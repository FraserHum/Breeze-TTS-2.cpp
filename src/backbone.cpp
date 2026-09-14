#include "breeze/backbone.h"

#include <cmath>
#include <string>

namespace breeze {

void BackboneState::init(BreezeModel & m, int max_seq) {
    kv.init(m.backend, m.cfg.bb.n_layer, m.cfg.bb.head_dim, m.cfg.bb.n_kv_head, max_seq);
    pos = 0;
}

static ggml_tensor * build_audio_embed(ggml_context * ctx, BreezeModel & m, ggml_tensor * idx, int n) {
    const int nc = m.cfg.num_codebooks;
    const int hidden = m.cfg.hidden_size;
    // get_rows only indexes along one axis, so the frames stay flat until after the lookup
    ggml_tensor * rows = ggml_get_rows(ctx, m.w("audio_embd.weight"), idx);    // [hidden, nc*n]
    rows = ggml_reshape_3d(ctx, rows, hidden, nc, n);
    ggml_tensor * perm = ggml_cont(ctx, ggml_permute(ctx, rows, 1, 0, 2, 3));  // [nc, hidden, n]
    ggml_tensor * summed = ggml_sum_rows(ctx, perm);                           // [1, hidden, n]
    return ggml_reshape_2d(ctx, summed, hidden, n);
}

std::vector<float> audio_embed_forward(BreezeModel & m, const std::vector<int> & codes, int n) {
    const int nc = m.cfg.num_codebooks;
    const int vs = m.cfg.audio_vocab_size;
    Graph g(256);
    std::vector<int32_t> idx((size_t) nc * n);
    for (int f = 0; f < n; f++)
        for (int cb = 0; cb < nc; cb++)
            idx[(size_t) f * nc + cb] = codes[(size_t) f * nc + cb] + cb * vs;
    ggml_tensor * t = g.input_i32(idx, nc * n);
    ggml_tensor * out = build_audio_embed(g.ctx, m, t, n);
    g.compute(m.backend, out);
    return tensor_to_f32(out);
}

static ggml_tensor * bb_layer(ggml_context * ctx, BreezeModel & m, Graph & g, BackboneState & st,
                              ggml_tensor * x, int il, ggml_tensor * pos, ggml_tensor * mask, int n) {
    const BackboneConfig & c = m.cfg.bb;
    const std::string p = "bb.blk." + std::to_string(il);
    const float scale = 1.0f / std::sqrt((float) c.head_dim);

    ggml_tensor * res = x;
    ggml_tensor * h = rms_norm(ctx, x, m.w(p + ".attn_norm.weight"), c.rms_eps);

    ggml_tensor * q = nullptr;
    ggml_tensor * k = nullptr;
    ggml_tensor * v = nullptr;
    if (m.wopt(p + ".attn_qkv.weight") != nullptr) {
        const int q_dim = c.head_dim * c.n_head;
        const int kv_dim = c.head_dim * c.n_kv_head;
        ggml_tensor * qkv = linear(ctx, m.w(p + ".attn_qkv.weight"), h);
        q = ggml_view_3d(ctx, qkv, c.head_dim, c.n_head, n, (size_t) c.head_dim * sizeof(float), qkv->nb[1], 0);
        k = ggml_view_3d(ctx, qkv, c.head_dim, c.n_kv_head, n, (size_t) c.head_dim * sizeof(float), qkv->nb[1], (size_t) q_dim * sizeof(float));
        v = ggml_view_3d(ctx, qkv, c.head_dim, c.n_kv_head, n, (size_t) c.head_dim * sizeof(float), qkv->nb[1], (size_t) (q_dim + kv_dim) * sizeof(float));
        if (!ggml_is_contiguous(q)) q = ggml_cont(ctx, q);
        if (!ggml_is_contiguous(k)) k = ggml_cont(ctx, k);
        if (!ggml_is_contiguous(v)) v = ggml_cont(ctx, v);
    } else {
        q = ggml_reshape_3d(ctx, linear(ctx, m.w(p + ".attn_q.weight"), h), c.head_dim, c.n_head, n);
        k = ggml_reshape_3d(ctx, linear(ctx, m.w(p + ".attn_k.weight"), h), c.head_dim, c.n_kv_head, n);
        v = ggml_reshape_3d(ctx, linear(ctx, m.w(p + ".attn_v.weight"), h), c.head_dim, c.n_kv_head, n);
    }
    q = rms_norm(ctx, q, m.w(p + ".attn_q_norm.weight"), c.rms_eps);
    k = rms_norm(ctx, k, m.w(p + ".attn_k_norm.weight"), c.rms_eps);
    q = ggml_rope_ext(ctx, q, pos, nullptr, c.head_dim, GGML_ROPE_TYPE_NEOX, 0, c.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    k = ggml_rope_ext(ctx, k, pos, nullptr, c.head_dim, GGML_ROPE_TYPE_NEOX, 0, c.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

    ggml_tensor * kfull = cache_append(ctx, g, st.kv.k[il], k, st.pos);
    ggml_tensor * vfull = cache_append(ctx, g, st.kv.v[il], v, st.pos);

    ggml_tensor * a = attention(ctx, q, kfull, vfull, mask, scale, c.n_head, c.n_kv_head);
    a = linear(ctx, m.w(p + ".attn_output.weight"), a);
    x = ggml_add(ctx, res, a);

    res = x;
    h = rms_norm(ctx, x, m.w(p + ".ffn_norm.weight"), c.rms_eps);
    if (m.wopt(p + ".ffn_gate_up.weight") != nullptr) {
        h = swiglu_ffn_packed(ctx, h, m.w(p + ".ffn_gate_up.weight"), m.w(p + ".ffn_down.weight"), c.ffn);
    } else {
        h = swiglu_ffn(ctx, h, m.w(p + ".ffn_gate.weight"), m.w(p + ".ffn_up.weight"), m.w(p + ".ffn_down.weight"));
    }
    return ggml_add(ctx, res, h);
}

StepOut backbone_run(BreezeModel & m, BackboneState & st, const std::vector<float> & embeds, int n) {
    const BackboneConfig & c = m.cfg.bb;
    const int total = st.pos + n;
    Graph g(8192);

    ggml_tensor * x = g.input_f32(embeds, c.hidden, n);
    std::vector<int32_t> pos_i(n);
    for (int i = 0; i < n; i++) pos_i[i] = st.pos + i;
    ggml_tensor * pos = g.input_i32(pos_i, n);
    std::vector<float> mask_v = build_causal_mask(n, total, st.pos, 0);
    ggml_tensor * mask = g.input_f32(mask_v, total, n);

    for (int il = 0; il < c.n_layer; il++) x = bb_layer(g.ctx, m, g, st, x, il, pos, mask, n);
    x = rms_norm(g.ctx, x, m.w("bb.output_norm.weight"), c.rms_eps);

    ggml_tensor * last = ggml_view_2d(g.ctx, x, c.hidden, 1, x->nb[1], (size_t) (n - 1) * x->nb[1]);
    last = ggml_cont(g.ctx, last);
    ggml_tensor * logits = linear(g.ctx, m.w("bb.lm_head.weight"), last);

    ggml_set_output(last);
    g.write(last);
    g.compute(m.backend, logits);

    StepOut out;
    out.hidden = tensor_to_f32(last);
    out.logits = tensor_to_f32(logits);
    st.pos += n;
    return out;
}

}
