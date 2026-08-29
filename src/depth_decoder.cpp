#include "breeze/depth_decoder.h"
#include "breeze/sampling.h"

#include <cmath>
#include <cstring>
#include <string>

namespace breeze {

static std::vector<float> llama3_freq_factors(const DepthConfig & c) {
    const int half = c.head_dim / 2;
    std::vector<float> ff(half, 1.0f);
    const float pi = 3.14159265358979323846f;
    const float low_wl = c.rope_orig_ctx / c.rope_low_freq;
    const float high_wl = c.rope_orig_ctx / c.rope_high_freq;
    for (int i = 0; i < half; i++) {
        float freq = std::pow(c.rope_theta, -2.0f * i / c.head_dim);
        float wavelen = 2.0f * pi / freq;
        if (wavelen > low_wl) {
            ff[i] = c.rope_factor;
        } else if (wavelen < high_wl) {
            ff[i] = 1.0f;
        } else {
            float smooth = (c.rope_orig_ctx / wavelen - c.rope_low_freq) / (c.rope_high_freq - c.rope_low_freq);
            ff[i] = 1.0f / ((1.0f - smooth) / c.rope_factor + smooth);
        }
    }
    return ff;
}

void DepthRunner::init(BreezeModel & m, int n_branches) {
    n_branch = n_branches;
    kv.init(m.backend, m.cfg.dd.n_layer, m.cfg.dd.head_dim, m.cfg.dd.n_kv_head,
            m.cfg.num_codebooks + 1, n_branches);
    freq_factors = llama3_freq_factors(m.cfg.dd);
    logits_buf.resize((size_t) m.cfg.audio_vocab_size * n_branches);
    combined_logits.resize(m.cfg.audio_vocab_size);
    flat_hiddens.resize((size_t) n_branches * m.cfg.hidden_size);
}

void DepthRunner::free() {
    kv.free();
    logits_buf.clear();
    combined_logits.clear();
    flat_hiddens.clear();
}

static ggml_tensor * dd_layer(ggml_context * ctx, BreezeModel & m, Graph & g, KVCache & kv,
                              ggml_tensor * x, int il, ggml_tensor * pos, ggml_tensor * ff,
                              ggml_tensor * mask, int start, int n) {
    const DepthConfig & c = m.cfg.dd;
    const std::string p = "dd.blk." + std::to_string(il);
    const float scale = 1.0f / std::sqrt((float) c.head_dim);

    ggml_tensor * res = x;
    ggml_tensor * h = rms_norm(ctx, x, m.w(p + ".attn_norm.weight"), c.rms_eps);
    ggml_tensor * q = ggml_reshape_3d(ctx, linear(ctx, m.w(p + ".attn_q.weight"), h), c.head_dim, c.n_head, n);
    ggml_tensor * k = ggml_reshape_3d(ctx, linear(ctx, m.w(p + ".attn_k.weight"), h), c.head_dim, c.n_kv_head, n);
    ggml_tensor * v = ggml_reshape_3d(ctx, linear(ctx, m.w(p + ".attn_v.weight"), h), c.head_dim, c.n_kv_head, n);
    q = ggml_rope_ext(ctx, q, pos, ff, c.head_dim, GGML_ROPE_TYPE_NEOX, 0, c.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    k = ggml_rope_ext(ctx, k, pos, ff, c.head_dim, GGML_ROPE_TYPE_NEOX, 0, c.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

    ggml_tensor * kfull = cache_append(ctx, g, kv.k[il], k, start);
    ggml_tensor * vfull = cache_append(ctx, g, kv.v[il], v, start);
    ggml_tensor * a = attention(ctx, q, kfull, vfull, mask, scale, c.n_head, c.n_kv_head);
    a = linear(ctx, m.w(p + ".attn_output.weight"), a);
    x = ggml_add(ctx, res, a);

    res = x;
    h = rms_norm(ctx, x, m.w(p + ".ffn_norm.weight"), c.rms_eps);
    h = swiglu_ffn(ctx, h, m.w(p + ".ffn_gate.weight"), m.w(p + ".ffn_up.weight"), m.w(p + ".ffn_down.weight"));
    return ggml_add(ctx, res, h);
}

// runs one depth position for every CFG branch at once and writes directly into pre-allocated logits_buf
static void depth_step(BreezeModel & m, DepthRunner & r, int start,
                       const std::vector<std::vector<float>> * hiddens,
                       int audio_code, int head_idx) {
    const DepthConfig & c = m.cfg.dd;
    const int nb = r.n_branch;
    const int n_pos = hiddens ? 2 : 1;
    const int n_tok = n_pos * nb;
    const int total = (start + n_pos) * nb;
    Graph g(1024);

    std::vector<int32_t> idx(nb, audio_code);
    ggml_tensor * aud = g.input_i32(idx, nb);
    ggml_tensor * embed = ggml_get_rows(g.ctx, m.w("audio_embd.weight"), aud); // [2048, nb]
    if (hiddens) {
        size_t off = 0;
        for (const auto & h : *hiddens) {
            std::memcpy(r.flat_hiddens.data() + off, h.data(), h.size() * sizeof(float));
            off += h.size();
        }
        ggml_tensor * h0 = g.input_f32(r.flat_hiddens, m.cfg.hidden_size, nb);
        embed = ggml_concat(g.ctx, h0, embed, 1); // [2048, 2*nb], position major
    }
    ggml_tensor * x = linear(g.ctx, m.w("dd.in_proj.weight"), embed); // [1024, n_tok]

    std::vector<int32_t> pos_i(n_tok);
    for (int i = 0; i < n_tok; i++) pos_i[i] = start + i / nb;
    ggml_tensor * pos = g.input_i32(pos_i, n_tok);
    ggml_tensor * ff = g.input_f32(r.freq_factors, (int) r.freq_factors.size());
    std::vector<float> mask_v = build_branch_causal_mask(n_tok, total, start, nb);
    ggml_tensor * mask = g.input_f32(mask_v, total, n_tok);

    for (int il = 0; il < c.n_layer; il++)
        x = dd_layer(g.ctx, m, g, r.kv, x, il, pos, ff, mask, start * nb, n_tok);
    x = rms_norm(g.ctx, x, m.w("dd.output_norm.weight"), c.rms_eps);

    ggml_tensor * last = ggml_cont(g.ctx, ggml_view_2d(g.ctx, x, c.hidden, nb, x->nb[1],
                                                       (size_t) (n_tok - nb) * x->nb[1]));
    ggml_tensor * head = m.w("dd.codebooks_head.weight");
    ggml_tensor * hw = ggml_view_2d(g.ctx, head, head->ne[0], head->ne[1], head->nb[1], (size_t) head_idx * head->nb[2]);
    ggml_tensor * logits = ggml_mul_mat(g.ctx, hw, last); // [vocab, nb]
    g.compute(m.backend, logits);
    ggml_backend_tensor_get(logits, r.logits_buf.data(), 0, r.logits_buf.size() * sizeof(float));
}

std::vector<int> DepthRunner::run(BreezeModel & m, const std::vector<std::vector<float>> & hiddens,
                                  int cb0, float cfg_scale, std::mt19937 & rng,
                                  const SampleParams * sp_in, const int * force, int n_force) {
    const int nc = m.cfg.num_codebooks;
    const int vs = m.cfg.audio_vocab_size;
    kv.reset();

    SampleParams sp;
    sp.temperature = m.cfg.depth_temperature;
    sp.top_k = m.cfg.depth_top_k;
    sp.top_p = m.cfg.depth_top_p;
    if (sp_in) sp = *sp_in;

    std::vector<int> codes = { cb0 };
    codes.reserve(nc);
    for (int j = 1; j < nc; j++) {
        const int head_idx = j - 1;
        if (j == 1) {
            depth_step(m, *this, 0, &hiddens, cb0, head_idx);
        } else {
            depth_step(m, *this, j, nullptr, codes[head_idx] + head_idx * vs, head_idx);
        }

        const int vocab = (int) logits_buf.size() / n_branch;
        if (n_branch > 1) {
            for (int i = 0; i < vocab; i++)
                combined_logits[i] = logits_buf[vocab + i] + cfg_scale * (logits_buf[i] - logits_buf[vocab + i]);
        } else {
            std::memcpy(combined_logits.data(), logits_buf.data(), vocab * sizeof(float));
        }
        // forced steps still run the graph, later codebooks are conditioned on this one
        codes.push_back(j <= n_force ? force[j - 1] : sample_token(combined_logits, sp, rng));
    }
    return std::vector<int>(codes.begin() + 1, codes.end());
}

}
