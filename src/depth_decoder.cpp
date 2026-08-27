#include "breeze/depth_decoder.h"
#include "breeze/sampling.h"

#include <cmath>
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
    kv.resize(n_branches);
    for (auto & c : kv) c.init(m.backend, m.cfg.dd.n_layer, m.cfg.dd.head_dim, m.cfg.dd.n_kv_head, m.cfg.num_codebooks + 1);
    freq_factors = llama3_freq_factors(m.cfg.dd);
}

void DepthRunner::free() {
    for (auto & c : kv) c.free();
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

// runs n depth tokens for one branch and returns logits over the codebook vocab for the last token
static std::vector<float> depth_step(BreezeModel & m, DepthRunner & r, int branch, int start,
                                     const std::vector<float> * hidden0, const std::vector<int> & audio_idx,
                                     int head_idx) {
    const DepthConfig & c = m.cfg.dd;
    const int n_audio = (int) audio_idx.size();
    const int n = n_audio + (hidden0 ? 1 : 0);
    const int total = start + n;
    Graph g(2048);

    std::vector<int32_t> idx(audio_idx.begin(), audio_idx.end());
    ggml_tensor * aud = g.input_i32(idx, n_audio);
    ggml_tensor * embed = ggml_get_rows(g.ctx, m.w("audio_embd.weight"), aud); // [2048, n_audio]
    if (hidden0) {
        ggml_tensor * h0 = g.input_f32(*hidden0, m.cfg.hidden_size, 1);
        embed = ggml_concat(g.ctx, h0, embed, 1); // [2048, n]
    }
    ggml_tensor * x = linear(g.ctx, m.w("dd.in_proj.weight"), embed); // [1024, n]

    std::vector<int32_t> pos_i(n);
    for (int i = 0; i < n; i++) pos_i[i] = start + i;
    ggml_tensor * pos = g.input_i32(pos_i, n);
    ggml_tensor * ff = g.input_f32(r.freq_factors, (int) r.freq_factors.size());
    std::vector<float> mask_v = build_causal_mask(n, total, start, 0);
    ggml_tensor * mask = g.input_f32(mask_v, total, n);

    for (int il = 0; il < c.n_layer; il++) x = dd_layer(g.ctx, m, g, r.kv[branch], x, il, pos, ff, mask, start, n);
    x = rms_norm(g.ctx, x, m.w("dd.output_norm.weight"), c.rms_eps);

    ggml_tensor * last = ggml_cont(g.ctx, ggml_view_2d(g.ctx, x, c.hidden, 1, x->nb[1], (size_t) (n - 1) * x->nb[1]));
    ggml_tensor * head = m.w("dd.codebooks_head.weight");
    ggml_tensor * hw = ggml_view_2d(g.ctx, head, head->ne[0], head->ne[1], head->nb[1], (size_t) head_idx * head->nb[2]);
    ggml_tensor * logits = ggml_mul_mat(g.ctx, hw, last); // [vocab, 1]
    g.compute(m.backend, logits);
    return tensor_to_f32(logits);
}

std::vector<int> DepthRunner::run(BreezeModel & m, const std::vector<std::vector<float>> & hiddens,
                                  int cb0, float cfg_scale, std::mt19937 & rng) {
    const int nb = (int) hiddens.size();
    const int nc = m.cfg.num_codebooks;
    const int vs = m.cfg.audio_vocab_size;
    for (auto & c : kv) c.reset();

    SampleParams sp;
    sp.temperature = m.cfg.depth_temperature;
    sp.top_k = m.cfg.depth_top_k;
    sp.top_p = m.cfg.depth_top_p;

    std::vector<int> codes = { cb0 };
    auto combine = [&](std::vector<std::vector<float>> & per_branch) {
        if (nb == 1) return per_branch[0];
        std::vector<float> out(per_branch[0].size());
        for (size_t i = 0; i < out.size(); i++)
            out[i] = per_branch[1][i] + cfg_scale * (per_branch[0][i] - per_branch[1][i]);
        return out;
    };

    for (int j = 1; j < nc; j++) {
        const int head_idx = j - 1;
        std::vector<std::vector<float>> br(nb);
        for (int b = 0; b < nb; b++) {
            if (j == 1) {
                std::vector<int> aidx = { cb0 }; // codebook 0 offset is 0
                br[b] = depth_step(m, *this, b, 0, &hiddens[b], aidx, head_idx);
            } else {
                std::vector<int> aidx = { codes[head_idx] + head_idx * vs };
                br[b] = depth_step(m, *this, b, j, nullptr, aidx, head_idx);
            }
        }
        std::vector<float> logits = combine(br);
        codes.push_back(sample_token(logits, sp, rng));
    }
    return std::vector<int>(codes.begin() + 1, codes.end());
}

}
