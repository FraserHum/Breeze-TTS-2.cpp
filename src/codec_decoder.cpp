#include "breeze/codec.h"

#include <chrono>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>

namespace breeze {
namespace codec_detail {

// snake beta activation: x + sin(x*alpha)^2 / beta, both params stored in log space
static ggml_tensor * snake_beta(ggml_context * ctx, ggml_tensor * x, ggml_tensor * la, ggml_tensor * lb) {
    ggml_tensor * alpha = ggml_reshape_2d(ctx, ggml_exp(ctx, la), 1, la->ne[0]);
    ggml_tensor * beta = ggml_reshape_2d(ctx, ggml_exp(ctx, lb), 1, lb->ne[0]);
    ggml_tensor * s = ggml_sin(ctx, ggml_mul(ctx, x, alpha));
    return ggml_add(ctx, x, ggml_div(ctx, ggml_sqr(ctx, s), beta));
}

// convnext block over [L, C]
static ggml_tensor * convnext(ggml_context * ctx, BreezeModel & m, const std::string & p, ggml_tensor * x) {
    ggml_tensor * dw = m.w(p + ".dw.weight");
    ggml_tensor * h = depthwise1d_causal(ctx, dw, m.w(p + ".dw.bias"), x, (int) dw->ne[1]);
    h = ggml_cont(ctx, ggml_transpose(ctx, h));
    h = layer_norm(ctx, h, m.w(p + ".norm.weight"), m.w(p + ".norm.bias"), 1e-6f);
    h = ggml_add(ctx, linear(ctx, m.w(p + ".pw1.weight"), h), m.w(p + ".pw1.bias"));
    h = ggml_gelu_erf(ctx, h);
    h = ggml_add(ctx, linear(ctx, m.w(p + ".pw2.weight"), h), m.w(p + ".pw2.bias"));
    h = ggml_mul(ctx, h, m.w(p + ".gamma"));
    return ggml_add(ctx, x, ggml_cont(ctx, ggml_transpose(ctx, h)));
}

static ggml_tensor * residual_unit(ggml_context * ctx, BreezeModel & m, const std::string & p,
                                   ggml_tensor * x, int dilation) {
    ggml_tensor * h = snake_beta(ctx, x, m.w(p + ".a1"), m.w(p + ".b1"));
    h = conv1d_causal(ctx, m.w(p + ".conv1.conv.weight"), m.w(p + ".conv1.conv.bias"), h, 1, dilation);
    h = snake_beta(ctx, h, m.w(p + ".a2"), m.w(p + ".b2"));
    h = conv1d_causal(ctx, m.w(p + ".conv2.conv.weight"), m.w(p + ".conv2.conv.bias"), h, 1, 1);
    return ggml_add(ctx, x, h);
}

// rebuild the continuous latent from the residual codebooks, first stage split from the rest
static ggml_tensor * quantizer_decode(ggml_context * ctx, BreezeModel & m, Graph & g,
                                      const std::vector<int> & codes, int n_cb, int T) {
    auto lookup = [&](const std::string & name, int cb) {
        std::vector<int32_t> idx(T);
        for (int t = 0; t < T; t++) idx[t] = codes[(size_t) t * n_cb + cb];
        ggml_tensor * ids = g.input_i32(idx, T);
        return ggml_get_rows(ctx, m.w(name), ids);
    };

    ggml_tensor * first = lookup("codec.dq.first.0.embed", 0);
    first = linear(ctx, m.w("codec.dq.first.out_proj.weight"), first);

    ggml_tensor * rest = nullptr;
    for (int cb = 1; cb < n_cb; cb++) {
        ggml_tensor * e = lookup("codec.dq.rest." + std::to_string(cb - 1) + ".embed", cb);
        rest = rest ? ggml_add(ctx, rest, e) : e;
    }
    if (rest) {
        rest = linear(ctx, m.w("codec.dq.rest.out_proj.weight"), rest);
        first = ggml_add(ctx, first, rest);
    }
    return ggml_cont(ctx, ggml_transpose(ctx, first));
}

ggml_tensor * vocoder_decode(ggml_context * ctx, BreezeModel & m, Graph & g,
                             const std::vector<int> & codes, int n_cb, int T) {
    const VocoderConfig & c = m.cfg.voc;

    ggml_tensor * h = quantizer_decode(ctx, m, g, codes, n_cb, T);
    h = conv1d_causal(ctx, m.w("codec.dpre.conv.weight"), m.w("codec.dpre.conv.bias"), h, 1, 1);

    h = ggml_cont(ctx, ggml_transpose(ctx, h));
    h = vocoder_transformer(ctx, m, g, h, T);
    h = ggml_cont(ctx, ggml_transpose(ctx, h));

    for (size_t i = 0; i < c.upsampling_ratios.size(); i++) {
        const std::string p = "codec.dup." + std::to_string(i);
        h = convtr1d_causal(ctx, m.w(p + ".up.conv.weight"), m.w(p + ".up.conv.bias"), h,
                            c.upsampling_ratios[i]);
        h = convnext(ctx, m, p, h);
    }

    h = conv1d_causal(ctx, m.w("codec.dhead.conv.weight"), m.w("codec.dhead.conv.bias"), h, 1, 1);
    const int dilations[3] = { 1, 3, 9 };
    for (size_t i = 0; i < c.upsample_rates.size(); i++) {
        const std::string p = "codec.dblk." + std::to_string(i);
        h = snake_beta(ctx, h, m.w(p + ".alpha"), m.w(p + ".beta"));
        h = convtr1d_causal(ctx, m.w(p + ".up.conv.weight"), m.w(p + ".up.conv.bias"), h,
                            c.upsample_rates[i]);
        for (int j = 0; j < 3; j++) {
            h = residual_unit(ctx, m, p + ".res." + std::to_string(j), h, dilations[j]);
        }
    }

    h = snake_beta(ctx, h, m.w("codec.dfin.alpha"), m.w("codec.dfin.beta"));
    h = conv1d_causal(ctx, m.w("codec.dfin.conv.weight"), m.w("codec.dfin.conv.bias"), h, 1, 1);
    return ggml_clamp(ctx, h, -1.0f, 1.0f);
}

} // codec_detail

using namespace codec_detail;

// streaming (BREEZE_VOC_STATEFUL) decode: decodes only the NEW frames, carrying the pre-RoPE
// transformer k/v in a host-side ring plus the convolution tails. bit-exact with the windowed
// path over [ctx_start, start + n_frames) restricted to the new frames:
//  - quantizer/dpre/transformer/decoder are causal LTI per frame, so new-frame values do not
//    depend on how much older context is present (the dpre and conv tails carry more history
//    than their receptive fields reach back);
//  - attention is exact because the full windowed k/v sequence is rebuilt in-graph
//    (zero prefix | ring | new) and RoPE'd at the window-relative positions 0..T_kv-1 the
//    windowed path would use; each new query row sees exactly the set the windowed
//    build_causal_mask gives it, and every zero-padded column is older than the sliding window
//    (or absent), hence -inf masked for all new rows;
//  - the kq/kqv matmuls keep the same K dimension (T_kv == windowed sub_T), so the per-element
//    reduction order matches the windowed run
void MimiCodec::init_state(VocoderState & st) const {
    const VocoderConfig & c = m->cfg.voc;
    st.n_layer = c.n_layer;
    st.n_kv_head = c.n_kv_head;
    st.head_dim = c.head_dim;
    st.latent_dim = (int) m->w("codec.dpre.conv.weight")->ne[2];
    st.ring_cap = c.sliding_window - 1;
    st.tail_latent = 16; // == the generation ctx slack; T1 measured the conv RF at ~5.9 latent frames
    st.tail_input = 6;   // > the dpre K=3 receptive field (2 frames)
    const size_t rstride = (size_t) st.head_dim * st.n_kv_head;
    st.ring_k.assign(c.n_layer, std::vector<float>(rstride * st.ring_cap));
    st.ring_v.assign(c.n_layer, std::vector<float>(rstride * st.ring_cap));
    st.latent_tail.assign((size_t) st.latent_dim * st.tail_latent, 0.0f);
    st.reset();
}

std::vector<float> MimiCodec::decode_stateful(VocoderState & st, const std::vector<int> & new_codes,
                                              int n_frames, int n_cb, int start) {
    if (n_cb <= 0) n_cb = m->cfg.num_codebooks;
    const auto tg0 = std::chrono::steady_clock::now();
    const VocoderConfig & c = m->cfg.voc;
    const int W = c.sliding_window;
    const int ctx = W + 16; // same window context generation.cpp hands the windowed path
    const int count = n_frames;
    const int ring_len = st.ring_len;      // invariant: min(start, ring_cap)
    const int in_frames = (int) st.input_tail.size() / n_cb;
    const int latent_len = st.latent_len;  // invariant: min(start, tail_latent)

    const int ctx_start = start > ctx ? start - ctx : 0;
    const int P = start - ctx_start; // context frames ahead of the new ones
    const int off = P - ring_len;    // context frames older than the ring: zero-padded
    const int T_kv = P + count;      // == sub_T of the windowed path
    const int T_in = in_frames + count;

    Graph g(32768);

    // 1) quantizer + dpre over (input tail + new frames); keep only the new frames
    std::vector<int> codes_in = st.input_tail;
    codes_in.insert(codes_in.end(), new_codes.begin(), new_codes.end());
    ggml_tensor * h = quantizer_decode(g.ctx, *m, g, codes_in, n_cb, T_in);
    h = conv1d_causal(g.ctx, m->w("codec.dpre.conv.weight"), m->w("codec.dpre.conv.bias"), h, 1, 1);
    h = ggml_view_2d(g.ctx, h, count, h->ne[1], h->nb[1], (size_t)(T_in - count) * h->nb[0]);
    const int Ld = (int) h->ne[1];
    GGML_ASSERT(Ld == st.latent_dim);
    ggml_tensor * x = ggml_cont(g.ctx, ggml_transpose(g.ctx, h)); // [Ld, count]

    // 2) transformer over the new frames; the ring supplies the pre-RoPE context k/v
    const float scale = 1.0f / std::sqrt((float) c.head_dim);
    std::vector<int32_t> pos_kv(T_kv);
    for (int i = 0; i < T_kv; i++) pos_kv[i] = i;
    ggml_tensor * pos = g.input_i32(pos_kv, T_kv);
    std::vector<int32_t> pos_q(count);
    for (int i = 0; i < count; i++) pos_q[i] = P + i;
    ggml_tensor * posq = g.input_i32(pos_q, count);
    ggml_tensor * mask = g.input_f32(build_causal_mask(count, T_kv, P, W), T_kv, count);
    const size_t fstride = (size_t) c.head_dim * c.n_kv_head;
    ggml_tensor * zpad = nullptr;
    if (off > 0) {
        std::vector<float> zeros((size_t) off * fstride, 0.0f);
        zpad = g.input_f32(zeros, c.head_dim, c.n_kv_head, off);
    }
    std::vector<ggml_tensor *> ks(c.n_layer), vs(c.n_layer);

    ggml_tensor * hu = ggml_add(g.ctx, linear(g.ctx, m->w("codec.dtf.in_proj.weight"), x),
                                m->w("codec.dtf.in_proj.bias"));
    for (int il = 0; il < c.n_layer; il++) {
        const std::string p = "codec.dtf.blk." + std::to_string(il);
        ggml_tensor * res = hu;
        ggml_tensor * cur = rms_norm(g.ctx, hu, m->w(p + ".attn_norm.weight"), c.rms_eps);
        ggml_tensor * q = ggml_reshape_3d(g.ctx, linear(g.ctx, m->w(p + ".attn_q.weight"), cur),
                                          c.head_dim, c.n_head, count);
        ggml_tensor * k = ggml_reshape_3d(g.ctx, linear(g.ctx, m->w(p + ".attn_k.weight"), cur),
                                          c.head_dim, c.n_kv_head, count);
        ggml_tensor * v = ggml_reshape_3d(g.ctx, linear(g.ctx, m->w(p + ".attn_v.weight"), cur),
                                          c.head_dim, c.n_kv_head, count);
        // k/v context: zero prefix | ring | new, == the windowed [ctx_start, start + count)
        ggml_tensor * kcat, * vcat;
        if (ring_len > 0) {
            const size_t r0 = (size_t) ring_len * fstride;
            std::vector<float> rk(st.ring_k[il].begin(), st.ring_k[il].begin() + r0);
            std::vector<float> rv(st.ring_v[il].begin(), st.ring_v[il].begin() + r0);
            kcat = g.input_f32(rk, c.head_dim, c.n_kv_head, ring_len);
            vcat = g.input_f32(rv, c.head_dim, c.n_kv_head, ring_len);
            if (off > 0) {
                kcat = ggml_concat(g.ctx, zpad, kcat, 2);
                vcat = ggml_concat(g.ctx, zpad, vcat, 2);
            }
            kcat = ggml_concat(g.ctx, kcat, k, 2);
            vcat = ggml_concat(g.ctx, vcat, v, 2);
        } else {
            kcat = k; // start == 0 here, so off == 0 as well
            vcat = v;
        }
        kcat = ggml_rope_ext(g.ctx, kcat, pos, nullptr, c.head_dim, GGML_ROPE_TYPE_NEOX, 0,
                             c.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        q = ggml_rope_ext(g.ctx, q, posq, nullptr, c.head_dim, GGML_ROPE_TYPE_NEOX, 0,
                          c.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        ggml_tensor * a = attention(g.ctx, q, kcat, vcat, mask, scale, c.n_head, c.n_kv_head);
        a = linear(g.ctx, m->w(p + ".attn_output.weight"), a);
        a = ggml_mul(g.ctx, a, m->w(p + ".attn_scale"));
        hu = ggml_add(g.ctx, res, a);

        res = hu;
        cur = rms_norm(g.ctx, hu, m->w(p + ".ffn_norm.weight"), c.rms_eps);
        cur = swiglu_ffn(g.ctx, cur, m->w(p + ".ffn_gate.weight"), m->w(p + ".ffn_up.weight"),
                         m->w(p + ".ffn_down.weight"));
        cur = ggml_mul(g.ctx, cur, m->w(p + ".ffn_scale"));
        hu = ggml_add(g.ctx, res, cur);
        ks[il] = k;
        vs[il] = v;
    }
    hu = rms_norm(g.ctx, hu, m->w("codec.dtf.norm.weight"), c.rms_eps);
    ggml_tensor * z_new = ggml_add(g.ctx, linear(g.ctx, m->w("codec.dtf.out_proj.weight"), hu),
                                   m->w("codec.dtf.out_proj.bias")); // [Ld, count]

    // 3) conv tail over (latent tail + new latent); causal LTI, so the extra tail frames only
    // shift the kept samples, never change them
    ggml_tensor * zcat;
    if (latent_len > 0) {
        zcat = ggml_concat(g.ctx, g.input_f32(st.latent_tail, Ld, latent_len), z_new, 1);
    } else {
        zcat = z_new;
    }
    ggml_tensor * h2 = ggml_cont(g.ctx, ggml_transpose(g.ctx, zcat)); // [latent_len + count, Ld]
    for (size_t i = 0; i < c.upsampling_ratios.size(); i++) {
        const std::string p = "codec.dup." + std::to_string(i);
        h2 = convtr1d_causal(g.ctx, m->w(p + ".up.conv.weight"), m->w(p + ".up.conv.bias"), h2,
                             c.upsampling_ratios[i]);
        h2 = convnext(g.ctx, *m, p, h2);
    }
    h2 = conv1d_causal(g.ctx, m->w("codec.dhead.conv.weight"), m->w("codec.dhead.conv.bias"), h2, 1, 1);
    const int dilations[3] = { 1, 3, 9 };
    for (size_t i = 0; i < c.upsample_rates.size(); i++) {
        const std::string p = "codec.dblk." + std::to_string(i);
        h2 = snake_beta(g.ctx, h2, m->w(p + ".alpha"), m->w(p + ".beta"));
        h2 = convtr1d_causal(g.ctx, m->w(p + ".up.conv.weight"), m->w(p + ".up.conv.bias"), h2,
                             c.upsample_rates[i]);
        for (int j = 0; j < 3; j++) {
            h2 = residual_unit(g.ctx, *m, p + ".res." + std::to_string(j), h2, dilations[j]);
        }
    }
    h2 = snake_beta(g.ctx, h2, m->w("codec.dfin.alpha"), m->w("codec.dfin.beta"));
    h2 = conv1d_causal(g.ctx, m->w("codec.dfin.conv.weight"), m->w("codec.dfin.conv.bias"), h2, 1, 1);
    h2 = ggml_clamp(g.ctx, h2, -1.0f, 1.0f);
    // keep only the new frames' samples; the audio tensor is [1, (latent_len + count) * spf]
    const int spf = m->cfg.samples_per_frame;
    ggml_tensor * audio = ggml_view_1d(g.ctx, h2, (size_t) count * spf,
                                       (size_t) latent_len * spf * h2->nb[0]);
    // step 4 reads the pre-RoPE k/v and the new latent back after compute; gallocr reuses
    // every non-output buffer, so mark them as graph outputs first (same pattern as
    // MimiCodec::encode and backbone_run) or the ring/latent tail fill with recycled garbage
    for (int il = 0; il < c.n_layer; il++) {
        g.mark_output(ks[il]);
        g.mark_output(vs[il]);
    }
    g.mark_output(z_new);
    const auto td0 = std::chrono::steady_clock::now();
    g.compute(m->backend, audio);
    std::vector<float> out = tensor_to_f32(audio);
    if (rt_timing_enabled()) {
        // same split as MimiCodec::decode: graph_ms = fresh context + every op/mask built;
        // decode_ms = topo-sort, graph alloc, input upload, backend compute, readback
        g_rt_last.graph_ms = std::chrono::duration<double, std::milli>(td0 - tg0).count();
        g_rt_last.decode_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - td0).count();
    }

    // 4) roll the state: ring (newest ring_cap), latent tail, input tail
    const int frames_after = start + count;
    const int new_ring = std::min(st.ring_cap, frames_after);
    const int drop = ring_len + count - new_ring;
    for (int il = 0; il < c.n_layer; il++) {
        std::vector<float> kh = tensor_to_f32(ks[il]);
        std::vector<float> vh = tensor_to_f32(vs[il]);
        for (int w = 0; w < 2; w++) {
            std::vector<float> & ring = w ? st.ring_v[il] : st.ring_k[il];
            const std::vector<float> & fresh = w ? vh : kh;
            if (drop >= ring_len)
                std::memcpy(ring.data(), fresh.data() + (size_t)(count - new_ring) * fstride,
                            (size_t)new_ring * fstride * sizeof(float));
            else {
                if (drop > 0) std::memmove(ring.data(), ring.data() + (size_t) drop * fstride,
                                           (size_t)(ring_len - drop) * fstride * sizeof(float));
                std::memcpy(ring.data() + (size_t)(ring_len - drop) * fstride, fresh.data(),
                            (size_t) count * fstride * sizeof(float));
            }
        }
    }
    st.ring_len = new_ring;

    std::vector<float> zh = tensor_to_f32(z_new); // [Ld, count], frame-major chunks
    const int new_lat = std::min(st.tail_latent, frames_after);
    const int ldrop = latent_len + count - new_lat;
    if (ldrop >= latent_len)
        std::memcpy(st.latent_tail.data(), zh.data() + (size_t)(count - new_lat) * Ld,
                    (size_t)new_lat * Ld * sizeof(float));
    else {
        if (ldrop > 0) std::memmove(st.latent_tail.data(), st.latent_tail.data() + (size_t) ldrop * Ld,
                                    (size_t)(latent_len - ldrop) * Ld * sizeof(float));
        std::memcpy(st.latent_tail.data() + (size_t)(latent_len - ldrop) * Ld, zh.data(),
                    (size_t) count * Ld * sizeof(float));
    }
    st.latent_len = new_lat;

    const int new_in = std::min(st.tail_input, frames_after);
    const int from_new = std::min(count, new_in);
    const int from_old = new_in - from_new;
    std::vector<int> it;
    it.reserve((size_t) new_in * n_cb);
    if (from_old > 0)
        it.assign(st.input_tail.begin() + (size_t)(in_frames - from_old) * n_cb, st.input_tail.end());
    it.insert(it.end(), new_codes.begin() + (size_t)(count - from_new) * n_cb, new_codes.end());
    st.input_tail.swap(it);

    return out;
}

} // breeze
