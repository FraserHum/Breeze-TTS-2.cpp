// tools/voc_measure.cpp
//
// T1 measurement harness for BREEZE-STATEFUL-VOCODER (standalone, default-off).
// Loads the model through the repo loader (BreezeModel, never hand-parses GGUF) and runs:
//   --print-voc-cfg : the live VocoderConfig the code actually uses + every codec.d* tensor ne[],
//                     plus the code-trace verdicts steps_per_frame / attn_positions
//   --voc-selftest  : (a) pipeline shift test (windowed vocoder_decode over [0,240) vs [40,280),
//                     emitting absolute frames [120,240) both ways), (b) conv-only shift test
//                     (dhead..dfin over shifted random transformer-output latents),
//                     (c) Mac CPU microbench of MimiCodec::decode over T in {40,56,80,104,128}
//
// Deciding lines for the verdicts (quoted in the T1 report):
//   steps_per_frame = 1:  src/codec_decoder.cpp:73 passes the audio-frame count T straight into
//                         vocoder_transformer as the sequence length, and src/codec.cpp:34 passes
//                         MimiCodec::decode's n_frames unchanged.
//   attn_positions = relative: src/codec_transformer.cpp:52-55 builds pos_i[i] = i over the current
//                         decode window (0..T-1) and build_causal_mask(T, T, 0, W) at q_offset 0.
//
// snake_beta / residual_unit below are verbatim copies of the statics in src/codec_decoder.cpp
// (lines 11-16 and 31-38); conv1d_causal / convtr1d_causal / convnext / quantizer_decode are
// the exported codec_detail helpers from src/codec_conv.cpp / src/codec_decoder.cpp.
#include "breeze/codec.h"
#include "breeze/model.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using namespace breeze;

// documented PRNG: 64-bit LCG, x_{n+1} = 6364136223846793005 * x_n + 1442695040888963407 (mod 2^64)
struct LCG {
    uint64_t x;
    explicit LCG(uint64_t seed) : x(seed) {}
    int next_mod(uint64_t n) {
        x = x * 6364136223846793005ULL + 1442695040888963407ULL;
        return (int) (x % n);
    }
    float next_f32() { // uniform in [-1, 1)
        x = x * 6364136223846793005ULL + 1442695040888963407ULL;
        return (float) (((x >> 11) % 9007199254740992ULL) / 9007199254740992.0 * 2.0 - 1.0);
    }
};

// verbatim from src/codec_decoder.cpp (static there, copied for the conv-only test)
static ggml_tensor * snake_beta(ggml_context * ctx, ggml_tensor * x, ggml_tensor * la, ggml_tensor * lb) {
    ggml_tensor * alpha = ggml_reshape_2d(ctx, ggml_exp(ctx, la), 1, la->ne[0]);
    ggml_tensor * beta = ggml_reshape_2d(ctx, ggml_exp(ctx, lb), 1, lb->ne[0]);
    ggml_tensor * s = ggml_sin(ctx, ggml_mul(ctx, x, alpha));
    return ggml_add(ctx, x, ggml_div(ctx, ggml_sqr(ctx, s), beta));
}

static ggml_tensor * residual_unit(ggml_context * ctx, BreezeModel & m, const std::string & p,
                                   ggml_tensor * x, int dilation) {
    ggml_tensor * h = snake_beta(ctx, x, m.w(p + ".a1"), m.w(p + ".b1"));
    h = codec_detail::conv1d_causal(ctx, m.w(p + ".conv1.conv.weight"), m.w(p + ".conv1.conv.bias"), h, 1, dilation);
    h = snake_beta(ctx, h, m.w(p + ".a2"), m.w(p + ".b2"));
    h = codec_detail::conv1d_causal(ctx, m.w(p + ".conv2.conv.weight"), m.w(p + ".conv2.conv.bias"), h, 1, 1);
    return ggml_add(ctx, x, h);
}

static void print_tensor(const char * name, ggml_tensor * t) {
    if (!t) { printf("%-44s MISSING\n", name); return; }
    printf("%-44s ne=[%d %d %d %d]\n", name, (int) t->ne[0], (int) t->ne[1], (int) t->ne[2], (int) t->ne[3]);
}

static void print_voc_cfg(BreezeModel & m) {
    const VocoderConfig & c = m.cfg.voc;
    const int nc = m.cfg.num_codebooks;
    printf("== VocoderConfig (live, after GGUF overrides) ==\n");
    printf("voc.hidden %d\nvoc.n_layer %d\nvoc.n_head %d\nvoc.n_kv_head %d\nvoc.head_dim %d\n",
           c.hidden, c.n_layer, c.n_head, c.n_kv_head, c.head_dim);
    printf("voc.ffn %d\nvoc.sliding_window %d\nvoc.latent_dim %d\nvoc.codebook_dim %d\n",
           c.ffn, c.sliding_window, c.latent_dim, c.codebook_dim);
    printf("voc.decoder_dim %d\nvoc.rms_eps %.1e\nvoc.rope_theta %.1f\n", c.decoder_dim, c.rms_eps, c.rope_theta);
    auto rates = [](const std::vector<int> & v) {
        std::string s;
        for (int r : v) s += s.empty() ? std::to_string(r) : " " + std::to_string(r);
        return s;
    };
    printf("voc.upsample_rates %s\n", rates(c.upsample_rates).c_str());
    printf("voc.upsampling_ratios %s\n", rates(c.upsampling_ratios).c_str());
    printf("num_codebooks %d\ncodec_codebook_size %d\nsample_rate %d\nsamples_per_frame %d\n",
           nc, (int) m.cfg.codec_codebook_size, m.cfg.sample_rate, m.cfg.samples_per_frame);

    printf("== codec dec tensors (wopt, ne[]) ==\n");
    print_tensor("codec.dq.first.0.embed", m.wopt("codec.dq.first.0.embed"));
    print_tensor("codec.dq.first.out_proj.weight", m.wopt("codec.dq.first.out_proj.weight"));
    for (int cb = 1; cb < nc; cb++)
        print_tensor(("codec.dq.rest." + std::to_string(cb - 1) + ".embed").c_str(),
                     m.wopt("codec.dq.rest." + std::to_string(cb - 1) + ".embed"));
    print_tensor("codec.dq.rest.out_proj.weight", m.wopt("codec.dq.rest.out_proj.weight"));
    print_tensor("codec.dpre.conv.weight", m.wopt("codec.dpre.conv.weight"));
    print_tensor("codec.dpre.conv.bias", m.wopt("codec.dpre.conv.bias"));
    print_tensor("codec.dtf.in_proj.weight", m.wopt("codec.dtf.in_proj.weight"));
    print_tensor("codec.dtf.in_proj.bias", m.wopt("codec.dtf.in_proj.bias"));
    for (int i = 0; i < c.n_layer; i++) {
        const std::string p = "codec.dtf.blk." + std::to_string(i) + ".";
        const char * keys[] = { "attn_norm.weight", "attn_q.weight", "attn_k.weight", "attn_v.weight",
                                "attn_output.weight", "attn_scale", "ffn_norm.weight", "ffn_gate.weight",
                                "ffn_up.weight", "ffn_down.weight", "ffn_scale" };
        for (const char * k : keys) print_tensor((p + k).c_str(), m.wopt(p + k));
    }
    print_tensor("codec.dtf.norm.weight", m.wopt("codec.dtf.norm.weight"));
    print_tensor("codec.dtf.out_proj.weight", m.wopt("codec.dtf.out_proj.weight"));
    print_tensor("codec.dtf.out_proj.bias", m.wopt("codec.dtf.out_proj.bias"));
    for (size_t i = 0; i < c.upsampling_ratios.size(); i++) {
        const std::string p = "codec.dup." + std::to_string(i) + ".";
        const char * keys[] = { "up.conv.weight", "up.conv.bias", "dw.weight", "dw.bias", "norm.weight",
                                "norm.bias", "pw1.weight", "pw1.bias", "pw2.weight", "pw2.bias", "gamma" };
        for (const char * k : keys) print_tensor((p + k).c_str(), m.wopt(p + k));
    }
    print_tensor("codec.dhead.conv.weight", m.wopt("codec.dhead.conv.weight"));
    print_tensor("codec.dhead.conv.bias", m.wopt("codec.dhead.conv.bias"));
    for (size_t i = 0; i < c.upsample_rates.size(); i++) {
        const std::string p = "codec.dblk." + std::to_string(i);
        print_tensor((p + ".up.conv.weight").c_str(), m.wopt(p + ".up.conv.weight"));
        print_tensor((p + ".up.conv.bias").c_str(), m.wopt(p + ".up.conv.bias"));
        print_tensor((p + ".alpha").c_str(), m.wopt(p + ".alpha"));
        print_tensor((p + ".beta").c_str(), m.wopt(p + ".beta"));
        for (int j = 0; j < 3; j++) {
            const std::string r = p + ".res." + std::to_string(j) + ".";
            const char * keys[] = { "a1", "b1", "a2", "b2", "conv1.conv.weight", "conv1.conv.bias",
                                    "conv2.conv.weight", "conv2.conv.bias" };
            for (const char * k : keys) print_tensor((r + k).c_str(), m.wopt(r + k));
        }
    }
    print_tensor("codec.dfin.alpha", m.wopt("codec.dfin.alpha"));
    print_tensor("codec.dfin.beta", m.wopt("codec.dfin.beta"));
    print_tensor("codec.dfin.conv.weight", m.wopt("codec.dfin.conv.weight"));
    print_tensor("codec.dfin.conv.bias", m.wopt("codec.dfin.conv.bias"));

    printf("== verdicts (code trace) ==\n");
    printf("steps_per_frame 1  # src/codec_decoder.cpp:73 vocoder_transformer(ctx, m, g, h, T); T = audio frames; src/codec.cpp:34\n");
    printf("attn_positions relative  # src/codec_transformer.cpp:52-55 pos_i[i]=i over the decode window; mask build_causal_mask(T,T,0,W)\n");
}

// dhead..dfin tail on the 50 Hz latent [T, latent_dim], where T = 4x the audio-frame count
// (dup.0/dup.1, the two 2x up-samplers, sit upstream of dhead); mirrors src/codec_decoder.cpp:83-97
static std::vector<float> run_conv_tail(BreezeModel & m, const std::vector<float> & lat, int from, int T) {
    const VocoderConfig & c = m.cfg.voc;
    const int C = c.latent_dim;
    Graph g(32768);
    // ggml contiguous [T, C] tensors lay i0 (time) FASTEST: element (t, c) lives at flat index
    // t + c*T, so a host buffer fed via input_f32 must be CHANNEL-MAJOR (channel c holds its
    // whole T time series contiguously). lat is frame-major; transpose it here. (The repo's own
    // path never hits this because its latents are in-graph tensors, and ggml ops use ne/nb.)
    std::vector<float> seg((size_t) T * C);
    for (int c = 0; c < C; c++)
        for (int t = 0; t < T; t++)
            seg[(size_t) c * T + t] = lat[(size_t) (from + t) * C + c];
    ggml_tensor * h = g.input_f32(seg, T, C); // [T, C, 1], time on ne0 like the repo's own conv inputs
    h = codec_detail::conv1d_causal(g.ctx, m.w("codec.dhead.conv.weight"), m.w("codec.dhead.conv.bias"), h, 1, 1);
    const int dilations[3] = { 1, 3, 9 };
    for (size_t i = 0; i < c.upsample_rates.size(); i++) {
        const std::string p = "codec.dblk." + std::to_string(i);
        h = snake_beta(g.ctx, h, m.w(p + ".alpha"), m.w(p + ".beta"));
        h = codec_detail::convtr1d_causal(g.ctx, m.w(p + ".up.conv.weight"), m.w(p + ".up.conv.bias"),
                                          h, c.upsample_rates[i]);
        for (int j = 0; j < 3; j++)
            h = residual_unit(g.ctx, m, p + ".res." + std::to_string(j), h, dilations[j]);
    }
    h = snake_beta(g.ctx, h, m.w("codec.dfin.alpha"), m.w("codec.dfin.beta"));
    h = codec_detail::conv1d_causal(g.ctx, m.w("codec.dfin.conv.weight"), m.w("codec.dfin.conv.bias"), h, 1, 1);
    h = ggml_clamp(g.ctx, h, -1.0f, 1.0f);
    ggml_tensor * out = ggml_cont(g.ctx, ggml_reshape_1d(g.ctx, h, h->ne[0]));
    g.compute(m.backend, out);
    return tensor_to_f32(out);
}

struct ShiftStats {
    double max_diff = 0.0;
    long nondiff = 0;
    int first_frame = -1;
};

static ShiftStats compare_emitted(const std::vector<float> & a, size_t a0, const std::vector<float> & b,
                                  size_t b0, size_t n, int abs_frame_of_first_sample) {
    ShiftStats st;
    for (size_t i = 0; i < n; i++) {
        float d = a[a0 + i] - b[b0 + i];
        if (d < 0.0f) d = -d;
        if (d > st.max_diff) st.max_diff = (double) d;
        if (d != 0.0f) {
            st.nondiff++;
            if (st.first_frame < 0) st.first_frame = abs_frame_of_first_sample + (int) (i / 1920);
        }
    }
    return st;
}

static int selftest(BreezeModel & m) {
    const int nc = m.cfg.num_codebooks;
    const int spf = m.cfg.samples_per_frame;
    const int book = (int) m.cfg.codec_codebook_size;
    const int total = 280, win = 240, off = 40, emit0 = 120, emit1 = 240;
    const int emit_frames = emit1 - emit0;
    const size_t emit_samples = (size_t) emit_frames * spf;

    // fixed-seed codebooks: LCG seed 20260903, codes in [0, codec_codebook_size), frame-major [f*nc+cb]
    std::vector<int> codes((size_t) total * nc);
    LCG crng(20260903);
    for (auto & v : codes) v = crng.next_mod(book);

    printf("== pipeline shift test: windowed vocoder_decode, [0,%d) vs [%d,%d), emit abs frames [%d,%d) ==\n",
           win, off, total, emit0, emit1);
    {
        MimiCodec codec;
        codec.init(m);
        std::vector<float> a = codec.decode(codes, win, nc); // absolute frames [0,240)
        std::vector<float> b = codec.decode(std::vector<int>(codes.begin() + off * nc, codes.begin() + total * nc),
                                            win, nc); // absolute frames [40,280)
        printf("decode_a samples %zu decode_b samples %zu\n", a.size(), b.size());
        // emitted absolute [120,240): in a = samples [120*spf, 240*spf); in b = local [80,200) = samples [80*spf, 200*spf)
        ShiftStats st = compare_emitted(a, (size_t) emit0 * spf, b, (size_t) (emit0 - off) * spf,
                                        emit_samples, emit0);
        printf("pipeline_shift_max_diff %.12f\n", st.max_diff);
        printf("pipeline_shift_nondiff %ld/%zu\n", st.nondiff, emit_samples);
        printf("first_diff_frame %d\n", st.first_frame);
    }

    // conv-only shift test: the dhead..dfin tail consumes the 50 Hz latent, i.e. 4x the audio-frame
    // rate (dup.0/dup.1 are the two 2x up-samplers upstream of dhead). Latent [0,1120) covers audio
    // frames [0,280); LCG seed 42424242, random values in [-1, 1)
    int r_dup = 1;
    for (int r : m.cfg.voc.upsampling_ratios) r_dup *= r; // {2,2} -> 4
    const int lat_total = total * r_dup, lat_win = win * r_dup, lat_off = off * r_dup;
    const int C = m.cfg.voc.latent_dim;
    std::vector<float> lat((size_t) lat_total * C);
    LCG lrng(42424242);
    for (auto & v : lat) v = lrng.next_f32();
    printf("== conv-only shift test: dhead..dfin over 50 Hz latent (x%d) C=%d, latent [0,%d) vs [%d,%d), emit abs frames [%d,%d) ==\n",
           r_dup, C, lat_win, lat_off, lat_total, emit0, emit1);
    {
        std::vector<float> a = run_conv_tail(m, lat, 0, lat_win); // audio frames [0,240)
        std::vector<float> b = run_conv_tail(m, lat, lat_off, lat_win); // audio frames [40,280)
        printf("conv_a samples %zu conv_b samples %zu\n", a.size(), b.size());
        ShiftStats st = compare_emitted(a, (size_t) emit0 * spf, b, (size_t) (emit0 - off) * spf,
                                        emit_samples, emit0);
        printf("conv_shift_max_diff %.12f\n", st.max_diff);
        printf("conv_shift_nondiff %ld/%zu\n", st.nondiff, emit_samples);
    }

    // backward receptive field of the post-transformer stack, chained forward from the transformer
    // output rate using the ACTUAL kernel sizes: conv-transpose (K,s) -> R = R*s + (K-1) in
    // downstream-sample units; dilated conv K,d -> R += (K-1)*d at its own rate. Result in samples
    // at 24 kHz, divided by spf -> frames.
    auto K = [&](const std::string & n) { return (int) m.w(n)->ne[0]; };
    double R = 0.0;
    for (size_t i = 0; i < m.cfg.voc.upsampling_ratios.size(); i++)
        R = R * m.cfg.voc.upsampling_ratios[i] + K("codec.dup." + std::to_string(i) + ".up.conv.weight") - 1;
    R += K("codec.dhead.conv.weight") - 1;
    const int dilations[3] = { 1, 3, 9 };
    for (size_t i = 0; i < m.cfg.voc.upsample_rates.size(); i++) {
        R = R * m.cfg.voc.upsample_rates[i] + K("codec.dblk." + std::to_string(i) + ".up.conv.weight") - 1;
        for (int j = 0; j < 3; j++)
            R += (K("codec.dblk." + std::to_string(i) + ".res." + std::to_string(j) + ".conv1.conv.weight") - 1) * dilations[j];
    }
    R += K("codec.dfin.conv.weight") - 1;
    printf("conv_rf_samples_24k %.1f\n", R);
    printf("conv_rf_frames %.10f\n", R / spf);
    printf("dpre_tail_frames %d\n", K("codec.dpre.conv.weight") - 1);

    // microbench: time exactly what MimiCodec::decode does (graph build + g.compute + readback)
    printf("== microbench: MimiCodec::decode over T frames, 3 warmups + 5 measured, mean ms/frame ==\n");
    MimiCodec codec;
    codec.init(m);
    const int Ts[5] = { 40, 56, 80, 104, 128 };
    double msf[5];
    for (int i = 0; i < 5; i++) {
        const int T = Ts[i];
        for (int w = 0; w < 3; w++) (void) codec.decode(codes, T, nc);
        double sum = 0.0;
        for (int r = 0; r < 5; r++) {
            const auto t0 = std::chrono::steady_clock::now();
            std::vector<float> out = codec.decode(codes, T, nc);
            const auto t1 = std::chrono::steady_clock::now();
            const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            sum += ms;
            printf("bench_T%d_run%d %.3f ms (%zu samples)\n", T, r, ms, out.size());
        }
        msf[i] = (sum / 5.0) / T;
        printf("bench_T%d_ms_f %.6f\n", T, msf[i]);
    }
    // least-squares slope of ms/frame vs T
    double mx = 0.0, my = 0.0, mxx = 0.0, mxy = 0.0;
    for (int i = 0; i < 5; i++) { mx += Ts[i]; my += msf[i]; mxx += (double) Ts[i] * Ts[i]; mxy += Ts[i] * msf[i]; }
    const double slope = (5.0 * mxy - mx * my) / (5.0 * mxx - mx * mx);
    printf("bench_slope_ms_per_frame %.6f\n", slope); // least-squares slope of ms/f vs T
    printf("bench_endpoints_ms_f %.6f %.6f\n", msf[0], msf[4]); // (f(128)-f(40))/88 = %.6f for reference
    return 0;
}

// ---------------------------------------------------------------------------
// --voc-stateful-diff: stage-level differential between the windowed path
// (MimiCodec::decode over [ctx_start, start + count), the generation.cpp
// schedule ctx = sliding_window + 16, skip-trimmed to the new frames) and the
// streaming path (MimiCodec::decode_stateful), one 40-frame chunk at a time.
// Every stage is isolated on CPU f32, where a correct stateful path must be
// BIT-EXACT (max abs diff 0) with the windowed path:
//   S    tool sanity: this file's windowed capture graph vs the real MimiCodec::decode
//   E    dpre input: stateful dpre over (input_tail + fresh), new-frame columns,
//        vs the windowed dpre output's new-frame columns
//   F    latent tail (before decode) vs the windowed z's last latent_len frames
//   K    ring k (before decode) vs the windowed pre-RoPE k for the ring's frames
//   C    transformer z: a VERBATIM replica of decode_stateful's transformer block
//        (same x, ring, positions, mask) vs the windowed z's new frames
//   K2   replica new-frame pre-RoPE k vs the windowed k (0 when E == 0)
//   A    END-TO-END: the real decode_stateful audio vs the windowed keep region
//   B    conv receptive field: the full post-transformer stack (dup/dhead/dblk/dfin)
//        over only the last (tail_latent + count) windowed-z frames vs the keep region
//   G    conv tail over (latent tail before + replica z) vs the real stateful audio
//        (sanity: replica + conv stack == the real decode_stateful)
//   D    ring k/v (after decode) vs the windowed pre-RoPE k/v for the new ring frames
//   Fp   latent tail (after decode) vs the windowed z's last new_lat frames
// Runs the whole loop twice: BREEZE_VOC_CONVT_MATMUL=0 and =1 (flag isolation).
// ---------------------------------------------------------------------------

static double maxdiff_range(const std::vector<float> & a, size_t a0,
                            const std::vector<float> & b, size_t b0, size_t n) {
    double d = 0.0;
    for (size_t i = 0; i < n; i++) {
        float e = a[a0 + i] - b[b0 + i];
        if (e < 0.0f) e = -e;
        if ((double) e > d) d = (double) e;
    }
    return d;
}

// the full post-transformer stack (dup.0/dup.1/dhead/dblk*4/dfin/clamp) over a host-side
// frame-major latent z [T, Ld]; mirrors src/codec_decoder.cpp:83-97 exactly
static std::vector<float> conv_tail_full(BreezeModel & m, const std::vector<float> & z, int T) {
    const VocoderConfig & c = m.cfg.voc;
    const int Ld = c.latent_dim;
    Graph g(32768);
    // input_f32 flat index is c + Ld * t, so transpose the frame-major buffer to channel-major
    std::vector<float> seg((size_t) T * Ld);
    for (int ch = 0; ch < Ld; ch++)
        for (int t = 0; t < T; t++)
            seg[(size_t) ch * T + t] = z[(size_t) t * Ld + ch];
    ggml_tensor * h = g.input_f32(seg, T, Ld); // [T, Ld], time fastest
    for (size_t i = 0; i < c.upsampling_ratios.size(); i++) {
        const std::string p = "codec.dup." + std::to_string(i);
        h = codec_detail::convtr1d_causal(g.ctx, m.w(p + ".up.conv.weight"), m.w(p + ".up.conv.bias"),
                                          h, c.upsampling_ratios[i]);
        h = codec_detail::convnext(g.ctx, m, p, h);
    }
    h = codec_detail::conv1d_causal(g.ctx, m.w("codec.dhead.conv.weight"), m.w("codec.dhead.conv.bias"), h, 1, 1);
    const int dilations[3] = { 1, 3, 9 };
    for (size_t i = 0; i < c.upsample_rates.size(); i++) {
        const std::string p = "codec.dblk." + std::to_string(i);
        h = snake_beta(g.ctx, h, m.w(p + ".alpha"), m.w(p + ".beta"));
        h = codec_detail::convtr1d_causal(g.ctx, m.w(p + ".up.conv.weight"), m.w(p + ".up.conv.bias"),
                                          h, c.upsample_rates[i]);
        for (int j = 0; j < 3; j++)
            h = residual_unit(g.ctx, m, p + ".res." + std::to_string(j), h, dilations[j]);
    }
    h = snake_beta(g.ctx, h, m.w("codec.dfin.alpha"), m.w("codec.dfin.beta"));
    h = codec_detail::conv1d_causal(g.ctx, m.w("codec.dfin.conv.weight"), m.w("codec.dfin.conv.bias"), h, 1, 1);
    h = ggml_clamp(g.ctx, h, -1.0f, 1.0f);
    ggml_tensor * out = ggml_cont(g.ctx, ggml_reshape_1d(g.ctx, h, h->ne[0]));
    g.compute(m.backend, out);
    return tensor_to_f32(out);
}

// reliable windowed transformer z: builds quantizer -> dpre -> transformer -> z and ends the
// graph AT z (z is the sole output, so its buffer is kept alive; no conv tail to reuse it).
static std::vector<float> windowed_z_reliable(BreezeModel & m, const std::vector<int> & sub, int sub_T) {
    const VocoderConfig & c = m.cfg.voc;
    const int nc = m.cfg.num_codebooks;
    const int W = c.sliding_window;
    const int Ld = c.latent_dim;
    Graph g(32768);
    ggml_tensor * h = codec_detail::quantizer_decode(g.ctx, m, g, sub, nc, sub_T);
    ggml_tensor * hpre = codec_detail::conv1d_causal(g.ctx, m.w("codec.dpre.conv.weight"),
                                                     m.w("codec.dpre.conv.bias"), h, 1, 1);
    ggml_tensor * x = ggml_cont(g.ctx, ggml_transpose(g.ctx, hpre)); // [Ld, sub_T]
    const float scale = 1.0f / std::sqrt((float) c.head_dim);
    std::vector<int32_t> pos_i(sub_T);
    for (int i = 0; i < sub_T; i++) pos_i[i] = i;
    ggml_tensor * pos = g.input_i32(pos_i, sub_T);
    ggml_tensor * mask = g.input_f32(build_causal_mask(sub_T, sub_T, 0, W), sub_T, sub_T);
    ggml_tensor * hu = ggml_add(g.ctx, linear(g.ctx, m.w("codec.dtf.in_proj.weight"), x),
                                m.w("codec.dtf.in_proj.bias"));
    for (int il = 0; il < c.n_layer; il++) {
        const std::string p = "codec.dtf.blk." + std::to_string(il);
        ggml_tensor * res = hu;
        ggml_tensor * cur = rms_norm(g.ctx, hu, m.w(p + ".attn_norm.weight"), c.rms_eps);
        ggml_tensor * q = ggml_reshape_3d(g.ctx, linear(g.ctx, m.w(p + ".attn_q.weight"), cur),
                                          c.head_dim, c.n_head, sub_T);
        ggml_tensor * k = ggml_reshape_3d(g.ctx, linear(g.ctx, m.w(p + ".attn_k.weight"), cur),
                                          c.head_dim, c.n_kv_head, sub_T);
        ggml_tensor * v = ggml_reshape_3d(g.ctx, linear(g.ctx, m.w(p + ".attn_v.weight"), cur),
                                          c.head_dim, c.n_kv_head, sub_T);
        q = ggml_rope_ext(g.ctx, q, pos, nullptr, c.head_dim, GGML_ROPE_TYPE_NEOX, 0, c.rope_theta,
                          1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        k = ggml_rope_ext(g.ctx, k, pos, nullptr, c.head_dim, GGML_ROPE_TYPE_NEOX, 0, c.rope_theta,
                          1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        ggml_tensor * a = attention(g.ctx, q, k, v, mask, scale, c.n_head, c.n_kv_head);
        a = linear(g.ctx, m.w(p + ".attn_output.weight"), a);
        a = ggml_mul(g.ctx, a, m.w(p + ".attn_scale"));
        hu = ggml_add(g.ctx, res, a);
        res = hu;
        cur = rms_norm(g.ctx, hu, m.w(p + ".ffn_norm.weight"), c.rms_eps);
        cur = swiglu_ffn(g.ctx, cur, m.w(p + ".ffn_gate.weight"), m.w(p + ".ffn_up.weight"),
                         m.w(p + ".ffn_down.weight"));
        cur = ggml_mul(g.ctx, cur, m.w(p + ".ffn_scale"));
        hu = ggml_add(g.ctx, res, cur);
    }
    hu = rms_norm(g.ctx, hu, m.w("codec.dtf.norm.weight"), c.rms_eps);
    ggml_tensor * z = ggml_add(g.ctx, linear(g.ctx, m.w("codec.dtf.out_proj.weight"), hu),
                               m.w("codec.dtf.out_proj.bias"));
    g.compute(m.backend, z);
    return tensor_to_f32(z);
}

// windowed reference with staged captures: dpre out, per-layer pre-RoPE k/v, z, conv tail.
// the transformer block is a verbatim copy of src/codec_transformer.cpp:48-83
struct WindowCapture {
    std::vector<float> hpre; // [Ld, sub_T]
    std::vector<float> hq; // [T, Ld] quantizer output before dpre
    std::vector<float> x; // [Ld, sub_T] actual windowed transformer input (transposed dpre)
    std::vector<std::vector<float>> kv; // per layer k, pre-RoPE [head_dim, n_kv_head, sub_T]
    std::vector<std::vector<float>> vv; // per layer v [head_dim, n_kv_head, sub_T]
    std::vector<float> z; // [Ld, sub_T]
    std::vector<float> audio; // sub_T * samples_per_frame
};

static WindowCapture windowed_capture(BreezeModel & m, const std::vector<int> & sub, int sub_T) {
    const VocoderConfig & c = m.cfg.voc;
    const int nc = m.cfg.num_codebooks;
    const int W = c.sliding_window;
    const int Ld = c.latent_dim;

    Graph g(32768);
    ggml_tensor * h = codec_detail::quantizer_decode(g.ctx, m, g, sub, nc, sub_T);
    g.mark_output(h);
    ggml_tensor * hpre = codec_detail::conv1d_causal(g.ctx, m.w("codec.dpre.conv.weight"),
                                                     m.w("codec.dpre.conv.bias"), h, 1, 1);
    g.mark_output(hpre);
    ggml_tensor * x = ggml_cont(g.ctx, ggml_transpose(g.ctx, hpre)); // [Ld, sub_T]
    g.mark_output(x);

    const float scale = 1.0f / std::sqrt((float) c.head_dim);
    std::vector<int32_t> pos_i(sub_T);
    for (int i = 0; i < sub_T; i++) pos_i[i] = i;
    ggml_tensor * pos = g.input_i32(pos_i, sub_T);
    ggml_tensor * mask = g.input_f32(build_causal_mask(sub_T, sub_T, 0, W), sub_T, sub_T);

    std::vector<ggml_tensor *> kt(c.n_layer), vt(c.n_layer);
    ggml_tensor * hu = ggml_add(g.ctx, linear(g.ctx, m.w("codec.dtf.in_proj.weight"), x),
                                m.w("codec.dtf.in_proj.bias"));
    for (int il = 0; il < c.n_layer; il++) {
        const std::string p = "codec.dtf.blk." + std::to_string(il);
        ggml_tensor * res = hu;
        ggml_tensor * cur = rms_norm(g.ctx, hu, m.w(p + ".attn_norm.weight"), c.rms_eps);
        ggml_tensor * q = ggml_reshape_3d(g.ctx, linear(g.ctx, m.w(p + ".attn_q.weight"), cur),
                                          c.head_dim, c.n_head, sub_T);
        ggml_tensor * k = ggml_reshape_3d(g.ctx, linear(g.ctx, m.w(p + ".attn_k.weight"), cur),
                                          c.head_dim, c.n_kv_head, sub_T);
        ggml_tensor * v = ggml_reshape_3d(g.ctx, linear(g.ctx, m.w(p + ".attn_v.weight"), cur),
                                          c.head_dim, c.n_kv_head, sub_T);
        kt[il] = k;
        vt[il] = v;
        g.mark_output(k);
        g.mark_output(v);
        q = ggml_rope_ext(g.ctx, q, pos, nullptr, c.head_dim, GGML_ROPE_TYPE_NEOX, 0, c.rope_theta,
                          1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        k = ggml_rope_ext(g.ctx, k, pos, nullptr, c.head_dim, GGML_ROPE_TYPE_NEOX, 0, c.rope_theta,
                          1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        ggml_tensor * a = attention(g.ctx, q, k, v, mask, scale, c.n_head, c.n_kv_head);
        a = linear(g.ctx, m.w(p + ".attn_output.weight"), a);
        a = ggml_mul(g.ctx, a, m.w(p + ".attn_scale"));
        hu = ggml_add(g.ctx, res, a);

        res = hu;
        cur = rms_norm(g.ctx, hu, m.w(p + ".ffn_norm.weight"), c.rms_eps);
        cur = swiglu_ffn(g.ctx, cur, m.w(p + ".ffn_gate.weight"), m.w(p + ".ffn_up.weight"),
                         m.w(p + ".ffn_down.weight"));
        cur = ggml_mul(g.ctx, cur, m.w(p + ".ffn_scale"));
        hu = ggml_add(g.ctx, res, cur);
    }
    hu = rms_norm(g.ctx, hu, m.w("codec.dtf.norm.weight"), c.rms_eps);
    ggml_tensor * z = ggml_add(g.ctx, linear(g.ctx, m.w("codec.dtf.out_proj.weight"), hu),
                               m.w("codec.dtf.out_proj.bias"));
    g.mark_output(z);

    ggml_tensor * h2 = ggml_cont(g.ctx, ggml_transpose(g.ctx, z)); // [sub_T, Ld]
    for (size_t i = 0; i < c.upsampling_ratios.size(); i++) {
        const std::string p = "codec.dup." + std::to_string(i);
        h2 = codec_detail::convtr1d_causal(g.ctx, m.w(p + ".up.conv.weight"), m.w(p + ".up.conv.bias"),
                                           h2, c.upsampling_ratios[i]);
        h2 = codec_detail::convnext(g.ctx, m, p, h2);
    }
    h2 = codec_detail::conv1d_causal(g.ctx, m.w("codec.dhead.conv.weight"), m.w("codec.dhead.conv.bias"), h2, 1, 1);
    const int dilations[3] = { 1, 3, 9 };
    for (size_t i = 0; i < c.upsample_rates.size(); i++) {
        const std::string p = "codec.dblk." + std::to_string(i);
        h2 = snake_beta(g.ctx, h2, m.w(p + ".alpha"), m.w(p + ".beta"));
        h2 = codec_detail::convtr1d_causal(g.ctx, m.w(p + ".up.conv.weight"), m.w(p + ".up.conv.bias"),
                                           h2, c.upsample_rates[i]);
        for (int j = 0; j < 3; j++)
            h2 = residual_unit(g.ctx, m, p + ".res." + std::to_string(j), h2, dilations[j]);
    }
    h2 = snake_beta(g.ctx, h2, m.w("codec.dfin.alpha"), m.w("codec.dfin.beta"));
    h2 = codec_detail::conv1d_causal(g.ctx, m.w("codec.dfin.conv.weight"), m.w("codec.dfin.conv.bias"), h2, 1, 1);
    h2 = ggml_clamp(g.ctx, h2, -1.0f, 1.0f);
    ggml_tensor * afull = ggml_cont(g.ctx, ggml_reshape_1d(g.ctx, h2, h2->ne[0]));
    g.compute(m.backend, afull);

    WindowCapture wc;
    wc.hpre = tensor_to_f32(hpre);
    wc.hq = tensor_to_f32(h);
    wc.x = tensor_to_f32(x);
    for (int il = 0; il < c.n_layer; il++) wc.kv.push_back(tensor_to_f32(kt[il]));
    for (int il = 0; il < c.n_layer; il++) wc.vv.push_back(tensor_to_f32(vt[il]));
    wc.z = tensor_to_f32(z);
    wc.audio = tensor_to_f32(afull);
    return wc;
}

// verbatim replica of the stateful transformer block (src/codec_decoder.cpp:165-232):
// zero prefix | ring | new, RoPE'd at window-relative positions 0..T_kv-1 / P..P+count-1,
// build_causal_mask(count, T_kv, P, W). x_new is the caller's dpre output for the new frames
struct ReplicaOut {
    std::vector<float> z; // [Ld, count]
    std::vector<std::vector<float>> k; // per layer [head_dim, n_kv_head, count], pre-RoPE
};

static ReplicaOut stateful_replica(BreezeModel & m, const VocoderState & st,
                                   const std::vector<float> & x_new, int P, int count) {
    const VocoderConfig & c = m.cfg.voc;
    const int W = c.sliding_window;
    const int Ld = st.latent_dim;
    const size_t fstride = (size_t) c.head_dim * c.n_kv_head;
    const int T_kv = P + count;
    const int off = P - st.ring_len;

    Graph g(32768);
    ggml_tensor * x = g.input_f32(x_new, Ld, count);
    std::vector<int32_t> pos_kv(T_kv);
    for (int i = 0; i < T_kv; i++) pos_kv[i] = i;
    ggml_tensor * pos = g.input_i32(pos_kv, T_kv);
    std::vector<int32_t> pos_q(count);
    for (int i = 0; i < count; i++) pos_q[i] = P + i;
    ggml_tensor * posq = g.input_i32(pos_q, count);
    ggml_tensor * mask = g.input_f32(build_causal_mask(count, T_kv, P, W), T_kv, count);
    ggml_tensor * zpad = nullptr;
    if (off > 0) {
        std::vector<float> zeros((size_t) off * fstride, 0.0f);
        zpad = g.input_f32(zeros, c.head_dim, c.n_kv_head, off);
    }

    const float scale = 1.0f / std::sqrt((float) c.head_dim);
    std::vector<ggml_tensor *> ks(c.n_layer);
    ggml_tensor * hu = ggml_add(g.ctx, linear(g.ctx, m.w("codec.dtf.in_proj.weight"), x),
                                m.w("codec.dtf.in_proj.bias"));
    for (int il = 0; il < c.n_layer; il++) {
        const std::string p = "codec.dtf.blk." + std::to_string(il);
        ggml_tensor * res = hu;
        ggml_tensor * cur = rms_norm(g.ctx, hu, m.w(p + ".attn_norm.weight"), c.rms_eps);
        ggml_tensor * q = ggml_reshape_3d(g.ctx, linear(g.ctx, m.w(p + ".attn_q.weight"), cur),
                                          c.head_dim, c.n_head, count);
        ggml_tensor * k = ggml_reshape_3d(g.ctx, linear(g.ctx, m.w(p + ".attn_k.weight"), cur),
                                          c.head_dim, c.n_kv_head, count);
        ggml_tensor * v = ggml_reshape_3d(g.ctx, linear(g.ctx, m.w(p + ".attn_v.weight"), cur),
                                          c.head_dim, c.n_kv_head, count);
        ggml_tensor * kcat, * vcat;
        if (st.ring_len > 0) {
            const size_t r0 = (size_t) st.ring_len * fstride;
            std::vector<float> rk(st.ring_k[il].begin(), st.ring_k[il].begin() + r0);
            std::vector<float> rv(st.ring_v[il].begin(), st.ring_v[il].begin() + r0);
            kcat = g.input_f32(rk, c.head_dim, c.n_kv_head, st.ring_len);
            vcat = g.input_f32(rv, c.head_dim, c.n_kv_head, st.ring_len);
            if (off > 0) {
                kcat = ggml_concat(g.ctx, zpad, kcat, 2);
                vcat = ggml_concat(g.ctx, zpad, vcat, 2);
            }
            kcat = ggml_concat(g.ctx, kcat, k, 2);
            vcat = ggml_concat(g.ctx, vcat, v, 2);
        } else {
            kcat = k;
            vcat = v;
        }
        kcat = ggml_rope_ext(g.ctx, kcat, pos, nullptr, c.head_dim, GGML_ROPE_TYPE_NEOX, 0, c.rope_theta,
                             1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        q = ggml_rope_ext(g.ctx, q, posq, nullptr, c.head_dim, GGML_ROPE_TYPE_NEOX, 0, c.rope_theta,
                          1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        ggml_tensor * a = attention(g.ctx, q, kcat, vcat, mask, scale, c.n_head, c.n_kv_head);
        a = linear(g.ctx, m.w(p + ".attn_output.weight"), a);
        a = ggml_mul(g.ctx, a, m.w(p + ".attn_scale"));
        hu = ggml_add(g.ctx, res, a);

        res = hu;
        cur = rms_norm(g.ctx, hu, m.w(p + ".ffn_norm.weight"), c.rms_eps);
        cur = swiglu_ffn(g.ctx, cur, m.w(p + ".ffn_gate.weight"), m.w(p + ".ffn_up.weight"),
                         m.w(p + ".ffn_down.weight"));
        cur = ggml_mul(g.ctx, cur, m.w(p + ".ffn_scale"));
        hu = ggml_add(g.ctx, res, cur);
        ks[il] = k;
        g.mark_output(k);
    }
    hu = rms_norm(g.ctx, hu, m.w("codec.dtf.norm.weight"), c.rms_eps);
    ggml_tensor * z = ggml_add(g.ctx, linear(g.ctx, m.w("codec.dtf.out_proj.weight"), hu),
                               m.w("codec.dtf.out_proj.bias"));
    g.mark_output(z);
    g.compute(m.backend, z);

    ReplicaOut ro;
    ro.z = tensor_to_f32(z);
    ro.k.reserve(c.n_layer);
    for (int il = 0; il < c.n_layer; il++) ro.k.push_back(tensor_to_f32(ks[il]));
    return ro;
}

static int stateful_diff(BreezeModel & m, int total_frames) {
    const VocoderConfig & c = m.cfg.voc;
    const int nc = m.cfg.num_codebooks;
    const int spf = m.cfg.samples_per_frame;
    const int book = (int) m.cfg.codec_codebook_size;
    const int ctx = c.sliding_window + 16; // the generation.cpp flush schedule
    const int Ld = c.latent_dim;
    const size_t fstride = (size_t) c.head_dim * c.n_kv_head;

    std::vector<int> codes((size_t) total_frames * nc);
    LCG crng(20260903); // same codes as the T1 selftest / A/B runs
    for (auto & v : codes) v = crng.next_mod(book);

    MimiCodec codec;
    codec.init(m);
    VocoderState st;
    codec.init_state(st);

    double overall_a = 0.0;
    for (int start = 0; start < total_frames; start += 40) {
        const int count = std::min(40, total_frames - start);
        const int ctx_start = start > ctx ? start - ctx : 0;
        const int sub_T = start + count - ctx_start;
        const int P = start - ctx_start;
        std::vector<int> sub(codes.begin() + (size_t) ctx_start * nc,
                             codes.begin() + (size_t)(start + count) * nc);
        std::vector<int> fresh(codes.begin() + (size_t) start * nc,
                               codes.begin() + (size_t)(start + count) * nc);

        // state snapshots BEFORE the real stateful decode
        const int ring_len0 = st.ring_len;
        std::vector<std::vector<float>> ringk0(st.ring_k), ringv0(st.ring_v);
        std::vector<float> lat0(st.latent_tail);
        const int lat_len0 = st.latent_len;
        std::vector<int> in0(st.input_tail);

        WindowCapture wc = windowed_capture(m, sub, sub_T);
        std::vector<float> z_rel = windowed_z_reliable(m, sub, sub_T); // reliable windowed z [Ld, sub_T]

        // S: tool sanity — capture graph vs the real windowed decode
        std::vector<float> aref = codec.decode(sub, sub_T, 0);
        const double S = maxdiff_range(aref, 0, wc.audio, 0, (size_t) sub_T * spf);

        // ZZ: confirm buffer-reuse — capture-graph z vs reliable z (0 => reuse confirmed if >0)
        const double ZZ = maxdiff_range(wc.z, 0, z_rel, 0, (size_t) Ld * sub_T);
        // FR: RELIABLE F — state latent_tail (before) holds z for frames [start-lat_len0, start),
        // which sit at window indices [P-lat_len0, P) in the [ctx_start, start+count) window.
        const double FR = maxdiff_range(lat0, 0, z_rel, (size_t)(P - lat_len0) * Ld,
                                        (size_t) lat_len0 * Ld);
        if (getenv("VOCM_DBG"))
            fprintf(stderr, "[dbg] ZZ(capture_z_vs_reliable_z)=%.3e  FR(reliable_latent_vs_z)=%.3e  lat_len0=%d\n",
                    ZZ, FR, lat_len0);

        // E: stateful dpre input (tail + fresh) vs the ACTUAL windowed transformer input.
        // wc.x is [Ld, sub_T] (channel-major, ne0=Ld). The last `count` columns are the new
        // frames. x_new must be [Ld, count] (channel-major) for input_f32.
        std::vector<float> x_new((size_t) Ld * count);
        for (int ch = 0; ch < Ld; ch++)
            for (int t = 0; t < count; t++)
                x_new[(size_t) ch + (size_t) Ld * t] = wc.x[(size_t) ch + (size_t) Ld * (P + t)];
        std::vector<float> x_sf;
        std::vector<float> hq_dbg;
        {
            std::vector<int> codes_in = in0;
            codes_in.insert(codes_in.end(), fresh.begin(), fresh.end());
            const int T_in = (int) codes_in.size() / nc;
            Graph g2(32768);
            ggml_tensor * h = codec_detail::quantizer_decode(g2.ctx, m, g2, codes_in, nc, T_in);
            g2.mark_output(h);
            ggml_tensor * hd = codec_detail::conv1d_causal(g2.ctx, m.w("codec.dpre.conv.weight"),
                                            m.w("codec.dpre.conv.bias"), h, 1, 1);
            hd = ggml_view_2d(g2.ctx, hd, count, hd->ne[1], hd->nb[1], (size_t)(T_in - count) * hd->nb[0]);
            ggml_tensor * x = ggml_cont(g2.ctx, ggml_transpose(g2.ctx, hd));
            g2.compute(m.backend, x);
            hq_dbg = tensor_to_f32(h);
            x_sf = tensor_to_f32(x);
        }
        const double E = maxdiff_range(x_sf, 0, x_new, 0, (size_t) Ld * count);

        // F: latent tail (before) vs windowed z
        const double F = maxdiff_range(lat0, 0, wc.z, (size_t)(P - lat_len0) * Ld,
                                       (size_t) lat_len0 * Ld);

        // K: ring k (before) vs windowed pre-RoPE k for the ring's absolute frames
        double K = 0.0;
        for (int il = 0; il < c.n_layer; il++) {
            const int sf0 = start - ring_len0 - ctx_start;
            K = std::max(K, maxdiff_range(ringk0[il], 0, wc.kv[il], (size_t) sf0 * fstride,
                                          (size_t) ring_len0 * fstride));
        }

        // C: stateful-transformer replica (real stateful inputs) vs windowed z new frames
        ReplicaOut ro = stateful_replica(m, st, x_new, P, count);
        const double C = maxdiff_range(ro.z, 0, wc.z, (size_t)(sub_T - count) * Ld,
                                       (size_t) Ld * count);
        double K2 = 0.0;
        for (int il = 0; il < c.n_layer; il++)
            K2 = std::max(K2, maxdiff_range(ro.k[il], 0, wc.kv[il], (size_t)(sub_T - count) * fstride,
                                            (size_t) count * fstride));

        // A: the real stateful decode vs the windowed keep region
        std::vector<float> audio_s = codec.decode_stateful(st, fresh, count, 0, start);
        const size_t keep0 = (size_t) P * spf;
        const double A = maxdiff_range(audio_s, 0, wc.audio, keep0, (size_t) count * spf);

        // B: conv receptive field — tail over only the last (tail_latent + count) windowed-z frames
        // (conv_tail_full wants frame-major z [T, Ld])
        const int T56 = st.tail_latent + count;
        std::vector<float> z56((size_t) T56 * Ld);
        for (int t = 0; t < T56; t++)
            for (int ch = 0; ch < Ld; ch++)
                z56[(size_t) t * Ld + ch] = wc.z[(size_t) ch * sub_T + (sub_T - T56 + t)];
        std::vector<float> a56 = conv_tail_full(m, z56, T56);
        const double B = maxdiff_range(a56, (size_t) st.tail_latent * spf, wc.audio, keep0,
                                       (size_t) count * spf);

        // G: conv tail over (latent tail before + replica z) vs the real stateful audio
        // (frame-major z [T, Ld])
        const int Tc = lat_len0 + count;
        std::vector<float> zc((size_t) Tc * Ld);
        for (int t = 0; t < lat_len0; t++)
            for (int ch = 0; ch < Ld; ch++)
                zc[(size_t) t * Ld + ch] = lat0[(size_t) ch * lat_len0 + t];
        for (int t = 0; t < count; t++)
            for (int ch = 0; ch < Ld; ch++)
                zc[(size_t)(lat_len0 + t) * Ld + ch] = ro.z[(size_t) ch * count + t];
        std::vector<float> a_g = conv_tail_full(m, zc, Tc);
        const double G = maxdiff_range(a_g, (size_t) lat_len0 * spf, audio_s, 0,
                                       (size_t) count * spf);

        // D: ring after the real decode vs windowed pre-RoPE k/v for the new ring frames
        const int new_ring = std::min(st.ring_cap, start + count);
        double Dk = 0.0, Dv = 0.0;
        int Dk_layer = -1, Dv_layer = -1;
        for (int il = 0; il < c.n_layer; il++) {
            const int sf0 = start + count - new_ring - ctx_start;
            double dk = maxdiff_range(st.ring_k[il], 0, wc.kv[il], (size_t) sf0 * fstride,
                                      (size_t) new_ring * fstride);
            double dv = maxdiff_range(st.ring_v[il], 0, wc.vv[il], (size_t) sf0 * fstride,
                                      (size_t) new_ring * fstride);
            if (dk > Dk) { Dk = dk; Dk_layer = il; }
            if (dv > Dv) { Dv = dv; Dv_layer = il; }
        }

        // Fp: latent tail (after) vs windowed z last new_lat frames
        const int new_lat = std::min(st.tail_latent, start + count);
        const double Fp = maxdiff_range(st.latent_tail, 0, wc.z, (size_t)(sub_T - new_lat) * Ld,
                                        (size_t) new_lat * Ld);

        if (start == 0 && getenv("VOCM_DBG")) {
            fprintf(stderr, "[dbg] E=%.3e x_sf[0..3]=%f %f %f %f x_new[0..3]=%f %f %f %f\n",
                    E, x_sf[0], x_sf[1], x_sf[2], x_sf[3], x_new[0], x_new[1], x_new[2], x_new[3]);
            fprintf(stderr, "[dbg] in0.size=%zu (frames=%zu) fresh.size=%zu sub.size=%zu  in0[0..2]=%d %d %d fresh[0..2]=%d %d %d sub[0..2]=%d %d %d\n",
                    in0.size(), in0.size() / nc, fresh.size(), sub.size(),
                    in0.size() ? in0[0] : -1, in0.size() > 1 ? in0[1] : -1, in0.size() > 2 ? in0[2] : -1,
                    fresh[0], fresh[1], fresh[2], sub[0], sub[1], sub[2]);
            fprintf(stderr, "[dbg] hq_sf(q)[0..3]=%f %f %f %f  hq_win(q)[0..3]=%f %f %f %f\n",
                    hq_dbg[0], hq_dbg[1], hq_dbg[2], hq_dbg[3],
                    wc.hq[0], wc.hq[1], wc.hq[2], wc.hq[3]);
            {
                const size_t n = std::min(hq_dbg.size(), wc.hq.size());
                size_t firstdiff = n; double md = 0.0;
                for (size_t i = 0; i < n; i++) {
                    double d = std::fabs((double) hq_dbg[i] - (double) wc.hq[i]);
                    if (d > md) md = d;
                    if (firstdiff == n && d > 1e-9) firstdiff = i;
                }
                fprintf(stderr, "[dbg] hq maxdiff=%.3e firstdiff_idx=%zu (t=%zu ch=%zu) sizes sf=%zu win=%zu\n",
                        md, firstdiff, firstdiff / Ld, firstdiff % Ld, hq_dbg.size(), wc.hq.size());
            }
            fprintf(stderr, "[dbg] hpre_sf[0..3]=%f %f %f %f  hpre_win[0..3]=%f %f %f %f\n",
                    x_sf[0], x_sf[1], x_sf[2], x_sf[3], wc.hpre[0], wc.hpre[1], wc.hpre[2], wc.hpre[3]);
            const std::vector<float> & rk0 = st.ring_k[0];
            const std::vector<float> & wk0 = wc.kv[0];
            fprintf(stderr, "[dbg] ring_k[0][0..3]=%f %f %f %f\n", rk0[0], rk0[1], rk0[2], rk0[3]);
            fprintf(stderr, "[dbg] win_k[0][0..3]=%f %f %f %f\n", wk0[0], wk0[1], wk0[2], wk0[3]);
            fprintf(stderr, "[dbg] ring_k[0]@fs[0..3]=%f %f %f %f\n",
                    rk0[fstride], rk0[fstride + 1], rk0[fstride + 2], rk0[fstride + 3]);
            fprintf(stderr, "[dbg] win_k[0]@fs[0..3]=%f %f %f %f\n",
                    wk0[fstride], wk0[fstride + 1], wk0[fstride + 2], wk0[fstride + 3]);
            fprintf(stderr, "[dbg] ro.k[0][0..3]=%f %f %f %f\n",
                    ro.k[0][0], ro.k[0][1], ro.k[0][2], ro.k[0][3]);
            fprintf(stderr, "[dbg] C=%.3e K2=%.3e Dk=%.3e Dv=%.3e\n", C, K2, Dk, Dv);
        }

        printf("chunk start=%4d count=%2d ctx_start=%3d sub_T=%3d ring_len0=%2d\n",
               start, count, ctx_start, sub_T, ring_len0);
        printf("  S_capture_vs_decode %.3e\n", S);
        printf("  E_dpre_input        %.3e\n", E);
        printf("  F_latent_pre        %.3e\n", F);
        printf("  K_ring_k_pre        %.3e\n", K);
        printf("  C_z                 %.3e\n", C);
        printf("  K2_replica_k        %.3e\n", K2);
        printf("  A_audio             %.3e\n", A);
        printf("  B_convRF            %.3e\n", B);
        printf("  G_tail_replica      %.3e\n", G);
        printf("  D_ring_k_post       %.3e (layer %d)\n", Dk, Dk_layer);
        printf("  D_ring_v_post       %.3e (layer %d)\n", Dv, Dv_layer);
        printf("  F_latent_post       %.3e\n", Fp);
        overall_a = std::max(overall_a, A);
    }
    printf("stateful_diff_overall_max_audio_diff %.3e\n", overall_a);
    return overall_a < 1e-4 ? 0 : 1;
}

int main(int argc, char ** argv) {
    std::string model_path = "/Users/fraser/.cache/breeze-models/breeze-tts-2-q4_k.gguf";
    bool print_cfg = false, do_selftest = false, do_diff = false;
    int diff_frames = 280;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--print-voc-cfg") print_cfg = true;
        else if (a == "--voc-selftest") do_selftest = true;
        else if (a == "--voc-stateful-diff") do_diff = true;
        else if (a.compare(0, 9, "--frames=") == 0) diff_frames = std::atoi(a.substr(9).c_str());
        else if (a.compare(0, 8, "--model=") == 0) model_path = a.substr(8);
        else if (!a.empty() && a[0] != '-') model_path = a;
        else { fprintf(stderr, "unknown arg %s\n", a.c_str()); return 1; }
    }
    if (!print_cfg && !do_selftest && !do_diff) {
        fprintf(stderr,
                "usage: voc-measure [--print-voc-cfg] [--voc-selftest] [--voc-stateful-diff] "
                "[--frames=N] [--model=PATH|POSARG]\n");
        return 1;
    }
    BreezeModel m;
    if (!m.load(model_path, false)) { fprintf(stderr, "model load failed: %s\n", model_path.c_str()); return 1; }
    printf("model_path %s\n", model_path.c_str());
    printf("backend %s\n", m.backend.name());
    if (print_cfg) print_voc_cfg(m);
    if (do_selftest) selftest(m);
    int rc = 0;
    if (do_diff) {
        for (int cv = 0; cv <= 1; cv++) {
            setenv("BREEZE_VOC_CONVT_MATMUL", cv ? "1" : "0", 1);
            printf("== stateful diff pass: BREEZE_VOC_CONVT_MATMUL=%d, %d frames, CPU f32 ==\n",
                   cv, diff_frames);
            rc |= stateful_diff(m, diff_frames);
        }
    }
    m.free();
    return do_diff ? (rc == 0 ? 0 : 2) : 0;
}
