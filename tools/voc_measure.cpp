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
// (lines 11-16 and 31-38); conv1d_causal / convtr1d_causal are the exported codec_detail helpers
// from src/codec_conv.cpp. No src/ file is modified.
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

int main(int argc, char ** argv) {
    std::string model_path = "/Users/fraser/.cache/breeze-models/breeze-tts-2-q4_k.gguf";
    bool print_cfg = false, do_selftest = false;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--print-voc-cfg") print_cfg = true;
        else if (a == "--voc-selftest") do_selftest = true;
        else if (a.rfind("--model=", 0) == 0) model_path = a.substr(8);
        else if (!a.empty() && a[0] != '-') model_path = a;
        else { fprintf(stderr, "unknown arg %s\n", a.c_str()); return 1; }
    }
    if (!print_cfg && !do_selftest) {
        fprintf(stderr, "usage: voc-measure [--print-voc-cfg] [--voc-selftest] [--model=PATH|POSARG]\n");
        return 1;
    }
    BreezeModel m;
    if (!m.load(model_path, false)) { fprintf(stderr, "model load failed: %s\n", model_path.c_str()); return 1; }
    printf("model_path %s\n", model_path.c_str());
    printf("backend %s\n", m.backend.name());
    if (print_cfg) print_voc_cfg(m);
    if (do_selftest) selftest(m);
    m.free();
    return 0;
}
