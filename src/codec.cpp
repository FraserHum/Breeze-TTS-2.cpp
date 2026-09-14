#include "breeze/codec.h"

#include <algorithm>
#include <chrono>
#include <cfloat>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <string>

namespace breeze {

using namespace codec_detail;

bool rt_timing_enabled() {
    static const int en = [] {
        const char * e = std::getenv("BREEZE_RT_TIMING");
        return e && std::strcmp(e, "1") == 0 ? 1 : 0;
    }();
    return en != 0;
}

bool vocoder_stateful_enabled() {
    static const int en = [] {
        const char * e = std::getenv("BREEZE_VOC_STATEFUL");
        return e && std::strcmp(e, "1") == 0 ? 1 : 0;
    }();
    return en != 0;
}

static bool vocoder_trim_enabled() {
    static const int en = [] {
        const char * e = std::getenv("BREEZE_VOC_TRIM");
        return !e || std::strcmp(e, "0") != 0 ? 1 : 0;
    }();
    return en != 0;
}

const RtTiming & rt_last_decode() { return g_rt_last; }

static ggml_tensor * transpose_cont(ggml_context * ctx, ggml_tensor * x) {
    return ggml_cont(ctx, ggml_transpose(ctx, x));
}

std::vector<float> MimiCodec::decode(const std::vector<int> & codes, int T, int n_cb, int trim_prefix) {
    if (n_cb <= 0) n_cb = m->cfg.num_codebooks;
    const auto tg0 = std::chrono::steady_clock::now();
    Graph g(32768);
    if (!vocoder_trim_enabled()) trim_prefix = -1;
    ggml_tensor * x = vocoder_decode(g.ctx, *m, g, codes, n_cb, T, trim_prefix);
    ggml_tensor * audio = ggml_cont(g.ctx, ggml_reshape_1d(g.ctx, x, x->ne[0]));
    const auto td0 = std::chrono::steady_clock::now();
    g.compute(m->backend, audio);
    std::vector<float> out = tensor_to_f32(audio);
    if (rt_timing_enabled()) {
        const auto td1 = std::chrono::steady_clock::now();
        // graph_ms: fresh ggml context + every op/mask built per call. decode_ms: topo-sort,
        // graph alloc, input upload, backend compute, waveform readback
        g_rt_last.graph_ms = std::chrono::duration<double, std::milli>(td0 - tg0).count();
        g_rt_last.decode_ms = std::chrono::duration<double, std::milli>(td1 - td0).count();
    }
    return out;
}

static int nearest(const std::vector<float> & book, const float * v, int dim, int n) {
    int best = 0;
    float best_d = FLT_MAX;
    for (int c = 0; c < n; c++) {
        const float * b = &book[(size_t) c * dim];
        float d = 0.0f;
        for (int k = 0; k < dim; k++) {
            float diff = v[k] - b[k];
            d += diff * diff;
        }
        if (d < best_d) { best_d = d; best = c; }
    }
    return best;
}

static std::vector<float> read_book(BreezeModel & m, const std::string & name) {
    return tensor_to_f32(m.w(name));
}

std::vector<int> MimiCodec::encode(const std::vector<float> & audio, int & n_frames) {
    const int nc = m->cfg.num_codebooks;
    const int n_sem = m->cfg.codec.num_semantic;
    const int n_ac = nc - n_sem;
    const int dim = m->cfg.codec.codebook_dim;
    const int book = m->cfg.codec_codebook_size;

    const int64_t total_samples = (int64_t) audio.size();
    constexpr int64_t kChunkSamples = 120000; // 5s of 24kHz audio = 125 transformer frames
    constexpr int64_t kChunkOverlap = 9600;   // 10 transformer frames to cover 5240-sample receptive field
    const int hidden = m->cfg.codec.hidden > 0 ? m->cfg.codec.hidden : 512;

    std::vector<float> all_conv_frames;
    int total_frames = 0;

    for (int64_t chunk_start = 0; chunk_start < total_samples; chunk_start += kChunkSamples) {
        int64_t chunk_audio_start = chunk_start;
        int64_t discard = 0;
        if (chunk_start > 0) {
            chunk_audio_start = std::max((int64_t) 0, chunk_start - kChunkOverlap);
            discard = (chunk_start - chunk_audio_start) / 960;
        }
        int64_t chunk_audio_end = std::min(total_samples, chunk_start + kChunkSamples);
        int64_t chunk_len = chunk_audio_end - chunk_audio_start;

        std::vector<float> chunk_audio(audio.begin() + chunk_audio_start, audio.begin() + chunk_audio_end);
        Graph g_chunk(4096);
        ggml_tensor * xc = g_chunk.input_f32(chunk_audio, (int) chunk_len, 1); // [chunk_len, 1]
        xc = seanet_encoder(g_chunk.ctx, *m, xc);                              // [T_chunk, hidden]
        xc = transpose_cont(g_chunk.ctx, xc);                                  // [hidden, T_chunk]
        g_chunk.compute(m->backend, xc);

        std::vector<float> xc_h = tensor_to_f32(xc);
        int n_chunk_frames = (int) xc->ne[1];
        if (discard < n_chunk_frames) {
            int keep_frames = n_chunk_frames - (int) discard;
            size_t src_offset = (size_t) discard * hidden;
            size_t count = (size_t) keep_frames * hidden;
            all_conv_frames.insert(all_conv_frames.end(), xc_h.begin() + src_offset, xc_h.begin() + src_offset + count);
            total_frames += keep_frames;
        }
    }

    Graph g(16384);
    ggml_tensor * x = g.input_f32(all_conv_frames, hidden, total_frames); // [hidden, total_frames]
    x = mimi_transformer(g.ctx, *m, g, x, "codec.enct", total_frames);
    x = transpose_cont(g.ctx, x);                                // [total_frames, hidden]
    x = conv1d_causal(g.ctx, m->w("codec.downsample.conv.weight"), nullptr, x, 2, 1); // [T2, hidden]
    ggml_tensor * emb = transpose_cont(g.ctx, x);                // [hidden, T2]
    ggml_tensor * rs = linear(g.ctx, m->w("codec.sq.in_proj.weight"), emb); // [dim, T2]
    ggml_tensor * ra = linear(g.ctx, m->w("codec.aq.in_proj.weight"), emb); // [dim, T2]
    ggml_set_output(rs);
    g.write(rs);
    g.compute(m->backend, ra);

    std::vector<float> rs_h = tensor_to_f32(rs);
    std::vector<float> ra_h = tensor_to_f32(ra);
    const int T = (int) ra->ne[1];
    n_frames = T;

    std::vector<float> sbook = read_book(*m, "codec.sq.0.embed");
    std::vector<std::vector<float>> abook(n_ac);
    for (int i = 0; i < n_ac; i++) abook[i] = read_book(*m, "codec.aq." + std::to_string(i) + ".embed");

    std::vector<int> codes((size_t) T * nc);
    std::vector<float> res(dim);
    for (int f = 0; f < T; f++) {
        codes[(size_t) f * nc + 0] = nearest(sbook, &rs_h[(size_t) f * dim], dim, book);
        for (int k = 0; k < dim; k++) res[k] = ra_h[(size_t) f * dim + k];
        for (int i = 0; i < n_ac; i++) {
            int c = nearest(abook[i], res.data(), dim, book);
            codes[(size_t) f * nc + (n_sem + i)] = c;
            const float * b = &abook[i][(size_t) c * dim];
            for (int k = 0; k < dim; k++) res[k] -= b[k];
        }
    }
    return codes;
}

}
