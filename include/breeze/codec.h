#pragma once

#include "breeze/model.h"

#include <algorithm>
#include <string>
#include <vector>

namespace breeze {

struct VocoderState; // defined below; referenced by MimiCodec's streaming-decode methods

struct MimiCodec {
    BreezeModel * m = nullptr;
    void init(BreezeModel & model) { m = &model; }

    // codes frame-major [f * n_cb + cb]; n_cb 0 means the full set. fewer codebooks drops detail,
    // 1 leaves only the semantic stage
    std::vector<float> decode(const std::vector<int> & codes, int n_frames, int n_cb = 0);
    // audio waveform in, returns codes frame-major, sets n_frames
    std::vector<int> encode(const std::vector<float> & audio, int & n_frames);

    // streaming (stateful) decode, armed by BREEZE_VOC_STATEFUL=1. new_codes holds the NEW frames
    // only, frame-major, arriving at absolute frame index `start`; returns exactly n_frames *
    // samples_per_frame audio samples. bit-exact with decode over the same windowed context
    // [start > ctx ? start - ctx : 0, start + n_frames), restricted to the new frames; rolls the
    // state. caller must init_state once and reset() the state at the start of every part
    void init_state(VocoderState & st) const;
    std::vector<float> decode_stateful(VocoderState & st, const std::vector<int> & new_codes,
                                       int n_frames, int n_cb, int start);
};

// opt-in streaming-vocoder timing probe, armed by BREEZE_RT_TIMING=1. numerically inert: it adds
// steady_clock reads and an opt-in print only, never a graph op or an RNG draw
struct RtTiming { double graph_ms = 0.0, decode_ms = 0.0; };
bool rt_timing_enabled();
// phase split of the most recent MimiCodec::decode / decode_stateful call; valid after every
// decode. shared with the streaming path, so it lives here instead of codec.cpp
inline thread_local RtTiming g_rt_last;
const RtTiming & rt_last_decode();

// BREEZE_VOC_STATEFUL=1 switches the generation flush from the windowed full-context re-decode to
// the streaming stateful path (MimiCodec::decode_stateful), which decodes only the NEW frames.
// default off; cached like rt_timing_enabled
bool vocoder_stateful_enabled();

// streaming vocoder state for one generation PART: the sliding-window key/value ring of the DTf
// transformer plus the convolution tails. the vocoder transformer attends with WINDOW-RELATIVE
// positions (pos 0..T-1 over the current decode window), so the ring holds PRE-RoPE k and RAW v;
// every flush re-applies RoPE at the local positions a windowed re-decode of
// [ctx_start, start + n_frames) would use, which keeps the new-frame outputs bit-exact with the
// windowed path by induction. one state per part; reset() it at the start of every part
struct VocoderState {
    int n_layer = 0, n_kv_head = 0, head_dim = 0, latent_dim = 0;
    int ring_cap = 0;       // sliding_window - 1: frames the ring holds
    int tail_latent = 16;   // transformer-output frames kept for the conv tail
    int tail_input = 6;     // codebook frames kept for the dpre conv
    int ring_len = 0;       // frames in the ring now (= min(frames so far, ring_cap))
    int latent_len = 0;     // frames in latent_tail now (= min(frames so far, tail_latent))
    std::vector<std::vector<float>> ring_k, ring_v; // per layer [head_dim, n_kv_head, ring_cap], oldest first
    std::vector<float> latent_tail;                 // [latent_dim, latent_len] frame-major, oldest first
    std::vector<int> input_tail;                    // frame-major [input frames * n_codebooks]

    void reset() {
        ring_len = 0;
        latent_len = 0;
        input_tail.clear();
        for (auto & r : ring_k) std::fill(r.begin(), r.end(), 0.0f);
        for (auto & r : ring_v) std::fill(r.begin(), r.end(), 0.0f);
    }
};

namespace codec_detail {

ggml_tensor * conv1d_causal(ggml_context * ctx, ggml_tensor * w, ggml_tensor * b,
                            ggml_tensor * x, int stride, int dilation);
ggml_tensor * convtr1d_causal(ggml_context * ctx, ggml_tensor * w, ggml_tensor * b,
                              ggml_tensor * x, int stride);
ggml_tensor * depthwise1d_causal(ggml_context * ctx, ggml_tensor * w, ggml_tensor * b,
                                 ggml_tensor * x, int kernel);
ggml_tensor * resnet_block(ggml_context * ctx, BreezeModel & m, const std::string & prefix, ggml_tensor * x);
ggml_tensor * seanet_encoder(ggml_context * ctx, BreezeModel & m, ggml_tensor * x);
ggml_tensor * mimi_transformer(ggml_context * ctx, BreezeModel & m, Graph & g, ggml_tensor * x,
                               const std::string & prefix, int seq_len);
ggml_tensor * vocoder_transformer(ggml_context * ctx, BreezeModel & m, Graph & g, ggml_tensor * x, int seq_len);
ggml_tensor * vocoder_decode(ggml_context * ctx, BreezeModel & m, Graph & g,
                             const std::vector<int> & codes, int n_codebooks, int seq_len);

}

}
