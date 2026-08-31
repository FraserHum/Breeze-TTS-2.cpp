#pragma once

#include "breeze/model.h"

#include <string>
#include <vector>

namespace breeze {

struct MimiCodec {
    BreezeModel * m = nullptr;
    void init(BreezeModel & model) { m = &model; }

    // codes frame-major [f * n_cb + cb]; n_cb 0 means the full set. fewer codebooks drops detail,
    // 1 leaves only the semantic stage
    std::vector<float> decode(const std::vector<int> & codes, int n_frames, int n_cb = 0);
    // audio waveform in, returns codes frame-major, sets n_frames
    std::vector<int> encode(const std::vector<float> & audio, int & n_frames);
};

// opt-in streaming-vocoder timing probe, armed by BREEZE_RT_TIMING=1. numerically inert: it adds
// steady_clock reads and an opt-in print only, never a graph op or an RNG draw
struct RtTiming { double graph_ms = 0.0, decode_ms = 0.0; };
bool rt_timing_enabled();
// phase split of the most recent MimiCodec::decode call; valid after every decode
const RtTiming & rt_last_decode();

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
