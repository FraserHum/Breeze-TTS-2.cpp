#pragma once

#include <string>
#include <vector>

namespace breeze {

struct TextEncoderConfig {
    int hidden = 0, n_layer = 0, n_head = 0, n_kv_head = 0, head_dim = 0, ffn = 0;
    int vocab = 0, sliding_window = 0;
    float rms_eps = 1e-6f;
    float rope_theta_full = 1000000.0f, rope_freq_scale_full = 0.125f;
    float rope_theta_sliding = 10000.0f;
    float query_pre_attn_scalar = 256.0f, embed_scale = 1.0f;
    std::vector<int> layer_is_full;
};

struct BackboneConfig {
    int hidden = 0, n_layer = 0, n_head = 0, n_kv_head = 0, head_dim = 0, ffn = 0;
    float rms_eps = 1e-6f, rope_theta = 1000000.0f;
};

struct DepthConfig {
    int hidden = 0, n_layer = 0, n_head = 0, n_kv_head = 0, head_dim = 0, ffn = 0;
    float rms_eps = 1e-5f, rope_theta = 500000.0f;
    float rope_factor = 32.0f, rope_high_freq = 0.0078125f, rope_low_freq = 0.001953125f;
    int rope_orig_ctx = 16;
};

struct CodecConfig {
    int hidden = 0, n_layer = 0, n_head = 0, head_dim = 0, ffn = 0, sliding_window = 250;
    int num_filters = 64, codebook_dim = 256, num_semantic = 1;
    float rope_theta = 10000.0f, layer_norm_eps = 1e-5f;
    std::vector<int> upsampling_ratios;
};

// the bundled vocoder that turns codebook indices back into a waveform
struct VocoderConfig {
    int hidden = 512, n_layer = 8, n_head = 16, n_kv_head = 16, head_dim = 64, ffn = 1024;
    int sliding_window = 72, latent_dim = 1024, codebook_dim = 512, decoder_dim = 1536;
    float rms_eps = 1e-5f, rope_theta = 10000.0f;
    std::vector<int> upsample_rates;
    std::vector<int> upsampling_ratios;
};

struct BreezeConfig {
    int hidden_size = 0, num_codebooks = 16, audio_vocab_size = 2051, codec_codebook_size = 2048;
    int audio_token_id = 0, audio_eos_token_id = 0, backbone_eos_token_id = 0;
    int codebook_pad_token_id = 0, codebook_eos_token_id = 0;
    int sample_rate = 24000, samples_per_frame = 1920, max_new_tokens = 750;
    float temperature = 0.9f, top_p = 1.0f, repetition_penalty = 1.1f;
    float depth_temperature = 0.9f, depth_top_p = 1.0f;
    int top_k = 50, depth_top_k = 50;

    TextEncoderConfig te;
    BackboneConfig bb;
    DepthConfig dd;
    CodecConfig codec;
    VocoderConfig voc;
};

struct GGUFModel;
BreezeConfig parse_config(const GGUFModel & gg);

}
