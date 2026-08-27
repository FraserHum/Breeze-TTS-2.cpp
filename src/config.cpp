#include "breeze/config.h"
#include "breeze/gguf_loader.h"

namespace breeze {

BreezeConfig parse_config(const GGUFModel & gg) {
    BreezeConfig c;
    c.hidden_size = gg.kv_u32("breeze.hidden_size", 2048);
    c.num_codebooks = gg.kv_u32("breeze.num_codebooks", 16);
    c.audio_vocab_size = gg.kv_u32("breeze.audio_vocab_size", 2051);
    c.codec_codebook_size = gg.kv_u32("breeze.codec_codebook_size", 2048);
    c.audio_token_id = gg.kv_u32("breeze.audio_token_id", 262144);
    c.audio_eos_token_id = gg.kv_u32("breeze.audio_eos_token_id", 262145);
    c.backbone_eos_token_id = gg.kv_u32("breeze.backbone_eos_token_id", 2051);
    c.codebook_pad_token_id = gg.kv_u32("breeze.codebook_pad_token_id", 2050);
    c.codebook_eos_token_id = gg.kv_u32("breeze.codebook_eos_token_id", 0);
    c.sample_rate = gg.kv_u32("breeze.sample_rate", 24000);
    c.samples_per_frame = gg.kv_u32("breeze.samples_per_frame", 1920);
    c.max_new_tokens = gg.kv_u32("breeze.max_new_tokens", 750);
    c.temperature = gg.kv_f32("breeze.temperature", 0.9f);
    c.top_k = gg.kv_u32("breeze.top_k", 50);
    c.top_p = gg.kv_f32("breeze.top_p", 1.0f);
    c.repetition_penalty = gg.kv_f32("breeze.repetition_penalty", 1.1f);
    c.depth_temperature = gg.kv_f32("breeze.depth_temperature", 0.9f);
    c.depth_top_k = gg.kv_u32("breeze.depth_top_k", 50);
    c.depth_top_p = gg.kv_f32("breeze.depth_top_p", 1.0f);

    auto & te = c.te;
    te.hidden = gg.kv_u32("breeze.te.hidden_size", 1152);
    te.n_layer = gg.kv_u32("breeze.te.block_count", 26);
    te.n_head = gg.kv_u32("breeze.te.head_count", 4);
    te.n_kv_head = gg.kv_u32("breeze.te.head_count_kv", 1);
    te.head_dim = gg.kv_u32("breeze.te.head_dim", 256);
    te.ffn = gg.kv_u32("breeze.te.feed_forward_length", 6912);
    te.rms_eps = gg.kv_f32("breeze.te.rms_eps", 1e-6f);
    te.rope_theta_full = gg.kv_f32("breeze.te.rope_theta_full", 1000000.0f);
    te.rope_freq_scale_full = gg.kv_f32("breeze.te.rope_freq_scale_full", 0.125f);
    te.rope_theta_sliding = gg.kv_f32("breeze.te.rope_theta_sliding", 10000.0f);
    te.sliding_window = gg.kv_u32("breeze.te.sliding_window", 512);
    te.query_pre_attn_scalar = gg.kv_f32("breeze.te.query_pre_attn_scalar", 256.0f);
    te.embed_scale = gg.kv_f32("breeze.te.embed_scale", 1.0f);
    te.layer_is_full = gg.kv_i32_array("breeze.te.layer_is_full");

    auto & bb = c.bb;
    bb.hidden = gg.kv_u32("breeze.bb.hidden_size", 2048);
    bb.n_layer = gg.kv_u32("breeze.bb.block_count", 28);
    bb.n_head = gg.kv_u32("breeze.bb.head_count", 16);
    bb.n_kv_head = gg.kv_u32("breeze.bb.head_count_kv", 8);
    bb.head_dim = gg.kv_u32("breeze.bb.head_dim", 128);
    bb.ffn = gg.kv_u32("breeze.bb.feed_forward_length", 6144);
    bb.rms_eps = gg.kv_f32("breeze.bb.rms_eps", 1e-6f);
    bb.rope_theta = gg.kv_f32("breeze.bb.rope_theta", 1000000.0f);

    auto & dd = c.dd;
    dd.hidden = gg.kv_u32("breeze.dd.hidden_size", 1024);
    dd.n_layer = gg.kv_u32("breeze.dd.block_count", 12);
    dd.n_head = gg.kv_u32("breeze.dd.head_count", 8);
    dd.n_kv_head = gg.kv_u32("breeze.dd.head_count_kv", 2);
    dd.head_dim = gg.kv_u32("breeze.dd.head_dim", 128);
    dd.ffn = gg.kv_u32("breeze.dd.feed_forward_length", 8192);
    dd.rms_eps = gg.kv_f32("breeze.dd.rms_eps", 1e-5f);
    dd.rope_theta = gg.kv_f32("breeze.dd.rope_theta", 500000.0f);
    dd.rope_factor = gg.kv_f32("breeze.dd.rope_factor", 32.0f);
    dd.rope_high_freq = gg.kv_f32("breeze.dd.rope_high_freq_factor", 0.0078125f);
    dd.rope_low_freq = gg.kv_f32("breeze.dd.rope_low_freq_factor", 0.001953125f);
    dd.rope_orig_ctx = gg.kv_u32("breeze.dd.rope_orig_ctx", 16);

    auto & cc = c.codec;
    cc.hidden = gg.kv_u32("breeze.codec.hidden_size", 512);
    cc.n_layer = gg.kv_u32("breeze.codec.block_count", 8);
    cc.n_head = gg.kv_u32("breeze.codec.head_count", 8);
    cc.head_dim = gg.kv_u32("breeze.codec.head_dim", 64);
    cc.ffn = gg.kv_u32("breeze.codec.feed_forward_length", 2048);
    cc.rope_theta = gg.kv_f32("breeze.codec.rope_theta", 10000.0f);
    cc.sliding_window = gg.kv_u32("breeze.codec.sliding_window", 250);
    cc.num_filters = gg.kv_u32("breeze.codec.num_filters", 64);
    cc.codebook_dim = gg.kv_u32("breeze.codec.codebook_dim", 256);
    cc.num_semantic = gg.kv_u32("breeze.codec.num_semantic", 1);
    cc.layer_norm_eps = gg.kv_f32("breeze.codec.layer_norm_eps", 1e-5f);
    cc.upsampling_ratios = gg.kv_i32_array("breeze.codec.upsampling_ratios");
    if (cc.upsampling_ratios.empty()) cc.upsampling_ratios = { 8, 6, 5, 4 };

    auto & vc = c.voc;
    vc.hidden = gg.kv_u32("breeze.codec.dec.hidden_size", 512);
    vc.n_layer = gg.kv_u32("breeze.codec.dec.block_count", 8);
    vc.n_head = gg.kv_u32("breeze.codec.dec.head_count", 16);
    vc.n_kv_head = gg.kv_u32("breeze.codec.dec.head_count_kv", 16);
    vc.head_dim = gg.kv_u32("breeze.codec.dec.head_dim", 64);
    vc.ffn = gg.kv_u32("breeze.codec.dec.feed_forward_length", 1024);
    vc.rms_eps = gg.kv_f32("breeze.codec.dec.rms_eps", 1e-5f);
    vc.rope_theta = gg.kv_f32("breeze.codec.dec.rope_theta", 10000.0f);
    vc.sliding_window = gg.kv_u32("breeze.codec.dec.sliding_window", 72);
    vc.latent_dim = gg.kv_u32("breeze.codec.dec.latent_dim", 1024);
    vc.codebook_dim = gg.kv_u32("breeze.codec.dec.codebook_dim", 512);
    vc.decoder_dim = gg.kv_u32("breeze.codec.dec.decoder_dim", 1536);
    vc.upsample_rates = gg.kv_i32_array("breeze.codec.dec.upsample_rates");
    if (vc.upsample_rates.empty()) vc.upsample_rates = { 8, 5, 4, 3 };
    vc.upsampling_ratios = gg.kv_i32_array("breeze.codec.dec.upsampling_ratios");
    if (vc.upsampling_ratios.empty()) vc.upsampling_ratios = { 2, 2 };

    return c;
}

}
