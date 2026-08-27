"""Codec conversion for the bundled Qwen3 TTS audio tokenizer.

The encoder is a Mimi model (SEANet + transformer + split RVQ). The decoder is a
separate vocoder: split RVQ, causal conv, a small transformer, ConvNeXt upsampling
and SnakeBeta decoder blocks.
"""
from __future__ import annotations

import numpy as np

EPS = 1e-5


def expand_depthwise(w):
    # ggml conv ops have no groups, so a depthwise (C,1,K) kernel becomes block diagonal (C,C,K)
    c, _, k = w.shape
    full = np.zeros((c, c, k), dtype=w.dtype)
    for i in range(c):
        full[i, i] = w[i, 0]
    return full


def squeeze1x1(w):
    return w[:, :, 0] if w.ndim == 3 and w.shape[-1] == 1 else w


def codebook(reader, prefix):
    names = reader.names()
    key = f"{prefix}.embedding_sum"
    if key not in names:
        key = f"{prefix}.embed_sum"
    embed_sum = reader.get(key)
    usage = reader.get(f"{prefix}.cluster_usage")
    return embed_sum / np.clip(usage, EPS, None)[:, None]


def write_encoder(add, reader, n_layers):
    for name in reader.names():
        if name.startswith("encoder.encoder.layers."):
            add("codec.enc." + name[len("encoder.encoder.layers."):], reader.get(name))
    for i in range(n_layers):
        src = f"encoder.encoder_transformer.layers.{i}"
        dst = f"codec.enct.blk.{i}"
        for a, b in TRANSFORMER_MIMI:
            add(dst + b, reader.get(src + a))
    add("codec.downsample.conv.weight", reader.get("encoder.downsample.conv.weight"))

    for tag, hf, count in (("sq", "semantic", 1), ("aq", "acoustic", 15)):
        base = f"encoder.quantizer.{hf}_residual_vector_quantizer"
        add(f"codec.{tag}.in_proj.weight", squeeze1x1(reader.get(base + ".input_proj.weight")))
        add(f"codec.{tag}.out_proj.weight", squeeze1x1(reader.get(base + ".output_proj.weight")))
        for i in range(count):
            add(f"codec.{tag}.{i}.embed", codebook(reader, f"{base}.layers.{i}.codebook"))


TRANSFORMER_MIMI = [
    (".input_layernorm.weight", ".attn_norm.weight"),
    (".input_layernorm.bias", ".attn_norm.bias"),
    (".post_attention_layernorm.weight", ".ffn_norm.weight"),
    (".post_attention_layernorm.bias", ".ffn_norm.bias"),
    (".self_attn.q_proj.weight", ".attn_q.weight"),
    (".self_attn.k_proj.weight", ".attn_k.weight"),
    (".self_attn.v_proj.weight", ".attn_v.weight"),
    (".self_attn.o_proj.weight", ".attn_output.weight"),
    (".self_attn_layer_scale.scale", ".attn_scale"),
    (".mlp.fc1.weight", ".ffn_up.weight"),
    (".mlp.fc2.weight", ".ffn_down.weight"),
    (".mlp_layer_scale.scale", ".ffn_scale"),
]

TRANSFORMER_QWEN = [
    (".input_layernorm.weight", ".attn_norm.weight"),
    (".post_attention_layernorm.weight", ".ffn_norm.weight"),
    (".self_attn.q_proj.weight", ".attn_q.weight"),
    (".self_attn.k_proj.weight", ".attn_k.weight"),
    (".self_attn.v_proj.weight", ".attn_v.weight"),
    (".self_attn.o_proj.weight", ".attn_output.weight"),
    (".self_attn_layer_scale.scale", ".attn_scale"),
    (".mlp.gate_proj.weight", ".ffn_gate.weight"),
    (".mlp.up_proj.weight", ".ffn_up.weight"),
    (".mlp.down_proj.weight", ".ffn_down.weight"),
    (".mlp_layer_scale.scale", ".ffn_scale"),
]


def write_decoder_quantizer(add, reader):
    for tag, hf, count in (("first", "rvq_first", 1), ("rest", "rvq_rest", 15)):
        base = f"decoder.quantizer.{hf}"
        add(f"codec.dq.{tag}.out_proj.weight", squeeze1x1(reader.get(base + ".output_proj.weight")))
        for i in range(count):
            add(f"codec.dq.{tag}.{i}.embed", codebook(reader, f"{base}.vq.layers.{i}._codebook"))


def write_decoder_transformer(add, reader, n_layers):
    add("codec.dtf.in_proj.weight", reader.get("decoder.pre_transformer.input_proj.weight"))
    add("codec.dtf.in_proj.bias", reader.get("decoder.pre_transformer.input_proj.bias"))
    add("codec.dtf.out_proj.weight", reader.get("decoder.pre_transformer.output_proj.weight"))
    add("codec.dtf.out_proj.bias", reader.get("decoder.pre_transformer.output_proj.bias"))
    add("codec.dtf.norm.weight", reader.get("decoder.pre_transformer.norm.weight"))
    for i in range(n_layers):
        src = f"decoder.pre_transformer.layers.{i}"
        dst = f"codec.dtf.blk.{i}"
        for a, b in TRANSFORMER_QWEN:
            add(dst + b, reader.get(src + a))


def write_decoder_upsample(add, reader, n_blocks):
    for i in range(n_blocks):
        add(f"codec.dup.{i}.up.conv.weight", reader.get(f"decoder.upsample.{i}.0.conv.weight"))
        add(f"codec.dup.{i}.up.conv.bias", reader.get(f"decoder.upsample.{i}.0.conv.bias"))
        dw = reader.get(f"decoder.upsample.{i}.1.dwconv.conv.weight")
        add(f"codec.dup.{i}.dw.weight", np.ascontiguousarray(dw[:, 0, :].T))
        add(f"codec.dup.{i}.dw.bias", reader.get(f"decoder.upsample.{i}.1.dwconv.conv.bias"))
        for a, b in (("norm.weight", "norm.weight"), ("norm.bias", "norm.bias"),
                     ("pwconv1.weight", "pw1.weight"), ("pwconv1.bias", "pw1.bias"),
                     ("pwconv2.weight", "pw2.weight"), ("pwconv2.bias", "pw2.bias"),
                     ("gamma", "gamma")):
            add(f"codec.dup.{i}.{b}", reader.get(f"decoder.upsample.{i}.1.{a}"))


def write_decoder_blocks(add, reader, n_blocks):
    for i in range(n_blocks):
        src = f"decoder.decoder.{i + 1}"
        dst = f"codec.dblk.{i}"
        add(f"{dst}.alpha", reader.get(f"{src}.block.0.alpha"))
        add(f"{dst}.beta", reader.get(f"{src}.block.0.beta"))
        add(f"{dst}.up.conv.weight", reader.get(f"{src}.block.1.conv.weight"))
        add(f"{dst}.up.conv.bias", reader.get(f"{src}.block.1.conv.bias"))
        for j in range(3):
            rs = f"{src}.block.{j + 2}"
            rd = f"{dst}.res.{j}"
            for a, b in (("act1.alpha", "a1"), ("act1.beta", "b1"),
                         ("act2.alpha", "a2"), ("act2.beta", "b2")):
                add(f"{rd}.{b}", reader.get(f"{rs}.{a}"))
            for c in ("conv1", "conv2"):
                add(f"{rd}.{c}.conv.weight", reader.get(f"{rs}.{c}.conv.weight"))
                add(f"{rd}.{c}.conv.bias", reader.get(f"{rs}.{c}.conv.bias"))


def write_codec(add, reader, cfg):
    enc = cfg["encoder_config"]
    dec = cfg["decoder_config"]
    n_up = len(dec["upsample_rates"])

    write_encoder(add, reader, enc["num_hidden_layers"])
    write_decoder_quantizer(add, reader)
    add("codec.dpre.conv.weight", reader.get("decoder.pre_conv.conv.weight"))
    add("codec.dpre.conv.bias", reader.get("decoder.pre_conv.conv.bias"))
    write_decoder_transformer(add, reader, dec["num_hidden_layers"])
    write_decoder_upsample(add, reader, len(dec["upsampling_ratios"]))
    add("codec.dhead.conv.weight", reader.get("decoder.decoder.0.conv.weight"))
    add("codec.dhead.conv.bias", reader.get("decoder.decoder.0.conv.bias"))
    write_decoder_blocks(add, reader, n_up)
    add("codec.dfin.alpha", reader.get(f"decoder.decoder.{n_up + 1}.alpha"))
    add("codec.dfin.beta", reader.get(f"decoder.decoder.{n_up + 1}.beta"))
    add("codec.dfin.conv.weight", reader.get(f"decoder.decoder.{n_up + 2}.conv.weight"))
    add("codec.dfin.conv.bias", reader.get(f"decoder.decoder.{n_up + 2}.conv.bias"))


def codec_metadata(writer, cfg):
    enc = cfg["encoder_config"]
    dec = cfg["decoder_config"]
    k, f = writer.add_uint32, writer.add_float32
    k("breeze.sample_rate", cfg["output_sample_rate"])
    k("breeze.samples_per_frame", cfg["decode_upsample_rate"])

    k("breeze.codec.hidden_size", enc["hidden_size"])
    k("breeze.codec.block_count", enc["num_hidden_layers"])
    k("breeze.codec.head_count", enc["num_attention_heads"])
    k("breeze.codec.head_dim", enc["head_dim"])
    k("breeze.codec.feed_forward_length", enc["intermediate_size"])
    f("breeze.codec.rope_theta", enc["rope_theta"])
    k("breeze.codec.sliding_window", enc["sliding_window"])
    k("breeze.codec.num_filters", enc["num_filters"])
    k("breeze.codec.codebook_dim", enc["codebook_dim"])
    k("breeze.codec.num_semantic", enc["num_semantic_quantizers"])
    f("breeze.codec.layer_norm_eps", enc["norm_eps"])
    writer.add_array("breeze.codec.upsampling_ratios", enc["upsampling_ratios"])

    k("breeze.codec.dec.hidden_size", dec["hidden_size"])
    k("breeze.codec.dec.block_count", dec["num_hidden_layers"])
    k("breeze.codec.dec.head_count", dec["num_attention_heads"])
    k("breeze.codec.dec.head_count_kv", dec["num_key_value_heads"])
    k("breeze.codec.dec.head_dim", dec["head_dim"])
    k("breeze.codec.dec.feed_forward_length", dec["intermediate_size"])
    f("breeze.codec.dec.rms_eps", dec["rms_norm_eps"])
    f("breeze.codec.dec.rope_theta", dec["rope_theta"])
    k("breeze.codec.dec.sliding_window", dec["sliding_window"])
    k("breeze.codec.dec.latent_dim", dec["latent_dim"])
    k("breeze.codec.dec.codebook_dim", dec["codebook_dim"])
    k("breeze.codec.dec.decoder_dim", dec["decoder_dim"])
    writer.add_array("breeze.codec.dec.upsample_rates", dec["upsample_rates"])
    writer.add_array("breeze.codec.dec.upsampling_ratios", dec["upsampling_ratios"])
