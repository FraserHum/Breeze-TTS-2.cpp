"""Convert a Breeze-TTS-2 Hugging Face checkpoint into a single GGUF file.

Usage:
    python scripts/convert_hf_to_gguf.py /path/to/breeze-tts-2 -o breeze-tts-2.gguf --dtype f16

Reads config.json, the safetensors shards and tokenizer.json, remaps every tensor
to the naming scheme the C++ runtime expects and writes all metadata as GGUF KV.
"""
from __future__ import annotations

import argparse
import json
import math
import struct
import sys
from pathlib import Path

import numpy as np

try:
    import gguf
except ImportError as exc:
    raise SystemExit("this converter needs the gguf package (pip install gguf)") from exc

ARCH = "breeze-tts-2"


class Safetensors:
    def __init__(self, path: Path):
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            self.header = json.loads(f.read(n))
        self.data_start = 8 + n
        self.mm = np.memmap(path, dtype=np.uint8, mode="r")

    def names(self):
        return [k for k in self.header if k != "__metadata__"]

    def get(self, name: str) -> np.ndarray:
        info = self.header[name]
        b, e = info["data_offsets"]
        raw = self.mm[self.data_start + b : self.data_start + e]
        dt = info["dtype"]
        if dt == "F32":
            arr = raw.view(np.float32)
        elif dt == "F16":
            arr = raw.view(np.float16).astype(np.float32)
        elif dt == "BF16":
            arr = (raw.view(np.uint16).astype(np.uint32) << 16).view(np.float32)
        else:
            raise ValueError(f"unsupported dtype {dt} for {name}")
        return np.ascontiguousarray(arr.reshape(info["shape"]))


class ShardReader:
    def __init__(self, model_dir: Path):
        self.dir = model_dir
        idx = model_dir / "model.safetensors.index.json"
        if idx.exists():
            self.weight_map = json.loads(idx.read_text())["weight_map"]
        else:
            st = Safetensors(model_dir / "model.safetensors")
            self.weight_map = {k: "model.safetensors" for k in st.names()}
        self._open: dict[str, Safetensors] = {}

    def handle(self, shard: str) -> Safetensors:
        if shard not in self._open:
            self._open[shard] = Safetensors(self.dir / shard)
        return self._open[shard]

    def keys(self):
        return self.weight_map.keys()

    def get(self, name: str) -> np.ndarray:
        return self.handle(self.weight_map[name]).get(name).astype(np.float32)


def to_dtype(arr: np.ndarray, dtype: str, name: str) -> np.ndarray:
    # conv kernels and anything used elementwise stay f32, vulkan conv ops reject f16
    keep_f32 = (
        arr.ndim <= 1
        or name.endswith("embed")
        or "codebook" in name
        or ".conv." in name
        or name.endswith(".dw.weight")
        or name.endswith("sample.weight")
    )
    if dtype == "f32" or keep_f32:
        return arr.astype(np.float32)
    return arr.astype(np.float16)


def add(writer, name, arr, dtype):
    # gemma rms norm scales by (1 + w), so bake the offset in and keep the runtime plain
    if name.startswith("te.") and name.endswith("_norm.weight"):
        arr = arr + 1.0
    writer.add_tensor(name, to_dtype(np.ascontiguousarray(arr), dtype, name))


def copy_block(writer, reader, src_tmpl, dst_tmpl, n, names, dtype, optional=()):
    for i in range(n):
        for src_suffix, dst_suffix in names:
            src = src_tmpl.format(i=i) + src_suffix
            if src not in reader.weight_map:
                if src_suffix in optional:
                    continue
                raise KeyError(f"missing tensor {src}")
            add(writer, dst_tmpl.format(i=i) + dst_suffix, reader.get(src), dtype)


ATTN = [
    (".self_attn.q_proj.weight", ".attn_q.weight"),
    (".self_attn.k_proj.weight", ".attn_k.weight"),
    (".self_attn.v_proj.weight", ".attn_v.weight"),
    (".self_attn.o_proj.weight", ".attn_output.weight"),
]
MLP = [
    (".mlp.gate_proj.weight", ".ffn_gate.weight"),
    (".mlp.up_proj.weight", ".ffn_up.weight"),
    (".mlp.down_proj.weight", ".ffn_down.weight"),
]
QK = [
    (".self_attn.q_norm.weight", ".attn_q_norm.weight"),
    (".self_attn.k_norm.weight", ".attn_k_norm.weight"),
]


def write_text_encoder(writer, reader, cfg, dtype):
    te = cfg["text_encoder_config"]
    add(writer, "te.token_embd.weight", reader.get("text_encoder.embed_tokens.weight"), dtype)
    add(writer, "te.output_norm.weight", reader.get("text_encoder.norm.weight"), dtype)
    add(writer, "te.proj.weight", reader.get("text_encoder_proj.weight"), dtype)
    norms = [
        (".pre_self_attn_layernorm.weight", ".attn_norm.weight"),
        (".post_self_attn_layernorm.weight", ".post_attn_norm.weight"),
        (".pre_feedforward_layernorm.weight", ".ffn_norm.weight"),
        (".post_feedforward_layernorm.weight", ".post_ffn_norm.weight"),
    ]
    copy_block(writer, reader, "text_encoder.layers.{i}", "te.blk.{i}",
               te["num_hidden_layers"], norms + ATTN + QK + MLP, dtype)


def write_backbone(writer, reader, cfg, dtype):
    bb = cfg["backbone_config"]
    add(writer, "audio_embd.weight", reader.get("depth_decoder.model.embed_tokens.weight"), dtype)
    add(writer, "bb.output_norm.weight", reader.get("backbone_model.norm.weight"), dtype)
    add(writer, "bb.lm_head.weight", reader.get("lm_head.weight"), dtype)
    norms = [
        (".input_layernorm.weight", ".attn_norm.weight"),
        (".post_attention_layernorm.weight", ".ffn_norm.weight"),
    ]
    copy_block(writer, reader, "backbone_model.layers.{i}", "bb.blk.{i}",
               bb["num_hidden_layers"], norms + ATTN + QK + MLP, dtype)


def write_depth(writer, reader, cfg, dtype):
    dd = cfg["depth_decoder_config"]
    add(writer, "dd.in_proj.weight", reader.get("depth_decoder.model.inputs_embeds_projector.weight"), dtype)
    add(writer, "dd.output_norm.weight", reader.get("depth_decoder.model.norm.weight"), dtype)
    # (n_cb-1, hidden, vocab) -> (n_cb-1, vocab, hidden) so a ggml slice is [hidden, vocab]
    head = reader.get("depth_decoder.codebooks_head.weight").transpose(0, 2, 1)
    add(writer, "dd.codebooks_head.weight", head, dtype)
    norms = [
        (".input_layernorm.weight", ".attn_norm.weight"),
        (".post_attention_layernorm.weight", ".ffn_norm.weight"),
    ]
    copy_block(writer, reader, "depth_decoder.model.layers.{i}", "dd.blk.{i}",
               dd["num_hidden_layers"], norms + ATTN + MLP, dtype)


def write_codec_all(writer, model_dir: Path, dtype: str):
    import convert_codec

    at_dir = model_dir / "audio_tokenizer"
    if not at_dir.is_dir():
        raise SystemExit(f"missing {at_dir}; the bundled audio tokenizer is required")
    cfg = json.loads((at_dir / "config.json").read_text())
    st = Safetensors(at_dir / "model.safetensors")

    class Reader:
        def names(self):
            return st.names()

        def get(self, name):
            return st.get(name).astype(np.float32)

    convert_codec.write_codec(lambda n, a: add(writer, n, a, dtype), Reader(), cfg)
    return cfg



def write_tokenizer(writer, model_dir: Path):
    tok = json.loads((model_dir / "tokenizer.json").read_text(encoding="utf-8"))
    model = tok["model"]
    vocab = model["vocab"]
    id2tok = {v: k for k, v in vocab.items()}
    added = tok.get("added_tokens", [])
    special_ids = set()
    for a in added:
        id2tok[a["id"]] = a["content"]
        if a.get("special"):
            special_ids.add(a["id"])
    size = max(id2tok) + 1
    tokens = [id2tok.get(i, f"<unused_{i}>") for i in range(size)]
    types = []
    for i in range(size):
        t = tokens[i]
        if i in special_ids:
            types.append(gguf.TokenType.CONTROL)
        elif len(t) == 6 and t.startswith("<0x") and t.endswith(">"):
            types.append(gguf.TokenType.BYTE)
        else:
            types.append(gguf.TokenType.NORMAL)
    merges = [f"{a} {b}" for a, b in model["merges"]]
    writer.add_tokenizer_model("breeze-bpe")
    writer.add_token_list(tokens)
    writer.add_token_types(types)
    writer.add_token_merges(merges)
    writer.add_bos_token_id(2)
    writer.add_eos_token_id(1)
    writer.add_pad_token_id(0)
    writer.add_unk_token_id(3)
    writer.add_bool("tokenizer.ggml.add_bos_token", False)


def write_metadata(writer, cfg):
    te = cfg["text_encoder_config"]
    bb = cfg["backbone_config"]
    dd = cfg["depth_decoder_config"]
    k = writer.add_uint32
    f = writer.add_float32
    writer.add_name("Breeze-TTS-2")
    k("breeze.hidden_size", cfg["hidden_size"])
    k("breeze.num_codebooks", cfg["num_codebooks"])
    k("breeze.audio_vocab_size", cfg["vocab_size"])
    k("breeze.codec_codebook_size", cfg["codec_config"]["codebook_size"])
    k("breeze.audio_token_id", cfg["audio_token_id"])
    k("breeze.audio_eos_token_id", cfg["audio_eos_token_id"])
    k("breeze.backbone_eos_token_id", cfg["vocab_size"])
    k("breeze.codebook_pad_token_id", cfg["codebook_pad_token_id"])
    k("breeze.codebook_eos_token_id", cfg["codebook_eos_token_id"])
    f("breeze.temperature", 0.9)
    k("breeze.top_k", 50)
    f("breeze.top_p", 1.0)
    f("breeze.repetition_penalty", 1.1)
    f("breeze.depth_temperature", 0.9)
    k("breeze.depth_top_k", 50)
    f("breeze.depth_top_p", 1.0)
    k("breeze.max_new_tokens", 750)

    def group(p, c, extra=None):
        k(f"{p}.hidden_size", c["hidden_size"])
        k(f"{p}.block_count", c["num_hidden_layers"])
        k(f"{p}.head_count", c["num_attention_heads"])
        k(f"{p}.head_count_kv", c["num_key_value_heads"])
        k(f"{p}.head_dim", c.get("head_dim", c["hidden_size"] // c["num_attention_heads"]))
        k(f"{p}.feed_forward_length", c["intermediate_size"])
        f(f"{p}.rms_eps", c.get("rms_norm_eps", 1e-6))
        (extra or (lambda: None))()

    group("breeze.te", te, lambda: [
        f("breeze.te.rope_theta_full", 1000000.0),
        f("breeze.te.rope_freq_scale_full", 1.0 / 8.0),
        f("breeze.te.rope_theta_sliding", 10000.0),
        k("breeze.te.sliding_window", te["sliding_window"]),
        f("breeze.te.query_pre_attn_scalar", float(te["query_pre_attn_scalar"])),
        f("breeze.te.embed_scale", math.sqrt(te["hidden_size"])),
        writer.add_array("breeze.te.layer_is_full",
                         [1 if t == "full_attention" else 0 for t in te["layer_types"]]),
    ] and None)
    group("breeze.bb", bb, lambda: f("breeze.bb.rope_theta", bb["rope_theta"]))
    rs = dd["rope_scaling"]
    group("breeze.dd", dd, lambda: [
        f("breeze.dd.rope_theta", dd["rope_theta"]),
        f("breeze.dd.rope_factor", rs["factor"]),
        f("breeze.dd.rope_high_freq_factor", rs["high_freq_factor"]),
        f("breeze.dd.rope_low_freq_factor", rs["low_freq_factor"]),
        k("breeze.dd.rope_orig_ctx", rs["original_max_position_embeddings"]),
    ] and None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir", type=Path)
    ap.add_argument("-o", "--output", type=Path, required=True)
    ap.add_argument("--dtype", choices=["f16", "f32"], default="f16")
    args = ap.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import convert_codec

    cfg = json.loads((args.model_dir / "config.json").read_text())
    reader = ShardReader(args.model_dir)
    writer = gguf.GGUFWriter(str(args.output), ARCH)

    write_metadata(writer, cfg)
    write_tokenizer(writer, args.model_dir)
    write_text_encoder(writer, reader, cfg, args.dtype)
    write_backbone(writer, reader, cfg, args.dtype)
    write_depth(writer, reader, cfg, args.dtype)
    at_cfg = write_codec_all(writer, args.model_dir, args.dtype)
    convert_codec.codec_metadata(writer, at_cfg)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
