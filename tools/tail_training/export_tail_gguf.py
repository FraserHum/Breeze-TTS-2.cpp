#!/usr/bin/env python3
"""Export trained student vocoder tail weights to GGUF format for C++ serving.

Usage:
  python export_tail_gguf.py --weights-dir RUN/best-weights --output breeze-tail-adam.gguf
"""
import argparse
import json
from pathlib import Path
import numpy as np

try:
    import gguf
except ImportError:
    raise ImportError("gguf python package is required. Install via pip install gguf.")


def export_to_gguf(weights_dir: Path, output_path: Path, prefix: str = "tail.", use_f16: bool = False):
    weights_json = weights_dir / "weights.json"
    if not weights_json.exists():
        raise FileNotFoundError(f"Missing weights manifest: {weights_json}")

    entries = json.loads(weights_json.read_text())
    if isinstance(entries, dict):
        entries = entries.get("weights", entries)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    part_path = output_path.with_suffix(".gguf.part")

    writer = gguf.GGUFWriter(part_path, "breeze-tail")
    writer.add_uint32("tail.n_layer", 4)
    writer.add_uint32("tail.hidden_dim", 512)
    writer.add_uint32("tail.input_dim", 1024)
    writer.add_uint32("tail.output_samples", 1920)

    # Detect head_dim
    head_dim = 1024
    for e in entries:
        if e["name"] in ("codec.dtf.out_proj.bias", "tail.out_proj.bias"):
            head_dim = e["shape"][0]
            break
    writer.add_uint32("tail.head_dim", head_dim)

    # Remap names to clean tail.* namespace
    name_remap = {
        "codec.dtf.in_proj.weight": "tail.in_proj.weight",
        "codec.dtf.in_proj.bias": "tail.in_proj.bias",
        "codec.dtf.norm.weight": "tail.norm.weight",
        "codec.dtf.out_proj.weight": "tail.out_proj.weight",
        "codec.dtf.out_proj.bias": "tail.out_proj.bias",
        "student.output.weight": "tail.output.weight",
    }
    for i in range(4):
        src_p = f"codec.dtf.blk.{i}."
        dst_p = f"tail.blk.{i}."
        for suffix in (
            "attn_norm.weight", "ffn_norm.weight", "attn_scale", "ffn_scale",
            "attn_q.weight", "attn_k.weight", "attn_v.weight",
            "ffn_gate.weight", "ffn_up.weight", "attn_output.weight", "ffn_down.weight"
        ):
            name_remap[src_p + suffix] = dst_p + suffix

    tensor_count = 0
    total_bytes = 0
    for entry in entries:
        src_name = entry["name"]
        dst_name = name_remap.get(src_name, src_name if src_name.startswith(prefix) else prefix + src_name)
        file_path = weights_dir / entry["file"]
        ggml_shape = entry["shape"]
        # In ggml shape is innermost-first; in numpy/torch shape is outermost-first
        np_shape = tuple(reversed(ggml_shape))

        data = np.fromfile(file_path, dtype="<f4")
        if data.size != int(np.prod(ggml_shape)):
            raise ValueError(f"Tensor {src_name} size mismatch: expected {np.prod(ggml_shape)}, got {data.size}")

        tensor_data = data.reshape(np_shape)
        if use_f16:
            tensor_data = tensor_data.astype(np.float16)

        writer.add_tensor(dst_name, tensor_data)
        tensor_count += 1
        total_bytes += tensor_data.nbytes

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    part_path.rename(output_path)
    print(f"Exported {tensor_count} tensors ({total_bytes / (1024*1024):.2f} MB) to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Export student tail weights to GGUF")
    parser.add_argument("--weights-dir", type=Path, required=True, help="Path to best-weights directory")
    parser.add_argument("--output", type=Path, required=True, help="Path to output .gguf file")
    parser.add_argument("--prefix", type=str, default="tail.", help="Tensor name prefix")
    parser.add_argument("--f16", action="store_true", help="Store weights in F16 instead of F32")
    args = parser.parse_args()

    export_to_gguf(args.weights_dir, args.output, prefix=args.prefix, use_f16=args.f16)


if __name__ == "__main__":
    main()
