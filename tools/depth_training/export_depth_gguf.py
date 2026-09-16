#!/usr/bin/env python3
"""Export distilled 9-block depth decoder into full GGUF models (F16 and Q4_K)."""
import argparse
import os
from pathlib import Path
import subprocess
import numpy as np
import torch
import gguf


def copy_metadata(reader: gguf.GGUFReader, writer: gguf.GGUFWriter, n_blocks: int = 9):
    for k, f in reader.fields.items():
        if k in ("GGUF.version", "GGUF.tensor_count", "GGUF.kv_count", "general.architecture"):
            continue
        val_type = f.types[0]
        if k == "breeze.dd.block_count":
            writer.add_uint32(k, n_blocks)
            continue
        if val_type == gguf.GGUFValueType.STRING:
            s = bytes(f.parts[f.data[0]]).decode("utf-8")
            writer.add_string(k, s)
        elif val_type == gguf.GGUFValueType.UINT32:
            writer.add_uint32(k, int(f.parts[f.data[0]][0]))
        elif val_type == gguf.GGUFValueType.INT32:
            writer.add_int32(k, int(f.parts[f.data[0]][0]))
        elif val_type == gguf.GGUFValueType.FLOAT32:
            writer.add_float32(k, float(f.parts[f.data[0]][0]))
        elif val_type == gguf.GGUFValueType.BOOL:
            writer.add_bool(k, bool(f.parts[f.data[0]][0]))
        elif val_type == gguf.GGUFValueType.ARRAY:
            elem_type = f.types[1]
            arr = [f.parts[idx] for idx in f.data]
            if elem_type == gguf.GGUFValueType.STRING:
                str_arr = [bytes(x).decode("utf-8") for x in arr]
                writer.add_array(k, str_arr)
            elif elem_type == gguf.GGUFValueType.INT32:
                i_arr = [int(x[0]) for x in arr]
                writer.add_array(k, i_arr)
            elif elem_type == gguf.GGUFValueType.FLOAT32:
                f_arr = [float(x[0]) for x in arr]
                writer.add_array(k, f_arr)


def export_gguf(
    base_gguf: str,
    checkpoint_pt: str,
    out_f16_gguf: str,
    out_q4k_gguf: str = None,
    breeze_quantize_bin: str = "/mnt/media/breeze-teacher/build/breeze-quantize",
):
    print(f"Reading base model from {base_gguf}...")
    reader = gguf.GGUFReader(base_gguf)

    print(f"Loading student checkpoint from {checkpoint_pt}...")
    ckpt = torch.load(checkpoint_pt, map_location="cpu")
    state_dict = ckpt.get("model_state_dict", ckpt)

    n_blocks = ckpt.get("n_layer", 9)
    print(f"Exporting student with {n_blocks} depth blocks to {out_f16_gguf}...")

    Path(out_f16_gguf).parent.mkdir(parents=True, exist_ok=True)
    writer = gguf.GGUFWriter(out_f16_gguf, "breeze")

    # 1. Copy metadata with block count = 9
    copy_metadata(reader, writer, n_blocks=n_blocks)

    # 2. Build tensor mapping from PyTorch state dict to GGUF tensor names
    # PyTorch names -> GGUF names
    pt_to_gguf = {}
    pt_to_gguf["in_proj.weight"] = "dd.in_proj.weight"
    pt_to_gguf["output_norm.weight"] = "dd.output_norm.weight"
    pt_to_gguf["codebooks_head"] = "dd.codebooks_head.weight"

    for i in range(n_blocks):
        p_pt = f"blocks.{i}."
        p_gguf = f"dd.blk.{i}."
        pt_to_gguf[p_pt + "attn_norm.weight"] = p_gguf + "attn_norm.weight"
        pt_to_gguf[p_pt + "attn_q.weight"] = p_gguf + "attn_q.weight"
        pt_to_gguf[p_pt + "attn_k.weight"] = p_gguf + "attn_k.weight"
        pt_to_gguf[p_pt + "attn_v.weight"] = p_gguf + "attn_v.weight"
        pt_to_gguf[p_pt + "attn_output.weight"] = p_gguf + "attn_output.weight"
        pt_to_gguf[p_pt + "ffn_norm.weight"] = p_gguf + "ffn_norm.weight"
        pt_to_gguf[p_pt + "ffn_gate.weight"] = p_gguf + "ffn_gate.weight"
        pt_to_gguf[p_pt + "ffn_up.weight"] = p_gguf + "ffn_up.weight"
        pt_to_gguf[p_pt + "ffn_down.weight"] = p_gguf + "ffn_down.weight"

    gguf_to_pt = {v: k for k, v in pt_to_gguf.items()}

    # 3. Add tensors
    skipped = 0
    replaced = 0
    passed = 0

    for t in reader.tensors:
        # Skip omitted depth layers (blocks 9, 10, 11)
        if any(t.name.startswith(f"dd.blk.{l}.") for l in range(n_blocks, 16)):
            skipped += 1
            continue

        if t.name in gguf_to_pt:
            # Replace with trained student weight
            pt_key = gguf_to_pt[t.name]
            tensor_val = state_dict[pt_key].cpu().float().numpy()

            # Preserve appropriate GGUF dtypes (F32 for norms, F16 for weights)
            if "norm" in t.name:
                arr = tensor_val.astype(np.float32)
            else:
                arr = tensor_val.astype(np.float16)

            writer.add_tensor(t.name, arr)
            replaced += 1
        else:
            # Pass through base tensor untouched
            writer.add_tensor(t.name, t.data)
            passed += 1

    print(f"Tensor summary: {passed} untouched, {replaced} replaced with student, {skipped} pruned.")

    print(f"Writing {out_f16_gguf}...")
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"Successfully exported {out_f16_gguf} ({os.path.getsize(out_f16_gguf) / (1024**3):.2f} GB)")

    # 4. Quantize to Q4_K if requested
    if out_q4k_gguf:
        print(f"\nQuantizing {out_f16_gguf} -> {out_q4k_gguf} (Q4_K)...")
        cmd = [breeze_quantize_bin, out_f16_gguf, out_q4k_gguf, "q4_k"]
        subprocess.run(cmd, check=True)
        print(f"Successfully quantized {out_q4k_gguf} ({os.path.getsize(out_q4k_gguf) / (1024**3):.2f} GB)")


def main():
    parser = argparse.ArgumentParser(description="Export 9-block depth student to GGUF")
    parser.add_argument("--base-gguf", default="/mnt/media/breeze-teacher/models/teacher-f16.gguf")
    parser.add_argument("--checkpoint", default="/mnt/media/breeze-teacher/depth-checkpoints/best_student_9block.pt")
    parser.add_argument("--out-f16", default="/mnt/media/breeze-teacher/models/breeze-dd9-f16.gguf")
    parser.add_argument("--out-q4k", default="/mnt/media/breeze-teacher/models/breeze-dd9-q4_k.gguf")
    parser.add_argument("--breeze-quantize", default="/mnt/media/breeze-teacher/build/breeze-quantize")
    args = parser.parse_args()

    export_gguf(
        base_gguf=args.base_gguf,
        checkpoint_pt=args.checkpoint,
        out_f16_gguf=args.out_f16,
        out_q4k_gguf=args.out_q4k,
        breeze_quantize_bin=args.breeze_quantize,
    )


if __name__ == "__main__":
    main()
