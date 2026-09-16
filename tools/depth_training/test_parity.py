#!/usr/bin/env python3
"""Test numerical parity between PyTorch DepthDecoder and GGUF Vulkan capture."""
import argparse
import numpy as np
import torch
import gguf

from model import DepthDecoder


def load_weights_into_model(model: DepthDecoder, gguf_path: str, device: str = "cpu"):
    print(f"Loading GGUF tensors from {gguf_path}...")
    reader = gguf.GGUFReader(gguf_path)
    tensors = {t.name: t for t in reader.tensors}

    def get_tensor(name):
        if name not in tensors:
            raise KeyError(f"Tensor {name} not found in GGUF")
        arr = tensors[name].data
        # GGUF tensor data may need conversion to float32
        return torch.from_numpy(arr.astype(np.float32)).to(device)

    # Embedding and in_proj
    with torch.no_grad():
        model.audio_embd.weight.copy_(get_tensor("audio_embd.weight"))
        model.in_proj.weight.copy_(get_tensor("dd.in_proj.weight"))
        model.output_norm.weight.copy_(get_tensor("dd.output_norm.weight"))
        model.codebooks_head.copy_(get_tensor("dd.codebooks_head.weight"))

        for i, block in enumerate(model.blocks):
            p = f"dd.blk.{i}"
            block.attn_norm.weight.copy_(get_tensor(f"{p}.attn_norm.weight"))
            block.attn_q.weight.copy_(get_tensor(f"{p}.attn_q.weight"))
            block.attn_k.weight.copy_(get_tensor(f"{p}.attn_k.weight"))
            block.attn_v.weight.copy_(get_tensor(f"{p}.attn_v.weight"))
            block.attn_output.weight.copy_(get_tensor(f"{p}.attn_output.weight"))

            block.ffn_norm.weight.copy_(get_tensor(f"{p}.ffn_norm.weight"))
            block.ffn_gate.weight.copy_(get_tensor(f"{p}.ffn_gate.weight"))
            block.ffn_up.weight.copy_(get_tensor(f"{p}.ffn_up.weight"))
            block.ffn_down.weight.copy_(get_tensor(f"{p}.ffn_down.weight"))

    print(f"Loaded {len(model.blocks)} blocks successfully.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gguf", default="/mnt/media/breeze-teacher/models/teacher-f16.gguf")
    parser.add_argument("--val-bin", default="/mnt/media/breeze-teacher/depth-corpus/validation.bin")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = args.device
    print(f"Using device: {device}")

    # Build 12-layer teacher model
    model = DepthDecoder(n_layer=12).to(device)
    model.eval()

    load_weights_into_model(model, args.gguf, device=device)

    # Read record 0 from validation.bin
    d = np.dtype([
        ("cb0", np.int32),
        ("h0", np.float32, (2048,)),
        ("codes", np.int32, (15,)),
        ("logits", np.float32, (15, 2051))
    ])
    memmap = np.memmap(args.val_bin, dtype=d, mode="r")
    print(f"Validation dataset has {len(memmap)} records.")

    rec0 = memmap[0]
    cb0 = torch.tensor([rec0["cb0"]], dtype=torch.long, device=device)
    h0 = torch.from_numpy(rec0["h0"].copy()).unsqueeze(0).to(device)
    codes = torch.from_numpy(rec0["codes"].copy()).unsqueeze(0).long().to(device)
    target_logits = torch.from_numpy(rec0["logits"].copy()).unsqueeze(0).to(device)

    with torch.no_grad():
        pred_logits = model(cb0, h0, codes)

    diff = (pred_logits - target_logits).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    pred_top1 = pred_logits.argmax(dim=-1)
    target_top1 = target_logits.argmax(dim=-1)
    top1_match = (pred_top1 == target_top1).float().mean().item()

    print(f"\n=== Parity Check Results (Record 0) ===")
    print(f"Max absolute diff:  {max_diff:.6f}")
    print(f"Mean absolute diff: {mean_diff:.6f}")
    print(f"Top-1 match:        {top1_match * 100:.2f}% (pred vs teacher capture)")
    print(f"Pred Top-1:   {pred_top1[0].cpu().numpy().tolist()}")
    print(f"Target Top-1: {target_top1[0].cpu().numpy().tolist()}")

    if max_diff < 0.05 and top1_match == 1.0:
        print("\n>>> SUCCESS: Full numerical parity verified between PyTorch and C++/Vulkan GGML! <<<")
    else:
        print("\n>>> Discrepancy observed. Inspect details above. <<<")


if __name__ == "__main__":
    main()
