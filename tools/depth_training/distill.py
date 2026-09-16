#!/usr/bin/env python3
"""Knowledge distillation trainer for 9-block Depth Decoder."""
import argparse
import math
import os
from pathlib import Path
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import gguf

from model import DepthDecoder

RECORD_DTYPE = np.dtype([
    ("cb0", np.int32),
    ("h0", np.float32, (2048,)),
    ("codes", np.int32, (15,)),
    ("logits", np.float32, (15, 2051))
])


class DepthRecordDataset(Dataset):
    def __init__(self, bin_path: str):
        self.path = bin_path
        if not os.path.exists(bin_path):
            raise FileNotFoundError(f"Binary records file not found: {bin_path}")
        self.memmap = np.memmap(bin_path, dtype=RECORD_DTYPE, mode="r")
        self.length = len(self.memmap)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int):
        rec = self.memmap[idx]
        return {
            "cb0": torch.tensor(rec["cb0"], dtype=torch.long),
            "h0": torch.from_numpy(rec["h0"].copy()),  # [2048]
            "codes": torch.from_numpy(rec["codes"].copy()).long(),  # [15]
            "teacher_logits": torch.from_numpy(rec["logits"].copy())  # [15, 2051]
        }


def init_student_from_teacher(student: DepthDecoder, gguf_path: str, device: str = "cpu"):
    print(f"Initializing student (9 blocks) from teacher GGUF: {gguf_path}...")
    reader = gguf.GGUFReader(gguf_path)
    tensors = {t.name: t for t in reader.tensors}

    def get_tensor(name):
        if name not in tensors:
            raise KeyError(f"Tensor {name} not found in GGUF")
        arr = tensors[name].data
        return torch.from_numpy(arr.astype(np.float32)).to(device)

    with torch.no_grad():
        student.audio_embd.weight.copy_(get_tensor("audio_embd.weight"))
        student.in_proj.weight.copy_(get_tensor("dd.in_proj.weight"))
        student.output_norm.weight.copy_(get_tensor("dd.output_norm.weight"))
        student.codebooks_head.copy_(get_tensor("dd.codebooks_head.weight"))

        # Copy first 9 blocks (dd.blk.0 .. dd.blk.8)
        for i in range(len(student.blocks)):
            p = f"dd.blk.{i}"
            student.blocks[i].attn_norm.weight.copy_(get_tensor(f"{p}.attn_norm.weight"))
            student.blocks[i].attn_q.weight.copy_(get_tensor(f"{p}.attn_q.weight"))
            student.blocks[i].attn_k.weight.copy_(get_tensor(f"{p}.attn_k.weight"))
            student.blocks[i].attn_v.weight.copy_(get_tensor(f"{p}.attn_v.weight"))
            student.blocks[i].attn_output.weight.copy_(get_tensor(f"{p}.attn_output.weight"))

            student.blocks[i].ffn_norm.weight.copy_(get_tensor(f"{p}.ffn_norm.weight"))
            student.blocks[i].ffn_gate.weight.copy_(get_tensor(f"{p}.ffn_gate.weight"))
            student.blocks[i].ffn_up.weight.copy_(get_tensor(f"{p}.ffn_up.weight"))
            student.blocks[i].ffn_down.weight.copy_(get_tensor(f"{p}.ffn_down.weight"))

    # Freeze audio_embd since it is shared with backbone
    student.audio_embd.weight.requires_grad_(False)
    print("Student initialization complete (layers 0..8 copied, audio_embd frozen).")


def compute_distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    target_codes: torch.Tensor,
    temperature: float = 1.0,
    alpha_kd: float = 0.8,
):
    """
    Args:
        student_logits: [B, 15, 2051]
        teacher_logits: [B, 15, 2051]
        target_codes:   [B, 15]
        temperature:    softening temperature for KD
        alpha_kd:       weight for KD loss (1 - alpha_kd for hard CE)
    """
    B, T, V = student_logits.shape

    # 1. Soft KL Divergence loss:
    # Scale logits by temperature
    s_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    t_probs = F.softmax(teacher_logits / temperature, dim=-1)
    # batchmean computes sum over all elements divided by B
    kl_loss = F.kl_div(s_log_probs, t_probs, reduction="batchmean") * (temperature ** 2) / T

    # 2. Hard Cross Entropy loss on teacher sampled codes:
    ce_loss = F.cross_entropy(
        student_logits.reshape(-1, V),
        target_codes.reshape(-1),
        reduction="mean"
    )

    total_loss = alpha_kd * kl_loss + (1.0 - alpha_kd) * ce_loss
    return total_loss, kl_loss, ce_loss


def evaluate(model: DepthDecoder, val_loader: DataLoader, device: str, temperature: float = 1.0, alpha_kd: float = 0.8):
    model.eval()
    total_loss = 0.0
    total_kl = 0.0
    total_ce = 0.0
    total_top1_match_codes = 0.0
    total_top1_match_teacher = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch in val_loader:
            cb0 = batch["cb0"].to(device)
            h0 = batch["h0"].to(device)
            codes = batch["codes"].to(device)
            t_logits = batch["teacher_logits"].to(device)

            s_logits = model(cb0, h0, codes)

            loss, kl, ce = compute_distillation_loss(
                s_logits, t_logits, codes, temperature=temperature, alpha_kd=alpha_kd
            )

            B = cb0.shape[0]
            total_loss += loss.item() * B
            total_kl += kl.item() * B
            total_ce += ce.item() * B

            s_top1 = s_logits.argmax(dim=-1)
            t_top1 = t_logits.argmax(dim=-1)

            match_codes = (s_top1 == codes).float().mean(dim=-1).sum().item()
            match_teacher = (s_top1 == t_top1).float().mean(dim=-1).sum().item()

            total_top1_match_codes += match_codes
            total_top1_match_teacher += match_teacher
            total_samples += B

    return {
        "loss": total_loss / total_samples,
        "kl": total_kl / total_samples,
        "ce": total_ce / total_samples,
        "top1_acc_codes": (total_top1_match_codes / total_samples) * 100.0,
        "top1_acc_teacher": (total_top1_match_teacher / total_samples) * 100.0,
    }


def main():
    parser = argparse.ArgumentParser(description="Distill 9-block depth decoder")
    parser.add_argument("--train-bin", default="/mnt/media/breeze-teacher/depth-corpus/train.bin")
    parser.add_argument("--val-bin", default="/mnt/media/breeze-teacher/depth-corpus/validation.bin")
    parser.add_argument("--teacher-gguf", default="/mnt/media/breeze-teacher/models/teacher-f16.gguf")
    parser.add_argument("--output-dir", default="/mnt/media/breeze-teacher/depth-checkpoints")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--alpha-kd", type=float, default=0.8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = args.device
    print(f"Training on device: {device}")

    # Datasets
    print(f"Loading training data from {args.train_bin}...")
    train_ds = DepthRecordDataset(args.train_bin)
    print(f"Train dataset: {len(train_ds)} records.")

    print(f"Loading validation data from {args.val_bin}...")
    val_ds = DepthRecordDataset(args.val_bin)
    print(f"Val dataset: {len(val_ds)} records.")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )

    # Initialize 9-block student model
    student = DepthDecoder(n_layer=9).to(device)
    init_student_from_teacher(student, args.teacher_gguf, device=device)

    # Baseline evaluation before training
    print("\n--- Evaluating untrained 9-block student (layers 0..8 direct truncate) ---")
    baseline = evaluate(student, val_loader, device, temperature=args.temperature, alpha_kd=args.alpha_kd)
    print(f"Untrained Baseline: Loss={baseline['loss']:.4f} (KL={baseline['kl']:.4f}, CE={baseline['ce']:.4f}) | Top-1 vs Codes: {baseline['top1_acc_codes']:.2f}% | Top-1 vs Teacher: {baseline['top1_acc_teacher']:.2f}%\n")

    # Optimizer & Scheduler
    trainable_params = [p for p in student.parameters() if p.requires_grad]
    n_params = sum(p.numel() for p in trainable_params)
    print(f"Trainable parameters: {n_params / 1e6:.2f}M")

    optimizer = torch.optim.AdamW(
        trainable_params, lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.98)
    )

    total_steps = len(train_loader) * args.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=args.min_lr
    )

    best_val_loss = float("inf")
    best_ckpt_path = out_dir / "best_student_9block.pt"

    for epoch in range(1, args.epochs + 1):
        student.train()
        epoch_loss = 0.0
        epoch_kl = 0.0
        epoch_ce = 0.0
        t0 = time.time()

        for step, batch in enumerate(train_loader):
            cb0 = batch["cb0"].to(device, non_blocking=True)
            h0 = batch["h0"].to(device, non_blocking=True)
            codes = batch["codes"].to(device, non_blocking=True)
            t_logits = batch["teacher_logits"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            s_logits = student(cb0, h0, codes)

            loss, kl, ce = compute_distillation_loss(
                s_logits, t_logits, codes, temperature=args.temperature, alpha_kd=args.alpha_kd
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            epoch_kl += kl.item()
            epoch_ce += ce.item()

        elapsed = time.time() - t0
        n_batches = len(train_loader)
        avg_train_loss = epoch_loss / n_batches
        avg_train_kl = epoch_kl / n_batches
        avg_train_ce = epoch_ce / n_batches

        # Validation
        val_metrics = evaluate(student, val_loader, device, temperature=args.temperature, alpha_kd=args.alpha_kd)

        current_lr = scheduler.get_last_lr()[0]
        print(
            f"Epoch [{epoch:2d}/{args.epochs:2d}] ({elapsed:.1f}s, lr={current_lr:.2e}): "
            f"Train Loss={avg_train_loss:.4f} (KL={avg_train_kl:.4f}, CE={avg_train_ce:.4f}) | "
            f"Val Loss={val_metrics['loss']:.4f} (KL={val_metrics['kl']:.4f}, CE={val_metrics['ce']:.4f}) | "
            f"Val Top-1 (Codes)={val_metrics['top1_acc_codes']:.2f}% | Val Top-1 (Teacher)={val_metrics['top1_acc_teacher']:.2f}%"
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            torch.save({
                "epoch": epoch,
                "model_state_dict": student.state_dict(),
                "val_metrics": val_metrics,
                "n_layer": 9,
            }, best_ckpt_path)
            print(f"  --> Saved new best checkpoint to {best_ckpt_path} (Val Loss: {best_val_loss:.4f})")

    print("\n=== Distillation Training Complete ===")
    print(f"Best validation loss: {best_val_loss:.4f} saved at {best_ckpt_path}")


if __name__ == "__main__":
    main()
