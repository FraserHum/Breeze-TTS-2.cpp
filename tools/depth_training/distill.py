#!/usr/bin/env python3
"""Knowledge distillation trainer for 9-block Depth Decoder.

N-voice corpus (per-voice bins from capture_dataset.py's per-voice subdirs):
    distill.py --train-bin <out>/calliope/train.bin <out>/steward/train.bin \
               --val-bin   <out>/calliope/validation.bin <out>/steward/validation.bin
Voice names default to each bin's parent directory name; pass --voice
(parallel to the bins) to override. Single-voice usage is unchanged.
"""
import argparse
import datetime
import json
import math
import os
from pathlib import Path
import platform
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import gguf

from model import DepthDecoder

RECORD_DTYPE = np.dtype([
    ("cb0", np.int32),
    ("h0", np.float32, (2048,)),
    ("codes", np.int32, (15,)),
    ("logits", np.float32, (15, 2051))
])


class DepthRecordDataset(Dataset):
    def __init__(self, bin_path: str, voice: str = ""):
        self.path = bin_path
        if not os.path.exists(bin_path):
            raise FileNotFoundError(f"Binary records file not found: {bin_path}")
        self.voice = voice
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


def build_dataset(bin_paths, voices=None):
    """Build per-voice DepthRecordDatasets from bin paths.

    voice names default to the bin's parent directory (the per-voice subdir
    written by capture_dataset.py); pass an explicit list to override.
    """
    ds_list = []
    for i, p in enumerate(bin_paths):
        p = str(p)
        v = voices[i] if (voices is not None and i < len(voices) and voices[i]) else Path(p).parent.name
        ds_list.append(DepthRecordDataset(p, voice=v))
    return ds_list


def combine_metrics(per_voice):
    """Aggregate per-voice eval metrics into combined metrics.

    Exact (order-independent): loss/kl/ce are per-record means and top-1
    values are per-(record,step) fractions, so the combined value is the
    record-count-weighted average of the per-voice values.
    """
    counts = {v: m["_n_records"] for v, m in per_voice.items()}
    n_total = sum(counts.values())
    combined = {}
    for key in ("loss", "kl", "ce", "top1_acc_codes", "top1_acc_teacher"):
        combined[key] = sum(m[key] * counts[v] for v, m in per_voice.items()) / n_total
    return combined


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
        "_n_records": total_samples,
    }


def write_receipt(out_dir: Path, args, train_counts: dict, val_counts: dict,
                  baseline: dict, baseline_voice: dict,
                  best_epoch: int, best_val_metrics: dict, best_per_voice: dict):
    """Persist distill_receipt.json at --output-dir.

    Contract for consumers (breeze-distill pipeline, safety-gate derivation):
    - per_voice_val: per-voice {loss, kl, ce, top1_acc_codes, top1_acc_teacher, _n_records}
      at the best checkpoint's epoch (combined = count-weighted aggregate via combine_metrics)
    - baseline_per_voice: untrained 9-block truncation, same keys
    - All *_acc_* values are percentages (0..100).
    """
    receipt = {
        "tool": "distill.py",
        "version": 1,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "host": platform.node(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": args.device,
        "n_layer": 9,
        "args": {
            "teacher_gguf": args.teacher_gguf,
            "output_dir": args.output_dir,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "lr": args.lr,
            "min_lr": args.min_lr,
            "weight_decay": args.weight_decay,
            "temperature": args.temperature,
            "alpha_kd": args.alpha_kd,
            "train_bin": args.train_bin,
            "val_bin": args.val_bin,
        },
        "train_counts": train_counts,
        "val_counts": val_counts,
        "baseline": {
            "per_voice": baseline_voice,
            "combined": baseline,
        },
        "best": {
            "epoch": best_epoch,
            "combined": best_val_metrics,
            "per_voice_val": best_per_voice,
        },
    }
    path = out_dir / "distill_receipt.json"
    with open(path, "w") as f:
        json.dump(receipt, f, indent=2)
    print(f"Receipt written: {path}")


def main():
    parser = argparse.ArgumentParser(description="Distill 9-block depth decoder")
    # N-voice: pass one bin per voice (parallel --voice labels). Single voice unchanged.
    parser.add_argument("--train-bin", nargs="+", default=["/mnt/media/breeze-teacher/depth-corpus/train.bin"])
    parser.add_argument("--val-bin", nargs="+", default=["/mnt/media/breeze-teacher/depth-corpus/validation.bin"])
    parser.add_argument("--voice", nargs="+", default=None,
                        help="Parallel to --train-bin/--val-bin voice labels; defaults to each bin's parent dir name")
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

    # Datasets (N-voice: one bin per voice, concatenated; per-voice val splits)
    train_ds_list = build_dataset(args.train_bin, args.voice)
    val_ds_list = build_dataset(args.val_bin, args.voice)
    # Guards: train and val voices must be parallel; each voice exactly once
    if [ds.voice for ds in train_ds_list] != [ds.voice for ds in val_ds_list]:
        raise SystemExit(f"Voice labels mismatch between train/val bins: "
                         f"{[d.voice for d in train_ds_list]} vs {[d.voice for d in val_ds_list]}")
    for name, ds_list in (("train", train_ds_list), ("val", val_ds_list)):
        voices = [ds.voice for ds in ds_list]
        if len(set(voices)) != len(voices):
            raise SystemExit(f"Duplicate voice labels in {name} bins: {voices}")
        paths = [str(ds.path) for ds in ds_list]
        if len(set(paths)) != len(paths):
            raise SystemExit(f"Duplicate {name} bin paths: {paths}")

    print("Loading training data...")
    for ds in train_ds_list:
        print(f"  train [{ds.voice}]: {len(ds)} records ({ds.path})")
    print("Loading validation data...")
    for ds in val_ds_list:
        print(f"  val   [{ds.voice}]: {len(ds)} records ({ds.path})")

    train_ds = ConcatDataset(train_ds_list)
    print(f"Train dataset: {len(train_ds)} records (combined).")
    if len(train_ds_list) > 1:
        train_counts = {ds.voice: len(ds) for ds in train_ds_list}
        print(f"Per-voice train counts: {train_counts}")
        if len(set(train_counts.values())) > 1:
            print("WARNING: per-voice train record counts differ — the N-voice protocol expects "
                  "the same prompt set for every voice; investigate before trusting per-voice metrics.")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True
    )
    val_loaders = {ds.voice: DataLoader(
        ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    ) for ds in val_ds_list}
    val_counts = {ds.voice: len(ds) for ds in val_ds_list}

    # Initialize 9-block student model
    student = DepthDecoder(n_layer=9).to(device)
    init_student_from_teacher(student, args.teacher_gguf, device=device)

    # Baseline evaluation before training (per-voice, combined aggregated from them)
    print("\n--- Evaluating untrained 9-block student (layers 0..8 direct truncate) ---")
    baseline_voice = {v: evaluate(student, ldr, device, temperature=args.temperature, alpha_kd=args.alpha_kd)
                     for v, ldr in val_loaders.items()}
    for v, m in baseline_voice.items():
        print(f"Untrained Baseline [{v}]: Loss={m['loss']:.4f} (KL={m['kl']:.4f}, CE={m['ce']:.4f}) | Top-1 vs Codes: {m['top1_acc_codes']:.2f}% | Top-1 vs Teacher: {m['top1_acc_teacher']:.2f}%")
    baseline = combine_metrics(baseline_voice)
    print(f"Untrained Baseline [combined]: Loss={baseline['loss']:.4f} (KL={baseline['kl']:.4f}, CE={baseline['ce']:.4f}) | Top-1 vs Codes: {baseline['top1_acc_codes']:.2f}% | Top-1 vs Teacher: {baseline['top1_acc_teacher']:.2f}%")
    print()

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
    best_epoch = None
    best_val_metrics = None
    best_per_voice = {}

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

        # Validation: per-voice passes only; combined is the record-count-
        # weighted aggregate of them (exact, no second forward pass over the data)
        per_voice = {}
        for v, ldr in val_loaders.items():
            vm = evaluate(student, ldr, device, temperature=args.temperature, alpha_kd=args.alpha_kd)
            per_voice[v] = vm
            print(
                f"    [{v}] Val Loss={vm['loss']:.4f} (KL={vm['kl']:.4f}, CE={vm['ce']:.4f}) | "
                f"Top-1 (Codes)={vm['top1_acc_codes']:.2f}% | Top-1 (Teacher)={vm['top1_acc_teacher']:.2f}%"
            )
        val_metrics = combine_metrics(per_voice)

        current_lr = scheduler.get_last_lr()[0]
        print(
            f"Epoch [{epoch:2d}/{args.epochs:2d}] ({elapsed:.1f}s, lr={current_lr:.2e}): "
            f"Train Loss={avg_train_loss:.4f} (KL={avg_train_kl:.4f}, CE={avg_train_ce:.4f}) | "
            f"Val Loss={val_metrics['loss']:.4f} (KL={val_metrics['kl']:.4f}, CE={val_metrics['ce']:.4f}) | "
            f"Val Top-1 (Codes)={val_metrics['top1_acc_codes']:.2f}% | Val Top-1 (Teacher)={val_metrics['top1_acc_teacher']:.2f}%"
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch
            best_val_metrics = val_metrics
            best_per_voice = per_voice
            torch.save({
                "epoch": epoch,
                "model_state_dict": student.state_dict(),
                "val_metrics": val_metrics,
                "per_voice_val": per_voice,
                "n_layer": 9,
            }, best_ckpt_path)
            print(f"  --> Saved new best checkpoint to {best_ckpt_path} (Val Loss: {best_val_loss:.4f})")

    print("\n=== Distillation Training Complete ===")
    print(f"Best validation loss: {best_val_loss:.4f} saved at {best_ckpt_path}")

    write_receipt(
        out_dir, args,
        train_counts={ds.voice: len(ds) for ds in train_ds_list},
        val_counts=val_counts,
        baseline=baseline,
        baseline_voice=baseline_voice,
        best_epoch=best_epoch,
        best_val_metrics=val_metrics,
        best_per_voice=best_per_voice,
    )


if __name__ == "__main__":
    main()
