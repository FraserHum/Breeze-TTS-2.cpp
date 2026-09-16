#!/usr/bin/env python3
"""Capture depth decoder distillation dataset (cb0, h0, codes, logits)

Runs teacher model generation over pilot-prompts.json and writes binary records.

N-voice corpus (per-voice record parity, N=2 = 160 prompts per voice):
    python3 capture_dataset.py --voice calliope
    python3 capture_dataset.py --voice steward
    # -> <out-dir>/calliope/{train,validation,test}.bin  (sidecar .bin.json written by the binary)
    # -> <out-dir>/steward/{...}
Then train with distill.py over the per-voice bins (see its --train-bin/--val-bin docstring).
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description="Capture depth decoder teacher records")
    parser.add_argument("--prompts", default="tools/tail_training/pilot-prompts.json")
    parser.add_argument("--breeze-cli", default="/mnt/media/breeze-teacher/build/breeze-cli")
    parser.add_argument("--model", default="/mnt/media/breeze-teacher/models/teacher-f16.gguf")
    parser.add_argument("--voices-dir", default="/mnt/media/breeze-teacher/models")
    parser.add_argument("--voice", default="calliope")
    parser.add_argument("--out-dir", default="/mnt/media/breeze-teacher/depth-corpus")
    parser.add_argument("--splits", nargs="+", default=["train", "validation", "test"])
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    # N-voice: capture each voice into its own subdir so the corpus keeps
    # per-voice record parity and per-voice val/test splits (no mixing).
    # The C++ binary appends "/depth_records.bin" if the path lacks a .bin suffix,
    # so point it directly at the per-split .bin inside the voice dir.
    voice_dir = out_dir / args.voice
    voice_dir.mkdir(parents=True, exist_ok=True)

    with open(args.prompts, "r") as f:
        all_prompts = json.load(f)

    for split in args.splits:
        split_prompts = [p for p in all_prompts if p.get("split") == split]
        if not split_prompts:
            print(f"No prompts found for split '{split}'")
            continue

        bin_path = voice_dir / f"{split}.bin"
        done_marker = voice_dir / f"{split}.done"
        progress_file = voice_dir / f"{split}_progress.json"

        completed_ids = set()
        if progress_file.exists():
            try:
                with open(progress_file, "r") as f:
                    completed_ids = set(json.load(f))
            except (json.JSONDecodeError, OSError) as e:
                print(f"WARNING: corrupt progress file {progress_file} ({e}); treating split as not resumed "
                      f"(re-run will re-capture — bin is append-only, so re-captured prompts duplicate records; "
                      f"delete the split's .bin/.json/.done/_progress files to start clean).")

        if done_marker.exists() and len(completed_ids) == len(split_prompts):
            print(f"Split '{split}' already complete ({len(completed_ids)}/{len(split_prompts)} prompts). Skipping.")
            continue

        if completed_ids:
            # Resume integrity: the C++ binary appends records per prompt and rewrites the
            # sidecar .json only at process exit (records = file size / record_bytes).
            # If the last process died mid-prompt, the bin holds a partial (non-integer
            # record-count) tail that memmap would misread — truncate it, then verify the
            # remaining count matches the completed prompt count.
            if done_marker.exists():
                print(f"Split '{split}' has a .done marker but only {len(completed_ids)}/{len(split_prompts)} "
                      f"prompts in progress; marker lost trust, resuming.")
            sidecar = Path(str(bin_path) + ".json")
            if sidecar.exists():
                try:
                    with open(sidecar) as f:
                        rb = json.load(f)["record_bytes"]
                    raw_size = bin_path.stat().st_size
                    n_recs = raw_size // rb
                    if raw_size % rb != 0:
                        # Partial record tail from a crashed mid-prompt run: memmap would
                        # misread the truncated bytes as a full record, so truncate to the
                        # last complete record before any data is read.
                        with open(bin_path, "r+b") as f:
                            f.truncate(n_recs * rb)
                        print(f"Truncated partial record tail: {raw_size} -> {n_recs * rb} bytes")
                    if n_recs != len(completed_ids):
                        print(f"WARNING: resume count mismatch ({n_recs} complete records in bin vs "
                              f"{len(completed_ids)} completed prompts). Continuing from progress file; "
                              f"verify the corpus before training.")
                except (KeyError, json.JSONDecodeError, OSError) as e:
                    print(f"WARNING: cannot verify bin integrity for resume ({e}); continuing anyway.")

        print(f"\n=== Starting capture for split '{split}': {len(split_prompts)} prompts (resuming from {len(completed_ids)}) ===")

        env = dict(os.environ)
        env["BREEZE_DEPTH_DISTILL_CAPTURE"] = str(bin_path)
        env["BREEZE_DD_FUSED"] = "0"
        env["GGML_VK_VISIBLE_DEVICES"] = "0"

        for idx, prompt in enumerate(split_prompts, 1):
            pid = prompt["id"]
            text = prompt["text"]

            if pid in completed_ids:
                continue

            t0 = time.time()
            cmd = [
                args.breeze_cli,
                args.model,
                "--text", text,
                "--voices-dir", args.voices_dir,
                "--voice", args.voice,
                "--seed", "42",
                "--cfg-scale", "1.0",
                "--output", "/tmp/breeze_capture_audio.wav"
            ]

            proc = subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            elapsed = time.time() - t0

            if proc.returncode != 0:
                print(f"[{idx}/{len(split_prompts)}] FAILED {pid} ({elapsed:.1f}s):\nSTDOUT: {proc.stdout}\nSTDERR: {proc.stderr}")
                sys.exit(1)

            completed_ids.add(pid)
            with open(progress_file, "w") as f:
                json.dump(list(completed_ids), f)

            print(f"[{idx}/{len(split_prompts)}] {pid}: OK ({elapsed:.1f}s)")

        with open(done_marker, "w") as f:
            f.write(f"Completed {len(split_prompts)} prompts\n")
        print(f"=== Completed split '{split}' ===")


if __name__ == "__main__":
    main()
