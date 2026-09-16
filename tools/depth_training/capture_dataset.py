#!/usr/bin/env python3
"""Capture depth decoder distillation dataset (cb0, h0, codes, logits)

Runs teacher model generation over pilot-prompts.json and writes binary records.
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
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.prompts, "r") as f:
        all_prompts = json.load(f)

    for split in args.splits:
        split_prompts = [p for p in all_prompts if p.get("split") == split]
        if not split_prompts:
            print(f"No prompts found for split '{split}'")
            continue

        bin_path = out_dir / f"{split}.bin"
        done_marker = out_dir / f"{split}.done"
        progress_file = out_dir / f"{split}_progress.json"

        completed_ids = set()
        if progress_file.exists():
            try:
                with open(progress_file, "r") as f:
                    completed_ids = set(json.load(f))
            except Exception:
                pass

        if done_marker.exists() and len(completed_ids) == len(split_prompts):
            print(f"Split '{split}' already complete ({len(completed_ids)}/{len(split_prompts)} prompts). Skipping.")
            continue

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
