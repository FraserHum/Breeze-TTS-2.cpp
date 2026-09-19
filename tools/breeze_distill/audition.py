"""Voice Design Audition Reel Generator & Candidate Selector for Breeze-TTS-2.

Solves Voice Design seed variance by generating a batch of candidate utterances
using the standardized phoneme-complete calibration script across multiple RNG seeds.
Allows the user/developer to audition clips and freeze their chosen timbre into a durable
reference WAV + transcript, then pre-encodes to a .breeze voice vector.
"""
import argparse
import json
import math
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple
import wave

from .calibration import DEFAULT_CALIBRATION_PASSAGE, analyze_phoneme_coverage

DEFAULT_SEEDS = [42, 77, 108, 256]


def analyze_wav_file(wav_path: Path) -> Dict[str, Any]:
    """Measure duration, channels, sample rate, peak amplitude, and RMS dBFS of a WAV file."""
    if not wav_path.exists():
        return {"error": f"File not found: {wav_path}"}

    try:
        with wave.open(str(wav_path), "rb") as wf:
            channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            framerate = wf.getframerate()
            n_frames = wf.getnframes()
            duration_s = n_frames / float(framerate) if framerate > 0 else 0.0

            raw_frames = wf.readframes(n_frames)

            # Analyze 16-bit PCM
            if sample_width == 2 and n_frames > 0:
                fmt = f"<{n_frames * channels}h"
                samples = struct.unpack(fmt, raw_frames)
                peak = max(abs(s) for s in samples)
                peak_norm = peak / 32768.0
                sum_sq = sum(s * s for s in samples)
                mean_sq = sum_sq / len(samples)
                rms = math.sqrt(mean_sq)
                rms_norm = rms / 32768.0
                rms_db = 20 * math.log10(rms_norm) if rms_norm > 0 else -100.0
            else:
                peak_norm = 1.0
                rms_db = -18.0

            return {
                "channels": channels,
                "sample_rate": framerate,
                "sample_width": sample_width,
                "n_frames": n_frames,
                "duration_s": round(duration_s, 2),
                "peak_amplitude": round(peak_norm, 3),
                "rms_dbfs": round(rms_db, 1),
            }
    except Exception as e:
        return {"error": str(e)}


def generate_audition_reel(
    instruction: str,
    voice_name: str,
    out_dir: str,
    model_path: str = "models/breeze-tts-2-q4_k.gguf",
    breeze_cli: str = "build/breeze-cli",
    seeds: Optional[List[int]] = None,
    text: str = DEFAULT_CALIBRATION_PASSAGE,
    dry_run: bool = False,
    extra_env: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Generate candidate audio clips for Voice Design across multiple seeds."""
    if seeds is None:
        seeds = list(DEFAULT_SEEDS)

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    report = analyze_phoneme_coverage(text)

    manifest: Dict[str, Any] = {
        "voice_name": voice_name,
        "instruction": instruction,
        "calibration_text": text,
        "phoneme_coverage_pct": round(report.coverage_ratio * 100.0, 1),
        "phonemes_covered": sorted(list(report.covered_phonemes)),
        "missing_phonemes": sorted(list(report.missing_phonemes)),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%SZ", time.gmtime()),
        "candidates": [],
    }

    env = dict(os.environ)
    if extra_env:
        env.update(extra_env)

    print(f"\n=======================================================")
    print(f"Breeze Voice Design Audition Studio: Voice '{voice_name}'")
    print(f"Instruction: \"{instruction}\"")
    print(f"Seeds: {seeds}")
    print(f"Calibration text: \"{text[:60]}...\" (Coverage: {report.covered_phonemes_count}/39)")
    print(f"Output directory: {out_path}")
    print(f"=======================================================\n")

    for idx, seed in enumerate(seeds, 1):
        clip_name = f"{voice_name}_seed{seed}.wav"
        clip_path = out_path / clip_name

        print(f"[{idx}/{len(seeds)}] Generating candidate for seed {seed}...")

        cmd = [
            breeze_cli,
            model_path,
            "--text", text,
            "--instruction", instruction,
            "--seed", str(seed),
            "--output", str(clip_path),
        ]

        if dry_run:
            print(f"  [DRY-RUN] Command: {' '.join(cmd)}")
            # Mock candidate entry
            candidate = {
                "seed": seed,
                "wav_file": clip_name,
                "wav_path": str(clip_path),
                "duration_s": 12.4,
                "sample_rate": 24000,
                "rms_dbfs": -18.2,
                "status": "dry_run",
            }
        else:
            t0 = time.time()
            proc = subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            elapsed = time.time() - t0

            if proc.returncode != 0:
                print(f"  FAILED seed {seed} ({elapsed:.1f}s):\nSTDERR: {proc.stderr}")
                candidate = {
                    "seed": seed,
                    "wav_file": clip_name,
                    "wav_path": str(clip_path),
                    "error": proc.stderr,
                    "status": "failed",
                }
            else:
                stats = analyze_wav_file(clip_path)
                print(f"  OK ({elapsed:.1f}s) -> {clip_name} ({stats.get('duration_s', 0)}s, {stats.get('rms_dbfs', 0)} dBFS)")
                candidate = {
                    "seed": seed,
                    "wav_file": clip_name,
                    "wav_path": str(clip_path),
                    "elapsed_s": round(elapsed, 1),
                    "status": "ready",
                    **stats,
                }

        manifest["candidates"].append(candidate)

    manifest_file = out_path / "audition_reel.json"
    with open(manifest_file, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nAudition reel saved to: {manifest_file}")
    print("Run `breeze-distill select` to freeze your chosen seed into a .breeze voice vector.")
    return manifest


def select_audition_candidate(
    voice_name: str,
    chosen_seed: int,
    audition_dir: str,
    voices_dir: str = "voices",
    model_path: str = "models/breeze-tts-2-q4_k.gguf",
    breeze_cli: str = "build/breeze-cli",
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Freeze a selected audition candidate into a reference WAV, TXT, and .breeze vector."""
    aud_path = Path(audition_dir)
    voices_path = Path(voices_dir)
    voices_path.mkdir(parents=True, exist_ok=True)

    manifest_file = aud_path / "audition_reel.json"
    transcript = DEFAULT_CALIBRATION_PASSAGE

    if manifest_file.exists():
        try:
            with open(manifest_file, "r") as f:
                data = json.load(f)
                transcript = data.get("calibration_text", transcript)
        except Exception:
            pass

    src_clip = aud_path / f"{voice_name}_seed{chosen_seed}.wav"
    dest_clip = voices_path / f"{voice_name}.wav"
    dest_txt = voices_path / f"{voice_name}.txt"
    dest_breeze = voices_path / f"{voice_name}.breeze"

    if not dry_run and not src_clip.exists():
        raise FileNotFoundError(f"Selected candidate clip does not exist: {src_clip}")

    print(f"\n=== Freezing Voice Candidate: '{voice_name}' (Seed {chosen_seed}) ===")
    print(f"Source clip: {src_clip}")
    print(f"Target reference WAV: {dest_clip}")
    print(f"Target transcript: {dest_txt}")
    print(f"Target .breeze vector: {dest_breeze}")

    if not dry_run:
        shutil.copy2(src_clip, dest_clip)
        with open(dest_txt, "w") as f:
            f.write(transcript + "\n")

    # Run breeze-cli --save-voice
    save_cmd = [
        breeze_cli,
        model_path,
        "--ref-audio", str(dest_clip),
        "--ref-text", transcript,
        "--save-voice", voice_name,
        "--voices-dir", str(voices_path),
    ]

    if dry_run:
        print(f"[DRY-RUN] Pre-encode command: {' '.join(save_cmd)}")
        return {
            "voice_name": voice_name,
            "chosen_seed": chosen_seed,
            "ref_wav": str(dest_clip),
            "ref_txt": str(dest_txt),
            "breeze_vector": str(dest_breeze),
            "status": "dry_run_selected",
        }

    t0 = time.time()
    proc = subprocess.run(save_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    elapsed = time.time() - t0

    if proc.returncode != 0:
        print(f"Pre-encode FAILED ({elapsed:.1f}s):\nSTDERR: {proc.stderr}")
        raise RuntimeError(f"breeze-cli --save-voice failed: {proc.stderr}")

    if not dest_breeze.exists():
        raise FileNotFoundError(f"Expected .breeze file was not created: {dest_breeze}")

    breeze_size = dest_breeze.stat().st_size
    print(f"SUCCESS: Encoded '{dest_breeze}' ({breeze_size:,} bytes in {elapsed:.1f}s)")

    return {
        "voice_name": voice_name,
        "chosen_seed": chosen_seed,
        "ref_wav": str(dest_clip),
        "ref_txt": str(dest_txt),
        "breeze_vector": str(dest_breeze),
        "breeze_bytes": breeze_size,
        "precompute_elapsed_s": round(elapsed, 2),
        "status": "ready",
    }


def main():
    parser = argparse.ArgumentParser(description="Generate voice design auditions or select a candidate")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # audition
    p_aud = subparsers.add_parser("audition", help="Generate multi-seed candidate clips")
    p_aud.add_argument("--voice", required=True, help="Voice name")
    p_aud.add_argument("--instruction", required=True, help="Voice description")
    p_aud.add_argument("--out-dir", default="auditions", help="Directory for audition clips")
    p_aud.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS, help="Candidate seeds")
    p_aud.add_argument("--model", default="models/breeze-tts-2-q4_k.gguf", help="Model path")
    p_aud.add_argument("--breeze-cli", default="build/breeze-cli", help="breeze-cli path")
    p_aud.add_argument("--text", default=DEFAULT_CALIBRATION_PASSAGE, help="Text to speak")
    p_aud.add_argument("--dry-run", action="store_true", help="Print commands without executing inference")

    # select
    p_sel = subparsers.add_parser("select", help="Freeze chosen candidate into .breeze vector")
    p_sel.add_argument("--voice", required=True, help="Voice name")
    p_sel.add_argument("--seed", type=int, required=True, help="Selected candidate seed")
    p_sel.add_argument("--audition-dir", default="auditions", help="Directory containing candidate clips")
    p_sel.add_argument("--voices-dir", default="voices", help="Target directory for saved voices")
    p_sel.add_argument("--model", default="models/breeze-tts-2-q4_k.gguf", help="Model path")
    p_sel.add_argument("--breeze-cli", default="build/breeze-cli", help="breeze-cli path")
    p_sel.add_argument("--dry-run", action="store_true", help="Dry run selection")

    args = parser.parse_args()

    if args.command == "audition":
        aud_dir = os.path.join(args.out_dir, args.voice)
        generate_audition_reel(
            instruction=args.instruction,
            voice_name=args.voice,
            out_dir=aud_dir,
            model_path=args.model,
            breeze_cli=args.breeze_cli,
            seeds=args.seeds,
            text=args.text,
            dry_run=args.dry_run,
        )
    elif args.command == "select":
        aud_dir = os.path.join(args.audition_dir, args.voice)
        select_audition_candidate(
            voice_name=args.voice,
            chosen_seed=args.seed,
            audition_dir=aud_dir,
            voices_dir=args.voices_dir,
            model_path=args.model,
            breeze_cli=args.breeze_cli,
            dry_run=args.dry_run,
        )


if __name__ == "__main__":
    main()
