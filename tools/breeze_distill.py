#!/usr/bin/env python3
"""breeze-distill: The Vulkan Voice Foundry and Deployment Compiler for Breeze-TTS-2.

Subcommands:
  calibrate: Evaluate the ARPAbet 39-phoneme coverage of a calibration text.
  audition:  Generate candidate clips for Voice Design across multiple seeds.
  select:    Freeze a candidate clip into a reference WAV, transcript, and .breeze vector.
  build:     Run the full ahead-of-time distillation and GGUF packaging compiler.
"""
import argparse
import os
from pathlib import Path
import sys

# Ensure repository root and tools are in sys.path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
tools_dir = Path(__file__).resolve().parent
if str(tools_dir) not in sys.path:
    sys.path.insert(0, str(tools_dir))

from breeze_distill.calibration import (
    DEFAULT_CALIBRATION_PASSAGE,
    analyze_phoneme_coverage,
)
from breeze_distill.audition import (
    DEFAULT_SEEDS,
    generate_audition_reel,
    select_audition_candidate,
)
from breeze_distill.pipeline import (
    DistillPipeline,
    DistillPipelineConfig,
    HARDWARE_PROFILES,
    VoiceSpec,
    parse_voice_flag,
)


def cmd_calibrate(args):
    text = args.text or DEFAULT_CALIBRATION_PASSAGE
    print(f"\nAnalyzing calibration text:\n\"{text}\"\n")
    report = analyze_phoneme_coverage(text)
    print(report.summary())
    if not report.is_complete and args.strict:
        print(f"\n[STRICT CHECK FAILED]: Text is missing {len(report.missing_phonemes)} ARPAbet phonemes.")
        sys.exit(1)


def cmd_audition(args):
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


def cmd_select(args):
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


def cmd_build(args):
    voices: list[VoiceSpec] = []
    if args.voice:
        for v_str in args.voice:
            voices.append(parse_voice_flag(v_str))
    else:
        # Default test voices: Calliope (clone) + Steward (design)
        voices = [
            VoiceSpec(name="calliope", voice_type="clone", audio_path="tests/calliope.wav"),
            VoiceSpec(name="steward", voice_type="design", instruction="Warm British butler", seed=42),
        ]

    config = DistillPipelineConfig(
        voices=voices,
        target_hardware=args.target_hardware,
        n_blocks=args.n_blocks,
        base_model=args.base_model,
        output_model=args.output,
        work_dir=args.work_dir,
        prompts_path=args.prompts,
        breeze_cli=args.breeze_cli,
        breeze_quantize=args.breeze_quantize,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        min_top1_threshold=args.min_top1,
        force=args.force,
        dry_run=args.dry_run,
    )

    pipeline = DistillPipeline(config)
    pipeline.run()


def main():
    parser = argparse.ArgumentParser(
        prog="breeze-distill",
        description="The Vulkan Voice Foundry & Ahead-of-Time Deployment Compiler for Breeze-TTS-2",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # 1. calibrate
    p_cal = subparsers.add_parser("calibrate", help="Analyze phoneme coverage of reference text")
    p_cal.add_argument("--text", default=None, help="Text to analyze (defaults to 39-ARPAbet pangram)")
    p_cal.add_argument("--strict", action="store_true", help="Exit non-zero if coverage is < 39/39")

    # 2. audition
    p_aud = subparsers.add_parser("audition", help="Generate Voice Design candidates across multiple seeds")
    p_aud.add_argument("--voice", required=True, help="Voice name identifier")
    p_aud.add_argument("--instruction", required=True, help="Voice description")
    p_aud.add_argument("--out-dir", default="auditions", help="Directory for candidate clips")
    p_aud.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS, help="RNG seeds to test")
    p_aud.add_argument("--model", default="models/breeze-tts-2-q4_k.gguf", help="Base model GGUF")
    p_aud.add_argument("--breeze-cli", default="build/breeze-cli", help="Path to breeze-cli")
    p_aud.add_argument("--text", default=DEFAULT_CALIBRATION_PASSAGE, help="Spoken calibration passage")
    p_aud.add_argument("--dry-run", action="store_true", help="Dry run without running model inference")

    # 3. select
    p_sel = subparsers.add_parser("select", help="Freeze selected audition candidate into .breeze voice vector")
    p_sel.add_argument("--voice", required=True, help="Voice name identifier")
    p_sel.add_argument("--seed", type=int, required=True, help="Chosen candidate seed")
    p_sel.add_argument("--audition-dir", default="auditions", help="Directory with candidate clips")
    p_sel.add_argument("--voices-dir", default="voices", help="Directory for saved voices")
    p_sel.add_argument("--model", default="models/breeze-tts-2-q4_k.gguf", help="Base model GGUF")
    p_sel.add_argument("--breeze-cli", default="build/breeze-cli", help="Path to breeze-cli")
    p_sel.add_argument("--dry-run", action="store_true", help="Dry run without running precompute")

    # 4. build
    p_bld = subparsers.add_parser("build", help="Compile specialized depth-distilled GGUF model")
    p_bld.add_argument(
        "--voice",
        action="append",
        help="Voice spec (e.g. name=steward,type=design,instruction='...',seed=42 or name=calliope,type=clone,audio=...)",
    )
    p_bld.add_argument(
        "--target-hardware",
        choices=list(HARDWARE_PROFILES.keys()),
        default="780m",
        help="Receipt-gated hardware profile (780m: 9 blocks @0.80 RTF, baseline: 12 blocks @0.93 RTF)",
    )
    p_bld.add_argument("--n-blocks", type=int, default=None, help="Explicit depth decoder block count")
    p_bld.add_argument("--base-model", default="models/teacher-f16.gguf", help="Teacher model GGUF")
    p_bld.add_argument("--output", default="models/breeze-distilled-q4_k.gguf", help="Output Q4_K model GGUF")
    p_bld.add_argument("--work-dir", default="distill_workspace", help="Working directory for artifacts")
    p_bld.add_argument("--prompts", default="tools/tail_training/pilot-prompts.json", help="Prompts file")
    p_bld.add_argument("--breeze-cli", default="build/breeze-cli", help="Path to breeze-cli")
    p_bld.add_argument("--breeze-quantize", default="build/breeze-quantize", help="Path to breeze-quantize")
    p_bld.add_argument("--epochs", type=int, default=15, help="Distillation epochs")
    p_bld.add_argument("--batch-size", type=int, default=32, help="Batch size")
    p_bld.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    p_bld.add_argument("--min-top1", type=float, default=None,
                       help="Safety-gate threshold for top1_acc_teacher, PERCENT 0..100 (derive from a "
                            "full-corpus receipt). Omit => UNVERIFIED, fail-closed (no PASS).")
    p_bld.add_argument("--force", action="store_true",
                       help="Override a failing safety gate; recorded as status=forced in scorecard.json (receipted)")
    p_bld.add_argument("--dry-run", action="store_true", help="Dry run end-to-end pipeline")

    args = parser.parse_args()

    if args.command == "calibrate":
        cmd_calibrate(args)
    elif args.command == "audition":
        cmd_audition(args)
    elif args.command == "select":
        cmd_select(args)
    elif args.command == "build":
        cmd_build(args)


if __name__ == "__main__":
    main()
