"""Unified Distillation Pipeline Orchestrator for Breeze-TTS-2.

Automates the complete deployment compilation lifecycle:
1. Voice Onboarding: Validates or pre-encodes reference WAVs and Voice Design seeds into .breeze vectors.
2. Balanced Vulkan Capture: Generates per-voice teacher depth activation records over calibration prompts.
3. Posterior Distillation: Trains the pruned depth decoder student (e.g. 9 blocks) with KL + CE loss.
4. GGUF Packaging: Prunes model blocks, embeds all active voices into GGUF metadata (breeze.embedded_voice.*).
5. Quantization: Quantizes to target format (Q4_K).
6. Telemetry Scorecard: Generates a validation and hardware latency report.
"""
from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

from .calibration import DEFAULT_CALIBRATION_PASSAGE, analyze_phoneme_coverage
from .audition import select_audition_candidate, generate_audition_reel

# Receipt-gated presets (Define lock, docs/plans/breeze-distill-define-lock.md).
# Every row must cite a measured receipt; the 6-block "edge" profile is cut
# until a 6-block checkpoint is trained, ear-tested, and canary-benchmarked.
HARDWARE_PROFILES = {
    "780m": {
        "description": "AMD Radeon 780M / 880M (RDNA3 iGPU, Mesa Vulkan)",
        "n_blocks": 9,
        "target_rtf": 0.80,
        "measured_rtf": (0.791, 0.800),
        "receipt": "benchmarks/depth-corpus/dd9-prod-pod-canary-receipt.json (0.791) + dd9-n2-rtf-control-receipt.json",
        "recommended_levers": {"BREEZE_DD_FUSED": "1", "BREEZE_DD_FLASH_ATTN": "1"},
    },
    "baseline": {
        "description": "Full 12-block generalist teacher (uncompressed)",
        "n_blocks": 12,
        "target_rtf": 0.93,
        "measured_rtf": (0.93, 0.93),
        "receipt": "benchmarks/depth-corpus/dd9-rtf-07917-verification-receipt.json (task 1846 control)",
        "recommended_levers": {"BREEZE_DD_FUSED": "1"},
    },
}


@dataclass
class VoiceSpec:
    name: str
    voice_type: str = "clone"  # 'clone', 'design', or 'precomputed'
    audio_path: Optional[str] = None      # primary reference (first utterance)
    transcript: Optional[str] = None      # primary transcript
    # F8: multi-utterance reference for cloning (list of (wav, transcript) pairs).
    utterances: List[Tuple[str, Optional[str]]] = field(default_factory=list)
    instruction: Optional[str] = None
    seed: int = 42
    breeze_path: Optional[str] = None

    def __post_init__(self):
        if self.audio_path and not self.utterances:
            self.utterances = [(self.audio_path, self.transcript)]

    def validate(self):
        if not self.name or not self.name.replace("-", "").replace("_", "").isalnum():
            raise ValueError(f"Invalid voice name '{self.name}'. Must be alphanumeric with '-' or '_'.")
        if self.voice_type not in ("clone", "design", "precomputed"):
            raise ValueError(f"Unknown voice type '{self.voice_type}'. Must be 'clone', 'design', or 'precomputed'.")
        if self.voice_type in ("clone", "precomputed") and not self.utterances and not self.breeze_path:
            raise ValueError(f"Voice '{self.name}' is type '{self.voice_type}' but no audio/text utterance "
                             f"or precomputed breeze_path was given.")
        # NOTE: the "cloning needs >= 2 reference utterances" rule (F8) is an execution-time
        # quality policy, enforced fail-closed in step_onboard_voices for real runs only.
        # validate() checks structural shape so dry-run wiring tests stay unblocked.
        if self.voice_type == "design" and not self.instruction and not self.breeze_path:
            raise ValueError(f"Voice '{self.name}' is type 'design' but instruction was not given.")


def _split_outside_quotes(s: str, sep: str = ",") -> List[str]:
    """Split on sep, but only when the separator is outside single/double quotes.

    F8 fix: instruction="Warm, British butler" must stay one key=value part.
    """
    parts: List[str] = []
    buf: List[str] = []
    quote = None
    for ch in s:
        if quote:
            buf.append(ch)
            if ch == quote:
                quote = None
        elif ch in "\"'":
            quote = ch
            buf.append(ch)
        elif ch == sep:
            parts.append("".join(buf))
            buf = []
        else:
            buf.append(ch)
    parts.append("".join(buf))
    return [p.strip() for p in parts if p.strip()]


def parse_voice_flag(v_str: str) -> VoiceSpec:
    """Parse a quote-aware voice specification: key=val,key=val...

    Example:
      name=steward,type=design,instruction="Warm, British butler",seed=42
      name=calliope,type=clone,audio=a1.wav,text="line one",audio=a2.wav,text="line two"
      name=calliope,type=precomputed,breeze=./calliope.breeze

    F8: repeated audio=/text= pairs become multi-utterance clone references. A
    text= pairs with the most recent preceding audio= (default if none).
    """
    kvs: Dict[str, str] = {}
    utterances: List[Tuple[str, Optional[str]]] = []
    pending_audio: Optional[str] = None
    for part in _split_outside_quotes(v_str):
        if "=" not in part:
            raise ValueError(f"Voice spec part must be key=value: {part!r}")
        k, v = part.split("=", 1)
        v = v.strip()
        if len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'":
            v = v[1:-1]
        k = k.strip()
        if k == "audio":
            if pending_audio is not None:
                utterances.append((pending_audio, None))
            pending_audio = v
        elif k == "text" and pending_audio is not None:
            utterances.append((pending_audio, v))
            pending_audio = None
        else:
            kvs[k] = v
    if pending_audio is not None:
        utterances.append((pending_audio, None))

    name = kvs.get("name")
    if not name:
        raise ValueError(f"Voice specification requires 'name': {v_str}")

    v_type = kvs.get("type", "clone")
    instruction = kvs.get("instruction")
    seed = int(kvs.get("seed", 42))
    breeze = kvs.get("breeze")
    audio = utterances[0][0] if utterances else None
    transcript = utterances[0][1] if utterances else None

    return VoiceSpec(
        name=name,
        voice_type=v_type,
        audio_path=audio,
        transcript=transcript,
        utterances=utterances,
        instruction=instruction,
        seed=seed,
        breeze_path=breeze,
    )


@dataclass
class DistillPipelineConfig:
    voices: List[VoiceSpec]
    target_hardware: str = "780m"
    n_blocks: Optional[int] = None
    base_model: str = "models/teacher-f16.gguf"
    output_model: str = "models/breeze-distilled-q4_k.gguf"
    work_dir: str = "distill_workspace"
    prompts_path: str = "tools/tail_training/pilot-prompts.json"
    breeze_cli: str = "build/breeze-cli"
    breeze_quantize: str = "build/breeze-quantize"
    epochs: int = 15
    batch_size: int = 32
    learning_rate: float = 2e-4
    # Safety gate threshold (percentage 0..100, matches receipt top1_acc_teacher).
    # None => UNVERIFIED (fail-closed: no threshold, no gate; a value must be
    # derived from a full-corpus run before [PASS] may ever be printed).
    min_top1_threshold: Optional[float] = None
    force: bool = False
    dry_run: bool = False

    def __post_init__(self):
        if self.target_hardware in HARDWARE_PROFILES:
            profile = HARDWARE_PROFILES[self.target_hardware]
            if self.n_blocks is None:
                self.n_blocks = profile["n_blocks"]
        elif self.n_blocks is None:
            self.n_blocks = 9
        if self.n_blocks not in (6, 9, 12):
            raise ValueError(f"n_blocks must be 6, 9, or 12; got {self.n_blocks}")
        if self.target_hardware in HARDWARE_PROFILES:
            want = HARDWARE_PROFILES[self.target_hardware]["n_blocks"]
            if self.n_blocks != want:
                raise ValueError(f"target_hardware {self.target_hardware!r} pins n_blocks={want}; "
                                 f"got {self.n_blocks} — presets are receipt-gated (Define lock)")


class DistillPipeline:
    """End-to-end execution pipeline for compiling specialized voice models."""

    def __init__(self, config: DistillPipelineConfig):
        self.cfg = config
        self.work_path = Path(self.cfg.work_dir)
        self.voices_dir = self.work_path / "voices"
        self.corpus_dir = self.work_path / "corpus"
        self.ckpt_dir = self.work_path / "checkpoints"
        self.telemetry: Dict[str, Any] = {}

    def setup_directories(self):
        self.work_path.mkdir(parents=True, exist_ok=True)
        self.voices_dir.mkdir(parents=True, exist_ok=True)
        self.corpus_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

    def step_onboard_voices(self) -> List[Tuple[str, Path]]:
        """Ensure all target voices have precomputed .breeze vectors."""
        print("\n=== Step 1/5: Voice Onboarding & Verification ===")
        embedded_specs: List[Tuple[str, Path]] = []

        for v in self.cfg.voices:
            v.validate()
            target_breeze = self.voices_dir / f"{v.name}.breeze"

            if target_breeze.exists():
                print(f"  Voice '{v.name}': using existing {target_breeze} ({target_breeze.stat().st_size:,} bytes)")
                embedded_specs.append((v.name, target_breeze))
                continue

            if v.breeze_path and Path(v.breeze_path).exists():
                shutil.copy2(v.breeze_path, target_breeze)
                print(f"  Voice '{v.name}': copied {v.breeze_path} -> {target_breeze}")
                embedded_specs.append((v.name, target_breeze))
                continue

            if v.voice_type == "design":
                print(f"  Voice '{v.name}': Design-once-then-clone (instruction: \"{v.instruction}\", seed: {v.seed})")
                aud_dir = self.work_path / "auditions" / v.name
                if not self.cfg.dry_run:
                    generate_audition_reel(
                        instruction=v.instruction or "A clear and natural voice",
                        voice_name=v.name,
                        out_dir=str(aud_dir),
                        model_path=self.cfg.base_model,
                        breeze_cli=self.cfg.breeze_cli,
                        seeds=[v.seed],
                    )
                    select_audition_candidate(
                        voice_name=v.name,
                        chosen_seed=v.seed,
                        audition_dir=str(aud_dir),
                        voices_dir=str(self.voices_dir),
                        model_path=self.cfg.base_model,
                        breeze_cli=self.cfg.breeze_cli,
                    )
                else:
                    print(f"    [DRY-RUN] Would generate audition and select candidate for '{v.name}'")
                    target_breeze.touch()
                embedded_specs.append((v.name, target_breeze))

            elif v.voice_type == "clone":
                n_u = len(v.utterances)
                print(f"  Voice '{v.name}': Cloning from {n_u} reference utterance(s) "
                      f"(multi-utterance required, F8)")
                if n_u < 2 and not self.cfg.dry_run:
                    raise ValueError(f"Voice '{v.name}': cloning requires >= 2 reference utterances "
                                     f"(audio=/text= pairs); single-clip cloning is fail-closed by design")
                if not self.cfg.dry_run:
                    for i, (ap, _) in enumerate(v.utterances, 1):
                        if not ap or not Path(ap).exists():
                            raise FileNotFoundError(f"Audio file {i} for voice '{v.name}' not found: {ap}")
                primary_wav, primary_text = v.utterances[0]
                dest_wav = self.voices_dir / f"{v.name}.wav"
                dest_txt = self.voices_dir / f"{v.name}.txt"
                transcript = primary_text or DEFAULT_CALIBRATION_PASSAGE

                if not self.cfg.dry_run:
                    shutil.copy2(primary_wav, dest_wav)
                    with open(dest_txt, "w") as f:
                        f.write(transcript + "\n")

                    cmd = [
                        self.cfg.breeze_cli,
                        self.cfg.base_model,
                        "--ref-audio", str(dest_wav),
                        "--ref-text", transcript,
                        "--save-voice", v.name,
                        "--voices-dir", str(self.voices_dir),
                    ]
                    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                    if proc.returncode != 0:
                        raise RuntimeError(f"Failed to pre-encode voice '{v.name}': {proc.stderr}")
                else:
                    print(f"    [DRY-RUN] Would execute breeze-cli --save-voice {v.name}")
                    target_breeze.touch()
                embedded_specs.append((v.name, target_breeze))

        if not self.cfg.dry_run:
            # F6: a dry-run stub or corrupted .breeze must not silently flow into export.
            for name, path in embedded_specs:
                raw = path.read_bytes()
                if len(raw) < 28 or raw[:4] != b"BRZV":
                    raise ValueError(f"voice '{name}': {path} is not a valid .breeze (BRZV) file — "
                                     f"likely a 0-byte dry-run stub; re-run without --dry-run or supply a real precomputed vector")
        return embedded_specs

    def step_capture_dataset(self):
        """Run teacher activation capture over prompts for each onboarded voice."""
        print("\n=== Step 2/5: Balanced Dataset Capture ===")
        for v in self.cfg.voices:
            voice_corpus = self.corpus_dir / v.name
            voice_corpus.mkdir(parents=True, exist_ok=True)
            print(f"  Capturing teacher records for voice '{v.name}' -> {voice_corpus}")

            cmd = [
                sys.executable,
                "tools/depth_training/capture_dataset.py",
                "--prompts", self.cfg.prompts_path,
                "--breeze-cli", self.cfg.breeze_cli,
                "--model", self.cfg.base_model,
                "--voices-dir", str(self.voices_dir),
                "--voice", v.name,
                "--out-dir", str(self.corpus_dir),
            ]

            if self.cfg.dry_run:
                print(f"    [DRY-RUN] Command: {' '.join(cmd)}")
                # Create dummy bins so subsequent step logic is testable
                for split in ("train", "validation", "test"):
                    (voice_corpus / f"{split}.bin").touch()
            else:
                subprocess.run(cmd, check=True)

    def step_train_distill(self) -> Path:
        """Run posterior distillation on the depth decoder student."""
        print(f"\n=== Step 3/5: Depth Decoder Distillation (Layers 0..{self.cfg.n_blocks - 1}) ===")
        train_bins = [str(self.corpus_dir / v.name / "train.bin") for v in self.cfg.voices]
        val_bins = [str(self.corpus_dir / v.name / "validation.bin") for v in self.cfg.voices]
        voice_names = [v.name for v in self.cfg.voices]

        # distill.py writes best_student_9block.pt + distill_receipt.json into --output-dir.
        best_ckpt = self.ckpt_dir / "best_student_9block.pt"

        cmd = [
            sys.executable,
            "tools/depth_training/distill.py",
            "--train-bin", *train_bins,
            "--val-bin", *val_bins,
            "--voice", *voice_names,
            "--teacher-gguf", self.cfg.base_model,
            "--epochs", str(self.cfg.epochs),
            "--batch-size", str(self.cfg.batch_size),
            "--lr", str(self.cfg.learning_rate),
            "--output-dir", str(self.ckpt_dir),
        ]

        if self.cfg.dry_run:
            print(f"  [DRY-RUN] Command: {' '.join(cmd)}")
            print("  [DRY-RUN] No metrics available — scorecard will be UNVERIFIED (F4: no fabricated numbers).")
        else:
            subprocess.run(cmd, check=True)
            receipt_file = self.ckpt_dir / "distill_receipt.json"
            if not receipt_file.exists():
                raise RuntimeError(f"distill.py completed but wrote no receipt at {receipt_file}")
            with open(receipt_file) as f:
                self.telemetry = json.load(f)

        return best_ckpt

    def step_export_and_quantize(self, best_ckpt: Path, voice_specs: List[Tuple[str, Path]]) -> Path:
        """Export pruned GGUF model with embedded voices and quantize to Q4_K."""
        print(f"\n=== Step 4/5: GGUF Block Pruning & Metadata Embedding ===")
        unquant_gguf = self.work_path / f"breeze-dd{self.cfg.n_blocks}-f16.gguf"
        final_gguf = Path(self.cfg.output_model)
        final_gguf.parent.mkdir(parents=True, exist_ok=True)

        # F2: exporter owns quantization; --embed-voice takes two args (NAME FILE);
        # block count comes from the checkpoint's n_layer, not a CLI flag.
        embed_args = []
        for name, breeze_path in voice_specs:
            embed_args.extend(["--embed-voice", name, str(breeze_path)])

        export_cmd = [
            sys.executable,
            "tools/depth_training/export_depth_gguf.py",
            "--base-gguf", self.cfg.base_model,
            "--checkpoint", str(best_ckpt),
            "--out-f16", str(unquant_gguf),
            "--out-q4k", str(final_gguf),
            "--breeze-quantize", self.cfg.breeze_quantize,
            *embed_args,
        ]
        if voice_specs and not self.cfg.dry_run:
            ckpt_md5 = hashlib.md5(best_ckpt.read_bytes()).hexdigest()
            export_cmd.extend(["--student-md5", ckpt_md5])
        if voice_specs and len(voice_specs) == 1:
            export_cmd.extend(["--default-voice", voice_specs[0][0]])

        if self.cfg.dry_run:
            print(f"  [DRY-RUN] Export Command: {' '.join(export_cmd)}")
            unquant_gguf.touch()
            final_gguf.touch()
        else:
            subprocess.run(export_cmd, check=True)

        return final_gguf

    def generate_scorecard(self, final_model_path: Path):
        """Generate and print the final telemetry and validation scorecard."""
        hw_info = HARDWARE_PROFILES.get(self.cfg.target_hardware, {})
        target_rtf = hw_info.get("target_rtf", 0.80)
        # F5: no fabricated RTF. The preset's measured RTF (with its receipt) is
        # context only — this specific artifact's RTF must be benchmarked.
        measured = hw_info.get("measured_rtf")
        measured_str = f"{measured[0]:.3f}–{measured[1]:.3f} ({hw_info.get('receipt', 'n/a')})" if measured else "n/a"
        rtf_line = f"Unmeasured (run canary benchmark on this artifact; {self.cfg.n_blocks}-block preset measured {measured_str} RTF)"

        print("\n=============================================================")
        print("          BREEZE-DISTILL DEPLOYMENT COMPILER SCORECARD        ")
        print("=============================================================")
        print(f"Target Hardware Profile: {self.cfg.target_hardware.upper()} ({hw_info.get('description', 'Custom')})")
        print(f"Architecture:            {self.cfg.n_blocks}-block Depth Decoder (Pruned from 12)")
        print(f"Output Artifact:         {final_model_path} ({final_model_path.stat().st_size:,} bytes)")
        print(f"Embedded Voices ({len(self.cfg.voices)}):   {', '.join(v.name for v in self.cfg.voices)}")
        print("-------------------------------------------------------------")

        threshold = self.cfg.min_top1_threshold
        # Fail-closed gate: no threshold => UNVERIFIED (never [PASS]);
        # threshold present => measured vs. derived; --force records FORCED.
        if threshold is None:
            gate_status = "unverified"
            gate_badge = "UNVERIFIED (no data-derived threshold)"
        elif self.cfg.force:
            gate_status = "forced"
            gate_badge = "FORCED (--force; see receipt safety_gate.status)"

        per_voice = self.telemetry.get("best", {}).get("per_voice_val", self.telemetry.get("per_voice_val", {}))
        voice_gate = {}
        for v in self.cfg.voices:
            metrics = per_voice.get(v.name, {})
            loss = metrics.get("loss", 0.0)
            kl = metrics.get("kl", 0.0)
            top1 = metrics.get("top1_acc_teacher", 0.0)  # percentage 0..100 (F3)
            if threshold is not None:
                voice_gate[v.name] = bool(top1 >= threshold)
                badge = ("PASS" if voice_gate[v.name] else "FAIL") if not self.cfg.force else "FORCED"
            else:
                badge = "UNVERIFIED"
            print(f"  - Voice '{v.name}': Loss: {loss:.3f} | KL: {kl:.3f} | Top-1 (Teacher): {top1:.1f}% [{badge}]")

        all_pass = all(voice_gate.values()) if voice_gate else False
        print("-------------------------------------------------------------")
        print(f"Latency Budget:          <= {target_rtf:.3f} RTF")
        print(f"Artifact RTF:            {rtf_line}")
        print(f"Safety Gate:             {gate_badge} (threshold: {threshold if threshold is not None else 'none — derive from full-corpus run'})")
        print(f"Runtime Levers Required: BREEZE_DD_FUSED=1 BREEZE_DD_FLASH_ATTN=1")
        print("=============================================================\n")

        scorecard_file = self.work_path / "scorecard.json"
        receipt = {
            "target_hardware": self.cfg.target_hardware,
            "n_blocks": self.cfg.n_blocks,
            "output_model": str(final_model_path),
            "voices": [v.name for v in self.cfg.voices],
            "artifact_rtf": None,
            "preset_measured_rtf": list(measured) if measured else None,
            "target_rtf": target_rtf,
            "safety_gate": {
                "threshold": threshold,
                "measured": {v.name: per_voice.get(v.name, {}).get("top1_acc_teacher") for v in self.cfg.voices},
                "status": gate_status,
            },
            "all_pass": all_pass if threshold is not None else None,
            "telemetry": self.telemetry,
        }
        with open(scorecard_file, "w") as f:
            json.dump(receipt, f, indent=2)

    def run(self) -> Path:
        """Execute the entire pipeline end-to-end."""
        self.setup_directories()
        voice_specs = self.step_onboard_voices()
        self.step_capture_dataset()
        best_ckpt = self.step_train_distill()
        final_model = self.step_export_and_quantize(best_ckpt, voice_specs)
        self.generate_scorecard(final_model)
        return final_model
