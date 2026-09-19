# `breeze-distill`: The Voice Foundry & Deployment Compiler

`breeze-distill` is the ahead-of-time deployment compiler and voice foundry for **Breeze-TTS-2.cpp**. It specializes the depth decoder to small-$N$ permanent voices (e.g. Calliope clone + Steward design) for real-time local serving on integrated GPUs (such as AMD Radeon 780M / 880M).

---

## 1. Architecture: Tier 1 vs. Tier 2

`Breeze-TTS-2.cpp` decouples broad general-purpose inference from specialized appliance deployment:

| Tier | Name | Target Use Case | Latency (780M) | Voices | Setup Cost |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Tier 1** | **Stock 9-Block GGUF** | General-purpose serving, dynamic voices | $\le 0.80$ RTF | Any loose `.breeze` | Instant (download & serve) |
| **Tier 2** | **`breeze-distill` Foundry** | Dedicated homelab / appliance agents | $\le 0.80$ RTF | 1–3 baked voices | Ahead-of-time distillation |

In Tier 2, voices are baked directly into the GGUF metadata (`breeze.embedded_voice.*`), eliminating filesystem `.breeze` dependencies, achieving **0.0 ms runtime reference encode latency**, and sharpening acoustic stability via student distillation.

---

## 2. CLI Workflow & Subcommands

The CLI entrypoint is `tools/breeze_distill.py`. It provides four primary subcommands:

```
tools/breeze_distill.py <subcommand> [options]
```

### 2.1 `calibrate` — Phonemic Coverage Analysis

Analyzes candidate calibration text against the full 39-phoneme ARPAbet dictionary (loaded from CMUdict). Ensures the reference speech prompts contain sufficient phonetic diversity for teacher capture.

```bash
python3 tools/breeze_distill.py calibrate \
  --text "The quick brown fox jumps over the lazy dog..." \
  --strict
```

Outputs total phonemes covered, missing phonemes, and vowel/consonant distribution.

### 2.2 `audition` — Voice Design Generation

Generates audio candidate clips across multiple random seeds for synthetic Voice Design:

```bash
python3 tools/breeze_distill.py audition \
  --voice steward \
  --instruction "Warm British butler, calm and articulate." \
  --seeds 42 101 202 303 \
  --out-dir runs/auditions
```

### 2.3 `select` — Freezing Audition Candidates

Auditions are reviewed by ear. Once a seed is approved, `select` freezes the chosen clip into a reference WAV, transcript, and pre-encoded `.breeze` vector ready for training:

```bash
python3 tools/breeze_distill.py select \
  --voice steward \
  --seed 42 \
  --audition-dir runs/auditions \
  --voices-dir voices/
```

### 2.4 `build` — End-to-End Distillation & Packaging Compiler

Runs the complete 5-stage ahead-of-time compilation pipeline:

```bash
python3 tools/breeze_distill.py build \
  --target-hardware 780m \
  --voice "calliope:clone:ref1.wav,ref2.wav" \
  --voice "steward:design:Warm British butler:42" \
  --base-model models/teacher-f16.gguf \
  --output models/breeze-dd9-n2-q4_k.gguf \
  --epochs 15 \
  --work-dir /tmp/distill_ws
```

---

## 3. The Calliope Method (Multi-Utterance Voice Cloning)

For cloned voices, `breeze-distill` implements the **Calliope Method** (F8 protocol):

1. **Acoustic Budget**: Pinned at 161 frames $\times$ 1920 samples = **309,120 samples** (12.88 s at 24 kHz).
2. **Pack Plan**: Multiple short reference utterances are concatenated in sequence with 250 ms (6,000 samples) acoustic silence padding until the budget is exhausted.
3. **Hold-out Acoustic Loss**: The first utterance that exceeds the 12.88 s budget is held out as an unseen evaluation sample. After distillation, the student synthesizes this held-out text, and `analyze_audio.py` verifies acoustic correlation, RMS ratio, and lag against ground truth.
4. **Fail-Closed Constraint**: Cloning requires $\ge 2$ reference utterances (one for reference packing and at least one for hold-out verification). Single-utterance cloning is rejected by the compiler.

---

## 4. GGUF Metadata Lineage Schema

`breeze-distill` embeds all active voices and compilation lineage into the exported GGUF file:

| Metadata Key | Type | Description |
| :--- | :--- | :--- |
| `breeze.dd.block_count` | `UINT32` | Number of active student depth blocks (e.g. `9`). |
| `breeze.student.ckpt_md5` | `STRING` | MD5 checksum of the PyTorch checkpoint (`best_student_9block.pt`). |
| `breeze.embedded_voice_names` | `STRING_ARRAY` | Ordered list of baked voice names (e.g. `["calliope", "steward"]`). |
| `breeze.embedded_voice.<name>` | `UINT8_ARRAY` | Raw `.breeze` (BRZV v1) binary bytes for voice `<name>`. |
| `breeze.embedded_voice.default` | `STRING` | Default fallback voice if `--voice` is omitted. |

The runtime CLI (`breeze-cli`) and server (`breeze-server`) automatically discover and load these embedded voices without requiring loose `.breeze` files or `--voices-dir`.

---

## 5. Hardware Profiles & Data-Derived Safety Gate

### 5.1 Hardware Presets (`--target-hardware`)
- **`780m`**: AMD Radeon 780M / 880M (RDNA3 iGPU, Mesa Vulkan). Enforces a 9-block architecture with target RTF $\le 0.800$. Requires runtime levers `BREEZE_DD_FUSED=1 BREEZE_DD_FLASH_ATTN=1`.
- **`baseline`**: Stock 12-block baseline (target RTF $\le 0.930$).

### 5.2 Fail-Closed Safety Gate
Dry runs and unmeasured builds report `UNVERIFIED` with `all_pass: null` — metrics are never fabricated.

When executed on hardware, the safety gate enforces a data-derived threshold ($\tau = 35.0\%$ Top-1 token match vs teacher, safely above the 19.9% random/untrained truncation floor). If any voice falls below $\tau$, the compiler fails closed.
