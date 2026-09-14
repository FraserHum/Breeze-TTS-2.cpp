# Breeze investigation handoff: Activation-aware low-rank depth compression

Date: 2026-09-06

## Purpose and context

The goal is q4_k generation at RTF 0.8 on the Radeon 780M. This handoff comes from a separate research conversation, not an execution handoff for the main optimization thread. The user wants transferable mathematical ideas whose benefit is amplified by Breeze's architecture. Literature findings are hypotheses for Breeze, not measured improvements.

Workspace: `/Users/fraser/workspace/github.com/fraserhum/Breeze-TTS-2.cpp`.

Read the existing artifacts rather than repeating their research:
- `/Users/fraser/workspace/github.com/fraserhum/Breeze-TTS-2.cpp/docs/research/breeze-specific-transferable-concepts.md` — architecture evidence, six concept families, caveats and research ordering.
- `/Users/fraser/workspace/github.com/fraserhum/Breeze-TTS-2.cpp/docs/research/rotation-quantization-780m.md` — quantization findings and proposed quality-first experiments.
- `/Users/fraser/workspace/github.com/fraserhum/Breeze-TTS-2.cpp/benchmarks/depth-frontier-780m.md` — measured checkpoint dimensions, traffic accounting and existing quality failures.
- `/Users/fraser/workspace/github.com/fraserhum/Breeze-TTS-2.cpp/docs/research/q4-k-rtf-080-investigation.md` — performance evidence and evaluation requirements; check for updates before relying on its baseline.

## Scope and coordination

Investigate low-rank FFN factors fitted to Breeze activations, not raw weight spectra alone. See concept 4. Depend on handoff 00.

Main-thread implementation work was ongoing when this handoff was written. Inspect current status and local instructions first; preserve unrelated edits and do not launch competing GPU jobs. No experiments, training or new performance measurements were performed in this research conversation. Required datasets, floating master weights and training resources have not been established. Do not interpret this document as approval to launch expensive training, deploy, commit or push. Avoid duplicate activation collection: coordinate through handoff 00 and reuse its artifacts.

## Bounded first investigation

1. Inspect IMPACT and Swift-SVD primary sources linked in the note and identify the simplest applicable output-distortion objective.
2. Measure activation-weighted spectra separately by layer and codebook position, then assess a shared basis across positions.
3. Build distortion-versus-stored-byte curves with explicit factor precision, scales and metadata. Compare against the existing Q4_K representation, not FP16.
4. Test the best small set of factorization choices on held-out activations; examine accumulated error before whole-model work.
5. If viable, specify one two-factor matrix-vector benchmark on the actual hardware.

## Required outcome and stopping condition

A rank/precision feasibility table and go/no-go conclusion. Reject factors that are smaller in parameter count but larger in bytes than Q4_K, or whose extra launches erase savings. Do not mistake paper compression-preparation speedups for inference speedups.

Use actual packed bytes and measured GPU time for eventual performance claims. Teacher-forced agreement alone is insufficient: approximations must eventually pass free-running speech and feedback evaluation. Preserve source provenance and label extrapolation explicitly. Follow the existing evaluation artifact for language, voice, duration and termination coverage.

## Suggested skills

Ask the next agent to call the Skill tool for these skills, if available (otherwise read their SKILL.md using its supported access mechanism):
- diagnosing-bugs — measure the bottleneck and candidate.
- concise-planning — scope rank/precision comparisons.
- ponytail — reuse numerical tooling.
