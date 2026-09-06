# Breeze investigation handoff: Rotation-aware three-bit depth quantization

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

Continue the earlier quantization investigation as a separate track; the rotation note contains the sources and ranking. Reuse shared calibration data from handoff 00.

Main-thread implementation work was ongoing when this handoff was written. Inspect current status and local instructions first; preserve unrelated edits and do not launch competing GPU jobs. No experiments, training or new performance measurements were performed in this research conversation. Required datasets, floating master weights and training resources have not been established. Do not interpret this document as approval to launch expensive training, deploy, commit or push. Avoid duplicate activation collection: coordinate through handoff 00 and reuse its artifacts.

## Bounded first investigation

1. Establish availability of original floating weights and representative Breeze calibration data.
2. Choose a bounded quality comparison: Q4_K control, calibrated ordinary 3-bit, and fixed-rotation 3-bit. Keep sensitive tensors at higher precision where justified.
3. Assess QTIP/YAQA tooling compatibility before creating custom code. Consider HARP only if fixed rotations fail quality and its additional overhead is justified.
4. Evaluate held-out output errors and full generated speech using existing gates.
5. Only after quality survives, scope one compressed matrix-vector Vulkan benchmark including transforms, packing and scratch traffic.

## Required outcome and stopping condition

A quality-versus-effective-byte result or explicit prerequisite blocker, plus the smallest runtime feasibility experiment. No full Vulkan port before quality evidence. Do not dequantize entire weights to global memory each step, and do not assume rotated formats combine freely with coordinate sparsity.

Use actual packed bytes and measured GPU time for eventual performance claims. Teacher-forced agreement alone is insufficient: approximations must eventually pass free-running speech and feedback evaluation. Preserve source provenance and label extrapolation explicitly. Follow the existing evaluation artifact for language, voice, duration and termination coverage.

## Suggested skills

Ask the next agent to call the Skill tool for these skills, if available (otherwise read their SKILL.md using its supported access mechanism):
- diagnosing-bugs — quality regression diagnosis.
- concise-planning — isolate quantizer comparisons.
- ponytail — use existing preparation tooling.
