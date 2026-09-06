# Breeze investigation handoff: Coordinate-conditioned parallel depth student

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

Investigate replacing the sequential depth predictor while preserving complete code tuples, backbone and codec. See concept 2 and the linked Samsung Interspeech 2025 paper. This is model research, not an attention-mask switch.

Main-thread implementation work was ongoing when this handoff was written. Inspect current status and local instructions first; preserve unrelated edits and do not launch competing GPU jobs. No experiments, training or new performance measurements were performed in this research conversation. Required datasets, floating master weights and training resources have not been established. Do not interpret this document as approval to launch expensive training, deploy, commit or push. Avoid duplicate activation collection: coordinate through handoff 00 and reuse its artifacts.

## Bounded first investigation

1. Read the paper’s coordinate-conditioned INR and masked-refinement mechanisms; establish their transferable assumptions and limitations.
2. Specify a small student conditioned on backbone hidden, cb0 and codebook index. Compare one-pass prediction with a small number of refinement rounds on paper before training.
3. Reuse the teacher corpus; identify losses, stochastic conditioning and joint-code dependence requirements.
4. Define a bounded pilot training proposal, compute budget, held-out evaluation and criteria for retaining both downstream interfaces.
5. Specify how parallel queries share weight reads on the 780M rather than independently executing the same weights fifteen times.

## Required outcome and stopping condition

A concrete student/pilot specification with data and compute prerequisites, then a go/no-go decision for a separately authorized pilot. Evaluate eventual free-running generation, not only token accuracy. Stop if the proposal requires replacing the backbone or codec without evidence that this is necessary.

Use actual packed bytes and measured GPU time for eventual performance claims. Teacher-forced agreement alone is insufficient: approximations must eventually pass free-running speech and feedback evaluation. Preserve source provenance and label extrapolation explicitly. Follow the existing evaluation artifact for language, voice, duration and termination coverage.

## Suggested skills

Ask the next agent to call the Skill tool for these skills, if available (otherwise read their SKILL.md using its supported access mechanism):
- concise-planning — bound the model pilot.
- research — primary-source follow-up only if its workflow is permitted in the next session.
- ponytail — smallest student and integration.
