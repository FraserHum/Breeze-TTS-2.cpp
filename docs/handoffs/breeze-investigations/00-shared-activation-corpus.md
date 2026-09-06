# Breeze investigation handoff: Shared activation and teacher corpus

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

Own only the design and bounded collection of reusable measurements for the other investigations. Do not implement compression or student models.

Main-thread implementation work was ongoing when this handoff was written. Inspect current status and local instructions first; preserve unrelated edits and do not launch competing GPU jobs. No experiments, training or new performance measurements were performed in this research conversation. Required datasets, floating master weights and training resources have not been established. Do not interpret this document as approval to launch expensive training, deploy, commit or push. Avoid duplicate activation collection: coordinate through handoff 00 and reuse its artifacts.

## Bounded first investigation

1. Read the architecture and evaluation artifacts; identify available model weights, traces, prompts and forced-token mechanisms.
2. Specify a compact corpus split by utterance, covering languages, voices, depth positions and representative frame positions. Keep calibration/training and held-out evaluation separate.
3. Define activation capture at FFN inputs/intermediates/outputs, plus backbone hidden, cb0, complete sampled codes, backbone feedback embedding and codec latent. Capture metadata needed to associate each tensor with its layer/frame/codebook; use selective or streamed capture to bound storage.
4. Determine which traces already exist. Propose minimal missing instrumentation only after tracing current code. Establish one owner and GPU schedule before collection.
5. Deliver a corpus manifest, schemas, reproducible collection recipe and cost estimate. If collection is authorized and resources are available, perform only a small validation capture first.

## Required outcome and stopping condition

A reusable manifest and a validated small sample, or an explicit resource blocker and ready collection specification. Check indexing and tensor provenance against an unmodified baseline. Do not collect a large corpus until consumers confirm the sample is sufficient.

Use actual packed bytes and measured GPU time for eventual performance claims. Teacher-forced agreement alone is insufficient: approximations must eventually pass free-running speech and feedback evaluation. Preserve source provenance and label extrapolation explicitly. Follow the existing evaluation artifact for language, voice, duration and termination coverage.

## Suggested skills

Ask the next agent to call the Skill tool for these skills, if available (otherwise read their SKILL.md using its supported access mechanism):
- concise-planning — bound collection and ownership.
- observability-and-instrumentation — only if capture instrumentation is needed.
- ponytail — keep capture minimal and reusable.
