# Breeze investigation handoff: Direct prediction of feedback embedding and codec latent

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

Investigate the most exploratory hypothesis: bypass residual code generation by predicting the two different downstream representations. See concept 1. This was inferred from Breeze source, not demonstrated by a cited paper.

Main-thread implementation work was ongoing when this handoff was written. Inspect current status and local instructions first; preserve unrelated edits and do not launch competing GPU jobs. No experiments, training or new performance measurements were performed in this research conversation. Required datasets, floating master weights and training resources have not been established. Do not interpret this document as approval to launch expensive training, deploy, commit or push. Avoid duplicate activation collection: coordinate through handoff 00 and reuse its artifacts.

## Bounded first investigation

1. Trace build_audio_embed, quantizer_decode and their callers. Document exact shapes, transforms and the proposed student boundary.
2. Verify teacher targets for BOTH backbone feedback and codec latent on the shared corpus; they are different spaces.
3. Compare a complete residual-loop replacement with a narrower early-code prefix plus predicted tail contributions.
4. Specify a small stochastic joint student and losses; assess whether mean regression collapses acoustic variability or produces off-manifold latents.
5. Design a closed-loop pilot that feeds predicted feedback into subsequent backbone generation, with explicit drift and termination checks. Identify required codec latent-input plumbing without implementing it prematurely.

## Required outcome and stopping condition

A falsifiable joint-student pilot proposal with interface evidence, distribution risks and resource requirements. Stop if acceptable prediction requires a model comparable in cost to the current depth decoder. Do not infer success from codec audio quality alone or ignore the loss of the original discrete-token interface.

Use actual packed bytes and measured GPU time for eventual performance claims. Teacher-forced agreement alone is insufficient: approximations must eventually pass free-running speech and feedback evaluation. Preserve source provenance and label extrapolation explicitly. Follow the existing evaluation artifact for language, voice, duration and termination coverage.

## Suggested skills

Ask the next agent to call the Skill tool for these skills, if available (otherwise read their SKILL.md using its supported access mechanism):
- codebase-design — establish the two downstream interfaces.
- concise-planning — bound the research pilot.
- ponytail — avoid building a new model framework.
