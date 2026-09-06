# Breeze investigation handoff: Feature forecasting feasibility (deferred, low-cost probe)

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

Lower-priority hypothesis, not part of the initial implementation shortlist. See concept 5. Test whether diffusion-style forecasting has any measurable basis in Breeze before adapting a cache.

Main-thread implementation work was ongoing when this handoff was written. Inspect current status and local instructions first; preserve unrelated edits and do not launch competing GPU jobs. No experiments, training or new performance measurements were performed in this research conversation. Required datasets, floating master weights and training resources have not been established. Do not interpret this document as approval to launch expensive training, deploy, commit or push. Avoid duplicate activation collection: coordinate through handoff 00 and reuse its artifacts.

## Bounded first investigation

1. Reuse shared traces to compare same-codebook features across frames and adjacent-codebook features within a frame separately.
2. Fit simple reuse and first-order forecasts; evaluate held-out prediction errors and sensitivity at speech transients.
3. Identify a cheap confidence signal and bound the amount of FFN work it could skip.
4. Initially retain exact attention and KV updates in any proposed approximation; explain error accumulation across the feedback loop.
5. Deliver the diagnostic result without building a cache if smoothness or a useful confidence signal is absent.

## Required outcome and stopping condition

A low-cost go/no-go on temporal predictability with plots or tables and no claimed speedup. Different RVQ positions are not successive diffusion timesteps. Stop if sampling and phonetic transitions make reuse unreliable.

Use actual packed bytes and measured GPU time for eventual performance claims. Teacher-forced agreement alone is insufficient: approximations must eventually pass free-running speech and feedback evaluation. Preserve source provenance and label extrapolation explicitly. Follow the existing evaluation artifact for language, voice, duration and termination coverage.

## Suggested skills

Ask the next agent to call the Skill tool for these skills, if available (otherwise read their SKILL.md using its supported access mechanism):
- diagnosing-bugs — validate assumptions with traces.
- concise-planning — keep this a small falsification test.
- ponytail — no caching framework before evidence.
