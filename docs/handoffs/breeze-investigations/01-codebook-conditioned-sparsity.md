# Breeze investigation handoff: Codebook-conditioned FFN sparsity

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

Investigate whether known codebook position predicts useful contiguous FFN subsets. See concept 3 of the transferable-concepts note. Depend on handoff 00 for data.

Main-thread implementation work was ongoing when this handoff was written. Inspect current status and local instructions first; preserve unrelated edits and do not launch competing GPU jobs. No experiments, training or new performance measurements were performed in this research conversation. Required datasets, floating master weights and training resources have not been established. Do not interpret this document as approval to launch expensive training, deploy, commit or push. Avoid duplicate activation collection: coordinate through handoff 00 and reuse its artifacts.

## Bounded first investigation

1. Measure per-layer/per-codebook activation and contribution distributions on the shared corpus.
2. Compare modest contiguous group masks: global static, codebook-specific static, and dynamic selection as a reference. Evaluate on held-out utterances.
3. Trace which gate/up/down weight blocks each proposal would actually avoid reading; account for quantization block alignment and mask overhead.
4. Test approximation sensitivity offline before proposing a sparse Vulkan kernel.
5. If quality headroom exists, specify one representative kernel benchmark and its required authorization/resources.

## Required outcome and stopping condition

A go/no-go report with error-versus-actual-byte curves, position-specialization evidence and the smallest next benchmark. Stop if masks generalize poorly or only zero activations while retaining dense weight reads. Do not claim all three FFN matrices benefit from a mask applied after gate/up.

Use actual packed bytes and measured GPU time for eventual performance claims. Teacher-forced agreement alone is insufficient: approximations must eventually pass free-running speech and feedback evaluation. Preserve source provenance and label extrapolation explicitly. Follow the existing evaluation artifact for language, voice, duration and termination coverage.

## Suggested skills

Ask the next agent to call the Skill tool for these skills, if available (otherwise read their SKILL.md using its supported access mechanism):
- diagnosing-bugs — evidence-driven performance investigation.
- concise-planning — define the ablation.
- ponytail — minimal offline probe.
