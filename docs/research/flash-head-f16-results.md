# F16 output heads with approved depth flash — 2026-09-15

Retain the F16-head model candidate: the paired ordinary Calliope development
fixture improves from **0.821579 to 0.795981 wall RTF**, with identical approved
audio. This reaches the 0.8 target on this fixture, not across all workloads.
The reserved held-out corpus remains untouched. The default model and
production deployment are unchanged.

## Model-only change

The [converter](../../tools/build_depth_head_f16.py) changes only
`dd.codebooks_head.weight`, shape `[1024,2051,15]`, from F32 to F16. It verifies
all **1,160 other tensor payloads**, tensor names/shapes/order and metadata
unchanged, with valid adjusted GGUF offsets. The source is never modified.

- Source SHA-256: `34483105709c75c38cdafcddcb4ec91e0b222097d30e76432507c448053a83de`.
- Candidate SHA-256: `31a8cc1fc2717fd623a6b7ff053dfdbfc36664159a8bce8db79e14f3cb124a56`.
- Storage reduction: **63,006,720 bytes**; candidate size 2,435,794,624 bytes.
- All 31,503,360 converted values are finite; maximum F32 roundtrip error is
  **2.98023224e-8**. Most source values were already F16-representable.

The approved calibrated Q3 FFNs and approved native depth flash path remain
in use. Runtime code slices each head using its tensor strides, so changing
storage precision needs no inference-code change. Weight rounding is tiny,
but changing tensor type can change GPU dispatch and arithmetic; the waveform
checks below remain necessary.

## Warm ABBA measurement

Same binary/backend, deployed Calliope reference, household text, instruction,
seed 42, CFG=1 and `BREEZE_DD_FLASH_ATTN=1` for both models. Three resident
repeats per block; exclude the first. All runs generate 115 frames / 9.20 seconds.

| Block, execution order | Wall RTF | Wall ms | Depth ms/frame |
|---|---:|---:|---:|
| Control A | 0.828152 | 7619.00 | 41.616 |
| F16 A | 0.793924 | 7304.10 | 38.886 |
| F16 B | 0.798038 | 7341.95 | 39.197 |
| Control B | 0.815005 | 7498.05 | 40.362 |
| Control pooled | 0.821579 | 7558.525 | 40.989 |
| F16 pooled | 0.795981 | 7323.025 | 39.042 |

Pooled depth time drops **1.947 ms/frame**, approximately 4.75%; actual request
wall time drops **235.5 ms**, or 3.12%. Frame counts match, so this RTF gain has
no changed-duration denominator. Both candidate blocks beat both controls.
However, control depth time drifts downward by 1.253 ms/frame between blocks;
the point estimate is not a confidence interval or a promise of 0.8 under
other host conditions. Inspect the paired experiment rather than comparing
against the previous session's approximately 0.81 baseline. The earlier Q4
head trial's 0.46 ms gain was a noisy prior, not an additive saving.

All twelve ordinary WAVs match the user-approved CFG=1 depth-flash hash
`3e35dafe2dd8a9d78c30fe271401d457e4765885d2b8bcf8c52bf384c64a6681`.
The saving is attributed to the model representation and resulting execution;
no isolated head-kernel bandwidth measurement was made in this experiment.

## Quality and scope

The independent CFG=1 tagged ABBA uses three repeats per block and excludes
the first. All runs produce 175 frames / 14.00 seconds.

| Block, execution order | Wall RTF | Depth ms/frame |
|---|---:|---:|
| Control A | 0.810936 | 39.613 |
| F16 A | 0.807379 | 39.110 |
| F16 B | 0.803393 | 38.637 |
| Control B | 0.814482 | 39.839 |
| Control pooled | 0.812709 | 39.726 |
| F16 pooled | 0.805386 | 38.873 |

Both candidate blocks again beat both controls. The tagged depth saving is
**0.853 ms/frame**, materially smaller than the ordinary estimate; actual
wall time falls 102.525 ms, approximately 0.90%. The tagged fixture **does not
reach 0.8**. Similar depth graph shapes but different observed gains suggest
host/runtime variability contributes to the ordinary result. Retain the
repeatable benefit without promising a universal 1.947 ms/frame saving.

All twelve tagged WAVs match the user-approved CFG=1 tagged hash
`c8925f3caaa2e58f3460fab6460fcb530330ed45c58dca9b436e90515a78b648`.
Together with the ordinary experiment, **all 24 WAVs match their corresponding
reviewed output exactly**.

All candidate ordinary and tagged logs have zero calculated playback deficit
after the initial chunk becomes available. First audio still requires
approximately 2.5–2.7 seconds; this is not zero startup latency.
Approval carries forward only where output exactly matches a reviewed fixture;
this does not establish equality for arbitrary text or seeds. Keep the user's
requirement for approval of any changed audio.

Commands, conversion verification, hashes and measurements are retained in the
[receipt](../../benchmarks/depth-corpus/flash-head-f16.json). Check it with
`python3 benchmarks/verify_flash_head_f16.py`. See the preceding
[depth-flash experiment](depth-flash-probe-results.md).
