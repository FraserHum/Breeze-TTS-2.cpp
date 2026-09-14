# Q3 depth costs and synchronization — 2026-09-15

Retain the regular depth path's coalesced graph/read synchronization
(`1644f52`). Same approved calibrated Q3 model, deployed Calliope reference,
instruction and seed. Production was not changed.

## Attribution

Existing `BREEZE_DEPTH_STEP_TIMING=1`, three ordinary household generations,
114 frames each; first generation excluded. Warm per-frame averages:

| Phase | ms/frame |
|---|---:|
| Host staging | 0.002 |
| Tensor set | 0.013 |
| Graph compute, including synchronization | 42.914 |
| Logits read, including separate barrier/wait | 1.374 |
| Host sampling | 0.290 |

All three timing WAVs match the reviewed Q3 waveform. Host preparation and
sampling cannot close the remaining RTF gap. The previous eight-frame Vulkan
profile identifies FFNs as the largest device category: median 1,948.75 us per
single-token depth graph, versus 358.93 us for other quantized projections,
129.35 us normalization, 119.61 us attention matmuls and 111.85 us output head.
These logger-perturbed category medians are diagnostic, cannot be summed into
a median graph time, and cannot be subtracted from uninstrumented wall time.

## Minimal change and validation

The old path calls synchronous graph compute, then a synchronous tensor read.
The latter submits another host-read barrier and waits on UMA. The new path
uses existing async graph compute and async logits read APIs, then synchronizes
before sampling, capture reads, or reuse of the next graph's shared scratch.
Vulkan's unpinned read can synchronize internally; the explicit final sync
also ensures correctness for other backends. No kernel or sampling changes.
With phase timing enabled, `d2h_ms` now includes the pending graph wait; compare
`comp_ms + d2h_ms` across versions.

Uninstrumented baseline/candidate/candidate/baseline order, three generations
per process, first excluded from timing. Original executable and candidate use
the byte-identical Vulkan library. Warm means:

| Run | Wall RTF | Depth ms/frame |
|---|---:|---:|
| Baseline A | 0.879879 | 44.907 |
| Candidate A | 0.861277 | 43.424 |
| Candidate B | 0.864775 | 43.573 |
| Baseline B | 0.870872 | 44.161 |
| Baseline pooled | 0.875376 | 44.534 |
| Candidate pooled | 0.863026 | 43.498 |

This saves 1.035 ms of depth time per frame and about 1.4% wall time on this
short development prompt. Backbone and vocoder averages remain similar.
Shared-node variation remains; this is not a general workload guarantee or
attainment of 0.8 RTF. No production requests appeared in the checked log window.

All twelve ordinary WAVs match the previously approved SHA. Two additional
Calliope `(sigh)`/`(laugh)` generations at CFG=2 also match the reviewed tagged
SHA exactly. They check the two-branch path, not a CFG=1 throughput comparison.
No new audio quality approval is needed for these identical outputs.

Receipt includes commands, binary/model/voice hashes, stage timings and audio
hashes: `benchmarks/depth-corpus/depth-sync-probe.json`. Run
`python3 benchmarks/verify_depth_sync_probe.py` to check it. Backend remains
`acc4d4d7`; no backend patch or new runtime flag was needed.

Correction from the subsequent [collaborative investigation](rtf-080-unexplored-20260915.md):
F16 heads were already tested on an older baseline (0.464 ms/frame gain), so
the proposal below is a replication, not an unexplored opportunity. Prioritize
the newly identified subgroup and computed-history probes instead.

Original next-candidate proposal: measure the existing F32 depth output head with F16
weights. The profile suggests only a small opportunity (roughly 1.7 ms/frame
for the entire head), so demand a measured gain before requesting listening
approval. It cannot independently close the remaining gap; larger progress
still needs faster FFN execution or a separately quality-approved model change.
