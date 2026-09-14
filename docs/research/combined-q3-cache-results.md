# Approved Q3 plus transposed V cache — 2026-09-15

Development validation after user approval of calibrated Q3 audio, including
the deployed Calliope reference. This is not the reserved final acceptance
matrix. No final held-out text is read or used, and production is unchanged.

Frozen engine: CLI MD5 `45190d015056b16ea4a0d1cbd631cf5c`, source `e6f15e5`,
ggml `acc4d4d7`. Frozen calibrated Q3 model SHA-256:
`34483105709c75c38cdafcddcb4ec91e0b222097d30e76432507c448053a83de`.
The optional value-cache layout is the only off/on difference.

The existing `cal-en-long-123` development passage is used verbatim, seed 123,
instruction `Speak clearly and naturally in English.`, maximum 1200 frames
per part, default splitting, packed weights and existing vocoder flags.
Design mode splits into three parts; the exact deployed Calliope saved
reference produces two parts. Static splitter checks reconstruct the text
without omissions. Calliope reference SHA-256:
`331f8daab3736cf24a31b0d1c54f0b87376ae6e196ec01411a084fc540a5d0d8`.

Each process has three resident repeats; the first is a warmup. Design uses
off/on ordering, Calliope on/off. This reverses ordering across workloads,
not within each workload, and cannot remove all thermal/shared-node noise.
The existing matrix parser records first audio, flush readiness and playback
deficit as well as wall and stage timings. Zero computed deficit is not an
audio-device underrun measurement. Aggregate frames alone cannot prove that
every individual split part terminated naturally; the CLI does not report
per-part termination. That limitation must not be relabelled a passed gate.

## Results

| Workload | Cache off RTF | Cache on RTF | Off/on audio | Max warmed first audio, on | Max delivery deficit, on |
|---|---:|---:|---:|---:|---:|
| long | 0.897033 | 0.860584 | 58.72 s | 2622 ms | 0.0 ms |
| calliope-long | 0.902915 | 0.896995 | 58.40 s | 2975 ms | 0.0 ms |

All twelve runs completed with identical off/on WAVs within each workload.
The design comparison is 4.06% faster overall, but backbone alone drops
17.313 → 15.857 ms/frame while unchanged depth also varies 44.794 → 43.370;
this is not clean attribution of the entire wall gain to the cache. Calliope
improves just 0.66%, which this single-order screen does not establish as a
reliable product improvement. Keep the cache optional and default off.

Neither mode reaches 0.8. The approved Q3 model remains the quality-approved
candidate, not a claimed final accepted deployment. No new cache-related
listening approval is needed for these bit-identical pairs. A bounded native
IQ2_S FFN screen follows to test whether lower weight traffic can close more
of the remaining gap; any changed audio requires separate user approval.

Receipts: `benchmarks/depth-corpus/combined-q3-cache.json`.
Verify: `python3 benchmarks/verify_combined_q3_cache.py`.
