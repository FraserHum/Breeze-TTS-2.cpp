# F32 strided backbone value cache — 2026-09-15

Opt-in experiment: `BREEZE_V_CACHE_TRANSPOSED=1`. Default remains off.
Root source: `e6f15e5`; ggml: `acc4d4d7` (local experimental commit,
reproducible patch in `benchmarks/depth-corpus/f32-strided-v-cache.patch`).
Test device: AMD Radeon 780M, RADV PHOENIX, shared queenbee dev pod.
Production deployment was not modified.

The backbone stores V as `[padded_capacity, head_dim, kv_heads]`, appends
transposed current values, and presents a live-length view during decode.
The existing F32 matvec shader reads the physical row stride without changing
its reduction bound or arithmetic. Prefill explicitly materializes contiguous
V. Saved-prefix snapshots retain their original canonical ordering across
cache layouts and capacities. Depth and other attention callers are unchanged.

## Correctness

Standalone check passes on CPU and Vulkan:

```
backbone-context-bench --backend cpu --check-cache-only
GGML_VK_ALLOW_GRAPHICS_QUEUE=1 backbone-context-bench --backend gpu --check-cache-only
```

It covers live lengths 1, 3, 4, 7, 8; grouped heads; NaN-poisoned unused
capacity; small-to-large snapshot restoration; and exact strided/contiguous
matvec equality. Full backbone runs at 64, 257 and 1024 tokens had identical
hidden/logit fingerprints across off/on, finite outputs, and exact repeats.

## Context timing

Each context used off/on/on/off order, three warmups and ten measured steps
per process, packed Q4_K weights. These are full host backbone calls, not
whole-speech RTF. Median across the twenty measured steps per mode:

| Context | Off | On | Interpretation |
|---|---:|---:|---|
| 64 | 12.767 ms | 12.756 ms | No meaningful change |
| 257 | 14.654 ms | 14.825 ms | No demonstrated improvement |
| 1024 | 27.897 ms | 21.584 ms | 22.6% faster, saves 6.31 ms/step |

The first 1024-token control contained an outlier; median is used above.
Both separate long-context on runs were faster than both off runs.

Audio parity is desirable, not a hard gate. Any changed audio requires the
user's listening approval before promotion; numerical checks alone cannot
provide that approval.

## Ordinary speech

Fixed English report, seed 42, packed Q4_K, three resident repetitions per
process, off/on/on/off. First repetition in each process excluded from means:
off **0.923128 RTF**, on **0.923386 RTF**. No whole-pipeline gain demonstrated.
Backbone decode averaged 14.384 versus 14.076 ms/frame, too small to move the
whole pipeline reliably. All twelve WAVs were byte-identical (`67984c45…db3e`).

## Long saved reference and decision

Same English text, synthetic saved `test68s` reference, seed and reversed
ordering. Warmed mean wall RTF fell from **1.099963 to 1.024546**, a **6.86%**
improvement. Backbone decode fell from **28.451 to 21.781 ms/frame**; depth
remained approximately 50 ms/frame. All twelve WAVs were identical
(`3b991f53…c66ff`). No production requests appeared in the inspected logs
during this test window, but the node is shared and these are not dedicated
hardware measurements.

Retain as an opt-in development experiment. It helps long-reference generation
but does not improve ordinary speech or achieve 0.8 RTF. Do not enable the
production default from this two-workload screen alone. The ggml commit is
local; its full patch is archived for reproduction. No audio approval is
requested because every tested off/on WAV matches exactly; this does not waive
the user's approval requirement for future changes to audio.

Next priority for ordinary-speech 0.8 RTF remains depth FFN weight traffic /
selective precision. Roughly 50 ms/frame in depth alone leaves too little of
the 64 ms/frame budget for the rest. Evaluate worthwhile precision candidates
with paired listening samples instead of stopping solely at numerical drift.

Receipts: `benchmarks/depth-corpus/strided-v-cache.json` (12 context calls,
24 speech WAV fingerprints, exact commands, per-run timings and binary hashes).
Verification: `python3 benchmarks/verify_strided_v_cache.py`.
