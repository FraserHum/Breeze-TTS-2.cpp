# RTF-080 implementation progress

## Current status

The new [depth quantization pilot](../docs/research/depth-quant-pilot.md) has a
separate **mixed depth Q3_K** candidate: warmed mean wall RTF **0.842859** on
the English report prompt, versus **0.932283** for the current model in this
session. UTMOS is 4.538 versus 4.551, and four repeats per case are deterministic.
This single-prompt result is not speech acceptance, a default change, or an
unchanged-Q4_K speedup. It prioritizes broader Q3_K validation before a custom
rotated backend. Raw runs and limitations are in
[quant-runtime.json](depth-corpus/quant-runtime.json).

The RTF 0.8 wall target is **not achieved**. The retained q4_k runtime path is
the exact conv-tail trim from source commit
`e0c8f9d250c8dc05a5ee6c806db9679f6ef60770`, with the graphics queue kept as
an explicit Radeon 780M runtime choice. Its three-run short-prompt smoke was
wall RTF `0.960376 / 0.944214 / 0.953075` (mean `0.952555`), with zero
measured playback deficit and the validated output SHA. The trim remains the
safe retained optimization; it still leaves the q4_k path above the 0.8
budget.

The bounded queue, trim, HIP, workgroup, backbone, duration, and head-F16
checks are complete for this pass. The rejected experiments and their exact
summary values are collected in [rtf-080-experiments.json](rtf-080-experiments.json).

## Historical queue receipt

This receipt records the already measured warm resident comparison on the AMD
Radeon 780M (`q4_k`, 213 frames, 17.04 s audio, six flushes per configuration).
The compact raw rows are in [rtf-080-queue.json](rtf-080-queue.json), copied
from `.beehive/agent/BREEZE-RTF-080-IMPL/queue-comparison.json`.

The benchmark CLI was commit `0ed7de69817e3b3380664c6d689606fb83541220`,
binary MD5 `a1bb5070e538f2bc45c117431207d023`. The same seeded request was used
for all twelve resident runs. Every run produced SHA-256
`67984c45dd90ffe4f6bba7c34b6c39401a793706f4dedbe9e0897705c1dddb3e`.

| configuration | runs | mean wall | mean wall RTF | first audio mean (range) | max playback deficit (range) |
| --- | ---: | ---: | ---: | ---: | ---: |
| control, queue variable absent | 6 | 18,263.2 ms | 1.07178599 | 3,026.3 ms (2,954–3,119) | 904.5 ms (761.4–1,042.0) |
| graphics queue enabled | 6 | 18,046.8 ms | 1.05908646 | 2,990.8 ms (2,955–3,083) | 717.1 ms (586.6–898.5) |

The graphics queue reduces mean resident wall time by 216.4 ms per 17.04-second
take, or `1.015962 ms` per audio frame. This is below the earlier 1.66 ms/frame
paired estimate. The wall RTF remains above 0.8, and the recorded playback
deficits do not support a no-underrun claim; the target is not achieved.

Retain the queue selection as an explicit Radeon 780M runtime recipe only:

```sh
# control: presence switch must be absent; assigning 0 still enables it
env -u GGML_VK_ALLOW_GRAPHICS_QUEUE ... breeze-cli ...

# measured candidate
GGML_VK_ALLOW_GRAPHICS_QUEUE=1 ... breeze-cli ...
```

The queue setting remains an explicit Radeon 780M runtime choice; no global
queue default changed. The trim default and bounded receipts follow.

## Conv-tail trim receipt

Commit `e0c8f9d250c8dc05a5ee6c806db9679f6ef60770` enables the exact discarded
convolution-history trim by default. The opt-in performance runs below come
from `915103802e4278360ad609118438719d57a42fd6` and used
`GGML_VK_ALLOW_GRAPHICS_QUEUE=1`, `BREEZE_VOC_TRIM=1`,
`BREEZE_VOC_STATEFUL=0`, and `BREEZE_DD_FUSED=0` for three resident repeats:

| quant | wall RTF per repeat | mean wall RTF | first audio mean | playback deficit | output SHA |
| --- | --- | ---: | ---: | ---: |
| q4_k, queue + trim | 0.949824 / 0.946438 / 0.940616 | 0.945626 | 2,955.7 ms | 0 ms in all runs | `67984c45…` |
| q8_0, queue + trim | 1.467017 / 1.454740 / 1.442618 | 1.454792 | 4,567.0 ms | 5,608.2–5,922.9 ms | `1a76fb50…` |

The q4_k trim output hash is identical to the prior validated matmul output;
the q8_0 trim hash is likewise identical to its prior validated matmul output.
The full hashes are q4_k
`67984c45dd90ffe4f6bba7c34b6c39401a793706f4dedbe9e0897705c1dddb3e` and q8_0
`1a76fb50b488ca71f655b5d87804970c725b222ff293d977bf9814ca063cd433`.
Existing quality scores therefore carry over only for this exact benchmark
prompt; no new PESQ or UTMOS scores were run. The GPU differential logs cover
`start={0,16,88}` and `count={1,4,40}` for both quants: all nine cases per
quant pass with infinite SNR, correlation 1.0, and maximum difference 0.
Raw runs and differential cases are in [rtf-080-trim.json](rtf-080-trim.json).

The default-unset q4_k smoke cited in the current-status section (`env -u
BREEZE_VOC_TRIM`, queue enabled) had zero measured playback deficit and the
same prior SHA. This is a separate three-run default check; the small spread
is normal resident timing variance, so it is not combined with the opt-in trim
table.

At 213 q4_k frames, the trim mean is 75.650 ms/frame against the 64 ms/frame
wall budget for RTF 0.8, leaving 11.650 ms/frame (15.4%) to remove. q8_0 is
116.383 ms/frame, leaving 52.383 ms/frame (45.0%). The exact tail trim remains
the retained q4_k optimization, but it does not meet the 0.8 target; stateful
vocoder and fused depth remain off.

The separate Mandarin control/trim check used 209 q4_k frames (16.72 s) and
binary MD5 `195f6d858dacb3221f7f753321d118b0`. Control (`BREEZE_VOC_TRIM=0`)
averaged wall RTF 1.069685 with 720.3–810.3 ms playback deficit; trim enabled
by the default commit (environment unset) averaged 0.957486 with zero deficit.
The trim reduced mean wall time from 17,885.1 ms to 16,009.2 ms, or 8.976
ms/frame, and all six runs produced SHA-256
`1cb3157b35e31f5a2bd753bcbfed6f4ce97632b443c980e1ebd771e33a760be8`.
This confirms the same output on the additional prompt while retaining the
q4 target gap above; it does not claim the 0.8 target is achieved. Raw rows
are recorded in [rtf-080-mandarin.json](rtf-080-mandarin.json).

## Additional bounded receipts

The large-workgroup A/B was rejected. Three candidate runs averaged wall RTF
`1.126708` versus `0.954849` for the control, despite byte-identical WAVs
(`67984c45…` in every run). The candidate depth decode was about 13.8 ms/frame
slower, so the larger workgroup is not retained. The full comparison is in
`.beehive/agent/BREEZE-RTF-080-IMPL/large-wg-comparison.json`.

The HIP/gfx1103 build and native run work, but the three-run mean wall RTF is
`1.183` versus `0.955` for the Vulkan reference. The current HIP deployment
path is therefore rejected on performance; HIP/Vulkan quality is unvalidated
because the runs produced different frame counts and durations. See the
[HIP feasibility report](hip-780m-feasibility.md).

The opt-in backbone timing profile made 639 `n=1` calls across three repeats.
Combined host graph build and allocation for audio embedding plus backbone was
`0.533 ms`; the whole audio-embedding call was `0.205 ms`. This is below the
1 ms retained-change gate, so graph reuse or an embedding merge is not carried
forward. The profile receipt is `.beehive/agent/BREEZE-RTF-080-IMPL/bb-profile-summary.json`.

The paired 212-frame reference workload averaged wall RTF `1.204` with trim
disabled (`1.214298 / 1.202211 / 1.196297`) and `1.094` with trim enabled
(`1.104729 / 1.081474 / 1.094941`), with the exact same output SHA. Trim left
one 18.3 ms readiness deficit and two zero-deficit runs, while control deficits
were 1,225.5–1,404.0 ms. Reference encoding is inside this wall timing, so
this remains a voice-reference workload receipt rather than a text-only or
general no-underrun claim. Raw rows are in
`.beehive/agent/BREEZE-RTF-080-IMPL/reference-control.json` and
`.beehive/agent/BREEZE-RTF-080-IMPL/reference-trim.json`.

## Long-output capped stress

The q4_k queue plus default-trim stress used a 3,373-character English prompt,
`--split-chars 10000`, and explicit `--max-new` caps of 125, 375, and 750
frames. The static check confirms one unsplit part. This is a capped-output
throughput test, not complete-utterance quality or listening evidence. It is
separate from the production default split of 600 and the existing short-prompt
receipts. All three repeats reached each cap and had the same SHA within each
case; p50/p95 below are descriptive sample statistics for `n=3`, not population
guarantees.

| cap | actual output | wall RTF repeats | mean | sample p50 | sample p95 | playback deficit range |
| ---: | ---: | --- | ---: | ---: | ---: | ---: |
| 125 frames | 10.0 s, 125 frames | 1.197980 / 1.180410 / 1.178520 | 1.185637 | 1.180410 | 1.196223 | 361.2–536.0 ms |
| 375 frames | 30.0 s, 375 frames | 1.129213 / 1.134257 / 1.142797 | 1.135422 | 1.134257 | 1.141943 | 2,169.5–2,570.1 ms |
| 750 frames | 60.0 s, 750 frames | 1.163035 / 1.154218 / 1.159962 | 1.159072 | 1.159962 | 1.162728 | 7,313.4–7,775.3 ms |

The compact rows, exact command receipts, raw stage timings, prompt hash, and
output hashes are in [rtf-080-duration.json](rtf-080-duration.json). These
long capped outputs remain well above the 0.8 target and show increasing
playback deficit with duration; they do not establish production playback
quality.

## Depth-head F16 A/B

The depth-head F16 A/B is complete: candidate wall RTF was `0.958586 /
0.938310 / 0.947576` (mean `0.948157`) versus the fresh F32 control mean
`0.953963`. All six runs had zero playback deficit and the same validated
SHA-256 `67984c45…`; the other 1,160 tensors were identical. The measured
gain is `0.464475 ms/frame` (0.46 ms/frame rounded), below the 1 ms retained-change gate, so F16 is
rejected for now and no default change is made. Raw rows are in
`.beehive/agent/BREEZE-RTF-080-IMPL/head-f16.json` and
`.beehive/agent/BREEZE-RTF-080-IMPL/head-f16-control.json`.

The compact machine-readable rows for the rejected workgroup, F16, backbone,
and reference A/B receipts are in
[rtf-080-experiments.json](rtf-080-experiments.json); the full raw rows remain
in the ledger paths cited there.

## Next bounded work

The subsequent [depth investigation](../docs/research/depth-next-results.md)
validated the small activation corpus, screened simple sparsity/low-rank
candidates, and rejected the tested integer multi-row shader variants. No
additional performance change was retained; RTF 0.8 remains unachieved.

The real-weight FFN graph probe separates isolated submission overhead from an already-batched product. A full-generation A/B of native SwiGLU preserved all 14 WAV hashes but saved only **0.872 ms/frame** in the warmed comparison, below the 1 ms retention gate; the original implementation was restored. The BF16 depth master is now verified locally, enabling the next quantization study. No new product default or RTF-0.8 claim follows. See [kernel results](../docs/research/ffn-kernel-feasibility.md) and [product A/B](depth-corpus/swiglu-product.json).

That report contains the updated priority order.

The model-track work uses the [shared activation and teacher corpus
handoff](../docs/handoffs/breeze-investigations/00-shared-activation-corpus.md)
for the [codebook-conditioned FFN sparsity](../docs/handoffs/breeze-investigations/01-codebook-conditioned-sparsity.md)
and [activation-aware low-rank FFN](../docs/handoffs/breeze-investigations/02-activation-aware-low-rank.md)
measurements, then assess the [parallel depth student](../docs/handoffs/breeze-investigations/03-parallel-depth-student.md)
and [direct downstream student](../docs/handoffs/breeze-investigations/04-direct-downstream-student.md)
proposals. Those handoffs define research prerequisites; they record no
completed training or performance measurements. Bespoke FFN-kernel and sampler
work is unimplemented and remains an open avenue, so this bounded pass does
not claim that engine work is exhausted.


## Broader mixed depth Q3_K matrix — 2026-09-06

Four resident repeats across 14 cases preserve identical WAVs within every case.
Q3_K is faster throughout, but warmed p50/p95 miss 0.8 on every workload. Mean
RTF is 0.847 held-out English, 0.870 Mandarin, 0.988 uncached reference and
1.045 at 60-second unsplit capped stress; that stress still has a 1.539-second
maximum warmed delivery deficit. Existing saved-voice caching lowers the reference
case to 0.892 with byte-identical audio and about 1.65 seconds earlier first audio.

ASR adds no Q3_K disagreement in these few natural fixtures; speaker cosine is
close to control. Quality remains inconclusive because these are limited proxy
checks with a synthetic reference and no listening/independent human-voice gate.
No defaults changed. Next: complete default-split product measurements, then
context-dependent backbone investigation and expanded Q3_K quality/calibration,
ahead of small sampler finishing work. See the [full report and ordered tasks](../docs/research/depth-quant-matrix.md),
[runtime receipt](depth-corpus/quant-matrix.json), and [speech receipt](depth-corpus/quant-speech.json).

## Default-split long-form and backbone context — 2026-09-06

Six complete-input cases, four resident repeats each, retain identical WAVs within each case. Mixed depth Q3_K mean RTF is **0.897 English, 0.899 Mandarin and 0.911 saved voice**, versus Q4 **0.977, 0.994 and 1.005**. Q3 has no warmed flush-readiness deficit, but still misses 0.8 by 7.8–8.9 ms/frame. This is not an audio-device underrun test.

Long Mandarin quality is a blocker: Whisper medium disagreements are 158/233 Q4 and 103/233 Q3. An instruction-only Mandarin control improves Q4 to 34/233 but leaves Q3 at 110/233. These are proxy diagnostics, not listening or accepted quality. Calibration and targeted precision now need broader language coverage before Q3 deployment.

The new diagnostic tool measures existing backbone calls at fixed contexts: **12.76 / 14.99 / 17.99 / 28.05 ms** for 64 / 256 / 512 / 1024 tokens, with exact repeated output checks passing. Separate profiler traces implicate QK and context-growing contiguous-copy work. Next engine experiment: existing native backbone flash attention, with full numerical/speech A/B; no speedup is yet claimed. No production defaults changed.

See [full findings](../docs/research/default-split-longform.md), [updated ordered tasks](../docs/research/q4-k-rtf-080-plan.md), [runtime receipt](depth-corpus/quant-longform.json), [ASR diagnostic](depth-corpus/quant-longform-asr-diagnostic.json), and [context receipt](depth-corpus/backbone-context.json).

## Native backbone flash attention rejection — 2026-09-07

The opt-in native `ggml_flash_attn_ext` backbone path failed the unchanged full-model numerical gate of `max_abs <= 0.01` and `relative_l2 <= 0.001`, so throughput was not measured and no saving is claimed. The isolated CPU F32 primitive passed five tested cases with worst `max_abs=2.98023224e-08` and `relative_l2=2.23709303e-07`; this validates only those small mappings. The full 28-layer Q3_K A/B remained finite with valid shapes and positions but failed on CPU (context 4 hidden `0.101967663/0.0308438608`, logits `0.199326381/0.0362246698`) and on Vulkan contexts 1, 4, and 64 (all return code 7). The exact cause remains unresolved; primitive agreement does not justify a definitive backend/operator diagnosis.

The runtime/tool changes were reverted; the candidate patch and numerical receipt are archived: [backbone-flash-candidate.patch](depth-corpus/backbone-flash-candidate.patch) and [backbone-flash-numerical.json](depth-corpus/backbone-flash-numerical.json). The baseline CLI MD5 remained `067a25764c424185bd684f60354852e0`. Native flash stays rejected under the current precision path and may reopen only with concrete precision-preserving evidence and a passing full-model numerical A/B.

Next order: execute the language-explicit Q4_K/Q3_K pairs and expand all-layer/late-frame calibration from the [manifest specification](depth-corpus/language-quality-manifest.json); compare calibrated standard Q3_K/selective precision; conditionally test rotation/packed feasibility; do small finishing work only if the measured gap warrants it; defer the student track until then.
