# q4_k to 0.8 RTF on the Radeon 780M: implementation plan

This is a bounded plan, not an implementation. The [investigation](q4-k-rtf-080-investigation.md) is the evidence record; current static 15-step depth graphs and CFG batching are already present, so they are not repeated work.

Current measured status is maintained in the [RTF-080 implementation progress](../../benchmarks/rtf-080-progress.md). The estimates and historical baseline below remain the pre-measurement plan; use the current receipts for decisions without rewriting this record.

## Current ordered work — after the broader Q3_K matrix, 2026-09-06

The [broader runtime and speech matrix](depth-quant-matrix.md) supersedes the
short-prompt pilot order. Q3_K improves every workload but misses 0.8 throughout;
ASR/speaker probes are encouraging but quality acceptance remains inconclusive.
Existing saved-voice caching preserves exact output and reduces reference Q3_K
mean wall RTF from 0.988 to 0.892. It needs no new cache implementation.

1. Measure **complete default-split long-form product workloads**, including
   saved voices. Keep the unsplit capped stress as a separate failed gate.
2. Profile **backbone context-dependent attention/KV/V-layout cost**, then
   test one justified existing ggml improvement. Backbone rises from about
   14 to 29 ms/frame in the unsplit stress; removable savings are unproven.
3. Expand **handoff 00 quality/calibration coverage** for standard Q3_K;
   use calibrated rounding/selective precision when held-out quality identifies
   the need. This CPU/data work can overlap investigation, with one GPU owner.
4. Reopen **exact sampling/readback plus small SwiGLU/head** finishing work
   only after the accepted candidate has a sufficiently small measured gap.
5. Keep **handoff 05 stronger rotation/IQ3/custom packing** conditional on a
   demonstrated standard-format shortfall and packed-operation feasibility.
6. Then smaller autoregressive depth student, handoff **03**, and conditional
   **04/06**. **01/02** remain parked on the failed sparsity/low-rank pilots.

The detailed task gates and effort are in the matrix report. The held-out gap
is 3.77 ms/frame, Mandarin 5.59, cached reference 7.35 and long unsplit stress
19.58. Do not treat the former 3.43 ms short-prompt gap as the whole goal.
No runtime default changes or unchanged-Q4_K RTF claim follows from this mixed
Q3_K candidate.

## Prior ordered work — before the quantization pilot

This section supersedes the historical order below and the initial priority table in `depth-next-results.md`. Investigate unchanged-Q4_K kernels and better quantization before substantial student training. Current short-prompt resident wall RTF is approximately 0.95 (76 ms per 80 ms audio frame); the 0.80 target is 64 ms, leaving approximately 12 ms/frame. Longer and reference-voice workloads have worse receipts: this short-prompt gap is not the full acceptance matrix.

Scope: existing Q4_K kernel execution, original-master depth quantization, shared calibration/evaluation, and conditional model reduction. No repeat of rejected settings without new evidence; no full custom quant backend before quality and one-kernel feasibility pass. A lower-bit depth candidate is a separately labelled mixed-format model, not an unchanged Q4_K result.

| Order | Atomic task | Initial effort / potential | Deliverable and decision gate |
|---|---|---|---|
| 1 | Establish shape-specific Q4_K FFN baselines and explain the bottleneck. | 1–2 days; gain unknown. | Reuse gate/up `[1024,8192]` and down `[8192,1024]` inputs; cover n=1 and n=2. Benchmark representative weight working sets, inspect shader loads/unpacking/register use and available profiling evidence. Separate dispatch from execution; report unavailable counters explicitly. Choose one concrete optimization or record insufficient evidence. |
| 2 | Establish original-master and calibration prerequisites for handoff 05. | Bounded inventory first; data collection depends on gaps. | Verify checkpoint provenance/conversion and quantizer tooling/resources. Extend handoff 00 with long, reference-voice, language, frame-position and codebook coverage; keep held-out data separate. Do not use requantized Q4_K as the primary source. This CPU/resource work can overlap task 1; serialize GPU jobs. |
| 3 | Prototype one evidence-backed existing-Q4_K kernel improvement. | 1–2 day probe; substantial implementation separately scoped. | Select load/unpack scheduling or gate/up plus activation fusion only from task 1 evidence. Preserve model topology and sampling. Require >=1 ms/frame repeated end-to-end saving and output validation before retention; stop after the bounded probe unless evidence justifies continuation. The old 8–12 ms target is unproven. |
| 4 | Compare calibrated ordinary and fixed-rotation 3-bit depth quantization against Q4_K. | Multi-day feasibility, dependent on task 2; potential traffic reduction, no measured speedup. | Reuse QTIP/YAQA tooling where compatible. Record effective packed bytes, sensitive tensors retained at higher precision, held-out errors and free-running speech quality. Floating-point emulation can establish quality, never speed. Predeclare quality criteria; add learned rotations only for a demonstrated quality shortfall. |
| 5 | Benchmark one packed low-bit depth matvec including transforms. | 1–2 day initial feasibility; port effort depends on format. | Only after task 4 quality passes: compare against the task 3 retained baseline on actual 780M shapes. Include decoding, transform, scratch and metadata costs; do not materialize full dequantized weights in global memory each step. Reject a format whose complete operation is slower or whose projected gain cannot justify integration. |
| 6 | Integrate and validate the winning quantization candidate. | Scope after task 5; gain remains unproven. | Add only necessary converter, metadata and runtime support. Run the acceptance matrix below and record combined measured improvement without adding overlapping estimates. Stop optimization if the target passes. |
| 7 | Implement exact sampling/readback improvements if the remaining gap is small. | 1–2 day investigation; current total cost ~1.8 ms/frame, not all removable. | Preserve RNG, top-k/top-p and forced-code semantics. Use only when the measured gap makes this finishing work worthwhile; otherwise advance directly to task 8. |
| 8 | Establish a smaller autoregressive depth student, then investigate handoff 03 parallel/refinement prediction. | Resource assessment then week-scale training; largest structural ceiling, quality uncertain. | Only if bounded kernel/quant work is insufficient or blocked. Reuse teacher/evaluation data; first preserve all codebooks and both downstream consumers. Compare the smaller AR baseline before a more complex student; validate free-running quality and actual 780M latency. |
| 9 | Screen handoff 04 direct downstream prediction, then handoff 06 feature forecasting. | Conditional research; no allocated speedup. | Advance only after task 8 evidence. Direct prediction must reproduce codec input and backbone feedback; forecasting must first show useful held-out predictability. Cheap offline forecasting screens may reuse existing data, but no runtime implementation without a passing screen. |
| 10 | Verify each retained candidate and the final target across the full workload matrix. | Per-candidate checks plus final benchmark pass. | Use one GPU consumer, resident wall timings with profiling/capture disabled, identical settings and >=3 repeats per workload; publish individual runs and sample counts, not only p50/p95. Apply the existing quality, latency and underrun gates below. Report an explicit remaining gap if any workload fails. This validation is a gate throughout, not deferred until all tasks finish. |

Handoffs 01/02: the initial sparsity/low-rank screens are complete, with no deployment-ready candidate. Pause runtime implementation; reopen only for an improved held-out offline result, using the shared corpus. These failures do not establish that all sparsity or factorization methods fail.

Already completed: graphics queue recipe (~1 ms/frame retained), convolution-history trim (~9 ms/frame, full transformer context retained), HIP feasibility (tested configuration slower), and initial FFN tuning probes (no winning change). Backbone reuse/embedding and head precision remain parked because measured costs/savings are small. Stateful vocoder and fused depth remain default-off under their existing quality/behavior decisions.

Execution checkpoint: the real-weight FFN graph probe and native SwiGLU full-generation A/B are complete. The existing product already batches its depth step; isolated graph batching is not a new product optimization. The warmed SwiGLU A/B saved 0.872 ms/frame with identical audio, below the 1 ms retention gate, so it was restored. Original BF16 depth weights are now cached and verified (112 tensors, 868,560,896 bytes). Calibration expansion and low-bit quality comparison remain next; no custom quant or student is implemented. See [kernel result](ffn-kernel-feasibility.md) and [master prerequisites](rotation-quantization-prerequisites.md).

Evidence: [current progress](../../benchmarks/rtf-080-progress.md), [bounded depth results](depth-next-results.md), [Vulkan research](780m-vulkan-opportunities.md), [rotation research](rotation-quantization-780m.md), and [handoff 05](../handoffs/breeze-investigations/05-rotation-aware-three-bit-depth.md).

## Historical plan — superseded ordering and estimates

## Goal and gates

The validated decode body is about 85.43 ms/frame for 80 ms of audio; RTF 0.8 is 64 ms/frame, and resident wall glue is outside the current stage sum. The queue result saves 1.66 ms/frame, leaving at least 19.77 ms of body reduction plus wall headroom. Treat 85.43 - 1.66 - 8 - 10 - 2 = 63.77 ms as illustrative only: the known ~0.7 ms setup already leaves about 64.47 ms, and overlap/unmeasured glue remain. Aim for <=62 ms decode body before calling the wall target plausible.

Use one GPU consumer, a warm resident process, and >=3 repeats for English, Mandarin, reference voice, and 10/30/60-second generations. Report p50/p95 resident wall RTF and stage RTF, first audio, depth phases, output SHA, quality and underruns. Success requires p50 and p95 wall RTF <=0.80, no underruns, and first audio no slower than the current roughly 3.1 s report-prompt result.

RNG-preserving changes must keep seeded repeatability and the established waveform gate (SNR >=30 dB, correlation >=0.999, max difference <0.03, equal duration). RNG/sampling/frame-count changes cannot use cross-take waveform SNR/PESQ; predeclare UTMOS/listening/intelligibility and forced-token/top-k/top-p gates before measuring, with no post-hoc relaxation.

## Ordered work

MMVQ is closed as an implementation branch in the investigation: its probe is slower/noisy, changes SHA/EOS and supplies no budgeted gain. Revisit only if a future shape-specific trace provides new evidence.

| # | atomic step | effort | expected retained gain | confidence / stop-go |
|---:|---|---:|---|---|
| 1 | Freeze the current q4_k matmul-only receipt, warm resident wall timer, phase timing and quality matrix. | 0.5 d | 0 | High; stop if three repeats are not stable. |
| 2 | A/B `GGML_VK_ALLOW_GRAPHICS_QUEUE` as a presence switch; retain the setting only if the 1.66 ms stage win survives wall timing and quality. | 0.25 d | 1.66 ms measured | High; GO at >=1 ms retained with unchanged SHA/gate. |
| 3 | Prototype exact conv-tail trim in [`codec_decoder.cpp`](../../src/codec_decoder.cpp): retain full 88-frame quantizer/dpre/transformer context, keep `min(P,16)` latent history before the conv stack, and move the caller's skip to that retained prefix. | 1 d | 5–9 ms estimated | Medium; GO only after starts 0/16/88, counts 1/4/40, waveform/duration/hash and underrun checks, with >=1 ms retained. |
| 4 | Run a HIP feasibility probe before bespoke Vulkan kernels: explicit HIP/gfx1103 build, exact APU/firmware/runtime, `/dev/kfd`, all codec ops including custom `COL2IM_1D`, q4_k full numerics and quality. | 1–2 d | Unknown | Medium; GO only if it runs correctly and beats the queue path; no driver upgrade. The ROCm matrix lists gfx1103 for Ryzen 200/PRO200, not this 8845HS. |
| 5 | Profile then tune the q4_K depth FFN gate/up/down families without logger-perturbed timings; preserve the existing 15-step graph topology and sampling. | 1–2 d initial experiment; 3–7+ d for a justified custom kernel | 8–12 ms target, unproven | Medium-low; the 34.9 ms logger attribution is not a gain. Retain any >=1 ms verified end-to-end win, recompute the remaining gap, and continue only when the larger target justifies the extra time. |
| 6 | If a gap remains, test backbone graph reuse plus audio-embedding merge; keep static depth graphs and RNG unchanged. | 1 d | 1–3 ms estimated | Low-medium; GO only at >=1 ms retained end-to-end with no first-audio regression. |
| 7 | A/B F32 depth-head to F16 only after profiling confirms traffic is material; validate Vulkan/HIP type support and q4_k waveform quality. | 0.5–1 d | ~1 ms traffic estimate | Low; GO only at >=1 ms measured, not from the byte estimate. |
| 8 | Prototype an exact GPU sampler/readback reduction only if the remaining gap justifies it; preserve top-k/top-p, forced codebooks and RNG semantics. | 1–2 d | 1–2 ms estimated | Low; reject any approximation that needs a relaxed quality gate. |
| 9 | If bounded engine work still misses 0.8, scope/data/train/evaluate a model branch: 12→6 depth-layer distillation or 16→8 codebook distillation, including changed backbone feedback. | Week-scale | Unknown, highest ceiling | Low until data and training are available; do not delete layers/codebooks from the checkpoint and call it deployment-ready. |
| 10 | Re-run the full matrix, choose the smallest passing default, and stop; leave flash attention/chunk-size tuning as measurement-only follow-ups. | 0.5–1 d | proves target or records gap | High for the gate; ship only when p50/p95 wall and quality both pass. |

Do not stack overlapping estimates. If the combined measured engine path still misses the wall target after the bounded queue, trim, HIP, DD, backbone and head/sampler work, open the model branch; no single kernel result is required to supply the whole gap. Deleting layers/codebooks or using `q4_k-dd2` is not a runtime switch (`dd2` UTMOS 2.9172).

Primary sources: [benchmark](../../benchmarks/audio-cpp-780m.md), [depth frontier](../../benchmarks/depth-frontier-780m.md), [Vulkan opportunities](780m-vulkan-opportunities.md), [static depth research](depth-decoder-static-graph.md), [`codec_decoder.cpp`](../../src/codec_decoder.cpp), [`codec_conv.cpp`](../../src/codec_conv.cpp), and the [probe receipts](../../.beehive/agent/BREEZE-RTF-080/).
