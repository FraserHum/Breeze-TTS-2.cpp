# q4_k to 0.8 RTF on the Radeon 780M: implementation plan

This is a bounded plan, not an implementation. The [investigation](q4-k-rtf-080-investigation.md) is the evidence record; current static 15-step depth graphs and CFG batching are already present, so they are not repeated work.

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
