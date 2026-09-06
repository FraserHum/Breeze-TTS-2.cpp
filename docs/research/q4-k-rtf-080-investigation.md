# q4_k to 0.8 RTF on the Radeon 780M: investigation

Research only; no production change or unmeasured speedup is claimed. One frame is 1,920 samples/80 ms, so RTF 0.8 means <=64 ms of resident generation per frame, including wall overhead.

## Baseline and evidence

The validated q4_k default has transposed-convolution matmul on, stateful vocoder off, and fused depth off. The exact report prompt produces 213 frames/17.04 s at 15.12 ms backbone + 52.34 ms depth + 17.98 ms vocoder = **about 85.43 ms/frame**; the fresh benchmark's stage RTF repeats span 1.0729–1.0807. The body is about 21.43 ms above 64 ms before unmeasured resident overhead, so the queue knob alone cannot reach the target.

The default passes the established waveform gate versus legacy: SNR 44.88 dB, correlation 0.999984, max difference 0.017365, PESQ 4.6401, UTMOS 4.5514. The fixed gate is SNR >=30 dB, correlation >=0.999, max difference <0.03, equal duration and repeatability. See the [benchmark](../../benchmarks/audio-cpp-780m.md) and [evidence bundle](../../benchmarks/780m-2026-09-06.json).

Stage RTF is not resident wall RTF: model load and some glue are outside the sum; `graph_compute` includes backend dispatch and waits. Final acceptance needs a resident-process wall timer.

The new Sep 6 queue/profile probes use rebuilt `/src/build/breeze-cli` MD5 `225a3e5afc8c6b3a7d76f6ac83c3a6b3`, seed 42, q4_k, mono 24 kHz, instruction `Speak clearly and naturally.`, one GPU consumer, and 40/40-frame chunks on the logged AMD Radeon 780M / RADV PHOENIX / Vulkan0 / UMA host. They are separate repeat controls, not the earlier full quality matrix.

| row | knob | frames | stage / decode RTF | first audio | SHA256 |
|---|---|---:|---:|---:|---|
| control 1 | graphics queue unset | 213 | 1.080446 / 1.071690 | 3036 ms | `67984c45dd90ffe4f6bba7c34b6c39401a793706f4dedbe9e0897705c1dddb3e` |
| graphics 1 | `GGML_VK_ALLOW_GRAPHICS_QUEUE=1` | 213 | 1.060211 / 1.051397 | 3046 ms | same |
| graphics 2 | `GGML_VK_ALLOW_GRAPHICS_QUEUE=1` | 213 | 1.061045 / 1.052230 | 3004 ms | same |
| control 2 | graphics queue unset | 213 | 1.082383 / 1.073580 | 3189 ms | same |
| submit 1000 | `GGML_VK_MAX_NODES_PER_SUBMIT=1000` | 213 | 1.083609 / 1.074747 | 3055 ms | same |

The two queue pairs save about 1.66 ms/frame in the stage sum with identical output SHA. `MAX_NODES_PER_SUBMIT=1000` is neutral/slower. Full rows are in [knobs.json](../../.beehive/agent/BREEZE-RTF-080/knobs.json).

## Depth profile and bounded opportunities

The current tree already has 15 static depth step graphs and CFG branches batched in one graph, but still does 15 sequential steps and 15 logits readbacks. RTD means are staging 0.002, tensor set 0.013, graph compute 50.544, readback 1.460, sampling 0.320, total **52.340 ms/frame** ([profile summary](../../.beehive/agent/BREEZE-RTF-080/profile-summary.json)). A 16-frame logger run showed about 34.9 ms/frame in q4_K FFN gate/up/down families, but instrumented depth was 57.57 versus 52.34 ms/frame, so that attribution is diagnostic only ([kernel log](../../.beehive/agent/BREEZE-RTF-080/kernel.log)).

The companion [Vulkan note](780m-vulkan-opportunities.md) covers source-level queue/MMVQ evidence. The presence-based `GGML_VK_DISABLE_MMVQ` probe gave stage 1.075171, decode 1.066362, 212 frames versus 213 control, depth 52.436 versus 52.715 ms/frame in that control and 52.34 baseline; SHA changed to `6ed8c21088b0e01b299118ace9f73e7c71ef478676372ddce4b9f73392499a74`. It has no material gain and an EOS/numerics change; do not budget it without a shape-specific trace.

The safest larger opportunity is in [`codec_decoder.cpp`](../../src/codec_decoder.cpp): keep the full 88-frame (`sliding_window + 16`) quantizer/dpre/transformer context, then trim the latent **after** the transformer to `min(P,16)` history plus new frames before ConvNeXt/dhead/decoder blocks, where `P` is the discarded left context. Move the caller's discarded prefix from 88 to the retained history; leave stateful attention untouched. This avoids reconvolving 72 of the 88 history frames without carrying approximate attention state.

The 16-frame history is conservative. Reverse output sample zero through dfin K=7, four causal transpose stages (strides 8/5/4/3, kernels 16/10/8/6), three K=7 residual units at dilations 1/3/9, dhead K=7 and two ConvNeXt K=7/stride-2 stages; using `ceil((n-K+1)/stride)` backwards gives about 10 latent frames. Keep 16 until deterministic tests prove a lower bound. The saving is estimated at 5–9 ms/frame, not measured; test starts 0/16/88 and counts 1/4/40 against the unchanged full-window output, offset, duration and underrun gates.

| candidate | evidence / expected gain | call |
|---|---|---|
| graphics queue | measured 1.66 ms/frame, SHA exact | cheap, retain after resident/quality checks |
| exact conv-tail trim | 5–9 ms/frame estimate | medium, prototype/prove |
| q4_K FFN kernel work | 8–12 ms/frame is an **unproven target**; logger's 34.9 is not a gain | profile first; bespoke Vulkan only after HIP gate |
| HIP feasibility | unknown gain; ggml HIP/CUDA sources include `COL2IM_1D`, runtime unverified | 1–2 day bounded probe; no driver upgrade |
| backbone graph/embedding merge | 1–3 ms estimate | profile first |
| F32 head to F16 | roughly 1 ms traffic estimate, not a ceiling | validate support/quality |
| exact GPU sampler | 1–2 ms estimate | preserve RNG/top-k/top-p/forced tokens |
| flash attention | likely small for depth context 17–32; maybe BB context | measure before coding |
| fewer layers/codebooks | largest ceiling, changes model | long-term distillation; dd2 UTMOS 2.9172 rejects it |

HIP is a feasibility gate, not a speed claim: the ROCm matrix lists gfx1103 for Ryzen 200/PRO200 on Ubuntu 26.04/kernel 7.0, not this observed 8845HS; verify exact APU/firmware, `/dev/kfd`, compiler/runtime, native gfx1103 build, every op/custom col2im, full q4_k numerics and quality. Compare any HIP result with the queue path and choose one; do not stack overlapping gains.

Illustrative only: 85.43 - 1.66 - 8 - 10 - 2 = 63.77 ms decode body, but setup is about 0.7 ms/frame on this prompt and wall glue is unknown. This does not prove 0.8; it needs roughly 2 ms headroom after overlap is measured.

Stateful vocoder reaches about 9.16 ms vocoder but fails q4_k waveform max-difference (0.039093); fused depth changes RNG/frame count and q4_k UTMOS is 4.2649 versus 4.5514; q4_k-dd2 UTMOS is 2.9172; no q4_0 result exists. These remain experimental/rejected defaults.

## Sources and acceptance

[depth static graph](depth-decoder-static-graph.md), [depth scratch analysis](depth-decoder-scratch-buffers.md), [`codec_conv.cpp`](../../src/codec_conv.cpp), [`codec.cpp`](../../src/codec.cpp), [`generation.cpp`](../../src/generation.cpp), [depth frontier](../../benchmarks/depth-frontier-780m.md), and the three receipt files above are the primary local sources.

For acceptance, run English and Mandarin, reference voice, and 10/30/60-second resident generations with one GPU consumer and >=3 controlled repeats; report p50/p95 wall and stage RTF, first audio <= current ~3.1 s, no underruns, hashes and quality. RNG-preserving changes keep the waveform gate and seeded repeatability. RNG-changing paths use a predeclared UTMOS/listening/intelligibility gate plus forced-token/top-k/top-p checks; do not retrofit a relaxed gate.
