# 780M Vulkan opportunities

Research-only note for BREEZE-RTF-080, pinned to the local ggml fork at
`e7d634a0a457d37eb4d3b70a90d3cdf19bcd306d` (2026-09-04). This note ranks
quality-gated, reversible paths. It does not change the product or claim a
speedup that was not measured on the 780M.

## What the current measurements say

The validated q4_k matmul-only run is 213 frames on the exact report prompt.
Its per-frame decode costs are backbone 15.12 ms, depth 52.34 ms, and vocoder
17.98 ms, or 85.43 ms/frame before the stage-only work. The source and quality
gates are recorded in the [780M benchmark](../../benchmarks/audio-cpp-780m.md).

The manager's exact-prompt RTD probe decomposed the 52.340 ms depth result as:

| phase | ms/frame | interpretation |
|---|---:|---|
| graph compute | 50.544 | `ggml_backend_graph_compute`; includes queue submission and waits, so it is not a GPU-kernel-only counter |
| tensor readback | 1.460 | 15 per-step logits readbacks in the static path |
| sampling | 0.320 | host CFG combine and sample |
| tensor set | 0.013 | per-step input uploads |
| staging/other | 0.002 | timing probe remainder |

The static depth path has 12 transformer layers and 15 sequential steps per
frame. The [depth scratch analysis](depth-decoder-scratch-buffers.md) and
[static-graph analysis](depth-decoder-static-graph.md) document why it still
does 15 graph computes and 15 readbacks. A 16-frame Vulkan performance-log run
measured 57.57 ms/frame with instrumentation and attributed about 34.9 ms/frame
to the three q4_K FFN matvec families (gate, up/SILU, down/add). That run is
diagnostic: the logger itself changes timing, so 34.9 ms must not be subtracted
from the 52.34 ms validated result. See the [manager's perf log](../../.beehive/agent/BREEZE-RTF-080/kernel.log).

### Bandwidth arithmetic, not a hardware counter

The checkpoint contains 187,662,336 bytes of stored q4_K transformer-block
weights and 126,013,440 bytes in the F32 codebook head
([checkpoint facts](../../benchmarks/depth-frontier-780m.md)). If every block byte
is streamed once for each of 15 steps, that is:

```
187,662,336 * 15 = 2,814,935,040 bytes = 2.62 GiB/frame
```

Reading one [1024, 2051] codebook slice at each step accounts for the full head,
another 126,013,440 bytes (0.117 GiB). Dividing these lower-bound byte counts by
52.34 ms gives 53.8 GB/s for the block alone or 56.2 GB/s including the head.
These are effective-stream estimates under an explicit one-read-per-byte
assumption; cache reuse, other tensors, compression layout and driver traffic
can all change actual memory traffic. No DRAM or UMA bandwidth counter was
collected, so these numbers are not a measured bandwidth claim.

## Best bounded backend results

### 1. Allow the RADV graphics queue: measured candidate

The fork exposes `GGML_VK_ALLOW_GRAPHICS_QUEUE`. The source comment says that
allowing the graphics queue can improve RADV performance, and the queue-family
selection is at
[`ggml-vulkan.cpp`](../../third_party/ggml/src/ggml-vulkan/ggml-vulkan.cpp#L6551-L6557).

On the exact q4_k prompt, the full comparison (two runs per configuration) was
stage RTF 1.060628 with `GGML_VK_ALLOW_GRAPHICS_QUEUE=1` versus 1.081414 for
the control, with identical output SHA. At 80 ms/audio frame this saves about
1.663 ms/frame in the measured stage sum. It is the best measured backend knob
in this run, but the result should still be retained as a repeated exact-prompt
receipt because stage RTF includes clock and pipeline effects. It changes queue
selection only; keep the output hash and quality gate attached to any deployment
decision. The control must have the variable absent: this is a presence check,
so `GGML_VK_ALLOW_GRAPHICS_QUEUE=0` still enables the option.

### 2. Disable Q8_1 MMVQ for an A/B: measured, unpromising single probe

The fork has two reversible switches: `GGML_VK_DISABLE_MMVQ=1` forces the
native dequant matvec path, while `GGML_VK_FORCE_MMVQ=1` forces MMVQ. The mode
selection is in
[`ggml-vulkan.cpp`](../../third_party/ggml/src/ggml-vulkan/ggml-vulkan.cpp#L7165-L7181).
With no override, AMD batches (`n > 1`) use MMVQ, and AMD q4_K with `k >= 2048`
also falls through to MMVQ. On non-GCN AMD, the Q4_K Q8_1 pipeline is configured
with `rm_kq_int=1`; only the AMD_GCN branch raises the K-quant row count to 4
([pipeline setup](../../third_party/ggml/src/ggml-vulkan/ggml-vulkan.cpp#L5255-L5268),
[selection logic](../../third_party/ggml/src/ggml-vulkan/ggml-vulkan.cpp#L9526-L9575)).

The primary upstream empirical report
[llama.cpp issue #21151](https://github.com/ggml-org/llama.cpp/issues/21151)
measured Q4_K MMVQ on an RX 7800 XT (gfx1101) at 2,544 us versus 166 us for
native f32 dequant in one shape (15.3x), and 11,465 us versus 1,696 us in
another (6.8x). It attributes the difference to one-row K-quant MMVQ, a
separate Q8_1 quantization dispatch/barrier, and the `repack4/get_dm_scale`
work. That report is hardware-specific and is not an upstream validation of
the 780M; it just makes this the most valuable low-effort A/B.

The manager's first exact-prompt `GGML_VK_DISABLE_MMVQ=1` probe reached decode
RTF 1.066362 and stage RTF 1.075171 over 212 frames, with backbone 14.974,
depth 52.436, and vocoder 17.899 ms/frame. The contemporaneous control was 213
frames with depth 52.715 ms/frame. The output SHA changed to
`6ed8c21088b0e01b299118ace9f73e7c71ef478676372ddce4b9f73392499a74`, and the
different duration means byte equivalence cannot be claimed; no additional
quality score was run because the knob showed no substantial benefit. Keep this
switch as a diagnostic for future driver/kernel changes, but do not prioritize
it as the large win suggested by the RX 7800 XT report and do not claim that
report's 6.8–15.3x as a Breeze result.

### 3. Submission batching: measured negative result

`GGML_VK_MAX_NODES_PER_SUBMIT` defaults to 100 and is clamped to at least one in
the fork ([source](../../third_party/ggml/src/ggml-vulkan/ggml-vulkan.cpp#L6524-L6531)).
The manager's exact-prompt `1000` test produced stage RTF 1.08361 versus 1.08238
for the control, so there is no observed win. The current scheduler also submits
on a FLOP threshold and near graph completion; the upstream change that replaced
weight-size batching with FLOPs is already in the fork:
[90985960](https://github.com/ggml-org/ggml/commit/909859607d438bb42b932e4eae4c714400a96678).
Do not spend implementation time on a larger fixed node cap unless a future
trace shows the node cap, rather than the FLOP threshold, is causing submits.

## Upstream work that is already present or conditional

| upstream source | relevance to this 780M depth path | action |
|---|---|---|
| [77a27eb, coalesce Q4_K/Q5_K scale loads](https://github.com/ggml-org/ggml/commit/77a27eb0408971486fbd44780576d88e2869dacb) | Packs scale loads to avoid compiler-generated redundant loads. The commit reports gains on Intel Xe2/Arc, not RDNA3. | Already in the fork; no new patch. |
| [5840e2a, configurable submit batches](https://github.com/ggml-org/ggml/commit/5840e2a2b1b468d473bf61da17070b216702d9b1) | Supplies `GGML_VK_MAX_NODES_PER_SUBMIT`; the local 1000 test was neutral/slower. | Keep default 100. |
| [ece08bc, host-visible UMA buffers](https://github.com/ggml-org/ggml/commit/ece08bcf1b08d69cfe722cc885d7820b12be37d4) and [c9b26635, contiguous transfer fast path](https://github.com/ggml-org/ggml/commit/c9b26635c1d624097dec5f52c4c52d7d3d630f73) | The device log says UMA; these reduce staging/copy overhead. The RTD probe saw only 0.013 ms of tensor set and 1.460 ms of readback. | Already in the fork; useful hygiene, not a 52 ms/frame explanation. |
| [b953b35, RDNA3 static mat-vec row tuning](https://github.com/ggml-org/ggml/commit/b953b35f964a015a8fe697995a5a02d516d68862) | Current upstream tunes static four-row matvecs for batched Strix Halo cases with more than four columns. The depth perf log's q4_K FFN shapes are n=1 for later steps and n=2 for the first path, so this condition is not hit. | Track for a future batching change; no backport claim. |
| [a4325bb, larger IQ3_S batches](https://github.com/ggml-org/ggml/commit/a4325bb7c0fbffca566eea8387464a6397c63d97) | A 5x result is reported for IQ3_S at batch 8, not q4_K. | Exclude from this plan. |
| [ad916634, MI50 MMVQ tuning](https://github.com/ggml-org/ggml/commit/ad916634268fc0037aafacb08a70dddc9c25534a) | AMD GCN/MI50-specific shader tuning does not establish a Phoenix/RDNA3 result. | Already in the fork; do not extrapolate. |

The fork also exposes `GGML_VK_PERF_LOGGER`,
`GGML_VK_PERF_LOGGER_FREQUENCY`, and `GGML_VK_SYNC_LOGGER`
([initialization](../../third_party/ggml/src/ggml-vulkan/ggml-vulkan.cpp#L7516-L7528)).
Use them for a short diagnostic run only. The logger adds work and its timings
must not be mixed with the validated 52.34 ms/frame result.

### Bounded HIP feasibility check before a custom shader

If the Vulkan A/Bs leave a large gap, a short HIP feasibility probe is a better
next gate than immediately writing a custom shader. The
[ROCm 7.14.1 compatibility matrix](https://rocm.docs.amd.com/en/docs-7.14.1/compatibility/compatibility-matrix.html)
lists Radeon 780M as `gfx1103` and covers Ubuntu 26.04/kernel 7.0, but the
local Vulkan log deliberately does not assign a gfx identifier. Verify the
actual APU/firmware target with the runtime before treating that matrix entry as
applicable. The pod has `/dev/kfd`, but no `hipcc`, so HIP compilation and a
working device runtime are not established.

The probe should be manager-owned and bounded: enumerate the exact device and
runtime, check whether the available ROCm development files can compile a small
HIP target, and verify the `col2im_1d` operator before estimating a vocoder port.
The local `ggml-hip` backend exists and its CMake file globs the CUDA `.cu`
sources ([CMake](../../third_party/ggml/src/ggml-hip/CMakeLists.txt#L60-L63)),
including [`col2im-1d.cu`](../../third_party/ggml/src/ggml-cuda/col2im-1d.cu).
The shared CUDA dispatch handles `GGML_OP_COL2IM_1D`
([dispatch](../../third_party/ggml/src/ggml-cuda/ggml-cuda.cu#L2328-L2329)) and
advertises F32/F16/BF16 contiguous support
([support check](../../third_party/ggml/src/ggml-cuda/ggml-cuda.cu#L5177-L5181)).
The pod still has no `hipcc`, so a compiled HIP artifact and runtime correctness
are unestablished. This is an environment feasibility check, not evidence of a
HIP speedup; stop if the runtime, operator coverage, or output correctness is
missing.

## Depth execution constraints

`BREEZE_DEPTH_STEP_TIMING=1` is a read-only timing probe. `BREEZE_DD_FUSED=1`
already chains the 15 steps into one graph, but it changes RNG/sampling and is
quality-gated: the q4_k measurement reports UTMOS 4.2649 versus 4.5514 for the
ordinary path and changes the frame count from 213 to 257
([quality record](../../benchmarks/audio-cpp-780m.md#quality-and-defaults)).
That makes fusion a separate model-behavior experiment, not a safe default
Vulkan optimization. The static path's host readback and sampling account for
about 1.78 ms/frame in the RTD probe, leaving most of the measured time inside
`graph_compute`; a host-only round-trip change cannot close the 21.43 ms gap to
the 64 ms target.

The current q4_k FFN matvec attribution and the measured graphics-queue result
make queue selection the actionable backend path. MMVQ remains a diagnostic
switch after the unpromising probe. Kernel rewrites, new fusion, or quantization
changes should wait for a trace that separates native dequant, Q8_1
quantization, and queue wait time on this exact Phoenix device.

## Reproduction matrix for the manager

Use one GPU consumer and the exact validated prompt for each row:

```bash
# control: the variable must be absent; assigning 0 still enables it
env -u GGML_VK_ALLOW_GRAPHICS_QUEUE ... breeze-cli ...

# measured queue candidate
GGML_VK_ALLOW_GRAPHICS_QUEUE=1 ... breeze-cli ...

# MMVQ diagnostic; the first probe was unpromising and changed the output SHA
GGML_VK_DISABLE_MMVQ=1 ... breeze-cli ...

# optional diagnostic only; current default is 100
GGML_VK_MAX_NODES_PER_SUBMIT=1000 ... breeze-cli ...
```

For each run retain stage/decode RTF, the RTD phase totals, output SHA, and the
existing waveform/UTMOS gate. The ellipses stand for the already recorded
command and are intentional here; this note does not run a product benchmark.
