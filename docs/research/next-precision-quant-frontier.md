# Next precision and quantization frontier

Research date: 2026-09-07. This is a bounded research specification, not a
quality, speed, or deployment result. The target remains Q4-equivalent speech
quality while reducing the depth decoder's repeated weight traffic. Every
candidate below must be measured against the existing controls; no numerical or
speech gate is relaxed.

## What the current graph permits

The depth decoder builds 15 static, step-shaped graphs in dedicated no-alloc
arenas. Each layer appends F32 K/V into a persistent cache with layout
`[head_dim, n_kv_head, sequence]`; the backbone/depth cache inputs remain F32
through this path. `attention()` explicitly permutes Q/K, uses
`ggml_cont(permute(V, 1,2,0,3))`, then returns a contiguous F32 output. The
same lifetime pattern is visible in [`src/common.cpp`](../../src/common.cpp),
[`src/depth_decoder.cpp`](../../src/depth_decoder.cpp), and
[`include/breeze/common.h`](../../include/breeze/common.h). The archived flash
candidate preserved that Q/K/V permutation, cast the mask to F16, called
`ggml_flash_attn_ext`, and set `GGML_PREC_F32`; it still failed the existing
full-model gate (`max_abs <= 0.01`, `relative_l2 <= 0.001`). The isolated CPU
primitive passed, so the cause of the full-model drift is unresolved.

The current capture seam is suitable for calibration if it remains narrow. In
`dd_layer()` capture marks the existing `norm_input`, post-SiLU-times-up, and
down-output nodes with `ggml_set_output()`; it does not add duplicate FFN
operators. `DepthRunner::run()` reads those tensors immediately after each
step's compute and before the next step can reuse the allocator slot. Capture
rejects `BREEZE_DD_FUSED=1` and, on Vulkan, requires fusion disabled because
marking outputs changes allocator reuse and fusion. A safe bounded record is
the default pilot of selected layers `0,5,11`, all 15 steps, frames `0,7,15`,
and a 16-frame cap, with raw F32 blobs plus a JSONL manifest. The concurrent
[language-quality handoff](language-quality-next.md) and [capture audit](depth-next-audit.md)
describe the opt-in `all`/late-frame selections; those expanded selections
are not the default pilot and must pass the same capture-off/capture-on parity
checks. Host-visible backbone hiddens, cb0, and sampled code vectors should be
recorded alongside them. Codec/downstream feedback fields remain deferred.

For GGML logical weight tensors, the depth FFN shapes are gate/up
`[1024,8192]` and down `[8192,1024]`; the corresponding activation views are
`[8192,n_tok]` and `[1024,n_tok]`. On the measured AMD Vulkan device, the current
`ggml_vk_should_use_mmvq()` heuristic rejects MMVQ for AMD `k < 2048`, so gate
and up use the ordinary activation path while down (`k=8192`) uses the MMVQ
integer-dot path. The force-MMVQ experiment changed sampling (216 versus 213
frames) and was not byte-equivalent; it is not evidence for a shippable
precision choice.

The pinned BF16 master is already available and locally verified at
`~/.cache/breeze-models/depth-master/depth-range.bin` (869 MB decimal; the
receipt is recorded in
[`rotation-quantization-prerequisites.md`](rotation-quantization-prerequisites.md)).
The inspected `/models` and pod inventory may list only quantized artifacts,
but that does not block calibration. Use this BF16 master as the reference;
never use Q4 as a BF16 reference by implication.

## Ranked bounded probes

### 1. Route strict F32 FlashAttention per operation

**Evidence.** The vendored Vulkan backend generates both scalar F32-body and
F16-body FlashAttention shaders, but on an FP16-capable device its normal
selection registers the F16 body even when `GGML_PREC_F32` selects an F32 score
accumulator. The cooperative-matrix-1 shader is generated only from the F16
body, and its `O_TYPE` is therefore F16 for the running attention output. The
AMD Radeon 780M receipt reports `fp16=1` and `KHR_coopmat`. See the local
backend/generator and shader sources
[`ggml-vulkan.cpp`](../../third_party/ggml/src/ggml-vulkan/ggml-vulkan.cpp),
[`vulkan-shaders-gen.cpp`](../../third_party/ggml/src/ggml-vulkan/vulkan-shaders/vulkan-shaders-gen.cpp),
[`flash_attn_cm1.comp`](../../third_party/ggml/src/ggml-vulkan/vulkan-shaders/flash_attn_cm1.comp),
and the upstream [shader generator](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-vulkan/vulkan-shaders/vulkan-shaders-gen.cpp).
A recent upstream report, [issue #28124](https://github.com/ggml-org/llama.cpp/issues/28124),
opened 2026-08-31 and still marked **Open** when checked on 2026-09-07,
reports this class of failure: on FP16 Vulkan hardware, requesting
`GGML_PREC_F32` can leave online output accumulators in half precision. Its
proposed fix routes the request to the true F32-body shader. This is evidence
for a fresh probe, not a confirmed fix or a Breeze-specific root-cause
finding.
The public ggml API still defines `GGML_PREC_F32` as an operation precision
hint ([`ggml.h`](https://github.com/ggml-org/llama.cpp/blob/master/ggml/include/ggml.h)),
so this can remain opt-in to FlashAttention without changing vocoder matmuls.

**Probe.** Add an opt-in dispatch experiment that, only for a FlashAttention
node carrying `GGML_PREC_F32`, selects the existing scalar F32-body SPIR-V and
forces the scalar path when the device would otherwise choose cooperative
matrix. Preserve the existing Q/K/V permutations and F32 Q/K/V inputs,
including F32 cache views, plus the mask,
GQA dimensions, and output reshape. Compare explicit attention versus strict
F32 at query/prefill contexts 1, 4, and 64, then run the existing full Q3_K
hidden/logit A/B. Keep the unchanged max-abs and relative-L2 gates. If they
pass, run fixed-seed short English and Mandarin code/WAV checks with capture
off and on; do not infer speech quality from the synthetic tensor gate.

**Prediction and decision.** If F16 `O_TYPE` accumulation is the dominant
source, scalar F32 should materially reduce the full-model drift and may pass
the gate; otherwise the rejected path remains rejected and the cause stays
unresolved. Stop on any non-finite output, shape/position mismatch, or gate
failure. Effort is low-to-medium because the shader is already generated;
there is no speed claim until numerical and speech checks pass.

### 2. Attention-only Q4 with FFN Q3 as a diagnostic control

**Evidence.** The current mixed Q3 control changes all 84 depth attention/FFN
matrices. The retained packed-byte inventory is Q4 attention `17,694,720`
bytes versus Q3 attention `13,516,800` bytes, a delta of `4,177,920` bytes
(about 4.18 MB decimal); FFN Q3 is `129,761,280` bytes versus Q4
`169,869,312`. This is a small, existing-format control rather than a new
quantizer. The upstream ggml format table records Q3_K at 3.4375 bpw and Q4_K
at 4.5 bpw ([format table](https://github.com/ggml-org/llama.cpp/wiki/Tensor-Encoding-Schemes)).

**Probe.** Build from the verified BF16 source, retain only the depth attention
projection/output tensors at Q4, and quantize the 36 FFN matrices to Q3_K.
Use the same seeded sampling, language fixtures, frame cap, and termination
checks as the Q3/Q4 controls. Reuse the capture seam above to compare layer
outputs, logits, depth codes, WAV hashes, and code/WAV parity with capture
disabled. This isolates attention sensitivity without changing the backend
operator or inventing a packed type.

**Prediction and decision.** If Mandarin or held-out code/logit error improves
while FFN-only error remains, attention precision is a likely sensitive
budget. If it does not, prioritize FFN calibration and do not spend the extra
4.18 MB on attention. A result is a quality diagnosis only; it says nothing
about RTF until separately timed.

### 3. Activation-weighted Q3_K, then selective precision

**Evidence.** The local quantizer calls
`ggml_quantize_chunk(..., nullptr)`, so its Q3_K is ordinary uncalibrated
rounding. The vendored ggml implementation accepts a per-element
`quant_weights` array for Q3_K and uses it when selecting scales/rounding; the
relevant seam is [`apps/quantize/main.cpp`](../../apps/quantize/main.cpp),
[`ggml-quants.c`](../../third_party/ggml/src/ggml-quants.c), and
[`ggml.c`](../../third_party/ggml/src/ggml.c). Upstream's imatrix tool describes
collecting activation importance and passing it to quantization
([imatrix README](https://github.com/ggml-org/llama.cpp/blob/master/tools/imatrix/README.md)).
GPTQ's approximate second-order error feedback
([paper](https://arxiv.org/abs/2210.17323)) and AWQ's activation-statistic
saliency ([paper](https://arxiv.org/abs/2306.00978)) support the direction,
but neither paper establishes this codec-conditioned TTS graph or GGML's
exact Q3_K grid.

**Probe.** From the BF16 master, collect activation-derived weights for the
same row/block geometry used by Q3_K. Start with the existing selected layers
and add all 12 layers before making a quality decision. Keep the
calibration/development utterances and held-out utterances disjoint. Quantize
the exact Q3_K representation with those weights, then freeze the candidate.
Use only the calibration/development split to rank sensitive tensors and, if
needed, select Q4 attention or a small set of Q4/Q5 FFN tensors. Record that
selection as a tensor manifest. Only after the manifest is frozen, compare the
untouched held-out split for FFN output error, code-logit KL/top-k agreement,
code sequences, termination, and short free-running English and Mandarin
speech; never use final held-out results to choose tensors or a follow-up
budget. Re-evaluate any frozen candidate on unseen prompts.

**Prediction and decision.** The falsifiable prediction is lower held-out
FFN/output error than ordinary Q3 at the same packed type, without any runtime
format change. If weighted Q3 does not improve the frozen candidate's held-out
error, stop before GPTQ/AWQ-style complexity. If it improves error but speech
still fails, record the method as calibrated but failing the speech acceptance
gate; calibrated is method provenance, not a quality acceptance claim. Any
follow-up selection must return to the development split rather than tuning on
the final held-out split. This is medium effort and has the clearest path to
the existing Vulkan Q3 matvec.

### 4. IQ3 triage, with a gated architecture-consistent rotation screen

**Evidence.** The vendored ggml tree already contains IQ3_S/IQ3_XXS quantizers,
importance-weight interfaces, and Vulkan matvec/dequant pipelines
([`ggml-quants.h`](../../third_party/ggml/src/ggml-quants.h),
[`ggml-vulkan.cpp`](../../third_party/ggml/src/ggml-vulkan/ggml-vulkan.cpp)).
The upstream format table describes IQ3 as an importance-matrix codebook format
at about 3.06--3.44 bpw, while Q3_K is ordinary k-quantization
([table](https://github.com/ggml-org/llama.cpp/wiki/Tensor-Encoding-Schemes)).
This makes IQ3 a more realistic first codebook test than QTIP: QTIP uses
stateful trellis codebooks and separately built CUDA kernels ([paper](https://arxiv.org/abs/2406.11235),
[implementation](https://github.com/Cornell-RelaxML/qtip)), so it has no direct
GGUF/Vulkan path here. IQ3 Vulkan presence proves dispatch feasibility only;
it is not an AMD performance result.

**Probe.** First expose no new runtime format: offline-quantize selected depth
FFN tensors as IQ3_S or IQ3_XXS using the same activation weights as probe 3,
then run the tensor-level held-out and short code/WAV screen. Stop if IQ3 does
not beat calibrated Q3 on both error and the predeclared quality gate. Only if
both ordinary Q3 and IQ3 fail for an identified outlier pattern, test a
full-width FWHT/Hadamard transform in an F32 reference graph before packing.
The transform must be placed consistently around the actual GGML logical
`[1024,8192]` gate/up and `[8192,1024]` down matrices, including the
SiLU/SwiGLU boundary
and inverse transform before the residual; a weight-only right transform is
not an equivalent network. Compare transformed BF16 identity first, then
calibrated 3-bit held-out error with transform and scratch traffic counted.
QuaRot's computational-invariance construction
([paper](https://arxiv.org/abs/2404.00456)) and SpinQuant's learned rotations
([paper](https://arxiv.org/abs/2405.16406)) justify the algebraic test, but
neither supplies a Breeze/SwiGLU/Vulkan implementation. The existing scalar
Hadamard pilot was already worse than Q3_K, so this must be a genuinely
full-width, placement-correct test rather than a repeat.

**Prediction and decision.** IQ3 may improve the quality/bit-rate tradeoff, but
no result is assumed. A placement-correct rotation that cannot beat calibrated
Q3 on held-out outputs is rejected before any packed kernel. A candidate that
passes quality still needs one representative AMD operation benchmark including
FWHT, metadata, scratch, register pressure, and dispatch; do not claim a
speedup from nominal bits or file size. Effort is medium for IQ3 and
high/conditional for rotation.

## Gate interpretation and deferred scope

The full-model FlashAttention failure is a valid rejection under the unchanged
engine numerical gate. It is not proof that generated speech failed: the
primitive passed, the full-model drift's cause is unresolved, and the archived
run did not establish a speech-quality comparison. Conversely, the existing
Q3 ASR/UTMOS and repeated-WAV observations are descriptive probes, not a
Q4-equivalent speech acceptance result. Keep numerical, code/WAV parity,
free-running speech, termination, and timing gates separate.

The next capture should therefore stay optional and bounded: selected FFN
inputs/intermediates/outputs plus host hiddens/cb0/codes, raw F32 blobs and a
manifest, captured immediately after each compute. Do not add a broad logging
framework, duplicate graph nodes, or downstream codec/feedback tensors until a
specific held-out error points there. The verified local BF16 master is ready
for calibration; QTIP still has no ready GGUF/Vulkan path here and remains
deferred.
