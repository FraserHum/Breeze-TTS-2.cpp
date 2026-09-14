# Native IQ depth candidates

Research date: 2026-09-07. This is a source and accounting review only. It
does not add a model format, runtime default, shader, dependency, or GPU
result.

## Decision

Do not switch the depth model to an IQ format before the weighted `Q3_K`
development comparison is complete. If that comparison still leaves a
quality/size shortfall, the smallest bounded native experiment is:

1. **`IQ3_XXS` first.** It is a native 3.0625 bpw codebook format, saves
   14,155,776 bytes over the 36 FFN matrices at the same 780M depth shape as
   the current `Q3_K` accounting, and has a real GGML Vulkan DMMV shader path.
2. **`IQ2_S` second.** It is a more aggressive 2.5625 bpw format, saves
   33,030,144 bytes over FFN `Q3_K`, and its local quantizer accepts a null
   importance vector. Its larger quality risk makes it a second probe, not a
   default.

`IQ3_S` is useful as a quality control because it has a larger codebook than
`IQ3_XXS`, but it is exactly the same packed size as `Q3_K` (110 bytes per
256 weights), so it is not a storage or traffic candidate. Defer `IQ2_XS` and `IQ2_XXS` because they are more aggressive. The
verified weighted-Q3 pipeline now provides importance weights for all 12
layers; the existing Breeze quantizer still passes a null pointer and must
not be used for either format.

These are candidates, not quality or speed claims. The current target is an
end-to-end speech result, and nominal bits or file bytes do not predict it.

## Exact formats and depth FFN accounting

The vendored definitions use `QK_K = 256` and include all scale, sign, high
bit, and lookup-index metadata in the `static_assert`ed struct size. See
[`ggml-common.h`](../../third_party/ggml/src/ggml-common.h#L378-L422). The
depth FFN has 12 layers and three matrices per layer. Each gate/up or down
matrix contains `1024 * 8192 = 8,388,608` weights, or 32,768 256-weight
blocks; the 36 matrices contain 301,989,888 weights.

| Type | Packed fields per 256 weights | Bytes/block | bpw | Bytes for 36 FFN matrices | Delta vs Q3_K | Delta vs Q4_K |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `Q4_K` control | `d`, `dmin`, 12-byte packed scales/mins, 128 quants | 144 | 4.5 | 169,869,312 | — | — |
| `Q3_K` control | 32-byte high mask, 64 quants, 12-byte scales, `d` | 110 | 3.4375 | 129,761,280 | 0 | -40,108,032 |
| `IQ3_S` | `d`, 64 quants, 8 high bits, 32 sign bits, 4 scales | 110 | 3.4375 | 129,761,280 | 0 | -40,108,032 |
| **`IQ3_XXS`** | `d`, 96 packed lookup indices | **98** | **3.0625** | **115,605,504** | **-14,155,776** | **-54,263,808** |
| **`IQ2_S`** | `d`, 64 quants, 8 high bits, 8 scales | **82** | **2.5625** | **96,731,136** | **-33,030,144** | **-73,138,176** |
| `IQ2_XS` | `d`, 32 16-bit index/sign words, 8 scales | 74 | 2.3125 | 87,293,952 | -42,467,328 | -82,575,360 |
| `IQ2_XXS` | `d`, 32 16-bit index/sign words | 66 | 2.0625 | 77,856,768 | -51,904,512 | -92,012,544 |

The deltas are a packed-payload ceiling, not a measured bandwidth or wall-time
gain. Runtime traffic also includes the activation, codebook lookup data,
dequantization instructions, workgroup staging, output, and any copy or
fallback. The existing selective-attention control costs about 4,177,920
extra bytes when it keeps attention at `Q4_K` instead of `Q3_K`
(`17,694,720 - 13,516,800`); retain that cost in any complete model
comparison.

## Quantization provenance and imatrix requirements

The core API already dispatches the native quantizers and initializes their
lookup tables: [`ggml.c`](../../third_party/ggml/src/ggml.c#L7951-L8036)
calls `quantize_iq2_*`, `quantize_iq3_*`, and `quantize_iq2_s` after
`ggml_quantize_init()`. The API-level hard requirement is explicit in
`ggml_quantize_requires_imatrix()`: `IQ2_XXS` and `IQ2_XS` assert when the
importance vector is null. Their implementations also directly dereference
the vector ([`ggml-quants.c`](../../third_party/ggml/src/ggml-quants.c#L3294-L3484)).

In this vendored revision, `IQ2_S`, `IQ3_XXS`, and `IQ3_S` have a null-safe
fallback: their quantizers use weight magnitude-derived weights when no
importance vector is supplied ([`ggml-quants.c`](../../third_party/ggml/src/ggml-quants.c#L3940-L4005),
[`ggml-quants.c`](../../third_party/ggml/src/ggml-quants.c#L4181-L4225),
[`ggml-quants.c`](../../third_party/ggml/src/ggml-quants.c#L5142-L5190)).
That is an implementation fact, not evidence that uncalibrated IQ is good
enough. The first experiment should pass the same activation-aware weights
used by the weighted `Q3_K` comparison whenever available.

The present Breeze CLI is not an IQ converter: it parses only `q2_k`, `q3_k`,
`q4_k`, `q6_k`, `q8_0`, and `f16`, then passes `nullptr` to
`ggml_quantize_chunk()` ([`apps/quantize/main.cpp`](../../apps/quantize/main.cpp#L10-L40),
[`apps/quantize/main.cpp`](../../apps/quantize/main.cpp#L75-L90)). Do not
extend that CLI as part of this research task. A bounded follow-up can reuse
the existing GGML API and a small depth-only conversion seam; no new
dependency or quantization format is needed. Quantize from the verified BF16
depth master at `~/.cache/breeze-models/depth-master/depth-range.bin`
(868,560,896 bytes), never by requantizing the current Q4 file. Both logical
FFN row widths are divisible by 256, satisfying the native block alignment.

The upstream llama.cpp encoding table independently describes IQ2/IQ3 as
importance-matrix codebook formats and records the same nominal rates
([Tensor Encoding Schemes](https://github.com/ggml-org/llama.cpp/wiki/Tensor-Encoding-Schemes)).
Its imatrix tool documents activation-squared statistics, calibration-file
generation, merging, and per-tensor selection
([`tools/imatrix/README.md`](https://github.com/ggml-org/llama.cpp/blob/master/tools/imatrix/README.md)).
Use the current calibration/development split to select settings, then use an
untouched private held-out split for acceptance. “Calibrated” describes how a
candidate was prepared; it does not mean speech quality passed.

## What Vulkan actually supports

This is more than an enum entry, but it is not a packed integer-dot promise.

* `ggml_vk_get_dequantize_mul_mat_vec()` accepts `IQ2_XXS`, `IQ2_XS`,
  `IQ2_S`, `IQ3_XXS`, and `IQ3_S` and returns the native f32/f16 RHS DMMV
  pipeline ([`ggml-vulkan.cpp`](../../third_party/ggml/src/ggml-vulkan/ggml-vulkan.cpp#L7856-L7941)).
* The shader generator emits dedicated packed decoders named
  `mul_mat_vec_iq2_xxs`, `mul_mat_vec_iq2_xs`, `mul_mat_vec_iq2_s`,
  `mul_mat_vec_iq3_xxs`, and `mul_mat_vec_iq3_s`, plus f32/f16 and subgroup
  variants ([`vulkan-shaders-gen.cpp`](../../third_party/ggml/src/ggml-vulkan/vulkan-shaders/vulkan-shaders-gen.cpp#L584-L590),
  [`vulkan-shaders-gen.cpp`](../../third_party/ggml/src/ggml-vulkan/vulkan-shaders/vulkan-shaders-gen.cpp#L734-L746)).
  The corresponding source files are present under
  `third_party/ggml/src/ggml-vulkan/vulkan-shaders/`.
* IQ2/IQ3 matrix pipelines are also registered for the larger matrix path
  ([`ggml-vulkan.cpp`](../../third_party/ggml/src/ggml-vulkan/ggml-vulkan.cpp#L4700-L4721)).
  Their lookup-table shared-memory sizing is accounted for in
  `ggml_vk_matmul_shmem_support`; device limits still decide whether a given
  variant is usable.
* The integer-dot `q8_1` switch deliberately includes legacy, K, and IQ1
  types but excludes IQ2/IQ3. When an FFN request tries MMVQ, the null
  q8_1 pipeline forces the ordinary native IQ DMMV fallback. Thus “Vulkan
  supports IQ3” means a generated packed decode/matvec shader exists; it does
  not mean the AMD integer-dot path used by `Q4_K` is available for IQ3.

For the actual depth shapes, the logical gate/up weights are `[1024,8192]`
and produce `m=8192, k=1024`; down weights are `[8192,1024]` and produce
`m=1024, k=8192`. With `n=1` or `n=2`, top-level dispatch selects
`ggml_vk_mul_mat_vec_q_f16()` ([`ggml-vulkan.cpp`](../../third_party/ggml/src/ggml-vulkan/ggml-vulkan.cpp#L10152-L10159)),
which selects the IQ DMMV pipeline after the q8_1 check. The existing Q4
diagnostic log records, for reference, gate/up `n=1` at 64.187 us and `n=2`
at 63.672 us, and down `n=1` at 68.714 us and `n=2` at 69.924 us
([`profile-8.log`](../../.beehive/agent/BREEZE-DEPTH-NEXT/profile-8.log#L130-L144)).
Those logger-perturbed Q4 numbers attribute operations; they are not an IQ
speed estimate.

The end-to-end IQ cost is therefore: offline BF16-to-F32 conversion and
codebook search; packed codebook/index/sign/scale reads at runtime; scalar
dequant and lookup instructions; f32/f16 activation conversion as selected by
the backend; shared-memory/register pressure; and possible shader-generation
or device-capability fallback. A candidate that falls back to CPU, generic
unpacked dequant, or a missing generated shader is rejected regardless of
its file size.

## Bounded experiment handoff

After weighted `Q3_K` selection, run only these two native controls:

1. Quantize the 36 depth FFN matrices from the BF16 master as `IQ3_XXS` and
   `IQ2_S`. Keep attention and every non-FFN tensor identical to the chosen
   Q3/selective-attention control. Record GGUF type names, per-tensor bytes,
   total bytes, imatrix provenance, and an untouched-tensor hash manifest.
2. First run a CPU dequant/tensor screen against BF16 for all 12 layers and
   the existing selected frames/steps. Then load each model on Vulkan and
   verify that every gate/up/down operation reports the native IQ DMMV path;
   reject a CPU or unexpected generic fallback before timing.
3. Use the existing development corpus for any calibration choice. Freeze it,
   then compare tensor error, code-logit KL/top-k agreement, complete codes,
   WAV hashes, and short free-running speech on the private held-out corpus.
   Compare capture-off and capture-on outputs before using captured blobs.
4. Only a candidate that passes the predeclared quality gate gets three
   resident, capture-off repeats at the real 15-step workload. Report decode,
   dequant, dispatch, and end-to-end wall time separately; do not infer a
   speed result from the byte table.

Reject on quality failure, output-parity failure, native-pipeline fallback, or
no complete-workload benefit. Keep `IQ3_S` as an optional same-size quality
control if `IQ3_XXS` fails for a codebook-specific reason. Try `IQ2_XS` or `IQ2_XXS` with the now-verified importance statistics only
if the less aggressive candidates still leave a measured shortfall.

## When a rotation, custom kernel, or student is justified

The native path should be exhausted first. A rotation is justified only after
calibrated standard `Q3_K` and at least one native IQ3/IQ2 candidate fail with
an identified outlier or layer pattern. It must be tested offline with the
actual depth graph placement and corresponding inverse transforms; the extra
transform, metadata, scratch, and precision costs belong in the complete
operation measurement. QuaRot reports that rotations can remove hidden-state
outliers in LLMs ([paper](https://arxiv.org/abs/2404.00456)); SpinQuant reports
that learned rotation choices matter ([paper](https://arxiv.org/abs/2405.16406)).
Those are transferable design hypotheses, not Breeze speech evidence.

A custom packed kernel is justified only if a native candidate passes the
quality gate and a representative AMD gate/up and down operation shows that
its complete scalar-DMMV cost, rather than its nominal bytes, is the remaining
gap. A student is the later option, after native formats and any bounded
precision/rotation probe cannot meet the quality/RTF target; it needs its own
teacher data, closed-loop speech evaluation, and actual 780M wall measurement.

## Sources and known unknowns

Local primary sources are the vendored GGML format structs and quantizers,
`ggml-vulkan.cpp`, the Vulkan shader generator and shader files, the existing
Q4 profile, and the current quantization reports
([depth quantization pilot](depth-quant-pilot.md),
 [precision frontier](next-precision-quant-frontier.md),
 [FFN feasibility](ffn-kernel-feasibility.md)). General PTQ guidance was
checked against GPTQ and AWQ ([GPTQ](https://arxiv.org/abs/2210.17323),
[AWQ](https://arxiv.org/abs/2306.00978)); their language-model results do not
establish a speech-quality threshold here.

Still unknown and requiring the bounded handoff are: measured AMD IQ shader
occupancy and codebook-LUT cost; actual n=1/n=2 pipeline selection in a real
IQ GGUF; Vulkan fallback behavior across device limits and generated-shader
builds; imatrix sensitivity for this speech/depth corpus; layer/codebook
quality variation; interaction with retained Q4 attention; and whether the
native IQ byte reduction survives complete 15-step wall timing.
