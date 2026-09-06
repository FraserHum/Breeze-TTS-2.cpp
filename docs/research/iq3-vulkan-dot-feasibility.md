# IQ3 Vulkan integer-dot feasibility

Repository pin: `8afd448f2b99bdcfca338f14c95455217f15f95d` (`git rev-parse HEAD`).
GGML pin: `e7d634a0a457d37eb4d3b70a90d3cdf19bcd306d`.
This is a source review only; no Vulkan implementation or GPU result is implied.

## Finding

CUDA already proves the IQ3 arithmetic can be expressed as signed packed-byte
dot products. `third_party/ggml/src/ggml-cuda/vecdotq.cuh:1160-1195` implements
`vec_dot_iq3_xxs_q8_1`; `:1198-1238` does the same for IQ3_S. The MMVQ host
selection in `third_party/ggml/src/ggml-cuda/mmvq.cu:9-65` returns both IQ3
functions and their VDR factors. CUDA MMQ also dispatches both types through
IQ3 tile loaders and q8_1 DP4A dots (`mmq.cuh:645-656`). This is an algorithmic
precedent, not evidence that RADV/AMD Vulkan will be faster.

The Vulkan float DMMV path is already complete: `mul_mat_vec_iq3_xxs.comp:10-48`
and `mul_mat_vec_iq3_s.comp:10-50` decode the same grids/signs and accumulate
FMA results; their row/reduction paths are `:50-88` and `:52-89`. The integer
q8_1 path is absent for IQ3, however:

* `vulkan-shaders-gen.cpp:766-776` generates q8_1 DMMV only for legacy/K,
  MXFP4, IQ1; `:1265-1268` applies the same filter to generated declarations.
* `mul_mat_vecq_funcs.glsl:160-193` has MMVQ implementations for Q2/legacy
  only; there is no IQ3 `repack`/`mmvq_dot_product` branch.
* `vulkan-shaders-gen.cpp:624-628` generates q8_1 MMQ only for legacy/K/MXFP4.
  `mul_mmq_shmem_types.glsl:1-90` and `mul_mmq_funcs.glsl:1-456` likewise have
  no IQ3 cache/load/dot branches.
* Host q8_1 pipeline creation omits IQ3 in `ggml-vulkan.cpp:5362-5384` (DMMV)
  and `:4975-4991` (MMQ). Selection rejects q8_1 IQ3 at `:7861-7880` and
  returns null for its MMQ availability through the int flags at `:8860-8865`.

## Smallest port seam

Start with q8_1 DMMV only, because decode is predominantly `n=1` in TTS.
Add IQ3_XXS packed-byte decode and `dotPacked4x8EXT` accumulation to
`mul_mat_vecq_funcs.glsl`, reusing `types.glsl` layouts and the existing
`mul_mat_vecq.comp` reduction. Extend the generator predicate and declarations,
then add the XXS host pipeline entries and the IQ3 cases in q8_1 selection. Keep
the existing float DMMV as the fallback. Implement MMQ separately only if a
measured workload routes there; it needs IQ3 cache structs, load functions,
generator entries, and host entries rather than a name-only alias.

The block layouts are already GGUF-native: `types.glsl:1521-1591` defines
IQ3_XXS (`K=256`, 98 bytes/block) and its 256-entry `uint32` grid; `:1593-1614`
defines IQ3_S (`K=256`, 110 bytes/block), with its 512-entry shared table and
barrier at `:1683-1695`. XXS initialization is `:1573-1585`.
The host shared-memory budget already accounts for 4*256 and 4*512 bytes
(`ggml-vulkan.cpp:3991-4024`), so the table is a reusable cost, not a new
format. No GGUF rotation or format change is required.

## Arithmetic and cost constraints

For XXS, match CUDA’s sign expansion and exact post-scale
`(ls*sumi + sumi/2)/2` (`vecdotq.cuh:1174-1195`); Vulkan’s float shader uses
`db=d*0.5*(0.5+(signscale>>28))` and the parity bit from `bitCount`
(`mul_mat_vec_iq3_xxs.comp:16-43`). For S, preserve qh’s ninth grid bit,
the eight sign bits, and `1 + 2*scale_nibble` (`vecdotq.cuh:1210-1238`,
`mul_mat_vec_iq3_s.comp:16-47`). Signed grids mean q8_1’s zero/sum correction
must not be copied from affine Q4 code; use the activation scale (`ds.x`) and
the same integer sum as CUDA.

q8_1 adds a separate activation quantization dispatch. `quantize_q8_1.comp:37-117`
finds a block max, rounds to int8, writes `d` and `sum*d`; host invokes it via
`ggml-vulkan.cpp:9214-9240` and before matmul at `:9489`/`:9768`. Include this
pass and its synchronization in timing: one vector may lose the weight-bandwidth
gain to quantize/read traffic. Integer accumulation and half scales also add
error versus the existing F32/F16 FMA path; compare outputs, do not infer
quality from throughput.

The depth FFN has gate/up input K=1024 and down input K=8192 (`tools/depth_ffn_bench.cpp:28-35,243-248`). Exercise both `n=1` and `n=2` for all
three projections (`:345-353,581-584`); larger batches may require the separate
MMQ seam. The CUDA implementation has architecture-specific IQ3 MMVQ batch
limits (`mmvq.cu:117-225`), so do not copy a CUDA gate to Vulkan without a
measured route decision.

## Minimal validation before a go decision

1. Use the actual IQ3 control at the path and SHA recorded by
   `benchmarks/depth-corpus/quant-iq3_xxs.json`; verify all 36 FFN tensors and
   finite native dequantization. Reuse the two 64-MiB capped teacher captures
   and hashes in `benchmarks/depth-corpus/capture-expanded.json` (frames
   0,7,15,31,63; 2705 records per fixture) when the temporary capture directory
   is available; regenerate it otherwise.
2. For each IQ3 type, K in {1024,8192}, and n in {1,2}, compare one Vulkan
   q8_1 DMMV output against a CPU ggml reference on identical packed weights
   and captured activation rows. Record finite status, max absolute error,
   relative L2, and checksum; separately compare q8 activation values and
   per-block scales. Include all sign/high-bit/scale branches.
3. Run the same graph with the existing float Vulkan IQ3 DMMV and CPU backend,
   then run a paired full-model speech pilot against the Q4 control using the
   established quality fixtures, ASR diagnostics, timings, and listening.
   The existing `depth_ffn_bench.cpp:243-248` rejects every non-Q4_K tensor and
   uses synthetic inputs (`:348-349,593-595`), so it is a Q4 control today and
   cannot by itself validate IQ3.

Recommendation: conditional go after favorable native IQ3 speech evidence and
a profile identifying enough scalar-DMMV cost to justify the port. A slow
native result can motivate this probe; it does not predict a successful port.
Until then, retain the float IQ3 path/fallback. CUDA IQ3 DP4A support is useful
porting evidence, but it is not an AMD/RADV performance claim.
