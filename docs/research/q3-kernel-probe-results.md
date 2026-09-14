# Approved Q3 FFN kernel probe — 2026-09-15

The user rejected IQ2 because `(laugh)` became another sigh. This experiment
keeps the approved calibrated Q3 model and deployed Calliope reference fixed.
No new quantization or production changes are made.

Model SHA-256: `34483105709c75c38cdafcddcb4ec91e0b222097d30e76432507c448053a83de`.
The existing real-weight FFN benchmark now accepts homogeneous Q3_K or Q4_K
weights, packs the actual type, and reports it accurately. Mixed-type inputs
are rejected. CPU Q3 n=1/n=2 smoke passes with exact variant equivalence.

## Attribution and existing integer route

An eight-frame logger-enabled Q3/Calliope profile identifies the packed gate/up
projection as the larger FFN cost. Across 112 single-token graph reports,
median reported per-layer gate/up is 108.105 us and down is 53.295 us. Early
reports were slower (roughly 145–150 us gate/up). These are perturbed profiling
measurements, not uninstrumented throughput or a steady-state guarantee.

Twelve-layer real-weight FFN graph, n=1/n=2, twenty warmups and one hundred
measured iterations per variant. Default/integer/integer/default order:

| Route | Packed n=1 mean | Packed n=2 mean |
|---|---:|---:|
| Default | 1844.637 us | 1874.904 us |
| Forced integer-dot | 1964.091 us | 1879.024 us |

The existing integer route is 6.48% slower at n=1 and essentially unchanged
at n=2. The FORCE flag is used only in this isolated FFN graph, never full
speech. It is not retained as a product change. Intra-mode equivalence passes;
that does not imply numerical equality across floating/integer routes.

Logical FFN weights total 129,761,280 bytes per pass, implying approximately
70.3 GB/s from the default graph time. This includes operators and dispatch
and is not a DRAM counter or proof that hardware bandwidth is saturated.

## Row-count specialization

The existing floating Q3/F32 single-column shader uses two rows per workgroup.
A four-row specialization reuses the existing shader and specialization
constants; workgroup dispatch divisor and NUM_ROWS change together. Only AMD
Q3/F32 n=1 is affected. Down projection normally uses the integer route;
n=2 and other types retain their original specialization. Candidate is opt-in.

Four-row pilot source: root `1596522`, ggml `4cb6105a`. Default/four/four/default
microbenchmark order:

| Run | n=1 | n=2 (unchanged specialization) |
|---|---:|---:|
| Default A | 1784.322 us | 1849.919 us |
| Four A | 2049.194 us | 2015.468 us |
| Four B | 1868.723 us | 1845.246 us |
| Default B | 1874.317 us | 1942.397 us |

All floating-path output checksums match across modes. No reliable speed gain
is demonstrated, and unrelated n=2 variation shows shared-host noise. Four
rows are not selected. The final bounded counterpart tests one row against
two (`GGML_VK_TEST_Q3_ROWS=1`); `4` selects the archived four-row variant and
all other values keep the default. Final probe ggml source is `2adf0138`.

One-row final comparison (`b5d7bdc` root / `2adf0138` ggml):

| Run | n=1 | n=2 (unchanged specialization) |
|---|---:|---:|
| Default A | 1945.948 us | 2003.410 us |
| One A | 3197.904 us | 2058.492 us |
| One B | 2003.873 us | 1996.225 us |
| Default B | 2037.427 us | 2070.802 us |

One row also shows no reliable benefit. Its first n=1 measurement is a large
outlier; the second merely overlaps controls. All floating-path checksums
match across one/two/four rows, and all variant equivalence checks pass.
Synthetic normalized inputs, fixed variant order, explicit intermediate
outputs, and shared GPU timing limit whole-product extrapolation.

## Decision and restoration

No new runtime optimization is retained. Keep the approved Q3 weights and
original two-row Vulkan specialization. Global integer dispatch is slower;
one/four-row variants provide no reliable gain. No candidate whole-speech
speed claim or quality change follows from these rejected microbenchmarks.
The baseline performs approximately 70 GB/s of logical FFN weight processing,
so another workgroup tweak is not an evidence-backed promise of the missing
4–8 ms/frame. Further work needs new attribution of remaining depth costs or
a specific shader instruction bottleneck, rather than repeating these knobs.

Backend pin restored in `8606fbc` to `acc4d4d7`. The pre-experiment CLI,
FFN benchmark and Vulkan library were restored byte-for-byte in the dev pod;
the restored library contains no experimental Q3 row selector. Backend source
was touched so the next CMake build recompiles the restored source rather
than reusing the probe object. Production was untouched throughout.

Retained: Q3 support in the existing benchmark, profile and microbenchmark
receipts, archived patches, and the user's IQ2 delivery-tag rejection.
The row-count patches are experimental evidence, not retained product code.

Post-restoration ordinary Calliope speech and `(sigh)`/`(laugh)` at guidance 2
match the previously reviewed Q3 WAVs byte-for-byte. No new listening approval
is requested. Receipt: `benchmarks/depth-corpus/q3-kernel-probe.json`.
Verify with `python3 benchmarks/verify_q3_kernel_probe.py`.
