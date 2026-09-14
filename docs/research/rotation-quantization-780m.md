# Rotation-assisted quantization for Breeze on the 780M

Research date: 2026-09-06. Literature review only; no new Breeze benchmarks or implementation changes.

## Objective and accounting

Reduce shared-DRAM weight reads per audio frame in the depth decoder. Rotations preserve full-precision linear algebra when applied consistently to weights and activations; they redistribute outliers so fewer quantization bits may become viable. Rotation alone does not reduce traffic.

Prior session accounting estimated 333,471,744 repeated depth-block parameters, 187,662,336 stored bytes, and 15 evaluations per frame: about 2.81 GB of nominal block-weight reads. This is an accounting model, not measured DRAM traffic. Ideal 3-bit payloads would total about 1.88 GB/frame, before metadata, transforms, padding, and retained higher-precision tensors. Runtime savings cannot be inferred directly from that ratio. A depth-only alternative creates a mixed-format model rather than preserving an entirely Q4_K deployment.

## Findings and priority

1. **QTIP with calibration-aware rounding: first quality experiment.** [QTIP](https://github.com/Cornell-RelaxML/qtip), NeurIPS 2024, combines incoherence processing with trellis-coded weight quantization. Public 2/3/4-bit matrix-vector kernels address batch-one inference directly; the demonstrated fast implementation uses CUDA. [YAQA / Model-Preserving Adaptive Rounding](https://github.com/Cornell-RelaxML/yaqa-quantization), 2025, improves rounding using an approximation to the full-model KL objective and supports QTIP quantizers. Its preparation-stage improvements are attractive before adding runtime complexity. Neither establishes Breeze quality or Vulkan performance.

2. **HARP: learned rotations if fixed rotations leave too much error.** [Paper](https://arxiv.org/abs/2605.29843), May 2026, revised September 2; [code](https://github.com/brain-lab-research/HARP). Calibration fits structured two-sided orthogonal processors. Reported quality improvements are strongest at 2 bits; 3-bit QTIP gains are smaller. The RTX 5080 microbenchmark retains about 90% of fixed-RHT throughput. Extra transforms and their parameter storage must be included in both latency and effective-bit accounting. Public QuIP#/QTIP integrations are useful references, not a ready Vulkan backend.

3. **Speech speculative verification: separate longer-term direction.** [Principled Coarse-Grained Acceptance for Speculative Decoding in Speech](https://arxiv.org/abs/2511.13732), ICASSP 2026, groups acoustically similar tokens to improve acceptance. Exactness is at the group level, not the original token distribution. Applying a small drafter plus batched verification across Breeze's depth codebooks is an untested adaptation; target-weight reuse depends on acceptance, verification kernels, and draft overhead.

4. **ConvRot: implementation reference, not an immediate q4 replacement.** The [original paper](https://arxiv.org/abs/2512.03673), December 2025, targets W4A4 diffusion transformers. Its speedups are not evidence for already-Q4 batch-one AMD decoding. An [August 23, 2026 GGUF fork PR](https://github.com/TheTom/llama-cpp-turboquant/pull/312) retains Q5/Q6/Q8 formats; experimental Q2/Q3/Q4 variants were removed after quality regressions. The PR explicitly lists Vulkan as unsupported. The retained formats do not directly reduce our Q4_K traffic. Treat early benchmark numbers carefully: the discussion contains corrected measurements and backend fixes.

5. **TurboQuant: limited direct relevance.** [Paper](https://arxiv.org/abs/2504.19874), 2025 / ICLR 2026, primarily concerns online vector and KV-cache compression. Breeze's short depth context makes KV compression a much smaller target than weights. [Randomized-Hadamard quantization theory](https://arxiv.org/abs/2605.13810), May 2026, supports efficient structured rotations but does not establish a weight-only runtime win.

## Smallest useful validation sequence

- Use original floating-point weights and representative English/Mandarin, voice, frame-position, and codebook-position calibration activations; avoid re-quantizing Q4_K as the primary experiment.
- Compare Q4_K, ordinary calibrated 3-bit, and fixed-rotation 3-bit depth weights. Retain sensitive tensors at higher precision where needed. Add learned rotations only if justified by quality failures.
- Evaluate output-distribution error per layer and codebook, followed by generated speech intelligibility, speaker similarity, perceptual quality, termination, and repeatability. Existing poor low-bit audio does not rule out better quantizers.
- If quality survives, benchmark one representative packed matrix-vector kernel on the 780M. Count effective bytes, transform cost, register pressure, and scratch traffic. Do not materialize full dequantized weights in global memory on every step.
- Only then consider a full Vulkan integration. No cited work establishes an RTF improvement for Breeze on this hardware.

## Cross-method interaction

Rotation spreads coordinate outliers and may weaken coordinate sparsity. Rotation-aware quantization and activation pruning therefore need a joint experiment; their independently reported savings must not be multiplied together.
