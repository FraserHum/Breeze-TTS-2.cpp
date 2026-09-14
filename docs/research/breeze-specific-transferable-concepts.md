# Transferable mathematical concepts for Breeze's depth bottleneck

Research date: 2026-09-06. Read-only source inspection and primary-source literature review; experiments below are proposals, not results. Companion: [rotation and quantization](rotation-quantization-780m.md).

## Architecture that changes the priorities

The [measured checkpoint account](../../benchmarks/depth-frontier-780m.md) identifies 16 codebooks, 15 sequential depth predictions, 12 shared transformer blocks, hidden width 1024 and FFN width 8192. FFN matrices contain 12 × 3 × 1024 × 8192 = 301,989,888 parameters, about 90.6% of the reported 333,471,744 block parameters. This is a parameter fraction, not a measured time or DRAM-traffic fraction.

Source inspected on this date:

- `DepthRunner::init` in [depth_decoder.cpp](../../src/depth_decoder.cpp) allocates a short, frame-local KV capacity of num_codebooks + 1. `dd_step_body` inserts the backbone hidden at the first step, then runs the same block body at every codebook position; the output head is position-specific.
- [generation.cpp](../../src/generation.cpp) completes all depth codes before feeding the frame into the next backbone step. Future frames cannot simply be batched without changing that dependency.
- `build_audio_embed` in [backbone.cpp](../../src/backbone.cpp) sums learned embeddings of all frame codes for temporal feedback.
- `quantizer_decode` in [codec_decoder.cpp](../../src/codec_decoder.cpp) separately projects codebook zero and a sum of the remaining codec embeddings, then adds the results. This is a different embedding space from backbone feedback.

These features are shared with parts of the dual-AR/RVQ model family; they are not claimed to be exclusive inventions of Breeze. Their combination makes the following concepts unusually relevant here.

## 1. Sufficient downstream representations: predict the two sums

**Mathematical principle:** an intermediate representation can be replaced if the quantities its consumers require are preserved. For code tuple c, Breeze's downstream interfaces depend on two functions: backbone embedding b(c), and reconstructed codec latent z(c). Matching only one is insufficient.

**Proposed adaptation, not a published Breeze result:** train a small conditional student of the backbone hidden and first codebook to predict both b(c) and z(c), possibly with shared stochastic input. It could replace all 15 residual-code predictions while retaining the backbone and waveform decoder. A narrower variant keeps the first k codes and predicts the missing tail contributions in both spaces.

The teacher-generated joint target distribution matters. Plain squared-error regression can average multiple valid acoustic outcomes and produce invalid or muffled latents. A probabilistic student, joint losses, and free-running rollouts are necessary investigations; matching teacher-forced embeddings is not sufficient. Direct latents also require a new internal codec entry point, and no longer provide the original discrete token interface.

**Related evidence:** [ReLMCodec](https://arxiv.org/abs/2608.08286), August 2026, demonstrates that reconstruction quality and token predictability are distinct codec design objectives. It trains a single-codebook representation to preserve phoneme organization while recovering acoustic detail. This motivates separating generation needs from reconstruction needs; it does not demonstrate our two-output student. Replacing Breeze's codec with ReLMCodec would require retraining and is not the recommended first experiment.

**First falsification test:** build teacher tuples (hidden, cb0, feedback embedding, codec latent), train a deliberately small student, and test held-out joint reconstruction plus closed-loop speech. Reject if latency/quality requires a student comparable to the current depth model. Research potential very high; training effort high.

## 2. Coordinate-conditioned generation: turn codebook depth into a query coordinate

**Mathematical principle:** represent a family of outputs as f(condition, coordinate), sharing feature extraction across coordinates. Codebook index can be that coordinate.

[Efficient Streaming TTS Acoustic Model with Depthwise RVQ Decoding Strategies in a Mamba Framework](https://www.isca-archive.org/interspeech_2025/lee25h_interspeech.pdf), Samsung Research, Interspeech 2025, applies implicit neural representations to predict RVQ levels in one parallel pass. It also studies masked iterative depth refinement. The paper reports a speed/quality tradeoff between those alternatives; it does not prove conditional independence of RVQ levels.

**Breeze adaptation:** preserve the temporal backbone and codec, train a depth student conditioned on backbone hidden, cb0, and codebook index. Predict all 15 residual codes together, or use a small number of parallel refinement rounds. This retains complete token tuples and both downstream sums. The shared computation across codebook queries is what could reduce repeated weight fetches; a nominally parallel implementation that independently reloads weights for every query may not realize the saving.

**First test:** compare one-pass and two/four-round students using the same teacher data and free-running evaluation. Larger potential than replacing individual kernels; high training effort. Existing AR weights cannot be converted by changing only their attention mask.

## 3. Conditional computation: specialize FFN work by codebook position

**Mathematical principle:** y = Wx is a sum of input-coordinate contributions. Small contributions can be omitted approximately; structured omissions allow corresponding weight blocks to remain unread.

[Dynamic Input Pruning](https://proceedings.mlsys.org/paper_files/paper/2025/file/afd6374c7f2839cba22f537f15f4f760-Paper-Conference.pdf), MLSys 2025, targets SwiGLU rather than assuming natural ReLU zeros. It prunes MLP inputs and intermediate activations, avoiding a separate sparsity predictor. Its flash-to-DRAM cache policy is not directly our UMA cache hierarchy. The transferable element is input-dependent weight selection.

**Breeze-specific hypothesis:** fifteen known depth positions may have different FFN usage patterns because their input tables, output heads, and RVQ roles differ. Measure per-position activation and output-error statistics. If stable specialization exists, use codebook-position masks with contiguous neuron groups, or a small shared core plus position-specific groups. This could avoid the selection overhead of fully dynamic pruning. It is not evidence that late codebooks are automatically easier or less important.

**First test:** offline block-mask ablations at modest sparsities, reporting both teacher-forced errors and free-running speech. Then benchmark packed sparse weight access. Zeroing values while a dense kernel still reads all weights saves no traffic. A mask applied after gate/up only saves down-projection reads; counting all three matrices as saved would be wrong. This has strong architecture relevance and moderate exploratory effort.

## 4. Data-dependent low rank: compress what the layer does on Breeze inputs

**Mathematical principle:** minimize expected output distortion E[||(W-UV)x||²], not merely weight error ||W-UV||². Real inputs may occupy a smaller subspace even if W is full rank.

[IMPACT](https://aclanthology.org/2026.acl-long.183/), ACL July 2026, uses activation structure and gradient-based importance to choose reconstruction bases. [Swift-SVD](https://arxiv.org/abs/2604.01609), ICML 2026, uses accumulated output covariance and adaptive rank allocation. Its advertised compression-time speedups are not inference speedups.

**Breeze adaptation:** collect activation spectra separately by layer and codebook position, then test shared factors across positions. Its 8192-wide FFNs dominate block parameters. Avoid storing fifteen separate factorizations that erase savings.

For an 8192 × 1024 matrix, a rank-r factorization stores 9216r values instead of 8,388,608. At equal bitwidth, r=512 uses 56.25% as many values. But FP16 factors must have r<256 to beat a 4.5-bit/weight Q4_K matrix even before metadata; a rank-512 FP16 factorization is larger than the existing quantized matrix. Both factor precision and two-kernel overhead matter.

**First test:** singular spectra of activation-weighted outputs and distortion-versus-byte curves for quantized factors. Low exploratory effort, medium implementation effort, unknown quality headroom.

## 5. Predictable evolution: forecast features and refresh on error

**Mathematical principle:** a smooth sequence permits local extrapolation; an inexpensive error signal can decide when to recompute an expensive function.

[TaylorSeer](https://arxiv.org/abs/2503.06923), ICCV 2025, forecasts diffusion features with Taylor expansions. [SeaCache](https://arxiv.org/abs/2602.18993), CVPR 2026, improves cache decisions using spectral structure. These are evidence for diffusion trajectories, not acoustic-codebook trajectories.

**Breeze hypothesis:** inspect same-codebook features across neighboring audio frames before adjacent-codebook features within a frame. Different RVQ indices have different semantics, so treating index as smooth time is especially risky. Cache only selected FFN outputs if a useful predictor exists; retain exact attention/KV updates initially. Reusing entire blocks could leave stale KV state and compound errors.

**First test:** trace feature differences, fit zero/first-order forecasts, and measure sensitivity to approximation. Speech transients and code sampling may destroy predictability. Test the hypothesis before porting a diffusion cache. Low diagnostic effort, high quality risk; lower priority than FFN compression.

## 6. Stage conditioning and progressive representation design

[Apple's temporal/depth detokenizer](https://arxiv.org/html/2607.23811v2), July 2026, revised August, uses DiT-style codebook-stage conditioning with shared projections. This is a direct example of transferring conditioning machinery from diffusion to autoregressive RVQ generation. It still generates RVQ levels autoregressively: “DiT” in the title does not mean parallel sampling. Its device and model timings cannot be transferred to Breeze.

For Breeze, stage-conditioned smaller depth students are worth comparing with an ordinary smaller transformer. Merely sharing its output heads has limited traffic benefit: the F32 heads total about 126 MB/frame versus about 2.81 GB/frame of nominal repeated block reads.

[HARP: Harmonic-Aware Residual Partitioning](https://arxiv.org/abs/2607.16657), Interspeech 2026, trains RVQ groups to refine frequency bands coherently. It explicitly identifies spectral entanglement in ordinary RVQ, explaining why naive truncation is unpredictable. This HARP is unrelated to rotation HARP. Its principle could support adaptive codebook budgets in a retrained Breeze codec, but current codebook identities and backbone feedback would also need adaptation.

## Recommended research order

1. Reuse a single representative activation/teacher corpus to measure codebook-conditioned sparsity and activation-weighted rank. These are the cheapest ways to discover architecture-specific headroom without first designing a new model.
2. If statistics support it, benchmark one block-pruned or factorized Q4 FFN, using actual stored bytes and GPU time rather than nominal FLOPs.
3. In the model-training track, prioritize a coordinate-conditioned parallel depth student and compare a two-output latent student as a higher-risk alternative. Both exploit Breeze's temporal/depth boundary.
4. Test feature forecasting only after confirming smoothness. Keep codec redesign as a longer-term option.

The evidence does not yet select a winning implementation or prove RTF 0.8. All approximation experiments must include full feedback rollouts, English/Mandarin, voice identity, long-form stability, and termination. Report generation distribution changes separately from seeded numerical parity. Do not multiply savings from quantization, sparsity, low rank, and caching without measuring their interaction.
