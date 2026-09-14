# Depth decoder frontier on the Radeon 780M

The existing `q4_k-dd2` checkpoint reaches realtime on this prompt, but its
UTMOS score drops substantially; it is not a deployment recommendation. Neither halving layers
nor predicting codebooks independently is a quality-preserving inference switch.
See [the measured comparison](audio-cpp-780m.md) for the current quality gates.

## Checkpoint facts

Read directly with `gguf.GGUFReader` from `breeze-tts-2-q4_k.gguf`:

| Property | Value |
|---|---:|
| Codebooks | 16 |
| Depth transformer layers | 12 |
| Hidden / feed-forward width | 1024 / 8192 |
| Attention / KV heads | 8 / 2 |
| Transformer block parameters | 333,471,744 |
| Transformer block stored bytes | 187,662,336 |
| All `dd.*` parameters / stored bytes | 367,073,280 / 314,859,520 |
| Three-dimensional codebook head | F32, 126,013,440 bytes |
| Depth sampling top-k / top-p | 50 / 1.0 |

The block matrices are Q4_K; normalization tensors and the codebook head are
F32. The separate audio.cpp structural report's four-layer description does
not describe this checkpoint. The roughly 338M estimate is close to the block
body plus projection, but excludes the codebook heads. Parameter counts alone
do not prove what percentage of latency is inherently unavoidable.

The pod has `q4_k`, `q8_0`, and `q4_k-dd2`. **No q4_0 measurement is available.**
The local quantizer's `parse_type` also does not accept q4_0. Q4_K and Q4_0 must
not be treated as interchangeable results. Requantizing an existing quantized
file would introduce another quality variable; use original floating weights
for a clean new quantization experiment.

## What can change

| Candidate | Expected benefit | What must be validated |
|---|---|---|
| Existing `q4_k-dd2` | Fewer depth-weight bytes, same 15-step architecture | Naturalness, intelligibility and drift with utterance length; existing documentation reports degradation past roughly 45 seconds |
| 16 → 8 codebooks | 15 → 7 depth steps, an optimistic 53% reduction of the depth stage | Train/distill or explicitly evaluate a lossy truncation; missing codebooks also change backbone feedback |
| 12 → 6 depth layers | Roughly half the block work; projection, heads and overhead remain | Train/distill a smaller decoder; removing layers from this checkpoint is an ablation, not a validated model |
| Parallel codebook prediction | Remove the sequential depth-step dependency | Train a predictor with a different conditional factorization; ordinary batching cannot remove the dependency |
| Quantize the F32 depth head | Reduce part of the remaining weight traffic | Quantization quality and 3-D tensor support; head reads are only a small fraction of repeated block traffic |

`DepthRunner::run` feeds each sampled code into the next step.
`build_fused` preserves that chain (`aud = amax`); fusion reduces submissions,
not the number of dependent predictions. `audio_embed_forward` sums all
codebook embeddings into the next backbone input. The vocoder then reconstructs
its latent from the semantic and acoustic codebooks. Reducing codebooks therefore
changes more than the output's residual acoustic detail.

For measured backbone, depth and vocoder costs B, D and V in ms/frame, optimistic
**decode** RTF estimates are `(B + D*7/15 + V)/80` for eight codebooks and
`(B + D/2 + V)/80` for six layers. These are sizing estimates, not achieved
performance: they assume unchanged output duration and near-linear depth cost,
and exclude setup. The layer estimate overstates savings by halving fixed work.
Batching independent requests may improve aggregate throughput, but does not
establish lower single-request latency.

The local quantizer keeps non-2-D tensors unchanged, which explains the F32
head. Reading all block weights 15 times would move about 2.81 GB/frame at Q4_K,
before heads and activations. That is a bandwidth-pressure estimate, not measured
DRAM traffic: caches, dispatches and kernel efficiency still matter.

## Measured model-side trial and next experiment

On the same `18ff504` Vulkan binary, exact prompt and seed 42, with matmul ON
and stateful/fused OFF, `q4_k-dd2` produced 244 frames / 19.52 seconds:

| Trial | Decode RTF | All measured stages RTF |
|---|---:|---:|
| Repeat 1 | 0.8899 | 0.8975 |
| Repeat 2 | 0.8984 | 0.9061 |

The WAVs reproduced byte-identically (SHA256
`81395b93863f906dde53a1e506f1e456ac0386d365438e4a6ce98b212e322188`).
UTMOS was **2.9172**, versus **4.5514** for the ordinary q4_k model with matmul.
Different code sampling and durations confound a direct waveform comparison,
but this large naturalness-proxy drop rejects this trial as evidence of a
quality-neutral realtime deployment. The documented long-form drift is not
the only concern: this sample is under 20 seconds. No cross-model PESQ/SNR claim
is made.

Next, evaluate intermediate depth quantization from floating master weights,
or train/distill a smaller depth decoder. Use several English and Mandarin
prompts, reference-voice cases and 10/30/60-second utterances, with intelligibility
and listening checks alongside UTMOS. A smaller or parallel depth model needs
a training/distillation dataset and evaluation corpus, neither supplied here.
Do not spend implementation effort simply deleting layers or codebooks.

Source: [configuration](../src/config.cpp),
[depth generation](../src/depth_decoder.cpp),
[backbone feedback](../src/backbone.cpp),
[vocoder reconstruction](../src/codec_decoder.cpp), and
[quantization policy](../apps/quantize/main.cpp).
