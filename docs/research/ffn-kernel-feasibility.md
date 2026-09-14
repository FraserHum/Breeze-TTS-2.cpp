# Q4_K FFN kernel feasibility baseline

## Full-generation SwiGLU verdict

A pod-only replacement of `swiglu_ffn` with the existing `ggml_swiglu_split` expression preserved every WAV byte across 14 generated takes. The initial three-versus-three comparison had a slow first control run. A reversed-order follow-up ran four generations per binary, excluding the first as a predeclared warmup and comparing the remaining three. Control mean wall RTF was **0.936884**, candidate **0.925986**: **0.871831 ms/frame saved**, below the ordered plan's 1 ms/frame retention gate. The experiment is not retained; original source and CLI are restored. This does not disprove the small saving, but it is finishing work rather than the missing ~12 ms/frame.

[Full product receipt](../../benchmarks/depth-corpus/swiglu-product.json) includes all runs, commands, hashes and the warmup policy. No separate speech-quality matrix is needed for this rejected, byte-identical candidate. The shared helper experiment covers backbone/codec callers as well as depth, so its wall saving is not solely depth-kernel time.

Independent manager graph verification used 100 timed iterations, separate from the worker's 50-iteration series. Its n=1 batched means were 2441.543 us (separate) and 2379.779 us (combined); n=2 was 2371.531 and 2400.639 us. Both series are retained separately in the graph receipt; their output hashes agree. These are isolated graph results, not additional additive product savings.

## Next probe

Extend representative calibration and compare calibrated ordinary versus fixed-rotation low-bit depth quantization from the now-verified BF16 master. Keep exact-kernel work open only for a concrete load/unpack or fusion proposal; the measured batching differences alone do not justify a bespoke shader.

## Baseline — real-weight graph, 2026-09-06

The current `depth-ffn-bench` replaces the initial synthetic-only harness. It loads all 36 actual Q4_K depth FFN matrices, uses bounded synthetic inputs, and compares identical operations as split submissions, one graph per FFN, or one graph with 12 independent FFN roots. It also compares the existing `ggml_swiglu_split` operation against separate SiLU and multiplication. The product already batches its 12-layer depth step; the split-to-batched saving is not a new product optimization.

## Evidence

Three process repeats, each with 20 warmup and 50 timed iterations, graphics queue enabled:

| Variant | n=1, microseconds per 12 FFNs | n=2, microseconds per 12 FFNs |
|---|---:|---:|
| Split, 60 submissions | 6847.937 | 6500.445 |
| One graph per FFN, 12 submissions | 4436.297 | 4136.350 |
| Batched, one submission | 2416.002 | 2394.358 |
| Combined SwiGLU, 12 submissions | 3257.689 | 3268.291 |
| Combined SwiGLU, one submission | 2326.927 | 2420.735 |

Every variant and repeat has the same output hash for its n value. CPU smoke checks also agree. This is tensor parity, not speech-quality evaluation. Variant order was fixed, inputs are synthetic, intermediate tensors are marked outputs for split-graph allocation, and the graph omits attention, normalization, residual addition and codebook sampling. Do not extrapolate these means into a claimed RTF.

The batched SwiGLU variant is 89.1 microseconds faster at n=1 and 26.4
microseconds slower at n=2 than the separate batched graph. Proceed only to a
bounded full-generation A/B of this existing operation (completed above); the product already
batches its depth step, so this isolated difference is unlikely to move RTF.

## Reproduction

Reproduce the current graph experiment:

```sh
env -u GGML_VK_DISABLE_MMVQ -u GGML_VK_FORCE_MMVQ -u GGML_VK_PERF_LOGGER \
  GGML_VK_ALLOW_GRAPHICS_QUEUE=1 /src/build/depth-ffn-bench \
  --backend gpu --model /models/breeze-tts-2-q4_k.gguf --layers 12 --warmup 20 --iterations 50
```

[Raw graph receipts](../../benchmarks/depth-corpus/ffn-graph.json) contain the
complete per-run settings, timings, output hashes and parity checks. The
superseded synthetic-only harness is no longer a supported mode of the
executable.

RADV can dump final AMD disassembly directly: `RADV_DEBUG=nocache,shaderstats,cs,asm` produced four compiler-statistics blocks, all with zero register spills and scratch allocation. This corrects the earlier implication that final ISA is unavailable merely because no separate disassembler is installed. Pipeline identities remain unmapped; these are static compiler statistics, not measured memory bandwidth or occupancy. The raw log is in the ignored work ledger and its hash/statistics are in the graph receipt. See [Mesa debug options](https://docs.mesa3d.org/envvars.html).

## Hypotheses and limitations

The isolated-cache hypothesis was inconclusive: the original 12-matrix rotation was not consistently slower than one-matrix reuse. The graph experiment confirms substantial per-call submission/synchronization cost in an artificial split harness, but production already batches its layers. Quantized unpacking/load scheduling remains an open kernel hypothesis; compiler statistics do not identify a runtime bottleneck. No attention, residual-add fusion or sampled-generation speedup should be inferred from the isolated graphs.
