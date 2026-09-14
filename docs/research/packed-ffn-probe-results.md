# Packed Q4_K FFN probe — September 14–15, 2026

**No runtime change is retained.** The shape-specific integer dispatch saved little and changed the generated speech. The benchmark now covers actual packed projections, and the rejected four-line probe is archived for reproducibility.

## Experiment

Extended the existing `depth-ffn-bench` with a packed variant: concatenate the original Q4_K gate/up payloads without requantization, run the same native SwiGLU as the product, then apply the original down projection. It loads all twelve real depth-layer weight sets and covers n=1/n=2. Existing separate variants remain numerical controls. CPU smoke and all Vulkan intra-mode packed/unpacked comparisons pass; these do not establish equality between floating and integer dispatch modes.

The current AMD heuristic rejects integer MMVQ for k<2048 regardless of output rows m. Packed gate/up has m=16384,k=1024,n=1, doubling m over an individual projection while leaving the activation quantization input size unchanged. The n=2 and down-projection paths already use MMVQ. This justified testing the existing kernel before writing a new shader.

Two GPU runs per mode, in default/integer/integer/default order, each with twenty warmups and one hundred timed iterations:

| Dispatch | Packed n=1, microseconds per 12 FFNs | Packed n=2, microseconds per 12 FFNs |
|---|---:|---:|
| Default, run A | 2478.581 | 2528.484 |
| Force integer, run A | 2449.119 | 2655.940 |
| Force integer, run B | 2433.316 | 2505.575 |
| Default, run B | 2457.429 | 2567.263 |

The n=1 means are 2468.005 versus 2441.217 us, approximately 1.1% apart. Extrapolating only that difference across fourteen single-token depth steps suggests **0.375 ms/frame**, not the remaining roughly 9.5 ms/frame. This is a projection from an isolated graph, not a promised product gain. The n=2 kernel selection is unchanged and its measured variation is larger than the n=1 difference.

## Cost attribution

Each twelve-layer FFN pass has 169,869,312 logical Q4_K weight bytes. Dividing by the default n=1 graph time implies about **68.8 GB/s** of weight throughput; n=2 doubles activation columns while graph time grows only about 3.2%. Together with the marginal integer-route result, this is consistent with weight traffic being a major constraint. It does not prove a DRAM bottleneck: no bandwidth/occupancy counters were collected, and the graph includes activation operations, dispatch and synchronization. Synthetic inputs, fixed variant order within each process, explicit intermediate buffers, and shared-host timing limit extrapolation.

The backend already caches activation Q8_1 conversions by input identity/pipeline/padded size. Gate/up and down consume different activations, so they cannot reuse one conversion. No new activation-cache optimization was identified. Previous larger-workgroup and integer-multirow failures were not rerun unchanged.

## Full-generation check

The prototype changes only AMD Q4_K dispatch at m=16384,k=1024,n=1, gated by a temporary environment variable. The microbenchmark's global FORCE switch was **not** used for full generation. Product source `48f4d08`, ggml probe `3c4b0270`; [archived patch](../../benchmarks/depth-corpus/packed-q4-dispatch.patch).

Same Q4_K model, report prompt, seed 42, 40/40 chunks, packing/graphics queue/trim/matmul enabled, stateful/fused-depth disabled. Three resident generations per mode; first is warmup, all are recorded.

| Mode | Warmed mean wall RTF | Depth ms/frame | Audio | Frames | Generation wall |
|---|---:|---:|---:|---:|---:|
| Original dispatch | 0.919275 | 50.013 | 17.04 s | 213 | 15.664 s |
| Shape-specific integer | 0.914753 | 49.645 | 18.80 s | 235 | 17.197 s |

Each mode repeats its own WAV hash, but the hashes differ between modes. Duration increases by 1.76 seconds, so the lower RTF does not demonstrate a quality-preserving improvement; absolute generation wall time also increases. The change is rejected before broader voice/listening or final-workload tests. This result does not claim the changed speech is perceptually bad; it fails the unchanged-output requirement and offers little speed incentive to open a separate quality-changing candidate.

The probe pin was reverted in `a58962a`. Original ggml source and the dev-pod runtime were restored; production was not changed. The benchmark extension remains.

## Reproduce and next decision

```sh
cmake -S . -B build -DBREEZE_BUILD_FFN_BENCH=ON
cmake --build build --target depth-ffn-bench
GGML_VK_ALLOW_GRAPHICS_QUEUE=1 build/depth-ffn-bench \
  --backend gpu --model /path/to/breeze-tts-2-q4_k.gguf \
  --layers 12 --warmup 20 --iterations 100 --json /tmp/packed-ffn.json
python3 benchmarks/verify_packed_ffn.py
```

[Receipt](../../benchmarks/depth-corpus/packed-ffn-probe.json) stores the microbenchmark commands, results, logs and equivalence errors, model/source identities, all six speech logs and hashes, and the probe backend-library hash. Hardware is queenbee's Radeon 780M/RADV; production logs were checked and showed no concurrent generation during timed tests, but other host activity and GPU clocks were not continuously measured.

This closes the small dispatch experiment. The next bounded runtime opportunity is the previously identified **F32 strided V-cache backend path**, preserving current reduction order and contiguous prefill. It is a separate implementation, not a repeat of the rejected flash-attention or layout-only changes. Its benefit must be measured at actual contexts; no remaining candidate here establishes an RTF-0.8 path by itself.
