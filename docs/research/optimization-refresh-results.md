# Optimization refresh results — September 14, 2026

The bounded pass is complete. Existing weight packing has a measured benefit; no additional runtime change is retained. RTF 0.8 remains unmet. These are calibration-prompt results, not the final five-workload acceptance matrix.

## Controlled packing comparison

Same binary (`431745a896cef38c88ff2b4104ed908c` MD5), source `1abeb7d`, unpacked Q4_K model, English report prompt, seed 42, 40/40 chunks, graphics queue, matmul and trim enabled; stateful vocoder and fused depth disabled. Run order was off/on/on/off, three resident repeats per block, with the first repeat explicitly excluded as warmup. All twelve WAVs have SHA-256 `67984c45dd90ffe4f6bba7c34b6c39401a793706f4dedbe9e0897705c1dddb3e`.

| Packing | Warmed wall RTF, block means | Combined warmed mean |
|---|---|---:|
| Off | 0.934137 / 0.933876 | 0.934006 |
| On | 0.921001 / 0.920678 | 0.920839 |

The observed saving is **1.053 ms/frame**, approximately 1.4% of wall time, on this prompt. Four warmed samples per setting do not establish population-tail performance. Packing-on still needs about **9.67 ms/frame** to reach 0.8 on this fixture. This comparison does not count the earlier vocoder trim as a new packing gain.

## Packed graph inspection

An eight-frame diagnostic using the existing Vulkan logger confirms actual `RMS_NORM_MUL` and `MUL_MAT_ADD` fusion, native GLU, and packed Q4_K gate/up shape `m=16384,k=1024` at n=1 and n=2. There is no demonstrated globally disabled native fusion to turn on.

Packed depth gate/up and down projections account for approximately **34.42 ms/frame in this instrumented run**. The logger perturbs execution (depth total 53.50 ms/frame here), so this attribution is not an available saving or an unprofiled latency result. It does identify the next substantial target: the existing packed Q4_K FFN execution, rather than a presumed missing normalization fusion. Raw logger output is included in the receipt.

## Remaining codec SwiGLU trial

Commit `b1d13b8` replaces separate SiLU/multiply with existing `ggml_swiglu_split` in the unpacked helper. With packing enabled this principally tests the remaining codec paths. Across the complete pass, all 21 plain-prompt outputs and all nine saved-voice outputs preserve their respective WAV hashes. Every block has three resident repeats and one excluded warmup.

| Workload | Original before | Candidate | Original after |
|---|---:|---:|---:|
| Plain English | 0.920839 | 0.920207 | 0.916452 |
| Saved `test17s` voice | 0.953939 | 0.962194 | 0.956812 |

Values are warmed mean wall RTF. Plain candidate uses two blocks; before uses the two packing-on blocks; the remaining cells use one block each. The nominal plain saving versus the early control is only 0.051 ms/frame, and the later original is faster. Saved voice is slower under the candidate. This does not establish a stable improvement; the trial was reverted in `526c1a3`. No universal 1 ms cutoff was applied: it failed to show a repeatable gain at all.

## Value-cache layout verdict

A graph-only transposed cache would not remove the Vulkan work. A live prefix of a fixed-capacity `[capacity, head_dim, heads]` V buffer has noncontiguous rows. The existing F32 matvec path copies that input into contiguous scratch internally; the prefill path can also convert noncontiguous F32 to F16. Removing the current `CONT` around the original permuted layout instead fails the matmul non-transposed-input requirement.

Therefore no layout-only patch was made. A separate backend experiment would need explicit row/batch stride support in the existing F32 matvec while preserving reduction order, contiguous prefill, and prefix snapshot/restore bytes across differing cache capacities. Simply reusing the F16-only noncontiguous kernel is not an exact F32 solution. Sources: [`ggml_vk_dim01_contiguous`, matvec and matmul paths](../../third_party/ggml/src/ggml-vulkan/ggml-vulkan.cpp), [attention and prefix snapshots](../../src/common.cpp). This is a source-based rejection of the simple proposal, not a measured rejection of every possible backend implementation.

## Reproduction, evidence, and limits

[Receipt](../../benchmarks/depth-corpus/optimization-refresh.json) includes all 30 generations, exact commands, source/binary/model/voice identities, raw timing logs, WAV hashes, warmup policy, and the diagnostic profile. Verify its arithmetic and paired hash evidence with:

```sh
python3 benchmarks/verify_optimization_refresh.py
```

The dev pod was on **queenbee**, sharing the Radeon 780M host with production. Production logs were checked during the timed sequence and showed no new generation after the pre-benchmark request; GPU clocks and other host activity were not continuously measured. No production configuration was changed. The dev pod was restored with `scripts/pod-sync.sh dev` to source `1abeb7d` and the original binary MD5 above.

The saved voice is the existing synthetic `test17s` fixture, not Calliope/Steward or a final held-out voice. Its file hash is recorded. No new ASR/MOS or human-listening claim follows; byte parity is the evidence for these exact fixtures. Longer English and real deployed voices remain part of acceptance before any future runtime promotion.
