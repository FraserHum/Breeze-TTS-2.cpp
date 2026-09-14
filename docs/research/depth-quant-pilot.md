# Depth quantization pilot — 2026-09-06

Standard mixed depth Q3_K is the next candidate to validate before implementing
a custom rotated format. Nothing in this experiment changes runtime defaults.
The unchanged Q4_K engine has not reached RTF 0.8.

## What was built and checked

The [control builder](../../tools/build_depth_quant_controls.py) replaces only
the 84 attention/FFN linear weights in the twelve depth blocks, sourced from
the verified original BF16 master. All 1,077 other tensors are byte-identical;
metadata and shapes are preserved. The intermediate F16 conversion has a
maximum absolute error of 2.9802322387695312e-8 across those matrices, with no
overflow. These are mixed models, not a full BF16 reference.

The existing quantizer produces ordinary, **uncalibrated** Q4_K and Q3_K
controls from that intermediate. The selected payload falls from 187,564,032
to 143,278,080 bytes (23.61%); total file size falls only from 2,538,909,376 to
2,494,623,424 bytes. Depth weights matter disproportionately because they are
read repeatedly. The rebuilt Q4_K payloads differ from the original model,
so both controls are measured. [Complete conversion receipt](../../benchmarks/depth-corpus/quant-controls.json).

## 780M generation smoke result

Four resident repeats per case, seed 42, exact English report prompt, first
repeat excluded from timing aggregates. The dev CLI is unchanged (MD5
`067a25764c424185bd684f60354852e0`). Graphics queue and conv-transpose matmul
are enabled; stateful vocoder and fused depth remain disabled, and exact
convolution-history trim remains at its enabled default.

| Model | Frames / audio | Warm mean wall RTF | UTMOS | Four WAVs identical within case |
|---|---:|---:|---:|---|
| Current Q4_K | 213 / 17.04 s | 0.932283 | 4.5514 | Yes |
| Rebuilt depth-block Q4_K | 195 / 15.60 s | 0.938885 | 4.5000 | Yes |
| Mixed depth-block Q3_K | 245 / 19.60 s | **0.842859** | 4.5381 | Yes |

Q3_K saves about 7.68 ms/frame relative to the rebuilt Q4_K control and leaves
3.43 ms/frame to the 64 ms/frame target on this prompt. Different frame counts
and sampled trajectories prevent treating this as a matched-shape kernel
comparison. UTMOS is a single-prompt quality proxy, not speech acceptance;
cross-take waveform/PESQ comparisons are deliberately omitted. First audio
for the three warmed Q3_K runs was 2593/2588/2615 ms. There is no evidence yet
that longer, Mandarin or reference-voice workloads meet the target.

[Raw logs, twelve runs, WAV hashes, model hashes, stage times and setup](../../benchmarks/depth-corpus/quant-runtime.json)
are preserved. Three warmed samples are too few to establish operational p95;
the receipt's sample p95 is descriptive only. Our GPU jobs and builds were
serialized; unrelated host activity was not continuously profiled. The rebuilt
controls terminated below their 400-frame cap; the original baseline used its
default cap and also terminated below 400.

To repeat, use the receipt's exact prompt/instruction and environment with
`breeze-cli MODEL --seed 42 --repeat 4 --max-new 400 --timings --output OUT.wav`.
UTMOS uses the cached SpeechMOS v1.2.0 `utmos22_strong` checkpoint whose SHA is
in the receipt. Models remain in the local cache; Q3_K/Q4_K controls are also
in the dev pod's `/tmp`, with full-file hashes checked before measurement.

## Conditional output-error screen

This uses the existing Q4_K-generated activation pilot: 96 calibration rows
and 96 disjoint held-out rows per selected layer, English/Mandarin, frames
0/7/15, all fifteen depth steps. Reference outputs use the original BF16 down
weights on those captured inputs. These are **not BF16-generated activations**,
full-FFN errors, or speech-quality measurements.

| Down-projection layer | Existing Q4_K | Rebuilt Q4_K | Standard Q3_K | Ordinary scalar 3-bit | Rotated scalar 3-bit |
|---|---:|---:|---:|---:|---:|
| 0 | 0.001088 | 0.001088 | 0.004979 | 0.013906 | 0.010076 |
| 5 | 0.003854 | 0.003858 | 0.016754 | 0.039569 | 0.033848 |
| 11 | 0.001035 | 0.001036 | 0.004552 | 0.015627 | 0.009808 |

Numbers are held-out relative output MSE. The scalar emulations use seven
symmetric levels, group size 128, F32 scales, and calibration-only clipping
selection. Their nominal payload is 3.25 bits/weight including scales, versus
Q3_K's 3.4375. They are not ggml Q3_K and have no packed runtime implementation.

The fixed randomized block Hadamard applies the same orthogonal right
transform to weights and inputs, after SwiGLU for this down projection. The
unquantized identity passed (maximum relative MSE 4.89e-12). Rotation reduces
error 14–37% against this scalar baseline, but standard Q3_K has roughly half
its error. This does not rule out full-width or learned rotations, better
rounding, or vector codebooks; it does not justify their runtime work yet.
[Screen and provenance](../../benchmarks/depth-corpus/quant-screen.json).

## Next order

1. Validate the standard depth Q3_K candidate across English, Mandarin,
   reference voices and 10/30/60-second workloads, with held-out intelligibility,
   speaker similarity, termination, repeatability, quality and resident RTF.
2. If quality passes, measure the remaining engine gap. Reconsider exact
   sampling/readback and the small native SwiGLU/head gains as a combined
   finishing experiment if warranted by that gap.
3. If quality needs recovery, expand calibration across all twelve layers,
   later frames and both CFG branches, using BF16-reference controls and
   disjoint held-out examples. Constrain calibrated rounding/error feedback to
   the exact Q3_K representation; identify sensitive layers before applying
   mixed precision. Do not call the present ordinary quantizer calibrated.
4. Investigate IQ3 and stronger rotations only for a demonstrated quality or
   speed shortfall. Custom packed kernels must pay for transformations and
   metadata traffic. Student distillation remains after these bounded tests.

The RTX 5090 is reachable, but the read-only check found 31,660/32,607 MiB
occupied. This pilot ran locally; no serving process was displaced. Capture
expansion and broad speech acceptance remain outstanding.

Reproduce the numerical screen with
`/tmp/breeze-quality-venv/bin/python tools/depth_quant_screen.py` and build or
verify the cached controls with
`/tmp/breeze-quality-venv/bin/python tools/build_depth_quant_controls.py`.
Both scripts have `--self-test`. The control builder requires the freshly
built `build-ffn-bench/breeze-quantize` target and the pinned source cache.
