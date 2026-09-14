# Q3 wave32 and computed vocoder history — 2026-09-15

Keep the computed convolution history; reject the wave32 specialization.
Production remains unchanged. No additional audio approval is required for the
tested trim outputs because they match the previously approved Q3 audio exactly.

## Wave32: no useful gain

Opt-in root `51b3e1d`, ggml `c0f98773`. Only RDNA3 floating Q3/F32 matvec
subgroup/workgroup sizes change; rows remain two and the integer path remains
unchanged. Live properties previously confirmed supported subgroup sizes 32–64.
The shader processes 16 threads per Q3 block; the changed 32/128-thread groups
remain compatible with that indexing. Enabled-run logs confirm the probe was
selected. No ISA or hardware-counter result is claimed.

Existing twelve-layer real-weight FFN benchmark, 20 warmups and 100 iterations
per variant, wave64/wave32/wave32/wave64 order:

| Run | Packed n=1 us | Packed n=2 us |
|---|---:|---:|
| Wave64 A | 1927.818 | 1880.154 |
| Wave32 A | 1935.695 | 1870.425 |
| Wave32 B | 1955.573 | 1943.041 |
| Wave64 B | 1813.146 | 1873.430 |
| Wave64 mean | 1870.482 | 1876.792 |
| Wave32 mean | 1945.634 | 1906.733 |

The main n=1 case is 4.02% slower. The n=2 route is a useful noise control,
not evidence that all matvec columns exercise the changed floating pipeline.
Normal/packed equivalence and repeat checks pass within each mode. The n=1
checksum changes between modes; no numerical or audio parity is claimed across
modes. No full speech trial was warranted after this negative microbenchmark.
This rejects the tested specialization, not every possible wave32 kernel design.

The backend pin returns to `acc4d4d7` in `33bf2d0`. Experimental patch is
archived at `benchmarks/depth-corpus/q3-wave32-probe.patch`; it is not retained
runtime code. Source hashes matched locally and in the pod, and the restored
library contains no wave32 probe message. Before full-speech timing, the saved
pre-probe Vulkan library was restored byte-for-byte for both executable paths.

## Computed history: retain the small change

Root `33bf2d0` changes `min(trim_prefix,16)` to
`min(trim_prefix,history)`, reusing the existing guarded receptive-field
calculation. For the approved model this retains 11 frames. Transformer context,
generation chunking and sampling are unchanged. The caller already calculates
the emitted-audio offset from the decoded length.

The existing `tools/voc_trim_test.cpp` now tests starts 0/11/16/88/89/104 and
counts 1/4/40 against full-window decoding, using the same moving context as
generation. All 18 cases have equal requested lengths, finite samples,
correlation 1 and max difference 0. The fixture is the reviewed 12.08-second
Calliope tagged waveform, encoded once to fixed codes for both paths.

Full ordinary Calliope runs use baseline/candidate/candidate/baseline order,
three repeats per process with the first excluded. Same approved Q3 model,
reference, instruction, seed and restored Vulkan library:

| Run | Warm wall RTF | Vocoder ms/frame | Depth ms/frame |
|---|---:|---:|---:|
| Baseline A | 0.868755 | 8.115 | 44.159 |
| History 11 A | 0.845866 | 7.570 | 42.946 |
| History 11 B | 0.831277 | 7.542 | 41.818 |
| Baseline B | 0.834693 | 8.263 | 41.490 |

Pooled vocoder time falls from **8.189 to 7.556 ms/frame**, a **0.634 ms/frame**
or 7.7% stage reduction. Both candidate blocks are below both controls.
Pooled wall RTF is 0.851724 versus 0.838572, but unchanged depth timing drifts
substantially during the run: do not credit the whole wall improvement to trim
or interpret this as attainment of 0.8. The small consistent stage saving is
appropriate for a one-line reuse of an existing calculation.

All twelve ordinary WAVs match the approved Q3 SHA. Two CFG=2 delivery-tag
runs also match the reviewed sigh/laugh WAV exactly. The latter is a two-branch
quality check, not a CFG=1 speed comparison. Production logs showed no new
requests in the checked benchmark window; other shared-host variability was
not continuously measured.

## Reproduction

`benchmarks/depth-corpus/wave-trim-probe.json` retains commands, benchmark
outputs, speech timings/hashes and boundary output. Verify the receipt with
`python3 benchmarks/verify_wave_trim_probe.py`.

The boundary tool is built against the current core, then run as follows in
the dev pod (the reviewed WAV needs at least 144 encoded frames):

```sh
c++ -O3 -std=c++17 -I/src/include -I/src/third_party/ggml/include \
  /src/tools/voc_trim_test.cpp /src/build/libbreeze_core.a \
  -L/src/build/third_party/ggml/src -L/src/build/third_party/ggml/src/ggml-vulkan \
  -Wl,-rpath,/src/build/third_party/ggml/src:/src/build/third_party/ggml/src/ggml-vulkan \
  -lggml -lggml-cpu -lggml-vulkan -lggml-base -pthread -o /tmp/voc-trim-test
GGML_VK_ALLOW_GRAPHICS_QUEUE=1 BREEZE_PACK_WEIGHTS=1 \
  BREEZE_VOC_CONVT_MATMUL=1 BREEZE_VOC_STATEFUL=0 \
  /tmp/voc-trim-test /tmp/breeze-current-ffn-q3-calibrated.gguf \
  /tmp/refresh-async-tags.wav
```

Next investigate exposed command-recording cost before implementing replay;
wave-size tuning has not produced a gain. Attention remains the next separate
kernel family to examine if recording overhead is too small.
