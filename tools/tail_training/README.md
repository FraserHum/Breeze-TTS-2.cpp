# Experimental tail preparation

This is an exact 17,704,960-parameter PyTorch student matching the four-block ggml cost probe. It is not a trained speech model. The production prefix remains frozen; targets come from the user-selected F16 teacher.

`student.py check CAPTURE_DIR --export UPDATED_DIR --threads 4` runs on CPU by default. It checks all saved ggml references, verifies finite gradients and a nonzero SGD update, then deliberately doubles the output head to make stale exports detectable. The exported diagnostic weights are **not for training initialization, listening, or deployment**. Use original capture weights for initialization; the synthetic sinusoidal head used for numerical tests should be replaced by seeded random initialization before speech fitting, with a fresh parity check.

`prepare_manifest.py CAPTURE_ROOT EXAMPLES_JSON RESULTS_DIR` checks frame alignment, finite input/target arrays, code lengths, split labels, and exact production PCM replay. It excludes `length-*` parity probes from the dataset. Run checks without Python's `-O` flag.

The standalone C++ `../tail_capture.cpp` accepts:

```
tail-capture MODEL CODES OUTDIR production
tail-capture F16_MODEL CODES OUTDIR teacher
tail-capture MODEL CODES OUTDIR roundtrip UPDATED_DIR
```

Build against the repository's `breeze_core` library, as with `teacher_compare.cpp`. Use production settings `BREEZE_VOC_TRIM=1 BREEZE_VOC_CONVT_MATMUL=1 BREEZE_VOC_STATEFUL=0`; `BREEZE_PACK_WEIGHTS=0` avoids unused generator packing. The capture tool operates on one generated part per code file. It saves native little-endian F32 arrays (validated on x86 hosts), a raw tensor manifest, and per-window data manifests. It invokes shared codec functions rather than reconstructing the teacher in PyTorch.

Input arrays are frame-major `[T,1024]`; output arrays `[N,1920]`. Weight shapes in `weights.json` are ggml innermost-first; reverse them when loading into PyTorch. Capture `production` before `teacher`; production mode saves inputs, original emitted samples, initialization weights and synthetic student references. Teacher mode writes only F16 targets. Roundtrip mode requires existing captured inputs and writes changed-weight ggml outputs; it does not regenerate targets.

Validated artifacts and instructions: [preparation results](../../docs/research/student-preparation-results.md). The final production GGUF exporter remains future work; the experimental trainer below now supplies losses and checkpoints.

## Tiny fitting

`train.py --self-check` checks the reconstruction losses. `train.py --weights CAPTURE/parity --out INITIALIZED --initialize-only` replaces the diagnostic head with seeded random weights and emits parity references. Verify those against ggml before running `train.py --weights INITIALIZED --dataset CAPTURE/dataset.json --out NEW_RUN --steps 2000 --minutes 30 --batch 4 --device cuda`. ROCm uses the same `cuda` device spelling in PyTorch. Each run requires a new output directory. The update loop respects the time cap; final evaluation and safe checkpoint writes add overhead. Checkpoints include optimizer/RNG state, but a resume CLI is not implemented yet.

The [first sunset run](../../docs/research/tiny-fit-sunset-results.md) completed; the user rejected its robotic, barely intelligible speech. `--wave-weight` changes the waveform L1 coefficient (default 1); compare fixed-step final checkpoints when comparing loss weights.

`analyze_audio.py TEACHER.wav STUDENT.wav` reports amplitude, correlation, bounded lag and frame/chunk boundary diagnostics. Run `--self-check` for identity, signed delay and injected seam checks. These metrics do not establish listening quality.

`train.py --objective mse` trains all student parameters with waveform MSE only; spectral metrics remain diagnostic. Best-checkpoint selection uses validation MSE in this mode. Use the fixed final checkpoint for controlled comparisons. The default `spectral` objective retains the previous waveform-L1 plus spectral losses.
