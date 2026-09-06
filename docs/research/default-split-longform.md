# Default-split long-form speech and backbone context cost

2026-09-06. This extends the [broader Q3_K matrix](depth-quant-matrix.md) using the actual product splitter. **RTF 0.8 remains unachieved.** No production runtime or default changes are retained. A diagnostic benchmark tool and a configurable speaker-fixture argument are the only executable additions.

## Complete-input timing matrix

English uses 885 weighted characters, Mandarin 783, and saved-voice English 858. The compiled current splitter produces 3, 3 and 2 parts respectively, with exact concatenation to the original input. The default budget is 600, with an opening budget of 200 when there is no supplied reference; CJK characters weigh three. No `--split-chars` override is used. Saved-voice English has a different text from unconditioned English, so their timing difference is not an isolated reference-overhead comparison.

Same verified Q4/Q3 controls, seed 42 and CLI MD5 `067a25764c424185bd684f60354852e0` as the previous matrix. Graphics queue, convolution matmul and trim enabled; stateful vocoder and fused depth disabled. Four resident repeats per case; discard the first for timing. All 24 WAV receipts are byte-identical within their case. Total frames stay below 1200 for every run, which also rules out any individual part reaching that cap. This does not prove every intended word was spoken. Runtime per-part completion/timing markers are unavailable and are recorded as missing, not inferred from static splitting.

| Workload | Q4 mean RTF | Q3 mean RTF | Q3 sample p50 / p95 | Q3 mean first audio ms | Q3 remaining ms/frame to 0.8 |
|---|---:|---:|---:|---:|---:|
| english | 0.97695 | 0.89722 | 0.89644 / 0.89909 | 2627 | 7.78 |
| mandarin | 0.99448 | 0.89911 | 0.89805 / 0.90096 | 2660 | 7.93 |
| saved-voice-en | 1.00454 | 0.91087 | 0.90939 / 0.91408 | 2923 | 8.87 |

Q3_K has zero warmed delivery deficit in all three cases; Q4_K reaches 180 ms in Mandarin and 536.9 ms in saved-voice English. This is the flush-readiness deficit after initial playback buffering, not an audio-device measurement. Sample p95 uses inclusive interpolation across three warmed observations, not a population guarantee. Saved-voice preparation and model loading are excluded from request wall timing.

Q3 backbone time is 17.97/18.25/19.74 ms/frame for English/Mandarin/saved voice, while depth stays about 43.8–43.9 and vocoder about 8.7. Default splitting avoids much of the prior unsplit backbone cost of 29 ms/frame, but still leaves 7.8–8.9 ms/frame to recover. Keep the earlier failed unsplit stress separately; this new measurement does not retroactively make it pass.

[Runtime receipt](../../benchmarks/depth-corpus/quant-longform.json) contains commands, logs, timing samples, hashes, splitter evidence and limitations.

## Speech checks and Mandarin blocker

Whisper small transcribes English with 2/142 disagreements for Q4 and 5/142 for Q3. Two are British/American spelling differences in both; Q3 additionally has `mend/men`, `finish/finished` and `ask/asked`. Saved-voice English has 4/136 for both, all spelling variants. These are recognizer disagreements, not human-annotated errors.

Long Mandarin triggers a substantial warning on both models:

| ASR diagnostic | Q4 character disagreements / 233 | Q3 character disagreements / 233 |
|---|---:|---:|
| Whisper small, whole clip | 198 | 128 |
| Whisper small, independent 10-second cuts | 159 | 136 |
| Whisper medium, whole clip | 158 | 103 |

The larger recognizer also produces unrelated/missing middle content; Q4 transcription includes Cantonese wording. Repeated small-model decoding of short cuts does not remove the warning. This is strong reason to block deployment acceptance of this fixture, but it does not establish the precise root cause or replace listening. Both model sizes belong to the Whisper family; do not treat them as independent human judgments. Do not attribute the baseline problem solely to Q3 quantization, because Q4 also exhibits it.

### Explicit language instruction control

Changing only `Speak clearly and naturally.` to `Speak clearly and naturally in Mandarin Chinese.` improves Q4's Whisper medium disagreement from **158/233 to 34/233**, but Q3 changes from **103/233 to 110/233**. Text, seed, split policy and runtime flags remain fixed. These are two single cold runs for quality diagnosis, not a warmed throughput comparison; [raw instruction receipt](../../benchmarks/depth-corpus/quant-mandarin-instruction.json) links their output hashes to the ASR diagnostic.

The instruction helps Q4 on this fixture but does not rescue Q3. An underspecified language instruction is therefore insufficient to explain away Q3's warning. This raises language-aware calibration and targeted precision above speculative custom formats. It does not establish a universal quantization failure from one seed, and even Q4's improved proxy score is not a declared quality pass. No listening has been performed.

Saved-voice cosine to the synthetic reference is 0.95872 for Q4 and 0.93294 for Q3, with Q3/Q4 cosine 0.96480. No speaker acceptance threshold is established. UTMOS was deliberately not rerun: it would not resolve the intelligibility warning. No cross-take waveform/PESQ gate applies to changed trajectories.

[Speech receipt](../../benchmarks/depth-corpus/quant-longform-speech.json) and [Mandarin ASR diagnostic](../../benchmarks/depth-corpus/quant-longform-asr-diagnostic.json) retain transcripts, normalization, model hashes, commands and the segment runner. The medium checkpoint was verified against the SHA-1 in the pinned whisper.cpp source; its SHA-256 is `6c14d5adee5f86394037b4e4e8b59f1673b6cee10e3cf0b11bbdbee79c156208`. No expected transcript was supplied as a recognition prompt, and no post-hoc score threshold is used as a quality pass.

## Fixed-context backbone probe

`backbone-context-bench` calls the existing `backbone_run` with real model weights. It initializes a deterministic synthetic prefix once, rewinds only the current position before each identical single-token query, and verifies output shapes, position, finite values and exact repeated hidden/logit equality. It does not generate speech. Three process rounds use five warmups and 30 timed calls per context, with context order reversed/rotated across rounds.

| Total context tokens | Mean full host backbone call, ms |
|---:|---:|
| 64 | 12.759 |
| 256 | 14.986 |
| 512 | 17.990 |
| 1024 | 28.049 |

All repeated-output checks pass. The 512-token value is close to the observed default-split backbone cost; 1024 tokens is a stress point, not the mean product context. These are full host call timings including graph work and readback, not generation RTF or projected savings.

Separate logger runs attribute the final single-query graph as follows. They are perturbed profiling measurements, not substitutes for the unprofiled timings above:

| Operation group across 28 layers | 64 tokens, ms | 1024 tokens, ms |
|---|---:|---:|
| CONT, 57 operations | 0.213 | 6.077 |
| Q×K, F32, head dimension 128, 16 query heads | 0.487 | 7.343 |
| V×probabilities, F32 | 0.222 | 1.599 |
| Softmax | 0.118 | 0.295 |

Current attention builds QK scores, softmax, a contiguous transposed V tensor, and the probability/value multiplication. CONT is aggregated: it also includes the fixed-size attention output and final-state copies; it is not a separate direct timer of V alone. The source and changing tensor sizes identify V layout as the principal context-growing copy to investigate. Do not add these logger groups to another end-to-end saving estimate.

The next engine experiment should use the existing `ggml_flash_attn_ext` at the **backbone call site only**. The vendored Vulkan implementation supports the current F32 Q/K/V and strides; it requires a contiguous F16 mask. Current causal masks use exactly representable zero/negative infinity. Request F32 accumulation explicitly, but do not equate that with byte-identical arithmetic: reduction order and internal kernel operands still require validation. Preserve full context, masks, positions and sampling, and leave codec/text/depth attention unchanged. Compare against this baseline and then full speech before retaining an opt-in path. No new shader is justified before that test.

The isolated probe links the pod's existing core and ggml libraries. Pod backbone and splitter sources match the local versions; the common helper differs only in explicit Vulkan backend selection. Its archive references ordinary SiLU, with no leftover temporary SwiGLU/flash helper. The benchmark does not rebuild or replace the existing CLI. [Context receipt](../../benchmarks/depth-corpus/backbone-context.json) records individual samples, profile blocks and provenance.

## Verification

```sh
python3 benchmarks/quant_matrix_780m.py --verify benchmarks/depth-corpus/quant-longform.json
python3 benchmarks/quant_speech_780m.py --verify benchmarks/depth-corpus/quant-longform-speech.json
cmake -S . -B build-context -DBREEZE_BUILD_BB_CONTEXT_BENCH=ON
cmake --build build-context --target backbone-context-bench
build-context/backbone-context-bench --model /path/to/model.gguf --backend cpu --context 4 --warmup 1 --iterations 2
```

The last command is the small numerical regression check, not a CPU/GPU performance comparison. Raw WAVs remain in the local long-form ledger and pod temporary directories; models and WAVs are not committed.
