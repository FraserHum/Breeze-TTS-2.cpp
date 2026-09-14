# Depth Q3_K: broader 780M runtime and speech matrix

2026-09-06. Current CLI MD5 `067a25764c424185bd684f60354852e0`; source workspace at `f71ed37`. No runtime code or defaults changed in this experiment. The candidate is mixed depth Q3_K derived from the BF16 master, not unchanged Q4_K. See the [quantization pilot](depth-quant-pilot.md) for conversion and tensor provenance.

## Decision

**RTF 0.8 is not achieved.** Q3_K improves every matched workload, but all warmed sample p50/p95 values remain above 0.8. At 60 seconds it still accumulates a delivery deficit. Speech probes show no additional ASR disagreement on this small set, but do not establish general quality acceptance. Keep Q3_K experimental; do not default it on or start a custom rotated format based on these results.

The existing saved-voice path is immediately useful: it removes per-request reference encoding/prefix construction while producing byte-identical speech. This functionality already exists; the old KV-prefix research is not an unimplemented optimization list.

## Runtime receipts

Each case has four resident repeats, discarding the first for timing summaries. One GPU consumer at a time; model order alternates across the original fixtures. The held-out and cached additions run Q4_K then Q3_K. Seed 42, CFG default 1.0, graphics queue enabled, convolution matmul and trim enabled, stateful vocoder and fused depth disabled. Models and CLI were hash-checked before each batch. Raw commands, logs, per-run WAV hashes, manifests and runner source are archived in [quant-matrix.json](../../benchmarks/depth-corpus/quant-matrix.json).

RTF is complete generation wall time divided by emitted audio duration. Model loading and saved-voice preparation are outside it. Sample p95 uses inclusive interpolation over only three warmed runs: descriptive, not a population tail guarantee. First audio below is warmed mean; deficit is warmed maximum.

| Workload | Q4_K p50 / p95 | Q3_K p50 / p95 | Q3 first audio ms | Q3 deficit ms | Q3 remaining ms/frame to 0.8, mean |
|---|---:|---:|---:|---:|---:|
| mandarin | 0.9327 / 0.9384 | 0.8681 / 0.8737 | 2746 | 0.0 | 5.59 |
| reference | 1.0830 / 1.0835 | 0.9877 / 0.9922 | 4427 | 0.0 | 15.07 |
| stress-125 | 1.1645 / 1.1678 | 1.0688 / 1.0808 | 4161 | 0.0 | 21.66 |
| stress-375 | 1.1024 / 1.1056 | 1.0123 / 1.0153 | 4161 | 0.0 | 16.99 |
| stress-750 | 1.1348 / 1.1369 | 1.0446 / 1.0483 | 4151 | 1539.3 | 19.58 |
| heldout-en | 0.9246 / 0.9352 | 0.8474 / 0.8504 | 2665 | 0.0 | 3.77 |
| cached-reference | 0.9887 / 0.9897 | 0.8924 / 0.8938 | 2773 | 0.0 | 7.35 |

All 14 cases have byte-identical output across their four repeats (56 WAVs). All natural cases terminate below the 400-frame cap. Stress cases reach their exact 125/375/750-frame caps. Q3_K first audio is faster than its matched Q4_K control in every case.

The stress input is 3,373 characters with `--split-chars 10000`, deliberately bypassing the production split of 600. These are truncated throughput tests, not complete long-form speech quality or a measurement of the default production splitter. Do not silently discard these failed gates; add a separate complete default-split workload before deciding which context optimization a streaming product needs.

Delivery deficit is computed from flush readiness after playback begins and audio already available. It is not a measured audio-device underrun. Q3_K has zero warmed deficit through the 30-second stress case despite RTF slightly above 1 because first-audio startup buffers the initial chunk; the 60-second case exhausts that cushion.

## Where the remaining time goes

| Q3_K workload | Backbone ms/frame | Depth ms/frame | Vocoder ms/frame |
|---|---:|---:|---:|
| heldout-en | 13.82 | 45.20 | 8.06 |
| mandarin | 14.67 | 45.28 | 8.83 |
| cached-reference | 17.85 | 44.17 | 8.77 |
| stress-750 | 29.00 | 44.11 | 8.71 |

Depth time stays roughly 44–45 ms/frame. Backbone rises from about 14 ms/frame on short text to 29 on long unsplit input. This is stage attribution, not proof of a particular shader bottleneck. Current `attention()` in `src/common.cpp` materializes the attention scores and a contiguous transposed V view on each call; profile these context-dependent operations before choosing an implementation. Host graph reuse remains parked: its earlier measured ceiling was small and is not evidence for this context-dependent cost.

The short held-out gap is 3.77 ms/frame, Mandarin 5.59, cached reference 7.35, and long unsplit stress 19.58. The earlier short-prompt 3.43 ms estimate therefore cannot justify treating the whole goal as a small sampler finishing task. Previously measured SwiGLU/head savings and sampler ceilings cannot be added as a guaranteed combined result.

## Existing saved-voice cache

Using an isolated saved voice reduced Q3_K warm mean RTF **0.98840 → 0.89187** and mean first audio **4427 → 2773 ms**. Q4_K improved **1.08090 → 0.98404**. Cached/uncached WAV SHA-256 matches exactly for each model. Saved voice creation used the Q4_K control; codec tensors are unchanged between controls.

`apps/cli/main.cpp` calls `build_voice_prefix()` before the repeated generation loop. `src/generation.cpp` restores that prefix for a matching saved voice and CFG 1.0. Codec encoding happens when saving the voice. Both costs are shifted outside the request, not eliminated from cold preparation. Use this existing route for persistent voices; no new cache implementation or claim about cold process latency is warranted. The fixture's reference recording is synthetic Q4_K speech, not a real human speaker.

## Speech evidence

[quant-speech.json](../../benchmarks/depth-corpus/quant-speech.json) stores raw transcripts, recognizer commands, normalized sequences, model hashes and speaker results. Whisper small multilingual runs independently without the expected text as a prompt. English uses lowercase NFKC alphanumeric words; Mandarin uses OpenCC traditional-to-simplified normalization and alphanumeric characters.

| Natural fixture | Q4_K ASR disagreement | Q3_K ASR disagreement | Interpretation |
|---|---:|---:|---|
| Mandarin | CER 1/75 | CER 1/75 | Recognizer substitutes one character in 品質 for both. |
| Reference English | WER 4/44 | WER 4/44 | `TTS two`/`TTS2` and `780 M`/`780M` tokenization account for the disagreements. |
| Held-out English | WER 1/29 | WER 0/29 | Q4 recognition spells the name `Mirra`, expected `Mira`. |

Cached reference scores duplicate identical audio, so they are not additional independent quality evidence. Recognizer disagreement is not human transcription error, and these results do not prove all speech is preserved. Held-out English duration changes from 13.04 s to 9.52 s while ASR recovers the text, demonstrating a pace/prosody change that text accuracy cannot assess.

WeSpeaker English VoxCeleb ResNet34 cosine to the synthetic reference is **0.98050 Q4_K**, **0.97533 Q3_K**; Q3/Q4 cosine is **0.96821**. Self-cosine is 1.0. These are descriptive scores with no task-specific acceptance threshold, not a speaker-quality pass.

Cached UTMOS scored each unique output once. Clips up to 10 seconds are scored whole; longer clips use contiguous 10-second windows and the unweighted mean of window scores. Window boundaries and short final windows differ between outputs of different length. The held-out scores (Q4 4.481, Q3 4.270) therefore deserve listening follow-up, not a declared quality regression or pass. Mandarin proxy scores (2.843/3.332) likewise do not establish multilingual acceptance. Full window scores and the exact checkpoint hash are in the runtime receipt. Cross-take waveform SNR/PESQ is inappropriate because quantization changes sampling trajectories and durations.

**Quality verdict: inconclusive.** Add independent human references, more held-out utterances/seeds and listening/prosody checks before deployment. Capped stress has no complete-utterance intelligibility verdict. No post-hoc score threshold is introduced.

## Updated implementation order

1. **Measure the actual long-form product route.** Keep the existing saved-voice path for the reference comparison; run complete default-split long English/Mandarin with Q4/Q3, recording per-part boundaries, wall RTF, first audio and delivery deficit. This is the highest-value next measurement: hours, no new engine code, and prevents optimizing only an artificial unsplit workload.
2. **Investigate the measured backbone context cost.** On the resulting real workload, isolate attention/V-layout/KV work from FFN and submission. Try one existing ggml attention/layout improvement only if attribution supports it; require correct masks/positions/full context and measured end-to-end benefit. Initial investigation about a day; savings unproven. This moves ahead of small head/sampler work because the observed context-associated increase is about 15 ms/frame, not because all of it is removable.
3. **Expand handoff 00 quality/calibration coverage for standard Q3_K.** Reuse the BF16 master and current probes; add disjoint utterances, human references, later frames/layers/codebooks and multiple seeds. Follow with exact-format calibrated Q3_K rounding or selective higher precision only when held-out quality identifies the need. Several days; no custom backend required. Corpus work can overlap the CPU analysis above, with serialized GPU collection. Q3_K remains experimental until acceptance is established.
4. **Use exact sampling/readback plus the small SwiGLU/head changes only as a measured finishing experiment.** Reopen when the accepted candidate's remaining gap is small enough. Preserve RNG/top-k/top-p/forced-code behavior and remeasure the combined path; earlier isolated savings are not additive promises. The current full matrix gap is too large to put this first.
5. **Handoff 05: stronger rotation/IQ3/custom packing, conditional on standard-format limits.** Show a held-out quality/packed-byte advantage over calibrated ordinary Q3_K first, then benchmark an actual 780M operation including transforms and metadata. The simple scalar rotation pilot did not beat standard Q3_K; neither that result nor these timings justify a new format yet.
6. **Structural student work if bounded engine/quant work remains insufficient.** Start a smaller autoregressive depth student preserving all codebooks and downstream feedback, then handoff 03 parallel/refinement, handoff 04 direct downstream prediction and handoff 06 forecasting as conditional branches. Week-scale; shared evaluation first. Handoffs 01/02 remain parked on their existing failed sparsity/low-rank screens.

Quality and the complete RTF/latency matrix remain gates throughout this order. The 0.8 target is not relaxed for reference or stress cases, and the original unchanged-Q4_K target is not relabelled as achieved by a mixed-format candidate.

## Reproduce and verify

The runtime receipt embeds all three serialized batch manifests and runner source. Local WAVs and scoring logs are in `.beehive/agent/BREEZE-Q3-MATRIX/results`; remote WAVs remain in `/tmp/breeze-q3-{matrix,heldout,cached}` in `breezetts-dev`. WAVs/models are not committed. The scripts download nothing.

```sh
python3 benchmarks/quant_matrix_780m.py --self-test
python3 benchmarks/quant_matrix_780m.py --verify
python3 benchmarks/quant_speech_780m.py --verify benchmarks/depth-corpus/quant-speech.json
```

Live rescoring requires the explicit local dependencies/model paths in the receipt. ASR uses whisper.cpp commit `52a939a2a762224e255d366c1182b2af4dd1a032` and multilingual small SHA-256 `1be3a9b2063867b937e64e2ec7483364a79917e157fa98c5d94b5c1fffea987b`; speaker weights SHA-256 `5ef208a9da1453335308a6b6f4e6dfbd7e183a38b604de0a57664f45d257fe94`. Tools run locally on Apple Silicon/CPU after GPU measurement, in isolated environments. The analysis-stage sum excludes the first-vocoder diagnostic because total vocoder already includes it.

Tool references: [official whisper.cpp](https://github.com/ggml-org/whisper.cpp), [official sherpa speaker documentation](https://k2-fsa.github.io/sherpa/onnx/speaker-identification/index.html), [official speaker example](https://github.com/k2-fsa/sherpa-onnx/blob/master/python-api-examples/speaker-identification.py).
