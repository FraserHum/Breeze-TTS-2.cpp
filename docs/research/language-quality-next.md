# Language-explicit quality corpus handoff

The ready corpus specification is [language-quality-manifest.json](../../benchmarks/depth-corpus/language-quality-manifest.json). It has eight disjoint text fixtures: English and Mandarin short calibration cases at seed 42, English and Mandarin default-split long calibration cases at seed 123, and matching held-out short/long evaluation cases at seeds 123/42. The unreferenced long cases are three-part inputs under the current 600 weighted-character budget with a 200-character opening budget; the saved-voice long cases use the runtime's 0-character opening budget and split into two parts. The two long evaluation cases use the available `matrix` saved synthetic voice. This file and the manifest describe inputs only; no calibration data or collection result is claimed.

Use the existing `breeze-quant-speech/v1` receipt shape and `benchmarks/quant_speech_780m.py` for the paired Q4_K/Q3_K ASR probe after a small language-field lookup is added: the current runner recognizes Mandarin only when the fixture id is literally `mandarin`, while this manifest keeps explicit disjoint IDs. Emit one sample per model for every fixture, retain both transcripts and WAV hashes, and report the per-fixture disagreement rates plus Q3-minus-Q4 deltas. English uses normalized word edit distance and Mandarin normalized character edit distance. These are recognizer disagreements, not human ground truth or acceptance evidence.

The current depth capture remains opt-in and narrow: `BREEZE_DEPTH_CAPTURE` must name a fresh empty directory; `BREEZE_DD_FUSED=0`; Vulkan capture requires `GGML_VK_DISABLE_FUSION` before backend initialization; only one or two branches are accepted; the 12-layer/15-step model is required; capture considers only the first 16 frames and currently selects frames 0, 7, 15 and layers 0, 5, 11; selected F32 tensors are written to `activations.f32` with `metadata.jsonl` under a 16 MiB validator limit. Compare capture-off and capture-on complete codes and WAV hashes before using any blob. The smallest missing instrumentation is parameterized layer/frame selection plus bounded streaming or file rotation for all 12 layers and late frames (for example 31 and 63), followed by parity and validator checks. No runtime change is part of this handoff.

Proposed order: validate the manifest and splitter; capture and parity-check Q4_K calibration inputs; run capture-off Q4_K/Q3_K pairs for all fixtures with sampling options omitted so each model's embedded GGUF values remain unchanged; run the paired ASR diagnostic; then obtain independent human listening and reference recordings before making any quality or deployment decision. The saved voice is synthetic and Q4_K-derived, and there is currently no human reference, listening panel, or acceptance threshold. The new texts avoid exact fixture reuse but share a workshop/logistics narrative pattern with the earlier library and museum cases, so they do not establish broad prompt-family independence.

Validation performed for this specification:

```sh
python3 -m json.tool benchmarks/depth-corpus/language-quality-manifest.json >/dev/null
git diff --check
```

Both commands pass. The existing compiled splitter helper `.beehive/agent/BREEZE-LONGFORM/split_probe` reports one part for each short fixture, three parts for each unreferenced long fixture, and two parts for each saved-voice long fixture, with exact concatenation in every case. No GPU, pod, network, download, corpus collection, or new runtime code was used.
