# Untouched final workload gates

[final-workload-manifest.json](../../benchmarks/depth-corpus/final-workload-manifest.json) is the frozen input specification for the last acceptance run. It contains nine new exact texts: six design-voice fixtures covering English and Mandarin at short, medium, and long buckets, plus three saved `matrix` voice fixtures at short English, medium Mandarin, and long English. The matrix voice is synthetic and Q4_K-derived; it is explicitly not a human reference.

No audio, model output, activation data, generation result, or acceptance claim is included here. The target durations are planning buckets. After candidate selection, record the emitted WAV duration and bucket coverage without truncating or rewriting text after seeing candidate quality.

## Fixed workload

| Bucket | Target audio | `max_new` | Design split parts | Saved matrix split parts |
|---|---:|---:|---:|---:|
| short | about 10 s | 400 | 1 | 1 |
| medium | about 30 s | 800 | 1 | 1 |
| long | about 60 s | 1600 | 3 | 2 |

The runtime must omit `--split-chars`, use chunk sizes 40/40, and preserve each manifest seed, exact text, language instruction, and voice path/hash. Sampling request overrides stay omitted; the receipt records the model-embedded values. Runtime flags match the existing quality path: graphics queue, convolution matmul, and trim enabled; stateful vocoder, fused depth, capture, profiling, strict FA, and BB flash disabled.

Use four resident repeats per fixture. Repeat 1 is an explicit warmup; repeats 2–4 are the timed sample. Report all four. The p50 is the median of repeats 2–4. The p95 is the inclusive 95th percentile over those three values, matching `statistics.quantiles(values, n=20, method="inclusive")[18]`; it is descriptive over three timed samples and is not a population-tail guarantee. A separate process or an ambiguous discard rule is not equivalent evidence.

For every repeat, retain the command, return code, timeout, log and WAV SHA-256, WAV format/duration, emitted frame count, termination status, wall timing, stage timing, first-audio time, flush readiness and delivery-deficit calculation. A cap hit, split mismatch, hash mismatch, malformed WAV, timeout, nonzero return code, or delivery deficit remains recorded as a failure; no bad case is excluded.

## Acceptance gates

The selected candidate must satisfy every runtime gate on all nine workloads. Run and report the paired Q4_K control with identical inputs and settings, but do not apply the candidate’s .8 verdict to Q4: Q4 is the measured control, and a slower Q4 baseline does not invalidate a faster candidate.

- wall RTF p50 ≤ **0.8** and wall RTF p95 ≤ **0.8**;
- every timed first-audio value ≤ the existing approximately **3.1 s** goal;
- maximum parsed delivery deficit ≤ **0 ms** across repeats 2–4;
- every repeat finishes below `max_new` and passes WAV/hash/termination checks.

Wall RTF is generation wall milliseconds divided by **1000 × emitted audio seconds**. Stage RTF is the reported stage total, excluding the first-vocoder diagnostic, divided by **1000 × emitted audio seconds**. Delivery deficit is the flush-readiness deficit after initial playback buffering; zero here is not a measurement of an audio-device underrun.

The candidate and Q4 control must use identical text, instruction, seed, voice mode/path/hash, split policy, cap, environment, chunk settings, and sampling request. Freeze candidate model identity/hash, CLI/source revision, and these settings before generating any held-out audio. Do not fit, tune, change thresholds, or rewrite fixtures from final results.

## Quality evidence

Run current local ASR, speaker, and MOS proxies where their inputs exist, retaining transcripts, commands, durations, hashes, model/checkpoint hashes, and paired candidate/Q4 values. English uses normalized WER and Mandarin uses normalized CER. The matrix speaker score is descriptive against a synthetic reference. None of these proxies is a human quality gate or proves intelligibility, prosody, speaker identity, or deployment quality. Paired human listening against Q4 remains required before acceptance; new independent human reference recordings are useful context but are not an added gate for this manifest.

The existing performance and quality receipts provide context only: [default-split-longform.md](default-split-longform.md), [depth-quant-matrix.md](depth-quant-matrix.md), [language-quality-runtime.json](../../benchmarks/depth-corpus/language-quality-runtime.json), and [quant-longform.json](../../benchmarks/depth-corpus/quant-longform.json). They do not pre-accept this untouched workload.

## Static checks

The existing splitter probe is run on every manifest text, adding `reference` for saved voice. Require the declared part count and `concat=ok`; its valid one-part return convention is recorded in the manifest. The static verification used for this specification is:

```sh
python3 -m json.tool benchmarks/depth-corpus/final-workload-manifest.json >/dev/null
python3 - <<'PY'
import json, subprocess
from pathlib import Path
m = json.loads(Path("benchmarks/depth-corpus/final-workload-manifest.json").read_text())
f = m["fixtures"]
assert len(f) == len({x["id"] for x in f}) == len({x["text"] for x in f}) == 9
assert sum(x["voice_mode"] == "design" for x in f) == 6
assert sum(x["voice_mode"] == "saved" for x in f) == 3
assert {x["language"] for x in f} == {"en", "zh"}
assert {x["seed"] for x in f} == {42, 123}
prior = set()
for name in m["provenance"]["prior_fixture_sources"]:
    def walk(value):
        if isinstance(value, dict):
            for key, child in value.items():
                if key == "text" and isinstance(child, str): prior.add(child)
                else: walk(child)
        elif isinstance(value, list):
            for child in value: walk(child)
    walk(json.loads(Path(name).read_text()))
assert not ({x["text"] for x in f} & prior)
for x in f:
    probe = [".beehive/agent/BREEZE-LONGFORM/split_probe"]
    if x["voice_mode"] == "saved": probe.append("reference")
    p = subprocess.run(probe, input=x["text"], text=True, capture_output=True)
    assert p.stdout.startswith(f"parts={x['expected_default_parts']} concat=ok ")
print("PASS: final workload IDs, counts, language/seed coverage, text disjointness, and split probes")
PY
git diff --check
```

This check validates the inputs only. It must not be reported as a runtime, quality, or acceptance result.
