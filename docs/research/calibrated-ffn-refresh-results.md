# Calibrated Q3 FFN on the current runtime — 2026-09-15

This experiment follows the strided-cache work committed through `cd30c5e`.
It reuses the existing activation-weighted Q3_K depth FFN tensors and copies
only those 36 tensors onto the exact current Q4 model. Depth attention and
all 1,125 other tensors, tensor ordering, shapes and metadata are verified
unchanged. No engine code, production default or deployment is changed.

- Baseline model SHA-256: `483418fbbb438f5f1c08dbe2b017e42db0f1d126765cd3f772d8622a30a4915c`.
- Calibrated donor SHA-256: `c5836c86c2a14c6b224f378b99ae289817c5d98335f3281d4f4cff41abf0a55d`.
- Candidate SHA-256: `34483105709c75c38cdafcddcb4ec91e0b222097d30e76432507c448053a83de`.
- CLI MD5: `45190d015056b16ea4a0d1cbd631cf5c`; engine source `e6f15e5`,
  ggml `acc4d4d7` (the later commit contains only experiment records).
- Donor calibration provenance: `benchmarks/depth-corpus/quant-calibrated.json`.
  This is reused calibration, not a new training or held-out acceptance run.
- FFN packed payload: 169,869,312 → 129,761,280 bytes (23.6% less).

The paired runtime screen uses the Radeon 780M dev pod, packed weights,
existing vocoder optimizations, and transposed V cache disabled in both modes.
The report prompt uses Q4/Q3/Q3/Q4 ordering and three resident repetitions per
process. The household announcement and synthetic saved-voice announcement
use one three-repeat process per mode; their ordering is not reversed and
therefore provides weaker timing evidence. First repetitions are warmups.
Audio trajectories may differ; depth milliseconds per frame is reported
alongside whole-pipeline RTF so duration changes cannot stand in for speed.

The household announcement is a fresh development prompt. No reserved held-out
manifest is opened or used. These three English cases are a bounded listening
screen, not broad speech acceptance. Any changed audio requires the user's
quality approval before retaining it as an accepted version or promoting it.

## Results

| Workload | Q4 RTF | Q3 RTF | Q4 depth ms/frame | Q3 depth ms/frame | Q4 / Q3 audio duration |
|---|---:|---:|---:|---:|---:|
| report | 0.924652 | 0.861077 | 50.419 | 45.231 | 17.04 / 16.80 s |
| home | 0.915042 | 0.829991 | 50.346 | 43.293 | 8.32 / 11.28 s |
| voice | 0.951912 | 0.880014 | 50.168 | 43.907 | 9.44 / 10.72 s |

## Decision

Report RTF improves 6.88%; household and saved-voice screens improve 9.29%
and 7.55%, respectively. All 24 generations completed below the model's
750-frame limit and repeated exactly within each case/model. All three
Q4/Q3 audio pairs differ. None reaches the 0.8 target. The household case
also increases absolute generation time because Q3 speaks for substantially
longer; lower RTF is not equivalent to a faster completed response.

**Candidate only; user quality approval pending.** No deployment or default
change is made. The paired WAVs are copied locally and verified against the
remote warmed-run hashes. Listening review:
`.beehive/agent/BREEZE-PRECISION-20260915/listening-review.md`.
No ASR/MOS score is used to substitute for the user's judgment. Calibration
includes prior English and Mandarin data, but this screen is English only.
The node is shared; no production requests appeared in inspected logs during
the window. Timing is not a dedicated-hardware result.

If quality is approved, freeze this exact model for a broader English timing
and listening matrix before promotion. The report remains approximately
4.89 ms/frame above the 64 ms/frame budget. Faster native precision choices
or bounded depth-kernel work can be assessed against this approved candidate;
the earlier long-context cache gain should not be added to short-speech savings.

Reproduce the model and cases with the scripts embedded in
`benchmarks/depth-corpus/calibrated-ffn-refresh.json`; it records all 1,161
output tensor hashes, the donor/base hashes, exact commands and runtime logs'
hashes. Run `python3 benchmarks/verify_calibrated_ffn_refresh.py` to check the
archived receipts. Model/WAV blobs remain outside Git.

## User listening approval — 2026-09-15

The user reviewed the pairs and stated: "the quality is fine". Audio quality
for this exact candidate is approved; the receipt's pending status records
the earlier measurement-time state. This is not a deployment or broader
speaker-consistency approval. The user also asked why the voices differ.
Every request used `Speak clearly and naturally.` and seed 42. Report/home
had no reference voice; the voice pair used the same `test17s` saved reference.
Each Q4/Q3 pair had identical text and conditioning. The generic instruction
specifies delivery, not speaker identity; the saved-reference case adds
speaker conditioning. Quantized weights can alter sampled audio codes even
with the same random seed, so identity/prosody consistency needs separate
assessment from acceptable overall sound quality.

## Deployed Calliope reference comparison

At the user's request, copied the exact deployed `/voices/calliope.breeze`
into the dev pod (SHA-256 `331f8daab3736cf24a31b0d1c54f0b87376ae6e196ec01411a084fc540a5d0d8`).
Both Q4/Q3 use that reference, the same household text, instruction
`Speak clearly and naturally.`, and seed 42. Three repetitions per mode
complete deterministically; warmed mean RTF is **0.934128 Q4 / 0.849720 Q3**.
Audio duration is **9.52 / 9.12 seconds**. This is a single-order screen,
not a reversed timing matrix. Copied warmed WAVs have verified remote hashes.
Receipt: `benchmarks/depth-corpus/calibrated-ffn-calliope.json`.
Production remains unchanged.
