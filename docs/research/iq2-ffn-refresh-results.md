# Native calibrated IQ2_S FFN follow-up — 2026-09-15

The approved Q3 plus cache screen remained above 0.8 RTF. This bounded
follow-up reuses the existing calibrated IQ2_S donor, copying only its 36
depth FFN tensors onto the exact current Q4 baseline. All 1,125 other tensors,
shapes, tensor ordering and metadata are verified unchanged. No new kernel
or runtime default is introduced; the optional transposed cache is off.

- Baseline SHA-256: `483418fbbb438f5f1c08dbe2b017e42db0f1d126765cd3f772d8622a30a4915c`.
- IQ2 donor: `1f9238ed40787716c416e055e22e73431cbdd4ab2edc8a956a72a95516943d10`.
- IQ2 candidate: `523c11521591727c47c7a9725bf6f443c3c7c465e17324c4caf5dfb2fa2891e0`.
- Approved Q3 control: `34483105709c75c38cdafcddcb4ec91e0b222097d30e76432507c448053a83de`.
- Engine source `e6f15e5`, CLI MD5 `45190d015056b16ea4a0d1cbd631cf5c`.
- Donor provenance is the existing `quant-iq2_s.json`; this is not fresh
  calibration. Its original receipt is not modified by this experiment.

Both modes use the same deployed Calliope reference, household announcement,
instruction `Speak clearly and naturally.`, seed 42 and packed-weight flags.
Order is Q3/IQ2/IQ2/Q3, three resident repeats per process, first excluded.

| Mode | Warmed mean wall RTF | Depth ms/frame | Audio duration |
|---|---:|---:|---:|
| Approved Q3 | 0.879197 | 45.104 | 9.12 s |
| IQ2_S | 0.852823 | 43.016 | 9.20 s |

IQ2 improves RTF by 3.00%, saving 2.09 ms per depth frame. Its FFN payload
is 96,731,136 bytes versus Q3's 129,761,280, so the byte saving is much larger
than the observed compute saving. No inference of proportional bandwidth
benefit or achievement of 0.8 is made. Compare the paired controls here;
absolute timings differ from earlier sessions on this shared GPU.

All twelve generations completed below the frame limit and repeat exactly
within each model, but Q3 and IQ2 audio differ. Warmed WAVs were downloaded
without audio processing and verified against their remote SHA-256 and
24-kHz mono WAV durations. **IQ2 quality approval remains pending.** The
approved Q3 candidate is unchanged and no production model/default is changed.
This one short workload does not establish a broader runtime or quality pass.

Receipt (including builder, runner, exact commands and tensor hashes):
`benchmarks/depth-corpus/iq2-ffn-refresh.json`.
Audio: `.beehive/agent/BREEZE-IQ2-REFRESH-20260915/audio/`.
Verify: `python3 benchmarks/verify_iq2_ffn_refresh.py`.

## Delivery-tag rejection

The user reported that IQ2 ignored `(laugh)` and produced a second sigh in
its place in the paired Calliope delivery test (`calliope-delivery-tags.json`,
guidance 2). IQ2 fails this user quality check and is rejected for promotion.
Q3 remains the approved candidate. The earlier pending status is historical.
