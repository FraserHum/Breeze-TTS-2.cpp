# Breeze and audio.cpp on the Radeon 780M

**The validated default is transposed-convolution matmul only.** It passes the quality gate on both quants. Stateful streaming misses the waveform gate; fused depth changes sampling and lowers q4_k UTMOS on this prompt. Both remain opt-in.

Fresh Breeze measurements below use the exact original report prompt. The old throughput conclusion was based on disabled optimizations and no longer describes the available paths. Faster experimental numbers must not be presented as the validated default.

## Setup and definitions

- Run date: 2026-09-06 Pacific/Auckland (2026-09-05 UTC). Ryzen 7 8845HS / Radeon 780M, RADV PHOENIX, Vulkan0, UMA, KHR_coopmat; dev pod `breezetts-dev`, namespace `hermes-voice`. The device log is authoritative; no unverified gfx identifier is assigned here.
- Benchmark source: `18ff504be236b5c599a974b978dab05fb737751f`, rebuilt binary MD5 `dbf13d6f9c18afffb020923c0158d6f6`. Earlier `b15e1fc` optimized WAVs were corrupted and are superseded. The receipt first lists the old binary before rebuilding; the later hash and `BUILD_RC=0` identify the measured build.
- Default change: `0c03bf4fa9519c5908d431e29dd09dc4aa7ee79a`; rebuilt binary MD5 `225a3e5afc8c6b3a7d76f6ac83c3a6b3`. Default and opt-out smoke WAVs are checked against the explicit matmul and legacy runs.
- Exact instruction: `Speak clearly and naturally.` Seed 42, no reference audio, mono 24 kHz. Default chunk-first/max are 40/40 frames (3.2s per full chunk). Two sequential runs per configuration, one GPU consumer at a time. Production logs showed no generation since 2026-08-31; other host activity and clock variation were not continuously measured.
- **Decode RTF** = (backbone decode + depth decode + vocoder milliseconds) / audio milliseconds. **Stage RTF** additionally includes reference encode, prompt build and backbone prefill. Lower is better; <1 is faster than realtime by that definition.
- Neither stage sum is a measured end-to-end generation wall clock: initialization, uninstrumented glue and model load are excluded. `first vocoder` is already included in `vocoder` and is never added twice. First-audio below is measured from generation start, excluding model load.

Exact prompt:

> The quick brown fox jumps over the lazy dog. We are benchmarking the Breeze TTS two model on the Radeon 780 M graphics card, measuring the real time factor of the full generation pipeline from the text encoder through the backbone and depth decoder.

## Fresh Breeze results

Ranges are the two repeats; stage means use full reported stage totals rather than rounded per-frame values.

| Configuration | Quant | Frames / audio | Decode RTF | Stage RTF | First audio |
|---|---|---|---:|---:|---:|
| Legacy (all off) | q4_k | 213 / 17.04s | 1.5407–1.5791 | 1.5494–1.5879 | 3610–3635ms |
| **Default: matmul only** | q4_k | 213 / 17.04s | 1.0639–1.0718 | 1.0729–1.0807 | 3048–3060ms |
| VOC2 (experimental) | q4_k | 213 / 17.04s | 0.9508–0.9624 | 0.9596–0.9711 | 3043–3050ms |
| All three (experimental) | q4_k | 257 / 20.56s | 0.9325–0.9405 | 0.9398–0.9478 | 2999–3019ms |
| Legacy (all off) | q8_0 | 202 / 16.16s | 1.9926–2.0049 | 2.0033–2.0139 | 5155–5258ms |
| **Default: matmul only** | q8_0 | 202 / 16.16s | 1.5842–1.5894 | 1.5932–1.5984 | 4734–4758ms |
| VOC2 (experimental) | q8_0 | 202 / 16.16s | 1.4671–1.4676 | 1.4761–1.4765 | 4628–4752ms |
| All three (experimental) | q8_0 | 238 / 19.04s | 1.4327–1.4373 | 1.4404–1.4449 | 4584–4590ms |

Mean decode cost, ms/frame:

| Configuration | Quant | Backbone | Depth | Vocoder |
|---|---|---:|---:|---:|
| legacy | q4_k | 15.07 | 52.10 | 57.62 |
| matmul | q4_k | 15.12 | 52.34 | 17.98 |
| voc2 | q4_k | 15.14 | 52.23 | 9.16 |
| all3 | q4_k | 15.42 | 50.35 | 9.16 |
| legacy | q8_0 | 23.52 | 84.24 | 52.14 |
| matmul | q8_0 | 23.58 | 84.52 | 18.84 |
| voc2 | q8_0 | 23.61 | 84.36 | 9.42 |
| all3 | q8_0 | 23.76 | 82.13 | 8.92 |

Actual Breeze file sizes: q4_k **2,538,909,376 bytes**; q8_0 **3,568,844,480 bytes**. No q4_0 file was tested. Quantization labels alone do not guarantee identical tensor policies between engines.

## Historical audio.cpp comparison

These audio.cpp 0.7.2 results are retained from the earlier report, not remeasured in this run. Their RTF uses `metrics.wall_ms`, so it is not precisely the same timing boundary as the Breeze stage sums.

| Engine / quant | Audio | Generation wall | Wall RTF | CLI first output including load |
|---|---:|---:|---:|---|
| audio.cpp q4_k | 19.20s | 26.1–26.7s | ~1.38 | ~128s, including ~100s runtime requantization |
| audio.cpp q8_0 | 19.20s | 35.2–36.2s | ~1.86 | ~40s |

The safe Breeze default has lower reported stage ratios (~1.08 / ~1.60) than these historical audio.cpp wall ratios (~1.38 / ~1.86), alongside streaming output. This supports the direction of the optimization win; it is not a controlled same-session engine-isolation percentage. The default still exceeds 1.0 RTF on both quants. VOC2 and all-three achieve q4_k stage RTF below 1, with the quality qualifications below.

The audio.cpp q4_k package was runtime-requantized from its q8_0 package (bf16→q8_0→q4_k), whereas Breeze uses separately prepared weights. Package sizes and tensor policies differ. audio.cpp emits a single CLI chunk; its 128-second load-inclusive output time should not be compared directly with Breeze’s 3-second generation-only first audio. Model load and conversion explain much of that absolute gap.

## Quality and defaults

Every configuration/model pair produced byte-identical repeats. Legacy, matmul and VOC2 preserve frame counts. Perceptual scoring: ffmpeg-resampled 16 kHz PESQ-WB and SpeechMOS v1.2.0 UTMOS22 strong; exact versions, checkpoint hash and waveform hashes are in [the evidence bundle](780m-2026-09-06.json). Legacy self-comparison PESQ is 4.6439.

Scoring implementations: [PESQ](https://github.com/ludlows/PESQ) and [SpeechMOS/UTMOS](https://github.com/tarepan/SpeechMOS/tree/v1.2.0).

The predeclared waveform gate requires SNR ≥ 30 dB, correlation ≥ 0.999, maximum normalized PCM difference < 0.03, equal duration and repeatability. No threshold was relaxed after measuring.

| Candidate vs legacy | Quant | SNR | Correlation | Max difference | PESQ | UTMOS | Gate |
|---|---|---:|---:|---:|---:|---:|---|
| matmul | q4_k | 44.88dB | 0.999984 | 0.017365 | 4.6401 | 4.5514 | PASS |
| voc2 | q4_k | 38.35dB | 0.999927 | 0.039093 | 4.6365 | 4.5514 | FAIL |
| matmul | q8_0 | 44.19dB | 0.999981 | 0.009125 | 4.6396 | 4.5095 | PASS |
| voc2 | q8_0 | 28.95dB | 0.999368 | 0.064453 | 4.6006 | 4.5091 | FAIL |

Legacy UTMOS is 4.5512 (q4_k) / 4.5095 (q8_0). Stateful output is perceptually close on this prompt, but q4_k misses maximum difference and q8_0 misses maximum difference and SNR. The remaining difference also reproduces in a random-code CPU two-chunk diagnostic after the committed buffer-lifetime fix; bit-exact equivalence is not established.

With fused depth, q4_k UTMOS falls to 4.2649 from 4.5514 (VOC2); q8_0 scores 4.5256 versus 4.5091. Frame counts grow 213→257 and 202→238. Separate RNG draws change speech timing, so waveform SNR/PESQ between these independent generations is not a valid naturalness test. Depth top-p is unsupported in the fused path; it is 1.0 in the inspected checkpoint, so its omission is inactive for this configuration. Caller compatibility, including forced codebooks, and broader quality remain unvalidated for making fusion a default.

Final gate behavior:

- `BREEZE_VOC_CONVT_MATMUL`: **on unless exactly `0`**.
- `BREEZE_VOC_STATEFUL`: **off**, enable experimentally with `1`.
- `BREEZE_DD_FUSED`: **off**, enable experimentally with `1`.

## Model-side trial

`q4_k-dd2` with matmul-only reached decode RTF 0.8899–0.8984, stage RTF 0.8975–0.9061, with 244 frames / 19.52s and byte-identical repeats. However, UTMOS dropped to **2.9172** from **4.5514** for ordinary q4_k/matmul. This is a rejected quality-neutral deployment candidate, despite its speed. See [depth decoder frontier](depth-frontier-780m.md) for checkpoint dimensions, dependency constraints and the next experiment.

## Reproduce

Use the exact prompt above and explicit flags to distinguish configurations. Each run below uses the same instruction, seed 42 and default 40/40 chunks. Do not run configurations concurrently on the shared GPU.

```bash
# legacy: STATEFUL=0, CONVT_MATMUL=0, DD_FUSED=0
# matmul: STATEFUL=0, CONVT_MATMUL=1, DD_FUSED=0
# VOC2:   STATEFUL=1, CONVT_MATMUL=1, DD_FUSED=0
# all3:   STATEFUL=1, CONVT_MATMUL=1, DD_FUSED=1
BREEZE_VOC_STATEFUL=0 BREEZE_VOC_CONVT_MATMUL=1 BREEZE_DD_FUSED=0 \
  /src/build/breeze-cli /models/breeze-tts-2-q4_k.gguf \
  --text "<EXACT PROMPT ABOVE>" --instruction "Speak clearly and naturally." \
  --seed 42 --timings --output /tmp/matmul-q4_k-s1.wav
# Repeat as s2; use q8_0 for the other model.

# Historical audio.cpp q4_k command:
audiocpp_cli --task tts --family breeze_tts \
  --model /audiocpp/models/breeze-tts-2-q8_0.gguf \
  --backend vulkan --device 0 --session-option weight_type=q4_k \
  --text "<EXACT PROMPT ABOVE>" --seed 42 --out /tmp/audio.wav --metrics
```

[quality_780m.py](quality_780m.py) accepts a directory containing all 16 WAVs named `<legacy|matmul|voc2|all3>-<q4_k|q8_0>-s<1|2>.wav`. It writes quality.json and checks the fixed waveform gates. Run in an isolated environment with the versions recorded in the evidence bundle and ffmpeg available. The UTMOS checkpoint downloads on first use.

```bash
python benchmarks/quality_780m.py /path/to/wavs
```

Raw stage logs, numeric results and build receipts are in [the evidence bundle](780m-2026-09-06.json). WAVs and model weights are not committed. One prompt and two repeats establish repeatability here, not general speech quality or performance across longer inputs, other languages or voice-cloning cases.

Verify the archived arithmetic and smoke hashes without models or extra dependencies:

```bash
python3 benchmarks/verify_780m.py
```
