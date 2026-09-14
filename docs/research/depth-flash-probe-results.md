# Depth flash attention and submission batching — 2026-09-15

Native depth flash attention is a promising **quality-review candidate**, not
an accepted default. The paired short Calliope experiment reduces depth time
by **1.817 ms/frame** and wall RTF from **0.830414 to 0.809698**. Audio changes;
user approval is pending. Production is unchanged and 0.8 remains unmet.

## Paired speech result

Approved calibrated Q3 FFNs, deployed Calliope reference, instruction
“Speak clearly and naturally.”, seed 42, CFG=1, and the existing household
prompt. Three resident repeats per process; exclude the first. Execution order
is candidate/control/control/candidate, one GPU benchmark at a time.

| Block | Warm wall RTF | Wall ms | Depth ms/frame | First audio ms |
|---|---:|---:|---:|---:|
| Flash A | 0.812614 | 7476.05 | 40.168 | 2589.5 |
| Control A | 0.826716 | 7539.65 | 41.566 | 2664.5 |
| Control B | 0.834112 | 7607.10 | 41.961 | 2633.0 |
| Flash B | 0.806783 | 7422.40 | 39.725 | 2622.0 |
| Control pooled | 0.830414 | 7573.375 | 41.763 | 2648.75 |
| Flash pooled | 0.809698 | 7449.225 | 39.947 | 2605.75 |

Both candidate blocks beat both controls in depth time. The pooled stage
reduction is 4.35%. Actual request wall time falls **124.15 ms, or 1.64%**.
The candidate produces 115 frames/9.20 seconds instead of 114/9.12, so its
2.49% RTF improvement also benefits from a 0.88% longer audio denominator.
Do not credit that entire RTF change to faster computation or assume identical
speech content from frame counts alone. This is one development fixture,
not the reserved final workload suite.

All six ordinary candidate WAVs have SHA-256
`3e35dafe2dd8a9d78c30fe271401d457e4765885d2b8bcf8c52bf384c64a6681`.
The six control WAVs match the previously approved Q3 Calliope hash
`f876f7bbcdaa65e73fc105e7151519f0f778780ff27255cd6d89adbac19ee0e3`.
Within-candidate repeatability does not establish listening quality.

The retained experiment is the opt-in `BREEZE_DD_FLASH_ATTN=1` graph path:
one native attention operation replaces score/softmax/value operations;
Q/K/V layout and cache writes retain their existing semantics. The causal/CFG
mask is cast once per step from F32 to F16, preserving zero and negative
infinity exactly. The default remains off.

## Numerical diagnosis: failed gates remain failed

The primitive test covers head dimension 128, eight query/two KV heads,
one/two CFG branches, initial queries and subsequent decoding at lengths
2/3/4/15/16. CPU passes all ten numerical cases. Native Vulkan fails eight
under the unchanged `max_abs < 0.001` and `relative_l2 < 0.001` gate;
worst differences are approximately **0.003065 / 0.001165**.

A separate zero-query test independently checks the expected uniform average,
branch isolation, grouped-head mapping, causal exclusion, and poisoned unused
cache capacity. CPU passes all ten. Native Vulkan passes lengths 2/4/16 but
fails the odd-length analytic checks, reaching 0.020836 absolute difference.
These failures were not relabelled passes or hidden by widening tolerances.

Source inspection identifies reduced-precision work despite requesting
`GGML_PREC_F32`: the cooperative shader stores scaled queries and softmax
weights in F16, and its value-product accumulator/output intermediates also
use `FLOAT_TYPE`. F32 controls the score accumulator, not every intermediate.

Running the unchanged test binary with `GGML_VK_DISABLE_F16=1` and
`GGML_VK_DISABLE_COOPMAT=1` selects the existing scalar F32 shader. All ten
numerical and all ten analytic cases pass, with worst numerical max difference
**8.18e-5** and analytic difference **7.63e-6**. Together with CPU agreement,
this strongly supports precision loss rather than an obvious tested-layout or
mask error. It does not prove every full-model state correct. These global
switches were diagnostic only, not the speech performance configuration.

## Submission batching: reject

The existing asynchronous depth timing exposes approximately 7.41 ms/frame
in compute enqueue and 33.36–33.80 ms/frame in readback/wait. Enqueue includes
recording that overlaps GPU work; it is not wholly removable wall time.

A shape-targeted probe changes the backend FLOP subdivision from divisor 40
to 4. Diagnostics confirm that the common depth graph drops from nine submits
to four, unlike the earlier node-cap-only experiment. However, paired
control/candidate/candidate/control warm depth times are
**41.658 / 42.193 / 42.231 / 42.168 ms/frame**; corresponding RTFs are
**0.829024 / 0.837023 / 0.835998 / 0.835148**. Fewer submissions did not improve
throughput. Reject the probe rather than retaining another runtime knob.

Recorded command replay remains a separate, larger change: current cleanup
resets command pools and recycles descriptor bindings. Keeping a command buffer
alone would not preserve its referenced resources. The failed batching probe
does not disprove replay, but supplies no reason to implement it now.

## Restored backend and next decision

Root `4fb9679` restores ggml `acc4d4d7`; the Vulkan library is restored to
SHA-256 `172a074e97c595f8c6c6d8eb1c4d1e54f5a3090573e9bb537ff62e1b05f8d1e8`.
The winning graph experiment needs no new backend shader or backend fork.
Three restored-backend ordinary runs per mode reproduce every corresponding
control/candidate WAV hash. Excluding warmup gives RTF 0.841458 versus
0.809049; this sequential restoration check is corroboration, not pooled into
the balanced experiment above.

Three restored-backend CFG=2 sigh/laugh runs per mode finish at 151 frames /
12.08 seconds. Controls retain approved SHA-256
`52f0a1c9b0d4cb358dc9b0600edd02db0a22ddd9b8025e76b8408cb63ceab194`;
all candidates share changed hash
`3dc18e86ee015c747840260d0aa08b7bd0fb96a3109e4442d35c6cb3ffa39cfb`.
These are quality/CFG checks, not speed acceptance: model hashing overlapped
tagged generation. CFG=2 remains slower than real time in these runs; the
approximately 0.81 result applies only to the ordinary CFG=1 fixture.

Restored ordinary runs deliver an initial 3.2 seconds of audio after
2.571–2.742 seconds across both modes. Their later flushes have zero calculated
playback deficit **when playback begins at that first flush**; this is not zero
startup latency. Tagged candidate startup is 3.873–3.979 seconds, and its
remaining stream has a calculated deficit of 167.1–571.0 ms; tagged controls
also have deficits. All flush sequences are monotonic and finish at the
recorded audio duration. Thus the tagged samples do not establish uninterrupted
real-time playback without additional buffering.

Obtain user quality approval before enabling or promoting the candidate.
Do not stack backbone flash changes into this listening decision. After
approval, the previously measured F16 output-head change is a cheap finishing
replication; backbone flash is the next separate higher-ceiling experiment,
especially for longer contexts. Neither gain is assumed in the current result.

Commands, hashes, diagnostics and speech timings are retained in the
[machine-readable receipt](../../benchmarks/depth-corpus/depth-flash-probe.json).
Run `python3 benchmarks/verify_depth_flash_probe.py` to check the receipt.
The [primitive tool](../../tools/depth_attention_test.cpp) retains both the
original numerical mode and the independent `--mask-only` diagnostic.

See [the collaborative investigation](rtf-080-unexplored-20260915.md) and
[the preceding wave/history experiment](wave32-trim11-results.md).
