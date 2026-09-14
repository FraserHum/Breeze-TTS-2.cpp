# Unexplored optimization opportunities: collaborative investigation

2026-09-15. Source baseline `132cda7`, ggml `acc4d4d7`. Investigation and
read-only device/model inspection only; no new performance benchmark, runtime
change, model change, or deployment. Companion: [research partner's findings
and challenges](rtf-080-research-partner-20260915.md).
The [inspection receipt](../../benchmarks/depth-corpus/research-inspection-20260915.json)
records device properties and the model-derived receptive-field arithmetic.

## Conclusion

The strongest concrete new kernel experiment is **Q3-only wave32 versus the
current explicitly requested wave64**. The cheapest additional graph experiment
is **using the computed 11-frame convolution history instead of retaining 16**.
Recorded Vulkan command reuse is a distinct, potentially useful area, but needs
evidence of exposed command-recording cost before a backend implementation.
These are hypotheses, not demonstrated savings.

Our current short Calliope development result is 0.863026 wall RTF, or 69.04 ms
per 80 ms audio frame. Reaching 0.8 on that fixture needs another **5.04 ms/frame**,
about 7.3% of current time. It does not establish the gap on every workload:
long-form context and CFG=2 tagged speech have different costs. Current warm
stages are depth 43.50, backbone 16.33 and vocoder 8.22 ms/frame; stage sums
exclude some setup/overhead. [Current timing and parity receipt](../../benchmarks/depth-corpus/depth-sync-probe.json).

## Corrections to the previous search

- **F16 depth heads were already tested.** The older Q4 trial saved 0.464 ms/frame
  with identical WAVs and was discarded under a then-fixed 1 ms cutoff. My last
  recommendation overlooked this receipt. Repeating it on approved Q3 would be
  a replication under changed conditions, not exploration. The later policy
  accepts small repeatable native gains in proportion to maintenance cost.
  [Old measurements](../../benchmarks/rtf-080-experiments.json),
  [revised decision policy](fresh-optimization-audit.md#decision-policy).
- **Static ggml graphs are not recorded GPU command replay.** The backend still
  walks graph nodes, checks fusions and records commands. Reusing command-buffer
  allocations is also different from resubmitting their unchanged contents.
  The current Vulkan code begins one-time command buffers and resets them.
  [Vulkan backend](../../third_party/ggml/src/ggml-vulkan/ggml-vulkan.cpp).
- **Numerical rejection and listening rejection differ.** Backbone flash
  attention failed the old tensor-difference gate; speech and speed were never
  measured. That does not prove acceptable speech, but it also does not prove
  unacceptable speech. In contrast, IQ2 has an actual user-reported delivery
  failure. The user's current instruction allows numerical differences while
  requiring approval of changed audio. [Flash record](backbone-flash-attention.md),
  [IQ2 record](iq2-ffn-refresh-results.md).

## Ranked experiments

| Priority | Candidate / status | Smallest useful test | Decision and likely scope |
|---|---|---|---|
| 1 | Q3 wave32; untested pipeline specialization | Map actual Q3 floating gate/up ISA and compare 32/64 subgroups in the existing real-weight FFN benchmark, with two output rows unchanged | Repeatable kernel gain first, then ordinary and tagged Calliope. No assumed speedup; altered reduction may change audio. |
| 2 | Convolution history 16→11; untested refinement of retained trim | Use existing receptive-field result, keep full transformer work, check chunk boundaries and repeated window shifts | Low implementation cost, likely a small finishing gain; cannot close 5 ms alone. |
| 3 | Recorded per-depth-step command replay; untested backend mechanism | Measure recording/submission CPU time and GPU gaps separately; prototype one fixed graph only if material | Higher maintenance cost: descriptors, buffers, scratch and barriers must remain valid. Preserve host sampler and 15 sequential decisions. |
| 4 | Short-context depth attention fusion; new application within a studied family | Isolated score/softmax/value fusion at lengths 2…16, n=1/2 and CFG branch layouts | Dispatch reduction, not long-sequence asymptotics. Generic flash kernel may be slower at these shapes. |
| 5 | Backbone flash attention; reopened under changed quality policy | Recheck mapping and numerical behavior, then timed full speech at actual Calliope contexts and user listening if useful | Most plausible at long contexts; old tensor drift needs investigation, not dismissal or automatic acceptance. |

### Q3 wave32: why this is more specific than another workgroup guess

The floating Q3 pipeline receives `force_subgroup_size16`, derived from device
subgroup size, explicitly at creation. The inspected device reports subgroup
64. The one/two/four-row experiments changed output rows and their dispatch
divisor, not this subgroup request. Merely adding RDNA3 to the RDNA1/RDNA2
default table would not override the explicit argument. Trace from matvec
subgroup setup near line 5280 to Q3 creation near 5318 and pipeline creation
near 3051 in the [pinned backend](../../third_party/ggml/src/ggml-vulkan/ggml-vulkan.cpp).

Live read-only `vulkaninfo` confirms RADV PHOENIX, Mesa 26.0.8-1ubuntu0.3,
subgroup size 64, minimum 32, maximum 64, subgroup-size control enabled, and
compute among the required-subgroup-size stages. Thus the API capability is
present; compiled ISA and resulting performance still need verification.
Vulkan supports an explicit subgroup request within the advertised limits.
[Vulkan subgroup-size specification](https://docs.vulkan.org/refpages/latest/refpages/source/VkPipelineShaderStageRequiredSubgroupSizeCreateInfo.html).

Change subgroup/workgroup/reduction parameters coherently for the Q3 floating
pipeline only; keep other formats, integer-dot route and row count unchanged.
Inspect n=1 and n=2 separately. Global `RADV_PERFTEST` switches are not an
equivalent targeted comparison. AMD's guidance explains why wave size changes
resource allocation, but warns that greater occupancy can hurt performance.
[AMD occupancy guidance](https://gpuopen.com/learn/occupancy-explained/).

### Smaller convolution history: a concrete shape calculation

The approved model has upsampling ratios `[2,2]`, decoder rates `[8,5,4,3]`,
and 1,920 samples/frame. Evaluating the existing `conv_tail_history()` recurrence
against its tensor shapes gives **19,928 samples**, hence `19928 / 1920 + 1`
with integer division gives **11 frames**. Yet `vocoder_decode()` retains
`min(trim_prefix,16)` after checking this computed history. The caller already
derives its discarded prefix from returned audio length.
[History calculation and trim](../../src/codec_decoder.cpp),
[flush offsets](../../src/generation.cpp).

This is a source-derived candidate, not proof of waveform equality. Validate
the receptive-field bound, output offset, phase alignment and boundary padding
with starts before/after the 88-frame context begins moving, chunk counts
1/4/40, partial final chunks and multi-part reset. First compare fixed-code
vocoder outputs, then full ordinary/tagged Calliope. Avoid tuning held-out final
acceptance texts.

For perspective only: five fewer frames out of a 56-frame convolution tail
would be about 9% less tail work on a full 40-frame chunk. Applying even that
ratio to the entire 8.22 ms vocoder stage gives only 0.73 ms/frame, and leaves
out the unchanged transformer and initial chunk. This is rough proportional
arithmetic, not a latency bound or measured estimate.

### Replay and hardware profiling: missing evidence, not a new framework

The recent host phase probe already rules out large savings in staging,
uploads and sampling. It does **not** isolate driver recording, synchronization
and device execution inside graph compute. Nor is logical weight-byte throughput
a DRAM-bandwidth counter. Capture a small representative Q3 graph, map its
pipelines to ISA/resource usage, and distinguish CPU recording from wait time.
Use GPU timeline/counters where available to discriminate memory stalls,
unpacking work, occupancy and queue gaps.

Mesa documents compute-only per-submit RGP capture and a profiling power-state
selection whose documented default is peak. Verify support and clock behavior
in installed 26.0.8 before using those options. `perf` and `rgp` were not found
on the dev pod PATH; this does not establish that no host-side analysis tool
is available. Do not let tooling setup become an open-ended prerequisite for
the two small discriminating experiments above. Captured durations cannot
replace ordinary uninstrumented A/Bs. [Mesa trace and profiling variables](https://docs.mesa3d.org/envvars.html).

Official Breeze provides a useful precedent: its CUDA/PyTorch path captures
the full depth loop including GPU sampling. Our proposed per-step replay
deliberately keeps the existing host decisions, so it is different from
re-enabling our fused-depth graph. No CUDA timing transfers to this 780M.
[Official depth graph, pinned source](https://github.com/breezeblue-ai/breeze-tts/blob/008f769016b0a24711becd7a4925030bc93f608c/models/cudagraph/depth_decoder_graph.py).
Reusable Vulkan recordings must preserve all referenced resource and command
pool lifetimes. [Khronos command-buffer guidance](https://github.khronos.org/Vulkan-Site/samples/latest/samples/performance/command_buffer_usage/README.html).

## Ideas we challenged or deprioritized

| Idea | Outcome of discussion |
|---|---|
| Generic residual/RMSNorm fusion | Already present; partner withdrew the generic proposal after checking executed fusion families. |
| Packed gate/up projection plus SwiGLU epilogue | Distinct, but pairs of rows are separated in storage; meeting both reductions may need a new layout. Standalone activation timing is only a small ceiling. Defer custom shader until mapping supports it. |
| Cache only convolution intermediates, keep transformer recomputation | Initially appealing. Old transformer latents can differ when context shifts, invalidating reused convolution state. Require overlap equality first; computed-history trim is simpler. |
| Selective Q3/IQ2 | New mixture, old family. Full IQ2's few-percent gain and actual laugh failure make hand-tuning layers to one clip unattractive. Use broader sensitivity data if revisited. |
| Codebook-conditioned FFN/layer skipping | Untested selective schedule within the previously researched model-compression family. Larger potential, but late codebooks also affect future backbone feedback. Screen sensitivity, then full rollouts; user approval required. |
| Parallel depth student / joint feedback-and-codec latent student | Largest structural opportunity, already identified in earlier research. Training and deployment effort are substantial; not a newly discovered shortcut. |
| Speculative depth decoding | Needs a draft, position-specific heads, distribution-correct verification and cache rollback. Partner's primary-source review found no basis for assuming a ready lossless gain here. |
| Another global backend or node-count switch | HIP, global integer routing and larger node cap already have negative measurements. Revisit only with a new mechanism or changed runtime evidence. |

The partner note carries primary sources for speculative decoding and trained
early exits. Earlier [Breeze-specific mathematical research](breeze-specific-transferable-concepts.md)
already covers parallel students, structured sparsity, low rank and feature
forecasting. Do not relabel those families unexplored because a particular
configuration remains untested.

## What to do next

1. Timebox Q3 pipeline/ISA mapping and recording attribution. If tools are
   unavailable, retain that uncertainty and use the existing paired FFN bench
   to falsify the specific wave32 hypothesis; do not start a profiler project.
2. Run targeted wave32 and computed-history probes independently. Stop a shader
   candidate at the microbenchmark if it cannot beat controls. Retain a small
   native change only for a repeatable end-to-end benefit appropriate to its cost.
3. Promote command replay to implementation only if exposed recording overhead
   warrants its resource-lifetime complexity. Otherwise move to the two
   attention investigations, with numerical differences and speech quality
   evaluated separately.
4. If these fail to close the gap, choose an explicit model-training/approximation
   project rather than an endless sequence of low-ceiling knobs.

All changed audio still requires the user's quality approval. Keep Calliope
identity, delivery tags, speech content, termination, first audio and playback
readiness in acceptance. Keep CFG=1 and CFG=2 results separate. Never add savings
from different baselines or tune on the reserved final-workload manifest.
