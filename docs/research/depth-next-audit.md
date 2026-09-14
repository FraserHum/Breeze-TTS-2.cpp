# BREEZE-DEPTH-NEXT capture and FFN seam audit

Implementation outcome: the probe marks existing nodes as outputs, without
adding copy nodes. Vulkan capture requires `GGML_VK_DISABLE_FUSION=1`; normal
seeded sampling then matched the retained baseline for all four pilot prompts
and the CFG check. Use the [implemented recipe](../depth-capture.md) and
[corpus manifest](../../benchmarks/depth-corpus/manifest.json) below in preference
to this initial design's proposed copy sinks or greedy/replay fallback.

Audit date: 2026-09-06. This is a capture specification, not a measurement
result. It is based on `docs/handoffs/breeze-investigations/00-shared-activation-corpus.md`,
the current `src/depth_decoder.cpp`, `src/common.cpp`, `src/backbone.cpp`,
`src/codec*.cpp`, and the pinned ggml Vulkan source. No GPU job or product
change was made for this audit.

## Current graph and tensor lifetimes

The deployed depth path has 16 codebooks and therefore 15 sequential residual
steps per frame. `DepthRunner::init` builds 15 static graphs, one for each
`j = 1..15`, and allocates them once on the dedicated `depth_alloc`. The
topology is not interchangeable: step 1 has the backbone hidden plus `cb0`,
steps 2..15 have one audio-code input, and each step bakes its KV view extent,
append offset, and codebook-head slice.

For a branch count `nb`, graph `j=1` has `n_tok = 2*nb`; every later graph has
`n_tok = nb`. The token order is interleaved by position and branch:
`token = position * nb + branch`. With CFG (`nb=2`), step 1 is four tokens and
the later steps are two tokens. Attention writes K/V through `dd_cache_append`
into the persistent `KVCache` backend buffer. The KV views and the append
offsets are graph structure; the cache bytes survive until `DepthRunner::free`.

The depth graph's persistent input leaves are:

| leaf | shape | provenance and update |
| --- | --- | --- |
| `aud` | `[nb]` i32 | current audio code, uploaded each step |
| `h0` | `[2048, nb]` f32, step 1 | last backbone hidden for each branch, uploaded at step 1 |
| `pos` | `[n_tok]` i32 | step-local positions, uploaded each step |
| `ff` | `[64]` f32 | RoPE frequency factors, constant for this model |
| `mask` | `[total, n_tok]` f32 | branch-causal mask, uploaded each step |

Only `logits` is a normal graph output. It is `[2051, nb]` f32, with vocabulary
index fastest and branch-major blocks. `run()` reads it with
`ggml_backend_tensor_get`, combines CFG on the host, samples, and stores the
depth codes in a host `std::vector<int>`. The `ggml_cpy` roots update the
persistent KV buffer and are already retained by the graph.

All other graph tensors are allocator scratch. In particular, the FFN input,
gate/up activations, SwiGLU product, down projection, and residual output can
be overwritten by later nodes or by the next graph compute. A capture path
must therefore retain a selected tensor as an output (or copy it to a
capture-only output tensor), call `ggml_backend_graph_compute`, read it back
before the next step can reuse the buffer, and then reuse the host storage.
Reading an unmarked intermediate after compute is not valid. Building capture
outputs also changes allocator reuse and can disable the existing fused
down-plus-residual path, so capture is an opt-in diagnostic graph built before
`DepthRunner::init`, never a silent production mode.

The backbone already has a host boundary: `backbone_run` marks the final
hidden tensor as an output and returns `StepOut::hidden` after
`tensor_to_f32`; its final hidden is `[2048]` for the last token. `cb0` is
sampled on the host immediately afterward. `audio_embed_forward` then returns
the `[2048]` feedback embedding after summing all 16 codebook rows. Complete
frame codes are also available without graph instrumentation through
`BREEZE_DEBUG_DEPTH_CODES=1`, which emits one `DEPTH_CODES` line containing
`cb0` followed by `cb1..cb15`.

The default vocoder path does not expose a stable host copy of its continuous
codec latent. `quantizer_decode` produces the latent inside the decode graph,
then dpre, transformer, and the decoder consume it; marking it for readback
would be another capture-only graph and another fusion/lifetime perturbation.
`decode_stateful` already reads `z_new`, but that is the experimental stateful
path and is not the default decode contract. Defer codec latent, waveform
intermediates, and downstream feedback capture until the depth sample proves a
need for them. The backbone hidden, `cb0`, complete codes, and selected depth
FFN tensors are sufficient for the first sparsity/rank screen.

## Minimal capture implementation

Keep the normal graph unchanged when capture is disabled. In capture mode,
record references to the selected FFN tensors while `dd_layer` constructs the
graph, add a `ggml_dup`/output sink for each selected reference, and retain the
sinks in the step graph. Allocate the capture graph's arena and dedicated
allocator with the extra output tensors accounted for. After each step:

1. upload the same `aud`, `h0`, `pos`, `ff`, and `mask` values;
2. compute the static graph;
3. read the selected output sinks and `logits` into preallocated host buffers;
4. attach `(utterance, frame, step, layer, branch, position, tensor)` metadata;
5. proceed with the existing host sampling path.

The selected tensors should be the normalized FFN input, the post-SwiGLU
product, and the down-projection output. They cover input covariance,
intermediate activity, and FFN output while keeping the pilot bounded. If a
later consumer needs separate gate and up statistics, run a second small
probe or stream reductions; do not retain every FFN tensor in the first
corpus. The source seam is `swiglu_ffn` in `src/common.cpp`: gate and up are
`[8192, n_tok]`, the product is `[8192, n_tok]`, and down returns
`[1024, n_tok]`. The surrounding `dd_layer` calls the FFN normalization at
`src/depth_decoder.cpp:265`, the SwiGLU at `:266`, and the residual add at
`:267`.

Use one little-endian float32 blob per run plus a JSON manifest. A record key
must include the layer and step; store the raw `ne[]` shape and token order,
not just a flattened array. The manifest should include the model path and
SHA256, git and ggml revisions, backend/device, quantization, seed, prompt or
utterance id, language, voice/reference id, CFG branch count, force mode,
frame index, codebook step, tensor name, dtype, shape, byte offset, and the
baseline/capture code and WAV hashes. This is enough to reconstruct every
sample without introducing a logging framework.

The bounded first probe is `cap16`: layers `{0, 5, 11}`, all 15 codebook
steps, and frame indices `{0, 7, 15}` from a 16-frame run. Use `cfg_scale=1`
(`nb=1`) for the first corpus so branch layout is not mixed with the primary
statistics; add one separate `nb=2` two-frame branch-order check after the
single-branch records validate. The three selected tensors cost about 1.9 MiB
per captured frame for the shipped model, including the two tokens at step 1
(`12` layers are built, but only three are retained); the cap16 raw payload is
therefore about 5.6 MiB before metadata. Stream summary statistics if the
consumer asks for more frames rather than raising the raw-file limit.

Keep calibration and held-out utterances disjoint. The ready matrix is one
English and one Mandarin utterance for calibration and a different English and
Mandarin utterance for held-out validation, each with the same frame positions
`0/7/15`. Add one design voice and one fixed reference voice when a reference
WAV or saved voice is available. The current inspected cache/pod has no
floating master weights or guaranteed reference corpus, so the first run can
be q4_k/design-only; label missing reference coverage and floating-weight
comparison as blockers instead of filling them with generated pseudo-data.

## Forced replay and parity gate

Normal `breeze-cli` generation has no flag to replay a complete code tuple.
The library seam exists: `DepthRunner::run(..., force, n_force)` replaces the
first `n_force` depth samples, and `convert_voice --keep-acoustic` reaches it.
For a complete teacher-forced frame, `keep_acoustic=15` plus `--feed-source`
keeps the source frame in the next backbone feedback step. This is the least
new machinery for a replay check, although it requires source codes/reference
inputs and still runs the surrounding conversion pipeline.

For an ordinary free-running pilot, freeze the step path with
`BREEZE_DD_FUSED=0`, `BREEZE_DEPTH_TOP_K=1`, `--top-k 1`, `--seed 42`, and
`--max-new 16`; dump `DEPTH_CODES` and use fixed 8/8-frame chunks. Run the
same request once with capture disabled and once with capture enabled. Compare
the complete code lines and WAV SHA256. A capture graph that changes codes or
WAV bytes has changed numerical execution through fusion or allocator reuse;
keep its tensors diagnostic only and do not call them baseline-equivalent.
For that case, use forced replay to validate indexing/provenance, and retain
the capture-off run as the quality reference. Also check every code is in
`[0, 2050]`, every record has the expected key exactly once, and all selected
values are finite.

## Exact FFN shapes and dispatch seams

The depth checkpoint has hidden width 1024, FFN width 8192, 12 layers, and
Q4_K block matrices. In ggml logical dimensions the per-layer matrices are:

| tensor | logical GGML shape | depth operation |
| --- | --- | --- |
| `dd.blk.i.ffn_gate.weight` | `[1024, 8192]` Q4_K | `[1024,n_tok] -> [8192,n_tok]` |
| `dd.blk.i.ffn_up.weight` | `[1024, 8192]` Q4_K | `[1024,n_tok] -> [8192,n_tok]` |
| `dd.blk.i.ffn_down.weight` | `[8192, 1024]` Q4_K | `[8192,n_tok] -> [1024,n_tok]` |

The direct seam is `linear()` -> `ggml_mul_mat()` -> Vulkan's quantized
matvec path. On AMD, `ggml_vk_should_use_mmvq` returns false for `n=1` and
`k < 2048`, so single-branch gate/up (`k=1024`) use the ordinary Q4_K
matvec/dequant path while down (`k=8192`) is eligible for MMVQ. For `n>1`
the heuristic selects MMVQ, so CFG-batched gate/up and down have a different
dispatch. `GGML_VK_FORCE_MMVQ` overrides the heuristic and
`GGML_VK_DISABLE_MMVQ` disables it; either changes arithmetic and must remain
diagnostic until a quality and seeded-output gate passes.

The current manager-owned 8-frame Vulkan profile records the expected seams:
Q4_K `m=8192,n=1,k=1024` gate/up and `m=1024,n=1,k=8192` down matvecs. The
profile serializes kernel work for attribution, so its microseconds are not a
wall-time claim. Use it to select the next FFN probe, not to multiply an
activation statistic into a speedup. Any compression experiment must measure
packed bytes and real graph time with capture disabled.

## Available resources and next order

The manager's current pod inventory contains `breeze-tts-2-q4_k.gguf`,
`breeze-tts-2-q8_0.gguf`, and `breeze-tts-2-q4_k-dd2.gguf`; q4_0 is absent.
The local cache has q4_k and a throwaway head-f16 artifact, but no floating
master checkpoint. Use q4_k for the first deployment-shaped capture, q8_0 only
as a control, and do not treat requantizing q4_k or the dd2 quality result as a
clean floating-weight experiment.

Next worker: implement the opt-in capture sinks and cap16 manifest/blob,
validate disabled/enabled parity, then rank FFN sparsity or activation-aware
low-rank only from the held-out records. Defer codec latent capture and any
kernel/product change until that sample is indexed and verified.
