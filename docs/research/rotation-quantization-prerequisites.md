# Rotation quantization prerequisites for the Breeze depth decoder

Research date: 2026-09-06. This note records the original depth checkpoint
receipt and the smallest quality experiment that can follow it. It does not
claim a quantized model, speech-quality result, or runtime improvement.

## Original-master receipt

The source is the public, immutable
[BreezeBlue/Breeze-TTS-2 revision `799624c0b4a1daa8db6d28bbd9850043c0270734`](https://huggingface.co/BreezeBlue/Breeze-TTS-2/tree/799624c0b4a1daa8db6d28bbd9850043c0270734).
The pinned `model.safetensors.index.json` maps all 112 `depth_decoder.*`
tensors to `model-00001-of-00002.safetensors`; no credentials or signed
redirect URL are stored here.

| item | verified value |
|---|---:|
| index SHA256 | `19977e3d96bb502ed3165b48adff094f6bd71ec3d349940d1a3ba6a0060bd5e3` |
| shard size | 4,961,989,890 bytes |
| safetensors JSON length | 101,880 bytes (101,888 including the 8-byte length field) |
| data section base | byte 101,888 |
| depth tensor count | 112 |
| depth dtype | BF16 for all 112 tensors |
| depth tensor bytes | 868,560,896 |
| selected absolute shard range | `[3,011,224,834, 3,879,785,730)` |
| cache range SHA256 | `6d69058ab262ea751881d263379afaf76161c10d3c2bf9465bca82dda4fdc40a` |

The selected range is contiguous in the shard, so one exact HTTP range request
avoids downloading the 6,966,413,058-byte two-shard model. The cache is
`~/.cache/breeze-models/depth-master/depth-range.bin`; each manifest entry
records its absolute safetensors offsets, local offset, shape, byte count and
SHA256. The BF16 scan rejects every exponent-255 value, covering both NaNs and
infinities. The full per-tensor receipt is in
[`benchmarks/depth-corpus/master-manifest.json`](../../benchmarks/depth-corpus/master-manifest.json).

Reproduce or verify the acquisition with:

```sh
python3 tools/fetch_depth_master.py --download
python3 tools/fetch_depth_master.py --verify
```

`--download` uses `curl`, validates the final HTTP `Content-Range` and
`Content-Length`, writes the range through a `.part` file, and atomically
renames it. A complete manifest/cache is verified and reused on later runs.
`--self-test` runs the offline negative range/header checks. This receipt does
not include a full conversion or any GPU work.

The depth inventory is four shared tensors (`codebooks_head`, `embed_tokens`,
`inputs_embeds_projector`, and final `norm`) plus nine tensors in each of 12
layers. The FFN tensors needed for the first quality comparison are:

| tensor family | shape | count | bytes each |
|---|---:|---:|---:|
| `mlp.down_proj.weight` | `[1024, 8192]` | 12 | 16,777,216 |
| `mlp.gate_proj.weight` | `[8192, 1024]` | 12 | 16,777,216 |
| `mlp.up_proj.weight` | `[8192, 1024]` | 12 | 16,777,216 |

The remaining per-layer tensors are attention projections and two 1024-element
norms. The shapes and offsets come from the pinned safetensors header, not from
the existing Q4_K file.

## Existing quantization baseline

[`apps/quantize/main.cpp`](../../apps/quantize/main.cpp) can select `q3_k` and
can apply a separate type to tensors named `dd.*` with `--depth`. Its call to
`ggml_quantize_chunk(..., nullptr)` passes no importance matrix, so this is an
ordinary uncalibrated Q3_K baseline. It is useful as a control, but it is not
the calibrated or rotation-aware method required by the next study. The
current C++ path also consumes GGUF/GGML formats; QTIP's trellis codebooks and
CUDA kernels are a separate format/runtime choice.

## Calibration coverage and exact gaps

The existing bounded corpus has two calibration utterances (`cal-en` and
`cal-zh`) and two held-out utterances (`eval-en` and `eval-zh`). Each uses one
branch, seed 42, 16 generated frames, and selected frames `0, 7, 15`; the FFN
records cover all 15 depth steps, but only layers `0, 5, 11`. Step 1 contributes
two token rows and steps 2--15 one each, giving exactly 48 rows per utterance
and 96 calibration rows per selected layer.

For a full-depth rotated study, the missing coverage is explicit:

- layers `1, 2, 3, 4, 6, 7, 8, 9, 10` have no calibration or held-out FFN rows;
- within the first 16-frame captures, frames `1--6` and `8--14` are absent,
  and there are no frames at or beyond 16 from a long utterance;
- there is one calibration utterance per language, with no reference-voice
  variation, additional voices, prompt families, or duration buckets;
- the main corpus is one branch only. The separate two-branch probe covers
  frame 0 only and is not a full calibration split;
- downstream codec latents, backbone feedback, and free-running speech gates
  are deferred.

Thus the selected layers have complete step coverage but sparse layer/frame
coverage. Reusing the 96-row pilot as if it represented all 12 layers would
make a rotation or rounding result an extrapolation.

## QTIP and YAQA compatibility check

[QTIP](https://github.com/Cornell-RelaxML/qtip) is an HF/PyTorch LLM
quantizer. Its primary README describes the `lib/codebook/bitshift.py`
trellis quantizer, a Llama-oriented `quantize_llama` flow, and separately built
`qtip-kernels` CUDA extensions for 2/3/4-bit batch-one matvecs. The published
[requirements](https://raw.githubusercontent.com/Cornell-RelaxML/qtip/main/requirements.txt)
include CUDA Python 12.6, PyTorch 2.4, `fast_hadamard_transform`, Transformers,
and other packages. The primary quantization script loads
`AutoModelForCausalLM` and uses `torch.cuda.device_count`, then quantizes the
Llama attention and MLP order. It has no Breeze depth-module mapping or Vulkan
backend, so it is an offline algorithm reference rather than a drop-in for the
GGUF/Q3_K runtime.

[YAQA](https://github.com/Cornell-RelaxML/yaqa-quantization) is a QTIP-derived
adaptive-rounding preparation path. Its README says to collect Hessians with
`hessian_llama` and otherwise follow QTIP. The primary
[Hessian instructions](https://raw.githubusercontent.com/Cornell-RelaxML/yaqa-quantization/main/hessian_llama/README.md)
recommend Sketch B (`ctx_size 2048`, `n_seqs 65536`) and show `torchrun` on
eight GPUs; they report fitting models below roughly 20B on one 8x80G node.
The current local environment has MPS rather than CUDA and lacks the QTIP
fast-Hadamard/safetensors stack. YAQA's published requirements also pull in a
large CUDA/PyTorch ecosystem. Installing or running that stack was therefore
not attempted. Those published multi-GPU figures are not a Breeze resource
estimate: direct execution is blocked here by the lack of CUDA, while resource
sizing for a bounded Breeze depth Hessian remains unresolved. The 869 MB
(about 828 MiB) depth master cache fits the available local disk.

## Smallest next quality experiment

First extend the capture recipe to a small disjoint set that covers all 12
layers, the existing 15 steps, several frame positions including a longer
utterance, and at least one extra voice/language case. Then run a tensor-level
screen on the three FFN families for layers `0, 5, 11` (expand to all layers
only after the screen is numerically sound), using the BF16 master from the
receipt as the reference. Compare:

1. the existing Q4_K dequantized control;
2. ordinary calibrated 3-bit rounding; and
3. a fixed Hadamard-rotated 3-bit variant, including the inverse activation
   transform in the error calculation.

Use calibration rows only to choose scales/rounding and held-out rows to report
per-layer, per-step output error. If that screen passes a predeclared error
gate, run one short free-running English/Mandarin generation before considering
YAQA/Hessian preparation or a packed Vulkan kernel. This is the smallest
quality-first experiment that tests the rotation premise without claiming
runtime speed. No result should be called deployment-ready until it includes
the missing coverage and free-running speech checks.

Primary method sources: [QTIP paper](https://arxiv.org/abs/2406.11235),
[QTIP source](https://github.com/Cornell-RelaxML/qtip), and
[YAQA paper](https://arxiv.org/abs/2505.22988).
