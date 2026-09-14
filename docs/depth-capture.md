# Depth activation capture

`BREEZE_DEPTH_CAPTURE=<empty-directory>` enables the bounded diagnostic probe in
the regular depth decoder. It refuses `BREEZE_DD_FUSED=1` and refuses to reuse a
directory, so each utterance or repeat needs a fresh output directory. With a
Vulkan backend, set `GGML_VK_DISABLE_FUSION=1` before starting the process;
capture refuses to run when that presence flag is missing. The probe retains
layers `0,5,11`, all 15 depth steps, and frames `0,7,15` (up to the first 16
depth frames). Each selected tensor is written as little-endian F32 in
`activations.f32`; `metadata.jsonl` records its frame, step, layer, shape,
branch/token layout, byte offset, and dtype. Host backbone hiddens and complete
`cb0..cb15` frame codes are indexed there as well.

Build and check locally:

```sh
cmake --build build-cpu --target breeze-cli -j4
python3 tools/depth_capture_check.py --self-test
git diff --check
```

Run one CPU probe with a fresh directory and seeded sampling:

```sh
capture_dir=$(mktemp -d /tmp/breeze-depth-cap.XXXXXX)
BREEZE_DD_FUSED=0 GGML_VK_DISABLE_FUSION=1 BREEZE_DEPTH_CAPTURE="$capture_dir" \
  build-cpu/breeze-cli ~/.cache/breeze-models/breeze-tts-2-q4_k.gguf \
  --cpu --seed 42 --max-new 16 --text 'Capture a short sample.' \
  --output "${capture_dir}.wav"
python3 tools/depth_capture_check.py --full "$capture_dir"
```

Capture changes allocator output lifetimes and can change fused down/residual
dispatch. Compare codes and WAV hashes with capture disabled before using the
values for compression measurements. Forced replay remains a deferred fallback
for indexing checks if a future backend cannot pass the parity gate.
