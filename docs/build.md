# Building

## Requirements

| Tool | Notes |
| --- | --- |
| CMake | 3.19 or newer |
| C++17 compiler | mingw-w64 (winlibs UCRT), MSVC, gcc or clang |
| Vulkan SDK | Only for the GPU backend. Provides `glslc` for shader compilation. |
| Ninja | Optional but much faster than the default generator |
| Python 3.9+ | Only for converting weights to GGUF |

The Vulkan backend works on any driver exposing Vulkan 1.2, which covers
NVIDIA, AMD and Intel on both Windows and Linux.

## Clone

The ggml backend is a submodule, so clone recursively:

```
git clone --recursive https://github.com/<owner>/Breeze-TTS-2.cpp
cd Breeze-TTS-2.cpp
```

If you already cloned without `--recursive`:

```
git submodule update --init --recursive
```

## Configure and build

GPU (Vulkan):

```
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

CPU only:

```
cmake -B build-cpu -G Ninja -DCMAKE_BUILD_TYPE=Release -DBREEZE_VULKAN=OFF
cmake --build build-cpu -j
```

## CMake options

| Option | Default | Effect |
| --- | --- | --- |
| `BREEZE_VULKAN` | `ON` | Build ggml with the Vulkan backend. |
| `BREEZE_BUILD_CLI` | `ON` | Build `breeze-cli`. |
| `BREEZE_BUILD_SERVER` | `ON` | Build `breeze-server`. |
| `BREEZE_BUILD_SHARED` | `ON` | Build the shared C library. |

## Outputs

| Target | Description |
| --- | --- |
| `breeze-cli` | One shot synthesis to WAV |
| `breeze-server` | HTTP server with optional web UI |
| `breeze-quantize` | Converts an F16 GGUF to Q8_0, Q6_K or Q4_K |
| `libbreeze` | Shared C library, see [c-api.md](c-api.md) |
| `libbreeze_core` | Static library if you would rather link directly |

The web UI is compiled into `breeze-server` at build time by
`cmake/embed_webui.cmake`, which turns the files in [webui](../webui) into a
generated header. Edit the sources and rebuild to pick up changes.

## Backend selection at runtime

Both apps try Vulkan first and fall back to CPU if no usable device is found.
Pass `--cpu` to skip the GPU entirely. The startup line tells you which one you
got:

```
backend: GPU, sample rate: 24000
```

## mingw specifics

mingw builds link the compiler runtime statically, so the binaries depend only
on system DLLs plus `vulkan-1.dll`. You can copy them to a machine with no
mingw installed and they will run.

OpenMP is disabled for mingw because there is no static `libgomp`, and linking
it dynamically would leak a `libgomp-1.dll` dependency into the shared library.
The ggml CPU backend falls back to its own thread pool, so it still runs
multithreaded.

## Troubleshooting

**`third_party/ggml is missing`** means the submodule was not initialised. Run
`git submodule update --init --recursive`.

**Vulkan shader generation crashes with `0xC0000139`** on Windows when another
program has put an incompatible `libstdc++-6.dll` earlier on `PATH`. Tesseract
OCR is a common culprit. Put your mingw `bin` directory first:

```powershell
$env:PATH = "C:\path\to\mingw64\bin;" + $env:PATH
```

**No Vulkan device found** usually means an outdated GPU driver, or running
inside a container or remote session without GPU passthrough. It is not fatal;
the model loads on CPU instead, just slowly.

**Out of video memory** during generation: use a smaller quantization. Q4_K
needs roughly 3 GB of VRAM against about 7 GB for F16. See
[models.md](models.md).
