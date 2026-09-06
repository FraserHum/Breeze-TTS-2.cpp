# HIP / gfx1103 build feasibility

Status: **build succeeds and native HIP runs with the kpack archive-path fix;
the current 780M deployment path fails the performance gate**.

The isolated NAS build produced `breeze-cli` with the HIP backend and native
gfx1103 offload target. It used root commit
`0ed7de69817e3b3380664c6d689606fb83541220` (the CLI timing commit) and ggml
submodule `e7d634a0a457d37eb4d3b70a90d3cdf19bcd306d`.

## Build

Host: `ansible@192.168.1.196` (`magicbox`), rootless Podman, no host package or
driver changes, and no `/dev/kfd` or GPU execution.

The existing pinned `/home/ansible/.cache/beehive-build/speaches-rocm/Containerfile`
was used for the first `ctranslate2-builder` attempt, tagged only as
`localhost/breeze-hip-sdk-080`. TheRock 7.14.0 download and CTranslate2
gfx1103 build passed:

```text
SDK image: localhost/breeze-hip-sdk-080
SDK image ID: 5642da07ed547033b21ac838f2e0c2f47eda41ae52ba323c89a5eb0d756277b4
SDK BUILD_RC=0
```

Breeze was configured in the SDK container with:

```text
BREEZE_VULKAN=OFF
BREEZE_CUDA=OFF
BREEZE_BUILD_CLI=ON
BREEZE_BUILD_SERVER=OFF
BREEZE_BUILD_SHARED=OFF
GGML_HIP=ON
GGML_NATIVE=OFF
GGML_BACKEND_DL=OFF
GGML_BUILD_TESTS=OFF
GGML_BUILD_EXAMPLES=OFF
CMAKE_C_COMPILER=/opt/rocm/bin/amdclang
CMAKE_CXX_COMPILER=/opt/rocm/bin/amdclang++
CMAKE_HIP_COMPILER=/opt/rocm/bin/amdclang++
CMAKE_HIP_ARCHITECTURES=gfx1103
```

The successful target command was:

```bash
cmake -S /src -B /src/build-hip-make \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=/opt/rocm/bin/amdclang \
  -DCMAKE_CXX_COMPILER=/opt/rocm/bin/amdclang++ \
  -DCMAKE_HIP_COMPILER=/opt/rocm/bin/amdclang++ \
  -DCMAKE_PREFIX_PATH=/opt/rocm \
  -DCMAKE_HIP_ARCHITECTURES=gfx1103 \
  -DBREEZE_VULKAN=OFF -DBREEZE_CUDA=OFF \
  -DBREEZE_BUILD_CLI=ON -DBREEZE_BUILD_SERVER=OFF -DBREEZE_BUILD_SHARED=OFF \
  -DGGML_HIP=ON -DGGML_NATIVE=OFF -DGGML_BACKEND_DL=OFF \
  -DGGML_BUILD_TESTS=OFF -DGGML_BUILD_EXAMPLES=OFF -DBUILD_SHARED_LIBS=OFF
cmake --build /src/build-hip-make --target breeze-cli --parallel 4
```

The SDK build ran from 12:42:15 to 12:51:36 NZST. The clean Breeze build ran
from 12:55:45 to 13:00:00 NZST and finished with `BUILD_RC=0`. The output is:

```text
/home/ansible/.cache/beehive-build/breeze-hip-080/source/build-hip-make/breeze-cli
size: 67,906,216 bytes
sha256: 4243ea40ed1cda245d0df49381c00065959dd8583e55b3503f49b454a95b170f
```

## Checks

`llvm-objdump --offloading` reports exactly:

```text
hipv4-amdgcn-amd-amdhsa--gfx1103
```

The executable is statically linked against ggml backends (`GGML_BACKEND_DL=OFF`)
and its symbol table contains `ggml_backend_cuda_reg`,
`ggml_backend_cpu_reg`, and `ggml_backend_init_by_type`. ggml's registry calls
the CUDA-named registration function when `GGML_USE_CUDA` is defined; ggml HIP
defines that macro for the static backend path. This is the expected HIP
registration path and retains CPU fallback.

The executable needs `libomp.so`, `libhipblas.so.3`, `librocblas.so.5`,
`libamdhip64.so.7`, and their ROCm transitive closure. The existing cached
runtime image resolves all of them and contains native gfx1103 data:

```text
192.168.1.196:5000/speaches-rocm:0.8.3-therock7.14.0-ct2.4.8.1-gfx1103-beehive.6
LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/lib/llvm/lib
/opt/rocm/lib/rocblas/library/Kernels.so-000-gfx1103.hsaco
```

No full SDK transfer is needed when the devpod uses that runtime image. The
runtime `ldd` evidence is saved at
`/home/ansible/.cache/beehive-build/breeze-hip-080/artifacts/runtime-image-ldd.txt`.

## Transfer

The smallest safe bundle is the CLI plus checksum and build metadata:

```text
NAS:   /home/ansible/.cache/beehive-build/breeze-hip-080/artifacts/breeze-cli-hip-gfx1103.tar.gz
SHA256: f1103662a20aa15e83bd0dc4725c51101d89ce49826849883980acfaa650a9de
Local: .beehive/agent/BREEZE-RTF-080-IMPL/hip/artifacts/breeze-cli-hip-gfx1103.tar.gz
```

For a devpod run, extract the bundle and use the runtime image above with
`LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/lib/llvm/lib`; the kpack archive path
must also be set as shown in the smoke section below. The manager's initial
smoke passed. The completed three-repeat performance result is recorded below;
quality remains unvalidated.

## Issues encountered

The SDK image did not contain Ninja, so the first configure's stale Ninja
cache was abandoned in favor of a fresh Unix Makefiles build directory. The
first source transfer also preserved macOS AppleDouble `._*` files; HIP then
tried to compile `._acc.cu`. Recreating the source directly from Git archives
removed all 2,406 metadata sidecars; the clean rebuild passed. No product
source or existing image was changed.

## Portable runtime and current-source refresh

The devpod does not contain `/opt/rocm`, so the runtime closure was exported
from the pinned cached image without exporting the SDK. The image's complete
`/opt/rocm` is 749,956,589 bytes (718 MiB) and contains only runtime libraries
and accelerator data under `/opt/rocm/lib`; the gzip archive is 216,199,561
bytes (207 MiB). The archive preserves tar symlinks and includes 249 `gfx1103`
entries, including:

```text
rocm/lib/hipblaslt/library/gfx1103/Kernels.so-000-gfx1103.hsaco
rocm/lib/rocblas/library/*gfx1103*
```

The cached image currently has zero symbolic links under `/opt/rocm` because
its runtime files are already flattened to their versioned names. No SDK
headers, compilers, or build tools are in this closure.

```text
NAS:   /home/ansible/.cache/beehive-build/breeze-hip-080/artifacts/rocm-runtime-gfx1103.tar.gz
SHA256: 3f70eaa596e44fc13c076b75f03b4e4e5f62a8688ab347eb3698dc5c750bf6f8
Local: /Users/fraser/workspace/github.com/fraserhum/Breeze-TTS-2.cpp/.beehive/agent/BREEZE-RTF-080-IMPL/hip/artifacts/rocm-runtime-gfx1103.tar.gz
```

Extract the archive under a writable staging directory; it creates
`<stage>/rocm/lib`:

```bash
mkdir -p "$stage"
tar --no-same-owner -xzf rocm-runtime-gfx1103.tar.gz -C "$stage"
export ROCM_ROOT="$stage/rocm"
export LD_LIBRARY_PATH="$ROCM_ROOT/lib:$ROCM_ROOT/lib/llvm/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export ROCM_PATH="$ROCM_ROOT"
export HIP_PATH="$ROCM_ROOT"
export HIPBLASLT_TENSILE_LIBPATH="$ROCM_ROOT/lib/hipblaslt/library"
```

Keeping the `rocm/lib/{rocblas,hipblaslt}/library` layout intact lets the
versioned libraries find their native data after relocation. The optional
`HIPBLASLT_TENSILE_LIBPATH` makes the hipBLASLt data location explicit.

The source was then refreshed from clean Git archives at Breeze
`e0c8f9d250c8dc05a5ee6c806db9679f6ef60770` with ggml still pinned to
`e7d634a0a457d37eb4d3b70a90d3cdf19bcd306d`; no AppleDouble files were present.
The existing `build-hip-make` directory and SDK image were reused for an
incremental `breeze-cli` rebuild (`BUILD_RC=0`). The refreshed binary is
67,911,104 bytes with SHA256
`9000a72ddc58cc09c8ebfa9cbb174ae67fb327b0a33f411e1f526354207e4421`; its
offload bundle still targets `gfx1103`.

```text
NAS:   /home/ansible/.cache/beehive-build/breeze-hip-080/artifacts/breeze-cli-hip-gfx1103-e0c8f9d.tar.gz
SHA256: 0f7c693a33c881601834b31e1da268890fd8fdb3c4ad55aaa12b58abc5328353
Local: /Users/fraser/workspace/github.com/fraserhum/Breeze-TTS-2.cpp/.beehive/agent/BREEZE-RTF-080-IMPL/hip/artifacts/breeze-cli-hip-gfx1103-e0c8f9d.tar.gz
```

The updated bundle contains the binary, checksum, build metadata, runtime
`ldd` output, and offload verification. The manager's smoke result and the
remaining 780M validation work are recorded below.

## Smoke failure diagnosis and resolved runtime packaging

The first devpod smoke reached the native 780M device and initialized the
depth decoder, then failed at the first rocBLAS `hipblasGemmBatchedEx` with
`hipErrorInvalidImage` (200). The retry with the e0 binary and
`ROCM_PATH`, `HIP_PATH`, and `ROCM_ROOT` set failed at the same call, so the
top-level ROCm path variables alone do not fix it. No GPU execution was done
by this build task.

The runtime image's own `Containerfile` copies the ELF library closure and the
`rocblas`/`hipblaslt` data directories, but not the SDK's hidden top-level
`.kpack` directory. The SDK contains this gfx1103 package:

```text
/opt/rocm/.kpack/blas_lib_gfx1103.kpack
size: 42546210 bytes
sha256: 9861590fa3c94439428fa27ec3ab5133f35767bfa4f3687d49a243c6cac56fc7
```

The cached runtime image has no `.kpack` directory. Both `librocblas.so.5`
and `libhipblaslt.so.1` contain the relative
`../.kpack/blas_lib_@GFXARCH@.kpack` lookup, while `librocm_kpack.so.0`
exports the `ROCM_KPACK_PATH` and `ROCM_KPACK_PATH_PREFIX` controls. A sample
`Kernels.so-000-gfx1103.hsaco` in the runtime data has ELF metadata target
`amdgcn-amd-amdhsa--gfx1103`.

The bounded additive probe contains only the missing gfx1103 kpack:

```text
NAS:   /home/ansible/.cache/beehive-build/breeze-hip-080/artifacts/rocm-kpack-gfx1103.tar.gz
SHA256: 13071896e02e4c047ab4819c46f7b0128eecaed38c18dce94d731d4002774121
Local: /Users/fraser/workspace/github.com/fraserhum/Breeze-TTS-2.cpp/.beehive/agent/BREEZE-RTF-080-IMPL/hip/artifacts/rocm-kpack-gfx1103.tar.gz
```

Apply it to the already extracted runtime stage for one controlled retry:

```bash
tar --no-same-owner -xzf rocm-kpack-gfx1103.tar.gz -C /tmp/breeze-hip-080
export ROCM_ROOT=/tmp/breeze-hip-080/rocm
export ROCM_KPACK_PATH="$ROCM_ROOT/.kpack"
export LD_LIBRARY_PATH="$ROCM_ROOT/lib:$ROCM_ROOT/lib/llvm/lib"
```

This creates only `/tmp/breeze-hip-080/rocm/.kpack/blas_lib_gfx1103.kpack`;
the existing libraries and native data remain unchanged.

That additive retry still failed with `hipErrorInvalidImage` at the same
rocBLAS call because the override was set to the `.kpack` directory. The final
`AMD_LOG_LEVEL=4` trace reports
`hip_fatbin.cpp:751: kpack_load_code_object failed with error: 13`. The
installed TheRock header defines kpack error 13 as
`KPACK_ERROR_ARCHIVE_NOT_FOUND`, so this was an archive lookup failure rather
than evidence that the gfx1103 code object was invalid. The archive is a valid
`KPAK 01` file and its table contains the matching rocBLAS/hipBLASLt library
names and gfx1103 entries.

The ROCm kpack loader treats each `ROCM_KPACK_PATH` entry as a direct archive
filename, not as a directory. The kpack reference metadata also resolves the
relative path to `$ROCM_ROOT/.kpack/blas_lib_gfx1103.kpack` when the library is
at `$ROCM_ROOT/lib/librocblas.so.5`. Passing that exact archive filename fixed
the runtime lookup:

```bash
export ROCM_KPACK_PATH="$ROCM_ROOT/.kpack/blas_lib_gfx1103.kpack"
export ROCM_KPACK_DEBUG=1
```

The successful manager-side smoke log records the archive opening and code
object load:

```text
kpack: opened and cached archive: /tmp/breeze-hip-080/rocm/.kpack/blas_lib_gfx1103.kpack
kpack:   found kernel: 3599480 bytes
kpack: loaded code object: 3599480 bytes
```

The 16-frame smoke exited 0 and produced 1.28 seconds of audio. It measured
1,918.7 ms generation wall time (wall RTF 1.499), including warmup, with
79.98 ms/frame depth decode and 10.91 ms/frame vocoder. The log is saved at
`.beehive/agent/BREEZE-RTF-080-IMPL/hip-smoke-kpack-file.log`; no GPU run was
performed by this build task.

## Full 780M performance gate

The manager then ran the refreshed e0c8f9d HIP binary three times with the
same seeded request and q4_k model. All three runs completed successfully,
produced 251 frames / 20.08 seconds of audio, and had the same output SHA256
`b10c70c500bb53e01fa0ddf8e2b88c4af0007e477f67e9e890c49e8a0d9e6fa5`:

```text
run              1             2             3          mean
wall RTF     1.197545      1.173411      1.178048      1.183001
wall ms     24046.7       23562.1       23655.2       23754.7
first audio  4053          3675          3653          3793.7 ms
deficit ms   2742.3        2691.7        2826.1        2753.4
depth ms/f     66.18         65.51         65.78         65.82
```

The current Vulkan deployment-path comparator is the three-run
`large-wg-control` result: 213 frames / 17.04 seconds, wall RTF
`0.959671`, `0.950939`, and `0.953938` (mean `0.954849`), with no playback
deficit. HIP is therefore `0.228152` wall-RTF points slower on this Radeon
780M configuration, about 24% slower by mean wall RTF, and misses the
real-time playback gate with deficits in every repeat. This rejects HIP as the
current deployment path on this GPU; it does not establish that HIP is slower
on other GPUs or configurations.

The raw three-repeat record, exact command, binary, runtime closure, kpack
archive, and environment paths are in
`benchmarks/rtf-080-hip.json`. Cross-engine waveform quality was not scored:
the HIP and Vulkan runs have different output durations and frame counts, so a
PESQ comparison between them would not be valid. No new quality score is
claimed for this HIP candidate.
