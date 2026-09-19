# Breeze-TTS Benchmarking & Telemetry Protocol

**Purpose:** Ensure all performance and latency claims in Breeze-TTS are rigorous, receipt-backed, reproducible, and immune to environmental confounding.

---

## 1. Golden Rules of Benchmarking

1. **Rule 1 (Target Hardware Verification):** No performance claim is valid or load-bearing without saved receipts from the target hardware (`breezetts-dev` or designated pod).
2. **Rule 2 (Ear Gate):** Throughput gains that degrade audio quality are invalid. Every model or sampling path change must pass auditory ratification on canonical voices before acceptance.
3. **Rule 3 (Full Provenance Invariant):** An unrecorded candidate run is an invalid run. A receipt that omits the exact environment flags or binary hash cannot be used as an anchor.

---

## 2. Mandatory Receipt Schema

Every JSON receipt committed under `benchmarks/` must include the following required fields:

```json
{
  "timestamp": "YYYY-MM-DDTHH:MM:SSZ",
  "purpose": "Precise description of the hypothesis or test",
  "target": {
    "host": "Node name (e.g. queenbee)",
    "pod": "Pod name and namespace (e.g. breezetts-dev in hermes-voice)",
    "gpu": "Exact Vulkan device string from ggml_vulkan log",
    "driver": "Driver name/version if readable"
  },
  "build": {
    "binary_path": "/src/build/breeze-cli",
    "binary_md5": "<live md5sum of binary>",
    "source_commit": "<git rev-parse HEAD>",
    "binary_mtime": "Timestamp of binary compilation"
  },
  "artifact": {
    "path": "/models/<model-name>.gguf",
    "size_bytes": 123456789,
    "md5": "<md5sum>",
    "sha256": "<sha256sum>"
  },
  "env": "EXACT command-line env string (e.g. BREEZE_VOC_TRIM=1 BREEZE_DD_FUSED=1 ...)",
  "host_state": {
    "loadavg_before": "1m 5m 15m load average",
    "loadavg_after": "1m 5m 15m load average",
    "uptime": "Host uptime string"
  },
  "command": "Exact argv string passed to breeze-cli",
  "runs": [
    {
      "repeat": 1,
      "warm": false,
      "wall_ms": 14000.0,
      "wall_rtf": 0.795,
      "frames": 220,
      "audio_s": 17.60,
      "backbone_ms_frame": 16.8,
      "depth_ms_frame": 37.5,
      "vocoder_ms_frame": 8.2,
      "audio_md5": "<md5sum of output WAV>"
    }
  ],
  "warm_mean": {
    "wall_rtf": 0.792,
    "backbone_ms_frame": 16.8,
    "depth_ms_frame": 37.5,
    "vocoder_ms_frame": 8.2,
    "total_compute_ms_frame": 62.5
  },
  "verdict": "PASS / FAIL against criteria"
}
```

---

## 3. Controlling for Environmental Drift

1. **Host Load & Thermal Regime:**
   - Shared SoCs (like the AMD Ryzen 7 8845HS / Radeon 780M) experience clock throttling and memory bandwidth contention under CPU/GPU load.
   - A benchmark measured at loadavg 0.40 cannot be directly compared to a benchmark measured at loadavg 3.0.
   - **Protocol:** Standardize gating measurements to **loadavg ≤ 1.5** on `breezetts-dev`. Always record loadavg immediately before and after execution.
2. **Controlled Dev Pod vs. Multi-Tenant Serving Pod:**
   - **Dev Pod (`breezetts-dev`):** The primary controlled test bench. Single-tenant, isolated execution, no incoming HTTP requests. Used for architectural gating.
   - **Production Pod (`breezetts`):** The operational canary. Carries ambient multi-tenant serving load and cluster background tasks. Used for deployment verification (validating latency reduction under live conditions), not for baseline gating.
3. **Audio Invariant Check:**
   - When running identical weights, prompt, seed, and environment across sessions, the output audio MD5 **must be bit-identical**.
   - If audio MD5 differs, do not assume identical execution: verify whether sampling flags (such as `BREEZE_DD_FUSED` or `BREEZE_DD_FLASH_ATTN`) were toggled.

---

## 4. Production Levers Reference

For AMD Radeon 780M / Vulkan deployments:

| Flag | Default | Production Value | Effect |
|---|---|---|---|
| `BREEZE_DD_FUSED` | `0` | `1` | Collapses 15 depth graph submits into 1 in-graph submit; eliminates submit overhead (~1.3 ms/f savings). |
| `BREEZE_DD_FLASH_ATTN` | `0` | `1` | Uses `ggml_flash_attn_ext` with f16 mask; halves attention mask memory bandwidth (~1.1 ms/f savings). |
| `BREEZE_VOC_TRIM` | `1` | `1` | Retains only causal vocoder convolution history. |
| `BREEZE_VOC_CONVT_MATMUL` | `1` | `1` | Replaces transposed conv with optimized GEMM. |
| `BREEZE_VOC_STATEFUL` | `0` | `0` | Keep 0 until streaming state cache passes waveform equivalence. |
| `BREEZE_V_CACHE_TRANSPOSED` | `0` | `0` | Standard contiguous V-cache layout. |
