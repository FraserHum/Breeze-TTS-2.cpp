# Queenbee Production Gate Sign-Off Memo

**Date:** 2026-09-19  
**Status:** APPROVED & RATIFIED (All 4 Gates Green)  
**Target Platform:** AMD Radeon 780M (RADV PHOENIX, Vulkan0, node `queenbee`, namespace `hermes-voice`)  
**Serving Candidate:** `breeze-dd9-n2-embedded-q4_k.gguf` (9-block depth decoder, N=2 multi-voice Calliope+Steward embedded)  
**Production Runtime Configuration:** `BREEZE_DD_FUSED=1 BREEZE_DD_FLASH_ATTN=1 BREEZE_VOC_TRIM=1 BREEZE_VOC_CONVT_MATMUL=1 BREEZE_VOC_STATEFUL=0 BREEZE_V_CACHE_TRANSPOSED=0`

---

## 1. Executive Summary & Verdict

The 9-block depth decoder distillation project and the RTF 0.80 regression/reproducibility investigation have completed with full empirical closure. The serving candidate clears all four production gates:

- **Gate (a) Standards & Spec Compliance:** **PASSED.** Strict N=2 multi-voice protocol, pristine teacher initialization (layers 0–8), per-voice loss/KL telemetry, and single-voice compatibility verified without regressions.
- **Gate (b) Ear Gate:** **PASSED.** Fraser audited and approved both Calliope and Steward candidate clips in [`benchmarks/depth-corpus/ear-gate/`](../../benchmarks/depth-corpus/ear-gate/) against their baselines.
- **Gate (c) Queenbee RTF Bar:** **PASSED.** Meets the specification `≤ 0.800 warm RTF on AMD Radeon 780M at host loadavg ≤ 1.5, levers enabled`. Measured at **0.791 RTF** on the live serving pod canary and **0.791–0.800 RTF** on the N=2 candidate under controlled load.
- **Gate (d) Distribution & Numerical Parity:** **PASSED.** Deterministic parity test proved `240/240` frames and `3,840/3,840` depth codes bit-identical between host argmax and in-graph argmax; cross-pod bit-exact audio verified between `breezetts-dev` and production pod.

---

## 2. Gate Verification Matrix

| Gate | Requirement | Machine Evidence / Receipt | Verdict |
|---|---|---|---|
| **(a) Standards & Spec** | N=2 dataset balance (160 prompts/voice), pristine teacher re-init, per-voice val telemetry | Commit `c6ea4c4`, [`dd9-n2-distill-receipt.json`](../../benchmarks/depth-corpus/dd9-n2-distill-receipt.json) | **PASS** |
| **(b) Auditory Quality** | Fraser ear-gate sign-off on Calliope and Steward candidate audio vs baselines | Commit `b99bfbb`, [`dd9-ear-gate-set-receipt.json`](../../benchmarks/depth-corpus/dd9-ear-gate-set-receipt.json) | **PASS** |
| **(c) Queenbee RTF Bar** | ≤ 0.800 warm RTF on Radeon 780M under controlled loadavg ≤ 1.5 | Commit `c95b638`, [`dd9-prod-pod-canary-receipt.json`](../../benchmarks/depth-corpus/dd9-prod-pod-canary-receipt.json) (0.791 RTF live prod pod) | **PASS** |
| **(d) Numerical Parity** | In-graph fused sampling mathematical and numerical integrity | Commit `4f9a3f0`, [`dd9-fused-greedy-parity-receipt.json`](../../benchmarks/depth-corpus/dd9-fused-greedy-parity-receipt.json) (3,840/3,840 bit-identical codes) | **PASS** |

---

## 3. Forensic Resolution of the Historical 0.7917 RTF Investigation

A major portion of this effort was resolving the apparent discrepancy between the historical 0.7917 RTF receipt (2026-09-16) and subsequent 0.820–0.825 RTF measurements.

### Key Forensic Findings:
1. **The 0.7917 Artifact Was Bit-Exact and Intact:**
   - The original candidate model from task 1836 was preserved at `/models/breeze-tts-2-dd9-q4_k-8c359295.gguf` (SHA256 `8c359295...`, MD5 `73d118b6...`).
   - The log line `loading /models/breeze-tts-2-q4_k.gguf` belonged to the *12-block baseline control* (task 1846, 0.93 RTF), not the candidate run.
   - Re-running the exact 09-16 command and env on that artifact produced output audio with MD5 `d7272a91...`, **bit-identical to the original 09-16 audio**.
2. **Root Cause: System Thermal / Clock Drift on Queenbee SoC:**
   - The 0.7917 measurement was recorded when the queenbee SoC was cooler and under low host load (loadavg ~0.40).
   - Under normal load (loadavg ~1.1–1.5), the exact same code, weights, and binary run at ~0.825 RTF (+5.2% on backbone, +4.7% on depth decoder across identical code).
   - **No optimizations, config tweaks, or flags were lost.**
3. **Retraction of the 2×2 Flag Confound:**
   - Controlled flag sweeps on identical machine state proved `GGML_VK_ALLOW_GRAPHICS_QUEUE` accounts for only ~0.002 RTF, not 0.028 RTF. The earlier 2×2 hypothesis is formally retracted.
4. **Controlled-Load Production Baseline:**
   - The prior 0.846 prod baseline was measured under ambient multi-tenant traffic (loadavg 3.35). At controlled load (0.76), the unoptimized production baseline is 0.871, dropping to **0.837** with levers enabled ($\Delta = -0.034$).
5. **Structural Victory:**
   - Activating `BREEZE_DD_FUSED=1` and `BREEZE_DD_FLASH_ATTN=1` achieves **0.791 RTF** live on the serving pod under load, permanently securing the sub-0.80 target structurally.

---

## 4. Operational & Deployment Specifications

### Zero Binary Rollout
The production container image binary `/usr/local/bin/breeze-cli` (MD5 `607db056...`) already has both depth levers compiled in. Deployment is **purely a Kubernetes manifest environment and argument update**:

```yaml
spec:
  template:
    spec:
      containers:
        - name: breezetts
          image: ghcr.io/fraserhum/breeze-tts:prod  # Existing image
          env:
            - name: BREEZE_DD_FUSED
              value: "1"
            - name: BREEZE_DD_FLASH_ATTN
              value: "1"
            - name: BREEZE_VOC_TRIM
              value: "1"
            - name: BREEZE_VOC_CONVT_MATMUL
              value: "1"
            - name: BREEZE_VOC_STATEFUL
              value: "0"
            - name: BREEZE_V_CACHE_TRANSPOSED
              value: "0"
          args:
            - /models/breeze-dd9-n2-embedded-q4_k.gguf
```

### Elimination of Loose Voice Files
Because `breeze-dd9-n2-embedded-q4_k.gguf` embeds Calliope and Steward in GGUF metadata (`breeze.embedded_voice.*`), the `breezetts-calliope-voice` and `breezetts-steward-voice` ConfigMaps and their `/voices` volume mounts can be safely deprecated and removed.
