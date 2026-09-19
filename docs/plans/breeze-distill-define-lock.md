# Define lock: `breeze-distill` product & validation contract

Date: 2026-09-19. Status: **LOCKED** — sign-off recorded 2026-09-19. Authors: Fraser & Antigravity; collaborator review incorporated. Source brief: [`product_design_brief.md`](/Users/fraser/.gemini/antigravity/brain/a8e9e24f-413b-4167-9072-f80cf85ad7e9/product_design_brief.md).

This document is the convergence record of the Define diamond for the `breeze-distill` work. Code changes are executed against it; the fix list below is the Develop plan. No number below is load-bearing without a receipt.

## Rulings (in order of precedence)

1. **Personal-project ruling (Fraser):** this project primarily exists for the builder's own usage. External/second-customer validation is **out of scope**. The persona below is a **design lens and goal criterion**, not a market hypothesis. Prioritization test for every feature: *"does this make my 1–3-voice always-on deployment more fluid or more self-contained?"*
2. **Levers default (two-signed, collaborator + Fraser):** the C++ runtime defaults are **not flipped**. The sub-0.80 path requires `BREEZE_DD_FUSED=1 BREEZE_DD_FLASH_ATTN=1`; the stock Tier 1 artifact ships with that **documented launch env**, and the env line is printed as a first-class field in the scorecard/CLI output. Engine defaults remain as gated on Queenbee.
3. **Stretch goal (Fraser):** integration with the **audio.cpp UI**. Interface commitment that makes it possible without future redesign: the **GGUF metadata schema** (`breeze.embedded_voice.*`) is the product interface, self-describing, with an audio.cpp-compatible exporter as the planned path. llama.cpp inference compatibility is **not claimed**. Upstream PRs remain deferred until at least one field-deployed Tier 2 artifact exists.

## Primary user model (design lens)

Always-on local assistant builders on integrated-GPU hardware (AMD 780M/880M, mini PCs, handhelds) who need conversational turn-taking and local privacy. Voice set is small and durable: 1–3 voices for months (e.g. Calliope + a household steward). Multi-voice / latency-tolerant use is served by Tier 1 zero-shot and is **not** the Tier 2 target.

## Decoupled two-tier architecture (approved)

- **Tier 1 — stock artifact, speed free at t=0.** `breeze-dd9-stock-q4_k.gguf` is the receipted `breeze-dd9-q4_k.gguf` (md5 `73d118b6effd199e72a96a819b0f1916`), 9-block N=2 student, canary-receipted **0.791–0.800 RTF** on 780M with the documented levers env (without levers: 0.811 student / 0.871 q3). Any-clip via `--ref-audio` zero-shot. No Python, no build.
- **Tier 2 — the voice foundry (`breeze-distill`).** Answers one question: "bake my 1–3 permanent voices into a single self-contained GGUF." One-time deployment-compile event, not an ongoing tax. Output: one GGUF, voices embedded, no sidecars.
- **Lineage rule:** every Tier 2 artifact embeds the **source student checkpoint hash** in GGUF metadata, so a stock-model update cannot silently orphan baked artifacts.

Receipts: `benchmarks/depth-corpus/dd9-prod-pod-canary-receipt.json` (0.791, md5 `73d118b6…`), `dd9-n2-rtf-control-receipt.json` (N=2 control), `dd9-rtf-07917-verification-receipt.json` (12-block baseline control **0.93 RTF**, task 1846).

## Presets (receipt-gated)

| Profile | Blocks | Target | Receipt status |
|---|---|---|---|
| `780m` (default) | 9 | ≤ 0.800 RTF, levers on, load ≤ 1.5 | **Receipted**: 0.791–0.800 (canary + N=2) |
| `baseline` | 12 | ≈ 0.93 RTF | **Receipted**: task 1846 control |
| ~~`edge`~~ | 6 | (was 0.65 — unmeasured) | **CUT from Iteration 1** until a 6-block checkpoint is trained, ear-tested, and canary-benchmarked. Active research milestone only. |

Arbitrary K remains reachable via config; the preset table is a receipt-discipline choice, not a capability cut.

## Validation gate (Tier 2 only)

- **Fail-closed by default:** if the per-voice teacher-agreement floor is not met, the pipeline exits non-zero and refuses quantization/export.
- **`--force` override for experimental work; the override is recorded in the receipt** (`"safety_gate": {"threshold": X, "measured": Y, "status": "forced"}`). A forced artifact is traceably forced, not silently shipped.
- **Threshold is data-derived** from the `top1_acc_teacher` distribution (combined, count-weighted) across a full-corpus distill run — never an arbitrary constant. Smoke-run receipts must carry first-class fields (`"smoke": {"prompts": N, "epochs": N}`) so they can never be cited as full-corpus evidence.
- **No receipt → no PASS.** Missing telemetry renders the scorecard `UNVERIFIED`; it never awards a passing badge on fabricated or absent data.
- The gate binds Tier 2 only. Tier 1 ships as a receipted artifact with its canary receipt attached; there is no runtime gate on a stock download.

## Contract facts verified against the real tools (for the patch; do not rediscover)

- `tools/depth_training/distill.py`: flag is `--output-dir` (not `--out-dir`). Writes **no** receipt today — persistence gap, not a measurement gap. `evaluate()` returns exactly `{loss, kl, ce, top1_acc_codes, top1_acc_teacher, _n_records}`; per-voice metrics and `combine_metrics` are already computed at the best-ckpt save point, and the checkpoint dict already carries `val_metrics`, `per_voice_val`, `n_layer`. `best_student_9block.pt` is written to `--output-dir`.
- `tools/depth_training/export_depth_gguf.py`: takes `--base-gguf --checkpoint --out-f16 --out-q4k --breeze-quantize` and **owns quantization** (`--breeze-quantize`) and **voice embedding** (`--embed-voice NAME FILE`, repeatable; `--default-voice`). There is **no `--n-blocks`** — block count is read from the checkpoint `n_layer`. All defaults point at prod host paths (`/mnt/media/breeze-teacher/...`) and must be overridden.
- `apps/cli/main.cpp`: `--instruction`, `--voice`, `--save-voice <name>` (requires `--ref-audio` + `--ref-text`), `--seed`, `--voices-dir` all exist; the audition real path is verified clean.
- Prototype WIP commit: `cd3a3f8` (local `dev`, unpushed) — status "wiring, unverified" until the empirical proof gate below lands.

## Fix list (Develop plan, in order)

1. **`distill.py` receipt persistence:** write `distill_receipt.json` at `--output-dir` on completion, carrying per-voice and combined metrics under the real keys above, the untrained-baseline metrics, the best-epoch metrics, and the args/env provenance. Update `pipeline.py` to `--output-dir` and the real metric keys (F1, F3).
2. **Export step (F2):** pass `--base-gguf` / `--checkpoint <output-dir>/best_student_9block.pt` / `--out-f16` / `--out-q4k` / `--breeze-quantize` / `--embed-voice` × N / `--default-voice`; drop the nonexistent `--n-blocks` and the redundant separate quantize call. Add the student checkpoint hash to the GGUF metadata (lineage rule).
3. **Purge speculative numbers (F5):** remove hardcoded `est_rtf`; scorecard RTF is measured-from-receipt or printed as `Unmeasured (run benchmark)`. Preset table per the receipt-gated section above.
4. **Harden dry-run & stubs (F4, F6):** dry-run scorecard prints `[DRY-RUN / FABRICATED METRICS]` and never renders `[PASS]`; stub `.breeze` files are detectable (marker or skipped exists-check in dry-run).
5. **Input robustness & G2P (F7, F8):** quote-aware `parse_voice_flag`; document the single-pangram coverage claim as lexical-analysis-based and swap the hand lexicon to a real G2P source (CMUdict/espeak-ng) before any user text is analyzed; multi-utterance reference required for cloning.
6. **Empirical proof gate:** one real smoke run through `breeze-distill build` on the dev pod (2 voices, small prompt slice, receipt-carrying slice fields), then a **full-corpus** distill on the dev pod to derive the safety threshold. Commit the receipts alongside the code. Until then, the prototype's status is "wiring, unverified."

## Open after this lock (not blocking)

- 6-block research milestone (train → ear-gate → canary → re-admit to presets).
- audio.cpp UI stretch goal (interface: GGUF metadata schema + planned exporter).
- `docs/distillation.md`; server-app embedded-voice fallback (`apps/server/voices.cpp`).
