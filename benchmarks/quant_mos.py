#!/usr/bin/env python3
"""Verify the bounded official UTMOSv2 naturalness-proxy receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "receipt",
        nargs="?",
        default="benchmarks/depth-corpus/language-quality-mos.json",
    )
    args = parser.parse_args()
    receipt_path = Path(args.receipt)
    root = Path.cwd()
    receipt = json.loads(receipt_path.read_text())
    if receipt.get("schema") != "breeze-language-quality-mos/v1":
        raise SystemExit("wrong MOS receipt schema")
    if receipt.get("status") != "complete":
        raise SystemExit("MOS receipt is not complete")

    source = receipt["source_runtime_receipt"]
    source_path = root / source["path"]
    if sha256(source_path) != source["sha256"]:
        raise SystemExit("source runtime receipt hash mismatch")
    order = source["case_order"]
    results = receipt["results"]
    if len(results) != source["case_count"] or [r["case_id"] for r in results] != order:
        raise SystemExit("MOS case order/count mismatch")

    runtime = json.loads(source_path.read_text())
    for row in results:
        case_id = row["case_id"]
        expected = runtime["cases"][case_id]["wav_sha256"]
        path = root / row["wav_path"]
        if row["wav_sha256"] != expected or sha256(path) != expected:
            raise SystemExit(f"WAV hash mismatch: {case_id}")
        if not math.isfinite(row["predicted_mos"]):
            raise SystemExit(f"non-finite MOS score: {case_id}")

    provenance = receipt["execution_provenance"]
    runner = root / provenance["runner"]["path"]
    if sha256(runner) != provenance["runner"]["sha256"]:
        raise SystemExit("inference runner hash mismatch")
    raw_output = root / provenance["raw_output"]["path"]
    if sha256(raw_output) != provenance["raw_output"]["sha256"]:
        raise SystemExit("score-output provenance hash mismatch")
    strict_path = root / provenance["strict_load_evidence"]["path"]
    if sha256(strict_path) != provenance["strict_load_evidence"]["sha256"]:
        raise SystemExit("strict-load evidence hash mismatch")
    strict_script = root / provenance["strict_load_evidence"]["script_path"]
    if sha256(strict_script) != provenance["strict_load_evidence"]["script_sha256"]:
        raise SystemExit("strict-load script hash mismatch")
    strict = json.loads(strict_path.read_text())
    if (
        strict.get("status") != "pass"
        or strict.get("missing_keys")
        or strict.get("unexpected_keys")
        or strict.get("shape_mismatches")
        or strict.get("checkpoint_sha256") != receipt["proxy"]["checkpoint_sha256"]
    ):
        raise SystemExit("strict-load evidence is not a clean checkpoint match")

    repeat = receipt["determinism"]["repeat"]
    if not repeat["equal_within_tolerance"] or repeat["absolute_delta"] > 1e-7:
        raise SystemExit("seeded repeat is not deterministic")
    if receipt["acceptance"]["threshold_selected"]:
        raise SystemExit("receipt must not select a MOS threshold")
    print(f"verified {len(results)} UTMOSv2 proxy scores; no acceptance threshold")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
