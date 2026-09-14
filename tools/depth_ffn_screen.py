#!/usr/bin/env python3
"""Screen the bounded depth capture before any compression work.

Each run directory contains ``activations.f32`` and ``metadata.jsonl``. A run
ID beginning with ``cal-`` is calibration and one beginning with ``eval-`` is
held out. Use ``--sample id=split:path`` when names do not follow that rule.

With ``--model`` this dequantizes the selected Q4_K down matrices through the
installed ``gguf`` package, checks captured outputs against that dense
reference, and measures contiguous 256-channel masks plus rounded F16
low-rank factors on disjoint utterances. It does not claim speech quality,
runtime, or generalization from this pilot.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import tempfile
from pathlib import Path
from typing import Any, Sequence

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover
    np = None  # type: ignore[assignment]
    NUMPY_ERROR = exc

Q4K_BLOCK_VALUES = 256
Q4K_BLOCK_BYTES = 144
FFN = 8192
HIDDEN = 1024
LAYERS = (0, 5, 11)
FRAMES = (0, 7, 15)
TENSORS = ("post_silu_times_up", "ffn_down_output")


def nparray() -> Any:
    if np is None:
        raise RuntimeError(f"numpy is required: {NUMPY_ERROR}")
    return np


def split_name(name: str) -> str:
    name = name.lower().replace("_", "-")
    if name.startswith("cal-"):
        return "calibration"
    if name.startswith("eval-"):
        return "evaluation"
    return "unknown"


def sample_specs(paths: Sequence[Path], specs: Sequence[str]) -> list[tuple[str, str, Path]]:
    if specs:
        result = []
        for spec in specs:
            match = re.fullmatch(r"([^=]+)=(calibration|evaluation):(.+)", spec)
            if not match:
                raise ValueError(f"--sample expects ID=calibration|evaluation:PATH, got {spec!r}")
            result.append((match.group(1), match.group(2), Path(match.group(3))))
        return result
    result = []
    for path in paths:
        dirs = [path] if (path / "metadata.jsonl").is_file() else sorted(
            child for child in path.iterdir()
            if child.is_dir() and (child / "metadata.jsonl").is_file() and split_name(child.name) != "unknown"
        )
        for capture in dirs:
            result.append((capture.name, split_name(capture.name), capture))
    return result


def normalize_specs(specs: Sequence[tuple[str, str, Path]]) -> list[tuple[str, str, Path]]:
    seen_ids: set[str] = set()
    seen_paths: dict[Path, str] = {}
    normalized = []
    for run_id, split, path in specs:
        resolved = path.expanduser().resolve()
        if run_id in seen_ids:
            raise ValueError(f"duplicate capture ID: {run_id}")
        if resolved in seen_paths:
            raise ValueError(f"capture path reused by {seen_paths[resolved]} and {run_id}: {resolved}")
        seen_ids.add(run_id)
        seen_paths[resolved] = run_id
        normalized.append((run_id, split, resolved))
    splits = {split for _, split, _ in normalized}
    if "calibration" not in splits or "evaluation" not in splits:
        raise ValueError("capture set must include calibration and evaluation runs")
    return normalized


def validate_specs(specs: Sequence[tuple[str, str, Path]]) -> tuple[list[tuple[str, str, Path]], list[dict[str, Any]]]:
    try:
        from depth_capture_check import validate
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("depth_capture_check.py is required for capture validation") from exc
    normalized = normalize_specs(specs)
    receipts = []
    for run_id, split, path in normalized:
        receipt = validate(path, full=True)
        header = None
        for line in (path / "metadata.jsonl").read_text(encoding="utf-8").splitlines():
            item = json.loads(line)
            if item.get("kind") == "header":
                header = item
                break
        if not header or int(header.get("branches", 0)) != 1:
            raise ValueError(f"{run_id}: screen requires a one-branch capture")
        receipts.append({"id": run_id, "split": split, "branches": 1, **receipt})
    return normalized, receipts


def read_run(run_id: str, split: str, capture: Path) -> dict[tuple[int, int, int, str], Any]:
    """Read exactly the capture worker's JSONL/blob schema."""
    if split == "unknown":
        raise ValueError(f"{run_id}: split is unknown; pass --sample {run_id}=calibration|evaluation:PATH")
    meta = capture / "metadata.jsonl"
    blob = capture / "activations.f32"
    if not meta.is_file() or not blob.is_file():
        raise ValueError(f"{capture}: expected metadata.jsonl and activations.f32")
    records: dict[tuple[int, int, int, str], Any] = {}
    with meta.open(encoding="utf-8") as stream, blob.open("rb") as data:
        for line_no, line in enumerate(stream, 1):
            item = json.loads(line)
            if item.get("kind") != "tensor" or item.get("tensor") not in TENSORS:
                continue
            layer, frame, step = (int(item[x]) for x in ("layer", "frame", "step"))
            shape = tuple(int(x) for x in item["shape"])
            if layer not in LAYERS or frame not in FRAMES or len(shape) != 3 or shape[2] != 1:
                continue
            width, tokens = shape[:2]
            expected_width = FFN if item["tensor"] == "post_silu_times_up" else HIDDEN
            if width != expected_width or tokens < 1:
                raise ValueError(f"{meta}:{line_no}: {item['tensor']} shape {shape} is not expected")
            if str(item.get("dtype", "")).lower() != "f32":
                raise ValueError(f"{meta}:{line_no}: selected capture is not f32")
            count = width * tokens
            offset = int(item["byte_offset"])
            nbytes = int(item["nbytes"])
            if offset != int(item["offset"]) or offset < 0 or nbytes != count * 4:
                raise ValueError(f"{meta}:{line_no}: invalid blob range")
            data.seek(offset)
            values = nparray().frombuffer(data.read(count * 4), dtype="<f4").copy()
            if len(values) != count or not nparray().isfinite(values).all():
                raise ValueError(f"{meta}:{line_no}: short or non-finite payload")
            # GGML ne[0] is contiguous: [width,tokens,1] becomes token rows.
            rows = values.reshape((tokens, width))
            key = (layer, frame, step, str(item["tensor"]))
            if key in records:
                raise ValueError(f"{meta}:{line_no}: duplicate record {key}")
            records[key] = rows
    return records


def load_model(path: Path) -> dict[int, Any]:
    try:
        import gguf
    except ImportError as exc:
        raise RuntimeError("--model needs the gguf package (use /tmp/breeze-quality-venv/bin/python)") from exc
    reader = gguf.GGUFReader(path)
    result = {}
    for layer in LAYERS:
        name = f"dd.blk.{layer}.ffn_down.weight"
        tensor = next((item for item in reader.tensors if item.name == name), None)
        if tensor is None or tensor.tensor_type != gguf.GGMLQuantizationType.Q4_K:
            raise ValueError(f"{path}: missing Q4_K tensor {name}")
        weight = gguf.dequantize(tensor.data, tensor.tensor_type)
        if weight.shape != (HIDDEN, FFN):
            raise ValueError(f"{name}: dequantized shape {weight.shape}, expected {(HIDDEN, FFN)}")
        result[layer] = nparray().asarray(weight, dtype="float32")
    return result


def q4k_bytes(rows: int, cols: int) -> int:
    if cols % Q4K_BLOCK_VALUES:
        raise ValueError("Q4_K dimension is not block aligned")
    return rows * (cols // Q4K_BLOCK_VALUES) * Q4K_BLOCK_BYTES


def items_for_layer(runs: dict[str, dict[tuple[int, int, int, str], Any]], layer: int) -> list[tuple[str, int, int, Any, Any]]:
    items = []
    for run_id, records in runs.items():
        for (record_layer, frame, step, tensor), rows in records.items():
            if record_layer != layer or tensor != "post_silu_times_up":
                continue
            down = records.get((layer, frame, step, "ffn_down_output"))
            if down is not None and down.shape[0] == rows.shape[0]:
                items.append((run_id, frame, step, rows, down))
    return items


def sample_matrix(items: Sequence[tuple[str, int, int, Any, Any]], run_ids: set[str], step: int | None = None) -> tuple[Any, Any]:
    chosen = [item for item in items if item[0] in run_ids and (step is None or item[2] == step)]
    if not chosen:
        return nparray().empty((0, FFN), dtype="float32"), nparray().empty((0, HIDDEN), dtype="float32")
    return nparray().concatenate([item[3] for item in chosen], axis=0), nparray().concatenate([item[4] for item in chosen], axis=0)


def rel_mse(a: Any, b: Any) -> float:
    den = float(nparray().mean(b * b))
    return float(nparray().mean((a - b) ** 2) / den) if den else 0.0


def window_start(energy: Any, keep: int) -> int:
    if keep >= len(energy):
        return 0
    score = nparray().convolve(energy, nparray().ones(keep), mode="valid")
    return int(nparray().argmax(score))


def apply_window(x: Any, start: int, keep: int) -> Any:
    mask = nparray().zeros(FFN // Q4K_BLOCK_VALUES, dtype=bool)
    mask[start:start + keep] = True
    return x * nparray().repeat(mask, Q4K_BLOCK_VALUES)[None, :]


def regression_factors(x: Any, y: Any) -> tuple[Any, Any, Any]:
    u, s, vt = nparray().linalg.svd(x, full_matrices=False)
    if not len(s):
        return u, s, vt
    keep = s > max(float(s[0]) * 1e-6, 1e-12)
    return u[:, keep], s[keep], vt[keep]


def screen(runs: dict[str, dict[tuple[int, int, int, str], Any]], splits: dict[str, str], weights: dict[int, Any] | None, fractions: Sequence[float], ranks: Sequence[int]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[str], list[str]]:
    cal = {run for run, split in splits.items() if split == "calibration"}
    eva = {run for run, split in splits.items() if split == "evaluation"}
    blockers: list[str] = []
    checks: list[str] = []
    if not cal:
        blockers.append("no calibration runs")
    if not eva:
        blockers.append("no held-out evaluation runs")
    if weights is None:
        blockers.append("--model omitted: output distortion and low-rank rows are deferred")
        return [], [], [], blockers, checks
    mask_rows: list[dict[str, Any]] = []
    rank_rows: list[dict[str, Any]] = []
    spectrum_rows: list[dict[str, Any]] = []
    for layer in LAYERS:
        items = items_for_layer(runs, layer)
        xcal, ycal = sample_matrix(items, cal)
        xeval, yeval = sample_matrix(items, eva)
        if not len(xcal) or not len(xeval):
            blockers.append(f"layer {layer}: incomplete disjoint pairs")
            continue
        weight = weights[layer]
        dense_cal, dense_eval = xcal @ weight.T, xeval @ weight.T
        cal_err, eval_err = rel_mse(ycal, dense_cal), rel_mse(yeval, dense_eval)
        checks.append(f"layer {layer}: captured down vs dequantized dense rel MSE calibration={cal_err:.6g}, evaluation={eval_err:.6g}")
        if max(cal_err, eval_err) > 1e-3:
            blockers.append(f"layer {layer}: capture/dequantized down mismatch exceeds 1e-3; do not rank compression")
            continue
        nblocks = FFN // Q4K_BLOCK_VALUES
        down_bytes = q4k_bytes(HIDDEN, FFN)
        all3_bytes = down_bytes * 3
        energy = nparray().mean(xcal.reshape(len(xcal), nblocks, Q4K_BLOCK_VALUES) ** 2, axis=(0, 2))
        for fraction in fractions:
            keep = max(1, min(nblocks, math.ceil(fraction * nblocks)))
            start = window_start(energy, keep)
            mask_rows.append({"layer": layer, "scope": "global", "fraction": fraction, "blocks": keep, "start": start, "eval_activation": rel_mse(apply_window(xeval, start, keep), xeval), "eval_output": rel_mse(apply_window(xeval, start, keep) @ weight.T, dense_eval), "down_bytes": round(down_bytes * keep / nblocks), "all3_bytes": round(all3_bytes * keep / nblocks), "cal_rows": len(xcal), "eval_rows": len(xeval)})
        for step in sorted({item[2] for item in items}):
            sxcal, _ = sample_matrix(items, cal, step)
            sxev, _ = sample_matrix(items, eva, step)
            if not len(sxcal) or not len(sxev):
                continue
            step_energy = nparray().mean(sxcal.reshape(len(sxcal), nblocks, Q4K_BLOCK_VALUES) ** 2, axis=(0, 2))
            step_dense = sxev @ weight.T
            for fraction in fractions:
                keep = max(1, min(nblocks, math.ceil(fraction * nblocks)))
                start = window_start(step_energy, keep)
                mask_rows.append({"layer": layer, "scope": f"step_{step}", "fraction": fraction, "blocks": keep, "start": start, "eval_activation": rel_mse(apply_window(sxev, start, keep), sxev), "eval_output": rel_mse(apply_window(sxev, start, keep) @ weight.T, step_dense), "down_bytes": round(down_bytes * keep / nblocks), "all3_bytes": round(all3_bytes * keep / nblocks), "cal_rows": len(sxcal), "eval_rows": len(sxev)})
        u, s, vt = regression_factors(xcal, dense_cal)
        supported = min(len(s), max(0, len(xcal) - 1))
        spectrum = nparray().cumsum(s * s) / max(float(nparray().sum(s * s)), 1e-30)
        spectrum_rows.append({"layer": layer, "rows": len(xcal), "cumulative": {rank: float(spectrum[rank - 1]) for rank in (4, 8, 16, 32, 64) if rank <= supported}})
        for rank in ranks:
            if rank < 1 or rank > supported:
                continue
            a = (u[:, :rank].T @ dense_cal) / s[:rank, None]
            b = vt[:rank]
            # Simulate actual proposed storage, then evaluate rounded factors.
            a16 = a.astype("float16").astype("float32")
            b16 = b.astype("float16").astype("float32")
            rank_rows.append({"layer": layer, "rank": rank, "f16_bytes": 2 * rank * (FFN + HIDDEN), "q4k_bytes": down_bytes, "train_error": rel_mse((xcal @ b16.T) @ a16, dense_cal), "eval_error": rel_mse((xeval @ b16.T) @ a16, dense_eval), "cal_rows": len(xcal), "eval_rows": len(xeval)})
    return mask_rows, rank_rows, spectrum_rows, blockers, checks


def report(records: int, runs: int, validations: Sequence[dict[str, Any]], masks: Sequence[dict[str, Any]], ranks: Sequence[dict[str, Any]], spectra: Sequence[dict[str, Any]], blockers: Sequence[str], checks: Sequence[str]) -> str:
    lines = [
        "# BREEZE-DEPTH-NEXT FFN numerical screen",
        "",
        "This is a bounded tensor-level pilot. It reports output distortion against dequantized dense Q4_K down references and rounded F16-factor reconstruction; it makes no speech-quality, runtime, or generalization claim.",
        "",
        f"Capture runs: {runs}; selected tensor records: {records}",
        "",
        "## Input validation",
        "",
    ]
    for receipt in validations:
        lines.append(f"- {receipt['id']} ({receipt['split']}): branches={receipt['branches']}, frames={receipt['frames']}, tensor records={receipt['tensors']}, blob bytes={receipt['blob_bytes']}")
    lines.extend(["", "## Capture checks", ""])
    lines.extend(f"- {item}" for item in checks)
    lines.extend(f"- blocker: {item}" for item in blockers)
    lines.extend([
        "",
        "## Aligned contiguous masks",
        "",
        "Windows are selected from calibration post-SwiGLU block energy. Output error is masked activation multiplied by the dequantized dense Q4_K down matrix, relative to the dense down reference. The byte columns are logical aligned bytes touched per call; they do not claim a dense masked matmul speedup or resident storage reduction. A codebook-specific window can reuse one original tensor; reduced resident storage would require an explicit union/repack, not 15 copies.",
        "",
        "| layer | scope | fraction | blocks | start | held-out activation rel MSE | held-out output rel MSE | logical down bytes/call | logical all3 bytes/call | cal/eval rows |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in masks:
        lines.append(f"| {row['layer']} | {row['scope']} | {row['fraction']:.6g} | {row['blocks']} | {row['start']} | {row['eval_activation']:.6g} | {row['eval_output']:.6g} | {row['down_bytes']} | {row['all3_bytes']} | {row['cal_rows']}/{row['eval_rows']} |")
    if not masks:
        lines.append("| — | — | — | — | — | — | — | — | — | — |")
    lines.extend([
        "",
        "## Activation spectrum and rounded F16 low-rank down factors",
        "",
        "The singular-energy rows describe the sampled post-SwiGLU activation subspace. Factors are a pilot truncated regression to the dequantized dense down reference, not an optimal activation-weighted weight-compression fit. Both factors are rounded to F16 before evaluation. With 96 calibration rows, the mathematical sample support is at most rank 95; this pilot does not justify extrapolation or deployment.",
        "",
        "| layer | calibration rows | cumulative singular energy at rank 4/8/16/32/64 |",
        "|---:|---:|---|",
    ])
    for row in spectra:
        values = ", ".join(f"{rank}:{energy:.6g}" for rank, energy in row["cumulative"].items())
        lines.append(f"| {row['layer']} | {row['rows']} | {values or '—'} |")
    if not spectra:
        lines.append("| — | — | — |")
    lines.extend([
        "",
        "| layer | rank | F16 bytes | Q4_K bytes | train rel MSE | held-out rel MSE | cal/eval rows |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in ranks:
        lines.append(f"| {row['layer']} | {row['rank']} | {row['f16_bytes']} | {row['q4k_bytes']} | {row['train_error']:.6g} | {row['eval_error']:.6g} | {row['cal_rows']}/{row['eval_rows']} |")
    if not ranks:
        lines.append("| — | — | — | — | — | — | — |")
    global75 = [row for row in masks if row["scope"] == "global" and math.isclose(row["fraction"], 0.75)]
    rank64 = [row for row in ranks if row["rank"] == 64]
    lines.extend(["", "## Pilot reading", ""])
    if global75:
        values = ", ".join(f"layer {row['layer']} {row['eval_output']:.3g}" for row in global75)
        lines.append(f"- Keeping 75% of contiguous post-SwiGLU blocks leaves held-out output rel MSE of {values}; the logical aligned read is 3,538,944 down bytes per call (10,616,832 for gate/up/down), not a resident-storage claim.")
    if rank64:
        values = ", ".join(f"layer {row['layer']} {row['eval_error']:.3g}" for row in rank64)
        lines.append(f"- Rounded F16 rank 64 uses 1,179,648 bytes versus 4,718,592 Q4_K bytes; held-out output rel MSE is {values}.")
    lines.append("- The sample is too small for deployment claims: 96 calibration rows, no reference or long-utterance coverage, and no free-running quality or timing measurement here.")
    lines.extend(["", "No compression deployment follows from this report.", ""])
    return "\n".join(lines)


def self_test() -> None:
    n = nparray()
    with tempfile.TemporaryDirectory() as root_name:
        root = Path(root_name)
        for name, delta in (("cal-en", 0.0), ("eval-en", 1.0)):
            run = root / name
            run.mkdir()
            blob = run / "activations.f32"
            lines = []
            with blob.open("wb") as out:
                offset = 0
                for layer in LAYERS:
                    for frame in FRAMES:
                        x = (n.arange(FFN, dtype="float32") + delta + layer + frame).reshape(FFN, 1)
                        y = n.ones((HIDDEN, 1), dtype="float32") * (delta + layer + frame)
                        for tensor, value in (("post_silu_times_up", x), ("ffn_down_output", y)):
                            payload = value.tobytes()
                            out.write(payload)
                            lines.append(json.dumps({"kind": "tensor", "frame": frame, "step": 1, "layer": layer, "tensor": tensor, "dtype": "f32", "shape": [value.shape[0], 1, 1], "offset": offset, "byte_offset": offset, "nbytes": len(payload)}))
                            offset += len(payload)
            (run / "metadata.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
        runs = {name: read_run(name, split_name(name), root / name) for name in ("cal-en", "eval-en")}
        assert len(runs["cal-en"]) == 18
        assert all(n.isfinite(value).all() for run in runs.values() for value in run.values())
        try:
            normalize_specs([("same", "calibration", root / "cal-en"), ("other", "evaluation", root / "cal-en")])
        except ValueError:
            pass
        else:
            raise AssertionError("duplicate capture path was not rejected")
    print("depth_ffn_screen self-test: ok")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("captures", nargs="*", type=Path)
    parser.add_argument("--sample", action="append", default=[], metavar="ID=SPLIT:PATH")
    parser.add_argument("--model", type=Path, help="Q4_K GGUF used for dense down references")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--fraction", action="append", type=float, default=[])
    parser.add_argument("--rank", action="append", type=int, default=[])
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.self_test:
            self_test()
            return 0
        specs = sample_specs(args.captures, args.sample)
        if not specs:
            parser.error("provide capture roots or --sample")
        specs, validations = validate_specs(specs)
        runs = {run_id: read_run(run_id, split, path) for run_id, split, path in specs}
        splits = {run_id: split for run_id, split, _ in specs}
        weights = load_model(args.model) if args.model else None
        masks, ranks, spectra, blockers, checks = screen(runs, splits, weights, tuple(args.fraction or (0.25, 0.5, 0.75)), tuple(args.rank or (4, 8, 16, 32, 64)))
        text = report(sum(len(run) for run in runs.values()), len(runs), validations, masks, ranks, spectra, blockers, checks)
        if args.report:
            args.report.write_text(text, encoding="utf-8")
        else:
            print(text)
        return 0
    except (AssertionError, OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"depth_ffn_screen: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
