#!/usr/bin/env python3
"""Bounded BF16 versus scalar three-bit down-projection pilot.

The two three-bit methods here are deliberately small offline emulations.  They
are not ggml Q3_K encoders: codes are kept as signed scalar values and the
reported payload is a logical bit count.  Calibration selects one clipping
factor per layer; held-out capture rows are only used for the final score.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Sequence

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover
    np = None  # type: ignore[assignment]
    NUMPY_ERROR = exc

TOOLS = Path(__file__).resolve().parent
ROOT = TOOLS.parent
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from depth_ffn_screen import (  # noqa: E402
    LAYERS,
    items_for_layer,
    load_model,
    nparray,
    read_run,
    sample_matrix,
    sample_specs,
    validate_specs,
)

HIDDEN = 1024
FFN = 8192
GROUP_SIZE = 128
QMAX = 3
ROTATION_SEED = 20260906
CLIP_GRID = (0.50, 0.60, 0.70, 0.80, 0.90, 1.00)
DEFAULT_CAPTURE_ROOT = ROOT / ".beehive/agent/BREEZE-DEPTH-NEXT/captures-verified"
DEFAULT_BF16_CACHE = Path("~/.cache/breeze-models/depth-master/depth-range.bin").expanduser()
DEFAULT_BF16_MANIFEST = ROOT / "benchmarks/depth-corpus/master-manifest.json"
DEFAULT_Q4_MODEL = Path("~/.cache/breeze-models/breeze-tts-2-q4_k.gguf").expanduser()
DEFAULT_OUTPUT = ROOT / "benchmarks/depth-corpus/quant-screen.json"


def require_numpy() -> Any:
    if np is None:
        raise RuntimeError(f"numpy is required: {NUMPY_ERROR}")
    return np


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def hash_region(path: Path, offset: int, nbytes: int) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        stream.seek(offset)
        remaining = nbytes
        while remaining:
            chunk = stream.read(min(1 << 20, remaining))
            if not chunk:
                raise ValueError(f"short read hashing {path} at {offset}")
            digest.update(chunk)
            remaining -= len(chunk)
    return digest.hexdigest()


def rel_mse(actual: Any, reference: Any) -> float:
    n = require_numpy()
    den = float(n.sum(reference.astype('float64') ** 2))
    err = float(n.sum((actual.astype('float64') - reference) ** 2))
    if not den:
        if err:
            raise ValueError('relative MSE is undefined for a zero reference')
        return 0.0
    return err / den


def load_bf16_layer(cache: Path, entry: dict[str, Any]) -> Any:
    """Read one BF16 tensor without materializing the whole range file."""
    n = require_numpy()
    shape = tuple(int(value) for value in entry["shape"])
    count = math.prod(shape)
    if entry.get("dtype") != "BF16" or shape != (HIDDEN, FFN):
        raise ValueError(f"unexpected BF16 down projection metadata: {entry}")
    offset = int(entry["local_offset"])
    nbytes = int(entry["nbytes"])
    if nbytes != count * 2 or offset < 0 or offset + nbytes > cache.stat().st_size:
        raise ValueError(f"invalid BF16 range for {entry['name']}")
    raw = n.memmap(cache, dtype="<u2", mode="r", offset=offset, shape=count)
    # NumPy 2.5 has no bfloat16 dtype; widening the bits is exact.
    bits = raw.astype(n.uint32) << 16
    result = bits.view(n.float32).reshape(shape).copy()
    del bits, raw
    if not n.isfinite(result).all():
        raise ValueError(f"non-finite BF16 values in {entry['name']}")
    return result


def hadamard_matrix(width: int) -> Any:
    n = require_numpy()
    if width < 1 or width & (width - 1):
        raise ValueError("Hadamard width must be a positive power of two")
    result = n.array([[1.0]], dtype="float32")
    while result.shape[0] < width:
        result = n.block([[result, result], [result, -result]])
    return result / math.sqrt(width)


def rotation_matrix(width: int, seed: int) -> tuple[Any, Any, Any]:
    """Return D_left H D_right and its signs.

    The two independent sign diagonals intentionally make the fixed transform
    generally non-symmetric.  Callers always apply this same right transform
    to row weights and row activations.
    """
    n = require_numpy()
    rng = n.random.default_rng(seed)
    left = n.where(rng.integers(0, 2, width), 1.0, -1.0).astype("float32")
    right = n.where(rng.integers(0, 2, width), 1.0, -1.0).astype("float32")
    matrix = (left[:, None] * hadamard_matrix(width)) * right[None, :]
    return matrix, left, right


def _fwht_rows(rows: Any) -> Any:
    """Apply normalized Walsh-Hadamard to the last axis in place."""
    n = require_numpy()
    flat = rows.reshape(-1, rows.shape[-1])
    width = flat.shape[-1]
    span = 1
    while span < width:
        for start in range(0, width, 2 * span):
            left = flat[:, start:start + span].copy()
            right = flat[:, start + span:start + 2 * span].copy()
            flat[:, start:start + span] = left + right
            flat[:, start + span:start + 2 * span] = left - right
        span *= 2
    flat /= math.sqrt(width)
    return rows


def right_transform(rows: Any, left_signs: Any, right_signs: Any) -> Any:
    """Apply the fixed block transform D_left H D_right to rows."""
    n = require_numpy()
    if rows.shape[-1] % len(left_signs):
        raise ValueError("row width must be a multiple of rotation width")
    blocks = n.array(rows, dtype="float32", copy=True).reshape(-1, rows.shape[-1] // len(left_signs), len(left_signs))
    blocks *= left_signs[None, None, :]
    _fwht_rows(blocks)
    blocks *= right_signs[None, None, :]
    return blocks.reshape(rows.shape)


def quantize_symmetric3(weight: Any, clipping: float) -> Any:
    """Dequantize signed scalar 3-bit groups using seven symmetric levels."""
    n = require_numpy()
    if weight.shape[-1] % GROUP_SIZE:
        raise ValueError("weight width is not group aligned")
    groups = weight.reshape(weight.shape[0], -1, GROUP_SIZE)
    max_abs = n.max(n.abs(groups), axis=-1)
    scale = n.maximum(max_abs * float(clipping) / QMAX, n.finfo("float32").tiny)
    codes = n.rint(n.clip(groups / scale[..., None], -QMAX, QMAX)).astype("int8")
    return (codes.astype("float32") * scale[..., None]).reshape(weight.shape)


def tune_symmetric3(weight: Any, xcal: Any, reference_cal: Any) -> tuple[float, Any, list[dict[str, float]]]:
    scores = []
    best_clip = None
    best_score = math.inf
    best_weight = None
    for clipping in CLIP_GRID:
        candidate = quantize_symmetric3(weight, clipping)
        score = rel_mse(xcal @ candidate.T, reference_cal)
        scores.append({"clipping": clipping, "calibration_rel_mse": score})
        if score < best_score:
            best_clip, best_score, best_weight = clipping, score, candidate
        else:
            del candidate
    assert best_clip is not None and best_weight is not None
    return best_clip, best_weight, scores


def selected_entry(manifest: dict[str, Any], layer: int) -> dict[str, Any]:
    suffix = f"layers.{layer}.mlp.down_proj.weight"
    entries = [entry for entry in manifest["tensors"] if entry["name"].endswith(suffix)]
    if len(entries) != 1:
        raise ValueError(f"manifest does not contain exactly one down projection for layer {layer}")
    return entries[0]


def capture_hashes(specs: Sequence[tuple[str, str, Path]]) -> list[dict[str, Any]]:
    return [
        {
            "id": run_id,
            "split": split,
            "path": str(path),
            "activations_sha256": hash_file(path / "activations.f32"),
            "metadata_sha256": hash_file(path / "metadata.jsonl"),
        }
        for run_id, split, path in specs
    ]


def run_pilot(
    specs: Sequence[tuple[str, str, Path]],
    manifest_path: Path,
    bf16_cache: Path,
    q4_model: Path,
) -> dict[str, Any]:
    n = require_numpy()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not bf16_cache.is_file() or not q4_model.is_file():
        raise ValueError(f"missing model input: BF16={bf16_cache}, Q4_K={q4_model}")
    selected = manifest.get("selected", {})
    if selected.get("dtype") != "BF16":
        raise ValueError("BF16 master manifest is not BF16")
    expected_cache_bytes = int(selected.get("tensor_bytes", 0))
    if expected_cache_bytes and bf16_cache.stat().st_size != expected_cache_bytes:
        raise ValueError(f"BF16 cache size mismatch: {bf16_cache.stat().st_size} != {expected_cache_bytes}")

    validated, receipts = validate_specs(specs)
    runs = {run_id: read_run(run_id, split, path) for run_id, split, path in validated}
    splits = {run_id: split for run_id, split, _ in validated}
    cal_ids = {run_id for run_id, split in splits.items() if split == "calibration"}
    eval_ids = {run_id for run_id, split in splits.items() if split == "evaluation"}
    q4_weights = load_model(q4_model)
    rotation, left_signs, right_signs = rotation_matrix(GROUP_SIZE, ROTATION_SEED)
    ortho_error = float(n.max(n.abs(rotation.T @ rotation - n.eye(GROUP_SIZE, dtype="float32"))))
    symmetry_error = float(n.max(n.abs(rotation - rotation.T)))
    results: list[dict[str, Any]] = []
    identity_rows: list[dict[str, Any]] = []
    tensor_provenance: list[dict[str, Any]] = []
    for layer in LAYERS:
        items = items_for_layer(runs, layer)
        xcal, _ = sample_matrix(items, cal_ids)
        xeval, _ = sample_matrix(items, eval_ids)
        if not len(xcal) or not len(xeval):
            raise ValueError(f"layer {layer}: missing disjoint calibration/evaluation pairs")
        entry = selected_entry(manifest, layer)
        tensor_provenance.append({
            "layer": layer,
            "name": entry["name"],
            "shape": entry["shape"],
            "local_offset": entry["local_offset"],
            "nbytes": entry["nbytes"],
            "sha256": hash_region(bf16_cache, int(entry["local_offset"]), int(entry["nbytes"])),
        })
        if tensor_provenance[-1]['sha256'] != entry['sha256']:
            raise ValueError(f"BF16 tensor hash mismatch: {entry['name']}")
        weight = load_bf16_layer(bf16_cache, entry)
        reference_cal = xcal @ weight.T
        reference_eval = xeval @ weight.T
        q4_cal = xcal @ q4_weights[layer].T
        q4_eval = xeval @ q4_weights[layer].T
        ordinary_clip, ordinary_weight, ordinary_grid = tune_symmetric3(weight, xcal, reference_cal)
        ordinary_cal = xcal @ ordinary_weight.T
        ordinary_eval = xeval @ ordinary_weight.T

        xcal_rot = right_transform(xcal, left_signs, right_signs)
        xeval_rot = right_transform(xeval, left_signs, right_signs)
        rotated_weight = right_transform(weight, left_signs, right_signs)
        rotated_reference_cal = xcal_rot @ rotated_weight.T
        rotated_reference_eval = xeval_rot @ rotated_weight.T
        identity_rows.append({
            "layer": layer,
            "calibration_rows": len(xcal),
            "evaluation_rows": len(xeval),
            "calibration_rel_mse": rel_mse(rotated_reference_cal, reference_cal),
            "evaluation_rel_mse": rel_mse(rotated_reference_eval, reference_eval),
        })
        rotated_clip, rotated_quantized, rotated_grid = tune_symmetric3(rotated_weight, xcal_rot, reference_cal)
        rotated_cal = xcal_rot @ rotated_quantized.T
        rotated_eval = xeval_rot @ rotated_quantized.T
        results.append({
            "layer": layer,
            "calibration_rows": len(xcal),
            "evaluation_rows": len(xeval),
            "q4_k": {
                "format": "existing GGUF Q4_K dequantized reference",
                "heldout_rel_mse": rel_mse(q4_eval, reference_eval),
                "calibration_rel_mse": rel_mse(q4_cal, reference_cal),
                "weight_rel_mse": rel_mse(q4_weights[layer], weight),
            },
            "ordinary_symmetric_3bit": {
                "format": "scalar signed seven-level 3-bit emulation; not ggml Q3_K",
                "group_size": GROUP_SIZE,
                "logical_bits_per_weight": 3,
                "selected_clipping": ordinary_clip,
                "clipping_grid": ordinary_grid,
                "calibration_rel_mse": rel_mse(ordinary_cal, reference_cal),
                "heldout_rel_mse": rel_mse(ordinary_eval, reference_eval),
                "weight_rel_mse": rel_mse(ordinary_weight, weight),
            },
            "rotated_symmetric_3bit": {
                "format": "fixed seeded randomized Hadamard plus scalar signed seven-level 3-bit emulation; not ggml Q3_K",
                "group_size": GROUP_SIZE,
                "logical_bits_per_weight": 3,
                "selected_clipping": rotated_clip,
                "clipping_grid": rotated_grid,
                "calibration_rel_mse": rel_mse(rotated_cal, reference_cal),
                "heldout_rel_mse": rel_mse(rotated_eval, reference_eval),
                "weight_rel_mse_in_rotated_basis": rel_mse(rotated_quantized, rotated_weight),
            },
        })
        del weight, ordinary_weight, rotated_weight, rotated_quantized, xcal_rot, xeval_rot

    manifest_hash = hash_file(manifest_path)
    cache_hash = hash_file(bf16_cache)
    if cache_hash != selected.get('range_sha256'):
        raise ValueError('BF16 cache hash mismatch')
    q4_hash = hash_file(q4_model)
    if q4_hash != '483418fbbb438f5f1c08dbe2b017e42db0f1d126765cd3f772d8622a30a4915c':
        raise ValueError('Q4_K model does not match the captured-model provenance')
    if any(row[key] > 1e-10 for row in identity_rows
           for key in ('calibration_rel_mse', 'evaluation_rel_mse')):
        raise ValueError('unquantized rotation identity failed')
    return {
        "schema": "breeze-quant-screen/v1",
        "status": "pilot_complete",
        "objective": "BF16-referenced ordinary versus fixed-rotation scalar three-bit down-projection pilot",
        "scope": {
            "layers": list(LAYERS),
            "frames": [0, 7, 15],
            "depth_steps": list(range(1, 16)),
            "branches": 1,
            "group_size": GROUP_SIZE,
            "source_dtype": "BF16",
            "reference": "original BF16 down_proj weight multiplied by captured post-SiLU-times-up activations",
            "activation_source": "verified one-branch captures generated by the existing Q4_K depth model",
        },
        "provenance": {
            "script": str(Path(__file__).resolve()),
            "script_sha256": hash_file(Path(__file__).resolve()),
            "bf16_manifest": str(manifest_path),
            "bf16_manifest_sha256": manifest_hash,
            "bf16_cache": str(bf16_cache),
            "bf16_cache_sha256": cache_hash,
            "bf16_range_sha256_from_manifest": selected.get("range_sha256"),
            "q4_k_model": str(q4_model),
            "q4_k_model_sha256": q4_hash,
            "rotation": {
                "seed": ROTATION_SEED,
                "definition": "R = D_left @ normalized Hadamard @ D_right; same row-right R applied to weights and activations",
                "left_signs_sha256": hashlib.sha256(left_signs.tobytes()).hexdigest(),
                "right_signs_sha256": hashlib.sha256(right_signs.tobytes()).hexdigest(),
                "matrix_sha256": hashlib.sha256(rotation.tobytes()).hexdigest(),
                "orthonormal_max_abs_error": ortho_error,
                "non_symmetry_max_abs": symmetry_error,
            },
            "bf16_tensors": tensor_provenance,
            "captures": capture_hashes(validated),
        },
        "capture_validation": receipts,
        "identity_check": {
            "description": "unquantized x @ W.T versus (x @ R) @ (W @ R).T",
            "rows": identity_rows,
            "max_rel_mse": max(max(row["calibration_rel_mse"] for row in identity_rows), max(row["evaluation_rel_mse"] for row in identity_rows)),
        },
        "results": results,
        "method_caveats": [
            "three bits counts codes only: F32 scales per 128 weights add 0.25 bits/weight; seven symmetric levels leave one three-bit code unused",
            "ordinary_symmetric_3bit and rotated_symmetric_3bit are offline scalar emulations, not ggml Q3_K and not packed GGUF tensors",
            "the rotated method requires applying the fixed block transform to activations at runtime; no runtime kernel or format was implemented",
            "Q4_K is measured by dequantizing the existing GGUF model, so Q4_K metadata/packing is not directly comparable to the emulation's logical three-bit payload",
            "captured activations came from the existing Q4_K model; BF16-reference scores therefore isolate down-weight approximation on a Q4_K-generated activation corpus",
            "this is a down-projection output-MSE pilot only; it does not measure speech quality, end-to-end runtime, or free-running generation",
            "clipping was selected from calibration rows only; held-out rows are evaluation-only",
        ],
    }


def self_test() -> None:
    n = require_numpy()
    rng = n.random.default_rng(7)
    rows = rng.normal(size=(3, FFN)).astype("float32")
    matrix, left, right = rotation_matrix(GROUP_SIZE, ROTATION_SEED)
    transformed = right_transform(rows, left, right)
    expected = rows.reshape(-1, GROUP_SIZE) @ matrix
    assert n.allclose(transformed.reshape(-1, GROUP_SIZE), expected, rtol=2e-5, atol=2e-5)
    weight = rng.normal(size=(HIDDEN, FFN)).astype("float32")
    x = rng.normal(size=(5, FFN)).astype("float32")
    transformed_weight = right_transform(weight, left, right)
    transformed_x = right_transform(x, left, right)
    identity = rel_mse(transformed_x @ transformed_weight.T, x @ weight.T)
    assert identity < 1e-10, identity
    dequantized = quantize_symmetric3(weight, 0.8)
    assert dequantized.shape == weight.shape and n.isfinite(dequantized).all()
    assert n.max(n.abs(matrix.T @ matrix - n.eye(GROUP_SIZE))) < 2e-6
    print("depth_quant_screen self-test: PASS")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("captures", nargs="*", type=Path, help="capture roots or individual run directories")
    parser.add_argument("--sample", action="append", default=[], metavar="ID=SPLIT:PATH")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_BF16_MANIFEST)
    parser.add_argument("--bf16-cache", type=Path, default=DEFAULT_BF16_CACHE)
    parser.add_argument("--model", "--q4-model", dest="q4_model", type=Path, default=DEFAULT_Q4_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.self_test:
            self_test()
            return 0
        if args.sample:
            specs = sample_specs(args.captures, args.sample)
        elif args.captures:
            specs = sample_specs(args.captures, [])
        else:
            specs = sample_specs([DEFAULT_CAPTURE_ROOT], [])
        report = run_pilot(specs, args.manifest.expanduser().resolve(), args.bf16_cache.expanduser().resolve(), args.q4_model.expanduser().resolve())
        args.output.expanduser().resolve().write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
        print(json.dumps({"output": str(args.output), "layers": list(LAYERS), "status": report["status"]}))
        return 0
    except (AssertionError, OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"depth_quant_screen: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
