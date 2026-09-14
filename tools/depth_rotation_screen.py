#!/usr/bin/env python3
"""Screen a fixed normalized Walsh-Hadamard basis on one depth FFN layer.

This is an offline projection diagnostic.  It reads only the BF16 depth master
range (rounded through F16 to F32), the two capped expanded calibration
captures, and the existing Q4_K control.  Native ggml IQ2_S and Q3_K packing
is used through ``ggml_quantize_chunk``; no GGUF or runtime artifact is
written.  English calibration rows provide the imatrix and Mandarin rows are a
diagnostic comparison, rather than a final held-out quality set.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Sequence

try:
    import gguf
    import numpy as np
except ImportError as exc:  # pragma: no cover - the quality venv supplies both
    gguf = None  # type: ignore[assignment]
    np = None  # type: ignore[assignment]
    IMPORT_ERROR = exc

TOOLS = Path(__file__).resolve().parent
ROOT = TOOLS.parent
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from build_depth_quant_controls import MAPPING  # noqa: E402
from depth_capture_check import validate as validate_capture  # noqa: E402

HIDDEN = 1024
FFN = 8192
LAYER = 5
EXPECTED_FIXTURES = ("cal-en-long-123", "cal-zh-long-123")
EXPECTED_FRAMES = (0, 7, 15, 31, 63)
ROLES = {
    "gate": {
        "gguf_name": f"dd.blk.{LAYER}.ffn_gate.weight",
        "master_suffix": "mlp.gate_proj.weight",
        "capture_tensor": "ffn_norm_input",
        "width": HIDDEN,
    },
    "up": {
        "gguf_name": f"dd.blk.{LAYER}.ffn_up.weight",
        "master_suffix": "mlp.up_proj.weight",
        "capture_tensor": "ffn_norm_input",
        "width": HIDDEN,
    },
    "down": {
        "gguf_name": f"dd.blk.{LAYER}.ffn_down.weight",
        "master_suffix": "mlp.down_proj.weight",
        "capture_tensor": "post_silu_times_up",
        "width": FFN,
    },
}

DEFAULT_CAPTURE_RECEIPT = ROOT / "benchmarks/depth-corpus/capture-expanded.json"
DEFAULT_MASTER_MANIFEST = ROOT / "benchmarks/depth-corpus/master-manifest.json"
DEFAULT_MASTER_CACHE = Path("~/.cache/breeze-models/depth-master/depth-range.bin").expanduser()
DEFAULT_Q4_MODEL = Path("~/.cache/breeze-models/depth-quant-controls/breeze-depth-blocks-q4_k.gguf").expanduser()
DEFAULT_QUANTIZER = ROOT / "build-ffn-bench/third_party/ggml/src/libggml-base.0.22.0.dylib"
DEFAULT_OUTPUT = ROOT / "benchmarks/depth-corpus/rotation-screen.json"


def require_numpy() -> Any:
    if np is None or gguf is None:
        raise RuntimeError(f"numpy and gguf are required: {IMPORT_ERROR}")
    return np


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def hash_bytes(value: Any) -> str:
    return hashlib.sha256(value.tobytes()).hexdigest()


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


def projection_metrics(actual: Any, reference: Any) -> dict[str, float]:
    n = require_numpy()
    delta = actual.astype(n.float64) - reference.astype(n.float64)
    ref_norm = float(n.linalg.norm(reference.astype(n.float64)))
    return {
        "rel_l2": float(n.linalg.norm(delta) / max(ref_norm, 1e-30)),
        "max_abs": float(n.max(n.abs(delta))) if delta.size else 0.0,
        "reference_rms": float(n.sqrt(n.mean(reference.astype(n.float64) ** 2))),
    }


def activation_stats(rows: Any) -> dict[str, float | int]:
    n = require_numpy()
    values = rows.astype(n.float64, copy=False)
    return {
        "rows": int(rows.shape[0]),
        "width": int(rows.shape[1]),
        "max_abs": float(n.max(n.abs(values))) if values.size else 0.0,
        "rms": float(n.sqrt(n.mean(values * values))) if values.size else 0.0,
    }


def fwht_rows(rows: Any) -> Any:
    """Apply a normalized, plain Walsh-Hadamard matrix on each row."""
    n = require_numpy()
    result = n.array(rows, dtype=rows.dtype, copy=True, order="C")
    flat = result.reshape(-1, result.shape[-1])
    width = flat.shape[-1]
    if width < 1 or width & (width - 1):
        raise ValueError(f"Hadamard width must be a power of two, got {width}")
    span = 1
    while span < width:
        for start in range(0, width, span * 2):
            left = flat[:, start:start + span].copy()
            right = flat[:, start + span:start + span * 2].copy()
            flat[:, start:start + span] = left + right
            flat[:, start + span:start + span * 2] = left - right
        span *= 2
    flat /= math.sqrt(width)
    return result


def load_bf16_f16(entry: dict[str, Any], master_cache: Path) -> tuple[Any, dict[str, Any]]:
    """Load one master tensor without materializing the complete range cache."""
    n = require_numpy()
    shape = tuple(int(value) for value in entry["shape"])
    count = math.prod(shape)
    if entry.get("dtype") != "BF16" or int(entry["nbytes"]) != count * 2:
        raise ValueError(f"bad BF16 entry: {entry}")
    offset = int(entry["local_offset"])
    if offset < 0 or offset + count * 2 > master_cache.stat().st_size:
        raise ValueError(f"master range outside cache: {entry['name']}")
    actual_hash = hash_region(master_cache, offset, count * 2)
    if actual_hash != entry.get("sha256"):
        raise ValueError(f"master tensor hash mismatch: {entry['name']}")
    raw = n.memmap(master_cache, dtype="<u2", mode="r", offset=offset, shape=count)
    bits = raw.astype(n.uint32) << 16
    bf32 = bits.view(n.float32)
    if not n.isfinite(bf32).all():
        raise ValueError(f"non-finite BF16 values: {entry['name']}")
    f16 = bf32.astype(n.float16)
    if not n.isfinite(f16).all():
        raise ValueError(f"F16 overflow: {entry['name']}")
    source = n.ascontiguousarray(f16.astype(n.float32).reshape(shape))
    delta = f16.astype(n.float64) - bf32.astype(n.float64)
    conversion = {
        "master_name": entry["name"],
        "shape": list(shape),
        "master_region_sha256": actual_hash,
        "changed_values": int(n.count_nonzero(delta)),
        "max_abs": float(n.max(n.abs(delta))),
        "relative_mse": float(n.sum(delta * delta) / max(float(n.sum(bf32.astype(n.float64) ** 2)), 1e-30)),
    }
    del raw, bits, bf32, f16, delta
    return source, conversion


def load_capture(capture: Path, tensor_name: str, layer: int = LAYER) -> Any:
    """Read selected rows from one expanded capture blob."""
    n = require_numpy()
    records: list[Any] = []
    seen: set[tuple[int, int]] = set()
    with (capture / "metadata.jsonl").open(encoding="utf-8") as metadata, (capture / "activations.f32").open("rb") as blob:
        for line_no, line in enumerate(metadata, 1):
            item = json.loads(line)
            if item.get("kind") != "tensor" or item.get("tensor") != tensor_name or int(item.get("layer", -1)) != layer:
                continue
            shape = tuple(int(value) for value in item["shape"])
            if len(shape) != 3 or shape[2] != 1 or shape[0] not in (HIDDEN, FFN) or shape[1] < 1:
                raise ValueError(f"{capture}/metadata.jsonl:{line_no}: invalid selected shape {shape}")
            if item.get("dtype", "").lower() != "f32":
                raise ValueError(f"{capture}/metadata.jsonl:{line_no}: selected tensor is not f32")
            key = (int(item["frame"]), int(item["step"]))
            if key in seen:
                raise ValueError(f"duplicate selected capture coordinate {key}")
            seen.add(key)
            count = shape[0] * shape[1]
            offset = int(item["byte_offset"])
            nbytes = int(item["nbytes"])
            if offset != int(item["offset"]) or nbytes != count * 4:
                raise ValueError(f"{capture}/metadata.jsonl:{line_no}: invalid selected range")
            blob.seek(offset)
            values = n.frombuffer(blob.read(nbytes), dtype="<f4").copy()
            if len(values) != count or not n.isfinite(values).all():
                raise ValueError(f"{capture}/metadata.jsonl:{line_no}: invalid selected payload")
            records.append(values.reshape(shape[1], shape[0]))
    if len(records) != len(EXPECTED_FRAMES) * 15:
        raise ValueError(f"{capture}: expected 75 {tensor_name} records for layer {layer}, got {len(records)}")
    result = n.concatenate(records, axis=0)
    if result.shape[0] != 80:
        raise ValueError(f"{capture}: expected 80 selected rows, got {result.shape}")
    return n.ascontiguousarray(result, dtype=n.float32)


def normalized_imatrix(rows: Any) -> tuple[Any, dict[str, Any]]:
    n = require_numpy()
    mean_square = n.mean(rows.astype(n.float64) ** 2, axis=0, dtype=n.float64)
    average = float(n.mean(mean_square))
    if not math.isfinite(average) or average <= 0.0:
        raise ValueError("activation imatrix has no positive mean square")
    weights = n.ascontiguousarray((mean_square / average).astype(n.float32))
    return weights, {
        "statistic": "float64 mean of squared calibration activations per input channel",
        "samples": int(rows.shape[0]),
        "channels": int(rows.shape[1]),
        "mean_square_sha256": hash_bytes(mean_square),
        "normalized_weight_sha256": hash_bytes(weights),
        "normalized_mean": float(n.mean(weights.astype(n.float64))),
        "normalized_min": float(n.min(weights)),
        "normalized_max": float(n.max(weights)),
    }


def load_master_entry(manifest: dict[str, Any], gguf_name: str) -> dict[str, Any]:
    master_name = MAPPING[gguf_name]
    matches = [entry for entry in manifest["tensors"] if entry["name"] == master_name]
    if len(matches) != 1:
        raise ValueError(f"master manifest does not contain exactly one {master_name}")
    return matches[0]


def native_quantizer(path: Path) -> tuple[Any, Any]:
    if not path.is_file():
        raise ValueError(f"missing native quantizer: {path}")
    library = ctypes.CDLL(str(path))
    function = library.ggml_quantize_chunk
    function.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_float), ctypes.c_void_p,
                         ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
                         ctypes.POINTER(ctypes.c_float)]
    function.restype = ctypes.c_size_t
    return library, function


def quantize(function: Any, qtype: Any, source: Any, weights: Any) -> tuple[bytes, Any]:
    n = require_numpy()
    block_size, type_size = gguf.GGML_QUANT_SIZES[qtype]
    rows, cols = source.shape
    if cols % block_size:
        raise ValueError(f"{qtype.name} requires width {cols} divisible by {block_size}")
    output = n.empty(rows * (cols // block_size) * type_size, dtype=n.uint8)
    source = n.ascontiguousarray(source, dtype=n.float32)
    weights = n.ascontiguousarray(weights, dtype=n.float32)
    written = int(function(int(qtype), source.reshape(-1).ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                           ctypes.c_void_p(output.ctypes.data), 0, rows, cols,
                           weights.ctypes.data_as(ctypes.POINTER(ctypes.c_float))))
    if written != output.nbytes:
        raise ValueError(f"native {qtype.name} size mismatch: {written} != {output.nbytes}")
    packed = output.tobytes()
    dequant = gguf.dequantize(output.reshape(rows, -1), qtype)
    if dequant.shape != source.shape or not n.isfinite(dequant).all():
        raise ValueError(f"native {qtype.name} dequantized shape mismatch: {dequant.shape}")
    return packed, n.ascontiguousarray(dequant, dtype=n.float32)


def q4_weight(reader: Any, name: str, expected_shape: tuple[int, int]) -> Any:
    n = require_numpy()
    tensor = next((item for item in reader.tensors if item.name == name), None)
    if tensor is None or tensor.tensor_type != gguf.GGMLQuantizationType.Q4_K:
        raise ValueError(f"Q4_K tensor missing: {name}")
    result = gguf.dequantize(tensor.data, tensor.tensor_type)
    if result.shape != expected_shape or not n.isfinite(result).all():
        raise ValueError(f"Q4_K shape mismatch for {name}: {result.shape}")
    return n.ascontiguousarray(result, dtype=n.float32)


def screen(
    capture_receipt_path: Path,
    master_manifest_path: Path,
    master_cache: Path,
    q4_model: Path,
    quantizer_path: Path,
) -> dict[str, Any]:
    n = require_numpy()
    receipt = json.loads(capture_receipt_path.read_text(encoding="utf-8"))
    if receipt.get("status") != "verified":
        raise ValueError("expanded capture receipt is not verified")
    fixtures = {item["id"]: item for item in receipt["capture"]["fixtures"]}
    if set(fixtures) != set(EXPECTED_FIXTURES):
        raise ValueError(f"expected only {EXPECTED_FIXTURES} expanded fixtures, got {sorted(fixtures)}")
    captures: dict[str, Path] = {}
    capture_provenance = []
    loaded: dict[str, dict[str, Any]] = {}
    for fixture_id in EXPECTED_FIXTURES:
        item = fixtures[fixture_id]
        directory = Path(item["capture_dir"]).expanduser()
        blob = directory / "activations.f32"
        metadata = directory / "metadata.jsonl"
        if hash_file(blob) != item["blob_sha256"] or hash_file(metadata) != item["metadata_sha256"]:
            raise ValueError(f"expanded capture hash changed: {fixture_id}")
        validation = validate_capture(directory, full=True)
        expected = {"frames": item["frames"], "tensors": item["tensor_records"], "blob_bytes": item["blob_bytes"]}
        if validation != expected:
            raise ValueError(f"expanded capture validation changed: {fixture_id}: {validation}")
        header = json.loads(metadata.read_text(encoding="utf-8").splitlines()[0])
        if header.get("capture_frames") not in (None, list(EXPECTED_FRAMES)) or header.get("branches") != 1:
            raise ValueError(f"unexpected expanded capture header: {fixture_id}")
        captures[fixture_id] = directory
        capture_provenance.append({
            "id": fixture_id,
            "split": "calibration_imatrix" if fixture_id == "cal-en-long-123" else "diagnostic_evaluation",
            "path": str(directory),
            "frames": item["frames"],
            "blob_bytes": item["blob_bytes"],
            "tensor_records": item["tensor_records"],
            "activations_sha256": item["blob_sha256"],
            "metadata_sha256": item["metadata_sha256"],
        })

    manifest = json.loads(master_manifest_path.read_text(encoding="utf-8"))
    selected = manifest.get("selected", {})
    if selected.get("dtype") != "BF16" or hash_file(master_cache) != selected.get("range_sha256"):
        raise ValueError("BF16 master cache provenance mismatch")
    if not q4_model.is_file():
        raise ValueError(f"missing Q4_K model: {q4_model}")
    master_entries = {role: load_master_entry(manifest, spec["gguf_name"]) for role, spec in ROLES.items()}

    # Keep the expanded fixture matrices in memory; each is only 80 x 8192.
    for fixture_id, capture in captures.items():
        loaded[fixture_id] = {spec["capture_tensor"]: load_capture(capture, spec["capture_tensor"]) for spec in ROLES.values()}
    calibration_id, diagnostic_id = EXPECTED_FIXTURES

    library, function = native_quantizer(quantizer_path)
    qtypes = (gguf.GGMLQuantizationType.IQ2_S, gguf.GGMLQuantizationType.Q3_K)
    native_results: list[dict[str, Any]] = []
    identity_rows: list[dict[str, Any]] = []
    q4_rows: list[dict[str, Any]] = []
    conversion_rows: list[dict[str, Any]] = []
    packed_sizes: dict[str, dict[str, int]] = {qtype.name: {} for qtype in qtypes}
    reader = gguf.GGUFReader(q4_model)
    try:
        for role, spec in ROLES.items():
            entry = master_entries[role]
            weight, conversion = load_bf16_f16(entry, master_cache)
            expected_shape = tuple(int(value) for value in entry["shape"])
            if weight.shape != expected_shape:
                raise ValueError(f"master shape mismatch for {role}: {weight.shape} != {expected_shape}")
            conversion_rows.append({"role": role, **conversion})
            xcal = loaded[calibration_id][spec["capture_tensor"]]
            xdiag = loaded[diagnostic_id][spec["capture_tensor"]]
            if xcal.shape[1] != weight.shape[1] or xdiag.shape[1] != weight.shape[1]:
                raise ValueError(f"activation width mismatch for {role}")

            # Float64 establishes the identity independently of native quantization.
            weight64 = weight.astype(n.float64)
            xcal64 = xcal.astype(n.float64)
            xdiag64 = xdiag.astype(n.float64)
            weight_rot64 = fwht_rows(weight64)
            xcal_rot64 = fwht_rows(xcal64)
            xdiag_rot64 = fwht_rows(xdiag64)
            ycal_ref64 = xcal64 @ weight64.T
            ydiag_ref64 = xdiag64 @ weight64.T
            identity_rows.append({
                "role": role,
                "shape": list(weight.shape),
                "calibration_rows": int(xcal.shape[0]),
                "diagnostic_rows": int(xdiag.shape[0]),
                "calibration": projection_metrics(xcal_rot64 @ weight_rot64.T, ycal_ref64),
                "diagnostic": projection_metrics(xdiag_rot64 @ weight_rot64.T, ydiag_ref64),
                "self_inverse_max_abs": float(n.max(n.abs(fwht_rows(fwht_rows(xcal64[:1])) - xcal64[:1]))),
            })

            weight_rot = n.ascontiguousarray(weight_rot64, dtype=n.float32)
            xcal_rot = n.ascontiguousarray(xcal_rot64, dtype=n.float32)
            xdiag_rot = n.ascontiguousarray(xdiag_rot64, dtype=n.float32)
            x_stats = {
                "unrotated_calibration": activation_stats(xcal),
                "rotated_calibration": activation_stats(xcal_rot),
                "unrotated_diagnostic": activation_stats(xdiag),
                "rotated_diagnostic": activation_stats(xdiag_rot),
            }
            imatrix_rows = {
                "unrotated": normalized_imatrix(xcal),
                "rotated": normalized_imatrix(xcal_rot),
            }
            q4 = q4_weight(reader, spec["gguf_name"], weight.shape)
            q4_rows.append({
                "role": role,
                "format": "Q4_K existing depth control",
                "calibration": projection_metrics(xcal @ q4.T, xcal @ weight.T),
                "diagnostic": projection_metrics(xdiag @ q4.T, xdiag @ weight.T),
            })

            for qtype in qtypes:
                for rotation_name, source, cal_x, diag_x, imatrix in (
                    ("unrotated", weight, xcal, xdiag, imatrix_rows["unrotated"]),
                    ("hadamard", weight_rot, xcal_rot, xdiag_rot, imatrix_rows["rotated"]),
                ):
                    weights, imatrix_stats = imatrix
                    packed, dequant = quantize(function, qtype, source, weights)
                    ycal_ref = xcal @ weight.T
                    ydiag_ref = xdiag @ weight.T
                    packed_sizes[qtype.name][f"{role}:{rotation_name}"] = len(packed)
                    native_results.append({
                        "role": role,
                        "format": qtype.name,
                        "rotation": rotation_name,
                        "shape": list(weight.shape),
                        "packed_bytes": len(packed),
                        "packed_sha256": hashlib.sha256(packed).hexdigest(),
                        "imatrix": imatrix_stats,
                        "calibration": projection_metrics(cal_x @ dequant.T, ycal_ref),
                        "diagnostic": projection_metrics(diag_x @ dequant.T, ydiag_ref),
                        "weight_basis_projection": projection_metrics(dequant, source),
                        "activation_stats": x_stats,
                    })
            del weight, weight64, weight_rot64, weight_rot, xcal64, xdiag64, xcal_rot64, xdiag_rot64, xcal_rot, xdiag_rot
    finally:
        free = getattr(library, "ggml_quantize_free", None)
        if free is not None:
            free()

    for qtype in qtypes:
        sizes = [size for key, size in packed_sizes[qtype.name].items() if key.startswith("gate:")]
        if not sizes or len(set(sizes)) != 1:
            raise ValueError(f"unexpected same-shape gate/up packed sizes for {qtype.name}")
        for role in ROLES:
            sibling = {packed_sizes[qtype.name][f"{role}:{variant}"] for variant in ("unrotated", "hadamard")}
            if len(sibling) != 1:
                raise ValueError(f"rotation changed packed byte size for {qtype.name}/{role}")

    identity_max = max(
        max(row["calibration"]["rel_l2"] for row in identity_rows),
        max(row["diagnostic"]["rel_l2"] for row in identity_rows),
    )
    source_hashes = {
        "script_sha256": hash_file(Path(__file__).resolve()),
        "capture_receipt": {"path": str(capture_receipt_path), "sha256": hash_file(capture_receipt_path)},
        "master_manifest": {"path": str(master_manifest_path), "sha256": hash_file(master_manifest_path)},
        "master_cache": {"path": str(master_cache), "bytes": master_cache.stat().st_size, "sha256": hash_file(master_cache)},
        "q4_model": {"path": str(q4_model), "bytes": q4_model.stat().st_size, "sha256": hash_file(q4_model)},
        "quantizer": {"path": str(quantizer_path), "bytes": quantizer_path.stat().st_size, "sha256": hash_file(quantizer_path)},
    }
    return {
        "schema": "breeze-rotation-screen/v1",
        "status": "diagnostic_complete",
        "objective": "fixed plain normalized Walsh-Hadamard projection screen on layer 5 depth FFN",
        "scope": {
            "layer": LAYER,
            "roles": list(ROLES),
            "formats": [qtype.name for qtype in qtypes],
            "native_quantizations": len(native_results),
            "calibration_fixture": calibration_id,
            "diagnostic_fixture": diagnostic_id,
            "diagnostic_is_not_final_heldout": True,
            "source_pipeline": "BF16 master -> F16 -> F32",
            "rotation": "X' = X H and W' = W H, with H the plain normalized Walsh-Hadamard matrix",
        },
        "provenance": {
            "sources": source_hashes,
            "captures": capture_provenance,
            "master_tensors": conversion_rows,
        },
        "float_identity": {
            "rows": identity_rows,
            "max_rel_l2": identity_max,
            "passed": identity_max < 1e-12,
            "tolerance": 1e-12,
        },
        "q4_unrotated_reference": q4_rows,
        "native_results": native_results,
        "packed_bytes": {
            "sizes_by_format_and_role_rotation": packed_sizes,
            "rotation_preserves_size_within_format": True,
        },
        "decision": {
            "deployment_recommendation": "diagnostic_only",
            "quality_or_runtime_claim": False,
            "interpretation": "This screen measures local projection error under native packed quantization. It does not test runtime rotation cost, Vulkan support, free-running speech, or final held-out quality.",
        },
        "caveats": [
            "English rows alone form the imatrix; Mandarin rows are a paired diagnostic evaluation within the expanded calibration corpus.",
            "The plain fixed rotation has no seed search, learned parameters, GGUF format, or runtime transform implementation.",
            "Q4_K is an unrotated dequantized control measured against the F16-rounded BF16 source, not a new quantization result.",
            "Packed byte sizes are compared within each native format; packed contents are expected to differ after rotation.",
        ],
    }


def self_test() -> None:
    n = require_numpy()
    rng = n.random.default_rng(20260907)
    for width in (16, 1024, 8192):
        rows = rng.normal(size=(2, width)).astype(n.float64)
        recovered = fwht_rows(fwht_rows(rows))
        assert float(n.max(n.abs(recovered - rows))) < 2e-12
    print("depth_rotation_screen self-test: PASS")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-receipt", type=Path, default=DEFAULT_CAPTURE_RECEIPT)
    parser.add_argument("--master-manifest", type=Path, default=DEFAULT_MASTER_MANIFEST)
    parser.add_argument("--master-cache", type=Path, default=DEFAULT_MASTER_CACHE)
    parser.add_argument("--q4-model", type=Path, default=DEFAULT_Q4_MODEL)
    parser.add_argument("--quantizer", type=Path, default=DEFAULT_QUANTIZER)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.self_test:
            self_test()
            return 0
        result = screen(
            args.capture_receipt.expanduser().resolve(),
            args.master_manifest.expanduser().resolve(),
            args.master_cache.expanduser().resolve(),
            args.q4_model.expanduser().resolve(),
            args.quantizer.expanduser().resolve(),
        )
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
        print(json.dumps({"output": str(output), "status": result["status"], "native_quantizations": result["scope"]["native_quantizations"]}))
        return 0
    except (AssertionError, OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"depth_rotation_screen: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
