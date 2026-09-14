#!/usr/bin/env python3
"""Build the native activation-weighted Q3_K depth FFN control.

The capture rows are used only to form per-input-channel mean-square weights.
The GGUF is based on the verified selective control so its Q4_K attention and
all non-FFN payloads remain byte-identical to the Q4_K control.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any

import gguf
import numpy as np

from build_depth_quant_controls import MAPPING, metadata
from depth_capture_check import validate as validate_capture
from depth_quant_screen import hash_file, hash_region


ROOT = Path(__file__).resolve().parents[1]
CACHE = Path("~/.cache/breeze-models/depth-quant-controls").expanduser()
Q3_PATH = CACHE / "breeze-depth-blocks-q3_k.gguf"
Q4_PATH = CACHE / "breeze-depth-blocks-q4_k.gguf"
SELECTIVE_PATH = CACHE / "breeze-depth-ffn-q3_k-attn-q4_k.gguf"
OUTPUT_PATH = CACHE / "breeze-depth-ffn-q3_k-calibrated-attn-q4_k.gguf"
PART_PATH = OUTPUT_PATH.with_suffix(".gguf.part")
RECEIPT_PATH = ROOT / "benchmarks/depth-corpus/quant-calibrated.json"
CAPTURE_RECEIPT = ROOT / "benchmarks/depth-corpus/capture-expanded.json"
MASTER_PATH = Path("~/.cache/breeze-models/depth-master/depth-range.bin").expanduser()
MASTER_MANIFEST = ROOT / "benchmarks/depth-corpus/master-manifest.json"
QUANTIZER = ROOT / "build-ffn-bench/third_party/ggml/src/libggml-base.0.22.0.dylib"
Q3_TYPE = gguf.GGMLQuantizationType.Q3_K
Q4_TYPE = gguf.GGMLQuantizationType.Q4_K
EXPECTED_LAYERS = 12
EXPECTED_STEPS = 15
EXPECTED_FRAMES = [0, 7, 15, 31, 63]
EXPECTED_TENSORS = 1161
FFN_NAMES = frozenset(name for name in MAPPING if ".ffn_" in name)
ATTENTION_NAMES = frozenset(name for name in MAPPING if ".attn_" in name)
OTHER_COUNT = EXPECTED_TENSORS - len(FFN_NAMES) - len(ATTENTION_NAMES)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _source(path: Path, digest: str | None = None) -> dict[str, Any]:
    actual = hash_file(path)
    if digest is not None and actual != digest:
        raise AssertionError(f"{path} hash changed: {actual} != {digest}")
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": actual}


def _load_bf16_f16(entry: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    shape = tuple(int(value) for value in entry["shape"])
    count = int(np.prod(shape))
    if entry.get("dtype") != "BF16" or int(entry["nbytes"]) != count * 2:
        raise AssertionError(f"bad BF16 master entry: {entry}")
    offset = int(entry["local_offset"])
    if offset < 0 or offset + count * 2 > MASTER_PATH.stat().st_size:
        raise AssertionError(f"master range outside file: {entry['name']}")
    raw = np.memmap(MASTER_PATH, dtype="<u2", mode="r", offset=offset, shape=count)
    bits = raw.astype(np.uint32) << 16
    bf32 = bits.view(np.float32)
    if not np.isfinite(bf32).all():
        raise AssertionError(f"non-finite BF16 master tensor: {entry['name']}")
    f16 = bf32.astype(np.float16)
    if not np.isfinite(f16).all():
        raise AssertionError(f"F16 overflow in {entry['name']}")
    changed = int(np.count_nonzero(f16.astype(np.float32) != bf32))
    delta = f16.astype(np.float64) - bf32.astype(np.float64)
    error = {
        "master_name": entry["name"],
        "changed_values": changed,
        "max_abs": float(np.max(np.abs(delta))),
        "relative_mse": float(np.sum(delta * delta) /
                               np.sum(bf32.astype(np.float64) ** 2)),
    }
    src = np.ascontiguousarray(f16.astype(np.float32))
    del raw, bits, bf32, f16, delta
    return src, error


class _Stats:
    def __init__(self, width: int) -> None:
        self.sumsq = np.zeros(width, dtype=np.float64)
        self.samples = 0
        self.records = 0

    def add(self, values: np.ndarray, tokens: int) -> None:
        matrix = values.astype(np.float64, copy=False).reshape(tokens, -1)
        self.sumsq += np.sum(matrix * matrix, axis=0, dtype=np.float64)
        self.samples += tokens
        self.records += 1


def _capture_stats() -> tuple[dict[tuple[int, str], _Stats], dict[str, Any]]:
    receipt = json.loads(CAPTURE_RECEIPT.read_text())
    if receipt.get("status") != "verified":
        raise AssertionError("capture receipt is not verified")
    expected = {"cal-en-long-123", "cal-zh-long-123"}
    fixtures = receipt["capture"]["fixtures"]
    if {item["id"] for item in fixtures} != expected:
        raise AssertionError("capture receipt does not contain exactly the calibration fixtures")
    stats = {(layer, role): _Stats(1024 if role == "gate_up" else 8192)
             for layer in range(EXPECTED_LAYERS) for role in ("gate_up", "down")}
    capture_evidence = []
    for fixture in sorted(fixtures, key=lambda item: item["id"]):
        directory = Path(fixture["capture_dir"])
        blob_path = directory / "activations.f32"
        meta_path = directory / "metadata.jsonl"
        if hash_file(blob_path) != fixture["blob_sha256"]:
            raise AssertionError(f"capture blob hash changed: {directory}")
        if hash_file(meta_path) != fixture["metadata_sha256"]:
            raise AssertionError(f"capture metadata hash changed: {directory}")
        result = validate_capture(directory, full=True)
        expected_result = {"frames": fixture["frames"], "tensors": fixture["tensor_records"],
                           "blob_bytes": fixture["blob_bytes"]}
        if result != expected_result:
            raise AssertionError(f"capture validation differs from receipt: {directory}")
        header = json.loads(meta_path.read_text().splitlines()[0])
        if (header.get("capture_frames") != EXPECTED_FRAMES or
                header.get("capture_layers") != list(range(EXPECTED_LAYERS)) or
                header.get("branches") != 1):
            raise AssertionError(f"capture selection differs from calibration contract: {directory}")
        blob = blob_path.read_bytes()
        seen: set[tuple[int, int, int, str]] = set()
        for line in meta_path.read_text().splitlines():
            record = json.loads(line)
            if record.get("kind") != "tensor" or record.get("tensor") not in {
                    "ffn_norm_input", "post_silu_times_up"}:
                continue
            frame = record["frame"]
            step = record["step"]
            layer = record["layer"]
            name = record["tensor"]
            role = "gate_up" if name == "ffn_norm_input" else "down"
            key = (frame, step, layer, name)
            if key in seen:
                raise AssertionError(f"duplicate calibration coordinate: {key}")
            seen.add(key)
            width = 1024 if role == "gate_up" else 8192
            tokens = 2 if step == 1 else 1
            if record["shape"] != [width, tokens, 1] or record["nbytes"] != width * tokens * 4:
                raise AssertionError(f"unexpected calibration shape: {record}")
            values = np.frombuffer(blob, dtype="<f4", count=width * tokens,
                                   offset=int(record["offset"]))
            if not np.isfinite(values).all():
                raise AssertionError(f"non-finite calibration tensor: {key}")
            stats[(layer, role)].add(values, tokens)
        capture_evidence.append({
            "id": fixture["id"],
            "capture_dir": str(directory),
            "frames": fixture["frames"],
            "layers": list(range(EXPECTED_LAYERS)),
            "blob_bytes": fixture["blob_bytes"],
            "metadata_bytes": fixture["metadata_bytes"],
            "blob_sha256": fixture["blob_sha256"],
            "metadata_sha256": fixture["metadata_sha256"],
            "validated": result,
        })
    for key, item in stats.items():
        expected_records = 2 * len(EXPECTED_FRAMES) * EXPECTED_STEPS
        expected_samples = 2 * len(EXPECTED_FRAMES) * (2 + EXPECTED_STEPS - 1)
        if item.samples != expected_samples or item.records != expected_records:
            raise AssertionError(f"unexpected calibration count for {key}: {item.samples}/{item.records}")
        if not np.isfinite(item.sumsq).all() or np.any(item.sumsq <= 0):
            raise AssertionError(f"invalid calibration statistics for {key}")
    return stats, {"fixtures": capture_evidence, "frames": EXPECTED_FRAMES,
                   "layers": list(range(EXPECTED_LAYERS)), "steps": EXPECTED_STEPS}


def _weights(stats: _Stats) -> tuple[np.ndarray, dict[str, Any]]:
    mean_square = stats.sumsq / float(stats.samples)
    if not np.isfinite(mean_square).all() or np.any(mean_square <= 0):
        raise AssertionError("mean-square activation weights must be finite and positive")
    normalization = float(np.mean(mean_square, dtype=np.float64))
    normalized = mean_square / normalization
    if not np.isfinite(normalized).all() or np.any(normalized <= 0):
        raise AssertionError("normalized activation weights must be finite and positive")
    normalized32 = np.ascontiguousarray(normalized.astype(np.float32))
    return normalized32, {
        "sample_count": stats.samples,
        "record_count": stats.records,
        "input_channels": len(mean_square),
        "mean_square_mean": float(np.mean(mean_square, dtype=np.float64)),
        "mean_square_min": float(np.min(mean_square)),
        "mean_square_max": float(np.max(mean_square)),
        "normalization": "divide by arithmetic mean of per-input-channel mean square",
        "normalization_factor": normalization,
        "floor": 0.0,
        "mean_square_sha256": _sha256_bytes(mean_square.tobytes()),
        "normalized_weight_sha256": _sha256_bytes(normalized32.tobytes()),
    }


def _quantizer() -> tuple[Any, Any]:
    if not QUANTIZER.is_file():
        raise AssertionError(f"missing native quantizer library: {QUANTIZER}")
    library = ctypes.CDLL(str(QUANTIZER))
    function = library.ggml_quantize_chunk
    function.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_float), ctypes.c_void_p,
                         ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
                         ctypes.POINTER(ctypes.c_float)]
    function.restype = ctypes.c_size_t
    return library, function


def _quantize(function: Any, source: np.ndarray, tensor: Any,
              weights: np.ndarray | None) -> bytes:
    n_per_row = int(tensor.shape[0])
    nrows = int(tensor.shape[1])
    expected = int(tensor.n_bytes)
    if source.size != n_per_row * nrows:
        raise AssertionError(f"source shape mismatch for {tensor.name}")
    output = np.empty(expected, dtype=np.uint8)
    source_ptr = source.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    weight_ptr = None if weights is None else weights.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    written = int(function(Q3_TYPE, source_ptr, ctypes.c_void_p(output.ctypes.data),
                           0, nrows, n_per_row, weight_ptr))
    if written != expected:
        raise AssertionError(f"native Q3_K size mismatch for {tensor.name}: {written} != {expected}")
    return output.tobytes()


def _clone_file(source: Path, destination: Path) -> str:
    libc = ctypes.CDLL(None)
    clone = getattr(libc, "clonefile", None)
    if clone is not None:
        clone.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint32]
        clone.restype = ctypes.c_int
        if clone(os.fsencode(source), os.fsencode(destination), 0) == 0:
            return "clonefile"
    shutil.copyfile(source, destination)
    return "copyfile"


def _open_sources() -> tuple[Any, Any, Any, dict[str, Any], dict[str, Any]]:
    if not all(path.is_file() for path in (Q3_PATH, Q4_PATH, SELECTIVE_PATH, MASTER_PATH)):
        raise AssertionError("missing quantization source or BF16 master")
    q3 = gguf.GGUFReader(Q3_PATH)
    q4 = gguf.GGUFReader(Q4_PATH)
    selective = gguf.GGUFReader(SELECTIVE_PATH)
    if metadata(q3) != metadata(q4) or metadata(q3) != metadata(selective):
        raise AssertionError("source metadata differs")
    names = [tensor.name for tensor in q3.tensors]
    if names != [tensor.name for tensor in q4.tensors] or names != [tensor.name for tensor in selective.tensors]:
        raise AssertionError("source tensor order differs")
    if len(names) != EXPECTED_TENSORS or len(FFN_NAMES) != 36 or len(ATTENTION_NAMES) != 48:
        raise AssertionError("unexpected depth tensor counts")
    for before, attention, mixed in zip(q3.tensors, q4.tensors, selective.tensors):
        if not (np.array_equal(before.shape, attention.shape) and np.array_equal(before.shape, mixed.shape)):
            raise AssertionError(f"source shape differs: {before.name}")
        if before.name in FFN_NAMES and before.tensor_type != Q3_TYPE:
            raise AssertionError(f"Q3 source type mismatch: {before.name}")
        if before.name in ATTENTION_NAMES and attention.tensor_type != Q4_TYPE:
            raise AssertionError(f"Q4 attention source type mismatch: {before.name}")
    sources = {
        "q3_k": _source(Q3_PATH),
        "q4_k": _source(Q4_PATH),
        "selective_q3_ffn_q4_attention": _source(SELECTIVE_PATH),
        "bf16_master": _source(MASTER_PATH),
        "master_manifest": _source(MASTER_MANIFEST),
        "quantizer": _source(QUANTIZER),
    }
    return q3, q4, selective, sources, {entry["name"]: entry for entry in json.loads(MASTER_MANIFEST.read_text())["tensors"]}


def _verify_output(q4: Any, candidate_path: Path, expected_ffn: dict[str, str] | None = None) -> dict[str, Any]:
    candidate = gguf.GGUFReader(candidate_path)
    if metadata(q4) != metadata(candidate):
        raise AssertionError("GGUF metadata changed")
    if [tensor.name for tensor in q4.tensors] != [tensor.name for tensor in candidate.tensors]:
        raise AssertionError("tensor order changed")
    attention = 0
    ffn = 0
    other = 0
    tensor_hashes = []
    for source, after in zip(q4.tensors, candidate.tensors):
        if not np.array_equal(source.shape, after.shape):
            raise AssertionError(f"tensor shape changed: {after.name}")
        actual = hash_region(candidate_path, int(after.data_offset), int(after.n_bytes))
        if after.name in FFN_NAMES:
            if after.tensor_type != Q3_TYPE:
                raise AssertionError(f"calibrated FFN type mismatch: {after.name}")
            ffn += 1
            if expected_ffn is not None and actual != expected_ffn[after.name]:
                raise AssertionError(f"calibrated FFN bytes changed: {after.name}")
        else:
            if after.tensor_type != source.tensor_type:
                raise AssertionError(f"unchanged tensor type changed: {after.name}")
            source_hash = hash_region(Q4_PATH, int(source.data_offset), int(source.n_bytes))
            if actual != source_hash:
                raise AssertionError(f"unchanged tensor bytes changed: {after.name}")
            if after.name in ATTENTION_NAMES:
                attention += 1
            else:
                other += 1
        if after.name in FFN_NAMES:
            tensor_hashes.append({"name": after.name, "sha256": actual,
                                  "bytes": int(after.n_bytes), "shape": after.shape.tolist(),
                                  "type": after.tensor_type.name})
    if (ffn, attention, other) != (36, 48, OTHER_COUNT):
        raise AssertionError(f"unexpected output tensor counts: {ffn}/{attention}/{other}")
    return {
        "path": str(candidate_path),
        "bytes": candidate_path.stat().st_size,
        "sha256": hash_file(candidate_path),
        "tensor_count": len(candidate.tensors),
        "ffn_tensor_count": ffn,
        "attention_tensor_count": attention,
        "other_tensor_count": other,
        "ffn_type": Q3_TYPE.name,
        "attention_type": Q4_TYPE.name,
        "attention_and_other_byte_identical_to_q4": True,
        "metadata_identical": True,
        "tensor_name_order_identical": True,
        "tensor_shapes_identical": True,
        "ffn_tensors": tensor_hashes,
    }


def _receipt_stats(stats: dict[tuple[int, str], _Stats]) -> list[dict[str, Any]]:
    result = []
    for (layer, role), item in sorted(stats.items()):
        _, evidence = _weights(item)
        result.append({"layer": layer, "role": role, **evidence})
    return result


def verify_artifact() -> dict[str, Any]:
    if not RECEIPT_PATH.is_file() or not OUTPUT_PATH.is_file():
        raise AssertionError("calibrated output and receipt must both exist")
    receipt = json.loads(RECEIPT_PATH.read_text())
    if receipt.get("schema") != "breeze-depth-calibrated/v1":
        raise AssertionError("unexpected calibrated receipt schema")
    if receipt.get("script_sha256") != hash_file(Path(__file__)):
        raise AssertionError("builder script changed since receipt")
    stats, calibration = _capture_stats()
    if receipt.get("calibration") != calibration:
        raise AssertionError("capture provenance changed")
    if receipt.get("activation_statistics") != _receipt_stats(stats):
        raise AssertionError("activation statistics changed")
    q3, q4, selective, sources, _ = _open_sources()
    if receipt.get("sources") != sources:
        raise AssertionError("source provenance changed")
    expected = {item["name"]: item["sha256"] for item in receipt["output"]["ffn_tensors"]}
    verification = _verify_output(q4, OUTPUT_PATH, expected)
    if receipt.get("output") != verification:
        raise AssertionError("output verification changed")
    if receipt.get("base_output") != str(SELECTIVE_PATH):
        raise AssertionError("base output provenance changed")
    del q3, q4, selective
    return verification


def build() -> dict[str, Any]:
    if OUTPUT_PATH.exists() or RECEIPT_PATH.exists():
        if not (OUTPUT_PATH.exists() and RECEIPT_PATH.exists()):
            raise AssertionError("refusing unreceipted calibrated artifact")
        result = verify_artifact()
        print(f"Existing calibrated control verified: {result['sha256']}")
        return result
    if PART_PATH.exists():
        raise AssertionError(f"refusing stale temporary artifact: {PART_PATH}")

    stats, calibration = _capture_stats()
    q3, q4, selective, sources, entries = _open_sources()
    stats_manifest = _receipt_stats(stats)
    library, function = _quantizer()
    expected_ffn: dict[str, str] = {}
    unweighted = []
    weighted = []
    conversion_errors = []
    clone_method = _clone_file(SELECTIVE_PATH, PART_PATH)
    try:
        candidate = gguf.GGUFReader(PART_PATH)
        candidate_by_name = {tensor.name: tensor for tensor in candidate.tensors}
        with PART_PATH.open("r+b") as stream:
            for tensor in q3.tensors:
                if tensor.name not in FFN_NAMES:
                    continue
                master_name = MAPPING[tensor.name]
                entry = entries[master_name]
                if list(reversed(entry["shape"])) != tensor.shape.tolist():
                    raise AssertionError(f"master shape differs: {tensor.name}")
                if hash_region(MASTER_PATH, int(entry["local_offset"]), int(entry["nbytes"])) != entry["sha256"]:
                    raise AssertionError(f"master tensor hash differs: {master_name}")
                source, error = _load_bf16_f16(entry)
                error["name"] = tensor.name
                conversion_errors.append(error)
                ordinary = _quantize(function, source, tensor, None)
                if ordinary != bytes(tensor.data):
                    raise AssertionError(f"native unweighted Q3_K mismatch: {tensor.name}")
                unweighted.append(tensor.name)
                layer = int(tensor.name.split(".")[2])
                role = "down" if tensor.name.endswith("ffn_down.weight") else "gate_up"
                weights, _ = _weights(stats[(layer, role)])
                packed = _quantize(function, source, tensor, weights)
                if len(packed) != int(tensor.n_bytes):
                    raise AssertionError(f"weighted packed size mismatch: {tensor.name}")
                expected_ffn[tensor.name] = _sha256_bytes(packed)
                weighted.append({"name": tensor.name, "sha256": expected_ffn[tensor.name],
                                 "bytes": int(tensor.n_bytes), "shape": tensor.shape.tolist(),
                                 "type": Q3_TYPE.name})
                stream.seek(int(candidate_by_name[tensor.name].data_offset))
                stream.write(packed)
                del source
    finally:
        library.ggml_quantize_free()

    if len(unweighted) != 36 or len(weighted) != 36:
        raise AssertionError("did not verify all 36 FFN tensors")
    try:
        verification = _verify_output(q4, PART_PATH,
                                      expected_ffn)
        os.replace(PART_PATH, OUTPUT_PATH)
        verification["path"] = str(OUTPUT_PATH)
        verification["sha256"] = hash_file(OUTPUT_PATH)
    except Exception:
        raise

    receipt = {
        "schema": "breeze-depth-calibrated/v1",
        "base_output": str(SELECTIVE_PATH),
        "sources": sources,
        "calibration": calibration,
        "activation_statistics": stats_manifest,
        "normalization": {
            "statistic": "float64 mean of squared captured values per input channel",
            "weight": "mean_square / arithmetic mean(mean_square)",
            "floor": 0.0,
        },
        "quantizer": {
            "library": str(QUANTIZER),
            "api": "ggml_quantize_chunk(GGML_TYPE_Q3_K, ..., imatrix)",
            "unweighted_matches_q3_tensor_bytes": len(unweighted) == 36,
            "unweighted_tensor_count": len(unweighted),
            "weighted_tensor_count": len(weighted),
            "expected_packed_bytes": int(sum(item["bytes"] for item in weighted)),
        },
        "f16_intermediate": conversion_errors,
        "output": verification,
        "script_sha256": hash_file(Path(__file__)),
        "clone_method": clone_method,
        "caveats": [
            "Calibration activations are sparse Q4-generated teacher data from two capped unsplit fixtures.",
            "FFN sources are verified BF16 master tensors rounded through an F16 intermediate.",
            "This is calibration provenance only; no speech-quality or runtime acceptance is implied.",
        ],
    }
    receipt_part = RECEIPT_PATH.with_suffix(".json.part")
    if receipt_part.exists():
        raise AssertionError(f"refusing stale receipt temporary artifact: {receipt_part}")
    receipt_part.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    os.replace(receipt_part, RECEIPT_PATH)
    print(f"Built and verified calibrated control: {verification['sha256']}")
    return verification


def self_test() -> None:
    assert len(MAPPING) == 84
    assert len(FFN_NAMES) == 36
    assert len(ATTENTION_NAMES) == 48
    fake = _Stats(4)
    fake.sumsq[:] = [1.0, 2.0, 3.0, 4.0]
    fake.samples = 1
    fake.records = 1
    weights, evidence = _weights(fake)
    assert np.allclose(weights, [0.4, 0.8, 1.2, 1.6])
    assert evidence["floor"] == 0.0
    print("calibrated builder self-test: PASS")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
    elif args.verify:
        result = verify_artifact()
        print(f"Calibrated control verified: {result['sha256']}")
    else:
        build()
