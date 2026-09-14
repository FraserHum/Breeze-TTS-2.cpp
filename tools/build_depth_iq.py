#!/usr/bin/env python3
"""Build and verify a calibrated native IQ depth FFN control.

The source weights and calibration statistics intentionally reuse the verified
Q3_K builder.  Only the 36 depth FFN tensors change; the Q4_K attention and
all other Q4 payloads are copied through a fresh GGUF writer.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any

import gguf
import numpy as np

from build_depth_calibrated import (
    ATTENTION_NAMES,
    EXPECTED_TENSORS,
    FFN_NAMES,
    MASTER_MANIFEST,
    MASTER_PATH,
    MAPPING,
    Q4_PATH,
    QUANTIZER,
    _capture_stats,
    _load_bf16_f16,
    _receipt_stats,
    _source,
    _weights,
)
from build_depth_quant_controls import metadata
from depth_quant_screen import hash_file, hash_region


ROOT = Path(__file__).resolve().parents[1]
CACHE = Path("~/.cache/breeze-models/depth-quant-controls").expanduser()
EXPECTED_FFN_COUNT = 36
EXPECTED_ATTENTION_COUNT = 48
EXPECTED_Q4_SHA256 = "49e4531863ccee2e4362fb06f86995f88659fb60bee37db45bfc662d5e9d497d"
CALIBRATED_BUILDER = ROOT / "tools/build_depth_calibrated.py"
DEFAULT_FORMAT = "iq3_xxs"
FORMAT_SPECS = {
    "iq3_xxs": {
        "type": gguf.GGMLQuantizationType.IQ3_XXS,
        "qk": 256,
        "bytes_per_block": 98,
        "expected_ffn_bytes": 115_605_504,
        "expected_per_projection_bytes": 3_211_264,
        "output": "breeze-depth-ffn-iq3_xxs-calibrated-attn-q4_k.gguf",
        "receipt": "quant-iq3_xxs.json",
        "schema": "breeze-depth-iq3-xxs/v1",
        "bytes_key": "iq3_xxs_ffn_bytes",
    },
    "iq2_s": {
        "type": gguf.GGMLQuantizationType.IQ2_S,
        "qk": 256,
        "bytes_per_block": 82,
        "expected_ffn_bytes": 96_731_136,
        "expected_per_projection_bytes": 2_686_976,
        "output": "breeze-depth-ffn-iq2_s-calibrated-attn-q4_k.gguf",
        "receipt": "quant-iq2_s.json",
        "schema": "breeze-depth-iq2-s/v1",
        "bytes_key": "iq2_s_ffn_bytes",
    },
}
IQ_TYPE: Any
QK: int
IQ_BYTES_PER_BLOCK: int
EXPECTED_FFN_BYTES: int
EXPECTED_PER_PROJECTION_BYTES: int
OUTPUT_PATH: Path
PART_PATH: Path
RECEIPT_PATH: Path
RECEIPT_PART: Path
ACTIVE_FORMAT = DEFAULT_FORMAT


def _configure_format(format_name: str) -> None:
    """Select the native format while keeping the IQ3 default unchanged."""
    global ACTIVE_FORMAT, IQ_TYPE, QK, IQ_BYTES_PER_BLOCK
    global EXPECTED_FFN_BYTES, EXPECTED_PER_PROJECTION_BYTES
    global OUTPUT_PATH, PART_PATH, RECEIPT_PATH, RECEIPT_PART
    if format_name not in FORMAT_SPECS:
        raise ValueError(f"unsupported native IQ format: {format_name}")
    spec = FORMAT_SPECS[format_name]
    ACTIVE_FORMAT = format_name
    IQ_TYPE = spec["type"]
    QK = int(spec["qk"])
    IQ_BYTES_PER_BLOCK = int(spec["bytes_per_block"])
    EXPECTED_FFN_BYTES = int(spec["expected_ffn_bytes"])
    EXPECTED_PER_PROJECTION_BYTES = int(spec["expected_per_projection_bytes"])
    OUTPUT_PATH = CACHE / str(spec["output"])
    PART_PATH = OUTPUT_PATH.with_suffix(".gguf.part")
    RECEIPT_PATH = ROOT / "benchmarks/depth-corpus" / str(spec["receipt"])
    RECEIPT_PART = RECEIPT_PATH.with_suffix(".json.part")


_configure_format(DEFAULT_FORMAT)


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
              weights: np.ndarray) -> np.ndarray:
    n_per_row = int(tensor.shape[0])
    nrows = int(tensor.shape[1])
    if n_per_row % QK != 0:
        raise AssertionError(f"{IQ_TYPE.name} row is not QK-aligned: {tensor.name}")
    expected = nrows * (n_per_row // QK) * IQ_BYTES_PER_BLOCK
    packed = np.empty(expected, dtype=np.uint8)
    source_ptr = source.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    weight_ptr = weights.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    written = int(function(int(IQ_TYPE), source_ptr, ctypes.c_void_p(packed.ctypes.data),
                           0, nrows, n_per_row, weight_ptr))
    if written != expected:
        raise AssertionError(f"native {IQ_TYPE.name} size mismatch for {tensor.name}: {written} != {expected}")
    return packed.reshape((nrows, expected // nrows))


def _add_metadata(writer: Any, reader: Any) -> None:
    for key, (types, value) in metadata(reader).items():
        writer.add_key_value(key, value, types[0], types[-1] if len(types) > 1 else None)


def _verify_output(base: Any, candidate_path: Path,
                   expected_ffn: dict[str, str] | None = None) -> dict[str, Any]:
    candidate = gguf.GGUFReader(candidate_path)
    if metadata(base) != metadata(candidate):
        raise AssertionError("GGUF metadata changed")
    if [tensor.name for tensor in base.tensors] != [tensor.name for tensor in candidate.tensors]:
        raise AssertionError("tensor name/order changed")

    ffn = 0
    attention = 0
    other = 0
    ffn_tensors: list[dict[str, Any]] = []
    dequant_finite = True
    for before, after in zip(base.tensors, candidate.tensors):
        if not np.array_equal(before.shape, after.shape):
            raise AssertionError(f"tensor shape changed: {after.name}")
        actual = hash_region(candidate_path, int(after.data_offset), int(after.n_bytes))
        if after.name in FFN_NAMES:
            if after.tensor_type != IQ_TYPE:
                raise AssertionError(f"{IQ_TYPE.name} type mismatch: {after.name}")
            if int(after.n_bytes) != EXPECTED_PER_PROJECTION_BYTES:
                raise AssertionError(f"{IQ_TYPE.name} bytes mismatch: {after.name}: {after.n_bytes}")
            if expected_ffn is not None and actual != expected_ffn[after.name]:
                raise AssertionError(f"{IQ_TYPE.name} bytes changed: {after.name}")
            dequant = gguf.dequantize(after.data, IQ_TYPE)
            if dequant.shape != tuple(reversed(after.shape.tolist())) or not np.isfinite(dequant).all():
                raise AssertionError(f"non-finite or malformed {IQ_TYPE.name} dequant: {after.name}")
            dequant_finite = dequant_finite and bool(np.isfinite(dequant).all())
            ffn_tensors.append({"name": after.name, "sha256": actual,
                                "bytes": int(after.n_bytes), "shape": after.shape.tolist(),
                                "type": after.tensor_type.name,
                                "dequant_shape": list(dequant.shape),
                                "dequant_finite": True})
            del dequant
            ffn += 1
        else:
            if after.tensor_type != before.tensor_type:
                raise AssertionError(f"unchanged tensor type changed: {after.name}")
            source_hash = hash_region(Q4_PATH, int(before.data_offset), int(before.n_bytes))
            if actual != source_hash:
                raise AssertionError(f"unchanged tensor bytes changed: {after.name}")
            if after.name in ATTENTION_NAMES:
                attention += 1
            else:
                other += 1

    if (ffn, attention, other) != (EXPECTED_FFN_COUNT, EXPECTED_ATTENTION_COUNT,
                                    EXPECTED_TENSORS - EXPECTED_FFN_COUNT - EXPECTED_ATTENTION_COUNT):
        raise AssertionError(f"unexpected tensor counts: {ffn}/{attention}/{other}")
    if sum(item["bytes"] for item in ffn_tensors) != EXPECTED_FFN_BYTES:
        raise AssertionError(f"unexpected total {IQ_TYPE.name} FFN bytes")
    return {
        "path": str(candidate_path),
        "bytes": candidate_path.stat().st_size,
        "sha256": hash_file(candidate_path),
        "tensor_count": len(candidate.tensors),
        "ffn_tensor_count": ffn,
        "attention_tensor_count": attention,
        "other_tensor_count": other,
        "ffn_type": IQ_TYPE.name,
        "attention_type": gguf.GGMLQuantizationType.Q4_K.name,
        FORMAT_SPECS[ACTIVE_FORMAT]["bytes_key"]: EXPECTED_FFN_BYTES,
        "attention_and_other_byte_identical_to_q4": True,
        "metadata_identical": True,
        "tensor_name_order_identical": True,
        "tensor_shapes_identical": True,
        "dequant_finite": dequant_finite,
        "ffn_tensors": ffn_tensors,
    }


def verify_artifact() -> dict[str, Any]:
    if not OUTPUT_PATH.is_file() or not RECEIPT_PATH.is_file():
        raise AssertionError(f"{IQ_TYPE.name} output and receipt must both exist")
    receipt = json.loads(RECEIPT_PATH.read_text())
    if receipt.get("schema") != FORMAT_SPECS[ACTIVE_FORMAT]["schema"]:
        raise AssertionError(f"unexpected {IQ_TYPE.name} receipt schema")
    # The receipt records the exact build recipe.  This verifier is intentionally
    # allowed to evolve while retaining strict source, calibration, tensor, and
    # output checks for older artifacts.
    stats, calibration = _capture_stats()
    if receipt.get("calibration") != calibration:
        raise AssertionError("calibration provenance changed")
    if receipt.get("activation_statistics") != _receipt_stats(stats):
        raise AssertionError("activation statistics changed")
    base = gguf.GGUFReader(Q4_PATH)
    if hash_file(Q4_PATH) != EXPECTED_Q4_SHA256:
        raise AssertionError("Q4 control no longer matches retained source hash")
    manifest = json.loads(MASTER_MANIFEST.read_text())
    if hash_file(MASTER_PATH) != manifest["selected"]["range_sha256"]:
        raise AssertionError("BF16 master no longer matches manifest")
    if receipt.get("sources", {}).get("q4_control", {}).get("sha256") != EXPECTED_Q4_SHA256:
        raise AssertionError("Q4 source changed")
    expected = {item["name"]: item["sha256"] for item in receipt["output"]["ffn_tensors"]}
    verification = _verify_output(base, OUTPUT_PATH, expected)
    if receipt.get("output") != verification:
        raise AssertionError("output verification changed")
    return verification


def build() -> dict[str, Any]:
    if OUTPUT_PATH.exists() or RECEIPT_PATH.exists():
        if not (OUTPUT_PATH.exists() and RECEIPT_PATH.exists()):
            raise AssertionError(f"refusing unreceipted {IQ_TYPE.name} artifact")
        result = verify_artifact()
        print(f"Existing {IQ_TYPE.name} control verified: {result['sha256']}")
        return result
    if PART_PATH.exists() or RECEIPT_PART.exists():
        raise AssertionError(f"refusing stale {IQ_TYPE.name} temporary artifact")
    if not Q4_PATH.is_file() or not MASTER_PATH.is_file() or not MASTER_MANIFEST.is_file():
        raise AssertionError("missing Q4 control, BF16 master, or manifest")

    manifest = json.loads(MASTER_MANIFEST.read_text())
    entries = {entry["name"]: entry for entry in manifest["tensors"]}
    q4_sha256 = hash_file(Q4_PATH)
    if q4_sha256 != EXPECTED_Q4_SHA256:
        raise AssertionError(f"unexpected Q4 control hash: {q4_sha256} != {EXPECTED_Q4_SHA256}")
    master_sha256 = hash_file(MASTER_PATH)
    expected_master_sha256 = manifest["selected"]["range_sha256"]
    if master_sha256 != expected_master_sha256:
        raise AssertionError(f"BF16 range hash differs from manifest: {master_sha256} != {expected_master_sha256}")
    base = gguf.GGUFReader(Q4_PATH)
    if len(base.tensors) != EXPECTED_TENSORS or len(FFN_NAMES) != EXPECTED_FFN_COUNT:
        raise AssertionError("unexpected Q4 tensor counts")
    expected_size = Q4_PATH.stat().st_size - sum(
        int(t.n_bytes) for t in base.tensors if t.name in FFN_NAMES
    ) + EXPECTED_FFN_BYTES
    disk_reserve = 512 * 1024 * 1024
    disk_free = shutil.disk_usage(CACHE).free
    if disk_free < expected_size + disk_reserve:
        raise AssertionError(
            f"insufficient disk space for {IQ_TYPE.name} output plus 512 MiB reserve: "
            f"free={disk_free} required={expected_size + disk_reserve}"
        )
    memory_available = None
    try:
        memory_available = (os.sysconf("SC_PAGE_SIZE") *
                            os.sysconf("SC_AVPHYS_PAGES"))
    except (AttributeError, OSError, ValueError):
        pass
    memory_reserve = 512 * 1024 * 1024
    if memory_available is not None and memory_available < memory_reserve:
        raise AssertionError(
            f"insufficient available memory for {IQ_TYPE.name} build: "
            f"available={memory_available} required={memory_reserve}"
        )
    preflight = {
        "disk_free_bytes": disk_free,
        "expected_output_bytes": expected_size,
        "disk_reserve_bytes": disk_reserve,
        "available_memory_bytes": memory_available,
        "memory_reserve_bytes": memory_reserve,
    }
    stats, calibration = _capture_stats()

    sources = {
        "q4_control": _source(Q4_PATH, EXPECTED_Q4_SHA256),
        "bf16_master": _source(MASTER_PATH, expected_master_sha256),
        "master_manifest": _source(MASTER_MANIFEST),
        "quantizer": _source(QUANTIZER),
        "calibrated_builder": _source(CALIBRATED_BUILDER),
        "calibrated_q3_receipt": _source(ROOT / "benchmarks/depth-corpus/quant-calibrated.json"),
    }
    writer = gguf.GGUFWriter(PART_PATH, "breeze-tts-2")
    _add_metadata(writer, base)
    library, function = _quantizer()
    expected_ffn: dict[str, str] = {}
    conversions: list[dict[str, Any]] = []
    weighted_stats: list[dict[str, Any]] = []
    try:
        for index, tensor in enumerate(base.tensors, start=1):
            if tensor.name not in FFN_NAMES:
                writer.add_tensor(tensor.name, tensor.data, raw_dtype=tensor.tensor_type)
                continue
            entry = entries[MAPPING[tensor.name]]
            if entry.get("dtype") != "BF16" or list(reversed(entry["shape"])) != tensor.shape.tolist():
                raise AssertionError(f"master shape/type differs: {tensor.name}")
            if hash_region(MASTER_PATH, int(entry["local_offset"]), int(entry["nbytes"])) != entry["sha256"]:
                raise AssertionError(f"master tensor hash differs: {tensor.name}")
            source, conversion = _load_bf16_f16(entry)
            conversion["name"] = tensor.name
            conversions.append(conversion)
            layer = int(tensor.name.split(".")[2])
            role = "down" if tensor.name.endswith("ffn_down.weight") else "gate_up"
            weights, evidence = _weights(stats[(layer, role)])
            weighted_stats.append({"name": tensor.name, **evidence})
            packed = _quantize(function, source, tensor, weights)
            expected_ffn[tensor.name] = hashlib.sha256(packed.tobytes()).hexdigest()
            writer.add_tensor(tensor.name, packed, raw_dtype=IQ_TYPE)
            print(f"quantized {index}/{len(base.tensors)} {tensor.name}", flush=True)
            del source, weights, packed
    finally:
        library.ggml_quantize_free()

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    verification = _verify_output(base, PART_PATH, expected_ffn)
    os.replace(PART_PATH, OUTPUT_PATH)
    verification["path"] = str(OUTPUT_PATH)
    verification["sha256"] = hash_file(OUTPUT_PATH)
    receipt = {
        "schema": FORMAT_SPECS[ACTIVE_FORMAT]["schema"],
        "base_output": str(Q4_PATH),
        "sources": sources,
        "preflight": preflight,
        "calibration": calibration,
        "activation_statistics": _receipt_stats(stats),
        "quantizer": {
            "library": str(QUANTIZER),
            "api": f"ggml_quantize_chunk(GGML_TYPE_{IQ_TYPE.name}, ..., imatrix)",
            "type": IQ_TYPE.name,
            "qk": QK,
            "bytes_per_block": IQ_BYTES_PER_BLOCK,
            "expected_per_projection_bytes": EXPECTED_PER_PROJECTION_BYTES,
            "expected_ffn_bytes": EXPECTED_FFN_BYTES,
            "weighted_tensor_count": EXPECTED_FFN_COUNT,
        },
        "f16_intermediate": conversions,
        "weighted_statistics": weighted_stats,
        "calibration_output_error": {
            "status": "deferred",
            "reason": f"Dense BF16-versus-{IQ_TYPE.name} projection products over all 160 captured samples are outside this bounded local build; no quality pass is claimed.",
        },
        "output": verification,
        "script_sha256": hash_file(Path(__file__)),
        "caveats": [
            "Only the 36 depth FFN tensors changed; 48 attention and 1077 other tensors are copied byte-identically from Q4_K.",
            "Calibration provenance uses the verified 160-sample, 150-record, all-12-layer capture receipt shared with weighted Q3_K.",
            "The model is a local quantization control, not a deployment choice; no GPU, speech-quality, or runtime acceptance is implied.",
        ],
    }
    RECEIPT_PART.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    os.replace(RECEIPT_PART, RECEIPT_PATH)
    print(f"Built and verified {IQ_TYPE.name} control: {verification['sha256']}")
    return verification


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--format", choices=tuple(FORMAT_SPECS), default=DEFAULT_FORMAT)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    _configure_format(args.format)
    if args.verify:
        result = verify_artifact()
        print(f"Verifier script SHA256: {hash_file(Path(__file__))}")
        print(f"{IQ_TYPE.name} control verified: {result['sha256']}")
    else:
        build()
