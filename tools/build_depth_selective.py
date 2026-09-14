#!/usr/bin/env python3
"""Build and verify a Q3_K FFN plus Q4_K attention depth control."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import gguf
import numpy as np

from build_depth_quant_controls import MAPPING, metadata
from depth_quant_screen import hash_file, hash_region


ROOT = Path(__file__).resolve().parents[1]
CACHE = Path("~/.cache/breeze-models/depth-quant-controls").expanduser()
Q3_PATH = CACHE / "breeze-depth-blocks-q3_k.gguf"
Q4_PATH = CACHE / "breeze-depth-blocks-q4_k.gguf"
OUTPUT_PATH = CACHE / "breeze-depth-ffn-q3_k-attn-q4_k.gguf"
PART_PATH = OUTPUT_PATH.with_suffix(".part")
RECEIPT_PATH = ROOT / "benchmarks/depth-corpus/quant-selective.json"
HASHES = {
    "q3_k": "65349eb6a24ca760b558e302b465c703c6fbbbca5a2fe2dd034aaee1d925fd50",
    "q4_k": "49e4531863ccee2e4362fb06f86995f88659fb60bee37db45bfc662d5e9d497d",
}
Q3_K = gguf.GGMLQuantizationType.Q3_K
Q4_K = gguf.GGMLQuantizationType.Q4_K
ATTENTION = frozenset(name for name in MAPPING if ".attn_" in name)
FFN = frozenset(MAPPING) - ATTENTION


def all_metadata(reader: gguf.GGUFReader) -> dict:
    """Include generated GGUF.* fields as well as model metadata."""
    return {key: (field.types, field.contents()) for key, field in reader.fields.items()}


def source_manifest() -> dict[str, dict]:
    result = {}
    for kind, path in (("q3_k", Q3_PATH), ("q4_k", Q4_PATH)):
        assert path.is_file(), f"missing {kind} source: {path}"
        actual = hash_file(path)
        assert actual == HASHES[kind], f"{kind} source hash mismatch: {actual}"
        result[kind] = {"path": str(path), "bytes": path.stat().st_size, "sha256": actual}
    return result


def open_sources() -> tuple[gguf.GGUFReader, gguf.GGUFReader, dict]:
    sources = source_manifest()
    q3, q4 = gguf.GGUFReader(Q3_PATH), gguf.GGUFReader(Q4_PATH)
    assert metadata(q3) == metadata(q4), "source metadata differs"
    assert all_metadata(q3) == all_metadata(q4), "source GGUF fields differ"
    assert [t.name for t in q3.tensors] == [t.name for t in q4.tensors], "source tensor order differs"
    assert len(q3.tensors) == len(q4.tensors) == 1161, "unexpected source tensor count"
    assert len([t for t in q3.tensors if t.name in MAPPING]) == 84
    assert len([t for t in q4.tensors if t.name in MAPPING]) == 84
    assert all(t.tensor_type == Q3_K for t in q3.tensors if t.name in MAPPING)
    assert all(t.tensor_type == Q4_K for t in q4.tensors if t.name in MAPPING)
    for before, selected in zip(q3.tensors, q4.tensors):
        assert np.array_equal(before.shape, selected.shape), f"source shape differs: {before.name}"
    return q3, q4, sources


def verify_model(q3: gguf.GGUFReader, q4: gguf.GGUFReader, path: Path) -> dict:
    candidate = gguf.GGUFReader(path)
    assert metadata(q3) == metadata(candidate), "GGUF metadata changed"
    assert all_metadata(q3) == all_metadata(candidate), "GGUF field changed"
    assert [t.name for t in q3.tensors] == [t.name for t in candidate.tensors], "tensor order changed"
    assert len(q3.tensors) == len(candidate.tensors), "tensor count changed"
    attention = []
    attention_bytes = 0
    ffn_bytes = 0
    unchanged = 0
    for base, selected, after in zip(q3.tensors, q4.tensors, candidate.tensors):
        assert base.name == selected.name == after.name
        assert np.array_equal(base.shape, selected.shape)
        assert np.array_equal(base.shape, after.shape), f"shape changed: {after.name}"
        if after.name in ATTENTION:
            assert selected.tensor_type == Q4_K and after.tensor_type == Q4_K
            assert after.n_bytes == selected.n_bytes
            assert np.array_equal(after.data, selected.data), f"packed bytes differ: {after.name}"
            source_sha = hash_region(Q4_PATH, selected.data_offset, selected.n_bytes)
            output_sha = hash_region(path, after.data_offset, after.n_bytes)
            assert source_sha == output_sha, f"packed hash differs: {after.name}"
            attention_bytes += after.n_bytes
            attention.append({
                "name": after.name,
                "type": after.tensor_type.name,
                "shape": after.shape.tolist(),
                "bytes": int(after.n_bytes),
                "source_offset": int(selected.data_offset),
                "source_sha256": source_sha,
                "output_offset": int(after.data_offset),
                "output_sha256": output_sha,
            })
        else:
            assert after.tensor_type == base.tensor_type
            assert after.n_bytes == base.n_bytes
            assert np.array_equal(after.data, base.data), f"unchanged bytes differ: {after.name}"
            unchanged += 1
            if after.name in FFN:
                assert base.tensor_type == Q3_K
                ffn_bytes += after.n_bytes
    assert len(attention) == 48
    assert len(FFN) == 36
    assert unchanged == len(q3.tensors) - len(attention)
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": hash_file(path),
        "tensor_count": len(candidate.tensors),
        "unchanged_tensor_count": unchanged,
        "unchanged_tensors_byte_identical": True,
        "metadata_field_count": len(candidate.fields),
        "metadata_identical": True,
        "tensor_name_order_identical": True,
        "tensor_shapes_identical": True,
        "attention_tensor_count": len(attention),
        "ffn_tensor_count": len(FFN),
        "attention_packed_bytes": attention_bytes,
        "ffn_packed_bytes": ffn_bytes,
        "selected_packed_bytes": attention_bytes,
        "selected_exact_bytes": True,
        "attention_tensors": attention,
    }


def receipt_for(verification: dict, sources: dict) -> dict:
    return {
        "schema": "breeze-depth-selective/v1",
        "sources": sources,
        "output": {key: verification[key] for key in ("path", "bytes", "sha256")},
        "script_sha256": hash_file(Path(__file__)),
        "selection": {
            "mapping_tensor_count": len(MAPPING),
            "attention_tensor_count": len(ATTENTION),
            "ffn_tensor_count": len(FFN),
            "attention_source_type": Q4_K.name,
            "ffn_source_type": Q3_K.name,
            "selected_packed_bytes": verification["selected_packed_bytes"],
        },
        "verification": verification,
        "caveats": [
            "Byte-preserving mixed control: Q3_K FFN and Q4_K depth attention.",
            "No speech-quality, runtime, or performance acceptance is implied.",
            "Built locally without GPU use or requantization.",
        ],
    }


def verify_artifact() -> dict:
    assert RECEIPT_PATH.is_file(), f"missing receipt: {RECEIPT_PATH}"
    receipt = json.loads(RECEIPT_PATH.read_text())
    assert receipt.get("schema") == "breeze-depth-selective/v1"
    q3, q4, sources = open_sources()
    assert receipt.get("sources") == sources, "receipt source evidence changed"
    assert receipt.get("script_sha256") == hash_file(Path(__file__)), "receipt script hash changed"
    verification = verify_model(q3, q4, OUTPUT_PATH)
    assert receipt.get("verification") == verification, "receipt verification changed"
    assert receipt.get("output") == {key: verification[key] for key in ("path", "bytes", "sha256")}
    assert receipt.get("selection") == {
        "mapping_tensor_count": len(MAPPING),
        "attention_tensor_count": len(ATTENTION),
        "ffn_tensor_count": len(FFN),
        "attention_source_type": Q4_K.name,
        "ffn_source_type": Q3_K.name,
        "selected_packed_bytes": verification["selected_packed_bytes"],
    }
    return verification


def build() -> dict:
    if RECEIPT_PATH.exists() or OUTPUT_PATH.exists():
        assert RECEIPT_PATH.exists() and OUTPUT_PATH.exists(), "refusing unreceipted selective artifact"
        result = verify_artifact()
        print(f"Existing selective control verified: {result['sha256']}")
        return result
    assert not PART_PATH.exists(), f"refusing stale temporary artifact: {PART_PATH}"
    q3, q4, sources = open_sources()
    q3_attention = sum(t.n_bytes for t in q3.tensors if t.name in ATTENTION)
    q4_attention = sum(t.n_bytes for t in q4.tensors if t.name in ATTENTION)
    expected_size = Q3_PATH.stat().st_size + q4_attention - q3_attention
    assert shutil.disk_usage(CACHE).free >= expected_size + (256 << 20), "insufficient output space"

    q4_by_name = {t.name: t for t in q4.tensors}
    writer = gguf.GGUFWriter(PART_PATH, "breeze-tts-2")
    try:
        for key, (types, value) in metadata(q3).items():
            writer.add_key_value(key, value, types[0], types[-1] if len(types) > 1 else None)
        for tensor in q3.tensors:
            source = q4_by_name[tensor.name] if tensor.name in ATTENTION else tensor
            writer.add_tensor(tensor.name, source.data, raw_dtype=source.tensor_type)
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
    finally:
        writer.close()

    # Verify the temporary file before the atomic rename; a failed verification
    # leaves an inspectable .part and cannot create an unreceipted final model.
    verification = verify_model(q3, q4, PART_PATH)
    PART_PATH.rename(OUTPUT_PATH)
    verification["path"] = str(OUTPUT_PATH)
    receipt_part = RECEIPT_PATH.with_suffix(".json.part")
    assert not receipt_part.exists(), f"refusing stale receipt temporary artifact: {receipt_part}"
    receipt_part.write_text(json.dumps(receipt_for(verification, sources), indent=2, allow_nan=False) + "\n")
    receipt_part.rename(RECEIPT_PATH)
    print(f"Built and verified selective control: {verification['sha256']}")
    return verification


def self_test() -> None:
    assert len(MAPPING) == 84
    assert len(ATTENTION) == 48
    assert len(FFN) == 36
    assert ATTENTION.isdisjoint(FFN) and ATTENTION | FFN == frozenset(MAPPING)
    assert MAPPING["dd.blk.11.ffn_down.weight"].endswith("layers.11.mlp.down_proj.weight")
    print("selective builder self-test: PASS")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    assert not (args.self_test and args.verify), "choose one mode"
    if args.self_test:
        self_test()
    elif args.verify:
        result = verify_artifact()
        print(f"Selective control verified: {result['sha256']}")
    else:
        build()
