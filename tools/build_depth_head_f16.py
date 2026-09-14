#!/usr/bin/env python3
"""Build a GGUF comparison model: convert only dd.codebooks_head.weight F32 to F16.

The source is never modified. GGUF requires contiguous tensor offsets, so
later offsets move left by the target's padded-size reduction; all other tensor
payloads and the metadata prefix are checked byte-for-byte.

  build_depth_head_f16.py rewrite SOURCE.gguf CANDIDATE.gguf
  build_depth_head_f16.py verify SOURCE.gguf CANDIDATE.gguf
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import sys

import gguf
import numpy as np


TARGET = "dd.codebooks_head.weight"
SHAPE = (1024, 2051, 15)  # GGUF dimension order.
CHUNK = 1 << 20
IO_CHUNK = 8 << 20


def die(message: str) -> "NoReturn":
    raise AssertionError(message)


def padded(size: int, alignment: int) -> int:
    return (size + alignment - 1) // alignment * alignment


def info_offset(tensor: gguf.ReaderTensor, part: int) -> int:
    return tensor.field.offset + sum(
        int(value.nbytes) for value in tensor.field.parts[:part]
    )


def copy_n(source, destination, size: int) -> None:
    remaining = size
    while remaining:
        data = source.read(min(IO_CHUNK, remaining))
        if not data:
            die(f"short copy ({remaining} bytes remain)")
        destination.write(data)
        remaining -= len(data)


def sha256(path: Path, offset: int = 0, size: int | None = None) -> str:
    digest = hashlib.sha256()
    remaining = size
    with path.open("rb") as stream:
        stream.seek(offset)
        while remaining is None or remaining:
            data = stream.read(IO_CHUNK if remaining is None else min(IO_CHUNK, remaining))
            if not data:
                if remaining:
                    die(f"short hash read at {offset}: {remaining} bytes remain")
                break
            digest.update(data)
            if remaining is not None:
                remaining -= len(data)
    return digest.hexdigest()


def check_layout(reader: gguf.GGUFReader, path: Path) -> None:
    expected = reader.data_offset
    if expected % reader.alignment:
        die(f"{path}: unaligned data section")
    for index, tensor in enumerate(reader.tensors):
        if tensor.data_offset != expected:
            die(f"{path}: tensor {index} {tensor.name} has non-contiguous offset")
        if (tensor.data_offset - reader.data_offset) % reader.alignment:
            die(f"{path}: tensor {index} {tensor.name} is unaligned")
        if len(tensor.field.parts) != 6:
            die(f"{path}: tensor {index} has unexpected tensor-info layout")
        expected += padded(tensor.n_bytes, reader.alignment)
    if expected != path.stat().st_size:
        die(f"{path}: layout ends at {expected}, file is {path.stat().st_size}")


def read_checked(path: Path) -> gguf.GGUFReader:
    reader = gguf.GGUFReader(path)
    if reader.endianess != gguf.GGUFEndian.LITTLE:
        die(f"{path}: only little-endian GGUF is supported")
    check_layout(reader, path)
    return reader


def find_target(reader: gguf.GGUFReader, require_f32: bool = True):
    matches = [(i, tensor) for i, tensor in enumerate(reader.tensors) if tensor.name == TARGET]
    if len(matches) != 1:
        die(f"expected one {TARGET}, found {len(matches)}")
    index, tensor = matches[0]
    if tuple(int(value) for value in tensor.shape) != SHAPE:
        die(f"{TARGET}: shape {tuple(tensor.shape)} != {SHAPE}")
    if tensor.n_elements != int(np.prod(SHAPE)):
        die(f"{TARGET}: inconsistent element count")
    if require_f32 and tensor.tensor_type != gguf.GGMLQuantizationType.F32:
        die(f"{TARGET}: expected F32, got {tensor.tensor_type}")
    return index, tensor


def metadata_same(
    source: gguf.GGUFReader,
    candidate: gguf.GGUFReader,
    source_path: Path,
    candidate_path: Path,
) -> None:
    if source.alignment != candidate.alignment or source.data_offset != candidate.data_offset:
        die("GGUF alignment/data-section metadata changed")
    if list(source.fields) != list(candidate.fields):
        die("GGUF metadata field names/order changed")
    for name in source.fields:
        left, right = source.fields[name], candidate.fields[name]
        if left.offset != right.offset or left.name != right.name:
            die(f"GGUF metadata field location changed: {name}")
        if left.data != right.data or left.types != right.types:
            die(f"GGUF metadata field changed: {name}")
        if len(left.parts) != len(right.parts) or any(
            not np.array_equal(a, b) for a, b in zip(left.parts, right.parts)
        ):
            die(f"GGUF metadata bytes changed: {name}")
    prefix_size = source.tensors[0].field.offset
    with source_path.open("rb") as left, candidate_path.open("rb") as right:
        if left.read(prefix_size) != right.read(prefix_size):
            die("bytes before tensor-info records changed")


def rewrite(source_path: Path, candidate_path: Path) -> dict[str, int]:
    if candidate_path.exists():
        die(f"refusing to overwrite {candidate_path}")
    source = read_checked(source_path)
    target_index, target = find_target(source)
    old_padded = padded(target.n_bytes, source.alignment)
    new_bytes = target.n_elements * 2
    new_padded = padded(new_bytes, source.alignment)
    delta = old_padded - new_padded
    if delta <= 0:
        die(f"target conversion has no storage reduction: {delta}")
    if struct.unpack_from("<I", source.data, info_offset(target, 4))[0] != 0:
        die("target raw dtype is not F32")
    if int(target.field.parts[5][0]) != target.data_offset - source.data_offset:
        die("target raw offset disagrees with reader")

    temp = candidate_path.with_name(f".{candidate_path.name}.tmp-{os.getpid()}")
    if temp.exists():
        die(f"temporary output exists: {temp}")
    candidate_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with source_path.open("rb") as source_file, temp.open("xb") as output:
            prefix = bytearray(source_file.read(source.data_offset))
            if len(prefix) != source.data_offset:
                die("short GGUF prefix read")
            struct.pack_into(
                "<I", prefix, info_offset(target, 4), gguf.GGMLQuantizationType.F16.value
            )
            for index, tensor in enumerate(source.tensors):
                if index <= target_index:
                    continue
                field = info_offset(tensor, 5)
                old_offset = struct.unpack_from("<Q", prefix, field)[0]
                if old_offset != tensor.data_offset - source.data_offset:
                    die(f"{tensor.name}: raw offset disagrees with reader")
                struct.pack_into("<Q", prefix, field, old_offset - delta)

            output.write(prefix)
            source_file.seek(source.data_offset)
            copy_n(source_file, output, target.data_offset - source.data_offset)

            values = np.memmap(
                source_path,
                mode="r",
                dtype="<f4",
                offset=target.data_offset,
                shape=(target.n_elements,),
            )
            for start in range(0, target.n_elements, CHUNK):
                converted = values[start : start + CHUNK].astype("<f2")
                if not np.isfinite(converted).all():
                    die(f"non-finite F16 target values at {start}")
                output.write(converted.tobytes())
            del values
            if new_padded > new_bytes:
                output.write(b"\0" * (new_padded - new_bytes))
            source_file.seek(target.data_offset + old_padded)
            while data := source_file.read(IO_CHUNK):
                output.write(data)
            output.flush()
            os.fsync(output.fileno())
        expected_size = source_path.stat().st_size - delta
        if temp.stat().st_size != expected_size:
            die(f"candidate size {temp.stat().st_size} != {expected_size}")
        os.replace(temp, candidate_path)
    finally:
        if temp.exists():
            temp.unlink()
    return {
        "source_bytes": source_path.stat().st_size,
        "candidate_bytes": candidate_path.stat().st_size,
        "target_old_bytes": target.n_bytes,
        "target_new_bytes": new_bytes,
        "offset_delta": delta,
        "target_index": target_index,
    }


def verify(source_path: Path, candidate_path: Path) -> dict[str, object]:
    source, candidate = read_checked(source_path), read_checked(candidate_path)
    if len(source.tensors) != len(candidate.tensors):
        die("tensor count changed")
    source_index, source_target = find_target(source)
    candidate_index, candidate_target = find_target(candidate, require_f32=False)
    if source_index != candidate_index:
        die("target tensor index changed")
    if candidate_target.tensor_type != gguf.GGMLQuantizationType.F16:
        die(f"candidate target is {candidate_target.tensor_type}, expected F16")
    if candidate_target.n_bytes != source_target.n_elements * 2:
        die("candidate target byte size is not F16 size")
    metadata_same(source, candidate, source_path, candidate_path)
    if candidate_target.data_offset != source_target.data_offset:
        die("target data offset changed")
    if source_path.stat().st_size - candidate_path.stat().st_size != (
        source_target.n_bytes - candidate_target.n_bytes
    ):
        die("file-size delta does not match target-size delta")

    verified = 0
    hash_chain = hashlib.sha256()
    for index, (left, right) in enumerate(zip(source.tensors, candidate.tensors)):
        if left.name != right.name or not np.array_equal(left.shape, right.shape):
            die(f"tensor {index} name/shape changed")
        if index == source_index:
            continue
        if left.tensor_type != right.tensor_type or left.n_bytes != right.n_bytes:
            die(f"tensor {index} {left.name} type/size changed")
        left_hash = sha256(source_path, left.data_offset, left.n_bytes)
        right_hash = sha256(candidate_path, right.data_offset, right.n_bytes)
        if left_hash != right_hash:
            die(f"tensor {index} {left.name} payload changed")
        hash_chain.update(left_hash.encode("ascii"))
        verified += 1

    source_values = np.memmap(
        source_path,
        mode="r",
        dtype="<f4",
        offset=source_target.data_offset,
        shape=(source_target.n_elements,),
    )
    candidate_values = np.memmap(
        candidate_path,
        mode="r",
        dtype="<f2",
        offset=candidate_target.data_offset,
        shape=(candidate_target.n_elements,),
    )
    max_error, finite = 0.0, 0
    for start in range(0, source_target.n_elements, CHUNK):
        left = source_values[start : start + CHUNK]
        right = candidate_values[start : start + CHUNK]
        if not np.isfinite(left).all() or not np.isfinite(right).all():
            die(f"non-finite target values at {start}")
        expected = left.astype("<f2")
        if not np.array_equal(expected, right):
            die(f"target F32->F16 bytes differ at {start}")
        finite += right.size
        max_error = max(max_error, float(np.max(np.abs(left - right.astype("<f4")))))
    del source_values, candidate_values
    return {
        "source": str(source_path),
        "candidate": str(candidate_path),
        "source_sha256": sha256(source_path),
        "candidate_sha256": sha256(candidate_path),
        "source_bytes": source_path.stat().st_size,
        "candidate_bytes": candidate_path.stat().st_size,
        "tensor_count": len(source.tensors),
        "target": TARGET,
        "target_index": source_index,
        "target_shape_gguf": list(SHAPE),
        "target_source_type": source_target.tensor_type.name,
        "target_candidate_type": candidate_target.tensor_type.name,
        "target_source_bytes": source_target.n_bytes,
        "target_candidate_bytes": candidate_target.n_bytes,
        "target_source_sha256": sha256(source_path, source_target.data_offset, source_target.n_bytes),
        "target_candidate_sha256": sha256(candidate_path, candidate_target.data_offset, candidate_target.n_bytes),
        "non_target_payloads_verified": verified,
        "non_target_payload_sha256_chain": hash_chain.hexdigest(),
        "target_f16_finite_elements": int(finite),
        "target_max_abs_f32_roundtrip_error": max_error,
        "metadata_prefix_unchanged": True,
        "layout_contiguous_aligned": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("rewrite", "verify"))
    parser.add_argument("source", type=Path)
    parser.add_argument("candidate", type=Path)
    args = parser.parse_args()
    source = args.source.resolve()
    candidate = args.candidate.resolve()
    if not source.is_file():
        die(f"source does not exist: {source}")
    if args.command == "rewrite":
        print(json.dumps(rewrite(source, candidate), sort_keys=True))
    print(json.dumps(verify(source, candidate), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (AssertionError, OSError, ValueError) as error:
        print(f"build_depth_head_f16.py: ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
