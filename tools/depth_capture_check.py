#!/usr/bin/env python3
"""Check the bounded depth FFN capture blob and JSONL index."""

import argparse
import json
import math
import struct
import tempfile
from pathlib import Path


FRAMES = {0, 7, 15}
LAYERS = {0, 5, 11}
TENSORS = {"ffn_norm_input", "post_silu_times_up", "ffn_down_output"}
MAX_BLOB_BYTES = 16 * 1024 * 1024


def _fail(message):
    raise AssertionError(message)


def validate(directory, full=False):
    directory = Path(directory)
    blob_path = directory / "activations.f32"
    meta_path = directory / "metadata.jsonl"
    if not blob_path.is_file() or not meta_path.is_file():
        _fail("capture needs activations.f32 and metadata.jsonl")
    blob = blob_path.read_bytes()
    if len(blob) > MAX_BLOB_BYTES:
        _fail(f"capture blob is too large: {len(blob)} bytes")

    records = []
    for line in meta_path.read_text().splitlines():
        if line.strip():
            records.append(json.loads(line))
    if not records:
        _fail("metadata is empty")

    keys = set()
    frames = {}
    tensor_records = []
    headers = 0
    header = None
    for record in records:
        key = record.get("key")
        if not key or key in keys:
            _fail(f"duplicate or missing record key: {key!r}")
        keys.add(key)
        kind = record.get("kind")
        if kind == "frame":
            frame = record.get("frame")
            codes = record.get("codes")
            if frame in frames or frame not in FRAMES or not isinstance(codes, list):
                _fail(f"bad frame record: {record}")
            if len(codes) != 16 or record.get("cb0") != codes[0]:
                _fail(f"frame {frame} does not contain cb0..cb15")
            if any(not isinstance(code, int) or not 0 <= code <= 2050 for code in codes):
                _fail(f"frame {frame} has an out-of-range code")
            frames[frame] = record
        elif kind == "header":
            if record.get("protocol") != 1 or record.get("endian") != "little":
                _fail(f"unsupported capture header: {record}")
            if str(record.get("backend", "")).startswith("Vulkan") and record.get("fusion_disabled") is not True:
                _fail("Vulkan capture requires recorded fusion_disabled=true")
            headers += 1
            header = record
        elif kind == "tensor":
            tensor_records.append(record)
            shape = record.get("shape")
            if (record.get("dtype") != "f32" or not isinstance(shape, list) or
                    not shape or any(not isinstance(n, int) or n <= 0 for n in shape)):
                _fail(f"bad tensor shape/dtype: {record}")
            offset = record.get("offset")
            nbytes = record.get("nbytes")
            count = 1
            for dim in shape:
                count *= dim
            if not isinstance(offset, int) or not isinstance(nbytes, int) or nbytes != count * 4:
                _fail(f"bad tensor byte range: {record}")
            if offset < 0 or offset + nbytes > len(blob):
                _fail(f"tensor points outside blob: {record}")
            values = struct.unpack_from("<%df" % count, blob, offset)
            if any(not math.isfinite(value) for value in values):
                _fail(f"non-finite tensor values: {record}")

            frame = record.get("frame")
            tensor = record.get("tensor")
            if frame not in FRAMES or record.get("branches") not in (1, 2):
                _fail(f"bad tensor frame/branch count: {record}")
            if tensor == "backbone_hidden":
                if record.get("layer") != -1 or record.get("step") != 0:
                    _fail(f"bad host hidden coordinates: {record}")
            elif (tensor not in TENSORS or record.get("layer") not in LAYERS or
                  not 1 <= record.get("step", 0) <= 15):
                _fail(f"bad FFN tensor coordinates: {record}")
        else:
            _fail(f"unknown metadata kind: {kind!r}")

    if not frames:
        _fail("no frame records")
    if headers != 1:
        _fail(f"expected one capture header, got {headers}")
    if (header.get("hidden") != 2048 or header.get("ffn") != 8192 or
            header.get("layers") != 12 or
            header.get("steps") != 15 or header.get("vocab") != 2051 or
            header.get("branches") not in (1, 2)):
        _fail(f"unexpected capture dimensions: {header}")
    if "fusion_disabled" in header and not isinstance(header["fusion_disabled"], bool):
        _fail(f"bad fusion_disabled header field: {header}")
    branches = header["branches"]
    for record in frames.values():
        if record.get("branches") != branches:
            _fail(f"frame branch count disagrees with header: {record}")
    by_frame = {}
    for record in tensor_records:
        by_frame.setdefault(record["frame"], []).append(record)
    for frame in frames:
        tensors = by_frame.get(frame, [])
        expected = {(layer, step, tensor) for layer in LAYERS for step in range(1, 16)
                    for tensor in TENSORS}
        if not {(r.get("layer"), r.get("step"), r.get("tensor")) for r in tensors
                if r.get("tensor") in TENSORS} == expected:
            _fail(f"frame {frame} is missing one or more selected tensors")
        if sum(r.get("tensor") == "backbone_hidden" for r in tensors) != 1:
            _fail(f"frame {frame} is missing its host hidden record")
        for record in tensors:
            if record.get("branches") != branches:
                _fail(f"tensor branch count disagrees with header: {record}")
            shape = record["shape"]
            if shape[2:] != [1]:
                _fail(f"unexpected tensor rank: {record}")
            if record["tensor"] == "backbone_hidden":
                expected_shape = [header["hidden"], branches, 1]
                if record.get("token_layout") != "branch_major":
                    _fail(f"bad host hidden layout: {record}")
            else:
                width = 8192 if record["tensor"] == "post_silu_times_up" else 1024
                tokens = (2 if record["step"] == 1 else 1) * branches
                expected_shape = [width, tokens, 1]
                if record.get("token_layout") != "position_major_branch_interleaved":
                    _fail(f"bad FFN token layout: {record}")
            if shape != expected_shape:
                _fail(f"unexpected tensor shape: {record}")
    if full and set(frames) != FRAMES:
        _fail(f"self-test expected frames {FRAMES}, got {set(frames)}")
    return {"frames": sorted(frames), "tensors": len(tensor_records), "blob_bytes": len(blob)}


def self_test():
    with tempfile.TemporaryDirectory(prefix="breeze-depth-capture-") as tmp:
        directory = Path(tmp)
        blob = bytearray()
        records = [{"kind": "header", "key": "header", "protocol": 1, "endian": "little",
                    "hidden": 2048, "ffn": 8192, "layers": 12,
                    "steps": 15, "vocab": 2051, "branches": 1, "fusion_disabled": False}]

        def add_tensor(frame, step, layer, name, shape, values):
            offset = len(blob)
            blob.extend(struct.pack("<%df" % len(values), *values))
            records.append({
                "kind": "tensor", "key": f"f{frame}/s{step}/l{layer}/{name}",
                "frame": frame, "step": step, "layer": layer, "tensor": name,
                "dtype": "f32", "shape": list(shape), "offset": offset,
                "nbytes": len(values) * 4, "branches": 1,
                "token_layout": "branch_major" if name == "backbone_hidden"
                else "position_major_branch_interleaved",
            })

        for frame in (0, 7, 15):
            add_tensor(frame, 0, -1, "backbone_hidden", (2048, 1, 1), [0.0] * 2048)
            for step in range(1, 16):
                for layer in (0, 5, 11):
                    add_tensor(frame, step, layer, "ffn_norm_input", (1024, 2 if step == 1 else 1, 1),
                               [0.0] * (1024 * (2 if step == 1 else 1)))
                    add_tensor(frame, step, layer, "post_silu_times_up", (8192, 2 if step == 1 else 1, 1),
                               [0.0] * (8192 * (2 if step == 1 else 1)))
                    add_tensor(frame, step, layer, "ffn_down_output", (1024, 2 if step == 1 else 1, 1),
                               [0.0] * (1024 * (2 if step == 1 else 1)))
            records.append({"kind": "frame", "key": f"frame/{frame}", "frame": frame,
                            "cb0": 0, "codes": list(range(16)), "branches": 1})
        (directory / "activations.f32").write_bytes(blob)
        (directory / "metadata.jsonl").write_text("\n".join(json.dumps(r) for r in records) + "\n")
        validate(directory, full=True)
        valid = (directory / "metadata.jsonl").read_text()
        first_tensor = next(line for line in valid.splitlines() if '"kind": "tensor"' in line)
        (directory / "metadata.jsonl").write_text(valid + first_tensor + "\n")
        try:
            validate(directory)
        except AssertionError:
            pass
        else:
            _fail("duplicate-key corruption was not rejected")
        (directory / "metadata.jsonl").write_text(valid)
        records[0]["backend"] = "Vulkan0"
        (directory / "metadata.jsonl").write_text("\n".join(json.dumps(r) for r in records) + "\n")
        try:
            validate(directory)
        except AssertionError:
            pass
        else:
            _fail("Vulkan capture without disabled fusion was not rejected")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", nargs="?", type=Path, help="capture output directory")
    parser.add_argument("--self-test", action="store_true", help="run the synthetic checker test")
    parser.add_argument("--full", action="store_true", help="require frames 0, 7, and 15")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        print("depth capture self-test: PASS")
    elif args.directory:
        print(json.dumps(validate(args.directory, full=args.full)))
    else:
        parser.error("provide a capture directory or --self-test")


if __name__ == "__main__":
    main()
