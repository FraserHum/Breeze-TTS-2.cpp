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
HARD_MAX_BLOB_BYTES = 256 * 1024 * 1024
MAX_CAPTURE_FRAMES = 64
MAX_CAPTURE_FRAME = 4095
MAX_CAPTURE_RECORDS = 65536


def _fail(message):
    raise AssertionError(message)


def _capture_selection(header, name, defaults, maximum, max_count):
    values = header.get(name)
    if values is None:
        return set(defaults)
    if (not isinstance(values, list) or not values or len(values) > max_count or
            any(type(value) is not int or value < 0 or value > maximum for value in values) or
            len(set(values)) != len(values) or values != sorted(values)):
        _fail(f"bad {name}: {header}")
    return set(values)


def validate(directory, full=False):
    directory = Path(directory)
    blob_path = directory / "activations.f32"
    meta_path = directory / "metadata.jsonl"
    if not blob_path.is_file() or not meta_path.is_file():
        _fail("capture needs activations.f32 and metadata.jsonl")
    blob = blob_path.read_bytes()
    if len(blob) > HARD_MAX_BLOB_BYTES:
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
            if (not isinstance(frame, int) or isinstance(frame, bool) or frame < 0 or
                    frame in frames or not isinstance(codes, list)):
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
            if (not isinstance(frame, int) or isinstance(frame, bool) or frame < 0 or
                    record.get("branches") not in (1, 2)):
                _fail(f"bad tensor frame/branch count: {record}")
            if tensor == "backbone_hidden":
                if record.get("layer") != -1 or record.get("step") != 0:
                    _fail(f"bad host hidden coordinates: {record}")
            elif (tensor not in TENSORS or type(record.get("layer")) is not int or
                  not 0 <= record.get("layer") <= 11 or
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
    capture_frames = _capture_selection(header, "capture_frames", FRAMES,
                                        MAX_CAPTURE_FRAME, MAX_CAPTURE_FRAMES)
    capture_layers = _capture_selection(header, "capture_layers", LAYERS,
                                        header["layers"] - 1, header["layers"])
    max_blob_bytes = header.get("max_blob_bytes", MAX_BLOB_BYTES)
    if (type(max_blob_bytes) is not int or not 0 < max_blob_bytes <= HARD_MAX_BLOB_BYTES or
            len(blob) > max_blob_bytes):
        _fail(f"capture blob exceeds declared max_blob_bytes: {header}")
    default_max_records = 1 + len(capture_frames) * (2 + len(capture_layers) * header["steps"] * 3)
    max_records = header.get("max_records", default_max_records)
    if (type(max_records) is not int or not 0 < max_records <= MAX_CAPTURE_RECORDS or
            len(records) > max_records):
        _fail(f"capture metadata exceeds declared max_records: {header}")
    branches = header["branches"]
    if set(frames) - capture_frames:
        _fail(f"unexpected frame coordinates: {set(frames) - capture_frames}")
    if any(record.get("frame") not in capture_frames for record in tensor_records):
        _fail("unexpected tensor frame coordinates")
    if any(record.get("tensor") != "backbone_hidden" and
           record.get("layer") not in capture_layers for record in tensor_records):
        _fail("unexpected tensor layer coordinates")
    for record in frames.values():
        if record.get("branches") != branches:
            _fail(f"frame branch count disagrees with header: {record}")
    by_frame = {}
    for record in tensor_records:
        by_frame.setdefault(record["frame"], []).append(record)
    for frame in frames:
        tensors = by_frame.get(frame, [])
        expected = {(layer, step, tensor) for layer in capture_layers for step in range(1, 16)
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
    if full and set(frames) != capture_frames:
        _fail(f"self-test expected frames {capture_frames}, got {set(frames)}")
    return {"frames": sorted(frames), "tensors": len(tensor_records), "blob_bytes": len(blob)}


def self_test():
    with tempfile.TemporaryDirectory(prefix="breeze-depth-capture-") as tmp:
        directory = Path(tmp)
        def build_capture(frames, layers, extended=False):
            blob = bytearray()
            header = {"kind": "header", "key": "header", "protocol": 1, "endian": "little",
                      "hidden": 2048, "ffn": 8192, "layers": 12,
                      "steps": 15, "vocab": 2051, "branches": 1, "fusion_disabled": False}
            if extended:
                header.update(capture_frames=list(frames), capture_layers=list(layers),
                              max_blob_bytes=16 * 1024 * 1024)
            records = [header]

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

            for frame in frames:
                add_tensor(frame, 0, -1, "backbone_hidden", (2048, 1, 1), [0.0] * 2048)
                for step in range(1, 16):
                    for layer in layers:
                        tokens = 2 if step == 1 else 1
                        add_tensor(frame, step, layer, "ffn_norm_input", (1024, tokens, 1),
                                   [0.0] * (1024 * tokens))
                        add_tensor(frame, step, layer, "post_silu_times_up", (8192, tokens, 1),
                                   [0.0] * (8192 * tokens))
                        add_tensor(frame, step, layer, "ffn_down_output", (1024, tokens, 1),
                                   [0.0] * (1024 * tokens))
                records.append({"kind": "frame", "key": f"frame/{frame}", "frame": frame,
                                "cb0": 0, "codes": list(range(16)), "branches": 1})
            return blob, records

        def write_capture(blob, records):
            (directory / "activations.f32").write_bytes(blob)
            (directory / "metadata.jsonl").write_text(
                "\n".join(json.dumps(record) for record in records) + "\n")

        def expect_failure(label, records, blob, full=False):
            write_capture(blob, records)
            try:
                validate(directory, full=full)
            except AssertionError:
                return
            _fail(f"{label} was not rejected")

        # Old headers remain valid, preserving the original pilot corpus format.
        blob, records = build_capture((0, 7, 15), (0, 5, 11))
        write_capture(blob, records)
        validate(directory, full=True)
        duplicate = json.loads(json.dumps(records))
        duplicate.append(dict(next(record for record in duplicate if record["kind"] == "tensor")))
        expect_failure("duplicate-key corruption", duplicate, blob)

        vulkan = json.loads(json.dumps(records))
        vulkan[0]["backend"] = "Vulkan0"
        expect_failure("Vulkan capture without disabled fusion", vulkan, blob)

        # All layers and late frames use the same bounded record/stream format.
        blob, records = build_capture((31,), tuple(range(12)), extended=True)
        write_capture(blob, records)
        validate(directory, full=True)
        bad_selection = json.loads(json.dumps(records))
        bad_selection[0]["capture_frames"] = [31, 31]
        expect_failure("duplicate frame selection", bad_selection, blob)
        bad_selection[0]["capture_frames"] = [-1]
        expect_failure("negative frame selection", bad_selection, blob)
        bad_selection[0]["capture_frames"] = [31]
        bad_selection[0]["capture_layers"] = [12]
        expect_failure("out-of-range layer selection", bad_selection, blob)

        missing = json.loads(json.dumps(records))
        missing.pop(next(i for i, record in enumerate(missing)
                         if record.get("kind") == "tensor" and record.get("layer") == 11))
        expect_failure("missing selected coordinate", missing, blob, full=True)
        unexpected = json.loads(json.dumps(records))
        frame_record = next(record for record in unexpected if record.get("kind") == "frame")
        frame_record["frame"] = 63
        expect_failure("unexpected frame coordinate", unexpected, blob)
        too_small = json.loads(json.dumps(records))
        too_small[0]["max_blob_bytes"] = len(blob) - 4
        expect_failure("declared file-size guard", too_small, blob)

        # Restore a valid custom capture after the corruption checks.
        write_capture(blob, records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", nargs="?", type=Path, help="capture output directory")
    parser.add_argument("--self-test", action="store_true", help="run the synthetic checker test")
    parser.add_argument("--full", action="store_true", help="require every selected frame")
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
