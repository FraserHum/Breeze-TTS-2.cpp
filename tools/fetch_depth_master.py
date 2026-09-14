#!/usr/bin/env python3
"""Fetch and verify the original Breeze depth tensors without the 6.96 GB model.

The depth tensors are contiguous in one immutable safetensors shard.  The
downloader therefore makes one exact HTTP range request, then records each
tensor's source and local byte range in ``master-manifest.json``.  It uses
``curl`` for TLS (the system Python on the capture host does not have a usable
certificate store) and only the Python standard library otherwise.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
from pathlib import Path
import re
import struct
import subprocess
import tempfile
from typing import Any


MODEL = "BreezeBlue/Breeze-TTS-2"
REVISION = "799624c0b4a1daa8db6d28bbd9850043c0270734"
SHARD = "model-00001-of-00002.safetensors"
BASE_URL = f"https://huggingface.co/{MODEL}/resolve/{REVISION}"
INDEX_URL = f"{BASE_URL}/model.safetensors.index.json"
SHARD_URL = f"{BASE_URL}/{SHARD}"

DEPTH_PREFIX = "depth_decoder."
EXPECTED_TENSORS = 112
EXPECTED_DTYPE = "BF16"
EXPECTED_HEADER_LENGTH = 101880
EXPECTED_TOTAL_BYTES = 868560896
DATA_ITEM = struct.Struct("<Q")
CHUNK = 1 << 20

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO_ROOT / "benchmarks/depth-corpus/master-manifest.json"
DEFAULT_CACHE = Path.home() / ".cache/breeze-models/depth-master"


class FetchError(RuntimeError):
    """A source or cache validation failure."""


def _curl_args(*extra: str) -> list[str]:
    return [
        "curl",
        "--fail",
        "--silent",
        "--show-error",
        "--location",
        "--proto",
        "=https",
        "--proto-redir",
        "=https",
        "--retry",
        "3",
        "--retry-delay",
        "1",
        "--retry-all-errors",
        "--connect-timeout",
        "20",
        "--max-time",
        "3600",
        *extra,
    ]


def curl_bytes(url: str) -> bytes:
    try:
        return subprocess.check_output(_curl_args(url), stderr=subprocess.STDOUT)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise FetchError(f"curl failed for {url}") from exc


def _final_header_block(text: str) -> dict[str, str]:
    blocks = re.split(r"(?m)(?=^HTTP/)", text.replace("\r\n", "\n"))
    blocks = [block for block in blocks if block.startswith("HTTP/")]
    if not blocks:
        raise FetchError("curl returned no HTTP response headers")
    lines = blocks[-1].splitlines()
    status = lines[0].split()
    if len(status) < 2 or not status[1].isdigit():
        raise FetchError(f"malformed HTTP status: {lines[0]!r}")
    result = {":status": status[1]}
    for line in lines[1:]:
        if ":" in line:
            key, value = line.split(":", 1)
            result[key.strip().lower()] = value.strip()
    return result


def validate_range_response(
    header_text: str,
    body_bytes: int,
    start: int,
    end: int,
    expected_total: int | None = None,
) -> int:
    """Reject a non-206 response, wrong range, or wrong body length."""

    headers = _final_header_block(header_text)
    if headers.get(":status") != "206":
        raise FetchError(f"range request returned HTTP {headers.get(':status', '?')}")
    content_range = headers.get("content-range", "")
    match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", content_range)
    if not match:
        raise FetchError(f"missing or malformed Content-Range: {content_range!r}")
    got_start, got_end, total = (int(value) for value in match.groups())
    expected_length = end - start + 1
    if (got_start, got_end) != (start, end):
        raise FetchError(f"wrong Content-Range {content_range!r}; expected bytes {start}-{end}")
    if expected_total is not None and total != expected_total:
        raise FetchError(f"wrong shard length {total}; expected {expected_total}")
    content_length = headers.get("content-length")
    if content_length is None or not content_length.isdigit() or int(content_length) != expected_length:
        raise FetchError(f"wrong Content-Length {content_length!r}; expected {expected_length}")
    if body_bytes != expected_length:
        raise FetchError(f"curl wrote {body_bytes} bytes; expected {expected_length}")
    return total


def fetch_range(url: str, start: int, end: int, destination: Path, expected_total: int | None = None) -> int:
    """Download one range to a temporary file and atomically install it."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    with tempfile.TemporaryDirectory(prefix="breeze-range-") as tmp:
        headers = Path(tmp) / "headers.txt"
        command = _curl_args(
            "--range",
            f"{start}-{end}",
            "--dump-header",
            str(headers),
            "--output",
            str(partial),
            url,
        )
        try:
            subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        except (OSError, subprocess.CalledProcessError) as exc:
            raise FetchError(f"curl failed for range {start}-{end}") from exc
        if not headers.is_file() or not partial.is_file():
            raise FetchError("curl did not produce range headers and body")
        total = validate_range_response(
            headers.read_text(encoding="utf-8", errors="replace"),
            partial.stat().st_size,
            start,
            end,
            expected_total,
        )
        with partial.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(partial, destination)
    return total


def parse_index(raw: bytes) -> tuple[dict[str, Any], str]:
    try:
        index = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise FetchError("model index is not JSON") from exc
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise FetchError("model index has no weight_map")
    names = sorted(name for name in weight_map if name.startswith(DEPTH_PREFIX))
    if len(names) != EXPECTED_TENSORS:
        raise FetchError(f"expected {EXPECTED_TENSORS} depth tensors in index, got {len(names)}")
    if any(weight_map[name] != SHARD for name in names):
        raise FetchError("depth tensors are not all in the expected first shard")
    return index, hashlib.sha256(raw).hexdigest()


def parse_header(raw: bytes) -> tuple[dict[str, Any], int, str]:
    if len(raw) < DATA_ITEM.size:
        raise FetchError("safetensors header is shorter than its length field")
    header_length = DATA_ITEM.unpack(raw[: DATA_ITEM.size])[0]
    if header_length != EXPECTED_HEADER_LENGTH:
        raise FetchError(f"unexpected safetensors header length {header_length}; expected {EXPECTED_HEADER_LENGTH}")
    total_header = DATA_ITEM.size + header_length
    if len(raw) != total_header:
        raise FetchError(f"header response has {len(raw)} bytes; expected {total_header}")
    try:
        header = json.loads(raw[DATA_ITEM.size : total_header])
    except json.JSONDecodeError as exc:
        raise FetchError("safetensors header JSON is invalid") from exc
    if not isinstance(header, dict):
        raise FetchError("safetensors header is not an object")
    return header, total_header, hashlib.sha256(raw).hexdigest()


def _product(shape: list[int]) -> int:
    result = 1
    for dimension in shape:
        if not isinstance(dimension, int) or dimension <= 0:
            raise FetchError(f"invalid tensor shape {shape!r}")
        result *= dimension
    return result


def depth_entries(index: dict[str, Any], header: dict[str, Any], data_base: int) -> list[dict[str, Any]]:
    weight_map = index["weight_map"]
    names = sorted(name for name in weight_map if name.startswith(DEPTH_PREFIX))
    entries: list[dict[str, Any]] = []
    for name in names:
        item = header.get(name)
        if not isinstance(item, dict):
            raise FetchError(f"header is missing indexed tensor {name}")
        dtype = item.get("dtype")
        shape = item.get("shape")
        offsets = item.get("data_offsets")
        if dtype != EXPECTED_DTYPE or not isinstance(shape, list) or not isinstance(offsets, list) or len(offsets) != 2:
            raise FetchError(f"unexpected safetensors metadata for {name}")
        if not all(isinstance(offset, int) and offset >= 0 for offset in offsets) or offsets[1] <= offsets[0]:
            raise FetchError(f"invalid data offsets for {name}: {offsets!r}")
        size = offsets[1] - offsets[0]
        expected_size = _product(shape) * 2
        if size != expected_size:
            raise FetchError(f"{name} has {size} bytes; shape/dtype require {expected_size}")
        entries.append(
            {
                "name": name,
                "dtype": dtype,
                "shape": shape,
                "data_offsets": offsets,
                "source_byte_start": data_base + offsets[0],
                "source_byte_end": data_base + offsets[1],
                "nbytes": size,
            }
        )
    if len(entries) != EXPECTED_TENSORS:
        raise FetchError(f"expected {EXPECTED_TENSORS} depth tensors, got {len(entries)}")
    ordered = sorted(entries, key=lambda entry: entry["source_byte_start"])
    for previous, current in zip(ordered, ordered[1:]):
        if previous["source_byte_end"] != current["source_byte_start"]:
            raise FetchError("depth tensor source ranges are not contiguous")
    if sum(entry["nbytes"] for entry in entries) != EXPECTED_TOTAL_BYTES:
        raise FetchError(f"unexpected depth byte total {sum(entry['nbytes'] for entry in entries)}")
    return entries


def _sha256(stream, length: int | None = None) -> str:
    digest = hashlib.sha256()
    remaining = length
    while remaining is None or remaining:
        want = CHUNK if remaining is None else min(CHUNK, remaining)
        block = stream.read(want)
        if not block:
            if remaining:
                raise FetchError("short tensor payload")
            break
        digest.update(block)
        if remaining is not None:
            remaining -= len(block)
    return digest.hexdigest()


def _assert_finite_bf16(block: bytes, name: str) -> None:
    if len(block) % 2:
        raise FetchError(f"{name} has an odd BF16 payload length")
    for index in range(0, len(block), 2):
        value = block[index] | (block[index + 1] << 8)
        if value & 0x7F80 == 0x7F80:
            raise FetchError(f"{name} contains a non-finite BF16 value")


def verify_tensor_file(stream, entry: dict[str, Any], range_start: int) -> str:
    offset = entry["source_byte_start"] - range_start
    if offset < 0:
        raise FetchError(f"{entry['name']} starts before the selected range")
    stream.seek(offset)
    digest = hashlib.sha256()
    remaining = entry["nbytes"]
    while remaining:
        block = stream.read(min(CHUNK, remaining))
        if not block:
            raise FetchError(f"short payload for {entry['name']}")
        digest.update(block)
        _assert_finite_bf16(block, entry["name"])
        remaining -= len(block)
    return digest.hexdigest()


def verify_cache(cache: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    selected = manifest.get("selected", {})
    file_name = selected.get("cache_file")
    range_start = selected.get("source_range", {}).get("start")
    range_bytes = selected.get("source_range", {}).get("bytes")
    range_sha = selected.get("range_sha256")
    if not isinstance(file_name, str) or not isinstance(range_start, int) or not isinstance(range_bytes, int):
        raise FetchError("manifest has no selected cache range")
    path = cache / file_name
    if not path.is_file() or path.stat().st_size != range_bytes:
        raise FetchError(f"cache range missing or wrong size: {path}")
    with path.open("rb") as stream:
        actual_range_sha = _sha256(stream)
    if range_sha and actual_range_sha != range_sha:
        raise FetchError(f"cache range SHA256 mismatch: {actual_range_sha} != {range_sha}")
    actual_tensors = []
    with path.open("rb") as stream:
        for entry in manifest.get("tensors", []):
            actual_sha = verify_tensor_file(stream, entry, range_start)
            expected_sha = entry.get("sha256")
            if expected_sha and actual_sha != expected_sha:
                raise FetchError(f"{entry['name']} SHA256 mismatch: {actual_sha} != {expected_sha}")
            actual_tensors.append(actual_sha)
    if len(actual_tensors) != EXPECTED_TENSORS:
        raise FetchError(f"manifest contains {len(actual_tensors)} tensors; expected {EXPECTED_TENSORS}")
    return {"cache_file": str(path), "range_bytes": range_bytes, "range_sha256": actual_range_sha,
            "tensor_count": len(actual_tensors), "tensor_bytes": sum(entry["nbytes"] for entry in manifest["tensors"])}


def make_manifest(index: dict[str, Any], index_sha: str, header: dict[str, Any], header_bytes: int,
                  header_sha: str, shard_size: int, entries: list[dict[str, Any]], cache_path: Path) -> dict[str, Any]:
    range_start = min(entry["source_byte_start"] for entry in entries)
    range_end = max(entry["source_byte_end"] for entry in entries)
    with cache_path.open("rb") as stream:
        range_sha = _sha256(stream)
        for entry in entries:
            entry["local_offset"] = entry["source_byte_start"] - range_start
            entry["sha256"] = verify_tensor_file(stream, entry, range_start)
    return {
        "schema": "breeze-depth-master/v1",
        "source": {
            "model": MODEL,
            "revision": REVISION,
            "index_url": INDEX_URL,
            "index_sha256": index_sha,
            "shard": SHARD,
            "shard_url": SHARD_URL,
            "shard_bytes": shard_size,
        },
        "safetensors": {
            "header_length": header_bytes - DATA_ITEM.size,
            "header_bytes": header_bytes,
            "header_sha256": header_sha,
            "data_base": header_bytes,
        },
        "selected": {
            "prefix": DEPTH_PREFIX,
            "tensor_count": len(entries),
            "dtype": EXPECTED_DTYPE,
            "tensor_bytes": sum(entry["nbytes"] for entry in entries),
            "source_range": {"start": range_start, "end_exclusive": range_end, "bytes": range_end - range_start},
            "cache_file": cache_path.name,
            "range_sha256": range_sha,
        },
        "verification": {
            "status": "passed",
            "finite_check": "all BF16 exponent patterns with exponent=255 rejected",
            "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        },
        "tensors": entries,
        "notes": [
            "Source offsets are absolute shard offsets; local_offset indexes cache_file.",
            "This cache contains depth tensors only; it is not a complete model checkpoint.",
        ],
    }


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as stream:
        temp_path = Path(stream.name)
        json.dump(value, stream, indent=2, sort_keys=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp_path, path)


def download(manifest_path: Path, cache: Path) -> dict[str, Any]:
    if manifest_path.is_file():
        try:
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
            if existing.get("source", {}).get("revision") == REVISION:
                return verify_cache(cache, existing)
        except (OSError, json.JSONDecodeError, FetchError):
            pass
    index_raw = curl_bytes(INDEX_URL)
    index, index_sha = parse_index(index_raw)
    header_path = cache / "safetensors-header.bin"
    cache.mkdir(parents=True, exist_ok=True)
    # The first response is exactly the 8-byte length plus the pinned 101880-byte JSON header.
    header_shard_size = fetch_range(SHARD_URL, 0, DATA_ITEM.size + EXPECTED_HEADER_LENGTH - 1, header_path)
    header_raw = header_path.read_bytes()
    header, data_base, header_sha = parse_header(header_raw)
    entries = depth_entries(index, header, data_base)
    expected_shard_size = data_base + max(
        int(item["data_offsets"][1])
        for item in header.values()
        if isinstance(item, dict) and "data_offsets" in item
    )
    if header_shard_size != expected_shard_size:
        raise FetchError(f"header range shard size {header_shard_size} disagrees with safetensors offsets")
    range_start = min(entry["source_byte_start"] for entry in entries)
    range_end = max(entry["source_byte_end"] for entry in entries) - 1
    range_path = cache / "depth-range.bin"
    shard_size = fetch_range(SHARD_URL, range_start, range_end, range_path, expected_total=expected_shard_size)
    manifest = make_manifest(index, index_sha, header, data_base, header_sha, shard_size, entries, range_path)
    write_json_atomic(manifest_path, manifest)
    return verify_cache(cache, manifest)


def verify(manifest_path: Path, cache: Path) -> dict[str, Any]:
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FetchError(f"cannot read manifest {manifest_path}") from exc
    if manifest.get("source", {}).get("revision") != REVISION:
        raise FetchError("manifest revision is not the pinned immutable revision")
    return verify_cache(cache, manifest)


def _expect_failure(function, label: str) -> None:
    try:
        function()
    except FetchError:
        return
    raise AssertionError(f"self-test accepted invalid {label}")


def self_test() -> None:
    valid_headers = "HTTP/2 302\n\nHTTP/2 206\nContent-Length: 4\nContent-Range: bytes 2-5/20\n\n"
    assert validate_range_response(valid_headers, 4, 2, 5) == 20
    _expect_failure(lambda: validate_range_response(valid_headers.replace("2-5/20", "2-4/20"), 4, 2, 5), "range")
    _expect_failure(lambda: validate_range_response(valid_headers.replace("Content-Length: 4", "Content-Length: 3"), 4, 2, 5), "length")
    _expect_failure(lambda: validate_range_response(valid_headers.replace("HTTP/2 206", "HTTP/2 200"), 4, 2, 5), "status")
    raw = DATA_ITEM.pack(2) + b'{}'  # parser's pinned-length guard is tested separately below
    _expect_failure(lambda: parse_header(raw), "safetensors header length")
    _assert_finite_bf16(struct.pack("<HH", 0x3F80, 0x0000), "finite")
    _expect_failure(lambda: _assert_finite_bf16(struct.pack("<H", 0x7F80), "infinite"), "non-finite BF16")
    _expect_failure(lambda: _product([0]), "tensor shape")
    print("fetch_depth_master self-test: PASS")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--download", action="store_true", help="download and verify the selected depth range")
    mode.add_argument("--verify", action="store_true", help="verify an existing cache from the manifest")
    mode.add_argument("--self-test", action="store_true", help="run offline range/header validation checks")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()
    try:
        if args.self_test:
            self_test()
            return
        cache = args.cache_dir.expanduser()
        if args.download:
            result = download(args.manifest, cache)
        else:
            result = verify(args.manifest, cache)
        print(json.dumps(result, sort_keys=True))
    except FetchError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
