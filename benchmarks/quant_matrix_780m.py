#!/usr/bin/env python3
"""Analyze an archived 780M Q3_K/Q4_K matrix run directory.

The runner deliberately stays elsewhere: this file only reads manifest,
logs, return codes, commands, and WAV receipts.  UTMOS is an optional,
cached-local descriptive metric; all timing and verification paths are
stdlib-only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import statistics
import subprocess
import tempfile
import wave
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT = Path(__file__).resolve().parent / "depth-corpus" / "quant-matrix.json"
STAGES = (
    "reference encode",
    "prompt build",
    "backbone prefill",
    "first vocoder",
    "backbone decode",
    "depth decode",
    "vocoder",
)
WALL_RE = re.compile(r"generation wall\s+([\d.]+) ms \(wall RTF ([\d.]+)\)")
WROTE_RE = re.compile(r"^\s*wrote\s+.*(?:\n|$)", re.MULTILINE)
FLUSH_RE = re.compile(
    # The CLI's progress carriage return can leave later flush markers after
    # ``generated ...`` on the same physical log line.
    r"flush\s+(\d+)\s+ready_ms=([\d.]+)\s+delivered_audio_s=([\d.]+)",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(4 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _p95(values: list[float]) -> float:
    if len(values) < 2:
        return values[0]
    return statistics.quantiles(values, n=20, method="inclusive")[18]


def max_playback_deficit(flushes: list[dict[str, float]]) -> float:
    """Return the prior receipt's delivery deficit formula, in milliseconds."""
    if not flushes:
        return 0.0
    first = flushes[0]["ready_ms"]
    return max(
        0.0,
        max(
            ready - first - (flushes[index - 1]["delivered_audio_s"] * 1000 if index else 0)
            for index, item in enumerate(flushes)
            for ready in [item["ready_ms"]]
        ),
    )


def _run_regions(log: str) -> list[str]:
    """Split a repeated CLI log while retaining each run's flush lines."""
    starts = [match.start() for match in re.finditer(r"^\s*flush\s+1\s+ready_ms=", log, re.MULTILINE)]
    walls = list(WALL_RE.finditer(log))
    if starts and len(starts) == len(walls):
        return [log[start : starts[index + 1] if index + 1 < len(starts) else len(log)]
                for index, start in enumerate(starts)]

    writes = list(WROTE_RE.finditer(log))
    if len(writes) >= len(walls):
        regions = []
        previous = 0
        for index, _wall in enumerate(walls):
            end = writes[index].end()
            regions.append(log[previous:end])
            previous = end
        return regions

    # This fallback is useful for a truncated synthetic receipt.  Parsing will
    # still fail with the missing timing field instead of silently mispairing runs.
    return [log[start:end] for start, end in zip(
        [0] + [match.end() for match in walls[:-1]],
        [match.start() for match in walls[1:]] + [len(log)],
    )]


def verify_flushes(flushes: list[dict[str, float]], audio_s: float) -> None:
    if [item["index"] for item in flushes] != list(range(1, len(flushes) + 1)):
        raise ValueError("flush indices must be consecutive")
    if not flushes or abs(flushes[-1]["delivered_audio_s"] - audio_s) > 0.0051:
        raise ValueError("last flush duration differs from generated audio")
    for previous, current in zip(flushes, flushes[1:]):
        if current["ready_ms"] < previous["ready_ms"] or current["delivered_audio_s"] <= previous["delivered_audio_s"]:
            raise ValueError("flush times/durations must be monotonic")


def parse_run_log(log: str, repeat: int) -> dict[str, Any]:
    """Parse one ``breeze-cli --timings`` resident repeat."""
    wall = WALL_RE.search(log)
    if not wall:
        raise ValueError(f"repeat {repeat}: missing generation wall timing")
    stages: dict[str, float] = {}
    per_frame: dict[str, float] = {}
    for stage in STAGES:
        match = re.search(
            rf"^\s*{re.escape(stage)}\s+([\d.]+) ms(?:\s+\(([\d.]+) ms/frame\))?",
            log,
            re.MULTILINE,
        )
        if not match:
            raise ValueError(f"repeat {repeat}: missing {stage} timing")
        stages[stage] = float(match.group(1))
        if match.group(2) is not None:
            per_frame[stage] = float(match.group(2))

    frames_match = re.search(r"^\s*(\d+) frames,\s*([\d.]+) s audio", log, re.MULTILINE)
    first_match = re.search(r"time to first audio\s+([\d.]+) ms over (\d+) flushes", log)
    if not frames_match or not first_match:
        raise ValueError(f"repeat {repeat}: missing frame or first-audio timing")
    frames = int(frames_match.group(1))
    audio_s = float(frames_match.group(2))
    flushes = [
        {"index": int(index), "ready_ms": float(ready), "delivered_audio_s": float(duration)}
        for index, ready, duration in FLUSH_RE.findall(log)
    ]
    expected_flushes = int(first_match.group(2))
    if len(flushes) != expected_flushes:
        raise ValueError(
            f"repeat {repeat}: expected {expected_flushes} flushes, parsed {len(flushes)}"
        )
    if frames <= 0 or audio_s <= 0:
        raise ValueError(f"repeat {repeat}: non-positive audio receipt")

    verify_flushes(flushes, audio_s)
    wall_ms = float(wall.group(1))
    if abs(wall_ms / (1000 * audio_s) - float(wall.group(2))) > 0.000501:
        raise ValueError("reported wall RTF differs from wall/duration")
    decode_ms = sum(stages[name] for name in ("backbone decode", "depth decode", "vocoder"))
    result: dict[str, Any] = {
        "repeat": repeat,
        "wall_ms": wall_ms,
        "wall_rtf": wall_ms / (1000.0 * audio_s),
        "wall_rtf_reported": float(wall.group(2)),
        "stages_ms": stages,
        "stage_rtf": sum(value for name, value in stages.items() if name != "first vocoder") / (1000.0 * audio_s),
        "decode_rtf": decode_ms / (1000.0 * audio_s),
        "frames": frames,
        "audio_s": audio_s,
        "first_audio_ms": float(first_match.group(1)),
        "flushes": expected_flushes,
        "flushes_ready": flushes,
        "max_playback_deficit_ms": max_playback_deficit(flushes),
        "max_delivery_deficit_ms": max_playback_deficit(flushes),
    }
    result.update({f"{name.replace(' ', '_')}_ms_per_frame": value for name, value in per_frame.items()})
    return result


def parse_case_log(log: str, repeats: int = 4) -> list[dict[str, Any]]:
    regions = _run_regions(log)
    if len(regions) != repeats:
        raise ValueError(f"expected {repeats} repeated timing blocks, parsed {len(regions)}")
    return [parse_run_log(region, index) for index, region in enumerate(regions, 1)]


def _wav_info(path: Path) -> tuple[int, float]:
    with wave.open(str(path), "rb") as wav:
        if (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) != (1, 2, 24000):
            raise ValueError(f"{path.name}: expected mono 16-bit 24000 Hz WAV")
        return wav.getframerate(), wav.getnframes() / wav.getframerate()


def _load_wav_window(path: Path, start_s: float, duration_s: float):
    """Load one bounded 16 kHz mono window for the optional UTMOS pass."""
    import numpy as np

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg is not available")
    output = subprocess.check_output(
        [
            ffmpeg,
            "-v",
            "error",
            "-ss",
            f"{start_s:.6f}",
            "-t",
            f"{duration_s:.6f}",
            "-i",
            str(path),
            "-f",
            "f32le",
            "-ar",
            "16000",
            "-ac",
            "1",
            "-",
        ],
        stderr=subprocess.STDOUT,
    )
    return np.frombuffer(output, dtype="<f4").copy()


def _utmos(cases: dict[str, Any], run_dir: Path) -> tuple[dict[str, Any], dict[str, float]]:
    """Score unique WAV hashes from a cached local SpeechMOS checkout only."""
    try:
        import numpy as np  # noqa: F401  (required by _load_wav_window)
        import torch
    except ImportError as exc:
        return {"status": "unavailable", "reason": f"optional dependency unavailable: {exc}",
                "descriptive_only": True}, {}

    hub_dir = Path(torch.hub.get_dir())
    source_dir = hub_dir / "tarepan_SpeechMOS_v1.2.0"
    checkpoint = hub_dir / "checkpoints" / "utmos22_strong_step7459_v1.pt"
    if not source_dir.is_dir() or not checkpoint.is_file():
        return {
            "status": "unavailable",
            "reason": "cached SpeechMOS source or checkpoint is missing; network access is disabled",
            "source_cache": str(source_dir),
            "checkpoint": str(checkpoint),
            "descriptive_only": True,
        }, {}

    try:
        model = torch.hub.load(
            str(source_dir), "utmos22_strong", source="local", progress=False, trust_repo=True
        )
        model.eval()
        torch.set_num_threads(2)
    except Exception as exc:  # cache incompatibility should not block timing analysis
        return {
            "status": "unavailable",
            "reason": f"cached UTMOS load failed: {type(exc).__name__}: {exc}",
            "source_cache": str(source_dir),
            "checkpoint": str(checkpoint),
            "descriptive_only": True,
        }, {}

    paths_by_hash: dict[str, Path] = {}
    for case in cases.values():
        for run in case["runs"]:
            paths_by_hash.setdefault(run["sha256"], run["wav_path"])

    scores: dict[str, Any] = {}
    values: dict[str, float] = {}
    for digest, path in paths_by_hash.items():
        _, duration = _wav_info(path)
        windows = [(0.0, duration, "full")] if duration <= 10.0 else [
            (start, min(10.0, duration - start), f"fixed10s-window-{index:03d}")
            for index, start in enumerate(range(0, math.ceil(duration), 10))
            if duration - start > 0.25
        ]
        scored = []
        try:
            for start, length, label in windows:
                audio = _load_wav_window(path, start, length)
                with torch.inference_mode():
                    score = float(model(torch.from_numpy(audio)[None, :], 16000)[0])
                scored.append({"label": label, "start_s": start, "duration_s": length, "score": score})
            values[digest] = statistics.mean(item["score"] for item in scored)
            scores[digest] = {
                "sha256": digest,
                "path": path.name,
                "duration_s": duration,
                "score": values[digest],
                "windows": scored,
            }
        except Exception as exc:
            scores[digest] = {"sha256": digest, "path": path.name, "duration_s": duration,
                              "error": f"{type(exc).__name__}: {exc}"}

    return {
        "status": "scored" if len(values) == len(paths_by_hash) else "partial" if values else "unavailable",
        "source": "cached local torch hub tarepan/SpeechMOS:v1.2.0",
        "source_cache": str(source_dir),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "window_policy": "full clip up to 10 s; fixed10s windows for longer clips",
        "descriptive_only": True,
        "scores": scores,
        "limitations": [
            "UTMOS is a speech-quality proxy, not an acceptance gate.",
            "No ASR intelligibility or speaker-similarity claim is made.",
        ],
    }, values


def _model_kinds(manifest: dict[str, Any], preflight: dict[str, Any]) -> list[str]:
    kinds = list(manifest.get("model_hashes", {})) or list(preflight.get("model_hashes", {}))
    return kinds or ["q4_k", "q3_k"]


def analyze_run_dir(run_dir: str | Path, output: str | Path | None = None, use_utmos: bool = True) -> dict[str, Any]:
    """Analyze all manifest fixtures and model kinds in ``run_dir``."""
    root = Path(run_dir).resolve()
    manifest = _read_json(root / "manifest.json")
    preflight = _read_json(root / "preflight.json")
    manifest_models = manifest.get("model_hashes", {})
    preflight_models = preflight.get("model_hashes", {})
    if manifest_models and preflight_models and manifest_models != preflight_models:
        raise ValueError("manifest and preflight model hashes differ")
    if manifest.get("cli_md5") and preflight.get("cli_md5") and manifest["cli_md5"] != preflight["cli_md5"]:
        raise ValueError("manifest and preflight CLI hashes differ")
    repeats = int(manifest.get("repeats", 4))
    cases: dict[str, Any] = {}
    expected_ids = []
    model_kinds = _model_kinds(manifest, preflight)
    for fixture in manifest["fixtures"]:
        for kind in model_kinds:
            case_id = f"{fixture['id']}-{kind}"
            expected_ids.append(case_id)
            log_path = root / f"{case_id}.log"
            rc_path = root / f"{case_id}.rc"
            if not log_path.is_file() or not rc_path.is_file():
                raise FileNotFoundError(f"missing receipt for {case_id}")
            rc = int(rc_path.read_text().strip())
            if rc != 0:
                raise RuntimeError(f"{case_id} exited with return code {rc}")
            log = log_path.read_text(encoding="utf-8", errors="replace")
            runs = parse_case_log(log, repeats)
            command_path = root / f"{case_id}.command.json"
            command = _read_json(command_path) if command_path.is_file() else None
            model_sha = preflight.get("model_hashes", {}).get(kind) or manifest.get("model_hashes", {}).get(kind)
            wav_paths = [root / f"{case_id}.wav"] + [root / f"{case_id}.repeat-{i}.wav" for i in range(2, repeats + 1)]
            for index, (run, wav_path) in enumerate(zip(runs, wav_paths), 1):
                if not wav_path.is_file():
                    raise FileNotFoundError(f"missing WAV receipt {wav_path.name}")
                run["wav"] = wav_path.name
                run["wav_path"] = wav_path
                run["sha256"] = _sha256(wav_path)
                run["wav_sample_rate"], run["wav_duration_s"] = _wav_info(wav_path)
                if abs(run["wav_duration_s"] - run["audio_s"]) > 0.0051:
                    raise ValueError(f"{case_id}: WAV duration differs from timing receipt")
                if abs(run["wav_duration_s"] - run["frames"] * 0.08) > 1 / 24000:
                    raise ValueError(f"{case_id}: WAV frames differ from timing receipt")
                if index != run["repeat"]:
                    raise ValueError(f"{case_id}: repeat ordering mismatch")
            rtfs = [run["wall_rtf"] for run in runs[1:]]
            frame_counts = [run["frames"] for run in runs]
            max_new = fixture.get("max_new")
            if max_new is None:
                termination = None
            else:
                at_cap = [frames >= int(max_new) for frames in frame_counts]
                expected = "at_cap" if fixture.get("kind") == "capped-throughput-only" else "below_cap"
                termination = {
                    "kind": fixture.get("kind"),
                    "max_new": int(max_new),
                    "cap_reached": all(at_cap),
                    "status": "reached" if all(at_cap) else "below-cap" if not any(at_cap) else "mixed",
                    "expected": expected,
                    "pass": all(at_cap) if expected == "at_cap" else not any(at_cap),
                    "frames": frame_counts,
                }
            cases[case_id] = {
                "fixture": fixture,
                "model": kind,
                "rc": rc,
                "command": command,
                "log": log,
                "log_sha256": _sha256(log_path),
                "model_sha256": model_sha,
                "cli_md5": preflight.get("cli_md5", manifest.get("cli_md5")),
                "runs": runs,
                "repeatable": len({run["sha256"] for run in runs}) == 1,
                "warm_mean_rtf": statistics.mean(rtfs),
                "warm_p50_rtf": statistics.median(rtfs),
                "warm_sample_p50_rtf": statistics.median(rtfs),
                "warm_sample_p95_rtf": _p95(rtfs),
                "termination": termination,
            }

    for case in cases.values():
        warm = case["runs"][1:]
        control = cases[f"{case['fixture']['id']}-q4_k"]
        case["gates"] = {
            "rtf": case["warm_sample_p50_rtf"] <= 0.8 and case["warm_sample_p95_rtf"] <= 0.8,
            "delivery": all(run["max_delivery_deficit_ms"] == 0 for run in warm),
            "first_audio_warm_mean": statistics.mean(run["first_audio_ms"] for run in warm) <= statistics.mean(run["first_audio_ms"] for run in control["runs"][1:]),
            "repeatability": case["repeatable"],
            "termination": case["termination"] is not None and case["termination"]["pass"],
        }
    utmos_report: dict[str, Any]
    utmos_values: dict[str, float]
    if use_utmos:
        utmos_report, utmos_values = _utmos(cases, root)
    else:
        utmos_report, utmos_values = {"status": "skipped", "descriptive_only": True}, {}
    for case in cases.values():
        for run in case["runs"]:
            if run["sha256"] in utmos_values:
                run["utmos"] = utmos_values[run["sha256"]]
            run.pop("wav_path", None)

    provenance = {
        "manifest": manifest,
        "preflight": preflight,
        "cli_md5": preflight.get("cli_md5", manifest.get("cli_md5")),
        "model_hashes": preflight.get("model_hashes", manifest.get("model_hashes", {})),
        "manifest_preflight_match": {
            "cli_md5": not manifest.get("cli_md5") or not preflight.get("cli_md5") or manifest["cli_md5"] == preflight["cli_md5"],
            "model_hashes": not manifest_models or not preflight_models or manifest_models == preflight_models,
        },
    }
    result = {
        "schema": "breeze-quant-matrix/v1",
        "run_dir": str(root),
        "provenance": provenance,
        "manifest": manifest,
        "preflight": preflight,
        "cases": cases,
        "repeat_count": repeats,
        "case_count": len(cases),
        "quality_method": f"Cached SpeechMOS v1.2.0 utmos22_strong: {utmos_report['status']}; descriptive only",
        "gate_verdict": {"performance": "pass" if all(all(case["gates"].values()) for case in cases.values() if case["model"] == "q3_k") else "fail", "quality": "inconclusive", "target_achieved": False},
        "utmos": utmos_report,
        "quality": {"utmos": utmos_report, "acceptance": "not assessed"},
        "limitations": [
            "UTMOS is descriptive only and supplies no speech acceptance decision.",
            "No ASR intelligibility or speaker-similarity claim is included.",
        ],
    }
    if list(cases) != expected_ids:
        raise AssertionError("case ordering mismatch")
    if output is not None:
        target = Path(output)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    return result


def verify_report(path: str | Path) -> None:
    """Verify archived arithmetic without importing torch or touching WAV files."""
    report = _read_json(Path(path))
    assert report.get("schema") == "breeze-quant-matrix/v1"
    assert report.get("case_count") == len(report["cases"])
    expected = {f"{fixture['id']}-{kind}" for fixture in report['manifest']['fixtures'] for kind in report['manifest']['model_hashes']}
    assert set(report['cases']) == expected
    for name, case in report["cases"].items():
        runs = case["runs"]
        assert len(runs) == report["repeat_count"]
        assert [run["repeat"] for run in runs] == list(range(1, len(runs) + 1)), name
        assert case["repeatable"] == (len({run["sha256"] for run in runs}) == 1)
        for run in runs:
            duration = run["audio_s"]
            assert duration > 0 and run["frames"] > 0
            assert abs(run["wav_duration_s"] - duration) <= 0.0051
            assert abs(run["wav_duration_s"] - run["frames"] * 0.08) <= 1 / 24000
            assert math.isclose(run["wall_rtf"], run["wall_ms"] / (1000 * duration), rel_tol=1e-12)
            assert math.isclose(run["stage_rtf"], sum(value for stage, value in run["stages_ms"].items() if stage != "first vocoder") / (1000 * duration), rel_tol=1e-12)
            decode = sum(run["stages_ms"][key] for key in ("backbone decode", "depth decode", "vocoder"))
            assert math.isclose(run["decode_rtf"], decode / (1000 * duration), rel_tol=1e-12)
            assert abs(run["wall_rtf_reported"] - run["wall_rtf"]) <= 0.000501
            assert len(run["flushes_ready"]) == run["flushes"]
            verify_flushes(run["flushes_ready"], duration)
            deficit = max_playback_deficit(run["flushes_ready"])
            assert math.isclose(run["max_playback_deficit_ms"], deficit, rel_tol=1e-12, abs_tol=1e-12)
            assert math.isclose(run["max_delivery_deficit_ms"], deficit, rel_tol=1e-12, abs_tol=1e-12)
        warm = [run["wall_rtf"] for run in runs[1:]]
        assert math.isclose(case["warm_mean_rtf"], statistics.mean(warm), rel_tol=1e-12)
        assert math.isclose(case["warm_p50_rtf"], statistics.median(warm), rel_tol=1e-12)
        if "warm_sample_p50_rtf" in case:
            assert math.isclose(case["warm_sample_p50_rtf"], statistics.median(warm), rel_tol=1e-12)
        assert math.isclose(case["warm_sample_p95_rtf"], _p95(warm), rel_tol=1e-12)

        assert case["gates"]["rtf"] == (case["warm_sample_p50_rtf"] <= 0.8 and case["warm_sample_p95_rtf"] <= 0.8)
        assert case["gates"]["delivery"] == all(run["max_delivery_deficit_ms"] == 0 for run in runs[1:])
        assert case["gates"]["repeatability"] == case["repeatable"]
        assert case["gates"]["termination"] == case["termination"]["pass"]
        control = report["cases"][f"{case['fixture']['id']}-q4_k"]
        assert case["gates"]["first_audio_warm_mean"] == (statistics.mean(run["first_audio_ms"] for run in runs[1:]) <= statistics.mean(run["first_audio_ms"] for run in control["runs"][1:]))
    passed = all(all(case["gates"].values()) for case in report["cases"].values() if case["model"] == "q3_k")
    assert report["gate_verdict"]["performance"] == ("pass" if passed else "fail")
    assert report["gate_verdict"]["quality"] == "inconclusive" and not report["gate_verdict"]["target_achieved"]


def _write_synthetic_wav(path: Path, seconds: float = 2.0) -> None:
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(24000)
        wav.writeframes(b"\0\0" * int(24000 * seconds))


def self_test() -> None:
    """Exercise repeated log splitting, flush deficit math, hashes, and verify."""
    with tempfile.TemporaryDirectory(prefix="quant-matrix-self-test-") as temporary:
        root = Path(temporary)
        manifest = {
            "schema": "breeze-q3-matrix/v1",
            "seed": 42,
            "repeats": 4,
            "fixtures": [{"id": "demo", "text": "demo", "max_new": 25, "kind": "capped-throughput-only"}],
            "model_hashes": {"q4_k": "q4", "q3_k": "q3"},
            "cli_md5": "cli",
        }
        preflight = {"model_hashes": {"q4_k": "q4", "q3_k": "q3"}, "cli_md5": "cli"}
        (root / "manifest.json").write_text(json.dumps(manifest))
        (root / "preflight.json").write_text(json.dumps(preflight))
        for kind in ("q4_k", "q3_k"):
            case = f"demo-{kind}"
            blocks = []
            for repeat in range(1, 5):
                ready = 100 + repeat * 1000
                blocks.append(
                    f"  flush 1 ready_ms={ready:.1f} delivered_audio_s=1.00\n"
                    f"  flush 2 ready_ms={ready + 1700:.1f} delivered_audio_s=2.00\n"
                    f"generation wall     {2000 + repeat * 100:.1f} ms (wall RTF {(2000 + repeat * 100) / 2000:.3f})\n"
                    "time to first audio 1000 ms over 2 flushes\n"
                    "  reference encode       0.0 ms\n"
                    "  prompt build          10.0 ms\n"
                    "  backbone prefill      20.0 ms\n"
                    "  first vocoder         30.0 ms  (40 frames)\n"
                    "  backbone decode       40.0 ms  (1.60 ms/frame)\n"
                    "  depth decode          50.0 ms  (2.00 ms/frame)\n"
                    "  vocoder               60.0 ms  (2.40 ms/frame)\n"
                    "  25 frames, 2.00 s audio\n"
                    f"wrote /tmp/{case}{'' if repeat == 1 else f'.repeat-{repeat}'}.wav (2.00 s)\n"
                )
            (root / f"{case}.log").write_text("".join(blocks))
            (root / f"{case}.rc").write_text("0\n")
            (root / f"{case}.command.json").write_text(json.dumps(["breeze-cli", kind]))
            wav = root / f"{case}.wav"
            _write_synthetic_wav(wav)
            for repeat in range(2, 5):
                (root / f"{case}.repeat-{repeat}.wav").write_bytes(wav.read_bytes())
        report_path = root / "report.json"
        report = analyze_run_dir(root, report_path, use_utmos=False)
        assert report["gate_verdict"]["performance"] == "fail"
        try:
            verify_flushes([{"index": 2, "ready_ms": 1, "delivered_audio_s": 2}], 2)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid flush accepted")
        assert report["case_count"] == 2
        assert report["cases"]["demo-q4_k"]["repeatable"]
        assert report["cases"]["demo-q4_k"]["runs"][0]["max_playback_deficit_ms"] == 700.0
        assert report["cases"]["demo-q4_k"]["runs"][0]["stage_rtf"] == 0.09
        verify_report(report_path)
    print("PASS: synthetic repeated log/flush parser and archived arithmetic verification")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", nargs="?", help="completed local run directory")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="report JSON path")
    parser.add_argument("--no-utmos", action="store_true", help="skip optional cached UTMOS scoring")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--verify", nargs="?", const=str(DEFAULT_OUTPUT), metavar="JSON")
    args = parser.parse_args(argv)
    if args.self_test:
        self_test()
        return
    if args.verify:
        verify_report(args.verify)
        print(f"PASS: verified {args.verify}")
        return
    if not args.run_dir:
        parser.error("run_dir is required unless --self-test or --verify is used")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    report = analyze_run_dir(args.run_dir, output, use_utmos=not args.no_utmos)
    print(f"wrote {output} ({report['case_count']} cases, {report['repeat_count']} repeats)")


if __name__ == "__main__":
    main()
