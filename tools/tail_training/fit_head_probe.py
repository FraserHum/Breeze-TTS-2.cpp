"""Train-set memorization probe: solve or gradient-fit the frozen student's head.

Uses float64 GELSD with the library's default rank tolerance; validation never
selects weights or solver settings. Success demonstrates memorization, not
generalization. --gradient uses waveform MSE and optionally --device cuda.
Run --self-check for a known full-row-rank linear problem.
"""
import argparse
import json
from pathlib import Path
import wave
import time

import numpy as np
import torch

from analyze_audio import analyze
from student import Student, read_f32


def solve(h, y):
    return torch.linalg.lstsq(h.double(), y.double(), rcond=None, driver="gelsd")


def gradient_fit(h, y, initial, steps, device, minutes):
    h, y = h.to(device), y.to(device)
    w = torch.nn.Parameter(initial.to(device).clone())
    optimizer = torch.optim.Adam([w], lr=1e-3)
    started, history = time.monotonic(), []
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = (h @ w - y).square().mean()
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite head MSE")
        loss.backward()
        optimizer.step()
        if step == 0 or (step + 1) % 250 == 0:
            event = dict(step=step + 1, mse_before_update=loss.item(),
                         elapsed_seconds=time.monotonic() - started)
            history.append(event)
            print(json.dumps(event), flush=True)
        if time.monotonic() - started >= minutes * 60:
            break
    return w.detach().cpu(), dict(steps=step + 1, history=history,
        elapsed_seconds=time.monotonic() - started, lr=1e-3,
        optimizer="Adam, no weight decay", batch="all emitted training frames")


def self_check():
    generator = torch.Generator().manual_seed(42)
    h = torch.randn(8, 16, dtype=torch.float64, generator=generator)
    y = torch.randn(8, 24, dtype=torch.float64, generator=generator)
    result = solve(h, y)
    assert result.rank.item() == 8
    torch.testing.assert_close(h @ result.solution, y, rtol=1e-10, atol=1e-10)
    fitted, _ = gradient_fit(torch.eye(4), torch.ones(4, 3) * 0.05,
                            torch.zeros(4, 3), 100, "cpu", 1)
    assert (fitted - 0.05).square().mean() < 1e-6
    print("Least-squares and gradient MSE checks: passed")


def run(args):
    args.out.mkdir(parents=True, exist_ok=False)
    records = json.loads(args.dataset.read_text())
    if isinstance(records, dict):
        records = records["records"]
    ids = {s: {r["utterance_id"] for r in records if r["split"] == s}
           for s in ("train", "validation")}
    if not all(ids.values()) or ids["train"] & ids["validation"]:
        raise ValueError("Require disjoint, nonempty train and validation utterances")
    model = Student(args.weights).cpu().eval()
    key = "student_output_weight"
    original = model.weights[key]
    original_norm = float(original.detach().norm())
    hidden, targets, cached = [], [], []
    with torch.no_grad():
        for record in records:
            if record["split"] not in ids:
                continue
            name = record["utterance_id"]
            if not name or not all(c.isalnum() or c in "-_" for c in name):
                raise ValueError("Unsafe utterance name")
            x = read_f32(args.dataset.parent / record["input"], (1, record["frames"], 1024))
            y = read_f32(args.dataset.parent / record["target"], (record["n"], 1920))
            reference = model(x, record["n"])[0]
            # Replacing only the final linear matrix with identity exposes its
            # input without duplicating the transformer or editing Student.
            model.weights[key] = torch.nn.Parameter(torch.eye(1024), requires_grad=False)
            try:
                h = model(x, record["n"])[0]
            finally:
                model.weights[key] = original
            torch.testing.assert_close(h @ original.T, reference, rtol=1e-5, atol=1e-6)
            if record["split"] == "train":
                hidden.append(h)
                targets.append(y)
            cached.append((record, x, y))
    h, y = torch.cat(hidden), torch.cat(targets)
    result = solve(h, y)
    w = result.solution
    error64 = h.double() @ w - y.double()
    gradient = None
    if args.gradient:
        w, gradient = gradient_fit(h, y, original.detach().T, args.steps,
                                   args.device, args.minutes)
    if not torch.isfinite(w).all():
        raise FloatingPointError("Nonfinite fitted head")
    with torch.no_grad():
        original.copy_(w.T.float())
    model.export(args.out / "weights")
    roundtrip = Student(args.out / "weights").cpu().eval()
    groups = {}
    with torch.no_grad():
        for record, x, target in cached:
            prediction = model(x, record["n"])[0]
            torch.testing.assert_close(roundtrip(x, record["n"])[0], prediction, rtol=0, atol=0)
            groups.setdefault(record["utterance_id"], []).append((record, prediction.numpy(), target.numpy()))
    metrics = {}
    for name, entries in groups.items():
        entries.sort(key=lambda entry: entry[0]["start"])
        offset = 0
        for record, _, _ in entries:
            if record["start"] != offset:
                raise ValueError("Noncontiguous utterance windows")
            offset += record["n"]
        p, t = [np.concatenate([entry[i].ravel() for entry in entries]) for i in (1, 2)]
        metrics[name] = analyze(t.astype(np.float64), p.astype(np.float64), 24000)
        metrics[name].update(split=entries[0][0]["split"],
                             max_absolute_error=float(np.abs(p - t).max()),
                             peak_absolute_prediction=float(np.abs(p).max()),
                             clipping_fraction=float(np.mean(np.abs(p) > 1)))
        for label, samples in (("student", p), ("teacher", t)):
            with wave.open(str(args.out / f"{name}-{label}.wav"), "wb") as wav:
                wav.setparams((1, 2, 24000, 0, "NONE", "not compressed"))
                wav.writeframes((np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes())
    rank = result.rank.item()
    report = dict(source_weights=str(args.weights), dataset=str(args.dataset),
                  solver="Adam waveform MSE" if args.gradient else "CPU float64 GELSD",
                  reference_solver="CPU float64 torch.linalg.lstsq GELSD; rcond=None (default)",
                  torch_version=torch.__version__, threads=torch.get_num_threads(),
                  gradient=gradient, device=args.device if args.gradient else "cpu",
                  feature_shape=list(h.shape), rank=rank,
                  singular_values=result.singular_values.tolist(),
                  retained_condition=float(result.singular_values[0] / result.singular_values[rank-1]) if rank else None,
                  head_frobenius_norm=float(w.norm()),
                  original_head_frobenius_norm=original_norm,
                  direct_reference_float64_mae=float(error64.abs().mean()),
                  direct_reference_float64_max_abs_error=float(error64.abs().max()),
                  raw_float32_metrics=metrics,
                  caveat="Train-frame memorization diagnostic; not a generalization or quality claim. WAVs clipped at serialization.")
    (args.out / "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "singular_values"}, indent=2, allow_nan=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--gradient", action="store_true")
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--minutes", type=float, default=3)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.steps < 1 or not 0 < args.minutes < float("inf"):
        parser.error("Require positive steps and finite positive minutes")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(4)
    if args.self_check:
        self_check()
    else:
        if not all((args.weights, args.dataset, args.out)):
            parser.error("--weights, --dataset and --out are required")
        run(args)
