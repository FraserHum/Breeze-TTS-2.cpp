"""Exact experimental waveform student and head widening utility.

Usage: 
  python student.py check CAPTURE_DIR --export UPDATED_DIR
  python student.py widen SOURCE_DIR OUT_DIR --head-dim 2048
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def expected_shapes(head_dim=1024):
    shapes = {
        "codec.dtf.in_proj.weight": [1024, 512],
        "codec.dtf.in_proj.bias": [512],
        "codec.dtf.norm.weight": [512],
        "codec.dtf.out_proj.weight": [512, head_dim],
        "codec.dtf.out_proj.bias": [head_dim],
        "student.output.weight": [head_dim, 1920],
    }
    for i in range(4):
        p = f"codec.dtf.blk.{i}."
        for name in ("attn_norm.weight", "ffn_norm.weight", "attn_scale", "ffn_scale"):
            shapes[p + name] = [512]
        for name in ("attn_q.weight", "attn_k.weight", "attn_v.weight",
                     "ffn_gate.weight", "ffn_up.weight"):
            shapes[p + name] = [512, 1024]
        shapes[p + "attn_output.weight"] = [1024, 512]
        shapes[p + "ffn_down.weight"] = [1024, 512]
    return shapes


def read_f32(path, shape):
    data = np.fromfile(path, dtype="<f4")
    if data.size != int(np.prod(shape)) or not np.isfinite(data).all():
        raise ValueError(f"Invalid finite F32 array: {path}, expected {shape}")
    return torch.from_numpy(data.reshape(shape).copy())


class Student(nn.Module):
    def __init__(self, directory):
        super().__init__()
        directory = Path(directory)
        entries = json.loads((directory / "weights.json").read_text())
        if isinstance(entries, dict):
            entries = entries["weights"]
        specs = {entry["name"]: entry for entry in entries}
        self.head_dim = specs["codec.dtf.out_proj.bias"]["shape"][0]
        shapes = expected_shapes(head_dim=self.head_dim)
        if len(specs) != len(entries) or set(specs) != set(shapes):
            raise ValueError("Student weight names must match the exact four-block contract")
        self.weights = nn.ParameterDict()
        for name, shape in shapes.items():
            spec = specs[name]
            if spec["shape"] != shape:
                raise ValueError(f"{name}: expected ggml shape {shape}, got {spec['shape']}")
            self.weights[name.replace(".", "_")] = nn.Parameter(
                read_f32(directory / spec["file"], tuple(reversed(shape))))

    def w(self, name):
        return self.weights[name.replace(".", "_")]

    def norm(self, x, name):
        return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-5) * self.w(name)

    @staticmethod
    def rope(x):
        theta = torch.arange(x.shape[-2], device=x.device, dtype=x.dtype)[:, None]
        theta = theta * (10000.0 ** (-torch.arange(0, 64, 2, device=x.device,
                                                  dtype=x.dtype) / 64))
        a, b = x.chunk(2, dim=-1)
        return torch.cat((a * theta.cos() - b * theta.sin(),
                          b * theta.cos() + a * theta.sin()), dim=-1)

    def forward(self, x, n):
        if x.ndim != 3 or x.shape[-1] != 1024 or not 1 <= n <= 12:
            raise ValueError("Expected [batch, frames, 1024] input and 1 <= n <= 12")
        batch, frames, _ = x.shape
        if not n <= frames <= 72 + n:
            raise ValueError("Input must contain the new frames and at most 72 history frames")
        positions = torch.arange(frames, device=x.device)
        delta = positions[:, None] - positions[None, :]
        mask = (delta >= 0) & (delta < 72)
        h = F.linear(x, self.w("codec.dtf.in_proj.weight"), self.w("codec.dtf.in_proj.bias"))
        for i in range(4):
            p = f"codec.dtf.blk.{i}."
            cur = self.norm(h, p + "attn_norm.weight")
            q, k, v = [F.linear(cur, self.w(p + f"attn_{kind}.weight"))
                       .reshape(batch, frames, 16, 64).transpose(1, 2)
                       for kind in ("q", "k", "v")]
            q, k = self.rope(q), self.rope(k)
            scores = (q @ k.transpose(-2, -1)) * 0.125
            a = scores.masked_fill(~mask, float("-inf")).softmax(-1) @ v
            a = a.transpose(1, 2).reshape(batch, frames, 1024)
            h = h + F.linear(a, self.w(p + "attn_output.weight")) * self.w(p + "attn_scale")
            cur = self.norm(h, p + "ffn_norm.weight")
            cur = F.silu(F.linear(cur, self.w(p + "ffn_gate.weight"))) * F.linear(
                cur, self.w(p + "ffn_up.weight"))
            h = h + F.linear(cur, self.w(p + "ffn_down.weight")) * self.w(p + "ffn_scale")
        h = self.norm(h[:, -n:], "codec.dtf.norm.weight")
        h = F.linear(h, self.w("codec.dtf.out_proj.weight"), self.w("codec.dtf.out_proj.bias"))
        return F.linear(F.gelu(h, approximate="none"), self.w("student.output.weight"))

    def export(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        entries = []
        for name, shape in expected_shapes(head_dim=self.head_dim).items():
            filename = name + ".f32"
            self.w(name).detach().cpu().numpy().astype("<f4").tofile(directory / filename)
            entries.append({"name": name, "shape": shape, "file": filename})
        (directory / "weights.json").write_text(json.dumps(entries, indent=2) + "\n")


def widen_head(source_dir, out_dir, head_dim=2048, seed=42):
    torch.manual_seed(seed)
    source_dir, out_dir = Path(source_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    source_entries = json.loads((source_dir / "weights.json").read_text())
    if isinstance(source_entries, dict):
        source_entries = source_entries["weights"]
    source_specs = {e["name"]: e for e in source_entries}

    shapes = expected_shapes(head_dim=head_dim)
    entries = []
    for name, shape in shapes.items():
        filename = name + ".f32"
        out_path = out_dir / filename
        if name == "codec.dtf.out_proj.bias":
            old_bias = read_f32(source_dir / source_specs[name]["file"], (1024,))
            tensor = torch.zeros(tuple(reversed(shape)))
            tensor[:1024] = old_bias
            tensor.numpy().astype("<f4").tofile(out_path)
        elif name == "codec.dtf.out_proj.weight":
            old_weight = read_f32(source_dir / source_specs[name]["file"], (1024, 512))
            tensor = torch.randn(tuple(reversed(shape))) * 0.02
            tensor[:1024, :] = old_weight
            tensor.numpy().astype("<f4").tofile(out_path)
        elif name == "student.output.weight":
            tensor = torch.randn(tuple(reversed(shape))) * 0.01
            tensor.numpy().astype("<f4").tofile(out_path)
        else:
            src_file = source_dir / source_specs[name]["file"]
            out_path.write_bytes(src_file.read_bytes())
        entries.append({"name": name, "shape": shape, "file": filename})
    (out_dir / "weights.json").write_text(json.dumps(entries, indent=2) + "\n")
    print(json.dumps({"widened_head": str(out_dir), "head_dim": head_dim, "total_weights": len(entries)}))


def check(args):
    torch.set_num_threads(args.threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    source = args.directory.resolve()
    model = Student(source).to(args.device)
    records = json.loads((source / "manifest.json").read_text())
    if isinstance(records, dict):
        records = records["records"]
    if not records:
        raise ValueError("No parity fixtures")
    errors = []
    for record in records:
        x = read_f32(source / record["input"], (1, record["frames"], 1024)).to(args.device)
        ref = read_f32(source / record["reference"], (1, record["n"], 1920))
        with torch.no_grad():
            actual = model(x, record["n"]).cpu()
        torch.testing.assert_close(actual, ref, rtol=1e-3, atol=1e-4)
        errors.append({"id": record.get("id", record["input"]),
                       "max_abs_error": (actual - ref).abs().max().item()})
    record = next((r for r in records if r.get("target")), None)
    if record is None:
        raise ValueError("One real target fixture is required for the gradient check")
    x = read_f32(source / record["input"], (1, record["frames"], 1024)).to(args.device)
    target = read_f32(source / record["target"], (1, record["n"], 1920)).to(args.device)
    before = model.w("student.output.weight").detach().clone()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
    prediction = model(x, record["n"])
    loss = F.mse_loss(prediction, target)
    loss.backward()
    for name, parameter in model.named_parameters():
        if parameter.grad is None or not torch.isfinite(parameter.grad).all():
            raise AssertionError(f"Missing or non-finite gradient: {name}")
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
    optimizer.step()
    update = (model.w("student.output.weight") - before).abs().max().item()
    if not update > 0:
        raise AssertionError("Diagnostic optimizer step did not change output weights")
    print(json.dumps({"parameters": sum(p.numel() for p in model.parameters()), "head_dim": model.head_dim}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    check_p = subparsers.add_parser("check")
    check_p.add_argument("directory", type=Path)
    check_p.add_argument("--export", type=Path)
    check_p.add_argument("--device", default="cpu")
    check_p.add_argument("--threads", type=int, default=4)

    widen_p = subparsers.add_parser("widen")
    widen_p.add_argument("source", type=Path)
    widen_p.add_argument("out", type=Path)
    widen_p.add_argument("--head-dim", type=int, default=2048)
    widen_p.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    if args.command == "check":
        check(args)
    elif args.command == "widen":
        widen_head(args.source, args.out, args.head_dim, args.seed)
