"""Bounded FP32 student training supporting Waveform MSE, Spectral, and T-Mimi composite objectives."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random
import time
import wave

import numpy as np
import torch
from torch.nn import functional as F

from student import Student, read_f32
from discriminator import MultiScaleSTFTDiscriminator
from loss import TMimiCompositeLoss, discriminator_loss, MultiScaleMelLoss


def loss_parts(prediction, target):
    if prediction.shape != target.shape or prediction.ndim != 2:
        raise ValueError("Loss expects matching [batch, samples] waveforms")
    convergence, log_magnitude = [], []
    for size in (256, 512, 1024, 2048):
        window = torch.hann_window(size, device=prediction.device)
        def spectrum(x):
            return torch.stft(x, size, hop_length=size // 4, window=window,
                              center=True, pad_mode="constant", return_complex=True).abs()
        p, t = spectrum(prediction), spectrum(target)
        convergence.append((p - t).flatten(1).norm(dim=1).div(
            t.flatten(1).norm(dim=1).clamp_min(1e-5)).mean())
        log_magnitude.append((p.clamp_min(1e-5).log() - t.clamp_min(1e-5).log()).abs().mean())
    return torch.stack((F.l1_loss(prediction, target),
                        torch.stack(convergence).mean(), torch.stack(log_magnitude).mean()))


def band_ratios(prediction, target, sr=24000):
    p_fft = np.abs(np.fft.rfft(prediction))
    t_fft = np.abs(np.fft.rfft(target))
    freqs = np.fft.rfftfreq(len(prediction), 1.0 / sr)
    ratios = {}
    for low, high in [(0, 500), (500, 2000), (2000, 4000), (4000, 8000)]:
        mask = (freqs >= low) & (freqs < high)
        p_eng = np.mean(p_fft[mask] ** 2)
        t_eng = np.mean(t_fft[mask] ** 2)
        ratios[f"ratio_{low}_{high}"] = float(np.sqrt(p_eng / (t_eng + 1e-12)))
    return ratios


def objective(prediction, target, parts, kind, wave_weight):
    if kind == "mse":
        return F.mse_loss(prediction, target)
    return wave_weight * parts[0] + parts[1:].sum()


def initialize(args):
    model = Student(args.weights)
    with torch.no_grad():
        model.w("student.output.weight").normal_(std=0.01)
    model.export(args.out)
    records = json.loads((args.weights / "manifest.json").read_text())
    if isinstance(records, dict):
        records = records["records"]
    updated = []
    for i, record in enumerate(records):
        source = (args.weights / record["input"]).resolve()
        x = read_f32(source, (1, record["frames"], 1024))
        with torch.no_grad():
            output = model(x, record["n"])
        filename = f"{i:04d}.torch.f32"
        output.numpy().astype("<f4").tofile(args.out / filename)
        entry = {**record, "input": str(source), "reference": filename}
        if record.get("target"):
            entry["target"] = str((args.weights / record["target"]).resolve())
        updated.append(entry)
    (args.out / "manifest.json").write_text(json.dumps(updated, indent=2) + "\n")
    (args.out / "initialization.json").write_text(json.dumps({
        "seed": args.seed, "head_normal_std": 0.01,
        "source_weights": str(args.weights.resolve()), "fixtures": len(updated)}, indent=2) + "\n")
    print(json.dumps({"initialized": str(args.out), "fixtures": len(updated)}), flush=True)


def write_wav(path, samples):
    with wave.open(str(path), "wb") as wav:
        wav.setparams((1, 2, 24000, 0, "NONE", "not compressed"))
        wav.writeframes((np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes())


class SilenceBank:
    """Bank of genuine silence latents extracted from in-distribution pause/tail frames."""
    def __init__(self, latents):
        self.latents = latents  # [N, 1024]

    def sample(self, k, device):
        indices = torch.randint(0, len(self.latents), (k,))
        return self.latents[indices].to(device)

    def augment(self, x, target, n):
        # x: [frames, 1024], target: [n * 1920]
        mode = random.choice(["tail", "lead", "pure"])
        frames = x.shape[0]
        n_hist = frames - n
        if mode == "tail":
            k = random.randint(1, max(1, min(n, 6)))
            x[-k:] = self.sample(k, x.device)
            target[-k * 1920:] = 0.0
        elif mode == "lead" and n_hist > 0:
            k = random.randint(1, min(n_hist, 12))
            x[:k] = self.sample(k, x.device)
        elif mode == "pure":
            x[:] = self.sample(frames, x.device)
            target[:] = 0.0


def train(args):
    started = time.monotonic()
    raw_manifest = args.dataset.read_bytes()
    records = json.loads(raw_manifest)
    if isinstance(records, dict):
        records = records["records"]
    records = [r for r in records if r["split"] in ("train", "validation")]
    ids = {split: {r["utterance_id"] for r in records if r["split"] == split}
           for split in ("train", "validation")}
    if not all(ids.values()) or ids["train"] & ids["validation"]:
        raise ValueError("Require nonempty, disjoint train and validation utterances")
    cache, groups, buckets = [], defaultdict(list), defaultdict(list)
    silence_latents = []
    for i, r in enumerate(records):
        name = r["utterance_id"]
        if not name or not all(c.isalnum() or c in "-_" for c in name):
            raise ValueError("Unsafe utterance ID")
        if not 1 <= r["n"] <= 12 or r["frames"] != min(r["start"], 72) + r["n"]:
            raise ValueError("Invalid production window dimensions")
        target_path = args.dataset.parent / r["target"]
        if r.get("target_sha256") and hashlib.sha256(target_path.read_bytes()).hexdigest() != r["target_sha256"]:
            raise ValueError("Target checksum mismatch")
        x_tensor = read_f32(args.dataset.parent / r["input"], (r["frames"], 1024))
        target_tensor = read_f32(target_path, (r["n"] * 1920,))
        cache.append((x_tensor, target_tensor))
        groups[name].append(i)
        if r["split"] == "train":
            buckets[r["frames"], r["n"]].append(i)
            for j in range(r["n"]):
                chunk = target_tensor[j * 1920 : (j + 1) * 1920]
                if torch.sqrt(torch.mean(chunk ** 2)) < 0.002:
                    silence_latents.append(x_tensor[r["frames"] - r["n"] + j].clone())
    silence_bank = SilenceBank(torch.stack(silence_latents)) if silence_latents else None
    if silence_bank:
        print(json.dumps({"silence_bank_frames": len(silence_latents)}), flush=True)
    for indices in groups.values():
        indices.sort(key=lambda i: records[i]["start"])
        start = 0
        for i in indices:
            if records[i]["start"] != start:
                raise ValueError("Utterance windows must be contiguous")
            start += records[i]["n"]
    train_indices = [i for i, r in enumerate(records) if r["split"] == "train"]
    model = Student(args.weights).to(args.device)
    if args.optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=tuple(args.betas), weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=tuple(args.betas), weight_decay=args.weight_decay)

    discriminator, optimizer_d, composite_loss = None, None, None
    mel_eval = MultiScaleMelLoss(sample_rate=24000).to(args.device)
    if args.objective == "tmimi":
        discriminator = MultiScaleSTFTDiscriminator().to(args.device)
        optimizer_d = torch.optim.AdamW(discriminator.parameters(), lr=args.lr_d, betas=(0.8, 0.99), weight_decay=0.01)
        composite_loss = TMimiCompositeLoss(sample_rate=24000).to(args.device)

    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(dataset_sha256=hashlib.sha256(raw_manifest).hexdigest(),
                  torch_version=torch.__version__, hip=torch.version.hip,
                  cuda=torch.version.cuda, parameters=sum(p.numel() for p in model.parameters()),
                  discriminator_parameters=sum(p.numel() for p in discriminator.parameters()) if discriminator else 0,
                  precision="FP32; TF32 disabled", loss_epsilon=1e-5,
                  stft_padding="center=True, constant zero; identical prediction and target",
                  source_sha256={p: hashlib.sha256(Path(__file__).with_name(p).read_bytes()).hexdigest()
                                 for p in ("student.py", "train.py", "loss.py", "discriminator.py") if Path(__file__).with_name(p).exists()})
    (args.out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    step, best, last_save = 0, float("inf"), time.monotonic()

    def emit(event):
        event.update(step=step, elapsed_seconds=time.monotonic() - started)
        if args.device.startswith("cuda"):
            event["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
            event["peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
        with (args.out / "metrics.jsonl").open("a") as f:
            f.write(json.dumps(event) + "\n")
        print(json.dumps(event), flush=True)

    def checkpoint(name):
        nonlocal last_save
        state = {"student": model.state_dict(), "optimizer": optimizer.state_dict(),
                 "step": step, "best_validation_loss": best, "config": config,
                 "rng_python": random.getstate(), "rng_numpy": np.random.get_state(),
                 "rng_torch": torch.get_rng_state(),
                 "rng_cuda": torch.cuda.get_rng_state_all() if args.device.startswith("cuda") else []}
        if discriminator:
            state["discriminator"] = discriminator.state_dict()
            state["optimizer_d"] = optimizer_d.state_dict()
        temporary = args.out / (name + ".tmp")
        torch.save(state, temporary)
        temporary.replace(args.out / (name + ".pt"))
        last_save = time.monotonic()

    def evaluate(label):
        nonlocal best
        model.eval()
        destination = args.out / f"audio-{label}-{step:05d}"
        destination.mkdir(exist_ok=True)
        sums = defaultdict(lambda: np.zeros(3))
        mel_sums = defaultdict(float)
        counts = defaultdict(int)
        utterances = {}
        with torch.no_grad():
            for name, indices in groups.items():
                predictions, targets = [], []
                for i in indices:
                    x, target = cache[i]
                    prediction = model(x[None].to(args.device), records[i]["n"]).flatten(1)
                    target = target[None].to(args.device)
                    parts = loss_parts(prediction, target)
                    if not torch.isfinite(parts).all():
                        raise FloatingPointError("Nonfinite evaluation loss")
                    split = records[i]["split"]
                    sums[split] += parts.cpu().numpy() * target.numel()
                    counts[split] += target.numel()
                    predictions.append(prediction.cpu().numpy().ravel())
                    targets.append(target.cpu().numpy().ravel())
                p, t = np.concatenate(predictions), np.concatenate(targets)
                p_t = torch.from_numpy(p)[None].to(args.device)
                t_t = torch.from_numpy(t)[None].to(args.device)
                parts = loss_parts(p_t, t_t).cpu().tolist()
                mel_l1 = mel_eval(p_t, t_t).item() if mel_eval else 0.0
                mel_sums[split] += mel_l1 * len(p)

                boundaries = np.cumsum([len(x) for x in predictions])[:-1]
                seam_error = np.abs((p[boundaries] - p[boundaries - 1]) -
                                    (t[boundaries] - t[boundaries - 1]))
                ratios = band_ratios(p, t)
                utterances[name] = dict(split=split, samples=len(p), wave_l1=parts[0],
                    wave_mse=float(np.mean((p - t) ** 2)),
                    mel_l1=mel_l1,
                    spectral_convergence=parts[1], log_magnitude=parts[2],
                    clipping_fraction=float(np.mean(np.abs(p) > 1)),
                    target_clipping_fraction=float(np.mean(np.abs(t) > 1)),
                    boundary_derivative_mae=float(seam_error.mean()) if len(seam_error) else 0.0,
                    **ratios)
                write_wav(destination / (name + "-student.wav"), p)
                write_wav(destination / (name + "-teacher.wav"), t)
        scores = {s: (sums[s] / counts[s]).tolist() for s in sums}
        mel_scores = {s: (mel_sums[s] / counts[s]) for s in mel_sums}
        event = {"event": "evaluation", "label": label, "parts_by_split": scores,
                 "mel_by_split": mel_scores,
                 "utterances": utterances, "audio": str(destination)}
        emit(event)
        (destination / "metrics.json").write_text(json.dumps(event, indent=2) + "\n")
        validation = [v for v in utterances.values() if v["split"] == "validation"]
        if args.objective == "tmimi":
            score = sum(v["mel_l1"] * v["samples"] for v in validation) / sum(v["samples"] for v in validation)
        elif args.objective == "mse":
            score = sum(v["wave_mse"] * v["samples"] for v in validation) / sum(v["samples"] for v in validation)
        elif args.objective == "combo":
            val_mse = sum(v["wave_mse"] * v["samples"] for v in validation) / sum(v["samples"] for v in validation)
            val_mel = sum(v["mel_l1"] * v["samples"] for v in validation) / sum(v["samples"] for v in validation)
            val_stft = sum((v["spectral_convergence"] + v["log_magnitude"]) * v["samples"] for v in validation) / sum(v["samples"] for v in validation)
            score = val_mse + args.mel_weight * val_mel + args.stft_weight * val_stft
        else:
            score = args.wave_weight * scores["validation"][0] + sum(scores["validation"][1:])

        if score < best:
            best = score
            checkpoint("best")
            model.export(args.out / "best-weights")
        checkpoint("latest")
        model.train()
        return event

    initial = evaluate("initial")
    recent_parts, recent_losses = [], []
    while step < args.steps and time.monotonic() - started < args.minutes * 60:
        anchor = random.choice(train_indices)
        r = records[anchor]
        indices = [anchor] + random.choices(buckets[r["frames"], r["n"]], k=args.batch - 1)
        x = torch.stack([cache[i][0] for i in indices]).to(args.device)
        target = torch.stack([cache[i][1] for i in indices]).to(args.device)

        # Silence augmentation: 10% probability edge-pad with real silence latents
        if args.silence_aug and silence_bank is not None and random.random() < 0.10:
            idx = random.randint(0, args.batch - 1)
            silence_bank.augment(x[idx], target[idx], r["n"])

        if args.objective == "tmimi":
            stage = 2 if (0 < args.stage2_steps < args.steps and step >= args.steps - args.stage2_steps) else 1

            # 1. Generator forward pass (once per step)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(x, r["n"]).flatten(1)

            # 2. Train Discriminator (Stage 1 only)
            loss_d_val = 0.0
            if stage == 1:
                optimizer_d.zero_grad(set_to_none=True)
                real_scores, _ = discriminator(target)
                fake_scores, _ = discriminator(prediction.detach())
                loss_d = discriminator_loss(real_scores, fake_scores)
                if not torch.isfinite(loss_d):
                    raise FloatingPointError("Nonfinite discriminator loss")
                loss_d.backward()
                torch.nn.utils.clip_grad_norm_(discriminator.parameters(), 1.0)
                optimizer_d.step()
                loss_d_val = loss_d.item()

            # 3. Train Generator (freeze discriminator params during G backward)
            for p in discriminator.parameters():
                p.requires_grad = False
            with torch.no_grad():
                _, fmaps_real = discriminator(target)
            fake_scores, fmaps_fake = discriminator(prediction)
            loss_g, comp_parts = composite_loss(
                prediction, target, None, fake_scores, fmaps_real, fmaps_fake, stage=stage
            )
            if not torch.isfinite(loss_g):
                raise FloatingPointError("Nonfinite generator loss")
            loss_g.backward()
            for p in discriminator.parameters():
                p.requires_grad = True
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            if any(p.grad is None for p in model.parameters()):
                raise RuntimeError("Missing student gradient")
            optimizer.step()
            step += 1
            recent_losses.append((comp_parts["total"], loss_d_val, comp_parts["mel"], comp_parts["feat"], comp_parts["adv"]))
            if step == 1 or step % 25 == 0:
                avgs = np.mean(recent_losses, axis=0)
                emit({"event": "training", "objective": "tmimi", "stage": stage,
                      "loss_g": avgs[0], "loss_d": avgs[1], "mel_l1": avgs[2],
                      "feat_loss": avgs[3], "adv_loss": avgs[4],
                      "gradient_norm": norm.item()})
                recent_losses.clear()

        else:
            optimizer.zero_grad(set_to_none=True)
            prediction = model(x, r["n"]).flatten(1)
            parts = loss_parts(prediction, target)
            if not torch.isfinite(parts).all():
                raise FloatingPointError("Nonfinite training loss")
            if args.objective == "combo":
                loss_mse = F.mse_loss(prediction, target)
                loss_mel = mel_eval(prediction, target)
                loss_stft = parts[1] + parts[2]
                loss = loss_mse + args.mel_weight * loss_mel + args.stft_weight * loss_stft
            else:
                loss = objective(prediction, target, parts, args.objective, args.wave_weight)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite training objective")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            if any(p.grad is None for p in model.parameters()):
                raise RuntimeError("Missing student gradient")
            optimizer.step()
            step += 1
            recent_parts.append(parts.detach().cpu().numpy())
            if step == 1 or step % 25 == 0:
                emit({"event": "training", "loss_parts": np.mean(recent_parts, axis=0).tolist(),
                      "gradient_norm": norm.item(), "objective": args.objective,
                      "last_batch_objective": loss.item()})
                recent_parts.clear()

        if step % args.eval_every == 0:
            evaluate("interval")
        elif time.monotonic() - last_save >= 300:
            checkpoint("latest")

    final = evaluate("final")
    model.export(args.out / "final-weights")
    (args.out / "summary.json").write_text(json.dumps({"steps": step,
        "elapsed_seconds": time.monotonic() - started, "best_validation_loss": best,
        "stop_reason": "steps" if step == args.steps else "wall_time",
        "initial": initial, "final": final}, indent=2) + "\n")


def self_check():
    for n in (1, 12):
        x = torch.linspace(-0.1, 0.1, n * 1920)[None].requires_grad_()
        assert x.flatten(1).shape == (1, n * 1920)
        assert torch.equal(loss_parts(x, x.detach()), torch.zeros(3))
        loss_parts(x, x.detach() * 0.5).sum().backward()
        assert torch.isfinite(x.grad).all()
    p = torch.tensor([[1.0, 3.0]], requires_grad=True)
    target = torch.zeros_like(p)
    mse = objective(p, target, None, "mse", 10)
    assert mse.item() == 5.0
    mse.backward()
    torch.testing.assert_close(p.grad, p.detach())
    parts = torch.tensor([1., 2., 3.])
    assert objective(p, target, parts, "spectral", 10).item() == 15
    fake_latents = torch.randn(10, 1024)
    bank = SilenceBank(fake_latents)
    x = torch.ones(20, 1024)
    t = torch.ones(12 * 1920)
    bank.augment(x, t, 12)
    assert x.shape == (20, 1024) and t.shape == (12 * 1920,)
    assert torch.isfinite(x).all() and torch.isfinite(t).all()
    print("Loss identity, short/long lengths, finite backward, objective selection and silence bank: passed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--initialize-only", action="store_true")
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--minutes", type=float, default=30)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--objective", choices=("spectral", "mse", "tmimi", "combo"), default="combo")
    parser.add_argument("--mel-weight", type=float, default=0.003)
    parser.add_argument("--stft-weight", type=float, default=0.002)
    parser.add_argument("--optimizer", choices=("adamw", "adam"), default="adamw")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--betas", nargs=2, type=float, default=[0.8, 0.99])
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--lr-d", type=float, default=2e-4)
    parser.add_argument("--stage2-steps", type=int, default=2000)
    parser.add_argument("--silence-aug", action="store_true", default=False,
                        help="Enable edge-padding silence augmentation (opt-in; off by default for A/B runs)")
    parser.add_argument("--wave-weight", type=float, default=1.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.self_check:
        self_check()
    else:
        if not args.weights or not args.out or (not args.initialize_only and not args.dataset):
            parser.error("--weights, --out and (for training) --dataset are required")
        if args.eval_every < 1 or args.steps < 1 or args.minutes <= 0 or args.batch < 1 or not np.isfinite(args.wave_weight) or args.wave_weight <= 0:
            parser.error("Training bounds and batch must be positive")
        args.out.mkdir(parents=True, exist_ok=False)
        initialize(args) if args.initialize_only else train(args)
