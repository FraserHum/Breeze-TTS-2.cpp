"""T-Mimi Composite Loss Formulation:
1. Multi-scale Mel-spectrogram L1 loss (weight 2.0)
2. Least-Squares GAN loss (weight 4.0)
3. Discriminator Feature Matching loss (weight 4.0)
4. Waveform L1 loss (weight 0.1)
"""
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F


def mel_filterbank(sr, n_fft, n_mels, fmin=0.0, fmax=None):
    if fmax is None:
        fmax = sr / 2.0

    def hz_to_mel(hz):
        return 2595.0 * torch.log10(1.0 + hz / 700.0)

    def mel_to_hz(mel):
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    mel_min = hz_to_mel(torch.tensor(fmin, dtype=torch.float32))
    mel_max = hz_to_mel(torch.tensor(fmax, dtype=torch.float32))
    mel_pts = torch.linspace(mel_min, mel_max, n_mels + 2)
    hz_pts = mel_to_hz(mel_pts)
    bin_pts = torch.floor((n_fft + 1) * hz_pts / sr).long()

    fb = torch.zeros(n_mels, n_fft // 2 + 1)
    for m in range(1, n_mels + 1):
        f_m_minus = bin_pts[m - 1].item()
        f_m = bin_pts[m].item()
        f_m_plus = bin_pts[m + 1].item()
        for k in range(f_m_minus, f_m):
            if f_m > f_m_minus:
                fb[m - 1, k] = (k - f_m_minus) / (f_m - f_m_minus)
        for k in range(f_m, f_m_plus):
            if f_m_plus > f_m:
                fb[m - 1, k] = (f_m_plus - k) / (f_m_plus - f_m)
    return fb


class MultiScaleMelLoss(nn.Module):
    def __init__(self, sample_rate=24000, n_mels=80, filters=(2048, 1024, 512, 256, 128)):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.filters = filters

        self.windows = nn.ParameterList()
        self.filterbanks = nn.ParameterList()
        for n_fft in filters:
            win = nn.Parameter(torch.hann_window(n_fft), requires_grad=False)
            fb = nn.Parameter(mel_filterbank(sample_rate, n_fft, n_mels), requires_grad=False)
            self.windows.append(win)
            self.filterbanks.append(fb)

    def mel_spectrogram(self, x, idx):
        n_fft = self.filters[idx]
        hop = n_fft // 4
        win = self.windows[idx]
        fb = self.filterbanks[idx]

        stft = torch.stft(
            x, n_fft, hop, n_fft, win,
            center=True, pad_mode="reflect", return_complex=True
        )
        mag = torch.abs(stft)  # [B, F, T]
        mel = torch.matmul(fb, mag)  # [B, n_mels, T]
        return torch.log(mel.clamp_min(1e-5))

    def forward(self, pred, target):
        loss = 0.0
        for i in range(len(self.filters)):
            pred_mel = self.mel_spectrogram(pred, i)
            target_mel = self.mel_spectrogram(target, i)
            loss = loss + F.l1_loss(pred_mel, target_mel)
        return loss / len(self.filters)


def discriminator_loss(disc_real_scores, disc_fake_scores):
    """Least-squares GAN loss for Discriminator."""
    loss = 0.0
    for real_score, fake_score in zip(disc_real_scores, disc_fake_scores):
        real_loss = torch.mean((real_score - 1.0) ** 2)
        fake_loss = torch.mean(fake_score ** 2)
        loss = loss + (real_loss + fake_loss) * 0.5
    return loss / len(disc_real_scores)


def generator_loss(disc_fake_scores):
    """Least-squares GAN loss for Generator."""
    loss = 0.0
    for fake_score in disc_fake_scores:
        loss = loss + torch.mean((fake_score - 1.0) ** 2)
    return loss / len(disc_fake_scores)


def feature_matching_loss(fmaps_real, fmaps_fake):
    """L1 feature matching across intermediate discriminator activations."""
    loss = 0.0
    total_layers = 0
    for scale_real, scale_fake in zip(fmaps_real, fmaps_fake):
        for r, f in zip(scale_real, scale_fake):
            loss = loss + F.l1_loss(f, r.detach())
            total_layers += 1
    return loss / max(1, total_layers)


class TMimiCompositeLoss(nn.Module):
    def __init__(self, sample_rate=24000):
        super().__init__()
        self.mel_loss = MultiScaleMelLoss(sample_rate=sample_rate)

    def forward(self, pred, target, disc_real_scores=None, disc_fake_scores=None,
                fmaps_real=None, fmaps_fake=None, stage=1):
        l_mel = self.mel_loss(pred, target)
        l_wave = F.l1_loss(pred, target)

        if stage == 2:
            # Stage 2: Fine-tune exclusively on feature matching
            l_feat = feature_matching_loss(fmaps_real, fmaps_fake)
            total = 4.0 * l_feat
            return total, {"mel": l_mel.item(), "wave_l1": l_wave.item(),
                           "adv": 0.0, "feat": l_feat.item(), "total": total.item()}

        # Stage 1: Full composite loss
        l_adv = generator_loss(disc_fake_scores) if disc_fake_scores is not None else torch.tensor(0.0, device=pred.device)
        l_feat = feature_matching_loss(fmaps_real, fmaps_fake) if fmaps_real is not None else torch.tensor(0.0, device=pred.device)

        total = 2.0 * l_mel + 4.0 * l_adv + 4.0 * l_feat + 0.1 * l_wave
        return total, {
            "mel": l_mel.item(),
            "wave_l1": l_wave.item(),
            "adv": l_adv.item() if isinstance(l_adv, torch.Tensor) else l_adv,
            "feat": l_feat.item() if isinstance(l_feat, torch.Tensor) else l_feat,
            "total": total.item()
        }


def self_check():
    sample_rate = 24000
    mel_loss = MultiScaleMelLoss(sample_rate=sample_rate)
    x = torch.randn(2, 24000, requires_grad=True)
    y = torch.randn(2, 24000)

    # Self-identity check
    l_ident = mel_loss(x, x.detach())
    assert l_ident.item() < 1e-4, f"Identity loss should be ~0, got {l_ident.item()}"

    # Difference and backward check
    l_diff = mel_loss(x, y)
    assert l_diff.item() > 0.1
    l_diff.backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()

    print("MultiScaleMelLoss self-check: PASSED")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        self_check()
