"""Multi-Scale STFT Discriminator (MS-STFT-D) based on DAC / T-Mimi.

Operates over multi-resolution 2D complex STFT representations to evaluate
spectral envelope, harmonics, and time-frequency structure.
"""
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch.nn.utils.parametrizations import weight_norm
except ImportError:
    from torch.nn.utils import weight_norm


class STFTDiscriminator(nn.Module):
    def __init__(self, n_fft=1024, hop_length=256, win_length=1024):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.register_buffer("window", torch.hann_window(win_length))

        self.convs = nn.ModuleList([
            weight_norm(nn.Conv2d(2, 32, kernel_size=(3, 9), padding=(1, 4))),
            weight_norm(nn.Conv2d(32, 32, kernel_size=(3, 9), stride=(1, 2), padding=(1, 4))),
            weight_norm(nn.Conv2d(32, 32, kernel_size=(3, 9), stride=(1, 2), padding=(1, 4))),
            weight_norm(nn.Conv2d(32, 32, kernel_size=(3, 9), stride=(1, 2), padding=(1, 4))),
            weight_norm(nn.Conv2d(32, 32, kernel_size=(3, 3), padding=(1, 1))),
        ])
        self.final_conv = weight_norm(nn.Conv2d(32, 1, kernel_size=(3, 3), padding=(1, 1)))

    def forward(self, x):
        # x: [B, T]
        stft = torch.stft(
            x, self.n_fft, self.hop_length, self.win_length,
            self.window, center=True, pad_mode="reflect", return_complex=True
        )
        # stft: [B, F, T_spec] -> [B, 2, F, T_spec]
        feat = torch.stack([stft.real, stft.imag], dim=1)

        fmaps = []
        for conv in self.convs:
            feat = F.leaky_relu(conv(feat), 0.2)
            fmaps.append(feat)
        out = self.final_conv(feat)
        fmaps.append(out)
        return out, fmaps


class MultiScaleSTFTDiscriminator(nn.Module):
    def __init__(self, filters=(2048, 1024, 512, 256, 128)):
        super().__init__()
        self.discriminators = nn.ModuleList([
            STFTDiscriminator(n_fft=f, hop_length=f // 4, win_length=f)
            for f in filters
        ])

    def forward(self, x):
        scores, fmaps = [], []
        for disc in self.discriminators:
            score, fmap = disc(x)
            scores.append(score)
            fmaps.append(fmap)
        return scores, fmaps


def self_check():
    disc = MultiScaleSTFTDiscriminator()
    x = torch.randn(2, 24000, requires_grad=True)
    scores, fmaps = disc(x)
    assert len(scores) == 5
    assert len(fmaps) == 5
    total_loss = sum(s.mean() for s in scores)
    total_loss.backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    print(f"MS-STFT-D self-check: PASSED (params: {sum(p.numel() for p in disc.parameters()):,})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        self_check()
