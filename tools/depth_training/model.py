import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., dim]
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight


def compute_freq_factors(
    head_dim: int = 128,
    rope_theta: float = 500000.0,
    rope_factor: float = 32.0,
    rope_low_freq: float = 0.001953125,
    rope_high_freq: float = 0.0078125,
    rope_orig_ctx: int = 16,
) -> np.ndarray:
    half = head_dim // 2
    ff = np.ones(half, dtype=np.float32)
    pi = np.pi
    low_wl = rope_orig_ctx / rope_low_freq
    high_wl = rope_orig_ctx / rope_high_freq
    for i in range(half):
        freq = rope_theta ** (-2.0 * i / head_dim)
        wavelen = 2.0 * pi / freq
        if wavelen > low_wl:
            ff[i] = rope_factor
        elif wavelen < high_wl:
            ff[i] = 1.0
        else:
            smooth = (rope_orig_ctx / wavelen - rope_low_freq) / (rope_high_freq - rope_low_freq)
            ff[i] = 1.0 / ((1.0 - smooth) / rope_factor + smooth)
    return ff


class Llama3RoPE(nn.Module):
    def __init__(
        self,
        head_dim: int = 128,
        max_seq_len: int = 16,
        rope_theta: float = 500000.0,
        rope_factor: float = 32.0,
        rope_low_freq: float = 0.001953125,
        rope_high_freq: float = 0.0078125,
        rope_orig_ctx: int = 16,
    ):
        super().__init__()
        self.head_dim = head_dim
        half = head_dim // 2
        ff = compute_freq_factors(
            head_dim, rope_theta, rope_factor, rope_low_freq, rope_high_freq, rope_orig_ctx
        )
        freqs = (rope_theta ** (-2.0 * np.arange(half, dtype=np.float32) / head_dim)) / ff
        positions = np.arange(max_seq_len, dtype=np.float32)
        angles = positions[:, None] * freqs[None, :]  # [max_seq_len, half]
        cos = np.cos(angles).astype(np.float32)
        sin = np.sin(angles).astype(np.float32)
        self.register_buffer("cos", torch.from_numpy(cos), persistent=False)
        self.register_buffer("sin", torch.from_numpy(sin), persistent=False)

    def forward(self, x: torch.Tensor, pos_offset: int = 0) -> torch.Tensor:
        # x: [B, n_heads, seq_len, head_dim]
        seq_len = x.shape[2]
        half = self.head_dim // 2
        cos = self.cos[pos_offset : pos_offset + seq_len, :]  # [seq_len, half]
        sin = self.sin[pos_offset : pos_offset + seq_len, :]  # [seq_len, half]
        cos = cos.unsqueeze(0).unsqueeze(1)  # [1, 1, seq_len, half]
        sin = sin.unsqueeze(0).unsqueeze(1)  # [1, 1, seq_len, half]

        x1 = x[..., :half]
        x2 = x[..., half:]
        # GGML NEOX rotation:
        # x1' = x1 * cos - x2 * sin
        # x2' = x1 * sin + x2 * cos
        out1 = x1 * cos - x2 * sin
        out2 = x1 * sin + x2 * cos
        return torch.cat([out1, out2], dim=-1)


class DepthDecoderBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 1024,
        n_head: int = 8,
        n_kv_head: int = 2,
        head_dim: int = 128,
        ffn_dim: int = 8192,
        rms_eps: float = 1e-5,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_head = n_head
        self.n_kv_head = n_kv_head
        self.head_dim = head_dim
        self.num_kv_groups = n_head // n_kv_head
        self.scale = 1.0 / math.sqrt(head_dim)

        self.attn_norm = RMSNorm(hidden_dim, eps=rms_eps)
        self.attn_q = nn.Linear(hidden_dim, n_head * head_dim, bias=False)
        self.attn_k = nn.Linear(hidden_dim, n_kv_head * head_dim, bias=False)
        self.attn_v = nn.Linear(hidden_dim, n_kv_head * head_dim, bias=False)
        self.attn_output = nn.Linear(n_head * head_dim, hidden_dim, bias=False)

        self.ffn_norm = RMSNorm(hidden_dim, eps=rms_eps)
        self.ffn_gate = nn.Linear(hidden_dim, ffn_dim, bias=False)
        self.ffn_up = nn.Linear(hidden_dim, ffn_dim, bias=False)
        self.ffn_down = nn.Linear(ffn_dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor, rope: Llama3RoPE, mask: torch.Tensor = None) -> torch.Tensor:
        # x: [B, seq_len, hidden_dim]
        B, T, D = x.shape
        h = self.attn_norm(x)

        q = self.attn_q(h).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.attn_k(h).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = self.attn_v(h).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)

        q = rope(q)
        k = rope(k)

        if self.num_kv_groups > 1:
            k = k.repeat_interleave(self.num_kv_groups, dim=1)
            v = v.repeat_interleave(self.num_kv_groups, dim=1)

        scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        if mask is not None:
            scores = scores + mask

        attn_probs = F.softmax(scores, dim=-1)
        attn_out = torch.matmul(attn_probs, v)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, -1)
        x = x + self.attn_output(attn_out)

        # FFN (SwiGLU)
        h_ffn = self.ffn_norm(x)
        gate = F.silu(self.ffn_gate(h_ffn))
        up = self.ffn_up(h_ffn)
        down = self.ffn_down(gate * up)
        x = x + down
        return x


class DepthDecoder(nn.Module):
    def __init__(
        self,
        n_layer: int = 9,
        hidden_dim: int = 1024,
        backbone_hidden_dim: int = 2048,
        vocab_size: int = 2051,
        n_steps: int = 15,
        n_head: int = 8,
        n_kv_head: int = 2,
        head_dim: int = 128,
        ffn_dim: int = 8192,
        rms_eps: float = 1e-5,
    ):
        super().__init__()
        self.n_layer = n_layer
        self.n_steps = n_steps
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size

        self.audio_embd = nn.Embedding(16 * vocab_size, backbone_hidden_dim)
        self.in_proj = nn.Linear(backbone_hidden_dim, hidden_dim, bias=False)

        self.rope = Llama3RoPE(head_dim=head_dim, max_seq_len=16)
        self.blocks = nn.ModuleList(
            [
                DepthDecoderBlock(
                    hidden_dim=hidden_dim,
                    n_head=n_head,
                    n_kv_head=n_kv_head,
                    head_dim=head_dim,
                    ffn_dim=ffn_dim,
                    rms_eps=rms_eps,
                )
                for _ in range(n_layer)
            ]
        )

        self.output_norm = RMSNorm(hidden_dim, eps=rms_eps)
        # 15 heads of [hidden_dim -> vocab_size]
        self.codebooks_head = nn.Parameter(torch.empty(n_steps, vocab_size, hidden_dim))

    def forward(
        self,
        cb0: torch.Tensor,
        h0: torch.Tensor,
        teacher_codes: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass for all 15 depth steps in a single batched causal sequence.

        Args:
            cb0: [B] int32/int64 token from backbone (codebook 0)
            h0: [B, 2048] float32 backbone hidden vector
            teacher_codes: [B, 15] int32/int64 teacher codes for codebooks 1..15

        Returns:
            logits: [B, 15, 2051] logits for codebooks 1..15
        """
        B = cb0.shape[0]
        device = cb0.device

        # Sequence construction (T = 16):
        # pos 0: h0 (backbone hidden)
        # pos 1: audio_embd[cb0]
        # pos 2: audio_embd[teacher_codes[:, 0] + 1 * 2051]
        # ...
        # pos 15: audio_embd[teacher_codes[:, 13] + 14 * 2051]

        cb0_emb = self.audio_embd(cb0).unsqueeze(1)  # [B, 1, 2048]
        offsets = (torch.arange(1, 15, device=device) * self.vocab_size).unsqueeze(0)  # [1, 14]
        tail_indices = teacher_codes[:, :14] + offsets  # [B, 14]
        tail_emb = self.audio_embd(tail_indices)  # [B, 14, 2048]

        h0_emb = h0.unsqueeze(1)  # [B, 1, 2048]
        seq_emb = torch.cat([h0_emb, cb0_emb, tail_emb], dim=1)  # [B, 16, 2048]

        x = self.in_proj(seq_emb)  # [B, 16, 1024]

        # Causal mask: [1, 1, 16, 16]
        causal_mask = torch.triu(
            torch.full((16, 16), float("-inf"), device=device), diagonal=1
        ).unsqueeze(0).unsqueeze(0)

        for block in self.blocks:
            x = block(x, self.rope, mask=causal_mask)

        x = self.output_norm(x)  # [B, 16, 1024]

        # Positions 1..15 correspond to step outputs 1..15:
        # Step j (1..15) reads pos j and projects through codebooks_head[j-1]
        step_features = x[:, 1:16, :]  # [B, 15, 1024]

        # Batch-apply the 15 codebook heads:
        # logits[b, j, v] = sum_d step_features[b, j, d] * codebooks_head[j, v, d]
        logits = torch.einsum("bjd,jvd->bjv", step_features, self.codebooks_head)
        return logits
