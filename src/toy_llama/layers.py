"""Core transformer layer components for Toy LLaMA 3.2."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeedForward(nn.Module):
    """SwiGLU feed-forward network.

    Implements the feed-forward layer with SwiGLU activation:
    FFN(x) = W_3(SiLU(W_1(x)) * W_2(x))
    """

    def __init__(self, cfg: "LlamaConfig") -> None:
        super().__init__()
        self.fc1 = nn.Linear(cfg.emb_dim, cfg.hidden_dim, dtype=cfg.dtype, bias=False)
        self.fc2 = nn.Linear(cfg.emb_dim, cfg.hidden_dim, dtype=cfg.dtype, bias=False)
        self.fc3 = nn.Linear(cfg.hidden_dim, cfg.emb_dim, dtype=cfg.dtype, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_fc1 = self.fc1(x)
        x_fc2 = self.fc2(x)
        x = F.silu(x_fc1) * x_fc2
        return self.fc3(x)


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    A simpler alternative to LayerNorm that normalizes by RMS.
    """

    def __init__(self, emb_dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.emb_dim = emb_dim
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(emb_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x = x.to(torch.float32)

        rms = x.pow(2).mean(dim=-1, keepdim=True)
        norm_x = x * torch.rsqrt(rms + self.eps)
        norm_x = norm_x * self.scale

        return norm_x.to(input_dtype)


def compute_rope_params(
    head_dims: int,
    theta_base: float = 10000.0,
    context_length: int = 4096,
    freq_config: Optional[dict] = None,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute rotary position embedding parameters.

    Args:
        head_dims: Dimension of attention heads (must be even).
        theta_base: Base theta for the rotary embedding.
        context_length: Maximum context length.
        freq_config: Optional frequency interpolation config.
        dtype: Output tensor dtype.

    Returns:
        Tuple of (cos, sin) tensors of shape (context_length, head_dims).
    """
    if head_dims % 2 != 0:
        msg = "Embedding dimension must be even"
        raise ValueError(msg)

    # Generate indices for dimensions: 0, 2, 4, ..., head_dims-2
    indices = torch.arange(0, head_dims, 2, dtype=dtype)
    # Take first half: effectively gives 2i/d formula
    indices = indices[: head_dims // 2]
    indices = indices / head_dims

    inv_freq = 1.0 / (theta_base**indices)

    if freq_config is not None:
        low_freq_wavelen = (
            freq_config["original_context_length"] / freq_config["low_freq_factor"]
        )
        high_freq_wavelen = (
            freq_config["original_context_length"] / freq_config["high_freq_factor"]
        )

        wavelen = 2 * torch.pi / inv_freq

        inv_freq_llama = torch.where(
            wavelen > low_freq_wavelen, inv_freq / freq_config["factor"], inv_freq
        )

        smooth_factor = (
            freq_config["original_context_length"] / wavelen
            - freq_config["low_freq_factor"]
        ) / (freq_config["high_freq_factor"] - freq_config["low_freq_factor"])

        smoothed_inv_freq = (1 - smooth_factor) * (
            inv_freq / freq_config["factor"]
        ) + smooth_factor * inv_freq

        is_medium_freq = (wavelen <= low_freq_wavelen) & (wavelen >= high_freq_wavelen)
        inv_freq_llama = torch.where(is_medium_freq, smoothed_inv_freq, inv_freq_llama)
        inv_freq = inv_freq_llama

    position = torch.arange(context_length, dtype=dtype)

    position = position.unsqueeze(1)
    frequencies = inv_freq.unsqueeze(0)

    angles = position * frequencies
    angles = torch.cat([angles, angles], dim=1)

    cos = torch.cos(angles)
    sin = torch.sin(angles)

    return cos, sin


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply rotary position embedding to input tensor.

    Args:
        x: Input tensor of shape (batch, num_heads, seq_len, head_dim).
        cos: Cosine values of shape (seq_len, head_dim).
        sin: Sine values of shape (seq_len, head_dim).

    Returns:
        Rotated tensor of same shape as input.
    """
    _, _, seq_len, head_dim = x.shape
    if head_dim % 2 != 0:
        msg = "Head dimensions must be even"
        raise ValueError(msg)

    x1 = x[..., : head_dim // 2]
    x2 = x[..., head_dim // 2 :]

    cos = cos[:seq_len, :].unsqueeze(0).unsqueeze(0)
    sin = sin[:seq_len, :].unsqueeze(0).unsqueeze(0)

    rotated = torch.cat([-x2, x1], dim=-1)
    x_rotated = (x * cos) + (rotated * sin)

    return x_rotated.to(dtype=x.dtype)


class GroupQueryAttention(nn.Module):
    """Grouped-Query Attention (GQA) module.

    GQA reduces memory bandwidth by sharing key/value heads across query heads.
    """

    def __init__(
        self,
        d_in: int,
        d_out: int,
        num_heads: int,
        num_kv_groups: int,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        if d_out % num_heads != 0:
            msg = "d_out must be divisible by num_heads"
            raise ValueError(msg)
        if num_heads % num_kv_groups != 0:
            msg = "num heads must be divisible by num kv groups"
            raise ValueError(msg)

        self.d_out = d_out
        self.num_heads = num_heads
        self.num_kv_groups = num_kv_groups
        self.group_size = num_heads // num_kv_groups
        self.head_dims = d_out // num_heads

        self.W_query = nn.Linear(d_in, d_out, bias=False, dtype=dtype)
        self.W_key = nn.Linear(
            d_in, num_kv_groups * self.head_dims, bias=False, dtype=dtype
        )
        self.W_value = nn.Linear(
            d_in, num_kv_groups * self.head_dims, bias=False, dtype=dtype
        )
        self.out_proj = nn.Linear(d_out, d_out, bias=False, dtype=dtype)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        b, num_tokens, _ = x.shape

        queries = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)

        queries = queries.view(b, num_tokens, self.num_heads, self.head_dims)
        keys = keys.view(b, num_tokens, self.num_kv_groups, self.head_dims)
        values = values.view(b, num_tokens, self.num_kv_groups, self.head_dims)

        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        queries = apply_rope(queries, cos, sin)
        keys = apply_rope(keys, cos, sin)

        keys = keys.repeat_interleave(self.group_size, dim=1)
        values = values.repeat_interleave(self.group_size, dim=1)

        attn_scores = queries @ keys.transpose(2, 3)
        attn_scores = attn_scores.masked_fill(mask, -1e9)

        attn_weights = torch.softmax(attn_scores / math.sqrt(self.head_dims), dim=-1)

        context_vec = (attn_weights @ values).transpose(1, 2)
        context_vec = context_vec.reshape(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)

        return context_vec


class TransformerBlock(nn.Module):
    """Pre-norm transformer block with GQA attention and SwiGLU FFN."""

    def __init__(self, cfg: "LlamaConfig") -> None:
        super().__init__()
        self.attn = GroupQueryAttention(
            d_in=cfg.emb_dim,
            d_out=cfg.emb_dim,
            num_heads=cfg.n_heads,
            num_kv_groups=cfg.n_kv_groups,
            dtype=cfg.dtype,
        )
        self.ff = FeedForward(cfg)
        self.norm1 = RMSNorm(cfg.emb_dim, eps=1e-6)
        self.norm2 = RMSNorm(cfg.emb_dim, eps=1e-6)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        shortcut = x
        x = self.norm1(x)
        x = self.attn(x, mask, cos, sin)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = x + shortcut

        return x
