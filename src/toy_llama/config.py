"""Model configuration for Toy LLaMA 3.2."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class LlamaConfig:
    """Configuration for the Llama3 model.

    Attributes:
        vocab_size: Number of unique tokens in the vocabulary.
        emb_dim: Token embedding dimension.
        n_heads: Total number of attention heads.
        n_kv_groups: Number of key/value groups for grouped-query attention.
        n_layers: Number of transformer layers.
        hidden_dim: Feed-forward network hidden dimension.
        context_length: Maximum sequence length.
        rope_base: Base theta for rotary position embeddings.
        dtype: Data type for model parameters.
        head_dim: Attention head dimension (computed if None).
        freq_config: Frequency configuration for RoPE interpolation.
    """

    vocab_size: int
    emb_dim: int
    n_heads: int
    n_kv_groups: int
    n_layers: int
    hidden_dim: int
    context_length: int
    rope_base: float = 10000.0
    dtype: torch.dtype = torch.float32
    head_dim: Optional[int] = None
    freq_config: Optional[dict] = None

    def __post_init__(self) -> None:
        """Compute head_dim if not set."""
        if self.head_dim is None:
            self.head_dim = self.emb_dim // self.n_heads


def create_config(
    vocab_size: int,
    emb_dim: int = 256,
    n_heads: int = 8,
    n_kv_groups: int = 2,
    n_layers: int = 6,
    hidden_dim: int = 512,
    context_length: int = 256,
    rope_base: float = 10000.0,
    dtype: torch.dtype = torch.float32,
    *,
    device: str = "cpu",
) -> LlamaConfig:
    """Create a LlamaConfig with automatic dtype selection.

    Args:
        vocab_size: Number of unique tokens.
        emb_dim: Token embedding dimension.
        n_heads: Total number of attention heads.
        n_kv_groups: Number of key/value groups.
        n_layers: Number of transformer layers.
        hidden_dim: Feed-forward hidden dimension.
        context_length: Maximum sequence length.
        rope_base: RoPE base theta.
        dtype: Default dtype (overridden for MPS/CUDA).
        device: Compute device for automatic dtype selection.

    Returns:
        Configured LlamaConfig instance.
    """
    if device == "mps":
        actual_dtype = torch.float32
    elif device == "cuda":
        actual_dtype = torch.bfloat16
    else:
        actual_dtype = dtype

    return LlamaConfig(
        vocab_size=vocab_size,
        emb_dim=emb_dim,
        n_heads=n_heads,
        n_kv_groups=n_kv_groups,
        n_layers=n_layers,
        hidden_dim=hidden_dim,
        context_length=context_length,
        rope_base=rope_base,
        dtype=actual_dtype,
        freq_config={
            "original_context_length": context_length,
            "low_freq_factor": 8,
            "high_freq_factor": 1,
            "factor": 4,
        },
    )
