"""Llama3 model implementation."""

from __future__ import annotations

import torch
import torch.nn as nn

from .config import LlamaConfig
from .layers import RMSNorm, TransformerBlock, compute_rope_params


class Llama3(nn.Module):
    """Complete Llama3 transformer model.

    A character-level language model using grouped-query attention and RoPE.
    """

    def __init__(self, cfg: LlamaConfig) -> None:
        super().__init__()

        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.emb_dim, dtype=cfg.dtype)

        self.trf_blocks = nn.ModuleList(
            TransformerBlock(cfg) for _ in range(cfg.n_layers)
        )

        self.final_norm = RMSNorm(cfg.emb_dim, eps=1e-5)
        self.out_head = nn.Linear(
            cfg.emb_dim, cfg.vocab_size, bias=False, dtype=cfg.dtype
        )

        cos, sin = compute_rope_params(
            head_dims=cfg.head_dim,
            theta_base=cfg.rope_base,
            context_length=cfg.context_length,
            freq_config=cfg.freq_config,
        )
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

        # Cache the causal mask
        self._cached_mask: torch.Tensor | None = None
        self._mask_context_len: int = 0

        self.cfg = cfg

    def _get_mask(self, num_tokens: int, device: torch.device) -> torch.Tensor:
        """Get or create cached causal mask."""
        if self._cached_mask is None or num_tokens > self._mask_context_len:
            self._cached_mask = torch.triu(
                torch.ones(
                    num_tokens, num_tokens, device=device, dtype=torch.bool
                ),
                diagonal=1,
            )
            self._mask_context_len = num_tokens
        return self._cached_mask[:num_tokens, :num_tokens]

    def forward(self, in_idx: torch.Tensor) -> torch.Tensor:
        token_embs = self.token_emb(in_idx)
        x = token_embs

        num_tokens = x.shape[1]
        mask = self._get_mask(num_tokens, x.device)

        for block in self.trf_blocks:
            x = block(x, mask, self.cos, self.sin)
        x = self.final_norm(x)
        logits = self.out_head(x.to(self.cfg.dtype))
        return logits

    @property
    def device(self) -> torch.device:
        """Return the device of the first parameter."""
        return next(self.parameters()).device
