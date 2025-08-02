import torch
import torch.nn as nn
from .layers import TransformerBlock, RMSNorm, compute_rope_params

class Llama3(nn.Module):
  def __init__(self, cfg):
    super().__init__()

    self.token_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"], dtype=cfg["dtype"])

    self.trf_blocks = nn.ModuleList(
      TransformerBlock(cfg) for _ in range(cfg["n_layers"])
    )

    self.final_norm = RMSNorm(cfg["emb_dim"], eps=1e-5)
    self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False, dtype=cfg["dtype"])

    cos, sin = compute_rope_params(
      head_dims=cfg["head_dim"],
      theta_base=cfg["rope_base"],
      context_length=cfg["context_length"],
      freq_config=cfg["freq_config"]
    )
    self.register_buffer("cos", cos, persistent=False)
    self.register_buffer("sin", sin, persistent=False)
    self.cfg = cfg 

  def forward(self, in_idx):
    token_embs = self.token_emb(in_idx)
    x = token_embs

    num_tokens = x.shape[1]
    mask = torch.triu(torch.ones(num_tokens, num_tokens, device=x.device, dtype=torch.bool), diagonal=1)

    for block in self.trf_blocks:
      x = block(x, mask, self.cos, self.sin)
    x = self.final_norm(x)
    logits = self.out_head(x.to(self.cfg["dtype"]))
    return logits