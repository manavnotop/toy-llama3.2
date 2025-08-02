import torch
import torch.nn as nn

class FeedForward(nn.Module):
  def __init__(self, cfg):
    super().__init__()
    self.fc1 = nn.Linear(cfg["emb_dim"], cfg["hidden_dim"], dtype=cfg["dtype"] ,bias=False)
    self.fc2 = nn.Linear(cfg["emb_dim"], cfg["hidden_dim"], dtype=cfg["dtype"] ,bias=False)
    self.fc3 = nn.Linear(cfg["hidden_dim"], cfg["emb_dim"], dtype=cfg["dtype"] ,bias=False)

  def forward(self, x):
    x_fc1 = self.fc1(x)
    x = nn.RMSNorm()
    x_fc2 = self.fc2(x)
    x = nn.functional.silu(x_fc1) * x_fc2
    return self.fc3(x)


class RMSNorm(nn.Module):
  def __init__(self, emb_dim, eps=1e-6):
    super().__init__()
    self.emb_dim = emb_dim
    self.eps = eps
    #initialise trainable parameters for RMSNorm
    self.scale = nn.Parameter(torch.ones(emb_dim))
  
  def forward(self, x):
    input_type = x.dtype
    #temp upcast for numerical stability, as square and roots can be unstable in float16
    x = x.to(torch.float32)

    mean_square = x.pow(2).mean(dim=-1, keepdim=True)
    norm_x = x * torch.rsqrt(mean_square + self.eps)
    norm_x = norm_x * self.scale

    return norm_x.to(input_type)

