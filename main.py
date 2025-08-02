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

def compute_rope_params(head_dims, theta_base=10000, context_length=4096, freq_config=None, dtype=torch.float32):
  assert head_dims % 2 == 0, "Embedding dimension must be even"

  #generate all powers of 2 beforehand
  even_indices = torch.arange(0, head_dims, 2, dtype=dtype) #[0, 2, 4, ..., head_dim-2]
  #take only first half of them
  powers = even_indices[: head_dims // 2] #first d/2 elements
  normalised_powers = powers / head_dims #kinda like 2i/d

  inv_freq= 1 / (theta_base ** normalised_powers) 

  if freq_config is not None:
    low_freq_wavelen = freq_config["original_context_length"] / freq_config["low_freq_factor"]
    high_freq_wavelen = freq_config["original_context_length"] / freq_config["high_freq_factor"]

    wavelen = 2 * torch.pi / inv_freq

    inv_freq_llama = torch.where(
      wavelen > low_freq_wavelen, inv_freq / freq_config["factor"], inv_freq
    )

    smooth_factor = (freq_config["original_context_length"] / wavelen - freq_config["low_freq_factor"]) / (
      freq_config["high_freq_factor"] - freq_config["low_freq_factor"]
    )

    smoothed_inv_freq = (
      (1 - smooth_factor) * (inv_freq / freq_config["factor"]) + smooth_factor * inv_freq
    )

    is_medium_freq = (wavelen <= low_freq_wavelen) & (wavelen >= high_freq_wavelen)
    inv_freq_llama = torch.where(is_medium_freq, smoothed_inv_freq, inv_freq_llama)
    inv_freq = inv_freq_llama

  position = torch.arange(context_length, dtype=dtype)

  position = position.unsqueeze(1)      # (context_length) -> (context_lenght, 1)
  frequencies = inv_freq.unsqueeze(0)   # (head_dims // 2) -> (1, head_dims // 2)

  angles = position * frequencies              #(context_length, head_dims // 2)
  angles = torch.cat([angles, angles], dim=1)  #(context_length, head_dims)

  cos = torch.cos(angles)
  sin = torch.sin(angles)

  return cos, sin

def apply_rope(x, cos, sin):
  batch_size, num_heads, seq_len, head_dim = x.shape
  assert head_dim % 2 == 0, "Head dimensions must be even"

  x1 = x[..., : head_dim // 2]
  x2 = x[..., head_dim // 2 :]

  ## cos/sin: (seq_len, head_dim) -> (1, 1, seq_len, head_dim)
  cos = cos[:seq_len, :].unsqueeze(0).unsqueeze(0)
  sin = sin[:seq_len, :].unsqueeze(0).unsqueeze(0)

  #(x1, x2) -> (-x2, x1)
  rotated = torch.cat([-x2, x1], dim=-1)

  # x = x * cos + (-x2, x1) * sin
  x_rotated = (x * cos) + (rotated * sin)

  return x_rotated.to(dtype=x.dtype)