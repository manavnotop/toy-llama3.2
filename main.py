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

class GroupQueryAttention(nn.Module):
  def __init__(self, d_in, d_out, num_heads, num_kv_groups, dtype=None):
    super().__init__()
    assert d_out % num_heads == 0, "d_out must be divisible by num_heads"
    assert num_heads % num_kv_groups == 0, "num heads must be divisible by num kv groups"

    self.d_out = d_out
    self.num_heads = num_heads
    self.num_kv_groups = num_kv_groups
    self.group_size = num_heads // num_kv_groups
    self.head_dims = d_out // num_heads

    self.W_query = nn.Linear(d_in, d_out, bias=False, dtype=dtype)
    self.W_key = nn.Linear(d_in, num_kv_groups * self.head_dims, bias=False, dtype=dtype)
    self.W_value = nn.Linear(d_in, num_kv_groups * self.head_dims, bias=False, dtype=dtype)

    self.out_proj = nn.Linear(d_out, d_out, bias=False, dtype=dtype)

  def forward(self, x, mask, cos, sin):
    b, num_tokens, d_in = x.shape()

    queries = self.W_query(x)
    keys = self.W_key(x)
    values = self.W_value(x)

    queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
    keys = keys.view(b, num_tokens, self.num_kv_groups, self.head_dim)
    values = values.view(b, num_tokens, self.num_kv_groups, self.head_dim)

    queries = queries.tranpose(1, 2) # -> (b, num_heads, num_tokens, head_dim)
    keys = keys.transpose(1, 2)      # -> (b, num_kv_groups, num_tokens, head_dim)
    values = values.transpose(1, 2)  # -> (b, num_kv_groups, num_tokens, head_dim)

    queries = apply_rope(queries, cos, sin)
    values = apply_rope(values, cos, sin)

    keys = keys.repeat_interleave(self.group_size, dim=1)
    values = values.repeat_interleave(self.group_size, dim=1)

    attn_scores = queries @ values.transpose(2, 3)

    attn_scores = attn_scores.masked_fill(mask, -torch.inf)

    attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
    assert keys.shape[-1] == self.head_dim

    context_vec = (attn_weights @ values).transpose(1, 2)

    context_vec = context_vec.reshape(b, num_tokens, self.d_out)
    context_vec = self.out_proj(context_vec)

    return context_vec

class TransformerBlock(nn.Module):
  def __init__(self, cfg):
    super().__init__()
    self.attn = GroupQueryAttention(
      d_in = cfg["emb_dim"],
      d_out = cfg["emb_dim"],
      num_heads = cfg["n_heads"],
      num_kv_groups = cfg["n_kv_groups"],
      dtype= cfg["dtype"]
    )
    self.ff = FeedForward(cfg)
    self.norm1 = RMSNorm(cfg["emb_dim"], eps=1e-6)
    self.norm2 = RMSNorm(cfg["emb_dim"], eps=1e-6)

  def forward(self, x, mask, cos, sin):
    shortcut = x
    x = self.norm1(x)
    x = self.attn(x)
    x = x + shortcut

    shortcut = x 
    x = self.norm2(x)
    x = self.ff(x)
    x = x + shortcut

    return x
  
class Llama3(nn.Module):
  def __init__(self, cfg):
    super().__init__()

    self.token_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"], dtype=cfg["cfg"])

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
    token_embs = token_embs(in_idx)
    x = token_embs

    num_tokens = x.shape[1]
    mask = torch.triu(torch.ones(num_tokens, num_tokens, device=x.device, dtype=torch.bool), diagonal=1)

    for block in self.trf_blocks:
      x = block(x, mask, self.cos, self.sin)
    x = self.final_norm(x)
    logits = self.out_head(x.to(self.cfg["dtype"]))
    return logits