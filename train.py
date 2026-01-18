"""Training script for Toy LLaMA 3.2."""

import contextlib
import json
import time

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm import tqdm

from src.toy_llama.config import create_config
from src.toy_llama.model import Llama3

# Load dataset
print("Loading dataset...")
with open("data/tiny_shakespeare.txt", "r") as f:
    text = f.read()

# Build character-level vocabulary
chars = sorted(list(set(text)))
print("chars =", repr(chars))
vocab_size = len(chars)

# Mapping characters and token indices
char_to_idx = {c: i for i, c in enumerate(chars)}
idx_to_char = {i: c for i, c in enumerate(chars)}


def encode(s: str) -> torch.Tensor:
    """Encode string to tensor of token indices."""
    return torch.tensor(
        [char_to_idx[c] for c in s if c in char_to_idx], dtype=torch.long
    )


def decode(tensor: torch.Tensor) -> str:
    """Decode tensor of token indices to string."""
    return "".join(idx_to_char[i.item()] for i in tensor)


# Encode dataset into token ids
data = encode(text)
print(f"Dataset loaded: {len(text):,} chars, vocab_size={vocab_size}")

# Device selection
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

# Create config
cfg = create_config(
    vocab_size=vocab_size,
    emb_dim=256,
    n_heads=8,
    n_kv_groups=2,
    n_layers=6,
    hidden_dim=512,
    context_length=256,
    rope_base=10000.0,
    device=device,
)

# Initialize model
model = Llama3(cfg).to(device)
if device != "mps":
    model = model.to(cfg.dtype)

# Optimizer with weight decay
optimizer = AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
loss_fn = nn.CrossEntropyLoss()

# Learning rate scheduler
scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=1000, T_mult=2, eta_min=1e-5)

# Training hyperparameters
seq_len = cfg.context_length
batch_size = 8  # Increased from 4
epochs = 24
steps_per_epoch = 250
gradient_accumulation_steps = 2  # Effective batch size = 16
losses = []

# Training mode
model.train()
print(f"\nStarting training: {epochs} epochs × {steps_per_epoch} steps")

# Automatic mixed precision for CUDA
use_autocast = torch.cuda.is_available()
autocast_context = (
    torch.autocast(device_type="cuda", dtype=cfg.dtype)
    if use_autocast
    else contextlib.nullcontext()
)

# Efficient data loading with advanced indexing
data = data.to(device)

# Training loop
for epoch in range(epochs):
    start_time = time.time()
    epoch_loss = 0.0
    progress_bar = tqdm(range(steps_per_epoch), desc=f"Epoch {epoch + 1}/{epochs}")

    for step in progress_bar:
        # Sample random sequences using advanced indexing
        ix = torch.randint(len(data) - seq_len - 1, (batch_size,))
        offsets = torch.arange(seq_len, device=device)
        xb = data[ix.unsqueeze(1) + offsets].view(batch_size, seq_len)
        yb = data[ix.unsqueeze(1) + 1 + offsets].view(batch_size, seq_len)

        # Accumulate gradients for larger effective batch size
        optimizer.zero_grad(set_to_none=True)

        with autocast_context:
            logits = model(xb)
            loss = loss_fn(logits.view(-1, vocab_size), yb.view(-1))
            loss = loss / gradient_accumulation_steps

        loss.backward()
        optimizer.step()
        scheduler.step()

        epoch_loss += loss.item() * gradient_accumulation_steps
        progress_bar.set_postfix({"loss": f"{loss.item():.3f}"})

    avg_loss = epoch_loss / steps_per_epoch
    losses.append(avg_loss)
    elapsed = time.time() - start_time
    current_lr = scheduler.get_last_lr()[0]
    print(
        f"Epoch {epoch + 1} | Avg Loss: {avg_loss:.3f} | LR: {current_lr:.2e} | Time: {elapsed:.0f}s"
    )

    # Sample generation
    model.eval()
    input_seq = xb[0].unsqueeze(0)
    generated = input_seq[0].tolist()
    with torch.no_grad():
        for _ in range(100):
            input_chunk = torch.tensor(generated[-seq_len:], device=device).unsqueeze(0)
            logits = model(input_chunk)
            probs = torch.softmax(logits[:, -1, :], dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()
            generated.append(next_token)
    print("Sample generation:", repr(decode(torch.tensor(generated))))
    model.train()

print("\nTraining complete! Model saved.")

# Plot loss curve
plt.figure(figsize=(8, 4))
plt.plot(losses, label="Training Loss", marker="o")
plt.title("Toy Llama3 - Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
print("Loss curve saved as 'loss_curve.png'")

# Save model
model_save_path = f"toy_llama3_{epochs}epochs.pth"
torch.save(model.state_dict(), model_save_path)
print(f"Model saved to '{model_save_path}'")

# Save loss curve
plot_save_path = f"assets/loss_curve_{epochs}epochs.png"
plt.savefig(plot_save_path)
print(f"Loss curve saved as '{plot_save_path}'")

# Save vocabulary
with open("char_vocab.json", "w") as f:
    json.dump(chars, f)
