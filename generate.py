"""Text generation script for Toy LLaMA 3.2."""

import contextlib
import json

import torch

from src.toy_llama.config import create_config
from src.toy_llama.model import Llama3

# Select device
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

# Load character vocabulary
with open("char_vocab.json", "r") as f:
    chars = json.load(f)

# Create character-index mapping
char_to_idx = {c: i for i, c in enumerate(chars)}
idx_to_char = {i: c for i, c in enumerate(chars)}


def decode(tensor: torch.Tensor) -> str:
    """Decode tensor of token indices to string."""
    return "".join(idx_to_char[i.item()] for i in tensor)


# Create config
cfg = create_config(
    vocab_size=len(chars),
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
model = Llama3(cfg).to(device).to(cfg.dtype)

# Load trained weights (suppress weights_only warning for custom model)
model.load_state_dict(
    torch.load("toy_llama3_24epochs.pth", weights_only=False, map_location=device)
)
model.eval()

print("Model loaded and ready for generation!\n")


def generate(
    model: Llama3,
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 0.85,
) -> str:
    """Generate text from a prompt.

    Args:
        model: The Llama3 model.
        prompt: Input text prompt.
        max_new_tokens: Number of tokens to generate.
        temperature: Sampling temperature (higher = more random).

    Returns:
        Generated text including the prompt.
    """
    model.eval()

    # Convert input prompt to tensor of token indices
    encoded = torch.tensor(
        [char_to_idx.get(c, 0) for c in prompt],
        dtype=torch.long,
        device=device,
    ).unsqueeze(0)

    autocast_context = (
        torch.autocast(device_type="cuda", dtype=cfg.dtype)
        if torch.cuda.is_available()
        else contextlib.nullcontext()
    )

    for _ in range(max_new_tokens):
        input_ids = encoded[:, -cfg.context_length :]
        with torch.no_grad(), autocast_context:
            logits = model(input_ids)
            logits = logits[:, -1, :]
            # Apply temperature AFTER softmax for proper distribution
            probs = torch.softmax(logits / temperature, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
        encoded = torch.cat([encoded, idx_next], dim=1)

    return decode(encoded[0])


# Example generation
prompt = "To be or not to be, that is the"
print(f"Prompt: {repr(prompt)}")
result = generate(model, prompt)
print(f"Generated: {repr(result)}")
