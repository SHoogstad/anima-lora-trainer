"""VAE and text-encoder forward passes for Anima.

These wrap the two genuinely model-specific operations — turning pixels into VAE
latents and prompts into conditioning embeddings — behind stable signatures. The
training loop and the dataset cache both call only these functions, so any tweak
needed to match the real Anima weights lives here and nowhere else.
"""

from __future__ import annotations

import logging

import torch

from .model import ModelBundle

logger = logging.getLogger(__name__)


@torch.no_grad()
def encode_images_to_latents(bundle: ModelBundle, pixel_values: torch.Tensor,
                             dtype: torch.dtype) -> torch.Tensor:
    """Encode pixels in [-1, 1] of shape (B, C, H, W) to scaled latents.

    The Qwen-Image VAE is a causal video VAE, so it expects a temporal dim. We add
    a single frame, encode, then drop the temporal axis to get (B, Cz, h, w).
    """
    vae = bundle.vae
    pixel_values = pixel_values.to(vae.device, dtype=dtype)

    # (B, C, H, W) -> (B, C, T=1, H, W) for the temporal VAE.
    x = pixel_values.unsqueeze(2)
    encoded = vae.encode(x)
    dist = getattr(encoded, "latent_dist", None)
    if dist is not None:
        latent = dist.sample()              # AutoencoderKL-style output
    elif hasattr(encoded, "sample") and not torch.is_tensor(encoded):
        latent = encoded.sample             # output object with a .sample tensor
    else:
        latent = encoded                    # already a tensor

    latent = latent * bundle.vae_scale_factor
    # Drop the temporal dim if present: (B, Cz, T, h, w) -> (B, Cz, h, w).
    if latent.dim() == 5:
        latent = latent[:, :, 0]
    return latent.to(dtype)


@torch.no_grad()
def decode_latents(bundle: ModelBundle, latents: torch.Tensor,
                   dtype: torch.dtype) -> torch.Tensor:
    """Inverse of :func:`encode_images_to_latents`, for sampling previews."""
    vae = bundle.vae
    latents = latents.to(vae.device, dtype=dtype) / bundle.vae_scale_factor
    if latents.dim() == 4:
        latents = latents.unsqueeze(2)  # add temporal frame
    out = vae.decode(latents)
    image = out.sample if hasattr(out, "sample") else out
    if image.dim() == 5:
        image = image[:, :, 0]
    return image.clamp(-1, 1)


@torch.no_grad()
def encode_prompts(bundle: ModelBundle, prompts: list[str], dtype: torch.dtype,
                   max_length: int = 512) -> dict[str, torch.Tensor]:
    """Encode a batch of prompts with the Qwen-3 text encoder.

    Returns ``{"embeds": (B, T, D), "mask": (B, T)}``. The DiT cross-attention
    consumes ``embeds`` (gated by ``mask``).
    """
    tokenizer, text_encoder = bundle.tokenizer, bundle.text_encoder
    tokens = tokenizer(
        prompts, padding="max_length", truncation=True,
        max_length=max_length, return_tensors="pt",
    )
    input_ids = tokens.input_ids.to(text_encoder.device)
    attention_mask = tokens.attention_mask.to(text_encoder.device)

    outputs = text_encoder(input_ids=input_ids, attention_mask=attention_mask,
                           output_hidden_states=True)
    # Use the last hidden state as the conditioning sequence.
    embeds = outputs.last_hidden_state if hasattr(outputs, "last_hidden_state") \
        else outputs.hidden_states[-1]
    return {"embeds": embeds.to(dtype), "mask": attention_mask}
