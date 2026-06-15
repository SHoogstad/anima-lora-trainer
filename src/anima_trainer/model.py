"""Download and load the Anima model components.

Anima (``circlestone-labs/Anima``) ships as ComfyUI-style single-file safetensors:

  * DiT          -> ``CosmosTransformer3DModel`` (Cosmos-Predict2 lineage)
  * Text encoder -> Qwen-3 0.6B  (transformers)
  * VAE          -> Qwen-Image VAE  (``AutoencoderKLQwenImage`` in diffusers)

Because the model is very new, this module concentrates *all* Anima-specific
loading in one place behind :class:`ModelBundle`. Each loader first tries the
diffusers/transformers single-file path and otherwise raises a clear, actionable
error instead of silently building the wrong graph. When you validate against the
real weights, this is the only file you should need to touch.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download

from .config import ModelConfig
from .device import Device

logger = logging.getLogger(__name__)


@dataclass
class ModelBundle:
    transformer: torch.nn.Module       # the DiT (LoRA is injected here)
    text_encoder: torch.nn.Module
    tokenizer: object
    vae: torch.nn.Module
    scheduler: object
    # Model-specific constants, resolved at load time:
    vae_scale_factor: float
    latent_channels: int

    def eval_frozen(self) -> None:
        """Freeze every base weight; LoRA params are added/unfrozen later."""
        for module in (self.transformer, self.text_encoder, self.vae):
            module.requires_grad_(False)
            module.eval()


def download_components(cfg: ModelConfig) -> dict[str, str]:
    """Fetch the three safetensors files, returning local paths."""
    common = dict(repo_id=cfg.repo_id, revision=cfg.revision, cache_dir=cfg.cache_dir)
    paths = {}
    for key, fname in (
        ("dit", cfg.dit_file),
        ("text_encoder", cfg.text_encoder_file),
        ("vae", cfg.vae_file),
    ):
        logger.info("Downloading %s (%s)...", key, fname)
        paths[key] = hf_hub_download(filename=fname, **common)
    return paths


# --------------------------------------------------------------------------
# Component loaders. These are deliberately isolated so they can be adjusted
# against the real weights without touching the training loop.
# --------------------------------------------------------------------------
def _load_transformer(path: str, dtype: torch.dtype) -> torch.nn.Module:
    from diffusers import CosmosTransformer3DModel

    if hasattr(CosmosTransformer3DModel, "from_single_file"):
        logger.info("Loading DiT via CosmosTransformer3DModel.from_single_file")
        return CosmosTransformer3DModel.from_single_file(path, torch_dtype=dtype)

    raise NotImplementedError(
        "Your installed diffusers cannot load the Anima DiT from a single ComfyUI "
        "safetensors file. Upgrade diffusers (pip install -U diffusers) so that "
        "CosmosTransformer3DModel.from_single_file is available, or convert the "
        "checkpoint to a diffusers folder and load it via from_pretrained. "
        "Loading is isolated in model.py:_load_transformer for exactly this reason."
    )


def _load_vae(path: str, dtype: torch.dtype) -> torch.nn.Module:
    # Anima uses the Qwen-Image VAE.
    try:
        from diffusers import AutoencoderKLQwenImage as VAEClass
    except ImportError as exc:  # pragma: no cover - depends on diffusers version
        raise NotImplementedError(
            "diffusers.AutoencoderKLQwenImage is unavailable. Upgrade diffusers to a "
            "version that ships the Qwen-Image VAE used by Anima."
        ) from exc

    if hasattr(VAEClass, "from_single_file"):
        return VAEClass.from_single_file(path, torch_dtype=dtype)
    raise NotImplementedError("AutoencoderKLQwenImage.from_single_file is unavailable.")


def _load_text_encoder(path: str, repo_id: str, cache_dir: str | None,
                       dtype: torch.dtype) -> tuple[torch.nn.Module, object]:
    """Load the Qwen-3 0.6B text encoder + tokenizer.

    The tokenizer/config is pulled from the repo's ``text_encoder`` /
    ``tokenizer`` subfolders when present; the weights come from the single file.
    """
    from transformers import AutoModel, AutoTokenizer

    # Tokenizer: prefer a repo subfolder, fall back to the stock Qwen3 0.6B tokenizer.
    tokenizer = None
    for subfolder in ("tokenizer", "text_encoder"):
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                repo_id, subfolder=subfolder, cache_dir=cache_dir)
            break
        except Exception:  # noqa: BLE001 - try the next candidate
            continue
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B-Base", cache_dir=cache_dir)

    # Model: try a diffusers/transformers subfolder; otherwise build from config +
    # load the raw state dict from the single safetensors file.
    try:
        text_encoder = AutoModel.from_pretrained(
            repo_id, subfolder="text_encoder", torch_dtype=dtype, cache_dir=cache_dir)
    except Exception:  # noqa: BLE001
        from safetensors.torch import load_file
        from transformers import AutoConfig, AutoModel as _AM

        config = AutoConfig.from_pretrained("Qwen/Qwen3-0.6B-Base", cache_dir=cache_dir)
        text_encoder = _AM.from_config(config, torch_dtype=dtype)
        missing, unexpected = text_encoder.load_state_dict(load_file(path), strict=False)
        if missing:
            logger.warning("Text encoder: %d missing keys (e.g. %s)",
                           len(missing), missing[:3])
        if unexpected:
            logger.warning("Text encoder: %d unexpected keys (e.g. %s)",
                           len(unexpected), unexpected[:3])
    return text_encoder, tokenizer


def _build_scheduler():
    """Flow-matching scheduler matching the rectified-flow training objective."""
    from diffusers import FlowMatchEulerDiscreteScheduler

    return FlowMatchEulerDiscreteScheduler()


def load_model(cfg: ModelConfig, device: Device, dtype: torch.dtype) -> ModelBundle:
    """Download (if needed) and load every Anima component onto ``device``."""
    paths = download_components(cfg)

    transformer = _load_transformer(paths["dit"], dtype)
    vae = _load_vae(paths["vae"], dtype)
    text_encoder, tokenizer = _load_text_encoder(
        paths["text_encoder"], cfg.repo_id, cfg.cache_dir, dtype)
    scheduler = _build_scheduler()

    # Resolve model-specific constants from the loaded config where possible.
    vae_cfg = getattr(vae, "config", None)
    vae_scale_factor = float(getattr(vae_cfg, "scaling_factor", 1.0) or 1.0)
    latent_channels = int(getattr(vae_cfg, "latent_channels", 16) or 16)

    bundle = ModelBundle(
        transformer=transformer,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        vae=vae,
        scheduler=scheduler,
        vae_scale_factor=vae_scale_factor,
        latent_channels=latent_channels,
    )
    bundle.eval_frozen()

    dev = device.torch_device
    transformer.to(dev)
    text_encoder.to(dev)
    vae.to(dev)
    logger.info("Loaded Anima components onto %s (vae_scale=%.5f, latent_ch=%d)",
                dev, vae_scale_factor, latent_channels)
    return bundle


def resolve_local_paths(cfg: ModelConfig) -> dict[str, str] | None:
    """Return cached local paths if all three files are already downloaded."""
    out = {}
    for key, fname in (("dit", cfg.dit_file), ("text_encoder", cfg.text_encoder_file),
                       ("vae", cfg.vae_file)):
        try:
            out[key] = hf_hub_download(
                repo_id=cfg.repo_id, filename=fname, revision=cfg.revision,
                cache_dir=cfg.cache_dir, local_files_only=True)
        except Exception:  # noqa: BLE001 - not cached yet
            return None
    return out
