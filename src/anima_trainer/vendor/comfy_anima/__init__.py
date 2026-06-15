"""Vendored Anima DiT (ComfyUI + Cosmos-Predict2), runnable standalone.

LICENSE NOTE: ``anima_model.py`` derives from ComfyUI (GPL-3.0); ``predict2.py``
and ``position_embedding.py`` derive from NVIDIA cosmos-predict2 (Apache-2.0);
``shims.py`` reimplements comfy_kitchen ops (Apache-2.0). Because the GPL-3.0
component is included, the project as distributed is effectively GPL-3.0.

The config below is derived directly from the anima-base-v1.0 checkpoint tensor
shapes (model dim 2048, 28 blocks, 16 heads, 16-ch latents, 2x2 patch, AdaLN-LoRA
dim 256, cross-attn context 1024, + the 6-block Qwen-3 llm_adapter).
"""

from __future__ import annotations

import torch

from .anima_model import Anima
from .ops import ops

# Architecture constants read off the checkpoint header.
ANIMA_DIT_CONFIG = dict(
    max_img_h=128,          # rope3d is computed dynamically; these don't size params
    max_img_w=128,
    max_frames=1,
    in_channels=16,
    out_channels=16,
    patch_spatial=2,
    patch_temporal=1,
    concat_padding_mask=True,
    model_channels=2048,
    num_blocks=28,
    num_heads=16,
    mlp_ratio=4.0,
    crossattn_emb_channels=1024,
    pos_emb_cls="rope3d",
    use_adaln_lora=True,
    adaln_lora_dim=256,
    extra_per_block_abs_pos_emb=False,
    rope_enable_fps_modulation=False,
)


def build_anima_dit(device=None, dtype=None) -> Anima:
    """Instantiate the Anima DiT (weights uninitialized)."""
    return Anima(device=device, dtype=dtype, operations=ops, **ANIMA_DIT_CONFIG)


def load_anima_dit(state_dict: dict, device=None, dtype=None,
                   strict: bool = True) -> Anima:
    """Build the DiT and load a checkpoint state_dict (keys prefixed with ``net.``)."""
    model = build_anima_dit(device=device, dtype=dtype)
    # Checkpoint keys look like 'net.blocks.0...'; our module root is the DiT itself.
    cleaned = {}
    for k, v in state_dict.items():
        cleaned[k[4:] if k.startswith("net.") else k] = v
    missing, unexpected = model.load_state_dict(cleaned, strict=strict)
    return model
