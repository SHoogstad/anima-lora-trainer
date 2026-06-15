"""LoRA injection and saving for the Anima DiT, built on PEFT.

We attach LoRA adapters to the attention projections of the Cosmos-Predict2
transformer blocks and export them in the diffusers/kohya ``*.safetensors``
convention so the result loads in ComfyUI and other Anima-aware tools.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from .config import LoRAConfig

logger = logging.getLogger(__name__)


def _module_matches(name: str, cfg: LoRAConfig) -> bool:
    """Decide whether a transformer submodule should receive a LoRA adapter."""
    # Self- vs cross-attention gating (Cosmos: attn1 self, attn2 cross).
    if ".attn1." in name and not cfg.train_self_attn:
        return False
    if ".attn2." in name and not cfg.train_cross_attn:
        return False

    # Optional block-index restriction, e.g. transformer_blocks.5.attn1.to_q
    if cfg.block_indices:
        parts = name.split(".")
        block_idx = None
        for i, p in enumerate(parts[:-1]):
            if p in ("transformer_blocks", "blocks") and parts[i + 1].isdigit():
                block_idx = int(parts[i + 1])
                break
        if block_idx is None or block_idx not in cfg.block_indices:
            return False

    # Match the configured projection suffixes (to_q, to_k, ...).
    return any(name == t or name.endswith("." + t) for t in cfg.target_modules)


def resolve_target_modules(transformer: torch.nn.Module, cfg: LoRAConfig) -> list[str]:
    """Expand the high-level LoRA config into concrete module names in this model."""
    targets = [
        name for name, module in transformer.named_modules()
        if isinstance(module, torch.nn.Linear) and _module_matches(name, cfg)
    ]
    if not targets:
        raise ValueError(
            "No LoRA target modules matched. Check lora.target_modules / block_indices "
            f"against the model's layer names. Configured suffixes: {cfg.target_modules}"
        )
    logger.info("LoRA will wrap %d linear modules.", len(targets))
    return targets


def inject_lora(transformer: torch.nn.Module, cfg: LoRAConfig):
    """Wrap ``transformer`` with a PEFT LoRA model and return it."""
    from peft import LoraConfig as PeftLoraConfig
    from peft import get_peft_model

    targets = resolve_target_modules(transformer, cfg)
    peft_cfg = PeftLoraConfig(
        r=cfg.rank,
        lora_alpha=cfg.alpha,
        lora_dropout=cfg.dropout,
        target_modules=targets,
        init_lora_weights="gaussian",
    )
    peft_model = get_peft_model(transformer, peft_cfg)
    trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    logger.info("LoRA trainable params: %s", f"{trainable:,}")
    return peft_model


def trainable_parameters(peft_model) -> list[torch.nn.Parameter]:
    return [p for p in peft_model.parameters() if p.requires_grad]


def save_lora(peft_model, output_dir: str | Path, name: str,
              metadata: dict[str, str] | None = None) -> Path:
    """Export LoRA weights to ``<output_dir>/<name>.safetensors``."""
    from peft import get_peft_model_state_dict

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{name}.safetensors"

    state = get_peft_model_state_dict(peft_model)
    # Contiguous CPU fp16/bf16-friendly tensors for portability.
    state = {k: v.detach().to("cpu").contiguous() for k, v in state.items()}

    meta = {"format": "pt", "anima_lora": "v1"}
    if metadata:
        meta.update({k: str(v) for k, v in metadata.items()})

    save_file(state, str(out_path), metadata=meta)
    logger.info("Saved LoRA -> %s (%d tensors)", out_path, len(state))
    return out_path


def load_lora_into(peft_model, path: str | Path) -> None:
    """Load a previously saved LoRA back into a PEFT model (for resume)."""
    from peft import set_peft_model_state_dict

    state = load_file(str(path))
    set_peft_model_state_dict(peft_model, state)
