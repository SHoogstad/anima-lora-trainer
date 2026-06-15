"""Typed training configuration with TOML load/save.

The defaults target Anima (``circlestone-labs/Anima``), a 2B Cosmos-Predict2 DiT
with a Qwen-3 text encoder and the Qwen-Image VAE.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import tomlkit

# Hugging Face repo + the ComfyUI-style file layout used by Anima.
ANIMA_REPO_ID = "circlestone-labs/Anima"
ANIMA_DIT_FILE = "split_files/diffusion_models/anima-base-v1.0.safetensors"
ANIMA_TE_FILE = "split_files/text_encoders/qwen_3_06b_base.safetensors"
ANIMA_VAE_FILE = "split_files/vae/qwen_image_vae.safetensors"

# Cosmos-Predict2 DiT attention projections — the standard LoRA targets.
# attn1 = self-attention, attn2 = cross-attention (text conditioning).
DEFAULT_LORA_TARGETS = ["to_q", "to_k", "to_v", "to_out.0"]


@dataclass
class ModelConfig:
    repo_id: str = ANIMA_REPO_ID
    dit_file: str = ANIMA_DIT_FILE
    text_encoder_file: str = ANIMA_TE_FILE
    vae_file: str = ANIMA_VAE_FILE
    # Where to download/cache weights. None -> default HF cache.
    cache_dir: str | None = None
    revision: str = "main"


@dataclass
class LoRAConfig:
    rank: int = 16
    alpha: int = 16
    dropout: float = 0.0
    target_modules: list[str] = field(default_factory=lambda: list(DEFAULT_LORA_TARGETS))
    # Restrict to specific block indices, e.g. [0, 1, 2]. Empty = all blocks.
    block_indices: list[int] = field(default_factory=list)
    train_self_attn: bool = True   # attn1
    train_cross_attn: bool = True   # attn2


@dataclass
class DatasetConfig:
    image_dir: str = ""
    # Caption source: ".txt" sidecar files next to each image, or a single
    # caption applied to all images.
    caption_ext: str = ".txt"
    default_caption: str = ""
    trigger_word: str = ""          # prepended to every caption if set
    resolution: int = 1024          # base bucket resolution (Anima: 512..1536)
    bucket_min: int = 512
    bucket_max: int = 1536
    bucket_step: int = 64
    enable_bucketing: bool = True
    repeats: int = 1
    shuffle_caption: bool = False
    cache_latents: bool = True
    cache_text_embeds: bool = True
    cache_dir: str = "cache"


@dataclass
class OptimConfig:
    learning_rate: float = 1e-4
    optimizer: str = "adamw"        # adamw | adamw8bit (cuda only) | adafactor
    weight_decay: float = 1e-2
    lr_scheduler: str = "constant"  # constant | cosine | linear | constant_with_warmup
    warmup_steps: int = 0
    max_grad_norm: float = 1.0
    gradient_accumulation_steps: int = 1


@dataclass
class TrainConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)

    # ---- runtime ----------------------------------------------------------
    backend: str = "auto"           # auto | cuda | xpu | cpu
    device_index: int = 0
    dtype: str = "bf16"             # bf16 | fp16 | fp32
    seed: int = 42

    # ---- schedule ---------------------------------------------------------
    batch_size: int = 1
    max_train_steps: int = 2000
    save_every_steps: int = 250
    sample_every_steps: int = 0     # 0 = disabled
    gradient_checkpointing: bool = True

    # ---- flow-matching objective -----------------------------------------
    # Anima/Cosmos-Predict2 train with a rectified-flow (flow-matching) loss.
    timestep_sampling: str = "logit_normal"  # uniform | logit_normal
    logit_mean: float = 0.0
    logit_std: float = 1.0

    # ---- output -----------------------------------------------------------
    output_dir: str = "outputs"
    output_name: str = "anima_lora"

    # ----------------------------------------------------------------------
    def to_toml(self) -> str:
        return tomlkit.dumps(_to_plain(asdict(self)))

    def save(self, path: str | Path) -> None:
        Path(path).write_text(self.to_toml(), encoding="utf-8")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TrainConfig":
        return cls(
            model=ModelConfig(**data.get("model", {})),
            lora=LoRAConfig(**data.get("lora", {})),
            dataset=DatasetConfig(**data.get("dataset", {})),
            optim=OptimConfig(**data.get("optim", {})),
            **{k: v for k, v in data.items()
               if k not in ("model", "lora", "dataset", "optim")},
        )

    @classmethod
    def load(cls, path: str | Path) -> "TrainConfig":
        data = tomlkit.parse(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(_to_plain(data))


def _to_plain(obj: Any) -> Any:
    """Recursively convert tomlkit/dataclass containers to plain dict/list."""
    if dataclasses.is_dataclass(obj):
        return _to_plain(asdict(obj))
    if isinstance(obj, dict):
        # TOML has no null; omit None values so they fall back to dataclass defaults on load.
        return {k: _to_plain(v) for k, v in obj.items() if v is not None}
    if isinstance(obj, (list, tuple)):
        return [_to_plain(v) for v in obj]
    return obj
