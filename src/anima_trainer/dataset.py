"""Image/caption dataset with aspect-ratio bucketing and optional latent/text caching.

A LoRA dataset is a folder of images, each with a ``.txt`` caption sidecar (or a
shared default caption). Images are grouped into aspect-ratio buckets so every
batch shares a tensor shape, then optionally pre-encoded to VAE latents and text
embeddings on disk to keep VRAM and per-step cost low.
"""

from __future__ import annotations

import hashlib
import logging
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file, save_file
from torch.utils.data import Dataset

from .config import DatasetConfig

logger = logging.getLogger(__name__)

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".avif"}


@dataclass
class Sample:
    image_path: Path
    caption: str
    bucket: tuple[int, int]  # (width, height)


def _build_buckets(cfg: DatasetConfig) -> list[tuple[int, int]]:
    """Generate (w, h) buckets near the target area, respecting min/max/step."""
    if not cfg.enable_bucketing:
        return [(cfg.resolution, cfg.resolution)]
    target_area = cfg.resolution * cfg.resolution
    buckets: set[tuple[int, int]] = set()
    w = cfg.bucket_min
    while w <= cfg.bucket_max:
        h = int(round(target_area / w / cfg.bucket_step) * cfg.bucket_step)
        h = max(cfg.bucket_min, min(cfg.bucket_max, h))
        buckets.add((w, h))
        w += cfg.bucket_step
    return sorted(buckets)


def _assign_bucket(size: tuple[int, int], buckets: list[tuple[int, int]]) -> tuple[int, int]:
    """Pick the bucket whose aspect ratio best matches the image."""
    w, h = size
    ar = w / h
    return min(buckets, key=lambda b: abs((b[0] / b[1]) - ar))


def _read_caption(image_path: Path, cfg: DatasetConfig) -> str:
    cap = cfg.default_caption
    sidecar = image_path.with_suffix(cfg.caption_ext)
    if sidecar.exists():
        cap = sidecar.read_text(encoding="utf-8").strip()
    if cfg.trigger_word:
        cap = f"{cfg.trigger_word}, {cap}".strip(", ")
    return cap


def discover_samples(cfg: DatasetConfig) -> list[Sample]:
    image_dir = Path(cfg.image_dir)
    if not image_dir.is_dir():
        raise FileNotFoundError(f"image_dir does not exist: {image_dir}")
    buckets = _build_buckets(cfg)

    samples: list[Sample] = []
    for path in sorted(image_dir.rglob("*")):
        if path.suffix.lower() not in IMAGE_EXTS:
            continue
        with Image.open(path) as im:
            bucket = _assign_bucket(im.size, buckets)
        caption = _read_caption(path, cfg)
        for _ in range(max(1, cfg.repeats)):
            samples.append(Sample(path, caption, bucket))
    if not samples:
        raise ValueError(f"No images found in {image_dir}")
    logger.info("Discovered %d samples across %d buckets.", len(samples), len(buckets))
    return samples


def _load_image_tensor(path: Path, bucket: tuple[int, int]) -> torch.Tensor:
    """Load, resize-crop to the bucket, and normalize an image to [-1, 1]."""
    w, h = bucket
    img = Image.open(path).convert("RGB")
    # Resize preserving aspect ratio to cover the bucket, then center-crop.
    scale = max(w / img.width, h / img.height)
    img = img.resize((max(w, int(img.width * scale)), max(h, int(img.height * scale))),
                     Image.LANCZOS)
    left = (img.width - w) // 2
    top = (img.height - h) // 2
    img = img.crop((left, top, left + w, top + h))
    arr = np.asarray(img, dtype=np.float32) / 127.5 - 1.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()  # (C, H, W)


def _cache_key(sample: Sample) -> str:
    raw = f"{sample.image_path}|{sample.bucket}|{sample.caption}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


class AnimaDataset(Dataset):
    """Dataset yielding pixel tensors (and captions) or cached latents/embeds."""

    def __init__(self, cfg: DatasetConfig, samples: list[Sample] | None = None):
        self.cfg = cfg
        self.samples = samples if samples is not None else discover_samples(cfg)
        self.cache_dir = Path(cfg.cache_dir)
        self._use_cache = False  # set True after precompute_cache()

    def __len__(self) -> int:
        return len(self.samples)

    def _cache_path(self, sample: Sample) -> Path:
        return self.cache_dir / f"{_cache_key(sample)}.safetensors"

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        item: dict = {"caption": sample.caption, "bucket": sample.bucket}

        if self._use_cache:
            cached = load_file(str(self._cache_path(sample)))
            item["latent"] = cached["latent"]
            if "embeds" in cached:
                item["embeds"] = cached["embeds"]
                item["mask"] = cached["mask"]
        else:
            item["pixel_values"] = _load_image_tensor(sample.image_path, sample.bucket)
        return item

    # ---- caching ----------------------------------------------------------
    def precompute_cache(self, bundle, device, dtype) -> None:
        """Encode every sample's latents (and optionally text embeds) to disk."""
        from .encoders import encode_images_to_latents, encode_prompts

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        for sample in self.samples:
            path = self._cache_path(sample)
            if path.exists():
                continue
            pixel = _load_image_tensor(sample.image_path, sample.bucket).unsqueeze(0)
            latent = encode_images_to_latents(bundle, pixel, dtype)[0].cpu().contiguous()
            tensors = {"latent": latent}
            if self.cfg.cache_text_embeds:
                enc = encode_prompts(bundle, [sample.caption], dtype)
                tensors["embeds"] = enc["embeds"][0].cpu().contiguous()
                tensors["mask"] = enc["mask"][0].cpu().contiguous()
            save_file(tensors, str(path))
        self._use_cache = True
        logger.info("Latent/text cache ready in %s", self.cache_dir)


class BucketBatchSampler(torch.utils.data.Sampler):
    """Yield batches whose samples all share one bucket (so shapes match)."""

    def __init__(self, samples: list[Sample], batch_size: int, shuffle: bool = True,
                 seed: int = 0):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        self.bucket_to_indices: dict[tuple[int, int], list[int]] = {}
        for i, s in enumerate(samples):
            self.bucket_to_indices.setdefault(s.bucket, []).append(i)

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        batches: list[list[int]] = []
        for indices in self.bucket_to_indices.values():
            idxs = list(indices)
            if self.shuffle:
                rng.shuffle(idxs)
            for i in range(0, len(idxs), self.batch_size):
                batches.append(idxs[i:i + self.batch_size])
        if self.shuffle:
            rng.shuffle(batches)
        self.epoch += 1
        yield from batches

    def __len__(self) -> int:
        return sum((len(v) + self.batch_size - 1) // self.batch_size
                   for v in self.bucket_to_indices.values())


def collate(batch: list[dict]) -> dict:
    """Stack a same-bucket batch into tensors."""
    out: dict = {"captions": [b["caption"] for b in batch]}
    if "pixel_values" in batch[0]:
        out["pixel_values"] = torch.stack([b["pixel_values"] for b in batch])
    if "latent" in batch[0]:
        out["latent"] = torch.stack([b["latent"] for b in batch])
    if "embeds" in batch[0]:
        out["embeds"] = torch.stack([b["embeds"] for b in batch])
        out["mask"] = torch.stack([b["mask"] for b in batch])
    return out
