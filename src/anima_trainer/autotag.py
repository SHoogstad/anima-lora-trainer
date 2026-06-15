"""Automatic dataset captioning with a WD14-style danbooru tagger.

Anime models like Anima are trained on danbooru tags (and natural language). This
module runs a `SmilingWolf` WD tagger (ONNX) over an image folder and writes a
``.txt`` caption sidecar next to each image — the same format the trainer reads.

ONNX Runtime keeps this light: it runs on CPU out of the box, and can use the
OpenVINO execution provider on Intel or DirectML on Windows. No torch needed for
tagging, so it works even before the (heavy) training stack is installed.
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# Default tagger. v3 models predict 448x448, BGR, float32 in [0, 255].
DEFAULT_TAGGER_REPO = "SmilingWolf/wd-swinv2-tagger-v3"
MODEL_FILE = "model.onnx"
LABELS_FILE = "selected_tags.csv"

# danbooru category ids in selected_tags.csv
CAT_GENERAL = 0
CAT_CHARACTER = 4
CAT_RATING = 9

# Tags that are kaomoji/punctuation — don't strip underscores from these.
_KAOMOJI = {
    "0_0", "(o)_(o)", "+_+", "+_-", "._.", "<o>_<o>", "<|>_<|>", "=_=", ">_<",
    "3_3", "6_9", ">_o", "@_@", "^_^", "o_o", "u_u", "x_x", "|_|", "||_||",
}


@dataclass
class TagConfig:
    repo_id: str = DEFAULT_TAGGER_REPO
    general_threshold: float = 0.35
    character_threshold: float = 0.85
    include_rating: bool = False
    replace_underscore: bool = True      # "long_hair" -> "long hair" (NL-friendly)
    max_tags: int = 0                    # 0 = unlimited
    caption_ext: str = ".txt"
    overwrite: bool = False
    providers: list[str] = field(default_factory=lambda: ["CPUExecutionProvider"])
    cache_dir: str | None = None


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".avif"}


# --------------------------------------------------------------------------
# Pure post-processing (unit-tested without a model)
# --------------------------------------------------------------------------
@dataclass
class LabelTable:
    names: list[str]
    categories: list[int]

    @property
    def general_idx(self) -> list[int]:
        return [i for i, c in enumerate(self.categories) if c == CAT_GENERAL]

    @property
    def character_idx(self) -> list[int]:
        return [i for i, c in enumerate(self.categories) if c == CAT_CHARACTER]

    @property
    def rating_idx(self) -> list[int]:
        return [i for i, c in enumerate(self.categories) if c == CAT_RATING]


def _format_name(name: str, replace_underscore: bool) -> str:
    if replace_underscore and name not in _KAOMOJI and len(name) > 3:
        return name.replace("_", " ")
    return name


def select_tags(probs: np.ndarray, labels: LabelTable, cfg: TagConfig) -> list[str]:
    """Turn a probability vector into an ordered tag list (character tags first)."""
    def pick(indices: list[int], threshold: float) -> list[tuple[str, float]]:
        hits = [(labels.names[i], float(probs[i])) for i in indices
                if probs[i] >= threshold]
        return sorted(hits, key=lambda x: x[1], reverse=True)

    chosen: list[tuple[str, float]] = []
    chosen += pick(labels.character_idx, cfg.character_threshold)
    chosen += pick(labels.general_idx, cfg.general_threshold)
    if cfg.include_rating:
        # Only the single most-likely rating tag is meaningful.
        ratings = pick(labels.rating_idx, 0.0)
        if ratings:
            chosen.append(max(ratings, key=lambda x: x[1]))

    names = [_format_name(n, cfg.replace_underscore) for n, _ in chosen]
    if cfg.max_tags > 0:
        names = names[:cfg.max_tags]
    return names


# --------------------------------------------------------------------------
# The tagger
# --------------------------------------------------------------------------
class WD14Tagger:
    def __init__(self, cfg: TagConfig):
        self.cfg = cfg
        self._session = None
        self._labels: LabelTable | None = None
        self._input_size = 448
        self._input_name = "input"

    def _load(self) -> None:
        if self._session is not None:
            return
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ImportError(
                "onnxruntime is required for auto-tagging. Install it with "
                "`pip install onnxruntime` (it's a base dependency, so a fresh "
                "`pip install -e .` also covers it). In the Docker container, run "
                "`pip install onnxruntime` in the venv, or set ANIMA_REINSTALL=1 "
                "and restart."
            ) from exc
        from huggingface_hub import hf_hub_download

        model_path = hf_hub_download(self.cfg.repo_id, MODEL_FILE,
                                     cache_dir=self.cfg.cache_dir)
        labels_path = hf_hub_download(self.cfg.repo_id, LABELS_FILE,
                                      cache_dir=self.cfg.cache_dir)
        self._session = ort.InferenceSession(model_path, providers=self.cfg.providers)
        inp = self._session.get_inputs()[0]
        self._input_name = inp.name
        # Input shape is (N, H, W, C); pull the spatial size.
        self._input_size = int(inp.shape[1]) if isinstance(inp.shape[1], int) else 448
        self._labels = _read_labels(labels_path)
        logger.info("Loaded tagger %s (size=%d, %d tags, providers=%s)",
                    self.cfg.repo_id, self._input_size, len(self._labels.names),
                    self._session.get_providers())

    def _preprocess(self, image: Image.Image) -> np.ndarray:
        # Flatten transparency onto white, pad to square, resize, RGB->BGR, 0..255.
        image = image.convert("RGBA")
        bg = Image.new("RGBA", image.size, (255, 255, 255, 255))
        bg.alpha_composite(image)
        rgb = bg.convert("RGB")
        w, h = rgb.size
        m = max(w, h)
        square = Image.new("RGB", (m, m), (255, 255, 255))
        square.paste(rgb, ((m - w) // 2, (m - h) // 2))
        square = square.resize((self._input_size, self._input_size), Image.BICUBIC)
        arr = np.asarray(square, dtype=np.float32)[:, :, ::-1]  # BGR
        return np.ascontiguousarray(arr)

    def tag_image(self, image: Image.Image) -> list[str]:
        self._load()
        batch = self._preprocess(image)[None, ...]
        probs = self._session.run(None, {self._input_name: batch})[0][0]
        return select_tags(np.asarray(probs), self._labels, self.cfg)

    def tag_path(self, path: Path) -> list[str]:
        with Image.open(path) as im:
            return self.tag_image(im)


def _read_labels(csv_path: str) -> LabelTable:
    names: list[str] = []
    categories: list[int] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            names.append(row["name"])
            categories.append(int(row["category"]))
    return LabelTable(names=names, categories=categories)


def tag_directory(image_dir: str | Path, cfg: TagConfig | None = None,
                  progress=None) -> dict[str, int]:
    """Tag every image in ``image_dir``, writing ``<image><caption_ext>`` sidecars.

    ``progress(done, total, filename)`` is called per image if provided.
    Returns a small summary dict.
    """
    cfg = cfg or TagConfig()
    image_dir = Path(image_dir)
    if not image_dir.is_dir():
        raise FileNotFoundError(f"image_dir does not exist: {image_dir}")

    images = [p for p in sorted(image_dir.rglob("*"))
              if p.suffix.lower() in IMAGE_EXTS]
    if not images:
        raise ValueError(f"No images found in {image_dir}")

    tagger = WD14Tagger(cfg)
    written = skipped = 0
    for i, path in enumerate(images, 1):
        sidecar = path.with_suffix(cfg.caption_ext)
        if sidecar.exists() and not cfg.overwrite:
            skipped += 1
        else:
            tags = tagger.tag_path(path)
            sidecar.write_text(", ".join(tags), encoding="utf-8")
            written += 1
        if progress:
            progress(i, len(images), path.name)
    summary = {"total": len(images), "written": written, "skipped": skipped}
    logger.info("Auto-tag complete: %s", summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser(prog="anima-tag",
                                description="Auto-caption a dataset with a WD14 tagger.")
    p.add_argument("image_dir", help="Folder of images to tag.")
    p.add_argument("--repo", default=DEFAULT_TAGGER_REPO, help="Tagger HF repo id.")
    p.add_argument("--general-threshold", type=float, default=0.35)
    p.add_argument("--character-threshold", type=float, default=0.85)
    p.add_argument("--include-rating", action="store_true")
    p.add_argument("--keep-underscores", action="store_true",
                   help="Keep danbooru underscores instead of converting to spaces.")
    p.add_argument("--max-tags", type=int, default=0)
    p.add_argument("--overwrite", action="store_true", help="Re-tag existing captions.")
    p.add_argument("--providers", nargs="+", default=["CPUExecutionProvider"],
                   help="ONNX Runtime execution providers, e.g. OpenVINOExecutionProvider.")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    cfg = TagConfig(
        repo_id=args.repo,
        general_threshold=args.general_threshold,
        character_threshold=args.character_threshold,
        include_rating=args.include_rating,
        replace_underscore=not args.keep_underscores,
        max_tags=args.max_tags,
        overwrite=args.overwrite,
        providers=args.providers,
    )

    def _prog(done, total, name):
        print(f"  [{done}/{total}] {name}", flush=True)

    summary = tag_directory(args.image_dir, cfg, progress=_prog)
    print(f"Done: {summary['written']} written, {summary['skipped']} skipped "
          f"of {summary['total']}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
