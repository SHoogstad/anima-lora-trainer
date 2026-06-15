"""Tests for the vendored native Anima DiT.

The forward smokes use tiny configs (no weights needed) to exercise the vendored
code path, RoPE, attention shims, AdaLN, and the llm_adapter. The architecture
check runs only if the real checkpoint is cached locally.
"""

import json
import struct
from pathlib import Path

import pytest
import torch

from anima_trainer.vendor.comfy_anima import build_anima_dit
from anima_trainer.vendor.comfy_anima.anima_model import LLMAdapter
from anima_trainer.vendor.comfy_anima.ops import ops
from anima_trainer.vendor.comfy_anima.predict2 import MiniTrainDIT


def test_base_dit_forward_shapes():
    torch.manual_seed(0)
    dit = MiniTrainDIT(
        max_img_h=64, max_img_w=64, max_frames=1, in_channels=16, out_channels=16,
        patch_spatial=2, patch_temporal=1, concat_padding_mask=True, model_channels=64,
        num_blocks=2, num_heads=4, mlp_ratio=4.0, crossattn_emb_channels=32,
        pos_emb_cls="rope3d", use_adaln_lora=True, adaln_lora_dim=16,
        extra_per_block_abs_pos_emb=False, rope_enable_fps_modulation=False, operations=ops)
    x = torch.randn(1, 16, 1, 8, 8)
    out = dit(x, torch.tensor([0.5]), torch.randn(1, 5, 32))
    assert tuple(out.shape) == (1, 16, 1, 8, 8)
    assert torch.isfinite(out).all()


def test_llm_adapter_forward_shapes():
    torch.manual_seed(0)
    adapter = LLMAdapter(source_dim=16, target_dim=16, model_dim=16, num_layers=2,
                         num_heads=2, use_self_attn=True, operations=ops)
    out = adapter(torch.randn(1, 5, 16), torch.randint(0, 32128, (1, 7)))
    assert tuple(out.shape) == (1, 7, 16)
    assert torch.isfinite(out).all()


def _find_cached_dit() -> Path | None:
    matches = list(Path("hf_home").rglob("anima-base-v1.0.safetensors")) if Path(
        "hf_home").exists() else []
    return matches[0] if matches else None


def test_architecture_matches_real_checkpoint():
    """Strict-load check: every param key+shape matches the real checkpoint."""
    ckpt_path = _find_cached_dit()
    if ckpt_path is None:
        pytest.skip("Anima checkpoint not cached; run a download first.")

    with open(ckpt_path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
    hdr.pop("__metadata__", None)
    ckpt = {(k[4:] if k.startswith("net.") else k): tuple(v["shape"])
            for k, v in hdr.items()}

    model = build_anima_dit(device=torch.device("meta"))
    msd = {k: tuple(v.shape) for k, v in model.state_dict().items()}

    assert set(msd) == set(ckpt), (
        f"key mismatch: missing={set(ckpt) - set(msd)}, extra={set(msd) - set(ckpt)}")
    mismatched = {k: (msd[k], ckpt[k]) for k in msd if msd[k] != ckpt[k]}
    assert not mismatched, f"shape mismatches: {mismatched}"
