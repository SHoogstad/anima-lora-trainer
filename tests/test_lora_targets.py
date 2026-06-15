"""Tests for LoRA target-module selection (pure logic, no peft needed)."""

import torch

from anima_trainer.config import LoRAConfig
from anima_trainer.lora import resolve_target_modules


class _Attn(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.to_q = torch.nn.Linear(8, 8)
        self.to_k = torch.nn.Linear(8, 8)
        self.to_v = torch.nn.Linear(8, 8)
        self.to_out = torch.nn.ModuleList([torch.nn.Linear(8, 8)])


class _Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.attn1 = _Attn()   # self-attention
        self.attn2 = _Attn()   # cross-attention
        self.ff = torch.nn.Linear(8, 8)


class _DiT(torch.nn.Module):
    def __init__(self, n=3):
        super().__init__()
        self.transformer_blocks = torch.nn.ModuleList([_Block() for _ in range(n)])


def test_default_targets_cover_all_attention():
    dit = _DiT(n=2)
    targets = resolve_target_modules(dit, LoRAConfig())
    # 2 blocks x 2 attn x 4 projections (to_q,to_k,to_v,to_out.0) = 16
    assert len(targets) == 16
    assert all("ff" not in t for t in targets)


def test_cross_attn_only():
    dit = _DiT(n=2)
    cfg = LoRAConfig(train_self_attn=False, train_cross_attn=True)
    targets = resolve_target_modules(dit, cfg)
    assert all(".attn2." in t for t in targets)
    assert len(targets) == 8


def test_block_index_restriction():
    dit = _DiT(n=4)
    cfg = LoRAConfig(block_indices=[0, 2])
    targets = resolve_target_modules(dit, cfg)
    assert {t.split(".")[1] for t in targets} == {"0", "2"}


def test_no_match_raises():
    dit = _DiT(n=1)
    cfg = LoRAConfig(target_modules=["does_not_exist"])
    try:
        resolve_target_modules(dit, cfg)
    except ValueError:
        return
    raise AssertionError("expected ValueError for unmatched targets")
