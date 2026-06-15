"""Minimal ``operations`` namespace expected by the vendored comfy modeling code.

ComfyUI passes an ``operations`` module providing layer classes with lazy weight
init and dtype casting. For a trainer we just need plain ``nn`` layers that accept
the same constructor kwargs (device/dtype/eps/elementwise_affine), plus an
``Embedding`` whose ``forward`` accepts the ``out_dtype`` kwarg the adapter uses.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class Embedding(nn.Embedding):
    def forward(self, input, out_dtype=None):  # noqa: A002 - mirror comfy signature
        weight = self.weight if out_dtype is None else self.weight.to(out_dtype)
        return F.embedding(input, weight, self.padding_idx, self.max_norm,
                           self.norm_type, self.scale_grad_by_freq, self.sparse)


class _Ops:
    """Namespace object passed as ``operations=`` to the vendored modules."""

    Linear = nn.Linear
    LayerNorm = nn.LayerNorm
    RMSNorm = nn.RMSNorm  # PyTorch >= 2.4
    Embedding = Embedding


ops = _Ops()
