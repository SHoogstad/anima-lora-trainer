"""Pure-PyTorch replacements for the ComfyUI/comfy_kitchen couplings used by the
vendored Anima/Cosmos-Predict2 DiT.

These let the vendored modeling code run standalone (CPU/CUDA/XPU) without the
ComfyUI runtime or the compiled comfy_kitchen kernels. Each function mirrors the
reference semantics exactly so the loaded weights behave identically.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def optimized_attention(q, k, v, heads, mask=None, skip_reshape=False,
                        skip_output_reshape=False, transformer_options=None, **kwargs):
    """SDPA stand-in for comfy.ldm.modules.attention.optimized_attention.

    With ``skip_reshape=True`` (the only path the DiT uses) q/k/v arrive as
    (B, heads, S, D). Returns (B, S, heads*D) unless ``skip_output_reshape``.
    """
    if skip_reshape:
        b = q.shape[0]
    else:
        b, s, _ = q.shape
        q = q.view(b, s, heads, -1).transpose(1, 2)
        k = k.view(b, -1, heads, q.shape[-1]).transpose(1, 2)
        v = v.view(b, -1, heads, q.shape[-1]).transpose(1, 2)

    out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    if skip_output_reshape:
        return out
    return out.transpose(1, 2).reshape(b, -1, heads * out.shape[-1])


def apply_rope_split_half1(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    # Verbatim semantics of comfy_kitchen.backends.eager.rope (Apache-2.0):
    # split the head dim into two halves and apply the 2x2 rotation matrices.
    t_ = x.reshape(*x.shape[:-1], 2, -1).movedim(-2, -1).unsqueeze(-2).to(freqs_cis.dtype)
    t_out = freqs_cis[..., 0] * t_[..., 0] + freqs_cis[..., 1] * t_[..., 1]
    return t_out.movedim(-1, -2).reshape(*x.shape).type_as(x)


def apply_rope_split_half(xq, xk, freqs_cis):
    return apply_rope_split_half1(xq, freqs_cis), apply_rope_split_half1(xk, freqs_cis)


def pad_to_patch_size(img, patch_size=(2, 2), padding_mode="circular"):
    """Pad spatial/temporal dims up to a multiple of the patch size.

    Copied from comfy.ldm.common_dit (GPL-3.0). For latents already divisible by
    the patch size (the common case) this is a no-op.
    """
    if padding_mode == "circular" and (torch.jit.is_tracing() or torch.jit.is_scripting()):
        padding_mode = "reflect"
    pad = ()
    for i in range(img.ndim - 2):
        pad = (0, (patch_size[i] - img.shape[i + 2] % patch_size[i]) % patch_size[i]) + pad
    return torch.nn.functional.pad(img, pad, mode=padding_mode)


# ``ck`` namespace shim so vendored ``comfy.quant_ops.ck.apply_rope_split_half`` works.
class _CK:
    apply_rope_split_half = staticmethod(apply_rope_split_half)


ck = _CK()
