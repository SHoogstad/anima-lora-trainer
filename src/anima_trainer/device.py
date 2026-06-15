"""Backend-agnostic device handling for Intel XPU, NVIDIA CUDA, and CPU.

Design goals
------------
* **No IPEX.** Intel GPU support relies entirely on PyTorch's *native* ``torch.xpu``
  backend (upstreamed in PyTorch >= 2.8). ``intel-extension-for-pytorch`` is being
  discontinued, so we never import it.
* A single :class:`Device` object answers every backend-specific question the rest
  of the trainer needs (autocast, memory stats, cache clearing, seeding, dtype
  support), so no other module has to branch on ``cuda``/``xpu``/``cpu``.
"""

from __future__ import annotations

import contextlib
import gc
import os
from dataclasses import dataclass

import torch

_VALID_BACKENDS = ("cuda", "xpu", "cpu")


def _xpu_available() -> bool:
    # torch.xpu only exists on PyTorch builds compiled with XPU support.
    return hasattr(torch, "xpu") and torch.xpu.is_available()


def _cuda_available() -> bool:
    return torch.cuda.is_available()


def detect_backend(preferred: str | None = None) -> str:
    """Pick a backend.

    ``preferred`` may be ``"auto"`` / ``None`` (autodetect), or an explicit backend.
    Autodetect order: CUDA, then XPU, then CPU.
    """
    if preferred and preferred != "auto":
        preferred = preferred.lower()
        if preferred not in _VALID_BACKENDS:
            raise ValueError(f"Unknown backend {preferred!r}; expected one of {_VALID_BACKENDS}")
        if preferred == "cuda" and not _cuda_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")
        if preferred == "xpu" and not _xpu_available():
            raise RuntimeError(
                "XPU requested but torch.xpu is unavailable. Install a PyTorch>=2.8 "
                "XPU build (pip install torch --index-url "
                "https://download.pytorch.org/whl/xpu)."
            )
        return preferred

    if _cuda_available():
        return "cuda"
    if _xpu_available():
        return "xpu"
    return "cpu"


@dataclass(frozen=True)
class Device:
    """A resolved compute device plus backend-aware helpers."""

    backend: str          # "cuda" | "xpu" | "cpu"
    index: int = 0

    # ---- construction -----------------------------------------------------
    @classmethod
    def resolve(cls, preferred: str | None = None, index: int = 0) -> "Device":
        return cls(backend=detect_backend(preferred), index=index)

    # ---- torch interop ----------------------------------------------------
    @property
    def torch_device(self) -> torch.device:
        if self.backend == "cpu":
            return torch.device("cpu")
        return torch.device(f"{self.backend}:{self.index}")

    @property
    def type(self) -> str:
        return self.backend

    def __str__(self) -> str:  # noqa: D105
        return str(self.torch_device)

    @property
    def is_gpu(self) -> bool:
        return self.backend in ("cuda", "xpu")

    # ---- module of the active backend (torch.cuda / torch.xpu) ------------
    @property
    def _mod(self):
        if self.backend == "cuda":
            return torch.cuda
        if self.backend == "xpu":
            return torch.xpu
        return None

    # ---- dtype support ----------------------------------------------------
    def supports_bf16(self) -> bool:
        if self.backend == "cuda":
            return torch.cuda.is_bf16_supported()
        if self.backend == "xpu":
            # All Arc (Alchemist) and newer Intel GPUs support bf16 in the XPU backend.
            return True
        # CPU bf16 is functionally supported (slow) on modern torch.
        return True

    def supports_fp16(self) -> bool:
        # fp16 compute is fine on CUDA and XPU. On CPU fp16 is poorly supported, avoid it.
        return self.is_gpu

    def resolve_dtype(self, name: str) -> torch.dtype:
        """Map a config string to a torch dtype, downgrading gracefully if unsupported."""
        name = (name or "bf16").lower()
        if name in ("bf16", "bfloat16"):
            return torch.bfloat16 if self.supports_bf16() else torch.float32
        if name in ("fp16", "float16", "half"):
            return torch.float16 if self.supports_fp16() else torch.float32
        if name in ("fp32", "float32", "float"):
            return torch.float32
        raise ValueError(f"Unknown dtype {name!r}")

    # ---- autocast ---------------------------------------------------------
    def autocast(self, dtype: torch.dtype | None = None):
        """Mixed-precision context manager for the active backend.

        ``torch.autocast`` accepts ``device_type="xpu"`` natively on PyTorch>=2.8,
        so the same call works for every backend.
        """
        if dtype is None or dtype == torch.float32 or self.backend == "cpu":
            return contextlib.nullcontext()
        return torch.autocast(device_type=self.backend, dtype=dtype)

    def grad_scaler(self, enabled: bool):
        """Return a GradScaler appropriate for the backend.

        fp16 training needs loss scaling. ``torch.amp.GradScaler`` is device-aware
        from PyTorch 2.4+ and supports ``"xpu"``.
        """
        return torch.amp.GradScaler(self.backend, enabled=enabled and self.is_gpu)

    # ---- memory -----------------------------------------------------------
    def empty_cache(self) -> None:
        if self._mod is not None:
            self._mod.empty_cache()
        gc.collect()

    def memory_summary(self) -> dict[str, float]:
        """Allocated / reserved / total VRAM in GiB (best effort)."""
        if self._mod is None:
            return {}
        gib = 1024 ** 3
        out: dict[str, float] = {
            "allocated_gib": self._mod.memory_allocated(self.index) / gib,
            "reserved_gib": self._mod.memory_reserved(self.index) / gib,
        }
        with contextlib.suppress(Exception):
            props = self._mod.get_device_properties(self.index)
            out["total_gib"] = props.total_memory / gib
        return out

    def device_name(self) -> str:
        if self._mod is None:
            return "CPU"
        with contextlib.suppress(Exception):
            return self._mod.get_device_properties(self.index).name
        return self.backend.upper()

    # ---- determinism ------------------------------------------------------
    def manual_seed(self, seed: int) -> None:
        torch.manual_seed(seed)
        if self.backend == "cuda":
            torch.cuda.manual_seed_all(seed)
        elif self.backend == "xpu":
            torch.xpu.manual_seed_all(seed)


def list_devices() -> list[dict[str, str]]:
    """Enumerate available compute devices for display in the UI."""
    devices: list[dict[str, str]] = []
    if _cuda_available():
        for i in range(torch.cuda.device_count()):
            devices.append({"backend": "cuda", "index": str(i),
                            "name": torch.cuda.get_device_properties(i).name})
    if _xpu_available():
        for i in range(torch.xpu.device_count()):
            devices.append({"backend": "xpu", "index": str(i),
                            "name": torch.xpu.get_device_properties(i).name})
    devices.append({"backend": "cpu", "index": "0", "name": "CPU"})
    return devices


def environment_report() -> str:
    """Human-readable backend report, handy for the UI and bug reports."""
    lines = [f"torch {torch.__version__}"]
    lines.append(f"CUDA available: {_cuda_available()}")
    lines.append(f"XPU available:  {_xpu_available()} (native torch.xpu, no IPEX)")
    if os.environ.get("ANIMA_FORCE_BACKEND"):
        lines.append(f"ANIMA_FORCE_BACKEND={os.environ['ANIMA_FORCE_BACKEND']}")
    for d in list_devices():
        lines.append(f"  - {d['backend']}:{d['index']}  {d['name']}")
    return "\n".join(lines)
