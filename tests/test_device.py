"""Tests for the backend-agnostic device layer (run on CPU CI)."""

import contextlib

import torch

from anima_trainer.device import (Device, detect_backend, environment_report,
                                  list_devices)


def test_detect_cpu_explicit():
    assert detect_backend("cpu") == "cpu"


def test_detect_auto_returns_valid():
    assert detect_backend("auto") in ("cuda", "xpu", "cpu")


def test_resolve_and_torch_device():
    dev = Device.resolve("cpu")
    assert dev.backend == "cpu"
    assert dev.torch_device == torch.device("cpu")
    assert dev.is_gpu is False


def test_dtype_resolution_downgrades_on_cpu():
    dev = Device.resolve("cpu")
    # fp16 is unsupported on CPU -> downgrade to fp32.
    assert dev.resolve_dtype("fp16") == torch.float32
    assert dev.resolve_dtype("fp32") == torch.float32


def test_autocast_is_nullcontext_on_cpu():
    dev = Device.resolve("cpu")
    ctx = dev.autocast(torch.bfloat16)
    assert isinstance(ctx, contextlib.nullcontext)


def test_grad_scaler_disabled_on_cpu():
    dev = Device.resolve("cpu")
    scaler = dev.grad_scaler(enabled=True)
    assert scaler.is_enabled() is False


def test_seed_and_reports_do_not_raise():
    dev = Device.resolve("cpu")
    dev.manual_seed(123)
    assert isinstance(environment_report(), str)
    assert any(d["backend"] == "cpu" for d in list_devices())
