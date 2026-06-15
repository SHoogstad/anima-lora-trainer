"""Tests for the flow-matching math."""

import torch

from anima_trainer import flow


def test_sample_timesteps_in_range():
    for mode in ("uniform", "logit_normal"):
        t = flow.sample_timesteps(64, mode, torch.device("cpu"))
        assert t.shape == (64,)
        assert (t > 0).all() and (t < 1).all()


def test_interpolate_endpoints():
    x0 = torch.zeros(4, 3, 8, 8)
    noise = torch.ones(4, 3, 8, 8)
    near0 = flow.interpolate(x0, noise, torch.full((4,), 1e-4))
    near1 = flow.interpolate(x0, noise, torch.full((4,), 1 - 1e-4))
    assert near0.mean() < 0.01
    assert near1.mean() > 0.99


def test_target_velocity():
    x0 = torch.randn(2, 3, 4, 4)
    noise = torch.randn(2, 3, 4, 4)
    assert torch.allclose(flow.target_velocity(x0, noise), noise - x0)


def test_flow_loss_zero_when_perfect():
    pred = torch.randn(3, 16, 8, 8)
    assert flow.flow_loss(pred, pred).item() < 1e-9
