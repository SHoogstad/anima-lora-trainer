"""Rectified-flow (flow-matching) helpers for the Anima/Cosmos-Predict2 objective.

Flow matching trains the network to predict the velocity ``v = noise - x0`` along
the straight path ``x_t = (1 - t) * x0 + t * noise`` for ``t in [0, 1]``. This is
the objective used by Cosmos-Predict2 and most recent DiT image models.
"""

from __future__ import annotations

import torch


def sample_timesteps(batch_size: int, sampling: str, device: torch.device,
                     logit_mean: float = 0.0, logit_std: float = 1.0) -> torch.Tensor:
    """Sample t in (0, 1). ``logit_normal`` biases toward mid timesteps (SD3/Flux style)."""
    if sampling == "uniform":
        t = torch.rand(batch_size, device=device)
    elif sampling == "logit_normal":
        normal = torch.randn(batch_size, device=device) * logit_std + logit_mean
        t = torch.sigmoid(normal)
    else:
        raise ValueError(f"Unknown timestep_sampling {sampling!r}")
    # Avoid exact endpoints for numerical stability.
    return t.clamp(1e-4, 1.0 - 1e-4)


def interpolate(x0: torch.Tensor, noise: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """x_t = (1 - t) * x0 + t * noise, broadcasting t over the sample dims."""
    t_b = t.view(-1, *([1] * (x0.dim() - 1)))
    return (1.0 - t_b) * x0 + t_b * noise


def target_velocity(x0: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
    """The flow-matching regression target."""
    return noise - x0


def flow_loss(model_pred: torch.Tensor, target: torch.Tensor,
              weighting: torch.Tensor | None = None) -> torch.Tensor:
    """Mean-squared error between predicted and target velocity, optional per-sample weight."""
    loss = (model_pred.float() - target.float()) ** 2
    loss = loss.mean(dim=list(range(1, loss.dim())))  # per-sample
    if weighting is not None:
        loss = loss * weighting.to(loss.device)
    return loss.mean()
