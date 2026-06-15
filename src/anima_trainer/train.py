"""The training loop: ties device, model, LoRA, dataset, and the flow-matching loss together.

Designed to be driven either from the CLI or the Gradio UI. Progress is reported
through a callback and the loop can be cooperatively stopped via ``stop_event``.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import torch
from torch.utils.data import DataLoader

from . import flow
from .config import TrainConfig
from .dataset import AnimaDataset, BucketBatchSampler, collate, discover_samples
from .device import Device
from .lora import inject_lora, save_lora, trainable_parameters
from .model import ModelBundle, load_model

logger = logging.getLogger(__name__)


@dataclass
class TrainState:
    step: int = 0
    loss: float = 0.0
    lr: float = 0.0
    ema_loss: float = 0.0
    started_at: float = field(default_factory=time.time)
    last_saved: str = ""
    finished: bool = False
    error: str = ""

    def steps_per_sec(self) -> float:
        elapsed = max(time.time() - self.started_at, 1e-6)
        return self.step / elapsed


ProgressFn = Callable[[TrainState], None]


def _build_optimizer(params, cfg: TrainConfig, device: Device):
    name = cfg.optim.optimizer.lower()
    if name == "adamw8bit":
        if device.backend != "cuda":
            logger.warning("adamw8bit is CUDA-only; falling back to adamw on %s.",
                           device.backend)
        else:
            import bitsandbytes as bnb
            return bnb.optim.AdamW8bit(params, lr=cfg.optim.learning_rate,
                                       weight_decay=cfg.optim.weight_decay)
    if name == "adafactor":
        from transformers.optimization import Adafactor
        return Adafactor(params, lr=cfg.optim.learning_rate, relative_step=False,
                         scale_parameter=False)
    return torch.optim.AdamW(params, lr=cfg.optim.learning_rate,
                             weight_decay=cfg.optim.weight_decay)


def _build_scheduler(optimizer, cfg: TrainConfig):
    from diffusers.optimization import get_scheduler

    return get_scheduler(
        cfg.optim.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=cfg.optim.warmup_steps,
        num_training_steps=cfg.max_train_steps,
    )


def _predict_velocity(transformer, noisy_latent: torch.Tensor, timestep: torch.Tensor,
                      embeds: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    """Call the Cosmos DiT and return its velocity prediction.

    Cosmos works on 5D video latents (B, C, T, H, W); Anima images use T=1. This is
    the single seam where the DiT forward signature is exercised — adjust here if the
    real Anima transformer expects different kwargs.
    """
    added_5d = False
    if noisy_latent.dim() == 4:
        noisy_latent = noisy_latent.unsqueeze(2)  # add temporal frame
        added_5d = True

    # Cosmos-Predict2 timesteps are scaled to [0, 1000].
    timestep_scaled = timestep * 1000.0
    out = transformer(
        hidden_states=noisy_latent,
        timestep=timestep_scaled,
        encoder_hidden_states=embeds,
        attention_mask=mask,
        return_dict=False,
    )
    pred = out[0] if isinstance(out, (tuple, list)) else out.sample
    if added_5d and pred.dim() == 5:
        pred = pred[:, :, 0]
    return pred


def train(cfg: TrainConfig, progress: ProgressFn | None = None,
          stop_event: threading.Event | None = None) -> TrainState:
    state = TrainState()
    stop_event = stop_event or threading.Event()

    try:
        device = Device.resolve(cfg.backend, cfg.device_index)
        dtype = device.resolve_dtype(cfg.dtype)
        device.manual_seed(cfg.seed)
        logger.info("Training on %s (%s), dtype=%s", device, device.device_name(), dtype)

        # ---- model + LoRA -------------------------------------------------
        bundle: ModelBundle = load_model(cfg.model, device, dtype)
        if cfg.gradient_checkpointing and hasattr(bundle.transformer,
                                                  "enable_gradient_checkpointing"):
            bundle.transformer.enable_gradient_checkpointing()
        peft_model = inject_lora(bundle.transformer, cfg.lora)
        peft_model.train()

        # ---- data ---------------------------------------------------------
        samples = discover_samples(cfg.dataset)
        dataset = AnimaDataset(cfg.dataset, samples)
        if cfg.dataset.cache_latents:
            logger.info("Pre-encoding latents/text embeds...")
            dataset.precompute_cache(bundle, device, dtype)

        sampler = BucketBatchSampler(samples, cfg.batch_size, shuffle=True, seed=cfg.seed)
        loader = DataLoader(dataset, batch_sampler=sampler, collate_fn=collate,
                            num_workers=0)

        # ---- optim --------------------------------------------------------
        params = trainable_parameters(peft_model)
        optimizer = _build_optimizer(params, cfg, device)
        lr_sched = _build_scheduler(optimizer, cfg)
        scaler = device.grad_scaler(enabled=(dtype == torch.float16))
        accum = max(1, cfg.optim.gradient_accumulation_steps)

        from .encoders import encode_prompts

        # ---- loop ---------------------------------------------------------
        data_iter = _infinite(loader)
        optimizer.zero_grad(set_to_none=True)
        while state.step < cfg.max_train_steps and not stop_event.is_set():
            micro_loss = 0.0
            for _ in range(accum):
                batch = next(data_iter)

                if "latent" in batch:
                    x0 = batch["latent"].to(device.torch_device, dtype=dtype)
                    embeds = batch["embeds"].to(device.torch_device, dtype=dtype)
                    mask = batch["mask"].to(device.torch_device)
                else:
                    from .encoders import encode_images_to_latents
                    x0 = encode_images_to_latents(bundle, batch["pixel_values"], dtype)
                    enc = encode_prompts(bundle, batch["captions"], dtype)
                    embeds, mask = enc["embeds"], enc["mask"]

                noise = torch.randn_like(x0)
                t = flow.sample_timesteps(x0.shape[0], cfg.timestep_sampling,
                                          device.torch_device, cfg.logit_mean, cfg.logit_std)
                noisy = flow.interpolate(x0, noise, t)
                target = flow.target_velocity(x0, noise)

                with device.autocast(dtype):
                    pred = _predict_velocity(peft_model, noisy, t, embeds, mask)
                    loss = flow.flow_loss(pred, target) / accum

                scaler.scale(loss).backward()
                micro_loss += loss.item()

            # optimizer step
            if cfg.optim.max_grad_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(params, cfg.optim.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            lr_sched.step()
            optimizer.zero_grad(set_to_none=True)

            state.step += 1
            state.loss = micro_loss
            state.ema_loss = micro_loss if state.step == 1 \
                else 0.98 * state.ema_loss + 0.02 * micro_loss
            state.lr = lr_sched.get_last_lr()[0]

            if state.step % 10 == 0:
                mem = device.memory_summary()
                logger.info("step %d/%d loss=%.4f ema=%.4f lr=%.2e %.2f it/s %s",
                            state.step, cfg.max_train_steps, state.loss, state.ema_loss,
                            state.lr, state.steps_per_sec(),
                            f"{mem.get('allocated_gib', 0):.1f}GiB" if mem else "")
            if progress:
                progress(state)

            if cfg.save_every_steps and state.step % cfg.save_every_steps == 0:
                _save(peft_model, cfg, state, device, dtype)

        # ---- final save ---------------------------------------------------
        _save(peft_model, cfg, state, device, dtype, final=True)
        state.finished = True
        return state

    except Exception as exc:  # noqa: BLE001 - surface to the UI
        logger.exception("Training failed")
        state.error = f"{type(exc).__name__}: {exc}"
        state.finished = True
        if progress:
            progress(state)
        return state


def _save(peft_model, cfg: TrainConfig, state: TrainState, device: Device,
          dtype: torch.dtype, final: bool = False) -> None:
    name = cfg.output_name if final else f"{cfg.output_name}-step{state.step:06d}"
    meta = {
        "step": state.step,
        "base_model": cfg.model.repo_id,
        "rank": cfg.lora.rank,
        "alpha": cfg.lora.alpha,
        "backend": device.backend,
        "dtype": cfg.dtype,
    }
    path = save_lora(peft_model, cfg.output_dir, name, meta)
    state.last_saved = str(path)


def _infinite(loader):
    while True:
        for batch in loader:
            yield batch
