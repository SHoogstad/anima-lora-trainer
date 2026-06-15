"""Minimal Gradio control panel for the Anima LoRA trainer.

Scope (per project decision): a thin control panel over the trainer core. It lets
you set the key knobs, shows the detected backend/devices, runs training in a
background thread, and streams live loss/step/VRAM status. Heavy lifting lives in
the trainer package; this file is only glue.
"""

from __future__ import annotations

import logging
import threading

import gradio as gr

from .config import (DEFAULT_LORA_TARGETS, DatasetConfig, LoRAConfig, ModelConfig,
                     OptimConfig, TrainConfig)
from .device import environment_report, list_devices
from .train import TrainState, train

logger = logging.getLogger(__name__)


class TrainerSession:
    """Owns the background training thread and the latest TrainState."""

    def __init__(self) -> None:
        self.thread: threading.Thread | None = None
        self.stop_event = threading.Event()
        self.state = TrainState()
        self._lock = threading.Lock()

    @property
    def running(self) -> bool:
        return self.thread is not None and self.thread.is_alive()

    def start(self, cfg: TrainConfig) -> str:
        if self.running:
            return "Training already running."
        self.stop_event = threading.Event()
        self.state = TrainState()

        def _progress(s: TrainState) -> None:
            with self._lock:
                self.state = s

        def _run() -> None:
            train(cfg, progress=_progress, stop_event=self.stop_event)

        self.thread = threading.Thread(target=_run, daemon=True)
        self.thread.start()
        return "Training started."

    def stop(self) -> str:
        if not self.running:
            return "Nothing to stop."
        self.stop_event.set()
        return "Stop requested; will halt after the current step."

    def status(self) -> str:
        with self._lock:
            s = self.state
        if s.error:
            return f"❌ Error: {s.error}"
        if not self.running and s.step == 0:
            return "Idle."
        head = "✅ Finished" if s.finished and not self.running else "🟢 Running"
        lines = [
            f"{head} — step {s.step}",
            f"loss {s.loss:.4f}  (ema {s.ema_loss:.4f})",
            f"lr {s.lr:.2e}   {s.steps_per_sec():.2f} it/s",
        ]
        if s.last_saved:
            lines.append(f"last save: {s.last_saved}")
        return "\n".join(lines)


def _cfg_from_inputs(**kw) -> TrainConfig:
    return TrainConfig(
        model=ModelConfig(repo_id=kw["repo_id"], cache_dir=kw["cache_dir"] or None),
        lora=LoRAConfig(
            rank=int(kw["rank"]), alpha=int(kw["alpha"]),
            target_modules=[t.strip() for t in kw["targets"].split(",") if t.strip()],
        ),
        dataset=DatasetConfig(
            image_dir=kw["image_dir"], trigger_word=kw["trigger"],
            resolution=int(kw["resolution"]), repeats=int(kw["repeats"]),
            cache_latents=kw["cache_latents"],
        ),
        optim=OptimConfig(
            learning_rate=float(kw["lr"]), optimizer=kw["optimizer"],
            lr_scheduler=kw["scheduler"], warmup_steps=int(kw["warmup"]),
        ),
        backend=kw["backend"], dtype=kw["dtype"], seed=int(kw["seed"]),
        batch_size=int(kw["batch_size"]), max_train_steps=int(kw["steps"]),
        save_every_steps=int(kw["save_every"]), output_dir=kw["output_dir"],
        output_name=kw["output_name"],
    )


def build_ui() -> gr.Blocks:
    session = TrainerSession()
    devices = [f"{d['backend']}:{d['index']}  {d['name']}" for d in list_devices()]

    with gr.Blocks(title="Anima LoRA Trainer") as demo:
        gr.Markdown("# Anima LoRA Trainer\n"
                    "kohya_ss-style LoRA training for the **Anima** DiT — "
                    "native Intel **XPU** and **CUDA** (no IPEX).")

        with gr.Accordion("Detected backends / devices", open=True):
            gr.Code(environment_report(), label="environment")
            gr.Markdown("Available: " + (", ".join(devices) if devices else "none"))

        with gr.Row():
            with gr.Column():
                gr.Markdown("### Model")
                repo_id = gr.Textbox(ModelConfig.repo_id, label="HF repo id")
                cache_dir = gr.Textbox("", label="Weights cache dir (blank = HF default)")

                gr.Markdown("### Dataset")
                image_dir = gr.Textbox("", label="Image folder",
                                       placeholder=r"E:\datasets\my_character")
                trigger = gr.Textbox("", label="Trigger word (optional)")
                resolution = gr.Slider(512, 1536, value=1024, step=64, label="Resolution")
                repeats = gr.Number(1, label="Repeats", precision=0)
                cache_latents = gr.Checkbox(True, label="Cache latents + text embeds")

                gr.Markdown("### LoRA")
                rank = gr.Slider(2, 128, value=16, step=2, label="Rank (dim)")
                alpha = gr.Slider(1, 128, value=16, step=1, label="Alpha")
                targets = gr.Textbox(", ".join(DEFAULT_LORA_TARGETS),
                                     label="Target modules")

            with gr.Column():
                gr.Markdown("### Compute")
                backend = gr.Radio(["auto", "cuda", "xpu", "cpu"], value="auto",
                                   label="Backend")
                dtype = gr.Radio(["bf16", "fp16", "fp32"], value="bf16", label="Precision")
                seed = gr.Number(42, label="Seed", precision=0)

                gr.Markdown("### Optimizer / schedule")
                lr = gr.Textbox("1e-4", label="Learning rate")
                optimizer = gr.Dropdown(["adamw", "adamw8bit", "adafactor"],
                                        value="adamw", label="Optimizer")
                scheduler = gr.Dropdown(
                    ["constant", "cosine", "linear", "constant_with_warmup"],
                    value="constant", label="LR scheduler")
                warmup = gr.Number(0, label="Warmup steps", precision=0)
                batch_size = gr.Number(1, label="Batch size", precision=0)
                steps = gr.Number(2000, label="Max train steps", precision=0)
                save_every = gr.Number(250, label="Save every (steps)", precision=0)

                gr.Markdown("### Output")
                output_dir = gr.Textbox("outputs", label="Output dir")
                output_name = gr.Textbox("anima_lora", label="Output name")

        with gr.Accordion("Auto-tag dataset (WD14)", open=False):
            gr.Markdown(
                "Generate danbooru-style `.txt` captions for the **Image folder** above "
                "using a WD14 tagger (ONNX, runs on CPU). Existing captions are kept "
                "unless *Overwrite* is checked.")
            with gr.Row():
                tagger_repo = gr.Textbox("SmilingWolf/wd-swinv2-tagger-v3",
                                         label="Tagger HF repo")
                tag_providers = gr.Dropdown(
                    ["CPUExecutionProvider", "OpenVINOExecutionProvider",
                     "DmlExecutionProvider", "CUDAExecutionProvider"],
                    value="CPUExecutionProvider", label="ONNX provider")
            with gr.Row():
                general_thr = gr.Slider(0.1, 0.9, value=0.35, step=0.05,
                                        label="General threshold")
                character_thr = gr.Slider(0.1, 0.95, value=0.85, step=0.05,
                                          label="Character threshold")
                max_tags = gr.Number(0, label="Max tags (0 = all)", precision=0)
            with gr.Row():
                keep_underscores = gr.Checkbox(False, label="Keep underscores")
                include_rating = gr.Checkbox(False, label="Include rating tag")
                tag_overwrite = gr.Checkbox(False, label="Overwrite existing")
            tag_btn = gr.Button("Auto-tag now")
            tag_status = gr.Textbox("", label="Tagging status", lines=3, interactive=False)

        def _autotag(image_dir, tagger_repo, tag_providers, general_thr, character_thr,
                     max_tags, keep_underscores, include_rating, tag_overwrite):
            if not image_dir:
                yield "Set an Image folder first."
                return
            from .autotag import TagConfig, tag_directory
            cfg = TagConfig(
                repo_id=tagger_repo, providers=[tag_providers],
                general_threshold=float(general_thr),
                character_threshold=float(character_thr),
                max_tags=int(max_tags), replace_underscore=not keep_underscores,
                include_rating=include_rating, overwrite=tag_overwrite)
            yield f"Loading tagger {tagger_repo} (first run downloads it)…"
            state = {"msg": "starting…"}

            def _p(done, total, name):
                state["msg"] = f"[{done}/{total}] {name}"

            # Run in a thread so we can stream progress back to the UI.
            result: dict = {}

            def _run():
                try:
                    result["summary"] = tag_directory(image_dir, cfg, progress=_p)
                except Exception as exc:  # noqa: BLE001
                    result["error"] = f"{type(exc).__name__}: {exc}"

            t = threading.Thread(target=_run, daemon=True)
            t.start()
            while t.is_alive():
                t.join(0.5)
                yield state["msg"]
            if "error" in result:
                yield f"❌ {result['error']}"
            else:
                s = result["summary"]
                yield (f"✅ Tagged {s['written']} image(s), skipped {s['skipped']} "
                       f"of {s['total']}.")

        tag_btn.click(_autotag,
                      inputs=[image_dir, tagger_repo, tag_providers, general_thr,
                              character_thr, max_tags, keep_underscores, include_rating,
                              tag_overwrite],
                      outputs=tag_status)

        with gr.Row():
            start_btn = gr.Button("Start training", variant="primary")
            stop_btn = gr.Button("Stop")
        status = gr.Textbox("Idle.", label="Status", lines=5, interactive=False)
        timer = gr.Timer(2.0)

        inputs = [repo_id, cache_dir, image_dir, trigger, resolution, repeats,
                  cache_latents, rank, alpha, targets, backend, dtype, seed, lr,
                  optimizer, scheduler, warmup, batch_size, steps, save_every,
                  output_dir, output_name]

        def _start(repo_id, cache_dir, image_dir, trigger, resolution, repeats,
                   cache_latents, rank, alpha, targets, backend, dtype, seed, lr,
                   optimizer, scheduler, warmup, batch_size, steps, save_every,
                   output_dir, output_name):
            if not image_dir:
                return "Please set an image folder first."
            cfg = _cfg_from_inputs(
                repo_id=repo_id, cache_dir=cache_dir, image_dir=image_dir,
                trigger=trigger, resolution=resolution, repeats=repeats,
                cache_latents=cache_latents, rank=rank, alpha=alpha, targets=targets,
                backend=backend, dtype=dtype, seed=seed, lr=lr, optimizer=optimizer,
                scheduler=scheduler, warmup=warmup, batch_size=batch_size, steps=steps,
                save_every=save_every, output_dir=output_dir, output_name=output_name)
            return session.start(cfg)

        start_btn.click(_start, inputs=inputs, outputs=status)
        stop_btn.click(lambda: session.stop(), outputs=status)
        timer.tick(lambda: session.status(), outputs=status)

    return demo


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Launch the Anima LoRA Trainer web UI.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="Create a public Gradio link.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    build_ui().queue().launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
