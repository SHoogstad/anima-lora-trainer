# Anima LoRA Trainer

A [kohya_ss](https://github.com/bmaltais/kohya_ss)-style **LoRA trainer + Gradio web UI** for the
**Anima** model — but built for **Intel GPUs (XPU)** as a first-class target, with
**CUDA** and **CPU** support too.

> **Intel GPU note.** `intel-extension-for-pytorch` (IPEX) is being discontinued.
> This project uses PyTorch's **native `torch.xpu` backend** (upstreamed in
> **PyTorch ≥ 2.8**) instead — **no IPEX required or imported.**

## What is Anima?

[`circlestone-labs/Anima`](https://huggingface.co/circlestone-labs/Anima) (CircleStone Labs / Comfy Org,
released May 2026) is a **2B-parameter text-to-image DiT** for anime/illustration. Unlike SDXL-based
anime models (Illustrious/NoobAI), Anima descends from **NVIDIA Cosmos-Predict2**:

| Component | What it is | diffusers class |
|-----------|------------|-----------------|
| DiT | Cosmos-Predict2 transformer, 28 blocks | `CosmosTransformer3DModel` |
| Text encoder | Qwen-3 0.6B | `transformers` `AutoModel` |
| VAE | Qwen-Image VAE | `AutoencoderKLQwenImage` |

LoRA is injected into the DiT attention projections (`to_q/to_k/to_v/to_out.0`) of the
self-attention (`attn1`) and cross-attention (`attn2`) blocks. Training uses a
**rectified-flow (flow-matching)** objective, matching the Cosmos-Predict2 lineage.

## Install

Install the **correct PyTorch build for your hardware first**, then this package. The
wheel index differs per backend, which is why torch is not pinned in `pyproject.toml`.

### Intel GPU (Arc / Battlemage / Lunar Lake iGPU / Data Center Max)
```bash
pip install "torch>=2.8" --index-url https://download.pytorch.org/whl/xpu
pip install -e .
```
On Linux you may also need the level-zero / oneAPI runtime. On Windows the Arc driver
ships the runtime — just install the wheel.

### NVIDIA (CUDA)
```bash
pip install "torch>=2.8" --index-url https://download.pytorch.org/whl/cu124
pip install -e ".[cuda-extras]"   # adds bitsandbytes for 8-bit Adam
```

### CPU (smoke-testing only)
```bash
pip install "torch>=2.8" --index-url https://download.pytorch.org/whl/cpu
pip install -e .
```

Verify the backend was detected:
```bash
anima-train --env
```

## Use

### Web UI (Gradio)
```bash
anima-trainer                 # http://127.0.0.1:7860
anima-trainer --port 8000 --share
```
Set your image folder, trigger word, rank, backend, etc., then **Start training**.
The status panel streams live step / loss / it/s; weights auto-download from Hugging
Face on first run.

### CLI
```bash
anima-train --write-default my_run.toml   # generate a config
#   edit my_run.toml -> set [dataset].image_dir
anima-train my_run.toml                    # train
```

### Dataset layout
A folder of images, each with an optional `.txt` caption sidecar:
```
my_dataset/
├── 001.png
├── 001.txt        # "1girl, silver hair, ..." (natural language also works)
├── 002.jpg
└── 002.txt
```
Aspect-ratio **bucketing** (512–1536, Anima's supported range) and on-disk **latent +
text-embedding caching** are on by default to keep VRAM and per-step cost low.

## Output

LoRAs are written to `outputs/<name>.safetensors` with metadata (base model, rank,
alpha, step, backend), in the PEFT/diffusers convention so they load in ComfyUI and
other Anima-aware tools.

## Project layout

```
src/anima_trainer/
├── device.py     # XPU/CUDA/CPU abstraction — autocast, AMP, memory, seeding (no IPEX)
├── config.py     # typed TrainConfig with TOML load/save
├── model.py      # HF download + load DiT / Qwen-3 TE / Qwen-Image VAE
├── encoders.py   # the only model-specific VAE/text-encode forward passes
├── lora.py       # PEFT LoRA injection + safetensors export
├── dataset.py    # bucketing, image loading, latent/text caching
├── flow.py       # rectified-flow timestep sampling + loss
├── train.py      # the training loop (CLI + UI driver)
├── webui.py      # minimal Gradio control panel
└── cli.py        # `anima-train`
```

## Tests
```bash
pip install pytest
pytest                # device, flow, config, and LoRA-target logic (run on CPU)
```

## Status & caveats (honest)

This is a **runnable foundation**, validated on CPU for everything that doesn't need
the weights:

- ✅ **Verified on CPU:** device abstraction (incl. XPU/CUDA code paths), flow-matching
  math, config round-trip, LoRA target-module selection, bucketing.
- ⚠️ **Needs validation against the real weights & a GPU:** Anima is one month old, so
  the exact single-file loading, the DiT forward signature, and the VAE/text-encoder
  forward passes may need tuning. Every such model-specific seam is deliberately
  isolated in **`model.py`** and **`encoders.py`** (and the DiT call in
  `train.py:_predict_velocity`) — those are the only places you should need to touch.
  Loaders fail **loudly with guidance** rather than silently building the wrong graph.

If a loader raises `NotImplementedError`, upgrade `diffusers`/`transformers` (Anima
support is actively landing) or convert the checkpoint to a diffusers folder.
