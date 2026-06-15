# Vendored code licenses

`src/anima_trainer/vendor/comfy_anima/` contains third-party model code so the
Anima DiT can run standalone (no ComfyUI runtime). Provenance:

| File | Derived from | License |
|------|--------------|---------|
| `anima_model.py` | [ComfyUI](https://github.com/comfyanonymous/ComfyUI) `comfy/ldm/anima/model.py` | **GPL-3.0** |
| `predict2.py` | ComfyUI `comfy/ldm/cosmos/predict2.py` (orig. [NVIDIA cosmos-predict2](https://github.com/nvidia-cosmos/cosmos-predict2)) | GPL-3.0 / Apache-2.0 |
| `position_embedding.py` | NVIDIA cosmos-predict2 (via ComfyUI) | Apache-2.0 |
| `shims.py` | reimplements `comfy_kitchen` eager ops ([Comfy-Org/comfy-kitchen](https://github.com/Comfy-Org/comfy-kitchen)) | Apache-2.0 |
| `ops.py`, `__init__.py` | original to this project | GPL-3.0 |

Because the **GPL-3.0** `anima_model.py` is included, **this project as a whole is
distributed under GPL-3.0-or-later**. The vendored files keep their original
attribution headers; the comfy-specific runtime couplings (compiled kernels, the
hook/patcher system, the `operations` abstraction) are replaced by the pure-PyTorch
shims in `shims.py` / `ops.py` so the model runs on CPU/CUDA/XPU unmodified.
