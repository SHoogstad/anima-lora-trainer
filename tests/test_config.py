"""Config roundtrip tests (requires tomlkit)."""

import pytest

tomlkit = pytest.importorskip("tomlkit")

from anima_trainer.config import TrainConfig  # noqa: E402


def test_roundtrip(tmp_path):
    cfg = TrainConfig()
    cfg.lora.rank = 32
    cfg.dataset.image_dir = "data/imgs"
    cfg.max_train_steps = 1234

    path = tmp_path / "cfg.toml"
    cfg.save(path)
    loaded = TrainConfig.load(path)

    assert loaded.lora.rank == 32
    assert loaded.dataset.image_dir == "data/imgs"
    assert loaded.max_train_steps == 1234
    assert loaded.model.repo_id == "circlestone-labs/Anima"


def test_defaults_target_anima():
    cfg = TrainConfig()
    assert "anima-base-v1.0" in cfg.model.dit_file
    assert cfg.lora.target_modules == ["to_q", "to_k", "to_v", "to_out.0"]
