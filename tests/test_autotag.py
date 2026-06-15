"""Tests for WD14 tag post-processing (pure, no ONNX model needed)."""

import numpy as np

from anima_trainer.autotag import LabelTable, TagConfig, select_tags

# index: 0 rating, 1-3 general, 4 character
LABELS = LabelTable(
    names=["general_rating", "long_hair", "blue_eyes", "smile", "hatsune_miku"],
    categories=[9, 0, 0, 0, 4],
)


def test_thresholds_and_ordering():
    probs = np.array([0.9, 0.8, 0.4, 0.2, 0.9])
    cfg = TagConfig(general_threshold=0.35, character_threshold=0.85,
                    include_rating=False, replace_underscore=False)
    tags = select_tags(probs, LABELS, cfg)
    # character first, then general by descending prob; "smile" (0.2) dropped, rating excluded
    assert tags == ["hatsune_miku", "long_hair", "blue_eyes"]


def test_character_threshold_excludes_low_confidence_character():
    probs = np.array([0.1, 0.8, 0.1, 0.1, 0.5])  # character below 0.85
    tags = select_tags(probs, LABELS, TagConfig())
    assert "hatsune_miku" not in tags
    assert "long hair" in tags  # underscores replaced by default


def test_underscore_replacement_and_max_tags():
    probs = np.array([0.1, 0.9, 0.9, 0.9, 0.1])
    cfg = TagConfig(max_tags=2, replace_underscore=True)
    tags = select_tags(probs, LABELS, cfg)
    assert len(tags) == 2
    assert all("_" not in t for t in tags)


def test_include_rating_adds_one():
    probs = np.array([0.9, 0.9, 0.1, 0.1, 0.1])
    cfg = TagConfig(include_rating=True, replace_underscore=False)
    tags = select_tags(probs, LABELS, cfg)
    assert "general_rating" in tags
