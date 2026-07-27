"""
Manufacturing Quality agent tests -- both classifiers. Needs the trained
.keras models (gitignored, produced by train/train_quality_model.py and
train/train_quality_model_neu_det.py). Auto-skips with a clear reason if
those aren't present yet.

Uses 4 small (10-21KB) real sample images committed under tests/fixtures/
rather than requiring the full raw dataset (which is gitignored and, unlike
the models, was never meant to be reproduced by anyone but the original
trainer) -- so these tests run for anyone who has the trained models, not
just whoever still has data/raw/ populated locally.
"""
from pathlib import Path
from PIL import Image

from conftest import skip_if_missing, MODELS_DIR

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"

skip_if_missing(MODELS_DIR / "quality_model.keras", MODELS_DIR / "quality_model_neu_det.keras")

from app.agents.quality_agent import get_quality_agent, get_surface_quality_agent  # noqa: E402


def test_casting_classifier_on_a_real_known_defective_image():
    agent = get_quality_agent()
    result = agent.predict(Image.open(FIXTURES_DIR / "casting_defective_sample.jpeg"))
    assert result["verdict"] == "defective"
    assert 0.5 <= result["confidence"] <= 1.0
    assert len(result["gradcam_overlay_png_base64"]) > 100


def test_casting_classifier_on_a_real_known_ok_image():
    agent = get_quality_agent()
    result = agent.predict(Image.open(FIXTURES_DIR / "casting_ok_sample.jpeg"))
    assert result["verdict"] == "ok"


def test_neu_det_classifier_returns_one_of_six_classes():
    expected_classes = {"crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"}
    agent = get_surface_quality_agent()
    result = agent.predict(Image.open(FIXTURES_DIR / "neu_det_crazing_sample.jpg"))
    assert result["verdict"] in expected_classes
    assert set(result["class_probabilities"].keys()) == expected_classes
    assert abs(sum(result["class_probabilities"].values()) - 1.0) < 0.01  # softmax sums to ~1


def test_neu_det_classifier_correct_on_two_real_known_samples():
    """Not a full accuracy sweep (that's train/train_quality_model_neu_det.py's
    job, at 99.7% on the real 360-image test set) -- just confirms the loaded
    model behaves sanely on unambiguous real examples from two different classes."""
    agent = get_surface_quality_agent()
    crazing_result = agent.predict(Image.open(FIXTURES_DIR / "neu_det_crazing_sample.jpg"))
    assert crazing_result["verdict"] == "crazing"
    scratches_result = agent.predict(Image.open(FIXTURES_DIR / "neu_det_scratches_sample.jpg"))
    assert scratches_result["verdict"] == "scratches"
