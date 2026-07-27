"""
Manufacturing Quality agent tests -- both classifiers. Needs the trained
.keras models (gitignored, produced by train/train_quality_model.py and
train/train_quality_model_neu_det.py) plus real sample test images
(gitignored, data/raw/). Auto-skips with a clear reason if either is missing.
"""
from pathlib import Path
from PIL import Image

from conftest import skip_if_missing, MODELS_DIR, PROJECT_ROOT

CASTING_TEST_DIR = PROJECT_ROOT / "data" / "raw" / "casting_defect" / "test"
NEU_DET_TEST_DIR = PROJECT_ROOT / "data" / "raw" / "neu_det" / "test"

skip_if_missing(
    MODELS_DIR / "quality_model.keras",
    MODELS_DIR / "quality_model_neu_det.keras",
    CASTING_TEST_DIR,
    NEU_DET_TEST_DIR,
)

from app.agents.quality_agent import get_quality_agent, get_surface_quality_agent  # noqa: E402


def _first_image(directory: Path) -> Path:
    for ext in ("*.jpeg", "*.jpg", "*.png"):
        found = list(directory.glob(ext))
        if found:
            return found[0]
    raise FileNotFoundError(f"no sample image found under {directory}")


def test_casting_classifier_on_a_real_known_defective_image():
    img_path = _first_image(CASTING_TEST_DIR / "def_front")
    agent = get_quality_agent()
    result = agent.predict(Image.open(img_path))
    assert result["verdict"] == "defective"
    assert 0.5 <= result["confidence"] <= 1.0
    assert len(result["gradcam_overlay_png_base64"]) > 100


def test_casting_classifier_on_a_real_known_ok_image():
    img_path = _first_image(CASTING_TEST_DIR / "ok_front")
    agent = get_quality_agent()
    result = agent.predict(Image.open(img_path))
    assert result["verdict"] == "ok"


def test_neu_det_classifier_returns_one_of_six_classes():
    expected_classes = {"crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"}
    img_path = _first_image(NEU_DET_TEST_DIR / "scratches")
    agent = get_surface_quality_agent()
    result = agent.predict(Image.open(img_path))
    assert result["verdict"] in expected_classes
    assert set(result["class_probabilities"].keys()) == expected_classes
    assert abs(sum(result["class_probabilities"].values()) - 1.0) < 0.01  # softmax sums to ~1


def test_neu_det_classifier_correct_on_a_real_known_sample():
    """Not a full accuracy sweep (that's train/train_quality_model_neu_det.py's
    job, at 99.7% on the real 360-image test set) -- just confirms the loaded
    model is actually behaving sanely on an unambiguous real example."""
    img_path = _first_image(NEU_DET_TEST_DIR / "crazing")
    agent = get_surface_quality_agent()
    result = agent.predict(Image.open(img_path))
    assert result["verdict"] == "crazing"
