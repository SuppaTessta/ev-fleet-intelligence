"""
Train/serve preprocessing parity for the vision classifiers.

The bug this guards against was silent and expensive: training resized with
tf.image.resize (bilinear, antialias off, via image_dataset_from_directory)
while serving resized with PIL, whose default resample for RGB is BICUBIC. On
the 715-image casting test set that cost defect recall 1.0000 -> 0.9823 -- the
live endpoint missed 8 of 453 real defects that the published evaluation caught.
The headline claim ("zero defect escapes") was true of the eval path and false of
the API path, and nothing in the test suite could tell.

There is a trap worth knowing about if these ever fail: switching PIL to
Image.BILINEAR looks like the obvious fix and is WORSE (recall 0.9581, 19 missed).
PIL's bilinear and tf.image.resize's antialias=False bilinear are different
operations. Resize with tf.image.resize, or not at all.
"""
import numpy as np
import pytest

# importorskip, not a bare import: with no tensorflow wheel for the running
# interpreter this module raised at COLLECTION, which pytest reports as an
# error and which stops the whole run -- taking the 100+ tests that need no
# TensorFlow down with it. A missing optional wheel is a skip.
tf = pytest.importorskip("tensorflow", reason="needs tensorflow-cpu (see SETUP.md)")

# Deliberately NO skip_if_missing here. These tests compare preprocessing
# tensors and never load a model, so gating them on a 98 MB artifact -- which is
# gitignored -- meant the one guard against the highest-impact bug in the vision
# pipeline never ran in CI. They now run everywhere.
from app.agents.quality_agent import IMG_SIZE, to_model_input  # noqa: E402
from conftest import PROJECT_ROOT

FIXTURES = PROJECT_ROOT / "tests" / "fixtures"
SAMPLES = ["casting_defective_sample.jpeg", "casting_ok_sample.jpeg",
           "neu_det_crazing_sample.jpg", "neu_det_scratches_sample.jpg"]


def _training_pipeline_tensor(path):
    """Reproduces exactly what image_dataset_from_directory feeds the model:
    TF's own decoder, then tf.image.resize with default (bilinear, no antialias)."""
    raw = tf.io.decode_image(tf.io.read_file(str(path)), channels=3, expand_animations=False)
    return tf.image.resize(tf.cast(raw, tf.float32), IMG_SIZE).numpy()


@pytest.mark.parametrize("name", SAMPLES)
def test_serving_bytes_path_is_bit_identical_to_training(name):
    """Passing raw bytes -- what the HTTP layer actually has -- must reproduce the
    training tensor exactly, so the published metric IS the served metric."""
    path = FIXTURES / name
    served, _ = to_model_input(path.read_bytes(), IMG_SIZE)
    expected = _training_pipeline_tensor(path)

    # to_model_input applies preprocess_input; undo it for comparison. ResNet50's
    # preprocess_input is RGB->BGR plus a per-channel mean subtraction.
    from tensorflow.keras.applications.resnet50 import preprocess_input
    np.testing.assert_allclose(
        served[0], preprocess_input(np.expand_dims(expected, 0))[0], rtol=0, atol=0,
        err_msg="serving preprocessing has drifted from the training pipeline")


@pytest.mark.parametrize("name", SAMPLES)
def test_pil_path_stays_close_to_training(name):
    """A PIL Image is accepted for convenience. It uses a different JPEG decoder,
    so it is not bit-identical -- but it must stay within a hair, and critically
    it must still go through tf.image.resize rather than PIL's resize."""
    from PIL import Image
    path = FIXTURES / name
    served, _ = to_model_input(Image.open(path), IMG_SIZE)
    from tensorflow.keras.applications.resnet50 import preprocess_input
    expected = preprocess_input(np.expand_dims(_training_pipeline_tensor(path), 0))
    assert np.abs(served - expected).max() < 8.0, (
        "PIL decode path diverges too far from the training pipeline")


def test_pil_default_resize_is_measurably_different():
    """Regression guard with teeth: if someone reintroduces PIL's resize, this
    fails. It asserts the OLD behaviour is genuinely different, so the test can
    never silently pass because both paths became the same thing again."""
    from PIL import Image
    path = FIXTURES / "casting_defective_sample.jpeg"
    correct, _ = to_model_input(path.read_bytes(), IMG_SIZE)
    old_way = np.expand_dims(
        tf.keras.utils.img_to_array(Image.open(path).convert("RGB").resize(IMG_SIZE)), 0)
    from tensorflow.keras.applications.resnet50 import preprocess_input
    assert np.abs(correct - preprocess_input(old_way)).max() > 1.0, (
        "expected PIL's default resize to differ from tf.image.resize -- if this "
        "fails, the comparison has stopped testing anything")
