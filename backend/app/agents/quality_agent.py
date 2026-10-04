"""Manufacturing Quality Intelligence: two ResNet50 classifiers with Grad-CAM.

- QualityAgent (casting): binary defective/OK gate over cast EV parts.
- SurfaceQualityAgent (NEU-DET): 6-way defect-TYPE classifier over steel
  surfaces. Every image is assumed to contain a defect; the question is which of
  the six known types, not whether one is present.

Both return a verdict, a confidence, and a base64 Grad-CAM overlay.

PREPROCESSING PARITY

Training decodes with TensorFlow and resizes with tf.image.resize -- bilinear,
antialias off. That is not interchangeable with PIL: PIL defaults to BICUBIC for
RGB, and PIL's BILINEAR is a different operation again because tf.image.resize
with antialias=False does not low-pass before downsampling. Measured on the 715
held-out casting images:

    tf decode + tf.image.resize     99.44% acc, defect recall 1.0000
    PIL decode + tf.image.resize    99.72% acc, defect recall 1.0000
    PIL decode + PIL BICUBIC        98.88% acc, defect recall 0.9823  (8 missed)
    PIL decode + PIL BILINEAR       97.34% acc, defect recall 0.9581 (19 missed)

Note row 4: switching PIL to BILINEAR is the obvious "match the training
interpolation" fix and it is WORSE. The only correct fix is to resize with
tf.image.resize, which is what to_model_input does -- and to pass raw bytes,
which additionally routes through TensorFlow's own JPEG decoder so the serving
path is bit-identical to the evaluation path.

tests/test_quality_parity.py asserts that and fails if the paths drift apart.
"""

import base64
import io
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import colormaps
from PIL import Image

from app import config

# TensorFlow is a hard requirement of this agent and an optional one of the
# service. Imported unguarded it is neither: app.main imports the quality
# router, so on any interpreter without a tensorflow-cpu wheel (3.13+, arm64
# Linux) uvicorn refuses to start and all 18 endpoints go down -- including
# /health and /ready, whose whole purpose is to report partial availability.
# Six of the seven agents never touch it.
#
# So absorb the ImportError, report it through /ready like a missing artifact,
# and raise it at the point of use.
try:
    import tensorflow as tf
    from tensorflow.keras import layers, models
    from tensorflow.keras.applications.resnet50 import preprocess_input
except ImportError as exc:   # pragma: no cover - depends on the interpreter
    tf = layers = models = preprocess_input = None
    TENSORFLOW_IMPORT_ERROR: ImportError | None = exc
else:
    TENSORFLOW_IMPORT_ERROR = None


def tensorflow_available() -> bool:
    """Whether the vision sub-agents can run at all in this process.

    Read by app.readiness so /ready cannot report `quality` as available on the
    strength of two .keras files it has no way to load.
    """
    return TENSORFLOW_IMPORT_ERROR is None


def _require_tensorflow() -> None:
    if TENSORFLOW_IMPORT_ERROR is not None:
        raise TENSORFLOW_IMPORT_ERROR

MODEL_PATH = config.QUALITY_CASTING_MODEL
IMG_SIZE = (128, 128)
CLASS_NAMES = ["def_front", "ok_front"]
LABEL_MAP = {"def_front": "defective", "ok_front": "ok"}

SURFACE_MODEL_PATH = config.QUALITY_SURFACE_MODEL
SURFACE_IMG_SIZE = (128, 128)
SURFACE_CLASS_NAMES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]

LAST_CONV_LAYER = "conv5_block3_out"
_JET = colormaps["jet"]


def to_model_input(source, size=IMG_SIZE):
    """Decode and resize exactly the way training did. See the module docstring.

    `source` may be raw bytes (preferred -- uses TensorFlow's decoder, giving
    bit-exact parity with training) or a PIL Image (accepted for convenience;
    differs from the training decoder on ~2 of 715 test images, inside noise but
    not identical, so prefer bytes where you have them).

    Returns (batched_preprocessed_input, display_rgb_uint8). The display copy is
    resized but NOT preprocessed, since preprocess_input mean-centres into a
    range that is meaningless to look at.
    """
    _require_tensorflow()
    if isinstance(source, (bytes, bytearray, memoryview)):
        raw = tf.io.decode_image(tf.constant(bytes(source)), channels=3, expand_animations=False)
        raw = tf.cast(raw, tf.float32)
    elif isinstance(source, Image.Image):
        raw = tf.convert_to_tensor(
            tf.keras.utils.img_to_array(source.convert("RGB")), dtype=tf.float32)
    else:
        raise TypeError(f"expected bytes or a PIL Image, got {type(source).__name__}")

    # bilinear, antialias=False -- image_dataset_from_directory's default, and
    # NOT interchangeable with PIL's BILINEAR (see module docstring)
    resized = tf.image.resize(raw, size).numpy()
    display = np.clip(resized, 0, 255).astype("uint8")
    return preprocess_input(np.expand_dims(resized, 0)), display


def _overlay_png_base64(display_rgb, heatmap, alpha=0.45) -> str:
    """Blend a Grad-CAM heatmap over the image and return a base64 PNG.

    Deliberately avoids pyplot: building a Figure per request goes through the
    global figure manager, which is not thread-safe, and FastAPI runs sync
    handlers in a threadpool. Pure numpy plus a colormap lookup instead -- no
    global state, and faster.
    """
    h = np.asarray(Image.fromarray(
        (np.clip(heatmap, 0, 1) * 255).astype("uint8")).resize(
            (display_rgb.shape[1], display_rgb.shape[0]), Image.BILINEAR)) / 255.0
    colored = (_JET(h)[..., :3] * 255).astype("float32")
    blended = (1 - alpha) * display_rgb.astype("float32") + alpha * colored
    buf = io.BytesIO()
    Image.fromarray(np.clip(blended, 0, 255).astype("uint8")).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


class _ResNetGradCamAgent:
    """Shared plumbing for both classifiers.

    Loads the model and builds the Grad-CAM sub-models once, at construction.
    Doing it per prediction meant two Keras Model constructions and a fresh
    layers.Input on the hot path of a CPU-bound endpoint.
    """

    def __init__(self, model_path: Path):
        _require_tensorflow()
        self.model = tf.keras.models.load_model(model_path)
        # the ResNet50 sub-model is the 2nd layer (index 1) of the functional model
        self.base_model = self.model.layers[1]
        self.grad_model = models.Model(
            [self.base_model.input],
            [self.base_model.get_layer(LAST_CONV_LAYER).output, self.base_model.output],
        )
        head_input = layers.Input(shape=self.base_model.output_shape[1:])
        h = layers.GlobalAveragePooling2D()(head_input)
        h = self.model.get_layer(index=-3)(h)
        h = self.model.get_layer(index=-2)(h)
        self.head_model = models.Model(head_input, self.model.get_layer(index=-1)(h))

    def _heatmap(self, img_array, target):
        """target: callable(preds) -> scalar tensor to explain."""
        with tf.GradientTape() as tape:
            conv_out, base_out = self.grad_model(img_array)
            tape.watch(conv_out)
            preds = self.head_model(base_out)
            loss = target(preds)

        grads = tape.gradient(loss, conv_out)
        pooled = tf.reduce_mean(grads, axis=(0, 1, 2))
        cam = tf.squeeze(conv_out[0] @ pooled[..., tf.newaxis])
        cam = tf.maximum(cam, 0) / (tf.math.reduce_max(cam) + 1e-8)
        return cam.numpy(), preds


class QualityAgent(_ResNetGradCamAgent):
    def __init__(self, model_path: Path = MODEL_PATH):
        super().__init__(model_path)

    def predict(self, source) -> dict:
        arr_batch, display = to_model_input(source, IMG_SIZE)

        # Peek at the score first so Grad-CAM can explain the class we actually
        # report. The single output neuron is sigmoid P(ok_front) -- CLASS_NAMES
        # is alphabetical, so index 0 is def_front and index 1 is ok_front.
        score = float(self.model.predict(arr_batch, verbose=0)[0, 0])
        is_ok = score > 0.5

        # Explain the class actually being reported. Grad-CAM ReLU-clips negative
        # contributions, so a fixed target of preds[:, 0] = P(ok_front) would
        # highlight the most OK-LOOKING regions on every 'defective' verdict --
        # the inverse of the explanation, on the only case an inspector opens the
        # heatmap for. For a defective verdict we explain -P(ok_front), whose
        # gradient points at the evidence against the part being sound.
        target = (lambda p: p[:, 0]) if is_ok else (lambda p: -p[:, 0])
        heatmap, _ = self._heatmap(arr_batch, target)

        return {
            "verdict": LABEL_MAP[CLASS_NAMES[1] if is_ok else CLASS_NAMES[0]],
            "confidence": round(score if is_ok else 1 - score, 4),
            "gradcam_overlay_png_base64": _overlay_png_base64(display, heatmap),
        }


class SurfaceQualityAgent(_ResNetGradCamAgent):
    """NEU-DET steel surface defect classifier.

    Genuinely multi-class rather than a binary gate, so the head is softmax(6)
    and the Grad-CAM target is the predicted class rather than a fixed neuron --
    there is no single "positive" output in a 6-way softmax.
    """

    def __init__(self, model_path: Path = SURFACE_MODEL_PATH):
        super().__init__(model_path)

    def predict(self, source) -> dict:
        arr_batch, display = to_model_input(source, SURFACE_IMG_SIZE)
        probs = self.model.predict(arr_batch, verbose=0)[0]
        class_idx = int(np.argmax(probs))

        heatmap, _ = self._heatmap(arr_batch, lambda p: p[:, class_idx])

        return {
            "verdict": SURFACE_CLASS_NAMES[class_idx],
            "confidence": round(float(probs[class_idx]), 4),
            "class_probabilities": {n: round(float(p), 4)
                                     for n, p in zip(SURFACE_CLASS_NAMES, probs, strict=False)},
            "gradcam_overlay_png_base64": _overlay_png_base64(display, heatmap),
        }


_agent_instance = None
_surface_agent_instance = None


def get_quality_agent() -> QualityAgent:
    global _agent_instance
    if _agent_instance is None:
        _agent_instance = QualityAgent()
    return _agent_instance


def get_surface_quality_agent() -> SurfaceQualityAgent:
    global _surface_agent_instance
    if _surface_agent_instance is None:
        _surface_agent_instance = SurfaceQualityAgent()
    return _surface_agent_instance
