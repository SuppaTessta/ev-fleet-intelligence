"""
Manufacturing Quality Intelligence Agent.

Two independent sub-agents, each wrapping its own trained ResNet50 transfer-learning
classifier + Grad-CAM:

- QualityAgent (cast-component): binary defective/OK gate over cast EV parts.
- SurfaceQualityAgent (NEU-DET): 6-way defect-TYPE classifier over sheet-metal /
  structural steel surface images (crazing, inclusion, patches, pitted_surface,
  rolled-in_scale, scratches). Every image is assumed to already contain a defect --
  there's no "OK" class here, the question is which of the 6 known types it is.

Both return a verdict, confidence, and a base64 Grad-CAM overlay so the caller can
see *why*.
"""

import base64
import io
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras import layers, models

MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "quality_model.keras"
IMG_SIZE = (128, 128)
CLASS_NAMES = ["def_front", "ok_front"]
LABEL_MAP = {"def_front": "defective", "ok_front": "ok"}


class QualityAgent:
    def __init__(self, model_path: Path = MODEL_PATH):
        self.model = tf.keras.models.load_model(model_path)
        # base ResNet50 sub-model is the 2nd layer (index 1) of the functional model
        self.base_model = self.model.layers[1]

    def _gradcam(self, img_array, last_conv_layer_name="conv5_block3_out"):
        grad_model = models.Model(
            [self.base_model.input],
            [self.base_model.get_layer(last_conv_layer_name).output, self.base_model.output],
        )
        head_input = layers.Input(shape=self.base_model.output_shape[1:])
        h = layers.GlobalAveragePooling2D()(head_input)
        h = self.model.get_layer(index=-3)(h)
        h = self.model.get_layer(index=-2)(h)
        head_output = self.model.get_layer(index=-1)(h)
        head_model = models.Model(head_input, head_output)

        with tf.GradientTape() as tape:
            conv_out, base_out = grad_model(img_array)
            tape.watch(conv_out)
            preds = head_model(base_out)
            loss = preds[:, 0]

        grads = tape.gradient(loss, conv_out)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_out = conv_out[0]
        heatmap = conv_out @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
        return heatmap.numpy(), float(preds[0, 0])

    def predict(self, pil_image) -> dict:
        img = pil_image.convert("RGB").resize(IMG_SIZE)
        arr = tf.keras.utils.img_to_array(img)
        arr_batch = preprocess_input(np.expand_dims(arr, 0))

        heatmap, score = self._gradcam(arr_batch)
        predicted_class = CLASS_NAMES[1] if score > 0.5 else CLASS_NAMES[0]
        confidence = score if score > 0.5 else 1 - score

        heatmap_resized = tf.image.resize(heatmap[..., tf.newaxis], IMG_SIZE).numpy().squeeze()
        fig, ax = plt.subplots(figsize=(2.5, 2.5))
        ax.imshow(arr.astype("uint8"))
        ax.imshow(heatmap_resized, cmap="jet", alpha=0.45)
        ax.axis("off")
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0, dpi=100)
        plt.close(fig)
        buf.seek(0)
        gradcam_b64 = base64.b64encode(buf.read()).decode("utf-8")

        return {
            "verdict": LABEL_MAP[predicted_class],
            "confidence": round(confidence, 4),
            "gradcam_overlay_png_base64": gradcam_b64,
        }


_agent_instance = None


def get_quality_agent() -> QualityAgent:
    global _agent_instance
    if _agent_instance is None:
        _agent_instance = QualityAgent()
    return _agent_instance


SURFACE_MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "quality_model_neu_det.keras"
SURFACE_IMG_SIZE = (128, 128)
SURFACE_CLASS_NAMES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]


class SurfaceQualityAgent:
    """
    NEU-DET steel surface / sheet-metal defect classifier -- second, independent
    Quality sub-agent alongside QualityAgent.

    Genuinely multi-class (not a binary gate like the casting model), so two things
    differ from QualityAgent's approach: the output head is softmax(6) instead of
    sigmoid(1), and the Grad-CAM loss term is keyed off the predicted class's softmax
    output (preds[:, class_idx]) instead of a fixed preds[:, 0] -- there's no single
    "positive" neuron in a 6-way softmax, so the gradient has to explain a specific
    class rather than "defective vs not".
    """

    def __init__(self, model_path: Path = SURFACE_MODEL_PATH):
        self.model = tf.keras.models.load_model(model_path)
        # base ResNet50 sub-model is the 2nd layer (index 1) of the functional model
        self.base_model = self.model.layers[1]

    def _gradcam(self, img_array, last_conv_layer_name="conv5_block3_out"):
        grad_model = models.Model(
            [self.base_model.input],
            [self.base_model.get_layer(last_conv_layer_name).output, self.base_model.output],
        )
        head_input = layers.Input(shape=self.base_model.output_shape[1:])
        h = layers.GlobalAveragePooling2D()(head_input)
        h = self.model.get_layer(index=-3)(h)
        h = self.model.get_layer(index=-2)(h)
        head_output = self.model.get_layer(index=-1)(h)
        head_model = models.Model(head_input, head_output)

        with tf.GradientTape() as tape:
            conv_out, base_out = grad_model(img_array)
            tape.watch(conv_out)
            preds = head_model(base_out)
            class_idx = tf.argmax(preds[0])
            loss = preds[:, class_idx]

        grads = tape.gradient(loss, conv_out)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_out = conv_out[0]
        heatmap = conv_out @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
        return heatmap.numpy(), preds[0].numpy(), int(class_idx.numpy())

    def predict(self, pil_image) -> dict:
        img = pil_image.convert("RGB").resize(SURFACE_IMG_SIZE)
        arr = tf.keras.utils.img_to_array(img)
        arr_batch = preprocess_input(np.expand_dims(arr, 0))

        heatmap, probs, class_idx = self._gradcam(arr_batch)
        predicted_class = SURFACE_CLASS_NAMES[class_idx]
        confidence = float(probs[class_idx])
        class_probabilities = {name: round(float(p), 4) for name, p in zip(SURFACE_CLASS_NAMES, probs)}

        heatmap_resized = tf.image.resize(heatmap[..., tf.newaxis], SURFACE_IMG_SIZE).numpy().squeeze()
        fig, ax = plt.subplots(figsize=(2.5, 2.5))
        ax.imshow(arr.astype("uint8"))
        ax.imshow(heatmap_resized, cmap="jet", alpha=0.45)
        ax.axis("off")
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0, dpi=100)
        plt.close(fig)
        buf.seek(0)
        gradcam_b64 = base64.b64encode(buf.read()).decode("utf-8")

        return {
            "verdict": predicted_class,
            "confidence": round(confidence, 4),
            "class_probabilities": class_probabilities,
            "gradcam_overlay_png_base64": gradcam_b64,
        }


_surface_agent_instance = None


def get_surface_quality_agent() -> SurfaceQualityAgent:
    global _surface_agent_instance
    if _surface_agent_instance is None:
        _surface_agent_instance = SurfaceQualityAgent()
    return _surface_agent_instance
