"""
Manufacturing Quality Intelligence Agent -- NEU-DET steel surface defect training pipeline.

Classifies sheet-metal / structural steel surface images into one of 6 real defect
types -- same transfer-learning technique (ResNet50 + Grad-CAM) as the cast-component
classifier (train_quality_model.py), but multi-class instead of binary: this is a
defect-TYPE classifier (which of 6 known defects is present), not a defect/OK gate
like the casting model -- NEU-DET has no "no defect" class.

Data: NEU surface defect database ("NEU-DET"), 1800 images, 300 per class, 6 classes:
crazing, inclusion, patches, pitted_surface, rolled-in_scale, scratches. Ships as a
flat object-detection layout (filename-encoded labels + unused XML bounding boxes);
run data/prepare_neu_det.py first to build the train/<class>/ + test/<class>/ split
this script expects (same layout convention as casting_defect_clean).

Approach
--------
- ResNet50 (ImageNet weights) as a frozen feature extractor -- only the head trains,
  same reasoning as casting: fast on CPU, avoids overfitting a dataset this size
  (1440 train images across 6 classes).
- Softmax(6) + sparse_categorical_crossentropy instead of casting's sigmoid(1) +
  binary_crossentropy -- this is genuinely multi-class, not a binary gate.
- Evaluated on a held-out test/ split (20%, stratified by class -- see prepare_neu_det.py),
  not a random slice of train.
- Grad-CAM loss term is keyed off the predicted class's softmax output
  (preds[:, class_idx]) instead of casting's preds[:, 0], since there's no single
  "positive" neuron in a 6-way softmax -- the gradient has to explain a specific class.

Outputs
-------
- backend/models/quality_model_neu_det.keras
- docs/quality_neu_det_model_eval.png       (6x6 confusion matrix + metrics)
- docs/quality_neu_det_gradcam_samples.png  (Grad-CAM overlays on sample predictions)
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras import layers, models
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input

SEED = 42
# Seeds python/numpy/tensorflow together. Without this, 99.7% (359/360) was a
# single unseeded run resting on one misclassification, and the Grad-CAM sample
# figure drew different images every time.
tf.keras.utils.set_random_seed(SEED)

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "raw" / "neu_det"
MODEL_DIR = Path(__file__).resolve().parents[1] / "backend" / "models"
DOCS_DIR = Path(__file__).resolve().parents[1] / "docs"
CASTING_MODEL_PATH = MODEL_DIR / "quality_model.keras"
for d in (MODEL_DIR, DOCS_DIR):
    d.mkdir(parents=True, exist_ok=True)

IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 4  # converges fast on this dataset -- val_accuracy hit 99.5%+ by epoch 4 and plateaued
# alphabetical = Keras default label order (also matches prepare_neu_det.py's CLASS_NAMES)
CLASS_NAMES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]
NUM_CLASSES = len(CLASS_NAMES)


def load_datasets():
    train_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR / "train", labels="inferred", label_mode="int",
        class_names=CLASS_NAMES, image_size=IMG_SIZE, batch_size=BATCH_SIZE,
        validation_split=0.15, subset="training", seed=42,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR / "train", labels="inferred", label_mode="int",
        class_names=CLASS_NAMES, image_size=IMG_SIZE, batch_size=BATCH_SIZE,
        validation_split=0.15, subset="validation", seed=42,
    )
    test_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR / "test", labels="inferred", label_mode="int",
        class_names=CLASS_NAMES, image_size=IMG_SIZE, batch_size=BATCH_SIZE, shuffle=False,
    )

    norm = lambda x, y: (preprocess_input(x), y)
    return (train_ds.map(norm).prefetch(tf.data.AUTOTUNE),
            val_ds.map(norm).prefetch(tf.data.AUTOTUNE),
            test_ds.map(norm).prefetch(tf.data.AUTOTUNE))


def build_model():
    if CASTING_MODEL_PATH.exists():
        # Reuse the ImageNet weights already loaded into the casting classifier's
        # frozen ResNet50 backbone instead of re-downloading ~90MB of identical
        # weights -- same architecture (ResNet50, no top) and same input shape, so
        # the weight arrays line up layer-for-layer. Both Quality sub-agents end up
        # sharing one frozen, unmodified ImageNet backbone, which is also more
        # consistent than two independently-downloaded copies.
        print(f"Reusing ImageNet backbone weights from {CASTING_MODEL_PATH.name} "
              f"(skips a second ~90MB download)...")
        base = ResNet50(weights=None, include_top=False, input_shape=IMG_SIZE + (3,))
        casting_model = tf.keras.models.load_model(CASTING_MODEL_PATH)
        casting_base = casting_model.layers[1]  # index 1 = the ResNet50 sub-model (see quality_agent.py)
        base.set_weights(casting_base.get_weights())
    else:
        print("No existing quality_model.keras found -- downloading ImageNet weights fresh...")
        base = ResNet50(weights="imagenet", include_top=False, input_shape=IMG_SIZE + (3,))

    base.trainable = False  # frozen feature extractor -- only the head trains

    inputs = layers.Input(shape=IMG_SIZE + (3,))
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)
    model = models.Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model, base


def evaluate(model, test_ds):
    y_true, y_pred = [], []
    for x, y in test_ds:
        y_true.extend(y.numpy().flatten().tolist())
        probs = model.predict(x, verbose=0)
        y_pred.extend(np.argmax(probs, axis=1).tolist())
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES, output_dict=True)
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(NUM_CLASSES)); ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right")
    ax.set_yticks(range(NUM_CLASSES)); ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(f"NEU-DET Surface Defect Classifier -- Test Set Confusion Matrix\n"
                 f"Accuracy: {report['accuracy']:.1%}")
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout()
    fig.savefig(DOCS_DIR / "quality_neu_det_model_eval.png", dpi=150)
    plt.close(fig)
    return report


def make_gradcam_heatmap(img_array, model, base_model, last_conv_layer_name="conv5_block3_out"):
    grad_model = models.Model(
        [base_model.input],
        [base_model.get_layer(last_conv_layer_name).output, base_model.output],
    )
    # Re-run the head on the base's pooled output to connect gradients end-to-end
    head_input = layers.Input(shape=base_model.output_shape[1:])
    h = layers.GlobalAveragePooling2D()(head_input)
    h = model.get_layer(index=-3)(h)
    h = model.get_layer(index=-2)(h)
    head_output = model.get_layer(index=-1)(h)
    head_model = models.Model(head_input, head_output)

    with tf.GradientTape() as tape:
        conv_out, base_out = grad_model(img_array)
        tape.watch(conv_out)
        preds = head_model(base_out)
        class_idx = tf.argmax(preds[0])
        # Multi-class: explain the predicted class's softmax output, not a fixed
        # index -- there's no single "positive" neuron like the binary case.
        loss = preds[:, class_idx]

    grads = tape.gradient(loss, conv_out)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_out = conv_out[0]
    heatmap = conv_out @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy(), preds[0].numpy(), int(class_idx.numpy())


def save_gradcam_samples(model, base_model, n_per_class=1):
    import random
    random.seed(SEED)  # same samples every run, so the figure is reproducible
    test_dir = DATA_DIR / "test"
    samples = []
    for cls in CLASS_NAMES:
        files = sorted((test_dir / cls).glob("*.jpg"))  # glob order is filesystem-dependent
        samples.extend([(f, cls) for f in random.sample(files, min(n_per_class, len(files)))])

    fig, axes = plt.subplots(2, len(samples), figsize=(3.0 * len(samples), 6.4))
    for i, (fpath, true_cls) in enumerate(samples):
        # tf.io.decode_image + tf.image.resize, matching image_dataset_from_directory.
        # tf.keras.utils.load_img(target_size=...) resizes via PIL NEAREST, so the
        # figure would be built from different pixels than the model was scored on.
        raw = tf.io.decode_image(tf.io.read_file(str(fpath)), channels=3, expand_animations=False)
        arr = tf.image.resize(tf.cast(raw, tf.float32), IMG_SIZE).numpy()
        arr_batch = preprocess_input(np.expand_dims(arr, 0))
        display = np.clip(arr, 0, 255).astype("uint8")  # unclipped cast wraps -0.0001 to 255

        heatmap, probs, class_idx = make_gradcam_heatmap(arr_batch, model, base_model)
        pred_cls = CLASS_NAMES[class_idx]
        pred_score = float(probs[class_idx])

        heatmap_resized = tf.image.resize(heatmap[..., tf.newaxis], IMG_SIZE).numpy().squeeze()

        axes[0, i].imshow(display)
        axes[0, i].set_title(f"True: {true_cls}", fontsize=9)
        axes[0, i].axis("off")

        axes[1, i].imshow(display)
        axes[1, i].imshow(heatmap_resized, cmap="jet", alpha=0.45)
        axes[1, i].set_title(f"Pred: {pred_cls} ({pred_score:.2f})", fontsize=9)
        axes[1, i].axis("off")

    fig.suptitle("Grad-CAM -- what the NEU-DET model is looking at")
    fig.tight_layout()
    fig.savefig(DOCS_DIR / "quality_neu_det_gradcam_samples.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    print(f"Loading NEU-DET surface defect dataset from {DATA_DIR} ...")
    print("(Run data/prepare_neu_det.py first if this folder doesn't exist yet.)")
    train_ds, val_ds, test_ds = load_datasets()

    print("\nBuilding ResNet50 transfer-learning model (frozen base, 6-way softmax)...")
    model, base_model = build_model()
    model.summary()

    print(f"\nTraining head for {EPOCHS} epochs...")
    history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, verbose=2)

    print("\n--- Evaluating on held-out test set ---")
    report = evaluate(model, test_ds)

    print("\n--- Generating Grad-CAM samples ---")
    save_gradcam_samples(model, base_model)

    model.save(MODEL_DIR / "quality_model_neu_det.keras")
    print(f"\nSaved model to {MODEL_DIR / 'quality_model_neu_det.keras'}")
    print(f"Saved plots to {DOCS_DIR}")
    print(f"\nFinal test accuracy: {report['accuracy']:.1%}")
