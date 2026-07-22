"""
Manufacturing Quality Intelligence Agent — training pipeline.

Classifies EV cast-component images (e.g., housings, impeller-style parts)
as Defective / OK using transfer learning on real casting inspection data —
same technique (ResNet50 + Grad-CAM) as the PhytoSentinel crop-disease project.

Data: "Real-life Industrial Dataset of Casting Product" (submersible pump
impeller castings, front view), ~7,348 images, pre-split train/test.

Approach
--------
- ResNet50 (ImageNet weights) as a frozen feature extractor — we only train
  a small classification head, which is fast even on CPU and avoids
  overfitting on a dataset this size.
- Evaluated on the dataset's own held-out test/ folder (not a random split
  of train), so the number reflects genuine generalisation.
- Grad-CAM overlay on sample predictions, so the agent can show *why* it
  flagged a part, not just a defective/OK label — same explainability
  pattern as PhytoSentinel.

Outputs
-------
- backend/models/quality_model.keras
- docs/quality_model_eval.png       (confusion matrix + metrics)
- docs/quality_gradcam_samples.png  (Grad-CAM overlays on sample predictions)
"""

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.metrics import confusion_matrix, classification_report

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "raw" / "casting_defect"
MODEL_DIR = Path(__file__).resolve().parents[1] / "backend" / "models"
DOCS_DIR = Path(__file__).resolve().parents[1] / "docs"
for d in (MODEL_DIR, DOCS_DIR):
    d.mkdir(parents=True, exist_ok=True)

IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 6
CLASS_NAMES = ["def_front", "ok_front"]  # alphabetical = Keras default label order


def load_datasets():
    train_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR / "train", labels="inferred", label_mode="binary",
        class_names=CLASS_NAMES, image_size=IMG_SIZE, batch_size=BATCH_SIZE,
        validation_split=0.15, subset="training", seed=42,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR / "train", labels="inferred", label_mode="binary",
        class_names=CLASS_NAMES, image_size=IMG_SIZE, batch_size=BATCH_SIZE,
        validation_split=0.15, subset="validation", seed=42,
    )
    test_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR / "test", labels="inferred", label_mode="binary",
        class_names=CLASS_NAMES, image_size=IMG_SIZE, batch_size=BATCH_SIZE, shuffle=False,
    )

    norm = lambda x, y: (preprocess_input(x), y)
    return (train_ds.map(norm).prefetch(tf.data.AUTOTUNE),
            val_ds.map(norm).prefetch(tf.data.AUTOTUNE),
            test_ds.map(norm).prefetch(tf.data.AUTOTUNE))


def build_model():
    base = ResNet50(weights="imagenet", include_top=False, input_shape=IMG_SIZE + (3,))
    base.trainable = False  # frozen feature extractor — only the head trains

    inputs = layers.Input(shape=IMG_SIZE + (3,))
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    model = models.Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss="binary_crossentropy", metrics=["accuracy"])
    return model, base


def evaluate(model, test_ds):
    y_true, y_pred_prob = [], []
    for x, y in test_ds:
        y_true.extend(y.numpy().flatten().tolist())
        y_pred_prob.extend(model.predict(x, verbose=0).flatten().tolist())
    y_true = np.array(y_true)
    y_pred = (np.array(y_pred_prob) > 0.5).astype(int)

    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES, output_dict=True)
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))

    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_xticklabels(CLASS_NAMES)
    ax.set_yticks([0, 1]); ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(f"Quality Agent — Test Set Confusion Matrix\n"
                 f"Accuracy: {report['accuracy']:.1%}")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout()
    fig.savefig(DOCS_DIR / "quality_model_eval.png", dpi=150)
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
        loss = preds[:, 0]

    grads = tape.gradient(loss, conv_out)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_out = conv_out[0]
    heatmap = conv_out @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy(), float(preds[0, 0])


def save_gradcam_samples(model, base_model, n=4):
    import random
    test_dir = DATA_DIR / "test"
    samples = []
    for cls in CLASS_NAMES:
        files = list((test_dir / cls).glob("*.jpeg"))
        samples.extend([(f, cls) for f in random.sample(files, min(n // 2, len(files)))])

    fig, axes = plt.subplots(2, len(samples), figsize=(3.2 * len(samples), 6.4))
    for i, (fpath, true_cls) in enumerate(samples):
        img = tf.keras.utils.load_img(fpath, target_size=IMG_SIZE)
        arr = tf.keras.utils.img_to_array(img)
        arr_batch = preprocess_input(np.expand_dims(arr, 0))

        heatmap, pred_score = make_gradcam_heatmap(arr_batch, model, base_model)
        pred_cls = CLASS_NAMES[1] if pred_score > 0.5 else CLASS_NAMES[0]

        heatmap_resized = tf.image.resize(heatmap[..., tf.newaxis], IMG_SIZE).numpy().squeeze()

        axes[0, i].imshow(arr.astype("uint8"))
        axes[0, i].set_title(f"True: {true_cls}", fontsize=9)
        axes[0, i].axis("off")

        axes[1, i].imshow(arr.astype("uint8"))
        axes[1, i].imshow(heatmap_resized, cmap="jet", alpha=0.45)
        axes[1, i].set_title(f"Pred: {pred_cls} ({pred_score:.2f})", fontsize=9)
        axes[1, i].axis("off")

    fig.suptitle("Grad-CAM — what the model is looking at")
    fig.tight_layout()
    fig.savefig(DOCS_DIR / "quality_gradcam_samples.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    print(f"Loading casting defect dataset from {DATA_DIR} ...")
    train_ds, val_ds, test_ds = load_datasets()

    print("\nBuilding ResNet50 transfer-learning model (frozen base)...")
    model, base_model = build_model()
    model.summary()

    print(f"\nTraining head for {EPOCHS} epochs...")
    history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, verbose=2)

    print("\n--- Evaluating on held-out test set ---")
    report = evaluate(model, test_ds)

    print("\n--- Generating Grad-CAM samples ---")
    save_gradcam_samples(model, base_model)

    model.save(MODEL_DIR / "quality_model.keras")
    print(f"\nSaved model to {MODEL_DIR / 'quality_model.keras'}")
    print(f"Saved plots to {DOCS_DIR}")
    print(f"\nFinal test accuracy: {report['accuracy']:.1%}")
