"""
NEU-DET steel surface defect dataset -- one-time prep script.

The dataset ships as a flat object-detection layout:
    IMAGES/<class>_<n>.jpg       (1800 images, 300 per class, class encoded in filename)
    ANNOTATIONS/<class>_<n>.xml  (PASCAL-VOC bounding boxes -- not needed for classification)

We only need image-level labels, which are already encoded in the filename, so we
skip the XML annotations entirely and do a stratified, class-preserving split into
the same train/<class>/ + test/<class>/ layout the casting-defect classifier uses.
That lets train_quality_model_neu_det.py reuse tf.keras.utils.image_dataset_from_directory
exactly like train_quality_model.py does -- same pattern, no new loading logic.

Setup: unzip NEU-DET.zip and place the resulting NEU-DET/ folder (containing
IMAGES/ and ANNOTATIONS/) at data/raw/neu_det_source/NEU-DET/.

Run once:
    python data/prepare_neu_det.py
"""
import re
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split

SOURCE_CANDIDATES = [
    Path(__file__).resolve().parent / "raw" / "neu_det_source" / "IMAGES",
    Path(__file__).resolve().parent / "raw" / "neu_det_source" / "NEU-DET" / "IMAGES",
]
OUT_DIR = Path(__file__).resolve().parent / "raw" / "neu_det"
CLASS_NAMES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]
TEST_SIZE = 0.2
SEED = 42


def find_source_dir():
    for cand in SOURCE_CANDIDATES:
        if cand.is_dir() and any(cand.iterdir()):
            return cand
    raise FileNotFoundError(
        "Couldn't find the NEU-DET IMAGES folder. Expected it at one of:\n"
        + "\n".join(f"  {c}" for c in SOURCE_CANDIDATES)
        + "\nUnzip NEU-DET.zip and place its IMAGES/ folder at one of those paths."
    )


def classify_filename(fname: str) -> str:
    # filenames look like "rolled-in_scale_142.jpg" -- strip the trailing _<number>.<ext>
    stem = re.sub(r"_\d+\.(jpg|jpeg|png)$", "", fname, flags=re.IGNORECASE)
    if stem not in CLASS_NAMES:
        raise ValueError(f"Unrecognized class parsed from filename '{fname}': got '{stem}'")
    return stem


def main():
    src = find_source_dir()
    print(f"Reading images from {src} ...")
    files = sorted(src.glob("*.jpg"))
    if not files:
        raise FileNotFoundError(f"No .jpg files found in {src}")

    labels = [classify_filename(f.name) for f in files]
    counts = {c: labels.count(c) for c in CLASS_NAMES}
    print(f"Found {len(files)} images: {counts}")

    train_files, test_files = train_test_split(
        files, test_size=TEST_SIZE, stratify=labels, random_state=SEED
    )

    for split_name, split_files in (("train", train_files), ("test", test_files)):
        for cls in CLASS_NAMES:
            (OUT_DIR / split_name / cls).mkdir(parents=True, exist_ok=True)
        for f in split_files:
            cls = classify_filename(f.name)
            shutil.copy2(f, OUT_DIR / split_name / cls / f.name)

    print("\nDone. Wrote:")
    for split_name, split_files in (("train", train_files), ("test", test_files)):
        split_counts = {c: sum(1 for f in split_files if classify_filename(f.name) == c) for c in CLASS_NAMES}
        print(f"  {OUT_DIR / split_name}: {len(split_files)} images  {split_counts}")


if __name__ == "__main__":
    main()
