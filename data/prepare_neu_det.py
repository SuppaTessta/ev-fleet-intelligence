"""
NEU-DET steel surface defect dataset -- one-time prep script.

The dataset is distributed in more than one shape, and this script accepts
either, because it previously accepted only the one nobody actually has.

  A. Flat object-detection layout, as the original NEU-DET release ships:
         IMAGES/<class>_<n>.jpg        1800 images, class encoded in the filename
         ANNOTATIONS/<class>_<n>.xml   PASCAL-VOC boxes -- not needed here

  B. Pre-split, class-foldered layout, which is how the copy in this
     repository's dataset folder is organised:
         NEU-DET/train/images/<class>/<class>_<n>.jpg        1440
         NEU-DET/validation/images/<class>/<class>_<n>.jpg    360
         NEU-DET/{train,validation}/annotations/...

Only layout A was recognised, via two hardcoded candidate paths under
data/raw/neu_det_source/. Pointed at layout B -- the one actually present --
find_source_dir raised FileNotFoundError, so the surface-defect model could not
be built at all from the dataset shipped alongside the project. Both layouts are
now discovered, and --source takes an explicit path for anything else.

WHY THE VENDOR SPLIT IS NOT REUSED
----------------------------------
Layout B arrives already split 1440/360. This script pools all 1800 images and
re-splits them stratified 80/20 at SEED anyway, deliberately: the published
99.7% is measured on that split, and silently switching to a different one --
even a reasonable one -- would mean the number in the README no longer describes
the model the script produces. Pooling first also makes A and B produce byte-for
-byte identical output, so which layout you happened to download cannot change
a published result.

Labels come from the filename in both cases (the class prefix), not the folder,
so a mislabelled directory cannot quietly relabel an image.

Run once:
    python data/prepare_neu_det.py
    python data/prepare_neu_det.py --source "path/to/NEU-DET"
"""
import argparse
import re
import shutil
from pathlib import Path

from sklearn.model_selection import train_test_split

DATA_RAW = Path(__file__).resolve().parent / "raw"
OUT_DIR = DATA_RAW / "neu_det"
CLASS_NAMES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]
TEST_SIZE = 0.2
SEED = 42
EXPECTED_TOTAL = 1800

# Searched in order. The first that yields images wins.
SOURCE_CANDIDATES = [
    DATA_RAW / "neu_det_source",
    DATA_RAW / "neu_det_source" / "NEU-DET",
    # the dataset folder this repository is distributed next to
    Path(__file__).resolve().parents[2] / "Dataset for ET" / "NEU Surface Defect Database" / "NEU-DET",
]


def classify_filename(fname: str) -> str:
    # filenames look like "rolled-in_scale_142.jpg" -- strip the trailing _<number>.<ext>
    stem = re.sub(r"_\d+\.(jpg|jpeg|png)$", "", fname, flags=re.IGNORECASE)
    if stem not in CLASS_NAMES:
        raise ValueError(f"Unrecognized class parsed from filename '{fname}': got '{stem}'")
    return stem


def collect_images(root: Path) -> list[Path]:
    """Every NEU-DET image under `root`, whatever the intermediate structure.

    rglob rather than a fixed IMAGES/ path: layout A puts them one level down,
    layout B puts them three levels down inside train/ and validation/. Files
    whose name does not parse to a known class are skipped rather than fataled,
    so a stray thumbnail or an ANNOTATIONS sibling cannot stop the run.
    """
    found = []
    for path in sorted(root.rglob("*")):
        if path.suffix.lower() not in (".jpg", ".jpeg", ".png") or not path.is_file():
            continue
        try:
            classify_filename(path.name)
        except ValueError:
            continue
        found.append(path)
    return found


def find_source_dir(explicit: Path | None) -> tuple[Path, list[Path]]:
    candidates = [explicit] if explicit else SOURCE_CANDIDATES
    for cand in candidates:
        if cand and cand.is_dir():
            images = collect_images(cand)
            if images:
                return cand, images
    raise SystemExit(
        "Couldn't find NEU-DET images. Looked under:\n"
        + "\n".join(f"  {c}" for c in candidates)
        + "\nPass --source with the folder containing the NEU-DET images "
          "(either a flat IMAGES/ folder or a NEU-DET/ with train/ and validation/)."
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", type=Path, default=None,
                    help="folder containing the NEU-DET images (either layout)")
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    args = ap.parse_args()

    src, files = find_source_dir(args.source)
    print(f"Reading images from {src} ...")

    labels = [classify_filename(f.name) for f in files]
    counts = {c: labels.count(c) for c in CLASS_NAMES}
    print(f"Found {len(files)} images: {counts}")

    # A duplicate filename across train/ and validation/ would mean the same
    # image lands in the pool twice and can straddle the re-split -- the exact
    # failure mode ADR-0003 documents for the casting set.
    names = [f.name for f in files]
    if len(set(names)) != len(names):
        dupes = sorted({n for n in names if names.count(n) > 1})[:5]
        raise SystemExit(f"duplicate image filenames across the source tree "
                         f"({len(names) - len(set(names))} of them, e.g. {dupes}). "
                         f"Re-splitting these would put one copy in train and the "
                         f"other in test.")
    if len(files) != EXPECTED_TOTAL:
        print(f"  note: expected {EXPECTED_TOTAL} images, found {len(files)} -- "
              f"the published 99.7% is measured on the full set")

    train_files, test_files = train_test_split(
        files, test_size=TEST_SIZE, stratify=labels, random_state=SEED
    )

    out = args.out
    if out.exists():
        shutil.rmtree(out)   # never merge with a previous run's split
    for split_name, split_files in (("train", train_files), ("test", test_files)):
        for cls in CLASS_NAMES:
            (out / split_name / cls).mkdir(parents=True, exist_ok=True)
        for f in split_files:
            shutil.copy2(f, out / split_name / classify_filename(f.name) / f.name)

    print("\nDone. Wrote:")
    for split_name, split_files in (("train", train_files), ("test", test_files)):
        split_counts = {c: sum(1 for f in split_files if classify_filename(f.name) == c)
                        for c in CLASS_NAMES}
        print(f"  {out / split_name}: {len(split_files)} images  {split_counts}")


if __name__ == "__main__":
    main()
