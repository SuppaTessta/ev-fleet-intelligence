"""
Casting-defect dataset -- leakage-free re-split.

WHY THIS EXISTS
---------------
The dataset ships two folders:

    casting_512x512/   1,300 ORIGINAL 512x512 images (781 def_front, 519 ok_front)
    casting_data/      7,348 AUGMENTED 300x300 images, pre-split 6,633 train / 715 test

The published 99.44% accuracy / 100% defect recall was measured on the vendor's
`casting_data` test split. That split was made AFTER augmentation, so rotations
of the same physical casting sit on both sides of it.

Measured with a 16x16 average-hash over all 715 test images against the full
6,633-image train set:

    exact perceptual duplicates (0 bits) : 80 / 715  = 11.2%
    within 2 bits                        : 142 / 715 = 19.9%
    within 8 bits                        : 452 / 715 = 63.2%
    median nearest-neighbour distance    : 7 / 256

Two of the 0-bit pairs were opened side by side and are visibly the same part,
same chipped rim, same background marks, differing only by a small rotation.

Removing exact duplicates barely moves the metric, which is why it is tempting
to call this minor. That measures the wrong thing. Mapping every augmented image
back to its nearest ORIGINAL shows the real scale:

    test images whose source part also appears in train : 697 / 715 = 97.5%
    distinct test parts also present in train           : 234 / 250 = 93.6%

So the "held-out" set is essentially not held out. This script splits the 1,300
ORIGINALS first, then augments only the training side, so no physical casting
can appear in both halves.

Run once:
    python data/prepare_casting.py

Then retrain:
    python train/train_quality_model.py --data-dir data/raw/casting_defect_clean
"""
import argparse
import shutil
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "raw" / "casting_defect_clean"

# Searched in order; the first one containing def_front/ and ok_front/ wins.
# This was a single hardcoded absolute path on one developer's D: drive, so the
# script's default could not work on any other machine -- including CI and any
# clone of the repository.
_DATASET_ROOT = ROOT.parent / "Dataset for ET" / "casting product image data for quality inspection"
SOURCE_CANDIDATES = [
    ROOT / "data" / "raw" / "casting_512x512",
    _DATASET_ROOT / "casting_512x512" / "casting_512x512",
    _DATASET_ROOT / "casting_512x512",
]
CLASS_NAMES = ["def_front", "ok_front"]
TEST_SIZE = 0.20
SEED = 42
IMG_SIZE = (300, 300)      # match the augmented set the model was designed around
AUG_PER_IMAGE = 5          # train side only


def augment(img: Image.Image, rng: np.random.Generator):
    """Rotation + flips only -- the same family of transforms the vendor used.

    Deliberately mild: the point is to restore the training-set size lost by
    splitting at part level, not to invent a new augmentation policy.
    """
    angle = float(rng.uniform(0, 360))
    out = img.rotate(angle, resample=Image.BILINEAR, fillcolor=0)
    if rng.random() < 0.5:
        out = out.transpose(Image.FLIP_LEFT_RIGHT)
    if rng.random() < 0.5:
        out = out.transpose(Image.FLIP_TOP_BOTTOM)
    return out


def _cluster_duplicates(files, side=16):
    """Union-find over perceptually identical originals.

    Returns a cluster id per file. Two files land in the same cluster when their
    16x16 average hashes are bit-identical, so a duplicated original can never be
    split across train and test.
    """
    def ahash(p):
        a = np.asarray(Image.open(p).convert("L").resize((side, side), Image.BILINEAR),
                       dtype=np.float32)
        return (a > a.mean()).ravel()

    h = np.array([ahash(f) for f in files], dtype=np.int32) * 2 - 1
    dist = (side * side - (h @ h.T)) // 2

    parent = list(range(len(files)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, j in zip(*np.where(dist <= 0)):
        a, b = find(int(i)), find(int(j))
        if a != b:
            parent[a] = b
    return [find(i) for i in range(len(files))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, default=None,
                    help="folder containing def_front/ and ok_front/ ORIGINALS. "
                         "If omitted, the known dataset locations are searched.")
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    ap.add_argument("--aug", type=int, default=AUG_PER_IMAGE)
    args = ap.parse_args()

    if args.source is None:
        args.source = next(
            (c for c in SOURCE_CANDIDATES
             if all((c / cls).is_dir() for cls in CLASS_NAMES)), None)
    if args.source is None or not args.source.is_dir():
        raise SystemExit(
            "could not find the casting originals. Looked under:\n"
            + "\n".join(f"  {c}" for c in SOURCE_CANDIDATES)
            + "\npoint --source at the casting_512x512 folder containing "
              "def_front/ and ok_front/")

    files, labels = [], []
    for cls in CLASS_NAMES:
        fs = sorted((args.source / cls).glob("*.jpeg"))
        if not fs:
            raise SystemExit(f"no .jpeg files in {args.source / cls}")
        files += fs
        labels += [cls] * len(fs)
    print(f"originals: {len(files)}  " +
          "  ".join(f"{c}={labels.count(c)}" for c in CLASS_NAMES))

    # The originals are not all distinct: two pairs are perceptually identical
    # (1,300 files -> 1,298 parts). Splitting per FILE would put one member of a
    # pair in train and the other in test, reintroducing exactly the leakage this
    # script exists to remove. Cluster first, split clusters.
    clusters = _cluster_duplicates(files)
    n_clusters = len(set(clusters))
    if n_clusters < len(files):
        print(f"  collapsed {len(files) - n_clusters} duplicate original(s) "
              f"-> {n_clusters} distinct parts")

    # one representative label per cluster, for stratification
    cluster_ids = sorted(set(clusters))
    cluster_label = {}
    for c, lab in zip(clusters, labels):
        cluster_label.setdefault(c, lab)

    # THE WHOLE POINT: split the ORIGINALS, before any augmentation exists
    train_c, test_c = train_test_split(
        cluster_ids, test_size=TEST_SIZE,
        stratify=[cluster_label[c] for c in cluster_ids], random_state=SEED)
    train_c, test_c = set(train_c), set(test_c)

    train_files = [f for f, c in zip(files, clusters, strict=True) if c in train_c]
    train_lab = [lab for lab, c in zip(labels, clusters, strict=True) if c in train_c]
    test_files = [f for f, c in zip(files, clusters, strict=True) if c in test_c]
    test_lab = [lab for lab, c in zip(labels, clusters, strict=True) if c in test_c]
    print(f"part-level split: {len(train_files)} train parts / {len(test_files)} test parts")

    if args.out.exists():
        shutil.rmtree(args.out)   # never merge with a previous run's split
    for split in ("train", "test"):
        for cls in CLASS_NAMES:
            (args.out / split / cls).mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(SEED)

    # test side: originals only, resized. No augmentation -- a test set should be
    # real images, and augmenting it would only inflate its apparent size.
    for f, cls in zip(test_files, test_lab):
        Image.open(f).convert("RGB").resize(IMG_SIZE, Image.BILINEAR).save(
            args.out / "test" / cls / f.name, quality=95)

    # train side: original + N augmentations each
    n_train = 0
    for f, cls in zip(train_files, train_lab):
        base = Image.open(f).convert("RGB")
        base.resize(IMG_SIZE, Image.BILINEAR).save(
            args.out / "train" / cls / f.name, quality=95)
        n_train += 1
        for k in range(args.aug):
            augment(base, rng).resize(IMG_SIZE, Image.BILINEAR).save(
                args.out / "train" / cls / f"{f.stem}_aug{k}.jpeg", quality=95)
            n_train += 1

    print(f"\nwrote {args.out}")
    for split in ("train", "test"):
        counts = {c: len(list((args.out / split / c).glob('*.jpeg'))) for c in CLASS_NAMES}
        print(f"  {split:5s} {sum(counts.values()):5d}  {counts}")
    print("\nNo physical casting appears in both splits -- verify with "
          "evaluation/leakage_check.py")


if __name__ == "__main__":
    main()
