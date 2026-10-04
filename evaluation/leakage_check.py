"""
Near-duplicate leakage check for image train/test splits.

Catches the failure that made the casting classifier's headline number
meaningless: a split made AFTER augmentation, so rotations of the same physical
part sit on both sides. Exact-filename comparison cannot see it; perceptual
hashing can.

Uses a 16x16 average hash (256 bits). Hamming distance 0 means the two images
are perceptually identical at that resolution -- for a dataset of near-identical
factory photographs that is strong evidence of a shared source image, though not
absolute proof, which is why the report includes the whole distance profile
rather than a single number.

    python evaluation/leakage_check.py data/raw/casting_defect_clean --max-exact 0

Exits non-zero when the exact-duplicate count exceeds --max-exact, so CI can
fail the build rather than quietly regressing.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

HASH_SIDE = 16
BITS = HASH_SIDE * HASH_SIDE
EXTS = ("*.jpg", "*.jpeg", "*.png", "*.bmp")


def ahash(path: Path, side: int = HASH_SIDE) -> np.ndarray:
    arr = np.asarray(
        Image.open(path).convert("L").resize((side, side), Image.BILINEAR),
        dtype=np.float32)
    return (arr > arr.mean()).ravel()


def collect(root: Path) -> list:
    return sorted(f for ext in EXTS for f in root.rglob(ext))


def hamming_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise Hamming distance via a single matmul on +/-1 encodings."""
    return (BITS - (a.astype(np.int32) @ b.T.astype(np.int32))) // 2


def check(train_dir: Path, test_dir: Path, max_exact: int):
    train_files, test_files = collect(train_dir), collect(test_dir)
    if not train_files or not test_files:
        raise SystemExit(f"no images found under {train_dir} or {test_dir}")
    print(f"train: {len(train_files)} images   test: {len(test_files)} images")

    tr = np.array([ahash(f) for f in train_files], dtype=np.int32) * 2 - 1
    te = np.array([ahash(f) for f in test_files], dtype=np.int32) * 2 - 1
    dist = hamming_matrix(te, tr)
    nearest = dist.min(axis=1)

    print(f"\n  {'nearest train neighbour within':38s} {'count':>7s} {'share':>8s}")
    for thr in (0, 2, 5, 8, 12):
        n = int((nearest <= thr).sum())
        label = f"{thr} bits" + (" (perceptual duplicate)" if thr == 0 else "")
        print(f"  {label:38s} {n:7d} {100 * n / len(nearest):7.1f}%")
    print(f"  {'median nearest distance':38s} {int(np.median(nearest)):7d} / {BITS}")

    exact = int((nearest == 0).sum())
    if exact:
        print("\n  examples of exact matches:")
        for i in np.where(nearest == 0)[0][:5]:
            print(f"    {test_files[i].name}  <->  {train_files[int(dist[i].argmin())].name}")

    ok = exact <= max_exact
    print(f"\n  {exact} exact duplicates (limit {max_exact}) -> {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", type=Path, help="folder containing train/ and test/")
    ap.add_argument("--train-name", default="train")
    ap.add_argument("--test-name", default="test")
    ap.add_argument("--max-exact", type=int, default=0,
                    help="fail above this many exact perceptual duplicates")
    args = ap.parse_args()
    ok = check(args.dataset / args.train_name, args.dataset / args.test_name, args.max_exact)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
