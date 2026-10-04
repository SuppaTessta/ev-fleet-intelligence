# ADR-0003: Re-split the casting dataset before augmentation

**Status:** accepted
**Supersedes:** the claim of 99.44% accuracy with 100% defect recall and "zero defect escapes"

## Context

The casting-defect dataset ships two folders:

```
casting_512x512/   1,300 ORIGINAL images   (781 def_front, 519 ok_front)
casting_data/      7,348 AUGMENTED images, pre-split 6,633 train / 715 test
```

The published result used the vendor's `casting_data` split. That split was made **after**
augmentation, so rotations of the same physical casting sit on both sides.

## Investigation

A 16×16 average hash over all 715 test images against the full 6,633-image train set:

| Nearest train neighbour within | Test images |
|---|---|
| **0 bits — perceptual duplicate** | **80 / 715 = 11.2%** |
| 2 bits | 142 / 715 = 19.9% |
| 8 bits | 452 / 715 = 63.2% |
| median nearest distance | **7 / 256** |

Two distance-0 pairs were opened side by side: visibly the same casting, same chipped rim, same
background marks, differing by a small rotation.

**Removing exact duplicates barely moves the accuracy, which is why this is tempting to dismiss.**
That measures the wrong thing. Mapping every augmented image to its nearest *original*:

- **97.5% of test images (697/715)** have their source part also in training
- **93.6% of distinct test parts (234/250)** appear in training

The test set is not held out in any meaningful sense.

## The leak was hiding a real failure, not just inflating a number

Both models were cross-evaluated on both test sets:

| | vendor test (leaky, 715) | clean part-level test (260) |
|---|---|---|
| **Model trained on the leaky split** | 99.44%, recall 1.000 | **88.08%** |
| **Model trained on the clean split** | 95.66% | **97.31%**, recall 0.955 |

The old model does not merely *measure* badly on unseen castings — it flags **29 of 104 good parts
(28%) as defective**. A production line running it would scrap a quarter of its good output, and the
vendor's test set could not reveal that because 97.5% of its parts were in training.

## Decision

`data/prepare_casting.py` splits the **1,300 originals** first, then augments only the training side.

It clusters perceptually-identical originals before splitting: the source folder contains two
duplicate pairs (1,300 files → 1,298 distinct parts), and a file-level split would have put one
member of a pair in train and the other in test, re-creating the leak it exists to remove.

`evaluation/leakage_check.py` verifies any image split and exits non-zero above a threshold, so CI
fails rather than regressing quietly. Result: **80 exact duplicates → 0**.

## Consequences

- Published casting accuracy drops **99.44% → 97.3%**, defect recall **1.000 → 0.955**.
- **"Zero defect escapes" is retired.** It does not survive a clean split. Seven of 156 defects are
  missed.
- Real-world performance *improves*: **88.08% → 97.31%** on genuinely unseen castings.
- The vendor's split remains in the repo, and `train_quality_model.py` still accepts `--data-dir`,
  so the comparison stays reproducible. The old model is kept as `quality_model_leaky_split.keras`.

## Rejected alternatives

- **Report the dedup-corrected number (99.39%).** Removing 64 exact hash matches leaves ~97%
  part-level contamination untouched. It answers a question nobody asked and reads as a defence.
- **Keep the vendor split with a caveat.** A caveat does not fix a model that rejects 28% of good
  parts; the number would still be the one quoted.
- **Augment the test side too.** Inflates apparent test size without adding information. A test set
  should be real images.

## Note on the sibling dataset

The identical check was run on NEU-DET as a control: **0 / 360 exact duplicates, median nearest
distance 33/256.** Its shipped split is clean, and its 99.7% stands. Same measurement, opposite
verdict — which is how you know the measurement is doing work.
