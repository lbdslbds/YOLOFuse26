# Crowd-aware paired detection

The current fixed train/val/test setup uses `configs/rgbir-tvt.yaml` (also the default
`configs/rgbir.yaml`): 5279/721/2676 aligned image pairs, with no split overlap.
All three splits retain original COCO IDs and crowd annotations. The compatibility
filename `configs/rgbir-crowd.yaml` now points to this same fixed split. See
[runtime and migration notes](MODERN_RUNTIME.md) for the verified dataset and commands.

Before the fixed split was introduced, the crowd configuration used the separate
dataset view `datasets_crowd_20261018`. That historical view retained
the then-existing train/val membership and did not resolve earlier use of the original
test split for selection. The current fixed split supersedes that membership.

The dataset view contains:

- `labels/{train,val,test}`: five-column YOLO labels for ordinary instances only.
- `crowd_labels/{train,val,test}`: the same format for crowd regions, with an empty file
  for every image without crowds. A missing file is an error.
- `annotations/{train,val,test}.json`: original COCO IDs, boxes, areas and `iscrowd`,
  with filenames mapped to the standardized paired-image names.
- `data.yaml`: six-channel data paths, `crowd_labels`, and split-specific `coco_annotations`.

Images are hard-linked into real directories, avoiding duplicate image storage and
keeping YOLO's image-to-label path mapping inside the new dataset view. The
initial two-split crowd preparation script is `tools/prepare_crowd_dataset.py`; it
refuses existing outputs. The fixed three-way split uses the migrated split lists
and manifest as its membership record.

Crowd regions follow the same Mosaic, MixUp, perspective, letterbox and flip
geometry as ordinary targets. They are separated before batching, never passed
to the positive assigner or bounding-box regression. Training ignores only the
crowd class's negative classification loss at unmatched anchor centers inside
these regions. Ordinary positive anchors take precedence. Both one-to-many and
one-to-one losses use this rule. This is an explicit training policy, not a claim
to reproduce COCO evaluation inside the loss.

CopyPaste, CutMix and custom Albumentations are rejected for this configuration;
set `copy_paste=0` and `cutmix=0`. Ordinary datasets without `crowd_labels` keep their
existing behavior. Crowd-aware original-COCO validation does not support `single_cls`.

Every validation epoch and standalone validation use the original COCO crowd
matching rules. `metrics/mAP50(B)`, `metrics/mAP50-95(B)` and checkpoint fitness
are synchronized with that evaluator. `max_det` is recorded explicitly (300 by
default here; the standard COCO default is 100). P/R use per-class maximum F1
on the COCO IoU=0.50 interpolated precision/recall grid; they are not the former
Ultralytics confidence operating point. Incompatible confusion matrices, PR/F1
curves and per-image matching metrics are disabled rather than mislabeled.

COCO matching uses annotation ID zero as an unmatched sentinel. Before evaluation,
non-positive annotation IDs are remapped to unused positive IDs in an in-memory
copy, above the original maximum. Positive IDs, image/category IDs, boxes, areas,
and `iscrowd` remain unchanged, and the original annotation file is never rewritten.
The evaluator records this remapping in `crowd_evaluation.json`; this avoids silently
treating a ground-truth annotation with ID zero as an unmatched detection.

Each validation writes `crowd_evaluation.json` with settings, annotation checksum,
actual image subset and scores; `predictions_coco.json` uses original COCO image
and category IDs for direct re-evaluation. `predictions.json` retains the normal
Ultralytics stem-ID export. Missing mappings and invalid categories fail explicitly.

Historical experiment `finalize.py` scripts retain their historical evaluation
policy. Use the updated `val_dual.py` with `configs/rgbir-crowd.yaml` for new evaluation.

The initial repair deployment did not start training. The subsequent runtime checks
used explicitly requested one-epoch smoke runs; their scores are functional evidence
and are not full-training benchmarks.
