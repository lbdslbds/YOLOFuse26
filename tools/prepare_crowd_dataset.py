"""Build a new paired dataset view with ordinary targets and separate crowd ignore regions.

Images are hard-linked into real split directories so YOLO's image-to-label mapping stays in the new view.
Original images, annotations, labels, experiments and split membership are never changed.
"""
import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path


def prepare(images_root, annotation_paths, output):
    images_root, output = Path(images_root).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite an existing dataset view: {output}")
    # Validate the entire annotation inventory before materializing a new view.
    splits, names, records = {}, None, []
    for split, annotation_path in annotation_paths.items():
        annotation_path = Path(annotation_path).resolve()
        raw = annotation_path.read_bytes()
        data = json.loads(raw)
        categories = sorted(data["categories"], key=lambda item: item["id"])
        if names is None:
            names = [c["name"] for c in categories]
        if [c["name"] for c in categories] != names:
            raise ValueError("Train/val category names disagree")
        category_map = {c["id"]: i for i, c in enumerate(categories)}
        if [c["id"] for c in categories] != list(range(1, len(names) + 1)):
            raise ValueError("This paired exporter requires contiguous one-based COCO categories")
        by_image = defaultdict(list)
        image_ids = {im["id"] for im in data["images"]}
        if len(image_ids) != len(data["images"]):
            raise ValueError("Duplicate COCO image ID")
        annotation_ids = set()
        for ann in data["annotations"]:
            if ann["id"] in annotation_ids or ann["image_id"] not in image_ids:
                raise ValueError("Duplicate annotation ID or unknown image ID")
            annotation_ids.add(ann["id"])
            if ann["category_id"] not in category_map or ann.get("iscrowd", 0) not in (0, 1):
                raise ValueError("Unknown category ID or invalid iscrowd")
            by_image[ann["image_id"]].append(ann)
        converted, stems, counts = deepcopy(data), set(), Counter()
        for image in converted["images"]:
            relative = Path(image["file_name"])
            if relative.parent.name not in ("day", "night") or not relative.name.endswith("_rgb.jpg"):
                raise ValueError(f"Unexpected source image naming: {relative}")
            stem = relative.parent.name + "_" + relative.name[:-8]
            if stem in stems:
                raise ValueError(f"Duplicate output image stem: {stem}")
            stems.add(stem)
            image["file_name"] = stem + ".jpg"
            width, height = image["width"], image["height"]
            normal, crowd = [], []
            for ann in by_image[image["id"]]:
                x, y, w, h = ann["bbox"]
                if (not all(math.isfinite(v) for v in (x, y, w, h)) or w <= 0 or h <= 0
                        or min(x, y) < 0 or x + w > width + 1e-6 or y + h > height + 1e-6):
                    raise ValueError(f"Invalid source bbox: {ann['id']}")
                row = [category_map[ann["category_id"]], (x + w / 2) / width, (y + h / 2) / height, w / width, h / height]
                (crowd if ann.get("iscrowd", 0) else normal).append(row)
            for folder in ("images", "imagesIR"):
                if not (images_root / folder / split / (stem + ".jpg")).is_file():
                    raise FileNotFoundError(images_root / folder / split / (stem + ".jpg"))
            records.append((split, stem, normal, crowd))
            counts.update(images=1, ordinary_boxes=len(normal), crowd_boxes=len(crowd), crowd_images=int(bool(crowd)))
        splits[split] = {"annotations": converted, "source": str(annotation_path),
                         "source_sha256": hashlib.sha256(raw).hexdigest(), "counts": dict(counts)}
    output.mkdir(parents=True)
    for split in splits:
        for folder in ("images", "imagesIR", "labels", "crowd_labels"):
            (output / folder / split).mkdir(parents=True)
    for split, stem, normal, crowd in records:
        for folder in ("images", "imagesIR"):
            os.link(images_root / folder / split / (stem + ".jpg"), output / folder / split / (stem + ".jpg"))
        for folder, rows in (("labels", normal), ("crowd_labels", crowd)):
            text = "".join(str(row[0]) + " " + " ".join(f"{v:.12f}" for v in row[1:]) + "\n" for row in rows)
            (output / folder / split / (stem + ".txt")).write_text(text, encoding="utf-8")
    (output / "annotations").mkdir()
    for split, info in splits.items():
        (output / "annotations" / f"{split}.json").write_text(json.dumps(info.pop("annotations"), ensure_ascii=False), encoding="utf-8")
    config = {"path": str(output), "train": "images/train", "val": "images/val", "channels": 6,
              "nc": len(names), "names": dict(enumerate(names)), "crowd_labels": str(output / "crowd_labels"),
              "coco_annotations": {split: str(output / "annotations" / f"{split}.json") for split in splits}}
    (output / "data.yaml").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "status": "ready_not_trained",
              "images_root": str(images_root), "output": str(output), "splits": splits,
              "split_policy": "Retain original split membership; this change fixes crowd semantics only",
              "training_policy": "Noncrowd positive targets; separate class-specific crowd ignore sidecars",
              "image_policy": "Hard links; source image bytes are not changed",
              "evaluation_policy": "Original COCO IDs, boxes, area and iscrowd retained; canonical image filenames"}
    (output / "preparation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images-root", required=True)
    parser.add_argument("--train-json", required=True)
    parser.add_argument("--val-json", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.images_root, {"train": args.train_json, "val": args.val_json}, args.output), indent=2))
