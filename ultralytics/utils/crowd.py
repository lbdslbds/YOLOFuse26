"""Original-COCO crowd evaluation shared by training and standalone detection validation.

YOLO labels cannot represent ``iscrowd``. Never reconstruct the evaluation ground truth
from those labels: retain the source image/category IDs and crowd flags instead.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np


class CrowdCOCOEvaluator:
    """Validate the original COCO contract and evaluate only images actually visited."""

    precision_recall_policy = (
        "Per-class maximum F1 on the COCO interpolated precision/recall grid at IoU=0.50, "
        "area=all, configured maxDet; these are not confidence-threshold Ultralytics P/R."
    )

    def __init__(self, annotation_file, names, split="val", max_det=300):
        self.annotation_file = Path(annotation_file)
        self.split = split
        self.max_det = int(max_det)
        if self.max_det < 1:
            raise ValueError("COCO crowd max_det must be positive")
        raw = self.annotation_file.read_bytes()
        self.annotation_sha256 = hashlib.sha256(raw).hexdigest()
        self.ground_truth = json.loads(raw)
        self.names = dict(enumerate(names)) if isinstance(names, (list, tuple)) else dict(names)
        if sorted(self.names) != list(range(len(self.names))):
            raise ValueError("Model class indices must be contiguous and zero-based")
        self.by_file_name = {}
        self.by_stem = {}
        image_ids = set()
        for image in self.ground_truth["images"]:
            image_id, name = image["id"], str(image["file_name"]).replace("\\", "/")
            if image_id in image_ids or name in self.by_file_name or Path(name).stem in self.by_stem:
                raise ValueError("Original COCO image IDs, file_names and stems must be unique")
            image_ids.add(image_id)
            self.by_file_name[name] = image_id
            self.by_stem[Path(name).stem] = (name, image_id)
        categories = self.ground_truth["categories"]
        category_ids = [c["id"] for c in categories]
        category_names = [c["name"] for c in categories]
        if len(set(category_ids)) != len(category_ids) or len(set(category_names)) != len(category_names):
            raise ValueError("Original COCO category IDs and names must be unique")
        by_name = {c["name"]: c["id"] for c in categories}
        if set(self.names.values()) != set(by_name) or len(set(self.names.values())) != len(self.names):
            raise ValueError("Model names must exactly match the original COCO category names")
        # Export directly to source IDs. In particular, never increment already one-based predictions.
        self.class_map = [by_name[self.names[i]] for i in range(len(self.names))]
        self.category_ids = set(category_ids)
        annotation_ids = set()
        for annotation in self.ground_truth["annotations"]:
            if type(annotation["id"]) is not int:
                raise ValueError("Original COCO annotation IDs must be integers")
            if annotation["id"] in annotation_ids:
                raise ValueError("Duplicate original COCO annotation ID")
            annotation_ids.add(annotation["id"])
            if annotation["image_id"] not in image_ids or annotation["category_id"] not in self.category_ids:
                raise ValueError("Original COCO annotation references an unknown image or category")
        # COCO uses annotation IDs in its match arrays, with 0 reserved for 'unmatched'.
        # Remap only non-positive IDs in a private copy, preserving the source annotation file,
        # its hash, and all image/category IDs, boxes, areas and crowd/ignore flags.
        self._evaluation_ground_truth = copy.deepcopy(self.ground_truth)
        self.annotation_id_remapping = []
        next_id = max(annotation_ids | {0}) + 1
        for annotation in self._evaluation_ground_truth["annotations"]:
            if annotation["id"] <= 0:
                self.annotation_id_remapping.append({"source_id": annotation["id"], "evaluation_id": next_id})
                annotation["id"] = next_id
                next_id += 1
        self._coco = None

    def file_name(self, image_file):
        """Resolve a paired image by its exact standardized COCO file_name, never by numeric stem."""
        name = Path(image_file).name
        if name not in self.by_file_name:
            raise ValueError(f"Validation image {name!r} is absent from original COCO file_names")
        return name

    def prepare_predictions(self, predictions, image_files):
        """Map exported stems to source IDs and fail on unknown files/classes or repeated ID offsets."""
        names = [self.file_name(path) for path in image_files]
        if len(set(names)) != len(names):
            raise ValueError("A COCO evaluation image was visited twice; use a non-padding validation sampler")
        selected = {self.by_file_name[name] for name in names}
        result = []
        for prediction in predictions:
            name = prediction.get("file_name")
            if name is None:
                stem = str(prediction["image_id"])
                if stem not in self.by_stem:
                    raise ValueError(f"Prediction stem {stem!r} is absent from original COCO images")
                name = self.by_stem[stem][0]
            if name not in self.by_file_name:
                raise ValueError(f"Prediction file_name {name!r} is absent from original COCO images")
            expected_stem = Path(name).stem
            if str(prediction["image_id"]) != expected_stem:
                raise ValueError(f"Prediction image_id must be its exact file_name stem ({name!r})")
            source_id = self.by_file_name[name]
            if source_id not in selected:
                raise ValueError(f"Prediction {name!r} was not part of the evaluated image subset")
            category = prediction["category_id"]
            if category not in self.category_ids:
                raise ValueError(f"Unknown COCO prediction category {category!r}; do not add 1 twice")
            box, score = prediction["bbox"], prediction["score"]
            if len(box) != 4 or not np.isfinite([*box, score]).all() or box[2] < 0 or box[3] < 0:
                raise ValueError("COCO predictions require finite xywh boxes and confidence scores")
            result.append({"image_id": source_id, "category_id": category, "bbox": box, "score": score})
        return result, [self.by_file_name[name] for name in names]

    def evaluate(self, predictions, image_files, metrics, logger=print):
        """Run native COCO crowd matching and synchronize the public Metric object and fitness."""
        from faster_coco_eval import COCO, COCOeval_faster

        predictions, image_ids = self.prepare_predictions(predictions, image_files)
        if not image_ids:
            raise ValueError("No validation images were visited for COCO crowd evaluation")
        self._coco = self._coco or COCO(self._evaluation_ground_truth)
        detected = (
            self._coco.loadRes(predictions)
            if predictions
            else COCO({**self._evaluation_ground_truth, "annotations": []})
        )
        evaluator = COCOeval_faster(self._coco, detected, iouType="bbox", print_function=logger)
        evaluator.params.imgIds = image_ids
        evaluator.params.catIds = self.class_map
        evaluator.params.maxDets = [1, min(10, self.max_det), self.max_det]
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()

        # COCO orders catIds while preparing evaluation. Resolve by ID, not assumed model order.
        precision = evaluator.eval["precision"]  # IoU, recall, category, area, maxDet
        recall_grid = np.asarray(evaluator.params.recThrs)
        category_index = {category: index for index, category in enumerate(evaluator.params.catIds)}
        aps, ps, rs, classes = [], [], [], []
        for class_index, category in enumerate(self.class_map):
            pr = precision[:, :, category_index[category], 0, -1]
            valid = pr >= 0
            if not valid.any():
                continue  # no non-ignored GT in this class/subset: COCO excludes it from the mean
            ap = np.array([row[row >= 0].mean() if (row >= 0).any() else 0.0 for row in pr])
            p = np.maximum(pr[0], 0)
            f1 = 2 * p * recall_grid / np.maximum(p + recall_grid, np.finfo(float).eps)
            best = int(f1.argmax())
            aps.append(ap)
            ps.append(p[best])
            rs.append(recall_grid[best] if p[best] > 0 else 0.0)
            classes.append(class_index)
        metrics.box.nc = len(self.names)
        metrics.box.all_ap = np.asarray(aps).reshape(-1, len(evaluator.params.iouThrs))
        metrics.box.ap_class_index = np.asarray(classes, dtype=int)
        metrics.box.p, metrics.box.r = np.asarray(ps), np.asarray(rs)
        metrics.box.f1 = 2 * metrics.box.p * metrics.box.r / np.maximum(
            metrics.box.p + metrics.box.r, np.finfo(float).eps
        )
        # Keep Metric's optional curve attributes valid but empty; ordinary matching curves are disabled.
        metrics.box.px = np.asarray([])
        for key in ("p_curve", "r_curve", "f1_curve", "prec_values"):
            setattr(metrics.box, key, np.empty((len(classes), 0)))
        metrics.box.clear_image_metrics()  # the native per-image Ultralytics matcher is not crowd-aware
        selected_ids = set(image_ids)
        inverse_classes = {category: i for i, category in enumerate(self.class_map)}
        counts = np.zeros(len(self.names), dtype=int)
        class_images = [set() for _ in self.names]
        crowd_count = 0
        for annotation in self.ground_truth["annotations"]:
            if annotation["image_id"] not in selected_ids:
                continue
            if annotation.get("iscrowd", 0):
                crowd_count += 1
            elif not annotation.get("ignore", 0):
                c = inverse_classes[annotation["category_id"]]
                counts[c] += 1
                class_images[c].add(annotation["image_id"])
        metrics.nt_per_class = counts
        metrics.nt_per_image = np.asarray([len(x) for x in class_images])
        metrics.confusion_matrix = None
        result = metrics.results_dict
        # Assert the public Metric object and backend agree before best.pt selection can use the result.
        backend_ap = max(0.0, float(evaluator.stats_as_dict["AP_all"]))
        backend_ap50 = max(0.0, float(evaluator.stats_as_dict["AP_50"]))
        if not np.allclose([metrics.box.map, metrics.box.map50], [backend_ap, backend_ap50], atol=1e-8):
            raise RuntimeError("COCO AP and internal Metric AP disagree")
        for area in ("small", "medium", "large"):
            result[f"metrics/mAP_{area}(B)"] = float(evaluator.stats_as_dict[f"AP_{area}"])
        policy = {
            "evaluator": "faster_coco_eval.COCOeval_faster",
            "ground_truth": str(self.annotation_file.resolve()),
            "ground_truth_sha256": self.annotation_sha256,
            "annotation_id_policy": (
                "Preserve positive original IDs; remap non-positive IDs in source order to unused positive IDs "
                "above the original maximum, in an evaluation-only copy"
            ),
            "annotation_id_remapping": self.annotation_id_remapping,
            "split": self.split,
            "images": len(image_ids),
            "image_ids": image_ids,
            "category_ids_by_model_class": self.class_map,
            "non_crowd_instances": int(counts.sum()),
            "crowd_regions": crowd_count,
            "predictions": len(predictions),
            "iscrowd_policy": "Original COCO iscrowd retained; crowd uses COCO intersection/detection-area matching",
            "max_detections": evaluator.params.maxDets,
            "iou_thresholds": np.asarray(evaluator.params.iouThrs).tolist(),
            "area_ranges": evaluator.params.areaRng,
            "precision_recall_policy": self.precision_recall_policy,
            "fitness": "COCO bbox AP@[0.50:0.95] (same as metrics.box.map)",
            "disabled_non_crowd_visualizations": ["confusion_matrix", "Ultralytics PR/F1 curves", "per_image_metrics"],
        }
        return result, policy
