"""Crowd-aware selection metrics must agree in training and standalone validation."""

import copy
import json
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("faster_coco_eval")

from ultralytics.models.yolo.detect.val import CrowdDetectionMetrics, DetectionValidator
from ultralytics.utils.crowd import CrowdCOCOEvaluator
from ultralytics.utils.metrics import DetMetrics


@pytest.fixture
def coco(tmp_path):
    data = {
        "info": {},
        "licenses": [],
        "images": [
            {"id": 41, "file_name": "day_000000.jpg", "height": 100, "width": 100},
            {"id": 99, "file_name": "night_000000.jpg", "height": 100, "width": 100},
        ],
        # Deliberately unordered, non-contiguous IDs expose positional mappings and repeated +1 bugs.
        "categories": [{"id": 13, "name": "bicycle"}, {"id": 3, "name": "person"}],
        "annotations": [
            {"id": 1, "image_id": 41, "category_id": 3, "bbox": [0, 0, 16, 16], "area": 256, "iscrowd": 0},
            {"id": 2, "image_id": 41, "category_id": 13, "bbox": [80, 0, 16, 16], "area": 256, "iscrowd": 0},
            {"id": 3, "image_id": 41, "category_id": 13, "bbox": [20, 20, 60, 60], "area": 3600, "iscrowd": 1},
            {"id": 4, "image_id": 99, "category_id": 3, "bbox": [0, 0, 16, 16], "area": 256, "iscrowd": 0},
        ],
    }
    path = tmp_path / "source_val.json"
    path.write_text(json.dumps(data))
    return path, data, {0: "person", 1: "bicycle"}


def predictions():
    return [
        {"image_id": "day_000000", "file_name": "day_000000.jpg", "category_id": category,
         "bbox": box, "score": score}
        for category, box, score in [
            (3, [0, 0, 16, 16], .90),
            (13, [80, 0, 16, 16], .90),
            # Low IoU but 100% intersection/detection-area: native crowd matching must ignore both.
            (13, [30, 30, 10, 10], .99),
            (13, [50, 50, 10, 10], .98),
        ]
    ]


def evaluate(coco, detections=None, max_det=300):
    path, _, names = coco
    evaluator = CrowdCOCOEvaluator(path, names, max_det=max_det)
    metrics = DetMetrics(names)
    result, policy = evaluator.evaluate(
        predictions() if detections is None else detections, ["day_000000.jpg"], metrics, logger=lambda _: None
    )
    return result, policy, metrics


def test_crowd_uses_ioa_multiple_matches_original_ids_and_actual_subset(coco):
    original = copy.deepcopy(coco[1])
    result, policy, metrics = evaluate(coco)
    assert result["metrics/mAP50(B)"] == pytest.approx(1)
    assert result["metrics/mAP50-95(B)"] == pytest.approx(1)
    assert result["fitness"] == pytest.approx(metrics.fitness)
    assert metrics.box.map == pytest.approx(1)
    assert policy["image_ids"] == [41]  # missing predictions on unvisited image 99 cannot lower AP
    assert policy["category_ids_by_model_class"] == [3, 13]
    assert policy["crowd_regions"] == 1
    assert policy["non_crowd_instances"] == 2
    assert policy["max_detections"] == [1, 10, 300]
    assert metrics.nt_per_class.tolist() == [1, 1]
    assert json.loads(coco[0].read_text()) == original


def test_wrong_class_inside_crowd_is_not_ignored(coco):
    detections = predictions()
    detections[2]["category_id"] = 3
    result, _, metrics = evaluate(coco, detections)
    assert result["metrics/mAP50-95(B)"] < 1
    assert metrics.box.ap[0] < metrics.box.ap[1]


def test_completely_empty_predictions_still_evaluate_original_ground_truth(coco):
    result, policy, metrics = evaluate(coco, [])
    assert result["metrics/mAP50(B)"] == 0
    assert result["metrics/mAP50-95(B)"] == result["fitness"] == metrics.fitness == 0
    assert policy["predictions"] == 0
    assert metrics.box.p.tolist() == metrics.box.r.tolist() == [0, 0]


def test_max_det_is_explicit_and_not_backend_default_100(coco):
    # Crowds of the same class can consume the prediction cap without becoming false positives.
    coco[1]["annotations"][2]["category_id"] = 3
    coco[0].write_text(json.dumps(coco[1]))
    detections = predictions()
    detections[2]["category_id"] = detections[3]["category_id"] = 3
    limited, policy, _ = evaluate(coco, detections, max_det=1)
    full, _, _ = evaluate(coco, detections, max_det=300)
    assert policy["max_detections"] == [1, 1, 1]
    assert limited["metrics/mAP50-95(B)"] < full["metrics/mAP50-95(B)"]
    assert full["metrics/mAP50-95(B)"] == pytest.approx(1)


@pytest.mark.parametrize("change", ("double_offset", "file_name", "stem", "unvisited"))
def test_invalid_prediction_mapping_fails_instead_of_silently_zeroing_ap(coco, change):
    evaluator = CrowdCOCOEvaluator(coco[0], coco[2])
    detections = predictions()
    if change == "double_offset":
        detections[1]["category_id"] += 1
    elif change == "file_name":
        detections[0]["file_name"] = "day_000000.png"
    elif change == "stem":
        detections[0]["image_id"] = 41
    else:
        detections[0].update(image_id="night_000000", file_name="night_000000.jpg")
    with pytest.raises(ValueError):
        evaluator.prepare_predictions(detections, ["day_000000.jpg"])


def test_duplicate_images_and_category_names_fail(coco):
    evaluator = CrowdCOCOEvaluator(coco[0], coco[2])
    with pytest.raises(ValueError, match="visited twice"):
        evaluator.prepare_predictions([], ["day_000000.jpg", "day_000000.jpg"])
    coco[1]["categories"][0]["name"] = "person"
    coco[0].write_text(json.dumps(coco[1]))
    with pytest.raises(ValueError, match="unique"):
        CrowdCOCOEvaluator(coco[0], coco[2])


def make_validator(coco, tmp_path, training):
    validator = DetectionValidator(args={"plots": True, "save_json": False}, save_dir=tmp_path)
    validator.training = training
    validator.device = torch.device("cpu")
    validator.data = {
        "val": str(tmp_path), "crowd_labels": str(tmp_path / "crowd"),
        "coco_annotations": {"val": str(coco[0])},
    }
    validator.dataloader = SimpleNamespace(dataset=SimpleNamespace(im_files=["day_000000.jpg"]), sampler=[0])
    validator.init_metrics(SimpleNamespace(names=coco[2], end2end=False))
    return validator


@pytest.mark.parametrize("training", (True, False))
@pytest.mark.parametrize("empty", (True, False))
def test_training_selection_and_standalone_share_coco_metrics(coco, tmp_path, training, empty):
    validator = make_validator(coco, tmp_path, training)
    assert validator.args.save_json and not validator.args.plots
    assert isinstance(validator.metrics, CrowdDetectionMetrics)
    batch = {
        "img": torch.zeros(1, 6, 100, 100), "batch_idx": torch.tensor([0, 0]),
        "cls": torch.tensor([[0.], [1.]]), "bboxes": torch.tensor([[.08, .08, .16, .16], [.88, .08, .16, .16]]),
        "ori_shape": [(100, 100)], "ratio_pad": [((1., 1.), (0., 0.))], "im_file": ["day_000000.jpg"],
    }
    saved = [] if empty else predictions()
    pred = {
        "bboxes": torch.tensor([[*p["bbox"][:2], p["bbox"][0] + p["bbox"][2], p["bbox"][1] + p["bbox"][3]]
                                for p in saved], dtype=torch.float32).reshape(-1, 4),
        "conf": torch.tensor([p["score"] for p in saved]),
        "cls": torch.tensor([0 if p["category_id"] == 3 else 1 for p in saved]),
        "extra": torch.zeros(len(saved), 0),
    }
    validator.update_metrics([pred], batch)
    result = validator.get_stats()
    assert result["metrics/mAP50-95(B)"] == pytest.approx(0 if empty else 1)
    assert result["metrics/mAP50(B)"] == validator.metrics.box.map50
    assert result["fitness"] == validator.metrics.fitness == validator.metrics.box.map
    assert result["metrics/precision(B)"] == validator.metrics.box.mp
    assert result["metrics/recall(B)"] == validator.metrics.box.mr
    assert validator.metrics.curves_results == []
    assert validator.metrics.box.image_metrics == {}
    validator.finalize_metrics()
    assert validator.metrics.confusion_matrix is None
    assert json.loads((tmp_path / "predictions.json").read_text()) == validator.jdict
    policy = json.loads((tmp_path / "crowd_evaluation.json").read_text())["policy"]
    assert policy["split"] == "val" and policy["image_ids"] == [41]
    # The base validator calls eval_json again only for nonempty standalone predictions; it must be idempotent.
    validator.crowd_evaluator.evaluate = lambda *a, **kw: pytest.fail("duplicate backend evaluation")
    assert validator.eval_json({}) == result


def test_missing_split_does_not_fall_back_to_ordinary_gt(coco, tmp_path):
    validator = make_validator(coco, tmp_path, True)
    validator.args.split = "train"
    with pytest.raises(ValueError, match="coco_annotations"):
        validator.init_metrics(SimpleNamespace(names=coco[2], end2end=False))


def test_no_crowd_configuration_keeps_standard_validator_behavior(coco, tmp_path):
    validator = DetectionValidator(args={"plots": True, "save_json": False}, save_dir=tmp_path)
    validator.training = True
    validator.data = {"val": str(tmp_path)}
    validator.init_metrics(SimpleNamespace(names=coco[2], end2end=False))
    assert validator.crowd_evaluator is None
    assert not isinstance(validator.metrics, CrowdDetectionMetrics)
    assert validator.args.plots and not validator.args.save_json
    assert validator.class_map == [1, 2]
