"""Crowd sidecars stay separate from positive targets and follow the same image geometry."""
from copy import deepcopy
from pathlib import Path
import random
import sys

import cv2
import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ultralytics.cfg import get_cfg
from ultralytics.data.augment import Format, LetterBox, RandomFlip, RandomPerspective
from ultralytics.data.dataset import YOLODataset
from ultralytics.utils.instance import Instances


@pytest.fixture
def crowd_data(tmp_path):
    for directory in ("images/train", "labels/train", "crowd_labels/train"):
        (tmp_path / directory).mkdir(parents=True)
    cv2.imwrite(str(tmp_path / "images/train/item.png"), np.zeros((32, 64, 3), dtype=np.uint8))
    (tmp_path / "labels/train/item.txt").write_text("0 0.25 0.25 0.2 0.2\n", encoding="utf-8")
    (tmp_path / "crowd_labels/train/item.txt").write_text("1 0.75 0.75 0.2 0.2\n", encoding="utf-8")
    return tmp_path


def dataset(root, *, crowd=True, augment=False, hyp=None, **kwargs):
    data = {"path": str(root), "nc": 2, "names": {0: "person", 1: "rider"}}
    if crowd:
        data["crowd_labels"] = "crowd_labels"
    return YOLODataset(img_path=str(root / "images/train"), imgsz=64, batch_size=1, augment=augment,
                       hyp=hyp or get_cfg(), data=data, cache=False, **kwargs)


def test_sidecars_cache_invalidation_and_batch_indices(crowd_data):
    first = dataset(crowd_data)
    initial_hash = first.get_cache_hash()
    sample = first[0]
    assert sample["cls"].tolist() == [[0.0]]
    assert sample["crowd_cls"].tolist() == [[1.0]]
    torch.testing.assert_close(sample["crowd_bboxes"], torch.tensor([[0.75, 0.625, 0.2, 0.1]]))
    batch = YOLODataset.collate_fn([sample, deepcopy(sample)])
    assert batch["batch_idx"].tolist() == [0, 1]
    assert batch["crowd_batch_idx"].tolist() == [0, 1]
    # The edit preserves byte length, which must still invalidate the label cache.
    sidecar = crowd_data / "crowd_labels/train/item.txt"
    sidecar.write_text("1 0.25 0.75 0.2 0.2\n", encoding="utf-8")
    second = dataset(crowd_data)
    assert initial_hash != second.get_cache_hash()
    assert second[0]["crowd_bboxes"][0, 0].item() == pytest.approx(0.25)
    assert first.labels[0]["crowd_bboxes"][0, 0] == pytest.approx(0.75)


@pytest.mark.parametrize("crowd_text", ["", "1 0.75 0.75 0.2 0.2\n"])
def test_empty_ordinary_labels_and_empty_sidecars(crowd_data, crowd_text):
    (crowd_data / "labels/train/item.txt").write_text("", encoding="utf-8")
    (crowd_data / "crowd_labels/train/item.txt").write_text(crowd_text, encoding="utf-8")
    sample = dataset(crowd_data)[0]
    assert sample["cls"].shape == (0, 1)
    assert sample["bboxes"].shape == (0, 4)
    assert sample["batch_idx"].shape == (0,)
    assert sample["crowd_bboxes"].shape == (int(bool(crowd_text)), 4)
    batch = YOLODataset.collate_fn([sample, deepcopy(sample)])
    assert batch["crowd_batch_idx"].tolist() == ([0, 1] if crowd_text else [])


def test_missing_sidecar_fails_with_warm_cache(crowd_data):
    dataset(crowd_data)
    (crowd_data / "crowd_labels/train/item.txt").unlink()
    with pytest.raises(FileNotFoundError, match="Missing crowd sidecar"):
        dataset(crowd_data)


@pytest.mark.parametrize("text", ["0 0.5 0.5 0.2", "-1 0.5 0.5 0.2 0.2", "2 0.5 0.5 0.2 0.2",
                                  "0.5 0.5 0.5 0.2 0.2", "0 nan 0.5 0.2 0.2", "0 0.5 0.5 0 0.2",
                                  "0 1.2 0.5 0.2 0.2"])
def test_invalid_sidecars_fail_explicitly(crowd_data, text):
    (crowd_data / "crowd_labels/train/item.txt").write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid crowd sidecar"):
        dataset(crowd_data)


def test_class_selection_single_cls_and_opt_in_boundary(crowd_data):
    sample = dataset(crowd_data, single_cls=True)[0]
    assert sample["cls"].tolist() == [[0.0]]
    assert sample["crowd_cls"].tolist() == [[0.0]]
    assert dataset(crowd_data, classes=[0])[0]["crowd_cls"].numel() == 0
    plain = dataset(crowd_data, crowd=False)[0]
    assert not any(key.startswith("crowd_") for key in plain)
    with pytest.raises(ValueError, match="task='detect'"):
        dataset(crowd_data, task="segment")


@pytest.mark.parametrize("option", [{"copy_paste": 1.0}, {"cutmix": 1.0}, {"augmentations": [object()]}])
def test_unsupported_augmentations_fail_instead_of_misaligning_regions(crowd_data, option):
    hyp = get_cfg()
    for key, value in option.items():
        setattr(hyp, key, value)
    with pytest.raises(ValueError, match="crowd_labels"):
        dataset(crowd_data, augment=True, hyp=hyp)


def labels(boxes, classes, *, shape=(32, 64), normalized=True, bbox_format="xywh"):
    return {"img": np.zeros((*shape, 6), dtype=np.uint8), "cls": np.array(classes, dtype=np.float32).reshape(-1, 1),
            "instances": Instances(np.array(boxes, dtype=np.float32).reshape(-1, 4),
                                   segments=np.zeros((0, 1000, 2), dtype=np.float32),
                                   bbox_format=bbox_format, normalized=normalized)}


def test_six_channel_letterbox_and_both_flips_use_one_geometry():
    sample = labels([[0.25, 0.25, 0.2, 0.2], [0.75, 0.75, 0.2, 0.2]], [0, -2])
    sample = LetterBox(new_shape=(64, 64), scaleup=False)(sample)
    sample = RandomFlip(p=1.0, direction="horizontal")(sample)
    sample = RandomFlip(p=1.0, direction="vertical")(sample)
    sample = Format(crowd=True)(sample)
    assert sample["img"].shape == (6, 64, 64)
    torch.testing.assert_close(sample["bboxes"], torch.tensor([[0.75, 0.625, 0.2, 0.1]]))
    torch.testing.assert_close(sample["crowd_bboxes"], torch.tensor([[0.25, 0.375, 0.2, 0.1]]))
    assert sample["crowd_cls"].tolist() == [[1.0]]


def test_perspective_keeps_visible_thin_crowd_but_removes_invisible_regions():
    sample = labels([[0, 0, 1, 20], [120, 120, 130, 130], [0, 0, 1, 20], [10, 10, 30, 30]],
                    [-2, -2, 0, 0], shape=(64, 64), normalized=False, bbox_format="xyxy")
    sample = RandomPerspective().apply_instances(sample, {"M": np.eye(3), "scale": 1.0,
                                                           "orig_shape": (64, 64), "size": (64, 64)})
    sample = Format(crowd=True)(sample)
    assert sample["cls"].tolist() == [[0.0]]
    assert sample["crowd_cls"].tolist() == [[1.0]]
    assert sample["crowd_bboxes"][0, 2].item() == pytest.approx(1 / 64)


def test_mosaic_mixup_and_perspective_keep_crowd_registered(crowd_data):
    random.seed(7)
    np.random.seed(7)
    (crowd_data / "crowd_labels/train/item.txt").write_text("1 0.25 0.25 0.2 0.2\n", encoding="utf-8")
    hyp = get_cfg(overrides={"mosaic": 1.0, "mixup": 1.0, "copy_paste": 0.0, "cutmix": 0.0,
                             "degrees": 10.0, "translate": 0.1, "scale": 0.2, "fliplr": 1.0, "flipud": 1.0})
    sample = dataset(crowd_data, augment=True, hyp=hyp)[0]
    assert sample["bboxes"].shape[0] > 0
    assert sample["crowd_bboxes"].shape[0] >= sample["bboxes"].shape[0]
    for box in sample["bboxes"]:
        assert torch.isclose(box, sample["crowd_bboxes"], atol=1e-5).all(dim=1).any()
    assert (sample["cls"] >= 0).all() and (sample["crowd_cls"] == 1).all()
