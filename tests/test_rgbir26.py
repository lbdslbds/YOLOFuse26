"""Regression coverage for the model and data boundaries changed by this migration."""
import os
from pathlib import Path
import sys

import cv2
import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.data.augment import Format, RandomHSV
from ultralytics.data.dataset import YOLODataset
from ultralytics.data.loaders import LoadImagesAndVideos
from ultralytics.data.rgbir import read_rgbir_image, rgb_to_ir_path
from ultralytics.nn.tasks import DetectionModel
from fusion_weights import initialize_from_yolo26

torch.set_num_threads(2)
FUSE = ROOT / "ultralytics/cfg/models/fuse26"
DEVICE = os.getenv("YOLOFUSE_TEST_DEVICE", "cpu")


@pytest.mark.parametrize("variant", ("decision", "easy", "mid", "early", "data", "defa", "deyolo"))
def test_yolo26_both_modalities_train_and_both_heads_infer(variant):
    torch.manual_seed(7)
    model = DetectionModel(str(FUSE / f"yolo26n-{variant}.yaml"), verbose=False).to(DEVICE)
    assert model.yaml["channels"] == 6
    assert model.model[-1].reg_max == 1
    assert model.stride.tolist() == ([8, 8, 16, 16, 32, 32] if variant == "decision" else [8, 16, 32])
    model.args = get_cfg(overrides={"epochs": 2})
    image = torch.rand(2, 6, 96, 96, device=DEVICE, requires_grad=True)
    batch = {"img": image, "batch_idx": torch.tensor([0, 1], device=DEVICE), "cls": torch.tensor([[0.], [2.]], device=DEVICE),
             "bboxes": torch.tensor([[.5, .5, .3, .4], [.4, .6, .2, .3]], device=DEVICE)}
    loss, items = model.loss(batch)
    assert torch.isfinite(loss).all()
    assert model.criterion.__class__.__name__ == "E2ELoss"
    loss.sum().backward()
    for modality in (image.grad[:, :3], image.grad[:, 3:]):
        assert torch.isfinite(modality).all() and modality.abs().sum() > 0
    assert sum(layer[-1].weight.grad.abs().sum() for layer in model.model[-1].one2one_cv3) > 0
    assert "dfl_loss" not in items and "l1_loss" in items
    model.eval()
    with torch.no_grad():
        model.end2end = True
        output = model(image.detach())[0]
        assert output.shape[0] == 2 and output.shape[-1] == 6
        model.end2end = False
        output, raw = model(image.detach())
        assert output.shape[1] == 8  # 4 box values + 4 dataset classes
        rgb_zero, ir_zero = image.detach().clone(), image.detach().clone()
        rgb_zero[:, :3] = 0
        ir_zero[:, 3:] = 0
        baseline = raw["one2many"]["scores"]
        assert not torch.equal(baseline, model(rgb_zero)[1]["one2many"]["scores"])
        assert not torch.equal(baseline, model(ir_zero)[1]["one2many"]["scores"])


@pytest.fixture
def pair(tmp_path):
    for name in ("images", "imagesIR", "labels"):
        (tmp_path / name / "train").mkdir(parents=True)
    rgb = np.full((64, 96, 3), (11, 43, 97), dtype=np.uint8)  # BGR
    ir = np.full((64, 96, 3), (151, 173, 211), dtype=np.uint8)
    file = tmp_path / "images/train/item.png"
    cv2.imwrite(str(file), rgb)
    cv2.imwrite(str(rgb_to_ir_path(file)), ir)
    (tmp_path / "labels/train/item.txt").write_text("0 0.5 0.5 0.25 0.5\n")
    return file


def dataset(file, cache=False):
    return YOLODataset(img_path=str(file.parent), imgsz=96, batch_size=1, augment=False,
                       hyp=get_cfg(), cache=cache, data={"nc": 4, "names": {i: str(i) for i in range(4)}, "channels": 6})


def test_pair_order_cache_and_thermal_invalidation(pair):
    # An old YOLOFuse cache must neither be consumed nor overwritten.
    old = pair.with_suffix(".npy")
    np.save(old, np.zeros((64, 96, 6), dtype=np.uint8))
    before = old.read_bytes()
    ds = dataset(pair, "disk")
    image = ds.load_image(0)[0]
    assert image[0, 0].tolist() == [97, 43, 11, 211, 173, 151]
    assert old.read_bytes() == before
    assert ds.npy_files[0].name.endswith(".rgbir26.npy")
    tensor = Format()._format_img(image)
    assert tensor[:, 0, 0].tolist() == [97, 43, 11, 211, 173, 151]
    loader = LoadImagesAndVideos(str(pair), channels=6)
    assert np.array_equal(next(iter(loader))[1][0], image)
    ir = rgb_to_ir_path(pair)
    cv2.imwrite(str(ir), np.full((64, 96, 3), 222, dtype=np.uint8))
    stamp = ds.npy_files[0].stat().st_mtime + 5
    os.utime(ir, (stamp, stamp))
    refreshed = dataset(pair, "disk").load_image(0)[0]
    assert refreshed[0, 0, 3:].tolist() == [222, 222, 222]
    assert old.read_bytes() == before


def test_missing_ir_fails_even_with_cache(pair):
    dataset(pair, "disk")
    rgb_to_ir_path(pair).unlink()
    with pytest.raises(FileNotFoundError, match="Missing IR"):
        dataset(pair, "disk")
    with pytest.raises(FileNotFoundError, match="Missing IR"):
        read_rgbir_image(pair)


def test_misalignment_and_invalid_pair_path_fail(pair):
    cv2.imwrite(str(rgb_to_ir_path(pair)), np.zeros((32, 32, 3), dtype=np.uint8))
    with pytest.raises(ValueError, match="registered"):
        read_rgbir_image(pair)
    with pytest.raises(ValueError, match="images"):
        rgb_to_ir_path("/data/my_images/train/item.png")


def test_hsv_keeps_thermal_values(pair):
    image = read_rgbir_image(pair)
    before = image[..., 3:].copy()
    RandomHSV(.5, .5, .5)({"img": image})
    assert np.array_equal(image[..., 3:], before)


def test_prediction_consistency_serialization_and_single_rgb(pair, tmp_path):
    model = YOLO(str(FUSE / "yolo26n-decision.yaml"))
    model.save(tmp_path / "fusion.pt")
    model = YOLO(str(tmp_path / "fusion.pt"))
    pair_image = read_rgbir_image(pair)
    kwargs = dict(device="cpu", imgsz=96, conf=0.00001, verbose=False, save=False, nms=False)
    a = model.predict(str(pair), **kwargs)[0]
    b = model.predict(pair_image, **kwargs)[0]
    assert a.orig_img[0, 0].tolist() == [11, 43, 97]
    assert a.ir.orig_img[0, 0].tolist() == [151, 173, 211]
    assert torch.allclose(a.boxes.data, b.boxes.data, atol=1e-5)
    assert torch.equal(a.boxes.data, a.ir.boxes.data)
    # NMS and one-to-one postprocessing must both work after reloading.
    model = YOLO(str(tmp_path / "fusion.pt"))
    model.predict(str(pair), **{**kwargs, "nms": None})
    with pytest.raises(ValueError, match="six channels"):
        model.predict(np.zeros((64, 96, 3), dtype=np.uint8), **kwargs)
    single = YOLO(str(ROOT / "ultralytics/cfg/models/26/yolo26.yaml"))
    assert single.model.yaml["channels"] == 3
    single.predict(np.zeros((64, 96, 3), dtype=np.uint8), **kwargs)


def test_pretrained_branches_are_mapped_by_role(tmp_path):
    source = YOLO(str(ROOT / "ultralytics/cfg/models/26/yolo26.yaml"))
    with torch.no_grad():
        source.model.model[0].conv.weight.fill_(0.125)
    path = tmp_path / "yolo26n.pt"
    source.save(path)
    target = YOLO(str(FUSE / "yolo26n-decision.yaml"))
    report = initialize_from_yolo26(target, path)
    assert report["copied_tensors"] > 200
    assert target.ckpt is not None  # public .train() preserves initialization
    for dst, src in target.model.yaml["pretrained_layers"].items():
        if src == 0:
            assert torch.equal(target.model.model[dst].conv.weight, source.model.model[0].conv.weight)


def test_wrong_dataset_channels_fail():
    with pytest.raises(ValueError, match="channels: 6"):
        DetectionModel(str(FUSE / "yolo26n-decision.yaml"), ch=3, verbose=False)
