"""Audit real paired data and run a small CPU or CUDA integration check."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("YOLO_AUTOINSTALL", "false")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import cv2
import numpy as np
from PIL import Image
import torch
import yaml
from ultralytics import YOLO, __version__
from ultralytics.data.rgbir import read_rgbir_image, rgb_to_ir_path
from fusion_weights import initialize_from_yolo26

torch.set_num_threads(2)
OUT = None


def dump(name, content):
    (OUT / name).write_text(json.dumps(content, indent=2, ensure_ascii=False) + "\n")


def main():
    global OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--imgsz", type=int, default=128)
    args = parser.parse_args()
    OUT = ROOT / "reports" / ("cpu_smoke" if args.device == "cpu" else "gpu_smoke")
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.time()
    data = yaml.safe_load((ROOT / "configs/rgbir.yaml").read_text())
    source = Path(data["path"])
    audit, selected = {}, {}
    for split, size in (("train", 8), ("val", 4)):
        files = sorted((source / "images" / split).glob("*.jpg"))
        irfiles = sorted((source / "imagesIR" / split).glob("*.jpg"))
        assert [f.name for f in files] == [f.name for f in irfiles]
        assert files
        labels_sha, classes, domains = hashlib.sha256(), Counter(), Counter()
        for file, irfile in zip(files, irfiles):
            with Image.open(file) as rgb, Image.open(irfile) as ir:
                assert rgb.size == ir.size, (file, rgb.size, ir.size)
            label = source / "labels" / split / (file.stem + ".txt")
            text = label.read_text()
            labels_sha.update(file.stem.encode() + b"\0" + text.encode())
            for line in text.splitlines():
                fields = [float(x) for x in line.split()]
                assert len(fields) == 5 and fields[0] == int(fields[0]) and 0 <= fields[0] < 4
                assert all(0 <= x <= 1 for x in fields[1:])
                classes[int(fields[0])] += 1
            domains[file.stem.split("_")[0]] += 1
        audit[split] = dict(pairs=len(files), instances=sum(classes.values()), classes=dict(classes),
                            domains=dict(domains), labels_sha256=labels_sha.hexdigest(), matching_dimensions=True)
        chosen = []
        for domain in ("day", "night"):
            candidates = [f for f in files if f.stem.startswith(domain)]
            chosen.extend(candidates[i] for i in np.linspace(0, len(candidates) - 1, size // 2, dtype=int))
        selected[split] = [str(f) for f in chosen]
        for file in chosen:
            for folder, original in (("images", file), ("imagesIR", rgb_to_ir_path(file)),
                                     ("labels", source / "labels" / split / (file.stem + ".txt"))):
                dest = OUT / "dataset" / folder / split / original.name
                dest.parent.mkdir(parents=True, exist_ok=True)
                if not dest.exists():
                    os.link(original, dest)
    dump("dataset_audit.json", audit)
    dump("samples.json", selected)
    print("Paired dataset audit passed:", {k: v["pairs"] for k, v in audit.items()}, flush=True)
    smoke_data = {**data, "path": str(OUT / "dataset")}
    yaml_file = OUT / "data.yaml"
    yaml_file.write_text(yaml.safe_dump(smoke_data, sort_keys=False))

    model = YOLO(str(ROOT / "ultralytics/cfg/models/fuse26/yolo26n-decision.yaml"))
    dump("pretrained_transfer.json", initialize_from_yolo26(model, ROOT / "weights/yolo26n.pt"))
    # Reuse the initialized weights through the normal trainer rebuild.
    initial = model.model.model[3].conv.weight.detach().clone()
    captured = {}

    def begin(trainer):
        torch.set_num_threads(2)
        assert torch.allclose(trainer.model.model[3].conv.weight.cpu(), initial)
        captured["initialization_preserved"] = True

    def batch_check(trainer):
        assert trainer.model.yaml["channels"] == 6

    model.add_callback("on_pretrain_routine_end", begin)
    model.add_callback("on_train_batch_end", batch_check)
    model.train(data=str(yaml_file), epochs=1, imgsz=args.imgsz, batch=2, workers=0, device=args.device,
                optimizer="MuSGD", cache="disk", amp=False, seed=0, deterministic=True, nms=False,
                project=str(OUT), name="train", exist_ok=True, plots=True, close_mosaic=0, patience=0)
    best = OUT / "train/weights/best.pt"
    assert best.is_file()
    restored = YOLO(str(best))
    # Both output modes go through the actual validator, including real labels.
    end2end = restored.val(data=str(yaml_file), device=args.device, imgsz=args.imgsz, batch=2, workers=0,
                           nms=False, project=str(OUT), name="val_end2end", exist_ok=True, plots=True)
    many = YOLO(str(best)).val(data=str(yaml_file), device=args.device, imgsz=args.imgsz, batch=2, workers=0,
                              nms=None, project=str(OUT), name="val_nms", exist_ok=True, plots=True)
    single_path = next((OUT / "dataset/images/val").glob("*.jpg"))
    prediction = YOLO(str(best)).predict(str(single_path), device=args.device, imgsz=args.imgsz, nms=False,
                                        conf=.001, verbose=False)[0]
    prediction.save(str(OUT / "prediction_rgb.jpg"))
    prediction.ir.save(str(OUT / "prediction_ir.jpg"))
    assert torch.equal(prediction.boxes.data, prediction.ir.boxes.data)

    # Export a six-channel TorchScript model and check numerical parity on a real pair.
    export_model = YOLO(str(best))
    torchscript = export_model.export(format="torchscript", imgsz=128, batch=1, device="cpu", nms=False)
    traced = torch.jit.load(torchscript, map_location="cpu")
    pair = cv2.resize(read_rgbir_image(single_path), (128, 128))
    tensor = torch.from_numpy(pair.transpose(2, 0, 1).copy()).float()[None] / 255
    with torch.no_grad():
        actual = traced(tensor)
        reference = export_model.model.eval()(tensor)
        reference = reference[0] if isinstance(reference, tuple) else reference
    assert actual.shape[0] == 1 and actual.shape[-1] == 6
    torch.testing.assert_close(actual, reference, rtol=1e-3, atol=1e-3)
    dump("result.json", dict(status="passed", device=args.device, cuda_available=torch.cuda.is_available(),
                             python=platform.python_version(), torch=torch.__version__, ultralytics=__version__,
                             elapsed_seconds=round(time.time() - started, 2), train_pairs=8, validation_pairs=4,
                             epochs=1, imgsz=args.imgsz, optimizer="MuSGD", checkpoint=str(best),
                             nms_free=True, traditional_nms=True, torchscript=torchscript,
                             torchscript_shape=list(actual.shape), channel_order=["RGB_R", "RGB_G", "RGB_B", "IR_R", "IR_G", "IR_B"],
                             **captured,
                             note="Integration smoke only. These sample metrics are not a full-dataset accuracy evaluation."))
    print("SMOKE_PASSED", args.device, flush=True)


if __name__ == "__main__":
    main()
