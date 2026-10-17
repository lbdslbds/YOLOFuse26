"""Predict paired RGB+IR images and save the same fused detections on both modalities."""
import argparse
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--source", required=True, help="RGB image, directory, or TXT list under an images directory")
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--nms", action="store_true")
    parser.add_argument("--name", default="rgbir")
    args = parser.parse_args()
    model = YOLO(args.weights)
    if model.model.yaml.get("channels") != 6:
        raise SystemExit("Expected a six-channel YOLOFuse26 checkpoint")
    for result in model.predict(source=args.source, device=args.device, imgsz=args.imgsz, batch=args.batch,
                                conf=args.conf, iou=args.iou, nms=None if args.nms else False, stream=True,
                                save=True, save_txt=True, save_conf=True,
                                project=str(ROOT / "runs/predict"), name=args.name):
        target = Path(result.save_dir) / "ir"
        target.mkdir(exist_ok=True)
        result.ir.save(filename=str(target / Path(result.path).name))
