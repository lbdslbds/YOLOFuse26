"""Validate a trained YOLOFuse26 checkpoint on paired RGB+IR images."""
import argparse
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--data", default=str(ROOT / "configs/rgbir.yaml"))
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--compile", nargs="?", const=True, default=False,
                        type=lambda value: {"true": True, "false": False}.get(value.lower(), value),
                        choices=(True, False, "default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"),
                        help="Enable torch.compile with Inductor; optionally select its mode.")
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.70)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--nms", action="store_true")
    parser.add_argument("--name", default="rgbir")
    args = parser.parse_args()
    model = YOLO(args.weights)
    if model.model.yaml.get("channels") != 6:
        raise SystemExit("Expected a six-channel YOLOFuse26 checkpoint")
    model.val(data=args.data, device=args.device, imgsz=args.imgsz, batch=args.batch, workers=args.workers,
              split=args.split, conf=args.conf, iou=args.iou, max_det=args.max_det,
              nms=None if args.nms else False, plots=True, save_json=True,
              project=str(ROOT / "runs/val"), name=args.name, compile=args.compile)
