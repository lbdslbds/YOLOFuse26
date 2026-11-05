"""Train YOLO26 RGB+IR fusion on a paired YOLO dataset."""
import argparse
import json
from pathlib import Path

import torch
import yaml

from ultralytics import YOLO
from fusion_weights import initialize_from_yolo26

ROOT = Path(__file__).resolve().parent
VARIANTS = ("decision", "easy", "mid", "early", "data", "defa", "deyolo", "defa-mala-core")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fusion", choices=VARIANTS, default="decision")
    parser.add_argument("--scale", choices=list("nsmlx"), default="n")
    parser.add_argument("--data", default=str(ROOT / "configs/rgbir.yaml"))
    parser.add_argument("--weights", help="Official YOLO26 weights; defaults to weights/yolo26{scale}.pt")
    parser.add_argument("--scratch", action="store_true")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=0,
                        help="Epochs without validation improvement before early stopping; 0 disables it. Resume keeps the checkpoint setting.")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="0")
    parser.add_argument("--optimizer", default="MuSGD", choices=("MuSGD", "SGD", "AdamW", "auto"))
    parser.add_argument("--cache", choices=("false", "disk", "ram"), default="false")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True,
                        help="Enable AMP for new training runs (default); --no-amp uses FP32. Resume keeps checkpoint precision.")
    parser.add_argument("--compile", nargs="?", const=True, default=None,
                        type=lambda value: {"true": True, "false": False}.get(value.lower(), value),
                        choices=(True, False, "default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"),
                        help="Enable torch.compile with Inductor; optionally select its mode.")
    parser.add_argument("--name")
    parser.add_argument("--fraction", type=float, default=1.0)
    parser.add_argument("--nms", action="store_true", help="Validate the one-to-many NMS head instead of the end-to-end head")
    args = parser.parse_args()
    if args.patience < 0:
        parser.error("--patience must be nonnegative")
    if args.device != "cpu" and not torch.cuda.is_available():
        raise SystemExit("No CUDA GPU is available. Enable GPU mode, or pass --device cpu for a small smoke run.")
    data = yaml.safe_load(Path(args.data).read_text())
    if data.get("channels") != 6:
        raise SystemExit("The dataset YAML must contain channels: 6 for paired RGB+IR training.")
    if args.resume:
        options = {"compile": args.compile} if args.compile is not None else {}
        YOLO(str(args.resume)).train(resume=True, device=args.device, **options)
        return
    config = ROOT / f"ultralytics/cfg/models/fuse26/yolo26{args.scale}-{args.fusion}.yaml"
    model = YOLO(str(config))
    if not args.scratch:
        weights = args.weights or ROOT / f"weights/yolo26{args.scale}.pt"
        transfer = initialize_from_yolo26(model, weights)
        (ROOT / "reports").mkdir(exist_ok=True)
        (ROOT / "reports" / f"pretrained_{args.scale}_{args.fusion}.json").write_text(json.dumps(transfer, indent=2))
    model.train(data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, workers=args.workers,
                device=args.device, optimizer=args.optimizer, cache=False if args.cache == "false" else args.cache,
                project=str(ROOT / "runs/train"), name=args.name or f"yolo26{args.scale}-{args.fusion}",
                nms=None if args.nms else False, seed=0, deterministic=True, amp=args.amp, patience=args.patience,
                close_mosaic=0, fraction=args.fraction, plots=True, compile=args.compile or False)


if __name__ == "__main__":
    main()
