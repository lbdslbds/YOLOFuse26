"""Generate fusion graphs from the vendored, unmodified official YOLO26 YAML."""
from copy import deepcopy
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "ultralytics/cfg/models/26/yolo26.yaml"
OUTPUT = ROOT / "ultralytics/cfg/models/fuse26"
VARIANTS = ("decision", "easy", "mid", "early", "data", "defa", "deyolo")


def build(variant, scale="n"):
    """Compose official YOLO26 stages with the selected YOLOFuse fusion topology."""
    official = yaml.safe_load(SOURCE.read_text())
    layers, mapping = [], {}

    def add(layer, source=None):
        index = len(layers)
        layers.append(deepcopy(layer))
        if source is not None:
            mapping[index] = source
        return index

    def backbone(selector, stop=11):
        branch = {}
        for old, layer in enumerate(official["backbone"][:stop]):
            layer = deepcopy(layer)
            layer[0] = selector if old == 0 else -1
            branch[old] = add(layer, old)
        return branch

    def neck(branch):
        branch = dict(branch)
        for old, layer in enumerate(official["head"][:-1], start=11):
            layer = deepcopy(layer)
            refs = layer[0] if isinstance(layer[0], list) else [layer[0]]
            refs = [branch[old - 1] if ref == -1 else branch[ref] for ref in refs]
            layer[0] = refs if isinstance(layer[0], list) else refs[0]
            branch[old] = add(layer, old)
        return [branch[i] for i in (16, 19, 22)]

    if variant == "data":
        branch = backbone(-1)
        boundary = len(layers)
        outputs = neck(branch)
    else:
        add([-1, 1, "IdentityInput", []])
        rgb_in = add([0, 1, "ModalitySelector", [1]])
        ir_in = add([0, 1, "ModalitySelector", [2]])
        rgb = backbone(rgb_in, 3 if variant == "early" else 11)
        ir = backbone(ir_in, 3 if variant == "early" else 11)
        if variant == "early":
            joined = add([[rgb[2], ir[2]], 1, "Concat", [1]])
            branch = {2: add([joined, 1, "Conv", [256, 1, 1]])}
            for old, layer in enumerate(official["backbone"][3:], start=3):
                branch[old] = add(layer, old)
            boundary = len(layers)
            outputs = neck(branch)
        elif variant == "decision":
            boundary = len(layers)
            rgb_out, ir_out = neck(rgb), neck(ir)
            outputs = [index for pair in zip(rgb_out, ir_out) for index in pair]
        else:
            boundary = len(layers)
            fused = {}
            for level, channels in ((4, 512), (6, 512), (10, 1024)):
                if variant in ("defa", "deyolo"):
                    module = "DEFA" if variant == "defa" else "DEA"
                    args = [] if module == "DEFA" else [{4: 80, 6: 40, 10: 20}[level]]
                    fused[level] = add([[rgb[level], ir[level]], 1, module, args])
                else:
                    joined = add([[rgb[level], ir[level]], 1, "Concat", [1]])
                    width = (256 if level == 4 else channels) if variant == "easy" else channels
                    fused[level] = add([joined, 2, "C3k2", [width, True]])
            outputs = [fused[i] for i in (4, 6, 10)] if variant == "easy" else neck(fused)
    add([outputs, 1, "Detect", ["nc"]])
    config = dict(nc=4, channels=6, ch=6, rgbir=True, scale=scale,
                  end2end=True, reg_max=1, scales=official["scales"],
                  fusion=variant, pretrained_layers=mapping,
                  pretrained_head_scales=[0, 0, 1, 1, 2, 2] if variant == "decision" else [0, 1, 2],
                  backbone=layers[:boundary], head=layers[boundary:])
    return config


if __name__ == "__main__":
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for variant in VARIANTS:
        for scale in "nsmlx":
            path = OUTPUT / f"yolo26{scale}-{variant}.yaml"
            path.write_text("# YOLOFuse26: RGB+IR on official Ultralytics 8.4.180 YOLO26\n" +
                            yaml.safe_dump(build(variant, scale), sort_keys=False), encoding="utf-8")
    print(f"Generated {len(VARIANTS) * 5} fusion configurations in {OUTPUT}")
