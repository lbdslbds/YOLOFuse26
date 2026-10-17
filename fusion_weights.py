"""Transfer official YOLO26 weights by graph role, never by shifted layer indices."""
from pathlib import Path

import torch

from ultralytics import YOLO


def initialize_from_yolo26(model, weights):
    """Seed both modality branches and both detector heads from a standard YOLO26 checkpoint."""
    source = YOLO(str(weights)).model.float()
    target = model.model
    if source.yaml.get("rgbir") or source.model[-1].reg_max != 1:
        raise ValueError("Use an official single-modality YOLO26 pretrained checkpoint; old YOLOFuse weights are incompatible")
    source_scale = source.yaml.get("scale") or next(iter(source.yaml.get("scales", {"n": None})))
    target_scale = target.yaml.get("scale")
    if source_scale != target_scale:
        raise ValueError(f"Pretrained scale {source_scale!r} does not match requested scale {target_scale!r}")
    source_head, target_head = source.model[-1], target.model[-1]
    target_state = target.state_dict()
    copied, skipped = [], []
    for target_index, source_index in target.yaml["pretrained_layers"].items():
        target_layer = target.model[int(target_index)]
        source_layer = source.model[int(source_index)]
        if type(target_layer) is not type(source_layer):
            raise ValueError(f"Pretrained graph mismatch at {target_index} <- {source_index}")
        for name, value in source_layer.state_dict().items():
            key = f"model.{target_index}.{name}"
            if target_state[key].shape == value.shape:
                target_state[key] = value
                copied.append(key)
            elif target.yaml["fusion"] == "data" and int(target_index) == 0 and name == "conv.weight":
                target_state[key] = value.repeat(1, 2, 1, 1) / 2
                copied.append(key)
            else:
                skipped.append(key)
    for head_name in ("cv2", "cv3", "one2one_cv2", "one2one_cv3"):
        src = getattr(source_head, head_name, None)
        dst = getattr(target_head, head_name, None)
        if src is None or dst is None:
            raise ValueError(f"Pretrained checkpoint does not contain both YOLO26 training heads: {head_name}")
        for index, source_index in enumerate(target.yaml["pretrained_head_scales"]):
            for name, value in src[source_index].state_dict().items():
                key = f"model.{len(target.model) - 1}.{head_name}.{index}.{name}"
                # Dataset classes differ from COCO: do not infer a class correspondence by shape.
                class_logits = head_name.endswith("cv3") and name in ("2.weight", "2.bias")
                if target_state[key].shape == value.shape and not class_logits:
                    target_state[key] = value
                    copied.append(key)
                else:
                    skipped.append(key)
    target.load_state_dict(target_state, strict=True)
    # The public object-load API marks these weights for the trainer's model rebuild.
    model.load(target)
    return dict(source=str(Path(weights)), copied_tensors=len(copied), skipped_tensors=len(skipped),
                copied_parameters=sum(target_state[k].numel() for k in copied),
                total_parameters=sum(v.numel() for v in target_state.values()),
                classification="Dataset classification logits initialized afresh", layer_map=target.yaml["pretrained_layers"])
