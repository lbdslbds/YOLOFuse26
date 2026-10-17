"""Shared, strict RGB+IR input contract for YOLOFuse26 training and inference."""
from pathlib import Path

import cv2
import numpy as np

from ultralytics.utils.patches import imread


def rgb_to_ir_path(file):
    """Resolve a paired image by replacing exactly one directory named images."""
    path = Path(file)
    parts = list(path.parts)
    indices = [i for i, part in enumerate(parts[:-1]) if part == "images"]
    if len(indices) != 1:
        raise ValueError(f"Expected exactly one 'images' directory for RGB+IR input: {file}")
    parts[indices[0]] = "imagesIR"
    return Path(*parts)


def read_rgbir_image(file):
    """Return HWC uint8 [RGB_R,G,B, IR_R,G,B]; never substitute visible for thermal."""
    infrared = rgb_to_ir_path(file)
    if not infrared.is_file():
        raise FileNotFoundError(f"Missing IR counterpart: {infrared}")
    rgb = imread(str(file), flags=cv2.IMREAD_COLOR)
    ir = imread(str(infrared), flags=cv2.IMREAD_COLOR)
    if rgb is None or ir is None:
        raise ValueError(f"Cannot decode paired RGB/IR images: {file}, {infrared}")
    if rgb.shape != ir.shape:
        raise ValueError(f"RGB/IR must be spatially registered with equal shapes: {rgb.shape} != {ir.shape}")
    return np.ascontiguousarray(np.concatenate((rgb[..., ::-1], ir[..., ::-1]), axis=2))
