"""YOLOFuse modality routing without changing the YOLO26 execution graph."""
import torch
from torch import nn


class IdentityInput(nn.Identity):
    """Retain the six-channel input so both branches can select it."""


class ModalitySelector(nn.Module):
    """Select RGB (1) or thermal (2) from the canonical six-channel tensor."""

    def __init__(self, out=1):
        super().__init__()
        if out not in (1, 2):
            raise ValueError("out must be 1 (RGB) or 2 (IR)")
        self.out = out

    def forward(self, x):
        return x[:, :3] if self.out == 1 else x[:, 3:6]


class ModalConcat(nn.Module):
    """Concatenate modality tensors, preserving their order."""

    def __init__(self, dimension=1):
        super().__init__()
        self.dimension = dimension

    def forward(self, x):
        return torch.cat(x, self.dimension) if isinstance(x, (list, tuple)) else x
