"""Regressions for class-specific crowd ignore in the real detection criterion."""
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from ultralytics.utils.loss import v8DetectionLoss


def criterion():
    model = nn.Module()
    model.register_parameter("dummy", nn.Parameter(torch.zeros(1)))
    model.model = [SimpleNamespace(nc=2, reg_max=1, stride=torch.tensor([8.]))]
    model.args = SimpleNamespace(box=1., cls=1., dfl=1.)
    return v8DetectionLoss(model)


def empty_batch():
    return {"batch_idx": torch.empty(0), "cls": torch.empty(0, 1), "bboxes": torch.empty(0, 4)}


def prediction():
    return {"boxes": torch.ones(1, 4, 2, requires_grad=True),
            "scores": torch.zeros(1, 2, 2, requires_grad=True),
            "feats": [torch.zeros(1, 1, 1, 2)]}


def test_crowd_only_masks_its_class_and_does_not_regress_group_box():
    loss_fn, pred = criterion(), prediction()
    batch = {**empty_batch(), "crowd_bboxes": torch.tensor([[.25, .5, .5, 1.]]),
             "crowd_cls": torch.tensor([[1.]]), "crowd_batch_idx": torch.tensor([0.])}
    _, losses, _ = loss_fn.get_assigned_targets_and_loss(pred, batch)
    losses.sum().backward()
    grad = pred["scores"].grad
    assert grad[0, 1, 0] == 0  # bicycle-like crowd category, anchor inside
    assert grad[0, 0, 0] > 0  # other categories keep their negative supervision
    assert grad[0, 1, 1] > 0  # same category outside crowd remains supervised
    assert losses[0] == 0 and losses[2] == 0


def test_ordinary_positive_anchor_overrides_crowd_ignore():
    loss_fn = criterion()
    batch = {"crowd_bboxes": torch.tensor([[.5, .5, 1., 1.]]),
             "crowd_cls": torch.tensor([[1.]]), "crowd_batch_idx": torch.tensor([0.])}
    mask = loss_fn.crowd_classification_mask(batch, torch.tensor([[4., 4.], [12., 4.]]),
                                            torch.tensor([[True, False]]), torch.tensor([16., 8., 16., 8.]))
    assert mask[0, 0].all()
    assert mask[0, 1, 0] and not mask[0, 1, 1]


def test_empty_crowd_is_identical_to_ordinary_loss():
    loss_fn, batch, pred = criterion(), empty_batch(), prediction()
    ordinary = loss_fn.get_assigned_targets_and_loss(pred, batch)[1]
    batch.update(crowd_bboxes=torch.empty(0, 4), crowd_cls=torch.empty(0, 1), crowd_batch_idx=torch.empty(0))
    assert torch.equal(ordinary, loss_fn.get_assigned_targets_and_loss(pred, batch)[1])


def test_invalid_crowd_class_fails():
    batch = {"crowd_bboxes": torch.tensor([[.5, .5, 1., 1.]]),
             "crowd_cls": torch.tensor([[2.]]), "crowd_batch_idx": torch.tensor([0.])}
    with pytest.raises(ValueError, match="Invalid"):
        criterion().crowd_classification_mask(batch, torch.tensor([[4., 4.]]), torch.tensor([[False]]),
                                             torch.tensor([8., 8., 8., 8.]))
