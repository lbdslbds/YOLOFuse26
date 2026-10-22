"""Regression checks for export-safe fusion and graph-capturable anchor generation."""
from copy import deepcopy
from pathlib import Path
import sys

import pytest
import torch
import torch.nn.functional as F
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ultralytics.nn.modules.DEFA import DEFA
from ultralytics.utils.tal import make_anchors


def test_defa_matches_original_fold_forward_and_backward():
    torch.manual_seed(41)
    modern = DEFA(channels=16, dynamic_k=3).train()
    legacy = deepcopy(modern)
    inputs = {}
    modern.enhance.register_forward_pre_hook(lambda module, args: inputs.update(modern=args[0].detach().clone()))
    legacy.enhance.register_forward_pre_hook(lambda module, args: inputs.update(legacy=args[0].detach().clone()))
    visible = torch.randn(2, 16, 6, 8, requires_grad=True)
    infrared = torch.randn_like(visible, requires_grad=True)
    rv, ri = visible.detach().clone().requires_grad_(), infrared.detach().clone().requires_grad_()
    actual = modern((visible, infrared))
    # Retain the original forward path as a numerical/gradient compatibility oracle.
    def legacy_forward(inputs):
        vi, ir = inputs
        weights = legacy.dynamic_conv(vi + ir)
        vi_seq, ir_seq = vi.flatten(2).transpose(1,2), ir.flatten(2).transpose(1,2)
        attended, _ = legacy.cross_attn(vi_seq, ir_seq, ir_seq, need_weights=False)
        feature = attended.transpose(1,2).reshape_as(vi)
        unfolded = F.unfold(feature, legacy.dynamic_k, padding=legacy.dynamic_k//2)
        b, _, n = unfolded.shape
        dynamic = torch.einsum('bkn,bckn->bcn', weights.view(b,9,1).expand(-1,-1,n),
                               unfolded.view(b,-1,9,n))
        dynamic = F.fold(dynamic, vi.shape[2:], kernel_size=1, stride=1)
        gates = legacy.gate(torch.cat((vi,ir),dim=1))
        return legacy.enhance((gates[:,1:2]*dynamic+vi)+(gates[:,0:1]*dynamic+ir))
    expected = legacy_forward((rv, ri))
    torch.testing.assert_close(inputs['modern'], inputs['legacy'], rtol=0, atol=0)
    # Reshape retains the contraction's channels-last strides; Fold materialized NCHW.
    # The following convolution may therefore use a different FP32 accumulation order.
    torch.testing.assert_close(actual, expected, rtol=2e-6, atol=1e-6)
    actual.square().mean().backward()
    expected.square().mean().backward()
    torch.testing.assert_close(visible.grad, rv.grad, rtol=1e-5, atol=1e-7)
    torch.testing.assert_close(infrared.grad, ri.grad, rtol=1e-5, atol=1e-7)
    for (name, parameter), (old_name, old_parameter) in zip(modern.named_parameters(), legacy.named_parameters()):
        assert name == old_name
        torch.testing.assert_close(parameter.grad, old_parameter.grad, rtol=1e-5, atol=1e-7)


@pytest.mark.skipif(not hasattr(torch,'compile'), reason='Requires torch.compile')
@pytest.mark.parametrize('levels', [3,6])
def test_tensor_strides_are_captured_in_one_graph(levels):
    torch.manual_seed(43)
    values = [8.,16.,32.] if levels == 3 else [8.,8.,16.,16.,32.,32.]
    feats = [torch.rand(1,4,int(64/s),int(64/s)) for s in values]
    strides = torch.tensor(values)
    expected_anchors, expected_strides = make_anchors(feats, strides)
    compiled = torch.compile(make_anchors, backend='eager', fullgraph=True)
    anchors, actual_strides = compiled(feats, strides)
    torch.testing.assert_close(anchors, expected_anchors, rtol=0, atol=0)
    torch.testing.assert_close(actual_strides, expected_strides, rtol=0, atol=0)
    manual = torch.cat([torch.full((f.shape[-2]*f.shape[-1],1),s) for f,s in zip(feats,values)])
    torch.testing.assert_close(actual_strides, manual, rtol=0, atol=0)


@pytest.mark.parametrize('fusion', ['decision', 'defa'])
def test_rgbir_gflops_use_external_six_channel_input(fusion):
    import importlib
    thop = pytest.importorskip('thop')
    if 'stride' not in __import__('inspect').signature(thop.profile).parameters:
        pytest.skip('Requires stride-aware ultralytics-thop')
    if not getattr(importlib.import_module('thop.profile'), '_COUNTS_FUNCTIONS', False):
        pytest.skip('Requires functional attention counting in ultralytics-thop>=2.2.3')
    from ultralytics import YOLO
    from ultralytics.utils.torch_utils import get_flops
    net = YOLO(str(ROOT / f'ultralytics/cfg/models/fuse26/yolo26n-{fusion}.yaml')).model.eval()
    net.fuse(verbose=False)
    # Both branches start with a 3-channel convolution; the external pair has 6 channels.
    assert next(net.parameters()).shape[1] == 3 and net.yaml['channels'] == 6
    macs, _ = thop.profile(net, inputs=(torch.zeros(1,6,128,128),), verbose=False)
    measured = get_flops(net, 128)
    assert measured > 0
    assert measured == pytest.approx(2 * macs / 1e9, rel=1e-9)


@pytest.mark.parametrize('variant', ['valid_boundary','wrong_class','outside_boundary'])
def test_topk_boundary_comparison_preserves_candidate_classes_and_cutoff(variant):
    from tools.validate_modern_runtime import compare_outputs
    raw = np.zeros((1,3,8),dtype=np.float32)
    raw[0,:,:4] = [[0,0,1,1],[2,2,3,3],[4,4,5,5]]
    raw[0,:,4:] = [[.9,.1,.1,.1],[.5,.1,.1,.1],[.01,.50000006,.01,.01]]
    expected = np.array([[[0,0,1,1,.9,0],[2,2,3,3,.5,0]]],dtype=np.float32)
    actual = np.array([[[0,0,1,1,.9,0],[4,4,5,5,.50000006,1]]],dtype=np.float32)
    if variant == 'valid_boundary':
        assert compare_outputs(actual,expected,'one2one',raw)['topk_boundary_ties'] == 1
    else:
        actual[0,1,5 if variant == 'wrong_class' else 4] = 2 if variant == 'wrong_class' else .8
        with pytest.raises(AssertionError):
            compare_outputs(actual,expected,'one2one',raw)
