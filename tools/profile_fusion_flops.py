"""Count full-resolution RGB+IR inference MACs; requires ultralytics-thop>=2.2.3.

Uses the YOLOFuse convention GFLOPs = 2 * THOP MACs / 1e9. Profiles an actual
640x640 pair rather than extrapolating a small image. THOP 2.2.3 accounts for
functional products, MultiheadAttention projections, QK/AV, and DEFA's einsum.
"""
import argparse
from importlib import import_module
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import thop
import torch
from torch import nn

from ultralytics import YOLO
from ultralytics.utils.torch_utils import get_flops


def attention_cost(module, inputs, output):
    """Independently break down an executed cross-attention call's MACs."""
    query, key, value = inputs
    assert query.ndim == key.ndim == value.ndim == 3 and module.batch_first
    assert module.bias_k is None and not module.add_zero_attn
    batch, nq, channels = query.shape
    nk = key.shape[1]
    projections = batch * (2 * nq * channels**2 + nk * channels * (module.kdim + module.vdim))
    products = 2 * batch * nq * nk * channels
    proxy = SimpleNamespace(**{name: getattr(module, name) for name in (
        'batch_first', 'bias_k', 'add_zero_attn', 'embed_dim', 'kdim', 'vdim', 'num_heads')}, total_ops=0)
    import_module('thop.profile').register_hooks[nn.MultiheadAttention](proxy, inputs, output)
    return dict(batch=batch, query_tokens=nq, key_tokens=nk, channels=channels,
                projection_macs=projections, qk_av_macs=products,
                softmax_thop_units=int(proxy.total_ops - projections - products),
                total_thop_macs=int(proxy.total_ops))


def profile_model(fusion, scale='n', imgsz=640, nc=4, head='one2many', cross_check=False):
    """Return arithmetic counts for one fused deployment head on CPU."""
    if fusion == "defa-mala-core":
        raise NotImplementedError(
            "DEFA-MALA-core profiling is not implemented; the Softmax attention counter is not applicable"
        )
    rules = import_module('thop.profile')
    if nn.MultiheadAttention not in rules.register_hooks or not getattr(rules, '_COUNTS_FUNCTIONS', False):
        raise RuntimeError('Install ultralytics-thop>=2.2.3 for attention and functional-product counting')
    torch.manual_seed(0)
    model = YOLO(str(ROOT / f'ultralytics/cfg/models/fuse26/yolo26{scale}-{fusion}.yaml')).model
    if nc != model.model[-1].nc:
        from ultralytics.nn.tasks import DetectionModel
        model = DetectionModel(model.yaml, ch=6, nc=nc, verbose=False)
    model = model.cpu().float().eval()
    model.end2end = head == 'one2one'
    model.fuse(verbose=False)
    detector = model.model[-1]
    assert detector.end2end == (head == 'one2one')
    assert (detector.cv2 is None) != (getattr(detector, 'one2one_cv2', None) is None)
    image = torch.zeros(1, 6, imgsz, imgsz)
    attention, dynamic = {}, {}
    handles = []
    for name, module in model.named_modules():
        if isinstance(module, nn.MultiheadAttention):
            def save_attention(m, inputs, output, name=name):
                attention[name] = attention_cost(m, inputs, output)
            handles.append(module.register_forward_hook(save_attention))
        if type(module).__name__ == 'DEFA':
            def save_dynamic(m, inputs, output, name=name):
                feature = inputs[0][0]
                dynamic[name] = dict(input_shape=list(feature.shape), kernel_size=m.dynamic_k,
                                     einsum_macs=feature.numel() * m.dynamic_k**2)
            handles.append(module.register_forward_hook(save_dynamic))
    try:
        # No stride= argument: this executes the requested full-resolution input.
        macs, params, details = thop.profile(model, inputs=(image,), verbose=False, ret_layer_info=True)
    finally:
        for handle in handles:
            handle.remove()
    result = dict(fusion=fusion, scale=scale, nc=nc, modality='RGB+IR', input_shape=list(image.shape),
                  head=head, fused=True, thop_version=thop.__version__, torch_version=torch.__version__,
                  macs=int(macs), gflops=2 * macs / 1e9, parameters=int(params),
                  attention=attention, dynamic_contraction=dynamic)
    result['stride_aware_get_flops'] = get_flops(model, imgsz)
    result['stride_vs_full_absolute_difference'] = abs(result['stride_aware_get_flops'] - result['gflops'])
    result['layers'] = details
    if cross_check:
        from torch.nn.attention import SDPBackend, sdpa_kernel
        from torch.utils.flop_counter import FlopCounterMode, sdpa_flop_count
        def cpu_attention_flops(query_shape, key_shape, value_shape, *args, **kwargs):
            return sdpa_flop_count(query_shape, key_shape, value_shape)
        # PyTorch 2.12 registers CUDA SDPA counters but omits this CPU operator.
        mapping = {torch.ops.aten._scaled_dot_product_flash_attention_for_cpu: cpu_attention_flops}
        # Count logical dense products while CPU FlashAttention avoids materializing N^2 scores.
        # The arithmetic counter omits THOP's normalization/pooling/softmax units.
        with torch.inference_mode(), sdpa_kernel(SDPBackend.FLASH_ATTENTION), \
                FlopCounterMode(display=False, custom_mapping=mapping) as counter:
            model(image)
        result['dense_product_cross_check_flops'] = counter.get_total_flops()
        result['dense_product_cross_check_gflops'] = counter.get_total_flops() / 1e9
        result['thop_additional_non_product_gflops'] = result['gflops'] - counter.get_total_flops() / 1e9
        result['dense_product_operator_flops'] = {
            str(op): count for op, count in counter.get_flop_counts()['Global'].items()}
        expected_attention = 2 * sum(value['qk_av_macs'] for value in attention.values())
        counted_attention = result['dense_product_operator_flops'].get(
            'aten._scaled_dot_product_flash_attention_for_cpu', 0)
        assert counted_attention == expected_attention, (counted_attention, expected_attention)
        assert result['thop_additional_non_product_gflops'] >= -1e-9
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fusion', nargs='+', choices=['decision', 'defa'], default=['decision', 'defa'])
    parser.add_argument('--scale', choices=list('nsmlx'), default='n')
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--nc', type=int, default=4)
    parser.add_argument('--head', choices=['one2one', 'one2many', 'both'], default='both')
    parser.add_argument('--cross-check', action='store_true')
    parser.add_argument('--output', type=Path, default=Path('reports/fusion_flops.json'))
    args = parser.parse_args()
    if args.imgsz < 32 or args.imgsz % 32:
        parser.error('--imgsz must be a positive multiple of 32')
    torch.set_num_threads(2)
    heads = ['one2many', 'one2one'] if args.head == 'both' else [args.head]
    report = dict(convention='GFLOPs = 2 * full-resolution THOP MACs / 1e9',
                  includes='THOP-counted products, normalization, pooling and attention softmax',
                  excludes='preprocessing, NMS, TopK/comparisons, memory movement and backward', models=[])
    for fusion in args.fusion:
        for head in heads:
            result = profile_model(fusion, args.scale, args.imgsz, args.nc, head, args.cross_check)
            report['models'].append(result)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
            print(json.dumps({k: v for k, v in result.items() if k != 'layers'}), flush=True)


if __name__ == '__main__':
    main()
