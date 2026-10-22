"""Numerical checks for six-channel ONNX, Inductor and DEFA's actual CUDA FlashAttention kernel."""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
import time
import traceback

os.environ.setdefault('TORCHINDUCTOR_COMPILE_THREADS', '2')
os.environ.setdefault('MAX_JOBS', '2')
os.environ.setdefault('OMP_NUM_THREADS', '2')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.nn.modules.DEFA import DEFA
from ultralytics.utils.torch_utils import attempt_compile

torch.set_num_threads(2)
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False


def sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def load(fusion):
    checkpoint = ROOT / f'runs/train/yolo26n-{fusion}-tvt-20261023-r1-smoke/weights/best.pt'
    return YOLO(str(checkpoint if checkpoint.exists() else ROOT / f'ultralytics/cfg/models/fuse26/yolo26n-{fusion}.yaml'))


def compare_outputs(actual, expected, head, raw=None):
    assert actual.shape == expected.shape, (actual.shape, expected.shape)
    assert np.isfinite(actual).all() and np.isfinite(expected).all()
    ties = 0
    if head == 'one2one':
        from scipy.optimize import linear_sum_assignment
        assert raw is not None and np.isfinite(raw).all()
        aligned = np.empty_like(expected)
        for b in range(len(actual)):
            np.testing.assert_allclose(np.sort(actual[b,:,4]),np.sort(expected[b,:,4]),rtol=2e-3,atol=2e-7)
            cutoff = expected[b,:,4].min()
            # Compare physical detections independently of their TopK row order. At the cutoff,
            # a nearly equal score may select a different valid anchor/class in the two runtimes.
            common = (actual[b,:,None,5] == expected[b,None,:,5])
            common &= np.isclose(actual[b,:,None,:4],expected[b,None,:,:4],rtol=2e-3,atol=2e-3).all(-1)
            common &= np.isclose(actual[b,:,None,4],expected[b,None,:,4],rtol=2e-3,atol=2e-7)
            different_actual = ~common.any(axis=1)
            different_expected = ~common.any(axis=0)
            assert np.all(np.abs(actual[b,different_actual,4]-cutoff) <= 2e-7)
            assert np.all(np.abs(expected[b,different_expected,4]-cutoff) <= 2e-7)
            ties += int(different_actual.sum())
            # Validate every emitted detection against the complete pre-TopK reference head.
            # Assignment is injective within each exact class, so duplicates, wrong classes,
            # incorrect boxes or fabricated confidence scores cannot pass as boundary ties.
            for category in np.unique(actual[b,:,5]):
                assert category == int(category) and 0 <= category < raw.shape[-1]-4
                chosen = np.flatnonzero(actual[b,:,5] == category)
                candidates = np.column_stack((raw[b,:,:4],raw[b,:,4+int(category)],
                                               np.full(raw.shape[1],category)))
                cost = np.abs(actual[b,chosen,None,:4]-candidates[None,:,:4]).sum(-1)
                cost += np.abs(actual[b,chosen,None,4]-candidates[None,:,4])*1e3
                rows, columns = linear_sum_assignment(cost)
                mapped = candidates[columns]
                np.testing.assert_allclose(actual[b,chosen[rows],:4],mapped[:,:4],rtol=2e-3,atol=2e-3)
                np.testing.assert_allclose(actual[b,chosen[rows],4],mapped[:,4],rtol=2e-3,atol=2e-7)
                aligned[b,chosen[rows]] = mapped
        expected = aligned
        np.testing.assert_array_equal(actual[..., 5], expected[..., 5])
        np.testing.assert_allclose(actual[..., 4], expected[..., 4], rtol=2e-3, atol=2e-7)
    np.testing.assert_allclose(actual, expected, rtol=2e-3, atol=2e-3)
    return {'max_abs_error':float(np.abs(actual - expected).max()), 'topk_boundary_ties':ties}


def before_topk(model, x, head):
    if head != 'one2one':
        return None
    from unittest.mock import patch
    with patch.object(model.model[-1], 'postprocess', lambda prediction:prediction), torch.inference_mode():
        return model(x).detach().cpu().numpy()


def check_onnx(fusion, output, emit):
    import onnx
    import onnxruntime as ort
    for dynamo, dynamic in [(False, False), (True, False), (True, True)]:
        for head in ['one2many', 'one2one']:
            if not dynamo and head == 'one2one':
                continue
            name = f'{fusion}_{head}_{"dynamo" if dynamo else "legacy"}_{"dynamic" if dynamic else "static"}'
            emit({'case': name, 'state': 'running'})
            model = load(fusion)
            model.model.pt_path = str(output / (name + '.pt'))
            captured = []
            model.add_callback('on_export_end', lambda exporter: captured.append(exporter))
            start = time.perf_counter()
            exported = model.export(format='onnx', device='cpu', imgsz=128, batch=1, opset=18,
                                    dynamic=dynamic, dynamo=dynamo, simplify=False,
                                    nms=False if head == 'one2one' else None)
            graph = onnx.load(exported)
            onnx.checker.check_model(graph)
            assert graph.graph.input[0].type.tensor_type.shape.dim[1].dim_value == 6
            options = ort.SessionOptions()
            options.intra_op_num_threads = 2
            options.inter_op_num_threads = 1
            session = ort.InferenceSession(str(exported), options, providers=['CPUExecutionProvider'])
            reference = captured[-1].model.eval()
            shapes = [(1, 6, 128, 128)]
            if dynamic:
                shapes += [(2, 6, 160, 192), (1, 6, 192, 160)]
                assert isinstance(session.get_inputs()[0].shape[0], str), session.get_inputs()[0].shape
            comparisons = []
            for shape in shapes:
                torch.manual_seed(13)
                x = torch.rand(shape)
                with torch.inference_mode():
                    expected = reference(x)
                if isinstance(expected, (tuple, list)):
                    expected = expected[0]
                actual = session.run(None, {'images': x.numpy()})[0]
                expected = expected.detach().cpu().numpy()
                error = compare_outputs(actual, expected, head, before_topk(reference,x,head))
                comparisons.append({'input': list(shape), 'output': list(actual.shape),
                                    **error})
            from ultralytics.data.rgbir import read_rgbir_image
            import cv2
            image = next((ROOT.parent / 'datasets_tvt_20261019/images/val').glob('*.jpg'), None)
            if image is None:
                image = next((ROOT.parent / 'datasets_tvt_20261019/images/val').glob('*.png'))
            paired = cv2.resize(read_rgbir_image(image), (128, 128))
            real = torch.from_numpy(np.ascontiguousarray(paired.transpose(2, 0, 1))).float().unsqueeze(0) / 255
            with torch.inference_mode():
                expected = reference(real)
            if isinstance(expected, (tuple, list)):
                expected = expected[0]
            actual = session.run(None, {'images':real.numpy()})[0]
            error = compare_outputs(actual, expected.numpy(), head, before_topk(reference,real,head))
            comparisons.append({'real_rgbir_image':str(image), **error})
            emit({'case': name, 'state': 'passed', 'file': str(exported),
                  'input_shape': session.get_inputs()[0].shape, 'comparisons': comparisons,
                  'seconds': round(time.perf_counter() - start, 2)})


def check_compile(fusion, output, emit, device, training=False, imgsz=128):
    from torch._dynamo.utils import counters
    from torch._inductor import metrics
    name = f'{fusion}_compile_{"train" if training else "infer"}_{device.type}_{imgsz}'
    emit({'case': name, 'state': 'running'})
    torch.manual_seed(17)
    base = load(fusion).model.to(device)
    if training:
        base.requires_grad_(True)  # Saved inference checkpoints have gradients disabled.
    base.train(training)
    base.end2end = False if not training else True
    if not training:
        base.fuse(verbose=False)
    reference = deepcopy(base)
    x = torch.rand(2 if training else 1, 6, imgsz, imgsz, device=device)
    batch = {'img': x, 'batch_idx': torch.tensor([0, 1], device=device),
             'cls': torch.tensor([[0.], [2.]], device=device),
             'bboxes': torch.tensor([[.5, .5, .3, .4], [.4, .6, .2, .3]], device=device)}
    if training:
        base.args = get_cfg(overrides={'epochs': 2})
        reference.args = get_cfg(overrides={'epochs': 2})
        # Match the trainer: compile the forward pass, calculate the variable-target loss separately.
        expected, _ = reference.loss(batch, preds=reference(x))
        expected.sum().backward()
    else:
        with torch.inference_mode():
            expected = reference(x)[0]
    torch._dynamo.reset()
    counters.clear()
    metrics.reset()
    torch._dynamo.config.suppress_errors = False
    start = time.perf_counter()
    compiled = attempt_compile(base, device=device, mode=True)
    assert hasattr(compiled, '_orig_mod'), 'compile=True silently fell back to eager'
    if training:
        actual, _ = base.loss(batch, preds=compiled(x))
        actual.sum().backward()
        torch.testing.assert_close(actual, expected, rtol=5e-3, atol=2e-3)
        finite, gradient_error = 0, 0.0
        for (name_a, param), (name_b, ref) in zip(base.named_parameters(), reference.named_parameters()):
            assert name_a == name_b
            if ref.grad is not None:
                assert param.grad is not None and torch.isfinite(param.grad).all(), name_a
                torch.testing.assert_close(param.grad, ref.grad, rtol=3e-2, atol=3e-3)
                gradient_error = max(gradient_error, float((param.grad-ref.grad).abs().max()))
                finite += 1
        torch.optim.SGD(base.parameters(), lr=1e-4).step()
        details = {'loss': actual.detach().cpu().tolist(), 'gradient_tensors': finite, 'max_gradient_error': gradient_error}
    else:
        with torch.inference_mode():
            actual = compiled(x)[0]
        torch.testing.assert_close(actual, expected, rtol=2e-3, atol=2e-3)
        details = {'input': list(x.shape), 'output': list(actual.shape),
                   'max_abs_error': float((actual-expected).abs().max())}
    sync(device)
    first_seconds = time.perf_counter() - start
    assert counters['stats']['unique_graphs'] > 0, dict(counters)
    assert metrics.generated_kernel_count > 0, 'No Inductor kernel was generated'
    if not training:
        timings = {}
        with torch.inference_mode():
            for _ in range(2):
                compiled(x)
            sync(device)
            for label, target in [('eager', reference), ('compiled', compiled)]:
                sync(device)
                start = time.perf_counter()
                for _ in range(5):
                    target(x)
                sync(device)
                timings[label + '_ms'] = (time.perf_counter() - start) * 1000 / 5
        details.update(timings, speedup=timings['eager_ms']/timings['compiled_ms'], fp32_tf32=False)
    emit({'case': name, 'state': 'passed', 'first_execution_seconds': round(first_seconds, 2),
          'inductor_kernels': metrics.generated_kernel_count,
          'unique_graphs': counters['stats']['unique_graphs'],
          'graph_breaks': dict(counters['graph_break']), **details})


def check_flash(output, emit, device, imgsz):
    from torch.nn.attention import SDPBackend, sdpa_kernel
    if device.type != 'cuda' or not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU must be enabled to validate the FlashAttention backend')
    from torch.profiler import profile, ProfilerActivity
    # Read channels from the actual model rather than assuming a backbone width.
    model = load('defa').model.to(device).requires_grad_(True).train()
    modules = [m for m in model.modules() if isinstance(m, DEFA)]
    assert len(modules) == 3
    for source_module, stride in zip(modules, [8, 16, 32]):
        channels = source_module.cross_attn.embed_dim
        spatial = imgsz // stride
        name = f'defa_flash_c{channels}_{spatial}x{spatial}'
        emit({'case': name, 'state': 'running'})
        torch.manual_seed(23)
        module = deepcopy(source_module).train()
        reference = deepcopy(module)
        vi = torch.rand(1, channels, spatial, spatial, device=device, requires_grad=True)
        ir = torch.rand_like(vi, requires_grad=True)
        rv, ri = vi.detach().clone().requires_grad_(), ir.detach().clone().requires_grad_()
        with sdpa_kernel(SDPBackend.MATH), torch.autocast('cuda', dtype=torch.float16):
            expected = reference((rv, ri))
            expected.float().square().mean().backward()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION), torch.autocast('cuda', dtype=torch.float16):
                actual = module((vi, ir))
                actual.float().square().mean().backward()
        sync(device)
        kernels = [event.key for event in prof.key_averages() if 'flash' in event.key.lower()]
        assert any('scaled_dot_product_flash_attention' in name for name in kernels), kernels
        assert any('backward' in name for name in kernels), kernels
        torch.testing.assert_close(actual, expected, rtol=3e-2, atol=5e-3)
        assert torch.isfinite(vi.grad).all() and vi.grad.abs().sum() > 0
        assert torch.isfinite(ir.grad).all() and ir.grad.abs().sum() > 0
        for grad, refgrad in [(vi.grad, rv.grad), (ir.grad, ri.grad)]:
            torch.testing.assert_close(grad, refgrad, rtol=5e-2, atol=2e-4)
        emit({'case': name, 'state': 'passed', 'amp_dtype': 'float16', 'backend': 'FLASH_ATTENTION only',
              'kernel_events': kernels, 'max_abs_error': float((actual-expected).abs().max()),
              'rgb_gradient_norm': float(vi.grad.norm()), 'ir_gradient_norm': float(ir.grad.norm())})
    # Exercise the original MultiheadAttention call inside a complete six-channel DEFA model as well.
    x = torch.rand(2, 6, min(imgsz, 320), min(imgsz, 320), device=device, requires_grad=True)
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION), torch.autocast('cuda', dtype=torch.float16):
        result = model(x)
        tensors = result['one2many']
        loss = tensors['boxes'].float().square().mean() + tensors['scores'].float().square().mean()
        loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(x.grad).all()
    assert x.grad[:, :3].abs().sum() > 0 and x.grad[:, 3:].abs().sum() > 0
    emit({'case': 'defa_full_model_flash_forward_backward', 'state': 'passed', 'input':list(x.shape),
          'loss':float(loss), 'backend': 'FLASH_ATTENTION only'})


def check_training_api(fusion, output, emit, device, imgsz):
    import yaml
    from torch._dynamo.utils import counters
    from torch._inductor import metrics
    name = f'{fusion}_train_api_compile_true_{imgsz}'
    emit({'case':name, 'state':'running'})
    config = yaml.safe_load((ROOT / 'configs/rgbir-tvt.yaml').read_text())
    dataset = Path(config['path'])
    # Preserve the full original COCO ground truth and let the existing evaluator select visited IDs.
    for split in ['train', 'val']:
        annotations = json.loads(Path(config['coco_annotations'][split]).read_text())
        crowd_ids = {a['image_id'] for a in annotations['annotations'] if a.get('iscrowd')}
        chosen = []
        for wanted in [True, False]:
            chosen += [dataset/'images'/split/im['file_name'] for im in annotations['images']
                       if (im['id'] in crowd_ids) == wanted][:4]
        assert len(chosen) == 8 and all(p.is_file() for p in chosen)
        listing = output / (split + '.txt')
        listing.write_text(''.join(str(p)+'\n' for p in chosen))
        config[split] = str(listing)
    data = output / 'data.yaml'
    data.write_text(yaml.safe_dump(config))
    model = load(fusion)
    confirmed = []
    model.add_callback('on_train_start', lambda trainer: confirmed.append(
        trainer.args.compile is True and hasattr(trainer.model, '_orig_mod')))
    counters.clear()
    metrics.reset()
    torch._dynamo.config.suppress_errors = False
    model.train(data=str(data), epochs=1, batch=2, imgsz=imgsz, workers=0,
                device=str(device), compile=True, amp=True, optimizer='SGD', cache=False,
                nms=None, project=str(output), name=fusion, exist_ok=True,
                plots=False, seed=0, deterministic=True, close_mosaic=0, mosaic=0,
                warmup_epochs=0, patience=0)
    assert confirmed == [True], 'Trainer failed to install the compiled module'
    assert counters['stats']['unique_graphs'] > 0 and metrics.generated_kernel_count > 0
    best = output / fusion / 'weights/best.pt'
    last = best.with_name('last.pt')
    assert best.is_file() and last.is_file()
    policy = json.loads((output / fusion / 'crowd_evaluation.json').read_text())['policy']
    assert policy['images'] == 8 and policy['crowd_regions'] > 0, policy
    # Reload a real compiled-training checkpoint and exercise both historical evaluation heads.
    heads = {}
    for head in ['one2many_nms', 'one2one_nms_free']:
        reloaded = YOLO(str(best))
        result = reloaded.val(data=str(data), split='val', device=str(device), imgsz=imgsz,
                              batch=2, workers=0, compile=True,
                              nms=None if head == 'one2many_nms' else False,
                              conf=0.001, iou=0.70, max_det=300, plots=False, save_json=True,
                              project=str(output), name=fusion+'_'+head, exist_ok=True)
        evaluation = json.loads((output / (fusion+'_'+head) / 'crowd_evaluation.json').read_text())
        assert evaluation['policy']['images'] == 8 and evaluation['policy']['crowd_regions'] > 0
        heads[head] = {'policy':evaluation['policy'], 'metrics':result.results_dict}
    emit({'case':name, 'state':'passed', 'best':str(best), 'last':str(last),
          'inductor_kernels':metrics.generated_kernel_count, 'unique_graphs':counters['stats']['unique_graphs'],
          'validation':heads})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--suite', required=True, choices=['onnx', 'compile', 'flash', 'train_api'])
    p.add_argument('--device', default='cpu')
    p.add_argument('--imgsz', type=int, default=128)
    p.add_argument('--train', action='store_true')
    p.add_argument('--fusion', nargs='+', choices=['decision', 'defa'], default=['decision', 'defa'])
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    device = torch.device('cuda:' + args.device if args.device.isdigit() else args.device)
    args.output.mkdir(parents=True, exist_ok=True)
    report = {'suite':args.suite, 'torch':torch.__version__, 'cuda_build':torch.version.cuda,
              'device':str(device), 'state':'running', 'cases':[]}
    if device.type == 'cuda':
        report['gpu'] = torch.cuda.get_device_name(device)
    def emit(item):
        print(json.dumps(item, ensure_ascii=False), flush=True)
        report['cases'].append(item)
        (args.output / 'results.json').write_text(json.dumps(report, indent=2))
    try:
        if args.suite == 'flash':
            check_flash(args.output, emit, device, args.imgsz)
        else:
            for fusion in args.fusion:
                if args.suite == 'onnx':
                    check_onnx(fusion, args.output, emit)
                elif args.suite == 'train_api':
                    check_training_api(fusion, args.output, emit, device, args.imgsz)
                else:
                    check_compile(fusion, args.output, emit, device, args.train, args.imgsz)
        report['state'] = 'passed'
    except Exception:
        report['state'] = 'failed'
        report['error'] = traceback.format_exc()
        raise
    finally:
        (args.output / 'results.json').write_text(json.dumps(report, indent=2))
        print(json.dumps({'suite':args.suite, 'state':report['state']}), flush=True)


if __name__ == '__main__':
    main()
