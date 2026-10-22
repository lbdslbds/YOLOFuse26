"""Exercise modern exporter options with real paired inputs and ONNX Runtime."""
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ultralytics.utils.checks import check_version
from ultralytics.utils.export.engine import torch2onnx


@pytest.mark.skipif(not check_version(torch.__version__, '>=2.6.0'), reason='Dynamo ONNX requires torch>=2.6')
def test_explicit_dynamic_shapes_paired_inputs_external_weights_and_verification(tmp_path):
    onnx = pytest.importorskip('onnx')
    ort = pytest.importorskip('onnxruntime')
    pytest.importorskip('onnxscript')

    class PairedProjection(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.projection = torch.nn.Conv2d(6, 16, 3, padding=1)

        def forward(self, visible, infrared):
            return self.projection(torch.cat((visible, infrared), dim=1)).relu()

    torch.manual_seed(31)
    model = PairedProjection().eval()
    example = (torch.rand(2, 3, 16, 24), torch.rand(2, 3, 16, 24))
    dims = {0:torch.export.Dim('batch', min=1, max=4),
            2:torch.export.Dim('height', min=8, max=64),
            3:torch.export.Dim('width', min=8, max=64)}
    output = tmp_path/'paired.onnx'
    torch2onnx(model, example, output, opset=18, input_names=['visible','infrared'],
               dynamo=True, dynamic_shapes=(dims,dims), external_data=True, report=True, verify=True)
    onnx.checker.check_model(onnx.load(output))
    assert list(tmp_path.glob('onnx_export_*.md')), 'No diagnostic report was written'
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    session = ort.InferenceSession(str(output), options, providers=['CPUExecutionProvider'])
    for shape in [(1,3,24,32), (3,3,40,16)]:
        visible, infrared = torch.rand(shape), torch.rand(shape)
        with torch.inference_mode():
            expected = model(visible, infrared).numpy()
        actual = session.run(None, {'visible':visible.numpy(), 'infrared':infrared.numpy()})[0]
        np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)
