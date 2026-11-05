"""Check MALA-core against dense spatial attention and preserve DEFA integration contracts."""

from copy import deepcopy
from pathlib import Path
import sys

import pytest
import torch
from torch import nn
from torch.nn import functional as F
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ultralytics.nn.modules.mala import MALACrossAttention


def dense_attention_reference(module, query, key, value):
    """Materialize every query/key weight, independently of the efficient K.T @ V contraction."""
    weights = module.in_proj_weight.chunk(3)
    biases = (None, None, None) if module.in_proj_bias is None else module.in_proj_bias.chunk(3)
    projected = [F.linear(x, weight, bias) for x, weight, bias in zip((query, key, value), weights, biases)]
    q, k, v = [x.reshape(x.shape[0], x.shape[1], module.num_heads, module.head_dim).transpose(1, 2)
               for x in projected]
    q, k = F.elu(q) + 1, F.elu(k) + 1
    # This oracle uses an Nq x Nk matrix. Its row mean is the magnitude term z.
    affinity = torch.matmul(q, k.transpose(-2, -1)) / module.head_dim**0.5
    magnitude = affinity.mean(dim=-1, keepdim=True)
    effective_weights = (affinity * (1 + 1 / (magnitude + 1e-6)) - magnitude) / key.shape[1]
    attended = torch.matmul(effective_weights, v)
    attended = attended.transpose(1, 2).reshape(query.shape[0], query.shape[1], module.embed_dim)
    return F.linear(attended, module.out_proj.weight, module.out_proj.bias)


@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("query_tokens,key_tokens", [(5, 7), (4, 4)])
def test_mala_matches_dense_attention_forward_and_gradients(bias, query_tokens, key_tokens):
    torch.manual_seed(71)
    actual_module = MALACrossAttention(12, 3, bias=bias).double()
    reference_module = deepcopy(actual_module)
    actual_inputs = [torch.randn(2, tokens, 12, dtype=torch.float64, requires_grad=True)
                     for tokens in (query_tokens, key_tokens, key_tokens)]
    reference_inputs = [x.detach().clone().requires_grad_() for x in actual_inputs]
    actual, attention_weights = actual_module(*actual_inputs)
    expected = dense_attention_reference(reference_module, *reference_inputs)
    assert attention_weights is None
    torch.testing.assert_close(actual, expected, rtol=2e-10, atol=2e-11)

    # A nonuniform cotangent checks all outputs, rather than only their sum.
    cotangent = torch.randn_like(actual)
    actual_variables = actual_inputs + list(actual_module.parameters())
    reference_variables = reference_inputs + list(reference_module.parameters())
    actual_gradients = torch.autograd.grad(actual, actual_variables, grad_outputs=cotangent)
    expected_gradients = torch.autograd.grad(expected, reference_variables, grad_outputs=cotangent)
    for actual_gradient, expected_gradient in zip(actual_gradients, expected_gradients):
        assert torch.isfinite(actual_gradient).all()
        torch.testing.assert_close(actual_gradient, expected_gradient, rtol=2e-9, atol=2e-10)


@pytest.mark.parametrize("bias", [True, False])
def test_mala_loads_standard_multihead_attention_parameters(bias):
    torch.manual_seed(73)
    original = nn.MultiheadAttention(16, 4, bias=bias, batch_first=True)
    adapted = MALACrossAttention(16, 4, bias=bias)
    result = adapted.load_state_dict(original.state_dict(), strict=True)
    assert not result.missing_keys and not result.unexpected_keys
    assert adapted.state_dict().keys() == original.state_dict().keys()
    for name, parameter in original.state_dict().items():
        torch.testing.assert_close(adapted.state_dict()[name], parameter, rtol=0, atol=0)
    # Structural compatibility is bidirectional; attention output equivalence is not promised.
    original.load_state_dict(adapted.state_dict(), strict=True)


@pytest.mark.parametrize("argument,message", [
    ("key_padding_mask", "attention masks"),
    ("attn_mask", "attention masks"),
    ("need_weights", "need_weights=False"),
    ("is_causal", "causal attention"),
])
def test_mala_rejects_unsupported_attention_options(argument, message):
    model = MALACrossAttention(12, 3)
    query, key, value = torch.randn(2, 5, 12), torch.randn(2, 7, 12), torch.randn(2, 7, 12)
    options = {"key_padding_mask": torch.zeros(2, 7, dtype=torch.bool),
               "attn_mask": torch.zeros(5, 7), "need_weights": True, "is_causal": True}
    with pytest.raises(ValueError, match=message):
        model(query, key, value, **{argument: options[argument]})


def test_mala_requires_batch_first_layout():
    with pytest.raises(ValueError, match="batch_first=True"):
        MALACrossAttention(12, 3, batch_first=False)


def test_mala_cpu_bfloat16_autocast_has_finite_outputs_and_gradients():
    if not hasattr(torch, "autocast"):
        pytest.skip("This PyTorch build does not provide autocast")
    # Probe only platform support. Errors from the MALA forward/backward must fail the test.
    try:
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            probe = F.linear(torch.ones(1, 4), torch.ones(4, 4))
    except (RuntimeError, TypeError, NotImplementedError) as error:
        pytest.skip(f"CPU bfloat16 autocast unavailable: {error}")
    if probe.dtype != torch.bfloat16:
        pytest.skip("CPU autocast does not produce bfloat16 projections")

    torch.manual_seed(79)
    model = MALACrossAttention(16, 4).train()
    inputs = [torch.randn(2, tokens, 16, requires_grad=True) for tokens in (5, 7, 7)]
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        output, weights = model(*inputs)
    assert output.dtype == torch.bfloat16 and weights is None
    assert torch.isfinite(output).all()
    output.float().square().mean().backward()
    for variable in inputs + list(model.parameters()):
        assert variable.grad is not None and torch.isfinite(variable.grad).all()
        assert variable.grad.abs().sum() > 0


@pytest.mark.parametrize("scale", list("nsmlx"))
def test_mala_model_configs_preserve_defa_graph_and_pretrained_mapping(scale):
    from tools.build_models import build

    base = build("defa", scale)
    adapted = build("defa-mala-core", scale)
    config_path = ROOT / f"ultralytics/cfg/models/fuse26/yolo26{scale}-defa-mala-core.yaml"
    assert yaml.safe_load(config_path.read_text(encoding="utf-8")) == adapted
    assert adapted["pretrained_layers"] == base["pretrained_layers"]
    assert adapted["pretrained_head_scales"] == base["pretrained_head_scales"]

    normalized = deepcopy(adapted)
    normalized["fusion"] = "defa"
    attention_nodes = [node for node in normalized["backbone"] + normalized["head"] if node[2] == "DEFA"]
    assert len(attention_nodes) == 3
    for node in attention_nodes:
        assert node[3] == [16, 8, 5, "mala-core"]
        node[3] = []
    assert normalized == base
