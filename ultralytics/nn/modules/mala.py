"""MALA-core cross-attention for DEFA with the original MHA projection layout.

Reference: Fan et al., "Rectifying Magnitude Neglect in Linear Attention", ICCV 2025.
Official core: https://github.com/qhfan/MALA/blob/main/classfication_release/model.py

This is a cross-modal adaptation of the magnitude-aware attention calculation,
not the complete official MALA block. It deliberately excludes RoPE, depthwise
convolution, and the official block's extra output gate. DEFA retains its own
dynamic convolution, fusion gates, residual paths, and feature enhancement.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class MALACrossAttention(nn.MultiheadAttention):
    """Unmasked batch-first MALA-core using standard MHA projection parameters.

    The inherited ``in_proj_weight``, ``in_proj_bias``, and ``out_proj`` keep
    their original names, shapes, and initialization. Loading MHA weights is
    therefore structurally compatible, but does not preserve MHA outputs.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        bias: bool = True,
        batch_first: bool = True,
    ) -> None:
        if not batch_first:
            raise ValueError("MALACrossAttention requires batch_first=True")
        super().__init__(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=0.0,
            bias=bias,
            batch_first=True,
        )

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        key_padding_mask: Tensor | None = None,
        need_weights: bool = False,
        attn_mask: Tensor | None = None,
        average_attn_weights: bool = True,
        is_causal: bool = False,
    ) -> tuple[Tensor, None]:
        """Map Q/K/V sequences to ``[B, Nq, C]``; attention weights are unavailable.

        ``average_attn_weights`` is accepted for MHA call compatibility and has
        no effect when ``need_weights=False``. Masks and causal attention are
        intentionally unsupported by this DEFA-specific implementation.
        """
        if key_padding_mask is not None or attn_mask is not None:
            raise ValueError("MALACrossAttention does not support attention masks")
        if is_causal:
            raise ValueError("MALACrossAttention does not support causal attention")
        if need_weights:
            raise ValueError("MALACrossAttention requires need_weights=False")
        if query.ndim != 3 or key.ndim != 3 or value.ndim != 3:
            raise ValueError("MALACrossAttention expects [B, N, C] tensors")
        if (
            query.shape[-1] != self.embed_dim
            or key.shape[-1] != self.embed_dim
            or value.shape[-1] != self.embed_dim
        ):
            raise ValueError("MALACrossAttention requires equal embedding dimensions")
        if query.shape[0] != key.shape[0] or query.shape[0] != value.shape[0]:
            raise ValueError("MALACrossAttention requires matching batch sizes")
        if key.shape[1] != value.shape[1] or key.shape[1] == 0:
            raise ValueError("MALACrossAttention requires matching nonempty K/V sequences")

        batch, query_tokens, _ = query.shape
        kv_tokens = key.shape[1]
        channels = self.embed_dim
        q_weight, k_weight, v_weight = self.in_proj_weight.chunk(3, dim=0)
        if self.in_proj_bias is None:
            q_bias = k_bias = v_bias = None
        else:
            q_bias, k_bias, v_bias = self.in_proj_bias.chunk(3, dim=0)

        q = F.linear(query, q_weight, q_bias)
        k = F.linear(key, k_weight, k_bias)
        v = F.linear(value, v_weight, v_bias)
        projection_dtype = q.dtype
        q = q.reshape(batch, query_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(batch, kv_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(batch, kv_tokens, self.num_heads, self.head_dim).transpose(1, 2)

        # Explicitly disable autocast: casting inputs alone does not keep matmul
        # in FP32. Preserve FP64 when requested instead of always downcasting.
        core_dtype = torch.float32 if projection_dtype in (torch.float16, torch.bfloat16) else projection_dtype
        with torch.autocast(device_type=query.device.type, enabled=False):
            q = F.elu(q.to(dtype=core_dtype)) + 1.0
            k = F.elu(k.to(dtype=core_dtype)) + 1.0
            v = v.to(dtype=core_dtype)
            scale = self.head_dim ** -0.5
            z = (q @ k.mean(dim=-2, keepdim=True).transpose(-2, -1)) * scale
            # KV has shape [B, heads, head_dim, head_dim], never [Nq, Nkv].
            kv_scale = (scale / kv_tokens) ** 0.5
            kv = (k.transpose(-2, -1) * kv_scale) @ (v * kv_scale)
            mixed = q @ kv
            output = mixed * (1.0 + (z + 1e-6).reciprocal()) - z * v.mean(dim=-2, keepdim=True)

        output = output.transpose(1, 2).reshape(batch, query_tokens, channels).to(dtype=projection_dtype)
        return self.out_proj(output), None
