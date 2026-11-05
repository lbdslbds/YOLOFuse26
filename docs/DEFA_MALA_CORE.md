# DEFA-MALA-core

CLI name: `defa-mala-core`. This optional mode replaces only DEFA's P3/P4/P5
cross-attention calculation. Both YOLO26 backbones, the shared PAN, dynamic-kernel
generator, dynamic convolution, fusion gates, residual paths, and enhancement
remain in place. RGB provides Q; IR provides K and V. Feature shapes are unchanged.

## Core adaptation

The attention calculation adapts Magnitude-Aware Linear Attention from
[Rectifying Magnitude Neglect in Linear Attention](https://arxiv.org/abs/2507.00698)
(Fan et al., ICCV 2025). The authors' [reference implementation](https://github.com/qhfan/MALA/blob/main/classfication_release/model.py)
provides the source formulation; the local implementation is
[`MALACrossAttention`](../ultralytics/nn/modules/mala.py).

This is a cross-modal core adaptation, not a reproduction of the complete
MALA/MAViT architecture. It adds no RoPE, local depthwise branch, extra output
gate, LayerNorm, downsampling, or reverse-direction attention. These scope
choices do not establish the cause of any measured accuracy difference.

For projected per-head features, let `M` be the number of key/value tokens,
`d` the head width, and `s = d**-0.5`. Means are over key/value tokens:

```text
q = ELU(Q) + 1
k = ELU(K) + 1
z = s * q @ mean(k).T
T = q @ ((s / M) * k.T @ V)
Y = T * (1 + 1 / (z + 1e-6)) - z * mean(V)
```

The implementation distributes `sqrt(s/M)` over K and V. It forms a per-head
`d × d` KV product rather than a full spatial QK matrix. FP16/BF16 projections
are promoted to FP32 inside an autocast-disabled core; FP64 is preserved. The
result returns to the projection dtype before the output projection. This
changes the attention operator and does not imply fewer training epochs.

`MALACrossAttention` retains the original MHA parameter names and shapes:
`in_proj_weight`, `in_proj_bias`, and `out_proj`. Parameter-layout compatibility
does not imply output equivalence. This implementation requires batch-first,
unmasked, non-causal attention with `need_weights=False`.

## Configuration and use

Five model configurations are supplied for `n/s/m/l/x`, starting with
[`yolo26n-defa-mala-core.yaml`](../ultralytics/cfg/models/fuse26/yolo26n-defa-mala-core.yaml).
Each keeps the original DEFA graph and pretrained layer map. The three DEFA
nodes use `[16, 8, 5, "mala-core"]` for reduction, heads, dynamic kernel size,
and attention type; the parser supplies the scaled channel count. Original
`defa` configurations retain `attention_type="softmax"`.

After configuring your own paired dataset, the following example uses the main
settings of the recorded early-stopping experiment. Replace `defa-mala-core`
with `defa` for the reference architecture:

```bash
python train_dual.py --fusion defa-mala-core --scale n --data configs/rgbir-tvt.yaml --device 0 --epochs 300 --patience 30 --imgsz 640 --batch 16 --workers 4 --optimizer SGD --cache disk --amp --compile --nms
```

`--patience` applies to new runs; its default is 0, which disables early
stopping. A positive value stops training after that many epochs without a
strict validation-fitness improvement. There is no minimum improvement
threshold, so even a small new best resets the wait. `--nms` selects the
one-to-many head for validation and checkpoint selection.

`--weights` expects official single-modality YOLO26 weights and initializes the
mapped backbone/neck/head layers. `--resume` retains the saved architecture and
training settings, including patience; it does not convert a Softmax DEFA
checkpoint into MALA-core. Validation and inference use the existing
weight-based `val_dual.py` and `infer_dual.py` entry points.

The original dataset, exact split lists/manifests, and trained experiment
checkpoints are not distributed with this repository. The command is a useful
training recipe, not a guarantee of exact historical reproduction; reproducing
the reported benchmark requires those original assets and the recorded setup.

## Recorded experiments

The recorded setup has four classes (person, rider, bicycle, car) and a fixed
RGB+IR split of 5279 training, 721 validation, and 2676 test pairs. The splits
contain 18955/2586/9902 ordinary boxes and 789/111/531 crowd regions. This table
reports that setup only; it is not a result for every dataset supported by the
repository.

Both models used nano scale, 640 input size, batch=16, workers=4, SGD, official
YOLO26n pretrained initialization, effective AMP, and Inductor compilation
(`compile=True`, default mode) on PyTorch 2.12.1+cu130. Checkpoints were selected
by validation COCO bbox mAP50–95, not by test scores. Independent final scoring
used FP32 eager execution and the one-to-many NMS head, with `conf=0.001`,
`iou=0.7`, and `max_det=300`.

All scores below are percentages using COCO crowd matching and the corrected
annotation-ID policy: non-positive annotation IDs are remapped only in an
evaluation copy; original source annotations and image/category IDs remain
unchanged. See [crowd evaluation](CROWD.md) for the matching policy.

| Training budget | Model | Best epoch | Completed epochs | Val mAP50–95 | Test mAP50–95 |
| --- | --- | ---: | ---: | ---: | ---: |
| 100 epochs | DEFA | — | 100 | 72.83 | 56.46 |
| 100 epochs | DEFA-MALA-core | — | 100 | 72.69 | 56.78 |
| 300 max, patience=30 | DEFA | 189 | 219 | 74.62 | 57.04 |
| 300 max, patience=30 | DEFA-MALA-core | 255 | 285 | 73.89 | 55.84 |

The latest early-stopping checkpoints have the following independent final
scores. P/R use each class's maximum-F1 point on the COCO IoU=0.50 interpolated
precision/recall grid, not a common confidence operating point:

| Model | Split | P | R | mAP50 | mAP50–95 |
| --- | --- | ---: | ---: | ---: | ---: |
| DEFA | val | 90.19 | 92.75 | 95.29 | 74.62 |
| DEFA-MALA-core | val | 92.28 | 89.75 | 94.23 | 73.89 |
| DEFA | test | 81.11 | 76.50 | 81.31 | 57.04 |
| DEFA-MALA-core | test | 82.33 | 74.50 | 79.68 | 55.84 |

Training-time AMP validation CSV values are a separate record from final FP32
scoring. The selected DEFA epoch has CSV mAP50–95 **74.573%**, versus final FP32
validation **74.6239%**; MALA-core has CSV **73.935%**, versus final FP32
validation **73.8900%**. Use the final evaluation table for checkpoint comparisons,
and the CSV series for training trajectories and epoch selection.

### Interpretation and limitations

- In the latest early-stopping experiment, MALA-core finished with lower
  validation and test mAP50–95 than DEFA, by about 0.73 and 1.20 percentage
  points. Its best epoch occurred later. The 100-epoch experiment instead had
  slightly higher MALA-core test mAP50–95, so these records do not establish
  a consistent accuracy benefit from replacing Softmax with MALA-core.
- Each configuration has one run per training budget. There are no multi-seed
  confidence intervals or statistical-significance estimates. The configured
  trainer seed was 0, but model construction precedes trainer seeding; identical
  initialization of newly introduced fusion parameters was not established.
- MALA-core was interrupted after epoch 144 and resumed at epoch 145. Optimizer,
  GradScaler, EMA, scheduler, and early-stopping states were restored. Exact
  equivalence to uninterrupted random-number and data-order histories was not
  established.
- The 100-epoch checkpoints were selected before the annotation-ID correction
  and rescored afterward without checkpoint reselection. The early-stopping
  experiment used the corrected evaluator. The two training budgets also use
  different learning-rate schedules; they are not interchangeable comparisons.
- A later best epoch or stopping epoch is not evidence that every phase of
  convergence is slower. Tiny validation improvements reset patience. No
  ablation isolates the cause of the accuracy difference, including the absence
  of components from the full MALA/MAViT architecture.

## Verification scope

Regression coverage includes an independent dense-attention reference for
forward values and input/parameter gradients, FP64 preservation, CPU BF16
autocast, MHA parameter-layout compatibility, unsupported-input contracts,
all five generated YAML graphs, CLI patience behavior, and nano-scale loss
gradients through both modalities and inference with both detection heads.

```bash
python -m pytest -c tests/pytest.ini tests/test_mala.py tests/test_train_cli.py tests/test_rgbir26.py tests/test_modern_runtime.py tests/test_crowd_eval.py -q
YOLOFUSE_TEST_DEVICE=cuda python -m pytest -c tests/pytest.ini tests/test_rgbir26.py -k 'yolo26_both and defa' -q
```

Nano-scale MALA-core completed full training with effective AMP and compiled
training graphs, followed by independent checkpoint evaluation. The fixed
100-epoch experiment evaluated both detection heads; the latest early-stopping
comparison above reports the one-to-many NMS head.

The other four model scales have configurations but no corresponding full
runtime validation in these records. MALA-core ONNX/export parity, dedicated
GFLOPs accounting, and controlled latency benchmarks remain unverified.
Existing Softmax DEFA FlashAttention, export, and GFLOPs results do not validate
MALA-core. The dedicated `tools/profile_fusion_flops.py` counter remains scoped
to `decision` and `defa`; its Softmax attention formula must not be applied to
this core. No MALA-core GFLOPs or checkpoint-size benchmark is reported here.
