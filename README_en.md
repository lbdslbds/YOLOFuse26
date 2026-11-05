<p align="center">English | <a href="README.md">中文</a></p>

<h1 align="center">YOLOFuse26</h1>
<p align="center">RGB + IR Multimodal Object Detection with YOLO26</p>

<p align="center">
  <a href="https://github.com/lbdslbds/YOLOFuse26"><img src="https://img.shields.io/badge/YOLO-26-2563eb" alt="YOLO26"></a>
  <a href="https://github.com/ultralytics/ultralytics/tree/v8.4.180"><img src="https://img.shields.io/badge/Ultralytics-8.4.180-blue" alt="Ultralytics 8.4.180"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-AGPL--3.0-green" alt="AGPL-3.0"></a>
  <a href="https://github.com/lbdslbds/YOLOFuse26/stargazers"><img src="https://img.shields.io/github/stars/lbdslbds/YOLOFuse26?style=social" alt="GitHub stars"></a>
</p>

**YOLOFuse26** brings the multimodal fusion approach of [YOLOFuse](https://github.com/WangQvQ/YOLOFuse) to **Ultralytics 8.4.180 / YOLO26**. It supports joint training, validation, and inference on paired visible (RGB) and infrared (IR) images, with eight fusion strategies for multimodal detection research in low light, surveillance, disaster response, and industrial inspection.

Migration and functional validation are complete, with full-training comparisons of DEFA and DEFA-MALA-core recorded below. Accuracy represents single runs on one fixed dataset split; see the [DEFA-MALA-core notes](docs/DEFA_MALA_CORE.md) for methods and limitations.

## 📋 Changelog

### 2026-11-06 · DEFA-MALA-core and full-training comparison

- Added `defa-mala-core`, replacing only DEFA's P3/P4/P5 cross-attention core while retaining dynamic convolution, gates, residuals, and enhancement. Five sizes (`n/s/m/l/x`) bring the current total to 40 fusion YAMLs.
- Nano runs completed both a 100-epoch comparison and a 300-epoch-limit comparison with patience=30, using effective AMP and Inductor compilation. MALA-core scored below DEFA on validation and test in the latest early-stopping experiment; no accuracy improvement is claimed.
- The training CLI exposes `--patience`, defaulting to 0 (disabled); resumed runs retain checkpoint settings. See [experiment details](docs/DEFA_MALA_CORE.md) for results and reproduction limits.

### 2026-10-23 · Compilation, export, and compute validation

- Decision / DEFA passed CPU and CUDA Inductor execution, gradient comparisons, and real paired-data smoke training with `compile=True + AMP`, followed by validation with both heads.
- Forced PyTorch SDPA FlashAttention for DEFA and verified CUDA forward/backward kernels.
- Added ONNX `dynamo=True` / `dynamic_shapes`, verified with ONNX Runtime at multiple shapes and with the legacy exporter.
- Retained original COCO crowd evaluation with fixed train/val/test splits of 5279/721/2676 pairs; measured full-resolution GFLOPs.
- See [runtime and validation notes](docs/MODERN_RUNTIME.md) for the old environment's AMP check failure and successful PyTorch 2.12.1 comparison.

### 2026-10-17 · YOLO26 migration

- Pinned the backbone to official `v8.4.180`, including the DFL-free YOLO26 head, one-to-one / one-to-many branches, and MuSGD.
- Rebuilt Data, Decision, Early, Mid, Easy, DEFA, and DEYOLO fusion graphs, with 35 YAML configurations across `n/s/m/l/x`.
- Unified six-channel inputs across training, validation, and inference. Geometric augmentation is shared; HSV augmentation affects RGB only.
- Added graph-based initialization from official YOLO26 pretrained weights and a separate RGB+IR disk cache.
- Passed 14 CPU and 7 CUDA regression cases, short training runs, both inference heads, and TorchScript export checks.

## ✨ Features

- **Joint RGB + IR detection** with matching filenames and shared YOLO annotations; grayscale IR is read as three channels.
- **Multiple fusion stages** spanning input fusion, dual backbones, multiscale features, and attention.
- **YOLO26 detection heads** with one-to-one NMS-free inference by default; `--nms` selects one-to-many + NMS.
- **Ultralytics Python API** through `YOLO(...).train()`, `.val()`, `.predict()`, and `.export()`.
- **Strict input checks** for missing IR images, decoding failures, and mismatched dimensions.
- **Single-modality support** for standard three-channel YOLO26 models.

## 🧩 Fusion strategies

| `--fusion` | Architecture | Nano configuration |
| --- | --- | --- |
| `data` | Six-channel input with a shared backbone and PAN | [yolo26n-data.yaml](ultralytics/cfg/models/fuse26/yolo26n-data.yaml) |
| `decision` (default) | Two YOLO26 backbones and PANs; interleaved modality features enter the detection head | [yolo26n-decision.yaml](ultralytics/cfg/models/fuse26/yolo26n-decision.yaml) |
| `early` | P2 fusion followed by a shared backbone and PAN | [yolo26n-early.yaml](ultralytics/cfg/models/fuse26/yolo26n-early.yaml) |
| `mid` | P3/P4/P5 fusion followed by a shared PAN | [yolo26n-mid.yaml](ultralytics/cfg/models/fuse26/yolo26n-mid.yaml) |
| `easy` | Fused P3/P4/P5 features from two backbones feed the detector directly | [yolo26n-easy.yaml](ultralytics/cfg/models/fuse26/yolo26n-easy.yaml) |
| `defa` | Dual backbones, Dynamic Enhanced Fusion Attention, and a shared PAN | [yolo26n-defa.yaml](ultralytics/cfg/models/fuse26/yolo26n-defa.yaml) |
| `defa-mala-core` | DEFA with MALA-core cross-attention; retains dynamic convolution, gates, and enhancement | [yolo26n-defa-mala-core.yaml](ultralytics/cfg/models/fuse26/yolo26n-defa-mala-core.yaml) |
| `deyolo` | Dual backbones, DEA attention from [DEYOLO](https://arxiv.org/abs/2412.04931), and a shared PAN | [yolo26n-deyolo.yaml](ultralytics/cfg/models/fuse26/yolo26n-deyolo.yaml) |

Select a model size with `--scale n/s/m/l/x`. Runtime validation currently covers **n**; larger configurations have been generated but have not been individually validated. DEFA attention can consume substantial memory at high resolution; adjust the batch size to your GPU.

## 📊 Model results and compute

| Architecture | Modality | Precision (P) | Recall (R) | mAP@50 | mAP@50:95 | Model size (MB) | Compute (GFLOPs) |
| --- | --- | --- | --- | --- | --- | --- | ---: |
| YOLO26n (baseline) | RGB | | | | | | |
| YOLO26n-Data | RGB+IR | | | | | | |
| YOLO26n-Decision | RGB+IR | | | | | | 10.626 |
| YOLO26n-Early | RGB+IR | | | | | | |
| YOLO26n-Mid | RGB+IR | | | | | | |
| YOLO26n-Easy | RGB+IR | | | | | | |
| YOLO26n-DEFA | RGB+IR | 81.11 | 76.50 | 81.31 | 57.04 | | 37.470 |
| YOLO26n-DEFA-MALA-core | RGB+IR | 82.33 | 74.50 | 79.68 | 55.84 | | |
| YOLO26n-DEYOLO | RGB+IR | | | | | | |

Detection metrics are percentages from independent **test** evaluation on a fixed RGB+IR split of 5279/721/2676 training/validation/test pairs, with person, rider, bicycle, and car classes. Both nano models used 640 inputs, batch=16, SGD, AMP, and `compile=True`, with a 300-epoch limit and patience=30. DEFA/MALA-core stopped at epochs 219/285; checkpoints from epochs 189/255 were selected by validation mAP50–95.

Final evaluation used FP32 eager execution, **one-to-many + NMS**, `conf=0.001`, `iou=0.7`, `max_det=300`, and COCO crowd matching. P/R use each class's maximum-F1 point on the COCO IoU=0.50 interpolated grid. Each configuration has one run; MALA-core resumed after epoch 144, and identical initialization of new fusion parameters was not established. No statistical significance or general superiority is claimed. See [experiment details](docs/DEFA_MALA_CORE.md#recorded-experiments) for validation scores and limitations. The original dataset, exact split lists/manifests, and trained checkpoints are not distributed with this repository. Exact reproduction requires those original assets; the example command provides training settings for your own data.

Compute was measured separately at nano scale, 4 classes, batch=1, with one 640×640 RGB+IR pair (six channels). It uses built-in model fusion, one deployment head, and THOP 2.2.3's `GFLOPs = 2 × MACs / 1e9` convention, including attention and dynamic-kernel products. Preprocessing and NMS are excluded. Both heads have identical counts for the measured Decision/DEFA models; a dedicated MALA-core count is unavailable. Checkpoint sizes and other blank entries remain unreported; smoke-test scores are not accuracy benchmarks. See [method and reproduction](docs/MODERN_RUNTIME.md#gflops-口径与复现).

## 🚀 Quick start

### 1. Environment

Python 3.10 or newer is recommended. Install a PyTorch / CUDA build compatible with your GPU environment.

```bash
git clone https://github.com/lbdslbds/YOLOFuse26.git
cd YOLOFuse26
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1`. Install `faster-coco-eval` before validation and `pytest` before running regression tests.

This repository vendors a modified `ultralytics` package. Use `pip install -e .` in a dedicated environment.

### 2. Dataset configuration

Edit [configs/rgbir.yaml](configs/rgbir.yaml): set `path` to the **absolute path** of your dataset, and set `nc` and `names` to match its labels. Keep `channels: 6`.

```yaml
path: /path/to/datasets
train: images/train
val: images/val
channels: 6
nc: 4
names:
  0: person
  1: rider
  2: bicycle
  3: car
```

### 3. Training

```bash
# Decision fusion; official YOLO26 nano pretrained weights download automatically
python train_dual.py --data configs/rgbir.yaml --fusion decision --scale n --device 0 --epochs 100 --batch 16

# Easy feature fusion
python train_dual.py --data configs/rgbir.yaml --fusion easy --scale n --device 0 --epochs 100 --batch 16

# MALA-core: at most 300 epochs, stopping after 30 epochs without validation improvement
python train_dual.py --data configs/rgbir-tvt.yaml --fusion defa-mala-core --scale n --device 0 --epochs 300 --patience 30 --imgsz 640 --batch 16 --workers 4 --optimizer SGD --cache disk --amp --compile --nms

```

The default pretrained path is `weights/yolo26{scale}.pt`. Ultralytics downloads the official asset if it is missing. Use `--weights /path/to/yolo26n.pt` for a local checkpoint or `--scratch` for random initialization. The pretrained scale must match `--scale`; classification outputs are initialized for the target dataset.

Defaults: `imgsz=640`, `optimizer=MuSGD`, `seed=0`, `amp=True`, `cache=False`, `patience=0`. For new runs, `--patience N` sets the early-stopping wait; 0 disables it. Resume retains the checkpoint's saved patience. New runs enable automatic mixed precision with the trainer's compatibility checks; use `--no-amp` for FP32. Options include `--cache disk`, `--cache ram`, `--optimizer SGD`, and `--device cpu`. Resumed runs retain the checkpoint's precision setting.

Add `--compile` to training or validation to enable `torch.compile`. See [runtime notes](docs/MODERN_RUNTIME.md) for the tested PyTorch 2.12.1 environment and modern ONNX export. See [crowd-aware detection](docs/CROWD.md) for the original-COCO evaluation and crowd training policy.

```bash
# Resume a checkpoint produced by this project
python train_dual.py --resume runs/train/yolo26n-decision/weights/last.pt --device 0
```

**Old YOLOFuse checkpoints are incompatible with the YOLO26 architecture and cannot resume training directly.** See [weights/README.md](weights/README.md).

### 4. Validation and inference

```bash
python val_dual.py --weights runs/train/yolo26n-decision/weights/best.pt --data configs/rgbir.yaml --device 0

python infer_dual.py --weights runs/train/yolo26n-decision/weights/best.pt --source /path/to/datasets/images/val --device 0
```

`--source` accepts an RGB image, directory, or TXT list. The `images` directory component is mapped to `imagesIR`. RGB visualizations are written under the corresponding `runs/predict/` run directory; IR visualizations are placed in its `ir/` subdirectory. Both show the same fused detections.

Add `--nms` to training, validation, or inference to select one-to-many + NMS. Paired file inputs support images. For live video, synchronize both streams yourself before supplying a six-channel input.

### 5. Python API

```python
from ultralytics import YOLO
from fusion_weights import initialize_from_yolo26

model = YOLO("ultralytics/cfg/models/fuse26/yolo26n-decision.yaml")
initialize_from_yolo26(model, "weights/yolo26n.pt")
model.train(data="configs/rgbir.yaml", device=0, epochs=100, nms=False,
            project="runs/train", name="yolo26n-decision")

trained = YOLO("runs/train/yolo26n-decision/weights/best.pt")
results = trained.predict("/path/to/datasets/images/val", device=0, nms=False)
```

## 📂 Dataset layout and input contract

```text
datasets/
├── images/                 # RGB images
│   ├── train/000001.jpg
│   └── val/000002.jpg
├── imagesIR/               # Registered IR images with matching names and dimensions
│   ├── train/000001.jpg
│   └── val/000002.jpg
└── labels/                 # Shared YOLO TXT annotations
    ├── train/000001.txt
    └── val/000002.txt
```

- Both modalities must have matching filenames and extensions. Each RGB path must contain exactly one directory named `images`.
- Register both modalities before use and ensure equal dimensions. The reader checks dimensions; registration quality remains a dataset preparation step.
- Each label row is `class_id x_center y_center width height`, with normalized coordinates in `[0, 1]`.
- Tensor and six-channel NumPy inputs use `RGB_R, RGB_G, RGB_B, IR_R, IR_G, IR_B` ordering. Tensors are BCHW in `[0, 1]`; NumPy images are HWC `uint8`.
- RGB+IR disk caches use `*.rgbir26.npy` and track modification times for both source images, independently of old YOLOFuse caches.

## 🧪 Validation and reproduction

Migration validation passed 14 CPU and 7 CUDA regression cases, covering gradients through both modalities, both inference heads, caching, serialization, missing IR, and single RGB compatibility. Decision / Easy passed short training, validation, and inference at 640 resolution. Six-channel TorchScript export passed numerical parity checks.

The 2026-10-23 validation on PyTorch 2.12.1 covers CPU/CUDA compilation, FlashAttention forward/backward, Decision/DEFA smoke training and both validation heads, plus 10 ONNX numerical export checks. See [runtime notes](docs/MODERN_RUNTIME.md) for commands and coverage. Accuracy changes require a separate full-training comparison.

```bash
python -m pytest -c tests/pytest.ini tests/test_rgbir26.py -q
```

See [migration and validation notes](docs/MIGRATION.md) for GPU tests, optional dataset integration checks, and coverage limits. Full-training DEFA/MALA-core summary metrics are reported above; trained weights are not distributed with the repository. Original DEFA export and FlashAttention checks do not validate MALA-core.

## ⚡ Cloud deployment

On AutoDL or another GPU server, clone the repository, configure PyTorch / CUDA, run `pip install -e .`, and update the dataset path. Then use the commands above. Tune batch size, optimizer, and training duration for your dataset and GPU memory.

## 📁 Repository layout

```text
YOLOFuse26/
├── configs/rgbir.yaml                  # Paired dataset template
├── docs/MIGRATION.md                   # Migration design and validation scope
├── ultralytics/cfg/models/fuse26/      # 40 YOLO26 fusion configurations
├── ultralytics/data/rgbir.py           # Paired reader and path mapping
├── ultralytics/nn/modules/            # Modality selection, DEA, and DEFA
├── fusion_weights.py                 # Pretrained parameter transfer
├── train_dual.py / val_dual.py / infer_dual.py
├── tools/                            # Graph generation and integration checks
└── tests/                            # RGB+IR regression tests
```

## 🙏 Acknowledgements and license

Built on [WangQvQ/YOLOFuse](https://github.com/WangQvQ/YOLOFuse) and [Ultralytics YOLO](https://github.com/ultralytics/ultralytics/tree/v8.4.180). Original attribution and the [AGPL-3.0 license](LICENSE) are retained. The official upstream README is preserved as [README_UPSTREAM.md](README_UPSTREAM.md).
