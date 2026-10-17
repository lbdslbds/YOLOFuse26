<p align="center"><a href="README_en.md">English</a> | 中文</p>

<h1 align="center">YOLOFuse26</h1>
<p align="center">基于 YOLO26 的 RGB + IR 多模态目标检测框架</p>

<p align="center">
  <a href="https://github.com/lbdslbds/YOLOFuse26"><img src="https://img.shields.io/badge/YOLO-26-2563eb" alt="YOLO26"></a>
  <a href="https://github.com/ultralytics/ultralytics/tree/v8.4.180"><img src="https://img.shields.io/badge/Ultralytics-8.4.180-blue" alt="Ultralytics 8.4.180"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-AGPL--3.0-green" alt="AGPL-3.0"></a>
  <a href="https://github.com/lbdslbds/YOLOFuse26/stargazers"><img src="https://img.shields.io/github/stars/lbdslbds/YOLOFuse26?style=social" alt="GitHub stars"></a>
</p>

**YOLOFuse26** 延续 [YOLOFuse](https://github.com/WangQvQ/YOLOFuse) 的双模态融合思路，将模型基座迁移到 **Ultralytics 8.4.180 / YOLO26**，支持可见光（RGB）与红外（IR）图像的联合训练、验证和推理。项目提供 7 种融合策略，适用于低照度感知、安防监控、灾害救援及工业巡检等多模态目标检测研究。

当前已完成迁移和功能验证；完整训练后的检测精度仍需在目标数据集上评估。

## 📋 更新日志

### 2026-10-17 · YOLO26 迁移

- 基座固定为官方 `v8.4.180`，接入 YOLO26 的无 DFL 检测头、one-to-one / one-to-many 双头和 MuSGD。
- 重建 Data、Decision、Early、Mid、Easy、DEFA、DEYOLO 共 7 种融合配置，生成 `n/s/m/l/x` 共 35 份模型 YAML。
- 统一训练、验证和推理的六通道输入，RGB/IR 采用同步几何增强，HSV 增强仅作用于 RGB。
- 提供官方 YOLO26 预训练权重的分支映射初始化，以及独立的 RGB+IR 磁盘缓存。
- 完成 14 项 CPU、7 项 CUDA 回归测试，以及短程训练、双头推理和 TorchScript 导出验证。

## ✨ 技术特性

- **RGB + IR 联合检测**：同名图像配对、共享 YOLO 标注，灰度 IR 自动读取为三通道。
- **多层次融合**：从六通道输入融合到双主干、多尺度特征和注意力融合，可按任务选择结构。
- **YOLO26 检测头**：默认使用 one-to-one 无 NMS 推理，`--nms` 切换到 one-to-many + NMS。
- **Ultralytics Python API**：保留 `YOLO(...).train()`、`.val()`、`.predict()` 和 `.export()` 入口。
- **严格配对检查**：缺少 IR、解码失败或尺寸不一致时明确报错，便于发现数据问题。
- **单模态兼容**：标准三通道 YOLO26 仍可独立使用。

## 🧩 融合策略

| `--fusion` | 结构 | 配置示例（nano） |
| --- | --- | --- |
| `data` | 六通道输入，共享主干与 PAN | [yolo26n-data.yaml](ultralytics/cfg/models/fuse26/yolo26n-data.yaml) |
| `decision`（默认） | 两个 YOLO26 主干与 PAN，多尺度特征交错输入检测头 | [yolo26n-decision.yaml](ultralytics/cfg/models/fuse26/yolo26n-decision.yaml) |
| `early` | P2 特征融合，后续主干与 PAN 共享 | [yolo26n-early.yaml](ultralytics/cfg/models/fuse26/yolo26n-early.yaml) |
| `mid` | P3/P4/P5 特征融合后进入共享 PAN | [yolo26n-mid.yaml](ultralytics/cfg/models/fuse26/yolo26n-mid.yaml) |
| `easy` | 双主干 P3/P4/P5 特征分别融合后直接检测 | [yolo26n-easy.yaml](ultralytics/cfg/models/fuse26/yolo26n-easy.yaml) |
| `defa` | 双主干 + 动态增强融合注意力 + 共享 PAN | [yolo26n-defa.yaml](ultralytics/cfg/models/fuse26/yolo26n-defa.yaml) |
| `deyolo` | 双主干 + [DEYOLO](https://arxiv.org/abs/2412.04931) 的 DEA 注意力 + 共享 PAN | [yolo26n-deyolo.yaml](ultralytics/cfg/models/fuse26/yolo26n-deyolo.yaml) |

使用 `--scale n/s/m/l/x` 选择模型尺寸。当前运行验证覆盖 **n**；其余尺寸已生成配置，尚未逐一验证。DEFA 的跨模态注意力开销较大，高分辨率训练时需按显存调整 batch。

## 🚀 快速开始

### 1. 环境初始化

建议使用 Python 3.10 或更新版本，并根据 GPU 环境安装适配的 PyTorch / CUDA。

```bash
git clone https://github.com/lbdslbds/YOLOFuse26.git
cd YOLOFuse26
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

Windows PowerShell 使用 `.venv\Scripts\Activate.ps1` 激活环境。运行验证命令前安装 `python -m pip install faster-coco-eval`，运行回归测试前安装 `python -m pip install pytest`。

本项目以源码形式提供修改后的 `ultralytics` 包，请在独立环境中使用 `pip install -e .`。

### 2. 准备数据

编辑 [configs/rgbir.yaml](configs/rgbir.yaml)，将 `path` 改为数据集的**绝对路径**，并按标注设置 `nc` 和 `names`。保持 `channels: 6`。

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

### 3. 模型训练

```bash
# 默认 Decision 融合，自动获取官方 YOLO26 nano 预训练权重
python train_dual.py --data configs/rgbir.yaml --fusion decision --scale n --device 0 --epochs 100 --batch 16

# Easy 特征融合
python train_dual.py --data configs/rgbir.yaml --fusion easy --scale n --device 0 --epochs 100 --batch 16
```

预训练权重默认位于 `weights/yolo26{scale}.pt`，缺失时由 Ultralytics 下载官方权重。可以用 `--weights /path/to/yolo26n.pt` 指定本地文件，或用 `--scratch` 从头训练。预训练权重尺寸必须与 `--scale` 一致；分类输出层按目标数据集重新初始化。

默认参数：`imgsz=640`、`optimizer=MuSGD`、`seed=0`、`amp=False`、`cache=False`。支持 `--cache disk`、`--cache ram`、`--optimizer SGD` 和 `--device cpu`。

```bash
# 继续训练本项目产生的 checkpoint
python train_dual.py --resume runs/train/yolo26n-decision/weights/last.pt --device 0
```

**旧 YOLOFuse checkpoint 与 YOLO26 架构不兼容，不能直接用于续训。** 权重说明见 [weights/README.md](weights/README.md)。

### 4. 验证与推理

```bash
python val_dual.py --weights runs/train/yolo26n-decision/weights/best.pt --data configs/rgbir.yaml --device 0

python infer_dual.py --weights runs/train/yolo26n-decision/weights/best.pt --source /path/to/datasets/images/val --device 0
```

`--source` 接受 RGB 图片、图片目录或 TXT 路径列表；IR 根据路径中的 `images` 目录自动映射到 `imagesIR`。RGB 可视化保存在 `runs/predict/` 下对应运行目录，IR 可视化位于其 `ir/` 子目录，两者展示同一份融合检测结果。

训练、验证或推理时加 `--nms` 可选择 one-to-many + NMS 路径。六通道文件输入面向配对图片；实时视频需要调用方先同步两路帧，再提供六通道输入。

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

## 📂 数据集结构与输入规范

```text
datasets/
├── images/                 # RGB 图像
│   ├── train/000001.jpg
│   └── val/000002.jpg
├── imagesIR/               # 同名、同尺寸、已配准的 IR 图像
│   ├── train/000001.jpg
│   └── val/000002.jpg
└── labels/                 # 两个模态共享的 YOLO TXT 标注
    ├── train/000001.txt
    └── val/000002.txt
```

- RGB/IR 文件名和扩展名一致，路径中应恰好包含一个名为 `images` 的目录。
- 两个模态必须事先完成空间配准，并具有相同尺寸；读取器会检查尺寸，配准质量需自行确认。
- 标注格式为 `class_id x_center y_center width height`，坐标归一化到 `[0, 1]`。
- 张量与六通道 NumPy 输入按 `RGB_R, RGB_G, RGB_B, IR_R, IR_G, IR_B` 排列；张量为 BCHW、归一化到 `[0, 1]`，NumPy 为 HWC `uint8`。
- 双模态磁盘缓存使用 `*.rgbir26.npy`，检查两路源文件修改时间，独立于旧 YOLOFuse 缓存。

## 🧪 验证与复现

迁移阶段已通过 14 项 CPU 回归测试与 7 项 CUDA 模型测试，覆盖双模态梯度、双头推理、缓存、序列化、缺失 IR 和单 RGB 兼容性。Decision / Easy 已通过 640 分辨率短程训练、验证与推理；六通道 TorchScript 已完成导出和数值一致性检查。

```bash
python -m pytest -c tests/pytest.ini tests/test_rgbir26.py -q
```

GPU 测试、可选数据集集成测试及验证范围见 [迁移与验证说明](docs/MIGRATION.md)。当前未发布完整训练后的融合权重或目标数据集精度结果。

## ⚡ 云端部署

在 AutoDL 等 GPU 服务器上克隆仓库，配置 PyTorch / CUDA，执行 `pip install -e .` 并修改数据集路径后，即可使用以上命令。完整训练的 batch、优化器和训练时长需按数据集及显存调整。

## 📁 项目结构

```text
YOLOFuse26/
├── configs/rgbir.yaml                  # 双模态数据集模板
├── docs/MIGRATION.md                   # 迁移设计与功能验证范围
├── ultralytics/cfg/models/fuse26/      # 35 份 YOLO26 融合配置
├── ultralytics/data/rgbir.py           # 配对图像读取与路径映射
├── ultralytics/nn/modules/            # 模态选择、DEA 和 DEFA
├── fusion_weights.py                 # 预训练参数映射
├── train_dual.py / val_dual.py / infer_dual.py
├── tools/                            # 配置生成与集成检查
└── tests/                            # RGB+IR 回归测试
```

## 🙏 致谢与许可证

本项目基于 [WangQvQ/YOLOFuse](https://github.com/WangQvQ/YOLOFuse) 和 [Ultralytics YOLO](https://github.com/ultralytics/ultralytics/tree/v8.4.180) 开发，保留原有来源说明与 [AGPL-3.0 许可证](LICENSE)。官方上游 README 保存在 [README_UPSTREAM.md](README_UPSTREAM.md)。
