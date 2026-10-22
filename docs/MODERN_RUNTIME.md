# 新服务器迁移与 PyTorch 2.12 运行方式

项目在 `/root/autodl-tmp/YOLOFuse26`，虚拟环境在项目的 `.venv`。新环境使用系统提供的 Python 3.12.3、PyTorch 2.12.1+cu130、torchvision 0.27.1+cu130 和 Triton 3.7.1。CUDA 实测设备为 RTX 4080 SUPER，32 GB 显存。详细包版本见 `reports/modern_runtime_20261023/environment.json`。

```bash
cd /root/autodl-tmp/YOLOFuse26
source .venv/bin/activate
```

## 数据与评估合同

固定数据根目录为 `/root/autodl-tmp/datasets_tvt_20261019`，使用 `configs/rgbir-tvt.yaml` 或同内容的 `configs/rgbir.yaml`。

| split | RGB/IR 图像对 | 普通正样本框 | crowd 区域 |
| --- | ---: | ---: | ---: |
| train | 5279 | 18955 | 789 |
| val | 721 | 2586 | 111 |
| test | 2676 | 9902 | 531 |

保留四类别顺序 person、rider、bicycle、car，以及六通道顺序 RGB_R/G/B、IR_R/G/B。迁移了原图、普通标注、crowd 标注、原始 COCO annotations、固定 split 列表和 manifest；未复制可重建的 NPY/cache。

迁移基线核对了 43,844 个文件的 SHA256，缺失和哈希差异均为零，三个 split 无重叠。证据在 `/root/autodl-tmp/migration_20261023/integrity.json`。随后修改的目标代码另有备份及补丁哈希，位于该目录的 `baseline/` 和 `patch.json`。源服务器的现有训练继续运行。

评估继续直接使用原始 COCO image/category ID 与 iscrowd 标志，按类别进行 crowd 匹配；训练选取 best.pt 和独立验证都使用同一实现。默认 `conf=0.001`、`iou=0.70`、`max_det=300`。P/R 继续取 COCO IoU=0.50 插值网格上的最大 F1 点，不改为普通 YOLO 的置信度曲线指标。

```bash
# one-to-one，无 NMS
python val_dual.py --weights /path/to/best.pt --data configs/rgbir-tvt.yaml --split val --device 0
# one-to-many + NMS
python val_dual.py --weights /path/to/best.pt --data configs/rgbir-tvt.yaml --split test --device 0 --nms
```

## compile=True

训练、验证入口都支持 `--compile`，也可追加 `default`、`reduce-overhead`、`max-autotune` 或 `max-autotune-no-cudagraphs` 模式。Python API 可直接传 `compile=True`。

```bash
python train_dual.py --fusion defa --scale n --data configs/rgbir-tvt.yaml --device 0 --epochs 100 --imgsz 640 --batch 16 --optimizer SGD --amp --nms --compile
python val_dual.py --weights /path/to/best.pt --device 0 --nms --compile
```

锚点生成不再把 stride 张量读取为 Python 标量，避免 `aten._local_scalar_dense` 导致的编译断图。测试要求实际安装 OptimizedModule、生成 Inductor 内核并执行数值/梯度对照；出现静默回退将判为失败。

FP32 对照统一禁用 TF32，避免 eager 卷积和编译卷积使用不同精度。速度报告是相同输入、相同精度、预热后的短测，不能外推为全量训练的加速比例。

## DEFA FlashAttention

DEFA 的 `MultiheadAttention(..., need_weights=False)` 调用保持原权重结构。测试通过 `torch.nn.attention.sdpa_kernel(SDPBackend.FLASH_ATTENTION)` 强制仅允许 FlashAttention，并用 CUDA Profiler 核对前向和反向内核。该路径使用 PyTorch 自带的 SDPA FlashAttention。

DEFA 的 `Fold(kernel_size=1, stride=1)` 改为等价的 reshape，规避 ONNX Col2Im 导出限制。回归逐元素核对增强层之前的特征，并核对输出、输入梯度和全部参数梯度。reshape 保留不同内存布局时，后续卷积允许正常的 FP32 舍入误差。

```bash
python tools/validate_modern_runtime.py --suite flash --device 0 --imgsz 640 --output reports/check_flash
python tools/validate_modern_runtime.py --suite compile --device 0 --imgsz 640 --output reports/check_compile
python tools/validate_modern_runtime.py --suite compile --train --device 0 --imgsz 128 --output reports/check_compile_train
```

## 新版 ONNX

`dynamo=True` 选择 torch.export 导出路径，需要 PyTorch>=2.6、ONNXScript 和 opset>=18。`dynamic=True` 在该路径转换为 `dynamic_shapes`，不继续传旧 `dynamic_axes`；batch=1 的示例会用 batch=2 建图，以保留可变 batch。默认 `dynamo=False` 保持旧环境的导出入口。

```python
from ultralytics import YOLO

model = YOLO('/path/to/best.pt')
model.export(format='onnx', device='cpu', imgsz=640, batch=1,
             dynamo=True, dynamic=True, opset=18, simplify=False, nms=False)
# nms=None 导出 one-to-many 原始头，推理后按现有评估流程执行 NMS。
```

底层 `ultralytics.utils.export.engine.torch2onnx` 还支持显式 `dynamic_shapes`、`external_data`、`report` 和 `verify`，已有双输入真实导出及 ONNX Runtime 多尺寸回归。

ONNX 验证覆盖两种融合、两种检测头、静态/动态输入和旧导出兼容路径。动态输入核对 batch=1/2、128×128、160×192、192×160；原始头另用真实配对图像核对 640×640。one-to-one 的 TopK 在近似同分时可能改变行顺序，因此使用一一匹配核对检测集合，同时严格检查类别、坐标和分数。

```bash
python tools/validate_modern_runtime.py --suite onnx --output reports/check_onnx
python -m pytest -c tests/pytest.ini tests/test_rgbir26.py tests/test_crowd_data.py tests/test_crowd_eval.py tests/test_crowd_loss.py tests/test_modern_onnx.py tests/test_modern_runtime.py -q
```

本次机器结果和日志保存在 `reports/modern_runtime_20261023/`，训练 API 测试使用 8 对训练图像和 8 对验证图像的一轮短程运行；这些 checkpoint 用于功能验证。

| 验证项 | 实测范围 | 结果 |
| --- | --- | --- |
| CPU 回归 | 数据、crowd 损失/评估、导出、编译与六通道 GFLOPs | 61 项通过 |
| CPU Inductor | Decision / DEFA，128 分辨率推理与数值对照 | 通过，无断图 |
| CUDA Inductor | 两种融合，640 推理；128 训练前向、反向和参数更新 | 通过，无断图 |
| 训练 API | 两种融合，640、batch=2、compile=True、AMP、一轮训练及重载后的两种头验证 | 通过 |
| CUDA FlashAttention | DEFA 实际 80×80、40×40、20×20 特征层前反向，以及完整 320 模型反向 | 通过 |
| ONNX | 每种融合：旧版静态 one-to-many、新版静态/动态两种头，共 10 项；另核对真实 640 配对图像原始头 | 通过 |

CPU 模式的最终回归耗时 210.21 秒；GPU 训练为短程功能验证，未进行目标数据集的完整训练性能比较。

## AMP 自检原因与新环境结果

`check_amp` 使用官方三通道 YOLO26n 和 `bus.jpg`，将同一批 8 张图片的 FP32 与 AMP 检测框直接比较，判断为 `torch.allclose(a, b.float(), atol=0.5)`。本次对照使用相同权重与图片 SHA256、seed=0 和 deterministic=True。

| 环境 | 卷积 TF32 | 最大坐标差（原图像素） | 原始自检 |
| --- | --- | ---: | --- |
| PyTorch 2.1.2+cu121 / cuDNN 8.9 | 默认开启 | 0.509964 | 失败 |
| 同一旧环境 | 关闭 | 0.371626 | 通过 |
| PyTorch 2.12.1+cu130 / cuDNN 9.2 | 默认开启 | 0.185730 | 通过 |
| 同一新环境 | 关闭 | 0.175034 | 通过 |

旧环境中，第 4 个检测框的 x1 坐标因数值舍入差异略超容差，触发 AssertionError；检测数量、类别均相同，所有输出均为有限值。日志中可能出现 NaN 或零 mAP 的文字是失败后的通用提示，并非本次观察到的现象。对照结果支持旧软件栈中 TF32 与 AMP 数值路径的差异触发阈值；PyTorch、CUDA、cuDNN 和驱动同时变化，因此无法单独归因于某一个版本。

新环境保留原始检查逻辑和容差，直接通过默认设置自检。Decision 和 DEFA 均完成 `compile=True, amp=True`、640 分辨率、batch=2、真实配对图像的一轮训练，以及重载 checkpoint 后的两种检测头验证。新训练默认 `--amp`；续训仍沿用 checkpoint 原有精度设置。完整机器对照在 `reports/modern_runtime_20261023/amp_diagnosis.json`。

旧环境若需单独复现，可在新进程调用自检前设置 `torch.backends.cudnn.allow_tf32 = False`，本次对照同时保持 `torch.backends.cuda.matmul.allow_tf32 = False`。本次没有更改源服务器正在运行的训练。

## GFLOPs 口径与复现

使用 `ultralytics-thop==2.2.3`，按 YOLOFuse 的 `GFLOPs = 2 × THOP MACs / 10^9` 口径，直接运行完整 `1×6×640×640` 输入。模型为 n 尺寸、4 类，计入 RGB 与 IR 两条分支和融合模块；调用模型原有 `fuse()`，仅保留选定的一个部署检测头。one-to-one 和 one-to-many 的计数分别核对，结果相同。预处理、NMS、TopK 比较、内存搬运和反向传播不计入。

| 模型 | MACs | GFLOPs |
| --- | ---: | ---: |
| YOLO26n-Decision | 5,313,024,000 | 10.626048 |
| YOLO26n-DEFA | 18,734,957,150 | 37.4699143 |

THOP 2.2.3 的规则计入函数式矩阵乘法、DEFA 的动态 einsum，以及 MultiheadAttention 的 Q/K/V、输出投影、QK/AV 与 softmax。该计数器仍沿用 THOP 对归一化、池化和 softmax 的计数约定；不应把它与只统计卷积/矩阵乘法的计数器直接混用。FlashAttention 改变计算实现与内存开销，不改变此处的逻辑运算量。

当前 GFLOPs 函数已修复六通道模型的输入推断：从 `model.yaml['channels']` 读取外部输入通道，避免首个三通道分支卷积使计数失败后返回 0。该函数的 stride-aware 结果与本次完整分辨率计数一致。这里的结论针对 THOP 2.2.3；旧版 THOP 对注意力和函数式运算可能缺少规则。

```bash
python -m pip install ultralytics-thop==2.2.3
python tools/profile_fusion_flops.py --fusion decision defa --scale n --imgsz 640 --nc 4 --head both --cross-check
```

脚本在 CPU 上计数，以 CPU FlashAttention 控制内存；这与 CUDA FlashAttention 的实测是独立步骤。交叉检查使用 PyTorch `FlopCounterMode`，为其未内置计数规则的 CPU SDPA 算子补充 PyTorch 自己的 `sdpa_flop_count`。完整输出含逐层计数、实际注意力形状和动态核乘加量，保存在 `reports/fusion_flops.json`。公开摘要见 [640 分辨率计数](benchmarks/fusion_flops_640.json)。

接口参考：[PyTorch 2.12 ONNX](https://docs.pytorch.org/docs/2.12/onnx.html)、[SDPA 后端选择](https://docs.pytorch.org/docs/2.12/generated/torch.nn.attention.sdpa_kernel.html)、[THOP 计数实现](https://github.com/ultralytics/thop/blob/main/thop/profile.py)。
