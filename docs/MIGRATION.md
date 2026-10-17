# YOLOFuse26 migration and validation

## Source and architecture

YOLOFuse26 ports the RGB+IR fusion capability of [WangQvQ/YOLOFuse](https://github.com/WangQvQ/YOLOFuse) onto the vendored [Ultralytics v8.4.180](https://github.com/ultralytics/ultralytics/tree/v8.4.180) source. The local predecessor used Ultralytics 8.3.74 with a YOLO12 configuration. Fusion configurations are rebuilt from the official YOLO26 graph rather than converted from old checkpoints.

The migration preserves official YOLO26 backbone blocks, SPPF, C2PSA, `reg_max=1` detection heads, one-to-many / one-to-one training paths, Progressive Loss, STAL, and MuSGD. Seven fusion strategies have configurations for five model scales. Use `python tools/build_models.py` to regenerate the 35 configurations from the vendored upstream YAML.

Core additions:

- `ultralytics/data/rgbir.py`: strict paired image loading and directory-level `images` → `imagesIR` mapping.
- `ultralytics/nn/modules/fusion.py`, `DEA.py`, and `DEFA.py`: modality selection and fusion layers.
- `fusion_weights.py`: pretrained mapping by graph role, including both modality branches and both detection heads.
- `train_dual.py`, `val_dual.py`, and `infer_dual.py`: paired dataset entry points.
- `tests/test_rgbir26.py`: regression coverage for model and data boundaries.

Changes to upstream data loading, augmentation, prediction, model parsing, and plotting carry six-channel data through the normal Ultralytics pipeline. RGB and IR share geometric transforms; HSV augmentation affects RGB only. The RGB+IR cache uses a distinct suffix and checks both source timestamps.

## Recorded functional validation

The following results were recorded during migration on 2026-10-17. They describe functional validation, not a full training accuracy benchmark.

| Check | Recorded result |
| --- | --- |
| CPU regression suite | 14 passed |
| CUDA modality and head tests | 7 passed, covering all nano fusion variants |
| Real dataset structure audit | 6,000 train pairs and 2,676 validation pairs; matching filenames, dimensions, and valid labels |
| Decision integration check | 1 epoch on 8 paired images at 640 resolution, batch 2, MuSGD; validation on 4 pairs |
| Easy CLI checks | Short training, validation with JSON/COCO metrics, and paired prediction passed |
| Serialization and visualization | Checkpoint reload and matching fused detections on RGB/IR passed |
| TorchScript export | Six-channel `1 × 6 × 128 × 128` input; `1 × 300 × 6` output; PyTorch numerical parity passed |

Recorded environment: Python 3.10.8, PyTorch 2.1.2+cu121, torchvision 0.16.2+cu121, NumPy 1.26.3, Ultralytics 8.4.180, RTX 4080 SUPER, pytest 8.4.2, and faster-coco-eval 1.7.2.

## Reproduce regression tests

After installing the repository:

```bash
python -m pip install pytest
python -m pytest -c tests/pytest.ini tests/test_rgbir26.py -q
```

Optional CUDA tests, on an available GPU:

```bash
YOLOFUSE_TEST_DEVICE=cuda:0 python -m pytest -c tests/pytest.ini tests/test_rgbir26.py -q -k both_modalities
```

PowerShell equivalent:

```powershell
$env:YOLOFUSE_TEST_DEVICE = "cuda:0"
python -m pytest -c tests/pytest.ini tests/test_rgbir26.py -q -k both_modalities
Remove-Item Env:YOLOFUSE_TEST_DEVICE
```

`tools/smoke_test.py` is an optional dataset integration check that **runs a short training job**, validation, prediction, and export. The migration version of this tool expects the four-class dataset in `configs/rgbir.yaml`, JPG files with `day` / `night` filename prefixes, and source files on the same filesystem as its output so hard links can be created. Prepare those inputs before running it:

```bash
python tools/smoke_test.py --device cpu --imgsz 128
# Or use an available GPU:
python tools/smoke_test.py --device 0 --imgsz 640
```

Integration output is written to the ignored `reports/` directory. Dataset images, local logs, downloaded weights, and machine-specific migration records are not part of this repository.

## Compatibility and limits

- Runtime coverage is for nano models. Larger scales are generated configurations requiring separate validation.
- Old YOLOFuse weights cannot resume YOLO26 training. Initialize from matching official YOLO26 weights or start from scratch.
- Paired images need equal dimensions and prior spatial registration. Dimension checks do not establish registration quality.
- The default validation/prediction path is one-to-one and NMS-free; `--nms` selects one-to-many + NMS.
- File prediction supports paired images. A caller providing live video must synchronize and combine RGB/IR frames.
- TorchScript export has been validated. Other deployment formats need independent verification.
- Short integration metrics do not establish final detection accuracy. This repository does not claim a completed full training run or publish fully trained fusion weights.

## Attribution

- [YOLOFuse](https://github.com/WangQvQ/YOLOFuse): original multimodal fusion framework and DEA/DEFA implementations.
- [Ultralytics](https://github.com/ultralytics/ultralytics/tree/v8.4.180): YOLO26 implementation and API.
- [DEYOLO](https://arxiv.org/abs/2412.04931): DEA fusion reference linked by YOLOFuse.

The repository retains the AGPL-3.0 license and upstream source notices.
