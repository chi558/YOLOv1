# YOLOv1 on PASCAL VOC

这是一个面向 PASCAL VOC 的 YOLOv1 训练、测试和推理项目。项目使用 PyTorch 实现，当前机器检测到的 Python 版本为 `3.11.15`；仓库本身不提交数据集、权重和训练输出。

## 功能

- VOC2007/VOC2012 数据集读取与 YOLOv1 标签编码
- YOLOv1 风格检测模型，默认使用 ResNet-50 特征提取器
- YOLOv1 损失函数，包括坐标、置信度、无目标和类别损失
- 训练脚本、评估脚本、单张图片 inference 脚本
- VOC 数据集下载脚本
- 基础单元测试

## 环境

建议使用虚拟环境：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

如果你使用 CUDA，请按照你的显卡和 CUDA 版本从 PyTorch 官网安装匹配的 `torch` / `torchvision` 版本，再安装其余依赖。

## 数据集

下载 VOC2007 trainval/test 和 VOC2012 trainval：

```powershell
python scripts/download_voc.py --root data
```

下载后目录应类似：

```text
data/
  VOCdevkit/
    VOC2007/
      Annotations/
      ImageSets/
      JPEGImages/
    VOC2012/
      Annotations/
      ImageSets/
      JPEGImages/
```

配置文件默认读取 `data/VOCdevkit`，可在 `configs/voc.yaml` 中修改。

## 训练

```powershell
python -m yolov1.train --config configs/voc.yaml
```

从已有 checkpoint 恢复：

```powershell
python -m yolov1.train --config configs/voc.yaml --resume checkpoints/last.pt
```

训练输出默认保存在 `checkpoints/`，该目录已被 `.gitignore` 忽略。

## 测试/评估

在 VOC2007 test split 上评估：

```powershell
python -m yolov1.evaluate --config configs/voc.yaml --checkpoint checkpoints/last.pt
```

当前评估脚本输出 `precision`、`recall` 和 `f1`。如果需要论文标准 VOC mAP，可以在此基础上扩展 AP 计算和 per-class 曲线。

运行项目单元测试：

```powershell
pytest
```

如果当前环境没有安装 PyTorch，相关测试会跳过。

## Inference

```powershell
python -m yolov1.infer `
  --config configs/voc.yaml `
  --checkpoint checkpoints/last.pt `
  --image path\to\image.jpg `
  --output runs\inference.jpg `
  --conf-threshold 0.25
```

输出图片会绘制检测框和类别分数。

## 配置

核心配置在 `configs/voc.yaml`：

- `dataset.root`: VOCdevkit 路径
- `dataset.image_sets.train`: 训练 split，默认 VOC2007 trainval + VOC2012 trainval
- `dataset.image_sets.val`: 验证/测试 split，默认 VOC2007 test
- `model.grid_size`: YOLO 网格大小，默认 7
- `model.num_boxes`: 每个网格预测框数，默认 2
- `train.batch_size`: batch size
- `train.epochs`: 训练轮数
- `train.learning_rate`: 初始学习率

## 项目结构

```text
configs/
  voc.yaml
scripts/
  download_voc.py
src/yolov1/
  box_ops.py
  config.py
  dataset.py
  evaluate.py
  infer.py
  loss.py
  model.py
  predict.py
  train.py
tests/
  test_box_ops.py
  test_predict.py
```

## GitHub

目标远程仓库：

```text
https://github.com/chi558/YOLOv1
```

如本机已经配置 GitHub 凭据，可以直接：

```powershell
git push -u origin main
```
