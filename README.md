# YOLOv1 on PASCAL VOC

这是一个面向 PASCAL VOC 的 YOLOv1 训练、测试和推理项目。项目使用 PyTorch 实现，当前机器检测到的 Python 版本为 `3.11.15`；仓库本身不提交数据集、权重和训练输出。

## 功能

- VOC2007/VOC2012 数据集读取与 YOLOv1 标签编码
- YOLOv1 风格检测模型，默认使用 ResNet18 特征提取器
- YOLOv1 损失函数，包括坐标、置信度、无目标和类别损失
- 训练脚本、评估脚本、单张图片 inference 脚本
- 基础单元测试

## 环境

使用 conda 创建环境：

```bash
conda create -n yolov1 python=3.11 -y
conda activate yolov1
pip install -r requirements.txt
pip install -e .
```

## 数据集

本项目使用原始 VOC XML 布局。

使用 Hugging Face 下载 VOC2007 和 VOC2012：

```bash
mkdir -p data/pascal_voc data/VOCdevkit
hf download HuggingFaceM4/pascal_voc \
  voc2007.tar.gz voc2012.tar.gz \
  --repo-type dataset \
  --local-dir data/pascal_voc

tar -xzf data/pascal_voc/voc2007.tar.gz -C data/VOCdevkit
tar -xzf data/pascal_voc/voc2012.tar.gz -C data/VOCdevkit
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

此时 `configs/voc.yaml` 应设置为：

```yaml
dataset:
  format: voc_xml
  root: /path/to/VOCdevkit
```

例如目录是 `/data/VOCdevkit/VOC2007` 和 `/data/VOCdevkit/VOC2012`，则 `root` 写成 `/data/VOCdevkit`。
默认训练集为 VOC2007 trainval + VOC2012 trainval，eval 集为 VOC2007 test。

## 训练

```bash
python -m yolov1.train --config configs/voc.yaml
```

从已有 checkpoint 恢复：

```bash
python -m yolov1.train --config configs/voc.yaml --resume checkpoints/last.pt
```

训练输出默认保存在 `checkpoints/`，该目录已被 `.gitignore` 忽略。
训练日志默认保存在 `runs/train.log`，包含每个 epoch 的 loss、学习率、eval 指标、epoch 耗时、累计耗时和 checkpoint 路径。
checkpoint 默认每个 epoch 保存并覆盖 `checkpoints/last.pt`，每 10 个 epoch 额外保存一次 `checkpoints/epoch_XXX.pt`。
训练中默认每 10 个 epoch 在 VOC2007 test 上 eval 一次，并按 best F1 保存 `checkpoints/best.pt`。

## 测试/评估

在 VOC2007 test split 上评估：

```bash
python -m yolov1.evaluate --config configs/voc.yaml --checkpoint checkpoints/last.pt
```

当前评估脚本输出 `precision`、`recall` 和 `f1`。如果需要论文标准 VOC mAP，可以在此基础上扩展 AP 计算和 per-class 曲线。

运行项目单元测试：

```bash
pytest
```

如果当前环境没有安装 PyTorch，相关测试会跳过。

## Inference

```bash
python -m yolov1.infer \
  --config configs/voc.yaml \
  --checkpoint checkpoints/last.pt \
  --image /path/to/image.jpg \
  --output runs/inference.jpg \
  --conf-threshold 0.25
```

输出图片会绘制检测框和类别分数。

## 配置

核心配置在 `configs/voc.yaml`：

- `dataset.root`: VOCdevkit 路径
- `dataset.format`: `voc_xml`
- `dataset.image_sets.train`: 训练 split，默认 VOC2007 trainval + VOC2012 trainval
- `dataset.image_sets.val`: 验证/测试 split，默认 VOC2007 test
- `model.grid_size`: YOLO 网格大小，默认 7
- `model.num_boxes`: 每个网格预测框数，默认 2
- `model.backbone`: 主干网络，默认 `resnet18`
- `train.batch_size`: batch size，默认 32
- `train.epochs`: 训练轮数，默认 150
- `train.learning_rate`: 初始学习率，默认 0.001
- `train.lr_decay`: 学习率衰减 epoch，默认 `[90, 120]`
- `train.checkpoint_interval`: 额外保存 checkpoint 的间隔，默认 10
- `train.eval_interval`: 训练中 eval 间隔，默认 10

## 项目结构

```text
configs/
  voc.yaml
yolov1/
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

```bash
git push -u origin main
```
