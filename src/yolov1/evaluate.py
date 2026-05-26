from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from yolov1.box_ops import box_iou
from yolov1.config import load_config
from yolov1.dataset import VOCDataset, parse_image_sets
from yolov1.model import YOLOv1
from yolov1.predict import decode_predictions


def _load_gt(annotation: Path, classes: list[str], image_size: tuple[int, int]) -> list[tuple[int, tuple[float, float, float, float]]]:
    width, height = image_size
    root = ET.parse(annotation).getroot()
    result = []
    for obj in root.findall("object"):
        name = obj.findtext("name")
        if int(obj.findtext("difficult", default="0")) or name not in classes:
            continue
        bnd = obj.find("bndbox")
        if bnd is None:
            continue
        result.append((
            classes.index(name),
            (
                float(bnd.findtext("xmin", default="0")),
                float(bnd.findtext("ymin", default="0")),
                float(bnd.findtext("xmax", default="0")),
                float(bnd.findtext("ymax", default="0")),
            ),
        ))
    return result


def evaluate(config_path: str, checkpoint_path: str, conf_threshold: float = 0.05, iou_threshold: float = 0.5) -> dict[str, float]:
    cfg = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = VOCDataset(
        root=cfg["dataset"]["root"],
        image_sets=parse_image_sets(cfg["dataset"]["image_sets"]["val"]),
        image_size=cfg["dataset"]["image_size"],
        grid_size=cfg["model"]["grid_size"],
        num_boxes=cfg["model"]["num_boxes"],
        classes=cfg["classes"],
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=cfg["dataset"]["num_workers"])
    model = YOLOv1(**cfg["model"]).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    true_positive = 0
    false_positive = 0
    false_negative = 0
    with torch.no_grad():
        for idx, (images, _) in enumerate(tqdm(loader, desc="evaluate")):
            image_path, anno_path = dataset.samples[idx]
            with ImageSize(image_path) as image_size:
                preds = model(images.to(device)).cpu()
                detections = decode_predictions(preds, image_size, cfg["model"]["grid_size"], cfg["model"]["num_boxes"], cfg["model"]["num_classes"], conf_threshold)
                gts = _load_gt(anno_path, cfg["classes"], image_size)
            matched: set[int] = set()
            for det in detections:
                candidates = [(gt_idx, gt_box) for gt_idx, (cls, gt_box) in enumerate(gts) if cls == det.class_id and gt_idx not in matched]
                if not candidates:
                    false_positive += 1
                    continue
                gt_boxes = torch.tensor([box for _, box in candidates], dtype=torch.float32)
                ious = box_iou(torch.tensor([det.box_xyxy], dtype=torch.float32), gt_boxes).squeeze(0)
                best = int(ious.argmax())
                if float(ious[best]) >= iou_threshold:
                    matched.add(candidates[best][0])
                    true_positive += 1
                else:
                    false_positive += 1
            false_negative += len(gts) - len(matched)
    precision = true_positive / max(1, true_positive + false_positive)
    recall = true_positive / max(1, true_positive + false_negative)
    return {"precision": precision, "recall": recall, "f1": 2 * precision * recall / max(1e-6, precision + recall)}


class ImageSize:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.image = None

    def __enter__(self) -> tuple[int, int]:
        from PIL import Image

        self.image = Image.open(self.path)
        return self.image.size

    def __exit__(self, exc_type, exc, traceback) -> None:
        if self.image is not None:
            self.image.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a YOLOv1 checkpoint on VOC.")
    parser.add_argument("--config", default="configs/voc.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--conf-threshold", type=float, default=0.05)
    args = parser.parse_args()
    metrics = evaluate(args.config, args.checkpoint, args.conf_threshold)
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")


if __name__ == "__main__":
    main()
