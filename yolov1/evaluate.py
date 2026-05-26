from __future__ import annotations

import argparse
import contextlib
import io
import xml.etree.ElementTree as ET
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from yolov1.config import load_config
from yolov1.dataset import build_dataset
from yolov1.model import YOLOv1
from yolov1.predict import decode_predictions


def _load_coco_annotations(annotation: Path, classes: list[str], image_id: int, start_id: int) -> tuple[list[dict], int]:
    root = ET.parse(annotation).getroot()
    anns = []
    ann_id = start_id
    for obj in root.findall("object"):
        name = obj.findtext("name")
        if int(obj.findtext("difficult", default="0")) or name not in classes:
            continue
        bnd = obj.find("bndbox")
        if bnd is None:
            continue
        xmin = float(bnd.findtext("xmin", default="0"))
        ymin = float(bnd.findtext("ymin", default="0"))
        xmax = float(bnd.findtext("xmax", default="0"))
        ymax = float(bnd.findtext("ymax", default="0"))
        width = max(0.0, xmax - xmin)
        height = max(0.0, ymax - ymin)
        anns.append({
            "id": ann_id,
            "image_id": image_id,
            "category_id": classes.index(name) + 1,
            "bbox": [xmin, ymin, width, height],
            "area": width * height,
            "iscrowd": 0,
        })
        ann_id += 1
    return anns, ann_id


def _mean_ap(coco_eval: COCOeval, iou: float | None, area: str, max_det: int = 100) -> float:
    precision = coco_eval.eval["precision"]
    area_idx = coco_eval.params.areaRngLbl.index(area)
    max_det_idx = coco_eval.params.maxDets.index(max_det)
    if iou is None:
        values = precision[:, :, :, area_idx, max_det_idx]
    else:
        iou_idx = min(range(len(coco_eval.params.iouThrs)), key=lambda idx: abs(float(coco_eval.params.iouThrs[idx]) - iou))
        values = precision[iou_idx, :, :, area_idx, max_det_idx]
    values = values[values > -1]
    return float(values.mean()) if values.size else 0.0


def format_coco_metrics(metrics: dict[str, float]) -> str:
    return "\n".join([
        f" - Average Precision (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = {metrics['ap']:.3f}",
        f" - Average Precision (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = {metrics['ap50']:.3f}",
        f" - Average Precision (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = {metrics['ap75']:.3f}",
        f" - Average Precision (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = {metrics['ap_small']:.3f}",
        f" - Average Precision (AP) @[ IoU=0.50      | area= small | maxDets=100 ] = {metrics['ap50_small']:.3f}",
        f" - Average Precision (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = {metrics['ap_medium']:.3f}",
        f" - Average Precision (AP) @[ IoU=0.50      | area=medium | maxDets=100 ] = {metrics['ap50_medium']:.3f}",
        f" - Average Precision (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = {metrics['ap_large']:.3f}",
        f" - Average Precision (AP) @[ IoU=0.50      | area= large | maxDets=100 ] = {metrics['ap50_large']:.3f}",
    ])


def evaluate_model(
    model: torch.nn.Module,
    cfg: dict,
    device: torch.device,
    conf_threshold: float = 0.05,
) -> dict[str, float]:
    dataset = build_dataset(cfg, split="val", augment=False)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=cfg["dataset"]["num_workers"])
    was_training = model.training
    model.eval()

    images_json = []
    annotations_json = []
    detections_json = []
    ann_id = 1
    with torch.no_grad():
        for idx, (images, _) in enumerate(tqdm(loader, desc="evaluate")):
            image_path, anno_path = dataset.samples[idx]
            image_id = idx + 1
            with ImageSize(image_path) as image_size:
                width, height = image_size
                images_json.append({"id": image_id, "file_name": image_path.name, "width": width, "height": height})
                anns, ann_id = _load_coco_annotations(anno_path, cfg["classes"], image_id, ann_id)
                annotations_json.extend(anns)
                preds = model(images.to(device)).cpu()
                detections = decode_predictions(
                    preds,
                    image_size,
                    cfg["model"]["grid_size"],
                    cfg["model"]["num_boxes"],
                    cfg["model"]["num_classes"],
                    conf_threshold,
                    cfg["train"].get("nms_threshold", 0.45),
                )
            for det in detections:
                x1, y1, x2, y2 = det.box_xyxy
                x1 = max(0.0, min(float(x1), float(width)))
                y1 = max(0.0, min(float(y1), float(height)))
                x2 = max(0.0, min(float(x2), float(width)))
                y2 = max(0.0, min(float(y2), float(height)))
                detections_json.append({
                    "image_id": image_id,
                    "category_id": det.class_id + 1,
                    "bbox": [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)],
                    "score": det.score,
                })
    if was_training:
        model.train()
    if not detections_json:
        return {key: 0.0 for key in ("ap", "ap50", "ap75", "ap_small", "ap50_small", "ap_medium", "ap50_medium", "ap_large", "ap50_large")}

    coco_gt = COCO()
    coco_gt.dataset = {
        "info": {},
        "licenses": [],
        "images": images_json,
        "annotations": annotations_json,
        "categories": [{"id": idx + 1, "name": name} for idx, name in enumerate(cfg["classes"])],
    }
    coco_gt.createIndex()
    with contextlib.redirect_stdout(io.StringIO()):
        coco_dt = coco_gt.loadRes(detections_json)
        coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
        coco_eval.params.maxDets = [1, 10, 100]
        coco_eval.evaluate()
        coco_eval.accumulate()
    return {
        "ap": _mean_ap(coco_eval, None, "all"),
        "ap50": _mean_ap(coco_eval, 0.5, "all"),
        "ap75": _mean_ap(coco_eval, 0.75, "all"),
        "ap_small": _mean_ap(coco_eval, None, "small"),
        "ap50_small": _mean_ap(coco_eval, 0.5, "small"),
        "ap_medium": _mean_ap(coco_eval, None, "medium"),
        "ap50_medium": _mean_ap(coco_eval, 0.5, "medium"),
        "ap_large": _mean_ap(coco_eval, None, "large"),
        "ap50_large": _mean_ap(coco_eval, 0.5, "large"),
    }


def evaluate(config_path: str, checkpoint_path: str, conf_threshold: float = 0.05) -> dict[str, float]:
    cfg = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = YOLOv1(**cfg["model"]).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    return evaluate_model(model, cfg, device, conf_threshold)


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
    print(format_coco_metrics(metrics))


if __name__ == "__main__":
    main()
