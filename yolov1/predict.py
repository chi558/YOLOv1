from __future__ import annotations

from dataclasses import dataclass

import torch

from yolov1.box_ops import nms, xywh_to_xyxy


@dataclass(frozen=True)
class Detection:
    class_id: int
    score: float
    box_xyxy: tuple[float, float, float, float]


def decode_predictions(
    preds: torch.Tensor,
    image_size: tuple[int, int],
    grid_size: int = 7,
    num_boxes: int = 2,
    num_classes: int = 20,
    conf_threshold: float = 0.2,
    nms_threshold: float = 0.45,
) -> list[Detection]:
    if preds.ndim == 4:
        preds = preds.squeeze(0)
    class_probs = torch.softmax(preds[..., :num_classes], dim=-1)
    boxes = preds[..., num_classes:].view(grid_size, grid_size, num_boxes, 5)
    xy = torch.sigmoid(boxes[..., :2])
    wh = boxes[..., 2:4].abs().clamp(max=1)
    conf = torch.sigmoid(boxes[..., 4])

    detections: list[Detection] = []
    width, height = image_size
    for row in range(grid_size):
        for col in range(grid_size):
            for b in range(num_boxes):
                cls_score, cls_id = class_probs[row, col].max(dim=0)
                score = float(cls_score * conf[row, col, b])
                if score < conf_threshold:
                    continue
                cx = (col + float(xy[row, col, b, 0])) / grid_size
                cy = (row + float(xy[row, col, b, 1])) / grid_size
                bw = float(wh[row, col, b, 0])
                bh = float(wh[row, col, b, 1])
                x1, y1, x2, y2 = xywh_to_xyxy(torch.tensor([[cx, cy, bw, bh]])).squeeze(0).tolist()
                detections.append(Detection(int(cls_id), score, (x1 * width, y1 * height, x2 * width, y2 * height)))

    final: list[Detection] = []
    for class_id in sorted({det.class_id for det in detections}):
        class_dets = [det for det in detections if det.class_id == class_id]
        boxes_tensor = torch.tensor([det.box_xyxy for det in class_dets], dtype=torch.float32)
        scores_tensor = torch.tensor([det.score for det in class_dets], dtype=torch.float32)
        keep = nms(boxes_tensor, scores_tensor, nms_threshold).tolist()
        final.extend(class_dets[i] for i in keep)
    return sorted(final, key=lambda det: det.score, reverse=True)
