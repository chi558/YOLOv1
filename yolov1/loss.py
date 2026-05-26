from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from yolov1.box_ops import box_iou, xywh_to_xyxy


class YOLOv1Loss(nn.Module):
    def __init__(
        self,
        grid_size: int = 7,
        num_boxes: int = 2,
        num_classes: int = 20,
        lambda_coord: float = 5.0,
        lambda_noobj: float = 0.5,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        self.grid_size = grid_size
        self.num_boxes = num_boxes
        self.num_classes = num_classes
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj
        self.label_smoothing = label_smoothing

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        class_preds = preds[..., :self.num_classes]
        class_targets = targets[..., :self.num_classes]
        if self.label_smoothing > 0:
            class_targets = class_targets * (1 - self.label_smoothing) + self.label_smoothing / self.num_classes
        pred_boxes = preds[..., self.num_classes:].view(*preds.shape[:3], self.num_boxes, 5)
        target_boxes = targets[..., self.num_classes:].view(*targets.shape[:3], self.num_boxes, 5)

        obj_mask = target_boxes[..., 0, 4] > 0
        pred_boxes = torch.sigmoid(pred_boxes)
        pred_xy = pred_boxes[..., :2]
        pred_wh_sqrt = pred_boxes[..., 2:4]
        pred_conf = pred_boxes[..., 4]
        target_xy = target_boxes[..., :2]
        target_wh_sqrt = target_boxes[..., 2:4]

        pred_xyxy = self._to_xyxy(pred_xy, pred_wh_sqrt.square())
        target_xyxy = self._to_xyxy(target_xy, target_wh_sqrt.square())
        ious = self._matched_iou(pred_xyxy, target_xyxy)
        max_iou, best_box = ious.max(dim=-1, keepdim=True)
        box_ids = torch.arange(self.num_boxes, device=preds.device).view(1, 1, 1, self.num_boxes)
        responsible = (box_ids == best_box) & obj_mask.unsqueeze(-1)

        zero = preds.sum() * 0
        if responsible.any():
            coord_loss = self.lambda_coord * (
                F.mse_loss(pred_xy[responsible], target_xy[responsible], reduction="sum")
                + F.mse_loss(pred_wh_sqrt[responsible], target_wh_sqrt[responsible], reduction="sum")
            )
            obj_loss = F.mse_loss(pred_conf[responsible], max_iou.expand_as(pred_conf)[responsible].detach(), reduction="sum")
            class_loss = F.mse_loss(class_preds[obj_mask], class_targets[obj_mask], reduction="sum")
        else:
            coord_loss = zero
            obj_loss = zero
            class_loss = zero
        noobj = ~responsible
        noobj_loss = self.lambda_noobj * F.mse_loss(pred_conf[noobj], torch.zeros_like(pred_conf[noobj]), reduction="sum")
        return (coord_loss + obj_loss + noobj_loss + class_loss) / preds.shape[0]

    def _to_xyxy(self, xy: torch.Tensor, wh: torch.Tensor) -> torch.Tensor:
        shifts_y, shifts_x = torch.meshgrid(
            torch.arange(self.grid_size, device=xy.device, dtype=xy.dtype),
            torch.arange(self.grid_size, device=xy.device, dtype=xy.dtype),
            indexing="ij",
        )
        shifts = torch.stack((shifts_x, shifts_y), dim=-1).view(1, self.grid_size, self.grid_size, 1, 2)
        centers = (xy + shifts) / float(self.grid_size)
        return xywh_to_xyxy(torch.cat((centers, wh), dim=-1))

    def _matched_iou(
        self,
        pred_xyxy: torch.Tensor,
        target_xyxy: torch.Tensor,
    ) -> torch.Tensor:
        ious = []
        for box_idx in range(self.num_boxes):
            ious.append(box_iou(
                pred_xyxy[..., box_idx, :].reshape(-1, 4),
                target_xyxy[..., box_idx, :].reshape(-1, 4),
            ).diag().view(*pred_xyxy.shape[:3]))
        return torch.stack(ious, dim=-1)
