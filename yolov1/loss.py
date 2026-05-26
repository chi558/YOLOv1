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
        noobj_mask = ~obj_mask
        pred_xy = torch.sigmoid(pred_boxes[..., :2])
        pred_wh = pred_boxes[..., 2:4].abs()
        pred_conf = torch.sigmoid(pred_boxes[..., 4])
        target_xy = target_boxes[..., :2]
        target_wh = target_boxes[..., 2:4]
        target_conf = target_boxes[..., 4]

        responsible = self._responsible_mask(pred_xy, pred_wh, target_xy[..., 0, :], target_wh[..., 0, :], obj_mask)

        zero = preds.sum() * 0
        if responsible.any():
            coord_loss = self.lambda_coord * (
                F.mse_loss(pred_xy[responsible], target_xy[responsible], reduction="sum")
                + F.mse_loss(torch.sqrt(pred_wh[responsible].clamp(min=1e-6)), torch.sqrt(target_wh[responsible].clamp(min=1e-6)), reduction="sum")
            )
            obj_loss = F.mse_loss(pred_conf[responsible], target_conf[responsible], reduction="sum")
            class_loss = F.mse_loss(class_preds[obj_mask], class_targets[obj_mask], reduction="sum")
        else:
            coord_loss = zero
            obj_loss = zero
            class_loss = zero
        noobj = noobj_mask.unsqueeze(-1).expand_as(pred_conf) | ~responsible
        noobj_loss = self.lambda_noobj * F.mse_loss(pred_conf[noobj], torch.zeros_like(pred_conf[noobj]), reduction="sum")
        return (coord_loss + obj_loss + noobj_loss + class_loss) / preds.shape[0]

    def _responsible_mask(
        self,
        pred_xy: torch.Tensor,
        pred_wh: torch.Tensor,
        target_xy: torch.Tensor,
        target_wh: torch.Tensor,
        obj_mask: torch.Tensor,
    ) -> torch.Tensor:
        mask = torch.zeros(pred_xy.shape[:-1], dtype=torch.bool, device=pred_xy.device)
        obj_indices = obj_mask.nonzero(as_tuple=False)
        for batch, row, col in obj_indices:
            pred = torch.cat((pred_xy[batch, row, col], pred_wh[batch, row, col]), dim=-1)
            tgt = torch.cat((target_xy[batch, row, col].unsqueeze(0), target_wh[batch, row, col].unsqueeze(0)), dim=-1)
            ious = box_iou(xywh_to_xyxy(pred), xywh_to_xyxy(tgt)).squeeze(1)
            best = int(ious.argmax())
            mask[batch, row, col, best] = True
        return mask
