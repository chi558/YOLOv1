import pytest

torch = pytest.importorskip("torch")

from yolov1.loss import YOLOv1Loss


def test_loss_accepts_sqrt_width_height_targets_and_backpropagates():
    preds = torch.zeros(2, 7, 7, 30, requires_grad=True)
    targets = torch.zeros(2, 7, 7, 30)
    targets[:, 3, 3, 0] = 1.0
    targets[:, 3, 3, 20:25] = torch.tensor([0.5, 0.5, 0.5, 0.5, 1.0])
    targets[:, 3, 3, 25:30] = torch.tensor([0.5, 0.5, 0.5, 0.5, 1.0])

    loss = YOLOv1Loss()(preds, targets)
    loss.backward()

    assert torch.isfinite(loss)
    assert preds.grad is not None
