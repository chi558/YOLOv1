import pytest

torch = pytest.importorskip("torch")

from yolov1.box_ops import box_iou, nms


def test_box_iou_identical_boxes():
    boxes = torch.tensor([[0.0, 0.0, 1.0, 1.0]])
    assert torch.allclose(box_iou(boxes, boxes), torch.ones(1, 1))


def test_nms_keeps_highest_scoring_overlap():
    boxes = torch.tensor([[0.0, 0.0, 1.0, 1.0], [0.1, 0.1, 1.1, 1.1], [2.0, 2.0, 3.0, 3.0]])
    scores = torch.tensor([0.9, 0.8, 0.7])
    keep = nms(boxes, scores, 0.5).tolist()
    assert keep == [0, 2]
