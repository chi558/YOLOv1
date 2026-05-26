import pytest

torch = pytest.importorskip("torch")

from yolov1.predict import decode_predictions


def test_decode_predictions_returns_detection():
    preds = torch.zeros(1, 7, 7, 30)
    preds[0, 3, 3, 0] = 10
    preds[0, 3, 3, 20 + 4] = 10
    detections = decode_predictions(preds, (448, 448), conf_threshold=0.2)
    assert detections
    assert detections[0].class_id == 0


def test_decode_predictions_squares_width_height_prediction():
    preds = torch.zeros(1, 7, 7, 30)
    preds[0, 0, 0, 1] = 1.0
    preds[0, 0, 0, 20 + 2] = 0.0
    preds[0, 0, 0, 20 + 3] = 0.0
    preds[0, 0, 0, 20 + 4] = 10.0

    detections = decode_predictions(preds, (448, 448), conf_threshold=0.2)

    assert detections
    x1, y1, x2, y2 = detections[0].box_xyxy
    assert (x2 - x1) == pytest.approx(112.0, abs=1.0)
    assert (y2 - y1) == pytest.approx(112.0, abs=1.0)
