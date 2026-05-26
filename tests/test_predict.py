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
