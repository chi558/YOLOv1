from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms

from yolov1.config import load_config
from yolov1.model import YOLOv1
from yolov1.predict import decode_predictions


def infer(config_path: str, checkpoint_path: str, image_path: str, output_path: str, conf_threshold: float = 0.25) -> None:
    cfg = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image = Image.open(image_path).convert("RGB")
    original_size = image.size
    transform = transforms.Compose([
        transforms.Resize((cfg["dataset"]["image_size"], cfg["dataset"]["image_size"])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    model = YOLOv1(**cfg["model"]).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    with torch.no_grad():
        preds = model(transform(image).unsqueeze(0).to(device)).cpu()
    detections = decode_predictions(
        preds,
        original_size,
        cfg["model"]["grid_size"],
        cfg["model"]["num_boxes"],
        cfg["model"]["num_classes"],
        conf_threshold,
        cfg["train"].get("nms_threshold", 0.45),
    )
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    for det in detections:
        x1, y1, x2, y2 = det.box_xyxy
        label = f"{cfg['classes'][det.class_id]} {det.score:.2f}"
        draw.rectangle((x1, y1, x2, y2), outline="red", width=3)
        draw.text((x1, max(0, y1 - 12)), label, fill="red", font=font)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run YOLOv1 inference on one image.")
    parser.add_argument("--config", default="configs/voc.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", default="runs/inference.jpg")
    parser.add_argument("--conf-threshold", type=float, default=0.25)
    args = parser.parse_args()
    infer(args.config, args.checkpoint, args.image, args.output, args.conf_threshold)


if __name__ == "__main__":
    main()
