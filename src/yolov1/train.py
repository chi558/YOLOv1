from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from yolov1.config import ensure_dir, load_config
from yolov1.dataset import build_dataset
from yolov1.loss import YOLOv1Loss
from yolov1.model import YOLOv1


def train(config_path: str, resume: str | None = None) -> None:
    cfg = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(int(cfg["train"].get("seed", 42)))

    train_set = build_dataset(cfg, split="train", augment=True)
    loader = DataLoader(train_set, batch_size=cfg["train"]["batch_size"], shuffle=True, num_workers=cfg["dataset"]["num_workers"], pin_memory=torch.cuda.is_available())
    model = YOLOv1(**cfg["model"]).to(device)
    criterion = YOLOv1Loss(
        grid_size=cfg["model"]["grid_size"],
        num_boxes=cfg["model"]["num_boxes"],
        num_classes=cfg["model"]["num_classes"],
        lambda_coord=cfg["train"]["lambda_coord"],
        lambda_noobj=cfg["train"]["lambda_noobj"],
        label_smoothing=cfg["train"].get("label_smoothing", 0.0),
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg["train"]["learning_rate"], momentum=cfg["train"]["momentum"], weight_decay=cfg["train"]["weight_decay"])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg["train"].get("lr_decay", []), gamma=0.1)
    start_epoch = 0
    if resume:
        checkpoint = torch.load(resume, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if "scheduler" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = int(checkpoint["epoch"]) + 1

    ckpt_dir = ensure_dir(cfg["train"]["checkpoint_dir"])
    for epoch in range(start_epoch, cfg["train"]["epochs"]):
        model.train()
        running = 0.0
        progress = tqdm(loader, desc=f"epoch {epoch + 1}/{cfg['train']['epochs']}")
        for images, targets in progress:
            images = images.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images), targets)
            loss.backward()
            optimizer.step()
            running += float(loss.item())
            progress.set_postfix(loss=running / max(1, progress.n))
        scheduler.step()
        checkpoint = {"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "config": cfg}
        torch.save(checkpoint, Path(ckpt_dir) / "last.pt")
        if (epoch + 1) % 10 == 0:
            torch.save(checkpoint, Path(ckpt_dir) / f"epoch_{epoch + 1:03d}.pt")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train YOLOv1 on PASCAL VOC.")
    parser.add_argument("--config", default="configs/voc.yaml")
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()
    train(args.config, args.resume)


if __name__ == "__main__":
    main()
