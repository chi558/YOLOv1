from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from yolov1.config import ensure_dir, load_config
from yolov1.dataset import build_dataset
from yolov1.evaluate import evaluate_model, format_coco_metrics
from yolov1.loss import YOLOv1Loss
from yolov1.model import YOLOv1


def _setup_logger(run_dir: str | Path) -> logging.Logger:
    ensure_dir(run_dir)
    logger = logging.getLogger("yolov1.train")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    file_handler = logging.FileHandler(Path(run_dir) / "train.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger


def _format_seconds(seconds: float) -> str:
    seconds = int(seconds)
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def train(config_path: str, resume: str | None = None) -> None:
    cfg = load_config(config_path)
    logger = _setup_logger(cfg["train"]["run_dir"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(int(cfg["train"].get("seed", 42)))
    logger.info("Start training")
    logger.info("Device: %s", device)
    logger.info("Config: %s", config_path)

    train_set = build_dataset(cfg, split="train", augment=True)
    loader = DataLoader(train_set, batch_size=cfg["train"]["batch_size"], shuffle=True, num_workers=cfg["dataset"]["num_workers"], pin_memory=torch.cuda.is_available())
    logger.info("Train samples: %d", len(train_set))
    logger.info("Batch size: %d, epochs: %d", cfg["train"]["batch_size"], cfg["train"]["epochs"])
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
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg["train"].get("lr_decay", []), gamma=cfg["train"].get("lr_gamma", 0.1))
    start_epoch = 0
    if resume:
        checkpoint = torch.load(resume, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if "scheduler" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        logger.info("Resumed from %s at epoch %d", resume, start_epoch + 1)

    ckpt_dir = ensure_dir(cfg["train"]["checkpoint_dir"])
    checkpoint_interval = int(cfg["train"].get("checkpoint_interval", 10))
    eval_interval = int(cfg["train"].get("eval_interval", 10))
    best_ap = -1.0
    total_start = time.perf_counter()
    for epoch in range(start_epoch, cfg["train"]["epochs"]):
        epoch_start = time.perf_counter()
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
        avg_loss = running / max(1, len(loader))
        lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        checkpoint = {"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "config": cfg}
        last_path = Path(ckpt_dir) / "last.pt"
        torch.save(checkpoint, last_path)
        saved_paths = [str(last_path)]
        if checkpoint_interval > 0 and (epoch + 1) % checkpoint_interval == 0:
            epoch_path = Path(ckpt_dir) / f"epoch_{epoch + 1:03d}.pt"
            torch.save(checkpoint, epoch_path)
            saved_paths.append(str(epoch_path))
        metrics = None
        if eval_interval > 0 and (epoch + 1) % eval_interval == 0:
            eval_start = time.perf_counter()
            metrics = evaluate_model(
                model=model,
                cfg=cfg,
                device=device,
                conf_threshold=cfg["train"].get("eval_conf_threshold", 0.05),
            )
            eval_time = time.perf_counter() - eval_start
            if metrics["ap"] > best_ap:
                best_ap = metrics["ap"]
                best_path = Path(ckpt_dir) / "best.pt"
                torch.save(checkpoint, best_path)
                saved_paths.append(str(best_path))
            logger.info(
                "Eval epoch %03d | AP %.4f | AP50 %.4f | AP75 %.4f | eval_time %s | best_AP %.4f",
                epoch + 1,
                metrics["ap"],
                metrics["ap50"],
                metrics["ap75"],
                _format_seconds(eval_time),
                best_ap,
            )
            for line in format_coco_metrics(metrics).splitlines():
                logger.info(line)
        epoch_time = time.perf_counter() - epoch_start
        total_time = time.perf_counter() - total_start
        metric_text = ""
        if metrics is not None:
            metric_text = f" | AP {metrics['ap']:.4f} | AP50 {metrics['ap50']:.4f} | AP75 {metrics['ap75']:.4f}"
        logger.info(
            "Epoch %03d/%03d | loss %.6f | lr %.6g%s | epoch_time %s | total_time %s | checkpoints %s",
            epoch + 1,
            cfg["train"]["epochs"],
            avg_loss,
            lr,
            metric_text,
            _format_seconds(epoch_time),
            _format_seconds(total_time),
            ", ".join(saved_paths),
        )
    logger.info("Finished training | total_time %s", _format_seconds(time.perf_counter() - total_start))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train YOLOv1 on PASCAL VOC.")
    parser.add_argument("--config", default="configs/voc.yaml")
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()
    train(args.config, args.resume)


if __name__ == "__main__":
    main()
