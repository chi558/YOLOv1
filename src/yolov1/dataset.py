from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


VOC_CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa",
    "train", "tvmonitor",
]


class VOCDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        image_sets: list[tuple[str, str]],
        image_size: int = 448,
        grid_size: int = 7,
        num_boxes: int = 2,
        classes: list[str] | None = None,
        augment: bool = False,
    ) -> None:
        self.root = Path(root)
        self.image_size = image_size
        self.grid_size = grid_size
        self.num_boxes = num_boxes
        self.classes = classes or VOC_CLASSES
        self.class_to_idx = {name: idx for idx, name in enumerate(self.classes)}
        self.samples = self._load_samples(image_sets)
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2) if augment else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def _load_samples(self, image_sets: list[tuple[str, str]]) -> list[tuple[Path, Path]]:
        samples: list[tuple[Path, Path]] = []
        for year, split in image_sets:
            base = self.root / year
            split_file = base / "ImageSets" / "Main" / f"{split}.txt"
            if not split_file.exists():
                raise FileNotFoundError(f"Missing VOC split file: {split_file}")
            for line in split_file.read_text(encoding="utf-8").splitlines():
                image_id = line.strip().split()[0]
                if not image_id:
                    continue
                samples.append((base / "JPEGImages" / f"{image_id}.jpg", base / "Annotations" / f"{image_id}.xml"))
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_path, anno_path = self.samples[index]
        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        target = self._encode_target(anno_path, width, height)
        return self.transform(image), target

    def _encode_target(self, anno_path: Path, width: int, height: int) -> torch.Tensor:
        target = torch.zeros((self.grid_size, self.grid_size, len(self.classes) + self.num_boxes * 5), dtype=torch.float32)
        root = ET.parse(anno_path).getroot()
        for obj in root.findall("object"):
            difficult = int(obj.findtext("difficult", default="0"))
            name = obj.findtext("name")
            if difficult or name not in self.class_to_idx:
                continue
            bnd = obj.find("bndbox")
            if bnd is None:
                continue
            xmin = float(bnd.findtext("xmin", default="0")) / width
            ymin = float(bnd.findtext("ymin", default="0")) / height
            xmax = float(bnd.findtext("xmax", default="0")) / width
            ymax = float(bnd.findtext("ymax", default="0")) / height
            cx = min(max((xmin + xmax) / 2, 0.0), 1.0)
            cy = min(max((ymin + ymax) / 2, 0.0), 1.0)
            bw = min(max(xmax - xmin, 0.0), 1.0)
            bh = min(max(ymax - ymin, 0.0), 1.0)
            cell_x = min(int(cx * self.grid_size), self.grid_size - 1)
            cell_y = min(int(cy * self.grid_size), self.grid_size - 1)
            if target[cell_y, cell_x, len(self.classes)] == 1:
                continue
            x_cell = cx * self.grid_size - cell_x
            y_cell = cy * self.grid_size - cell_y
            target[cell_y, cell_x, self.class_to_idx[name]] = 1
            for box_idx in range(self.num_boxes):
                start = len(self.classes) + box_idx * 5
                target[cell_y, cell_x, start:start + 5] = torch.tensor([x_cell, y_cell, bw, bh, 1.0])
        return target


def parse_image_sets(raw_sets: list[list[str]] | list[tuple[str, str]]) -> list[tuple[str, str]]:
    return [(str(year), str(split)) for year, split in raw_sets]
