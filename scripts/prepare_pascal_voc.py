#!/usr/bin/env python3
"""Convert raw PASCAL VOC 2007/2012 into images/labels split folders.

Expected input:
    <voc-root>/VOC2007/JPEGImages, Annotations, ImageSets/Main
    <voc-root>/VOC2012/JPEGImages, Annotations, ImageSets/Main

Output:
    <output-root>/images/{train2007,val2007,test2007,train2012,val2012}
    <output-root>/labels/{train2007,val2007,test2007,train2012,val2012}
"""

from __future__ import annotations

import argparse
import shutil
import tarfile
import xml.etree.ElementTree as ET
from pathlib import Path


VOC_CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa",
    "train", "tvmonitor",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare HuggingFaceM4/pascal_voc files for this YOLOv1 repo.")
    parser.add_argument("--download-dir", type=Path, default=None, help="Directory containing voc2007.tar.gz and voc2012.tar.gz.")
    parser.add_argument("--voc-root", type=Path, required=True, help="Directory containing or receiving VOC2007 and VOC2012.")
    parser.add_argument("--output-root", type=Path, required=True, help="Prepared dataset root used by configs/voc.yaml.")
    parser.add_argument("--copy", action="store_true", help="Copy images instead of creating symlinks.")
    parser.add_argument("--skip-extract", action="store_true", help="Skip tar.gz extraction and only convert existing VOC folders.")
    return parser.parse_args()


def extract_archives(download_dir: Path, voc_root: Path) -> None:
    archives = [download_dir / "voc2007.tar.gz", download_dir / "voc2012.tar.gz"]
    for archive in archives:
        if not archive.is_file():
            raise FileNotFoundError(f"Missing archive: {archive}")
        with tarfile.open(archive, "r:gz") as tar:
            tar.extractall(voc_root)


def link_or_copy(src: Path, dst: Path, copy_file: bool) -> None:
    if dst.exists():
        return
    if copy_file:
        shutil.copy2(src, dst)
        return
    try:
        dst.symlink_to(src)
    except OSError:
        shutil.copy2(src, dst)


def convert_annotation(xml_path: Path, label_path: Path, cls2id: dict[str, int]) -> None:
    annotation = ET.parse(xml_path).getroot()
    size = annotation.find("size")
    if size is None:
        raise ValueError(f"Missing image size in {xml_path}")
    width = float(size.findtext("width", "0"))
    height = float(size.findtext("height", "0"))
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image size in {xml_path}: {width}x{height}")

    lines = []
    for obj in annotation.findall("object"):
        name = obj.findtext("name")
        if name not in cls2id:
            continue
        box = obj.find("bndbox")
        if box is None:
            continue
        xmin = float(box.findtext("xmin", "0"))
        ymin = float(box.findtext("ymin", "0"))
        xmax = float(box.findtext("xmax", "0"))
        ymax = float(box.findtext("ymax", "0"))
        x_center = ((xmin + xmax) / 2.0) / width
        y_center = ((ymin + ymax) / 2.0) / height
        box_width = (xmax - xmin) / width
        box_height = (ymax - ymin) / height
        lines.append(f"{cls2id[name]} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}")
    label_path.write_text("\n".join(lines), encoding="utf-8")


def convert_split(voc_dir: Path, split: str, output_name: str, output_root: Path, cls2id: dict[str, int], copy_images: bool) -> int:
    ids_file = voc_dir / "ImageSets" / "Main" / f"{split}.txt"
    if not ids_file.is_file():
        raise FileNotFoundError(f"Missing split file: {ids_file}")
    image_out = output_root / "images" / output_name
    label_out = output_root / "labels" / output_name
    image_out.mkdir(parents=True, exist_ok=True)
    label_out.mkdir(parents=True, exist_ok=True)
    image_ids = [line.strip().split()[0] for line in ids_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    for image_id in image_ids:
        src_image = voc_dir / "JPEGImages" / f"{image_id}.jpg"
        src_xml = voc_dir / "Annotations" / f"{image_id}.xml"
        if not src_image.is_file():
            raise FileNotFoundError(f"Missing image: {src_image}")
        if not src_xml.is_file():
            raise FileNotFoundError(f"Missing annotation: {src_xml}")
        link_or_copy(src_image, image_out / src_image.name, copy_images)
        convert_annotation(src_xml, label_out / f"{image_id}.txt", cls2id)
    return len(image_ids)


def find_voc_dir(voc_root: Path, name: str) -> Path:
    candidates = [
        voc_root / name,
        voc_root / "VOCdevkit" / name,
        voc_root / name.upper(),
        voc_root / "VOCdevkit" / name.upper(),
    ]
    for candidate in candidates:
        if (candidate / "JPEGImages").is_dir() and (candidate / "Annotations").is_dir():
            return candidate
    matches = [path for path in voc_root.rglob(name) if path.is_dir() and (path / "JPEGImages").is_dir()]
    if matches:
        return matches[0]
    raise FileNotFoundError(f"Could not find {name} under {voc_root}")


def main() -> None:
    args = parse_args()
    voc_root = args.voc_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    voc_root.mkdir(parents=True, exist_ok=True)

    if not args.skip_extract:
        if args.download_dir is None:
            raise ValueError("--download-dir is required unless --skip-extract is set")
        extract_archives(args.download_dir.expanduser().resolve(), voc_root)

    cls2id = {class_name: i for i, class_name in enumerate(VOC_CLASSES)}
    voc2007 = find_voc_dir(voc_root, "VOC2007")
    voc2012 = find_voc_dir(voc_root, "VOC2012")
    jobs = [
        (voc2007, "train", "train2007"),
        (voc2007, "val", "val2007"),
        (voc2007, "test", "test2007"),
        (voc2012, "train", "train2012"),
        (voc2012, "val", "val2012"),
    ]

    total = 0
    for voc_dir, split, output_name in jobs:
        count = convert_split(voc_dir, split, output_name, output_root, cls2id, args.copy)
        total += count
        print(f"{output_name}: {count} images")
    print(f"Done: {output_root}")
    print("Set configs/voc.yaml dataset.format to yolo_labels")
    print(f"Set configs/voc.yaml dataset.root to {output_root}")


if __name__ == "__main__":
    main()
