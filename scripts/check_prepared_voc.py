from __future__ import annotations

import argparse
from pathlib import Path


EXPECTED_SPLITS = {
    "train2007": 2501,
    "val2007": 2510,
    "test2007": 4952,
    "train2012": 5717,
    "val2012": 5823,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Check prepared PASCAL VOC images/labels layout.")
    parser.add_argument("--root", type=Path, required=True, help="Prepared root containing images/ and labels/.")
    args = parser.parse_args()
    root = args.root.expanduser().resolve()

    ok = True
    for split, expected in EXPECTED_SPLITS.items():
        image_dir = root / "images" / split
        label_dir = root / "labels" / split
        images = sorted(image_dir.glob("*.jpg")) if image_dir.is_dir() else []
        labels = sorted(label_dir.glob("*.txt")) if label_dir.is_dir() else []
        missing_labels = [image.stem for image in images if not (label_dir / f"{image.stem}.txt").is_file()]
        status = "OK" if len(images) == expected and len(labels) == expected and not missing_labels else "FAIL"
        print(f"{split}: {status} images={len(images)} labels={len(labels)} expected={expected}")
        if missing_labels[:5]:
            print(f"  missing labels sample: {', '.join(missing_labels[:5])}")
        ok = ok and status == "OK"

    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
