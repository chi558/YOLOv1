from __future__ import annotations

import argparse
from pathlib import Path

from torchvision.datasets import VOCDetection


def main() -> None:
    parser = argparse.ArgumentParser(description="Download PASCAL VOC 2007 and 2012 into data/VOCdevkit.")
    parser.add_argument("--root", default="data")
    args = parser.parse_args()
    root = Path(args.root)
    for year, image_set in [("2007", "trainval"), ("2007", "test"), ("2012", "trainval")]:
        VOCDetection(root=str(root), year=year, image_set=image_set, download=True)


if __name__ == "__main__":
    main()
