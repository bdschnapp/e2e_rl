"""Export and crop Gazebo screenshots for the thesis."""

import argparse
from pathlib import Path

try:
    from PIL import Image
except ImportError:
    raise SystemExit("Pillow is required: pip install Pillow")


def process_screenshot(src: Path, dst: Path, crop: tuple | None = None):
    img = Image.open(src)
    if crop:
        img = img.crop(crop)  # (left, upper, right, lower)
    dst.parent.mkdir(parents=True, exist_ok=True)
    img.save(dst)
    print(f"Saved {dst}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("src", type=Path, help="Source screenshot")
    parser.add_argument("dst", type=Path, help="Output path")
    parser.add_argument("--crop", nargs=4, type=int, metavar=("L", "U", "R", "D"))
    args = parser.parse_args()
    process_screenshot(args.src, args.dst, tuple(args.crop) if args.crop else None)
