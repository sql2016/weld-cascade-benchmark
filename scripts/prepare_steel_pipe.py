#!/usr/bin/env python3
"""Normalize Steel Pipe dataset to YOLO layout under prepared/_steel_yolo/."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from config import DATA_ROOT, RAW_STEEL, STEEL_PIPE_NAMES
from io_utils import find_images, label_path_for_image

PREP_STEEL = DATA_ROOT / "prepared" / "_steel_yolo"


def _discover_steel_root() -> Path:
    candidates = [
        RAW_STEEL,
        RAW_STEEL / "steel-tube-dataset-all",
        RAW_STEEL / "YOLO",
        RAW_STEEL / "yolo",
    ]
    for c in candidates:
        if find_images(c):
            return c
    raise FileNotFoundError(
        f"No images under {RAW_STEEL}. Unzip steel-tube-dataset-all.zip there."
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--link", action="store_true", help="Hardlink instead of copy")
    args = parser.parse_args()

    root = _discover_steel_root()
    images = find_images(root)
    if PREP_STEEL.exists():
        shutil.rmtree(PREP_STEEL)
    img_out = PREP_STEEL / "images"
    lbl_out = PREP_STEEL / "labels"
    img_out.mkdir(parents=True)
    lbl_out.mkdir(parents=True)

    n = 0
    for img in images:
        lbl_src = label_path_for_image(img)
        if not lbl_src.exists():
            for alt in [
                img.parent.parent / "labels" / f"{img.stem}.txt",
                root / "labels" / f"{img.stem}.txt",
            ]:
                if alt.exists():
                    lbl_src = alt
                    break
        dst_img = img_out / f"steel_{img.stem}{img.suffix.lower()}"
        dst_lbl = lbl_out / f"steel_{img.stem}.txt"
        if args.link:
            dst_img.hardlink_to(img)
        else:
            shutil.copy2(img, dst_img)
        if lbl_src.exists():
            shutil.copy2(lbl_src, dst_lbl)
        else:
            dst_lbl.write_text("", encoding="utf-8")
        n += 1

    names_file = PREP_STEEL / "classes.txt"
    names_file.write_text("\n".join(STEEL_PIPE_NAMES) + "\n", encoding="utf-8")
    print(f"Steel Pipe: {n} images -> {PREP_STEEL}")


if __name__ == "__main__":
    main()
