#!/usr/bin/env python3
"""Merge Steel + SWRD into stage-1 binary YOLO (class 0 = defect)."""

from __future__ import annotations

import random
import shutil
from pathlib import Path

from config import (
    PREPARED,
    RANDOM_SEED,
    STAGE1_ROOT,
    TEST_RATIO,
    TRAIN_RATIO,
    VAL_RATIO,
)
from io_utils import find_images, read_yolo_labels, split_ids, write_yolo_labels

STEEL = PREPARED / "_steel_yolo"
SWRD = PREPARED / "_swrd_yolo"


def _collect_sources() -> list[tuple[Path, Path, str]]:
    sources: list[tuple[Path, Path, str]] = []
    for tag, base in (("steel", STEEL), ("swrd", SWRD)):
        if (base / "images").exists():
            sources.append((base / "images", base / "labels", tag))
    if not sources:
        raise FileNotFoundError("Run prepare_steel_pipe.py and/or prepare_swrd.py first.")
    return sources


def main() -> None:
    if STAGE1_ROOT.exists():
        shutil.rmtree(STAGE1_ROOT)
    for split in ("train", "val", "test"):
        (STAGE1_ROOT / "images" / split).mkdir(parents=True)
        (STAGE1_ROOT / "labels" / split).mkdir(parents=True)

    records: list[tuple[str, Path, Path]] = []
    for img_dir, lbl_dir, tag in _collect_sources():
        for img in find_images(img_dir):
            lbl = lbl_dir / f"{img.stem}.txt"
            sid = f"{tag}_{img.stem}"
            records.append((sid, img, lbl))

    split_map = split_ids([r[0] for r in records], RANDOM_SEED, TRAIN_RATIO, VAL_RATIO)
    stats = {"train": 0, "val": 0, "test": 0, "defect_imgs": 0, "good_imgs": 0, "boxes": 0}

    for sid, img, lbl in records:
        sp = split_map[sid]
        dst_img = STAGE1_ROOT / "images" / sp / f"{sid}{img.suffix.lower()}"
        dst_lbl = STAGE1_ROOT / "labels" / sp / f"{sid}.txt"
        shutil.copy2(img, dst_img)
        rows = read_yolo_labels(lbl)
        binary = [(0, xc, yc, w, h) for _, xc, yc, w, h in rows]
        write_yolo_labels(dst_lbl, binary)
        stats[sp] += 1
        if binary:
            stats["defect_imgs"] += 1
            stats["boxes"] += len(binary)
        else:
            stats["good_imgs"] += 1

    yaml_path = STAGE1_ROOT / "data.yaml"
    yaml_path.write_text(
        f"path: {STAGE1_ROOT.as_posix()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"test: images/test\n"
        f"nc: 1\n"
        f"names: ['defect']\n",
        encoding="utf-8",
    )
    print(f"Stage-1 dataset -> {STAGE1_ROOT}")
    print(stats)


if __name__ == "__main__":
    main()
