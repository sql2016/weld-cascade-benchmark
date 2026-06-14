#!/usr/bin/env python3
"""Crop defect ROIs from Steel + SWRD for multi-class classification (stage 2)."""

from __future__ import annotations

import random
import shutil
from pathlib import Path

from PIL import Image

from config import (
    MIN_ROI_PX,
    PREPARED,
    RANDOM_SEED,
    ROI_PAD,
    SWRD_NAME_TO_UNIFIED,
    STAGE2_ROOT,
    STEEL_TO_UNIFIED,
    STEEL_PIPE_NAMES,
    UNIFIED_CLASSES,
)
from io_utils import find_images, read_yolo_labels, split_ids, yolo_to_xyxy

STEEL = PREPARED / "_steel_yolo"
SWRD = PREPARED / "_swrd_yolo"


def _swrd_class_map(swrd_root: Path) -> dict[int, str]:
    cls_file = swrd_root / "swrd_classes.txt"
    if not cls_file.exists():
        return {}
    names = [ln.strip() for ln in cls_file.read_text(encoding="utf-8").splitlines() if ln.strip()]
    out: dict[int, str] = {}
    for i, name in enumerate(names):
        key = name.lower().replace(" ", "_")
        out[i] = SWRD_NAME_TO_UNIFIED.get(key, "inclusion_other")
    return out


def _crop_roi(img: Image.Image, x1: int, y1: int, x2: int, y2: int, pad: float) -> Image.Image | None:
    iw, ih = img.size
    bw, bh = x2 - x1, y2 - y1
    if bw < 2 or bh < 2:
        return None
    px = int(bw * pad)
    py = int(bh * pad)
    xa = max(0, x1 - px)
    ya = max(0, y1 - py)
    xb = min(iw, x2 + px)
    yb = min(ih, y2 + py)
    if xb - xa < MIN_ROI_PX or yb - ya < MIN_ROI_PX:
        return None
    return img.crop((xa, ya, xb, yb))


def _process_source(
    img_dir: Path,
    lbl_dir: Path,
    tag: str,
    cls_map: dict[int, str],
    split_map: dict[str, str],
    counter: dict[str, int],
) -> int:
    n = 0
    for img_path in find_images(img_dir):
        sid = f"{tag}_{img_path.stem}"
        sp = split_map.get(sid, "train")
        lbl = lbl_dir / f"{img_path.stem}.txt"
        rows = read_yolo_labels(lbl)
        if not rows:
            continue
        img = Image.open(img_path).convert("L")
        iw, ih = img.size
        for j, (cls, xc, yc, w, h) in enumerate(rows):
            _, x1, y1, x2, y2 = yolo_to_xyxy(cls, xc, yc, w, h, iw, ih)
            crop = _crop_roi(img, x1, y1, x2, y2, ROI_PAD)
            if crop is None:
                continue
            ucls = cls_map.get(cls)
            if ucls is None or ucls not in UNIFIED_CLASSES:
                continue
            counter[ucls] = counter.get(ucls, 0) + 1
            idx = counter[ucls]
            out_dir = STAGE2_ROOT / sp / ucls
            out_dir.mkdir(parents=True, exist_ok=True)
            crop.save(out_dir / f"{sid}_{j}_{idx}.png")
            n += 1
    return n


def main() -> None:
    if STAGE2_ROOT.exists():
        shutil.rmtree(STAGE2_ROOT)

    steel_map = {i: STEEL_TO_UNIFIED[i] for i in range(len(STEEL_PIPE_NAMES))}
    swrd_map = _swrd_class_map(SWRD)

    all_ids: list[str] = []
    for tag, base in (("steel", STEEL), ("swrd", SWRD)):
        if (base / "images").exists():
            for img in find_images(base / "images"):
                all_ids.append(f"{tag}_{img.stem}")
    split_map = split_ids(all_ids, RANDOM_SEED, 0.8, 0.1)

    counter: dict[str, int] = {}
    total = 0
    if (STEEL / "images").exists():
        total += _process_source(
            STEEL / "images", STEEL / "labels", "steel", steel_map, split_map, counter
        )
    if (SWRD / "images").exists():
        total += _process_source(
            SWRD / "images", SWRD / "labels", "swrd", swrd_map, split_map, counter
        )

    meta = STAGE2_ROOT / "classes.txt"
    meta.write_text("\n".join(UNIFIED_CLASSES) + "\n", encoding="utf-8")
    print(f"Stage-2 ROIs: {total} crops -> {STAGE2_ROOT}")
    print("Per class:", counter)


if __name__ == "__main__":
    main()
