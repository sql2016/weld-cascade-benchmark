"""Shared I/O for YOLO layouts and image discovery."""

from __future__ import annotations

import random
import shutil
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def find_images(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(
        p for p in root.rglob("*") if p.suffix.lower() in IMG_EXTS and p.is_file()
    )


def label_path_for_image(img: Path, labels_root: Path | None = None) -> Path:
    """Map image path to parallel labels/*.txt (YOLO convention)."""
    if labels_root is not None:
        rel = img.name
        return labels_root / f"{Path(rel).stem}.txt"
    parts = list(img.parts)
    if "images" in parts:
        idx = len(parts) - 1 - parts[::-1].index("images")
        parts[idx] = "labels"
        return Path(*parts).with_suffix(".txt")
    return img.with_suffix(".txt")


def read_yolo_labels(txt: Path) -> list[tuple[int, float, float, float, float]]:
    if not txt.exists():
        return []
    rows: list[tuple[int, float, float, float, float]] = []
    for line in txt.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        cls = int(float(parts[0]))
        xc, yc, w, h = map(float, parts[1:5])
        rows.append((cls, xc, yc, w, h))
    return rows


def write_yolo_labels(txt: Path, rows: list[tuple[int, float, float, float, float]]) -> None:
    txt.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{c} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}" for c, xc, yc, w, h in rows]
    txt.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def yolo_to_xyxy(
    cls: int, xc: float, yc: float, w: float, h: float, iw: int, ih: int
) -> tuple[int, int, int, int, int]:
    x1 = int((xc - w / 2) * iw)
    y1 = int((yc - h / 2) * ih)
    x2 = int((xc + w / 2) * iw)
    y2 = int((yc + h / 2) * ih)
    x1 = max(0, min(x1, iw - 1))
    y1 = max(0, min(y1, ih - 1))
    x2 = max(0, min(x2, iw - 1))
    y2 = max(0, min(y2, ih - 1))
    return cls, x1, y1, x2, y2


def split_ids(ids: list[str], seed: int, train: float, val: float) -> dict[str, str]:
    rng = random.Random(seed)
    shuffled = ids[:]
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(n * train)
    n_val = int(n * val)
    out: dict[str, str] = {}
    for i, sid in enumerate(shuffled):
        if i < n_train:
            out[sid] = "train"
        elif i < n_train + n_val:
            out[sid] = "val"
        else:
            out[sid] = "test"
    return out


def link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        dst.hardlink_to(src)
    except OSError:
        shutil.copy2(src, dst)
