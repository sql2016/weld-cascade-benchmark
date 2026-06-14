#!/usr/bin/env python3
"""Train YOLO stage-1 binary defect detector."""

from __future__ import annotations

import argparse
from pathlib import Path

from config import GDXRAY_EVAL, STAGE1_ROOT


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0")
    parser.add_argument("--eval-gdxray", action="store_true")
    args = parser.parse_args()

    yaml_path = STAGE1_ROOT / "data.yaml"
    if not yaml_path.exists():
        raise SystemExit(f"Missing {yaml_path}. Run prepare_stage1.py first.")

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("pip install ultralytics") from exc

    model = YOLO(args.model)
    model.train(
        data=str(yaml_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=0,
        project=str(STAGE1_ROOT.parent / "runs"),
        name="stage1_defect",
    )
    metrics = model.val(data=str(yaml_path), workers=0)
    print(f"Val mAP50={metrics.box.map50:.4f} mAP50-95={metrics.box.map:.4f}")

    if args.eval_gdxray and (GDXRAY_EVAL / "data.yaml").exists():
        gdx = model.val(data=str(GDXRAY_EVAL / "data.yaml"))
        print(f"GDXray mAP50={gdx.box.map50:.4f} (external)")


if __name__ == "__main__":
    main()
