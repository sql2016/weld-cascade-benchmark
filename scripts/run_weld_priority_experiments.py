#!/usr/bin/env python3
"""Priority follow-up experiments: routed eval, stage-1 retrain, YOLO multiclass, GDXray."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import f1_score
from torchvision import models, transforms

ROOT = Path(__file__).resolve().parents[1]
PLAN_B = ROOT / "scripts" / "weld_plan_b"
if str(PLAN_B) not in sys.path:
    sys.path.insert(0, str(PLAN_B))
OUT = ROOT / "artifacts/rc-weld-two-stage/deliverables/results/priority_experiments.json"
BENCH = ROOT / "artifacts/rc-weld-two-stage/deliverables/results/revision_benchmark.json"
CHARTS = ROOT / "artifacts/rc-weld-two-stage/deliverables/charts"


def _load_cfg():
    import importlib.util

    spec = importlib.util.spec_from_file_location("weld_cfg", PLAN_B / "config.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    xa1, ya1, xa2, ya2 = a
    xb1, yb1, xb2, yb2 = b
    xi1, yi1 = max(xa1, xb1), max(ya1, yb1)
    xi2, yi2 = min(xa2, xb2), min(ya2, yb2)
    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    if inter <= 0:
        return 0.0
    area_a = max(0, xa2 - xa1) * max(0, ya2 - ya1)
    area_b = max(0, xb2 - xb1) * max(0, yb2 - yb1)
    return inter / max(area_a + area_b - inter, 1e-9)


def _find_yolo_weights(cfg) -> Path | None:
    runs = cfg.STAGE1_ROOT.parent / "runs"
    cands = sorted(runs.glob("stage1_defect*/weights/best.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0] if cands else None


def _train_stage1(cfg, epochs: int, device: str, name: str = "stage1_defect") -> dict:
    from ultralytics import YOLO

    yaml_path = cfg.STAGE1_ROOT / "data.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError(f"Missing {yaml_path}")
    t0 = time.time()
    model = YOLO("yolov8n.pt")
    model.train(
        data=str(yaml_path),
        epochs=epochs,
        imgsz=640,
        batch=8,
        device=device,
        workers=0,
        project=str(cfg.STAGE1_ROOT.parent / "runs"),
        name=name,
        exist_ok=True,
    )
    metrics = model.val(data=str(yaml_path), workers=0)
    run_dir = cfg.STAGE1_ROOT.parent / "runs" / name
    csv_path = run_dir / "results.csv"
    n_epochs = 0
    if csv_path.exists():
        n_epochs = len(list(csv.DictReader(csv_path.open(encoding="utf-8"))))
    return {
        "epochs": n_epochs,
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "weights": str(run_dir / "weights" / "best.pt"),
        "elapsed_sec": round(time.time() - t0, 1),
        "run_dir": str(run_dir),
    }


def _ensure_classifier(cfg, device: str = "cpu") -> Path:
    ckpt = cfg.STAGE2_ROOT.parent / "runs" / "stage2_cls" / "best.pt"
    if ckpt.exists():
        return ckpt
    import subprocess

    print("Training stage-2 classifier (5 ep) for routed eval...", flush=True)
    subprocess.run(
        [sys.executable, str(PLAN_B / "train_stage2.py"), "--epochs", "5", "--device", device, "--batch", "16"],
        cwd=str(PLAN_B),
        check=True,
    )
    if not ckpt.exists():
        raise FileNotFoundError(f"Classifier checkpoint not created: {ckpt}")
    return ckpt


def _load_classifier(ckpt_path: Path, device: str):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    classes = ckpt["classes"]
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, len(classes))
    model.load_state_dict(ckpt["model"])
    model.eval().to(device)
    tf = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.485, 0.456, 0.406]),
    ])
    cls_to_idx = {c: i for i, c in enumerate(classes)}
    return model, classes, cls_to_idx, tf


def _resolve_mc_label_path(cfg, image_stem: str) -> Path | None:
    if image_stem.startswith("steel_"):
        orig = image_stem[len("steel_") :]
        p = cfg.PREPARED / "_steel_yolo" / "labels" / f"{orig}.txt"
        return p if p.exists() else None
    if image_stem.startswith("swrd_"):
        orig = image_stem[len("swrd_") :]
        p = cfg.PREPARED / "_swrd_yolo" / "labels" / f"{orig}.txt"
        return p if p.exists() else None
    return None


def _swrd_class_map(cfg) -> dict[int, str]:
    cls_file = cfg.PREPARED / "_swrd_yolo" / "swrd_classes.txt"
    if not cls_file.exists():
        return {}
    names = [ln.strip() for ln in cls_file.read_text(encoding="utf-8").splitlines() if ln.strip()]
    out: dict[int, str] = {}
    for i, name in enumerate(names):
        key = name.lower().replace(" ", "_")
        out[i] = cfg.SWRD_NAME_TO_UNIFIED.get(key, "inclusion_other")
    return out


def _gt_boxes_unified(cfg, lbl_path: Path, iw: int, ih: int) -> list[tuple[str, tuple[int, int, int, int]]]:
    from io_utils import read_yolo_labels, yolo_to_xyxy

    rows = read_yolo_labels(lbl_path)
    swrd_map = _swrd_class_map(cfg) if "_swrd_yolo" in lbl_path.as_posix() else {}
    out: list[tuple[str, tuple[int, int, int, int]]] = []
    for cls, xc, yc, w, h in rows:
        if "_steel_yolo" in lbl_path.as_posix():
            u = cfg.STEEL_TO_UNIFIED.get(cls)
        else:
            u = swrd_map.get(cls, "inclusion_other")
        if u is None or u not in cfg.UNIFIED_CLASSES or u == "inclusion_other":
            continue
        _, x1, y1, x2, y2 = yolo_to_xyxy(cls, xc, yc, w, h, iw, ih)
        out.append((u, (x1, y1, x2, y2)))
    return out


def _routed_eval(cfg, yolo_weights: Path, cls_ckpt: Path, split: str = "test", conf: float = 0.25, iou_thr: float = 0.5) -> dict:
    from ultralytics import YOLO
    from io_utils import find_images

    device = "cpu"
    det = YOLO(str(yolo_weights))
    clf, classes, cls_to_idx, tf = _load_classifier(cls_ckpt, device)
    pad = cfg.ROI_PAD

    y_true: list[int] = []
    y_pred: list[int] = []
    n_gt = n_matched = n_missed = 0

    img_dir = cfg.STAGE1_ROOT / "images" / split
    for img_path in find_images(img_dir):
        stem = img_path.stem
        mc_lbl = _resolve_mc_label_path(cfg, stem)
        if mc_lbl is None:
            continue
        img = Image.open(img_path).convert("L")
        iw, ih = img.size
        gt_boxes = _gt_boxes_unified(cfg, mc_lbl, iw, ih)
        if not gt_boxes:
            continue

        res = det.predict(str(img_path), conf=conf, verbose=False)[0]
        det_boxes: list[tuple[int, int, int, int]] = []
        if res.boxes is not None and len(res.boxes):
            for box in res.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                det_boxes.append((x1, y1, x2, y2))

        used_det: set[int] = set()
        for gt_cls, gt_xy in gt_boxes:
            n_gt += 1
            best_i, best_iou = -1, 0.0
            for j, db in enumerate(det_boxes):
                if j in used_det:
                    continue
                iou = _iou(gt_xy, db)
                if iou > best_iou:
                    best_iou, best_i = iou, j
            if best_i < 0 or best_iou < iou_thr:
                n_missed += 1
                continue
            n_matched += 1
            used_det.add(best_i)
            x1, y1, x2, y2 = det_boxes[best_i]
            bw, bh = x2 - x1, y2 - y1
            crop = img.crop((
                max(0, int(x1 - bw * pad)),
                max(0, int(y1 - bh * pad)),
                min(iw, int(x2 + bw * pad)),
                min(ih, int(y2 + bh * pad)),
            ))
            x = tf(crop).unsqueeze(0)
            with torch.no_grad():
                pred_idx = int(clf(x).argmax(1).item())
            gt_idx = cls_to_idx.get(gt_cls)
            if gt_idx is None:
                continue
            y_true.append(gt_idx)
            y_pred.append(pred_idx)

    macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0)) if y_true else 0.0
    weighted = float(f1_score(y_true, y_pred, average="weighted", zero_division=0)) if y_true else 0.0
    acc = float(np.mean(np.array(y_true) == np.array(y_pred))) if y_true else 0.0
    n_cls_correct = int(sum(t == p for t, p in zip(y_true, y_pred)))
    n_cls_wrong = int(n_matched - n_cls_correct)
    return {
        "split": split,
        "conf_threshold": conf,
        "iou_match_threshold": iou_thr,
        "n_gt_boxes": n_gt,
        "n_matched_boxes": n_matched,
        "n_missed_boxes": n_missed,
        "n_matched_correct": n_cls_correct,
        "n_matched_misclassified": n_cls_wrong,
        "detection_recall_on_gt": float(n_matched / max(n_gt, 1)),
        "routed_test_acc": acc,
        "routed_test_f1_macro": macro,
        "routed_test_f1_weighted": weighted,
        "class_names": classes,
    }


def _train_multiclass_yolo(cfg, epochs: int, device: str) -> dict:
    import subprocess

    prep = PLAN_B / "prepare_stage1_multiclass.py"
    if not (cfg.PREPARED / "stage1_multiclass" / "data.yaml").exists():
        subprocess.run([sys.executable, str(prep)], cwd=str(PLAN_B), check=True)
    from ultralytics import YOLO

    yaml_path = cfg.PREPARED / "stage1_multiclass" / "data.yaml"
    t0 = time.time()
    model = YOLO("yolov8n.pt")
    model.train(
        data=str(yaml_path),
        epochs=epochs,
        imgsz=640,
        batch=8,
        device=device,
        workers=0,
        project=str(cfg.STAGE1_ROOT.parent / "runs"),
        name="stage1_multiclass",
        exist_ok=True,
    )
    metrics = model.val(data=str(yaml_path), workers=0)
    return {
        "epochs": epochs,
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "elapsed_sec": round(time.time() - t0, 1),
        "weights": str(cfg.STAGE1_ROOT.parent / "runs" / "stage1_multiclass" / "weights" / "best.pt"),
    }


def _gdxray_eval(yolo_weights: Path, cfg) -> dict:
    raw = cfg.RAW_GDXRAY
    if not raw.exists():
        return {"available": False, "note": f"GDXray not found at {raw.parent}"}
    import subprocess

    if str(PLAN_B) not in sys.path:
        sys.path.insert(0, str(PLAN_B))
    if not (cfg.GDXRAY_EVAL / "data.yaml").exists():
        subprocess.run([sys.executable, str(PLAN_B / "prepare_gdxray.py")], cwd=str(PLAN_B), check=True)
    from ultralytics import YOLO

    model = YOLO(str(yolo_weights))
    metrics = model.val(data=str(cfg.GDXRAY_EVAL / "data.yaml"), workers=0)
    return {
        "available": True,
        "n_images": len(list((cfg.GDXRAY_EVAL / "images").glob("*"))),
        "map50": float(metrics.box.map50),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
    }


def _plot_oracle_vs_routed(oracle_f1: float, routed_f1: float, det_rec: float) -> None:
    import matplotlib.pyplot as plt

    CHARTS.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    labels = ["Oracle ROI\nmacro F1", "Detector-routed\nmacro F1", "GT box match\nrate (IoU≥0.5)"]
    vals = [oracle_f1, routed_f1, det_rec]
    colors = ["#4477AA", "#EE6677", "#CCBB44"]
    ax.bar(labels, vals, color=colors, edgecolor="black", linewidth=0.6)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Oracle vs Detector-Routed Stage-2 (test split)")
    for i, v in enumerate(vals):
        ax.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(CHARTS / "fig_oracle_vs_routed.png", dpi=300)
    plt.close(fig)


def _merge_benchmark(priority: dict) -> None:
    if BENCH.exists():
        bench = json.loads(BENCH.read_text(encoding="utf-8"))
    else:
        bench = {}
    bench["priority_experiments"] = priority
    if priority.get("stage1_retrain"):
        bench["stage1"] = {
            **bench.get("stage1", {}),
            **priority["stage1_retrain"],
            "note": "Retrained stage-1 (priority experiments)",
        }
    BENCH.write_text(json.dumps(bench, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1-epochs", type=int, default=10, help="Stage-1 YOLO retrain epochs")
    parser.add_argument("--multiclass-epochs", type=int, default=10, help="Single-stage YOLO multiclass epochs")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--skip-stage1-train", action="store_true")
    parser.add_argument("--skip-multiclass", action="store_true")
    parser.add_argument("--skip-gdxray", action="store_true")
    args = parser.parse_args()

    t0 = time.time()
    cfg = _load_cfg()
    out: dict = {"generated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}

    # P1: Stage-1 retrain
    weights = _find_yolo_weights(cfg)
    if args.skip_stage1_train and weights:
        out["stage1_retrain"] = {"skipped": True, "weights": str(weights)}
    else:
        print(f"P1: Training stage-1 YOLO ({args.stage1_epochs} epochs)...", flush=True)
        out["stage1_retrain"] = _train_stage1(cfg, args.stage1_epochs, args.device)
        weights = Path(out["stage1_retrain"]["weights"])

    if not weights or not Path(weights).exists():
        raise FileNotFoundError("Stage-1 weights missing after training")

    # P0: Detector-routed evaluation
    print("P0: Detector-routed stage-2 evaluation...", flush=True)
    cls_ckpt = _ensure_classifier(cfg, args.device)
    out["routed_eval"] = _routed_eval(cfg, Path(weights), cls_ckpt, split="test")

    oracle_f1 = 0.0
    if BENCH.exists():
        oracle_f1 = float(json.loads(BENCH.read_text(encoding="utf-8"))["stage2_summary"]["f1_macro_mean"])
    _plot_oracle_vs_routed(
        oracle_f1,
        out["routed_eval"]["routed_test_f1_macro"],
        out["routed_eval"]["detection_recall_on_gt"],
    )

    # P2: Single-stage multiclass YOLO baseline
    if not args.skip_multiclass:
        print(f"P2: Single-stage YOLO multiclass ({args.multiclass_epochs} epochs)...", flush=True)
        out["yolo_multiclass_baseline"] = _train_multiclass_yolo(cfg, args.multiclass_epochs, args.device)

    # P3: GDXray external validation
    if not args.skip_gdxray:
        print("P3: GDXray external validation...", flush=True)
        out["gdxray_external"] = _gdxray_eval(Path(weights), cfg)

    out["elapsed_sec"] = round(time.time() - t0, 1)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    _merge_benchmark(out)
    print(f"Wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
