#!/usr/bin/env python3
"""Routed-eval confusion matrix + qualitative failure panels for the manuscript."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import confusion_matrix, f1_score

ROOT = Path(__file__).resolve().parents[1]
PLAN_B = ROOT / "scripts" / "weld_plan_b"
if str(PLAN_B) not in sys.path:
    sys.path.insert(0, str(PLAN_B))

OUT_JSON = ROOT / "artifacts/rc-weld-two-stage/deliverables/results/routed_viz.json"
PRIO_JSON = ROOT / "artifacts/rc-weld-two-stage/deliverables/results/priority_experiments.json"
BENCH_JSON = ROOT / "artifacts/rc-weld-two-stage/deliverables/results/revision_benchmark.json"
CHARTS = ROOT / "artifacts/rc-weld-two-stage/deliverables/charts"
CKPT_DIR = ROOT / "data/weld_plan_b/prepared/runs/routed_eval_ckpts"

# Reuse helpers from priority experiments
from run_weld_priority_experiments import (  # noqa: E402
    _ensure_classifier,
    _find_yolo_weights,
    _gt_boxes_unified,
    _iou,
    _load_cfg,
    _load_classifier,
    _resolve_mc_label_path,
)


def _routed_eval_viz(cfg, yolo_weights: Path, cls_ckpt: Path, split: str = "test") -> dict:
    from io_utils import find_images
    from ultralytics import YOLO

    conf, iou_thr = 0.25, 0.5
    pad = cfg.ROI_PAD
    det = YOLO(str(yolo_weights))
    clf, classes, cls_to_idx, tf = _load_classifier(cls_ckpt, "cpu")

    y_true: list[int] = []
    y_pred: list[int] = []
    n_gt = n_matched = n_missed = 0
    missed_cases: list[dict] = []
    miscls_cases: list[dict] = []

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
                if len(missed_cases) < 12:
                    missed_cases.append({
                        "image": str(img_path),
                        "stem": stem,
                        "gt_class": gt_cls,
                        "gt_box": list(gt_xy),
                        "best_iou": round(best_iou, 3),
                    })
                continue

            n_matched += 1
            used_det.add(best_i)
            x1, y1, x2, y2 = det_boxes[best_i]
            bw, bh = x2 - x1, y2 - y1
            crop_box = (
                max(0, int(x1 - bw * pad)),
                max(0, int(y1 - bh * pad)),
                min(iw, int(x2 + bw * pad)),
                min(ih, int(y2 + bh * pad)),
            )
            crop = img.crop(crop_box)
            x = tf(crop).unsqueeze(0)
            with torch.no_grad():
                pred_idx = int(clf(x).argmax(1).item())
            gt_idx = cls_to_idx.get(gt_cls)
            if gt_idx is None:
                continue
            y_true.append(gt_idx)
            y_pred.append(pred_idx)
            if pred_idx != gt_idx and len(miscls_cases) < 12:
                miscls_cases.append({
                    "image": str(img_path),
                    "stem": stem,
                    "gt_class": gt_cls,
                    "pred_class": classes[pred_idx],
                    "det_box": [x1, y1, x2, y2],
                    "crop_box": list(crop_box),
                })

    n_cls = len(classes)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_cls))).tolist()
    macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0)) if y_true else 0.0
    acc = float(np.mean(np.array(y_true) == np.array(y_pred))) if y_true else 0.0
    n_cls_correct = int(sum(t == p for t, p in zip(y_true, y_pred)))
    n_cls_wrong = int(n_matched - n_cls_correct)

    # Pick diverse failure examples for the figure
    def _pick(cases: list[dict], n: int) -> list[dict]:
        if len(cases) <= n:
            return cases
        step = max(len(cases) // n, 1)
        return [cases[i] for i in range(0, len(cases), step)][:n]

    return {
        "split": split,
        "class_names": classes,
        "confusion_matrix": cm,
        "n_gt_boxes": n_gt,
        "n_matched_boxes": n_matched,
        "n_missed_boxes": n_missed,
        "n_matched_correct": n_cls_correct,
        "n_matched_misclassified": n_cls_wrong,
        "routed_test_acc": acc,
        "routed_test_f1_macro": macro,
        "missed_examples": _pick(missed_cases, 3),
        "misclassified_examples": _pick(miscls_cases, 3),
    }


def _plot_routed_confusion(cm: list[list[int]], classes: list[str], path: Path) -> None:
    cm_arr = np.array(cm)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm_arr, cmap="Oranges")
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    short = [c.replace("_", "\n") for c in classes]
    ax.set_xticklabels(short, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(short, fontsize=8)
    ax.set_xlabel("Predicted (detector-routed crop)")
    ax.set_ylabel("Ground truth")
    ax.set_title("Detector-Routed Stage-2 Confusion Matrix (test split)")
    for i in range(cm_arr.shape[0]):
        for j in range(cm_arr.shape[1]):
            ax.text(j, i, str(cm_arr[i, j]), ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _draw_missed(ax, case: dict) -> None:
    img = Image.open(case["image"]).convert("L")
    ax.imshow(img, cmap="gray")
    x1, y1, x2, y2 = case["gt_box"]
    rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor="#e74c3c", facecolor="none")
    ax.add_patch(rect)
    ax.set_title(f"Missed: {case['gt_class'].replace('_', ' ')}\n(best IoU={case['best_iou']:.2f})", fontsize=8)
    ax.axis("off")


def _draw_miscls(ax, case: dict) -> None:
    img = Image.open(case["image"]).convert("L")
    x1, y1, x2, y2 = case["crop_box"]
    crop = img.crop((x1, y1, x2, y2))
    ax.imshow(crop, cmap="gray")
    gt = case["gt_class"].replace("_", " ")
    pr = case["pred_class"].replace("_", " ")
    ax.set_title(f"GT: {gt}\nPred: {pr}", fontsize=8, color="#c0392b")
    ax.axis("off")


def _plot_failure_cases(viz: dict, path: Path) -> None:
    missed = viz.get("missed_examples") or []
    miscls = viz.get("misclassified_examples") or []
    ncols = 3
    nrows = 2 if missed and miscls else 1
    fig, axes = plt.subplots(nrows, ncols, figsize=(9, 3.2 * nrows))
    if nrows == 1:
        axes = np.array([axes])
    for j in range(ncols):
        if j < len(missed):
            _draw_missed(axes[0, j], missed[j])
        else:
            axes[0, j].axis("off")
            axes[0, j].set_title("(no example)", fontsize=8)
    if nrows > 1:
        for j in range(ncols):
            if j < len(miscls):
                _draw_miscls(axes[1, j], miscls[j])
            else:
                axes[1, j].axis("off")
    fig.suptitle("Detector-Routed Qualitative Failures (Steel Pipe test split)", fontsize=10, y=1.02)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _merge_into_priority(viz: dict) -> None:
    prio = json.loads(PRIO_JSON.read_text(encoding="utf-8")) if PRIO_JSON.exists() else {}
    re = prio.setdefault("routed_eval", {})
    for k in (
        "confusion_matrix", "class_names", "n_matched_correct", "n_matched_misclassified",
        "missed_examples", "misclassified_examples",
    ):
        if k in viz:
            re[k] = viz[k]
    PRIO_JSON.write_text(json.dumps(prio, indent=2), encoding="utf-8")
    if BENCH_JSON.exists():
        bench = json.loads(BENCH_JSON.read_text(encoding="utf-8"))
        bench["priority_experiments"] = prio
        bench["routed_viz"] = {"confusion_matrix": viz["confusion_matrix"], "class_names": viz["class_names"]}
        BENCH_JSON.write_text(json.dumps(bench, indent=2), encoding="utf-8")


def main() -> None:
    cfg = _load_cfg()
    weights = _find_yolo_weights(cfg)
    if not weights:
        raise FileNotFoundError("Stage-1 YOLO weights not found")
    cls_ckpt = CKPT_DIR / "seed_0.pt"
    if not cls_ckpt.exists():
        cls_ckpt = _ensure_classifier(cfg, "cpu")
    else:
        print(f"Using P0 routed checkpoint {cls_ckpt}", flush=True)
    print("Running routed eval with visualization...", flush=True)
    viz = _routed_eval_viz(cfg, weights, cls_ckpt, split="test")
    CHARTS.mkdir(parents=True, exist_ok=True)
    _plot_routed_confusion(viz["confusion_matrix"], viz["class_names"], CHARTS / "fig_routed_confusion_matrix.png")
    _plot_failure_cases(viz, CHARTS / "fig_failure_cases.png")
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(viz, indent=2), encoding="utf-8")
    _merge_into_priority(viz)
    print(f"Wrote {OUT_JSON}", flush=True)
    print(f"Wrote {CHARTS / 'fig_routed_confusion_matrix.png'}", flush=True)
    print(f"Wrote {CHARTS / 'fig_failure_cases.png'}", flush=True)


if __name__ == "__main__":
    main()
