#!/usr/bin/env python3
"""Revision benchmark: multi-seed Stage-2, per-class metrics, baselines, bootstrap CI."""

from __future__ import annotations

import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

ROOT = Path(__file__).resolve().parents[1]
PLAN_B = ROOT / "scripts" / "weld_plan_b"
OUT = ROOT / "artifacts/rc-weld-two-stage/deliverables/results/revision_benchmark.json"
CHARTS = ROOT / "artifacts/rc-weld-two-stage/deliverables/charts"


def _load_cfg():
    import importlib.util

    spec = importlib.util.spec_from_file_location("weld_cfg", PLAN_B / "config.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _stage1_metrics(cfg) -> dict:
    runs = cfg.STAGE1_ROOT.parent / "runs"
    cands = sorted(runs.glob("stage1_defect*/results.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not cands:
        return {"epochs": 0, "map50": 0.0, "precision": 0.0, "recall": 0.0}
    rows = list(csv.DictReader(cands[0].open(encoding="utf-8")))
    last = rows[-1]
    return {
        "epochs": len(rows),
        "map50": float(last.get("metrics/mAP50(B)", 0) or 0),
        "precision": float(last.get("metrics/precision(B)", 0) or 0),
        "recall": float(last.get("metrics/recall(B)", 0) or 0),
        "source": str(cands[0]),
    }


def _dataset_stats(cfg) -> dict:
    out = {"splits": {}, "classes": []}
    for split in ("train", "val", "test"):
        d = cfg.STAGE2_ROOT / split
        counts: dict[str, int] = {}
        if not d.exists():
            continue
        for cls_dir in sorted(d.iterdir()):
            if not cls_dir.is_dir():
                continue
            n = len(list(cls_dir.glob("*.jpg"))) + len(list(cls_dir.glob("*.png")))
            counts[cls_dir.name] = n
        out["splits"][split] = {"total": sum(counts.values()), "per_class": counts}
    if "val" in out["splits"]:
        out["classes"] = list(out["splits"]["val"]["per_class"].keys())
    return out


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _eval_loader(model, loader, device: str) -> tuple[list[int], list[int]]:
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            preds.extend(model(x).argmax(1).cpu().tolist())
            labels.extend(y.tolist())
    return preds, labels


def _metrics_from_preds(labels: list[int], preds: list[int], class_names: list[str]) -> dict:
    labels_arr = np.array(labels)
    preds_arr = np.array(preds)
    report = classification_report(
        labels, preds, target_names=class_names, output_dict=True, zero_division=0
    )
    per_class = {
        cls: {
            "precision": report[cls]["precision"],
            "recall": report[cls]["recall"],
            "f1": report[cls]["f1-score"],
            "support": int(report[cls]["support"]),
        }
        for cls in class_names
    }
    return {
        "acc": float((preds_arr == labels_arr).mean()),
        "f1_macro": float(f1_score(labels, preds, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(labels, preds, average="weighted", zero_division=0)),
        "per_class": per_class,
        "confusion_matrix": confusion_matrix(labels, preds).tolist(),
        "class_names": class_names,
    }


def _train_eval_resnet(
    cfg,
    seed: int,
    epochs: int = 5,
    lr: float = 1e-3,
    batch: int = 16,
    device: str = "cpu",
    ckpt_path: Path | None = None,
) -> dict:
    _set_seed(seed)
    train_dir = cfg.STAGE2_ROOT / "train"
    val_dir = cfg.STAGE2_ROOT / "val"
    test_dir = cfg.STAGE2_ROOT / "test"
    tf_train = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.485, 0.456, 0.406]),
    ])
    tf_eval = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.485, 0.456, 0.406]),
    ])
    train_ds = datasets.ImageFolder(train_dir, transform=tf_train)
    val_ds = datasets.ImageFolder(val_dir, transform=tf_eval)
    test_ds = datasets.ImageFolder(test_dir, transform=tf_eval)
    train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch, shuffle=False, num_workers=0)

    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, len(train_ds.classes))
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()

    best_val_acc = -1.0
    best_state: dict | None = None
    for _ in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            crit(model(x), y).backward()
            opt.step()
        val_preds, val_labels = _eval_loader(model, val_loader, device)
        val_acc = float((np.array(val_preds) == np.array(val_labels)).mean())
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
        if ckpt_path is not None:
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {"model": best_state, "classes": train_ds.classes},
                ckpt_path,
            )

    val_preds, val_labels = _eval_loader(model, val_loader, device)
    test_preds, test_labels = _eval_loader(model, test_loader, device)
    val_m = _metrics_from_preds(val_labels, val_preds, val_ds.classes)
    test_m = _metrics_from_preds(test_labels, test_preds, test_ds.classes)
    return {
        "seed": seed,
        "epochs": epochs,
        "lr": lr,
        "batch": batch,
        "val_acc": val_m["acc"],
        "f1_macro": test_m["f1_macro"],
        "f1_weighted": test_m["f1_weighted"],
        "test_acc": test_m["acc"],
        "test_f1_macro": test_m["f1_macro"],
        "test_f1_weighted": test_m["f1_weighted"],
        "val_f1_macro": val_m["f1_macro"],
        "val_f1_weighted": val_m["f1_weighted"],
        "per_class": test_m["per_class"],
        "confusion_matrix": test_m["confusion_matrix"],
        "class_names": test_m["class_names"],
    }


def _majority_baseline(cfg, split: str = "test") -> dict:
    data_dir = cfg.STAGE2_ROOT / split
    counts: dict[str, int] = {}
    labels = []
    classes = sorted([d.name for d in data_dir.iterdir() if d.is_dir()])
    cls_to_idx = {c: i for i, c in enumerate(classes)}
    for cls in classes:
        n = len(list((data_dir / cls).glob("*.jpg"))) + len(list((data_dir / cls).glob("*.png")))
        counts[cls] = n
    for cls in classes:
        labels.extend([cls_to_idx[cls]] * counts[cls])
    maj = max(counts, key=counts.get)
    maj_idx = cls_to_idx[maj]
    preds = [maj_idx] * len(labels)
    return {
        "name": "majority_class",
        "predicted_class": maj,
        "test_acc": float((np.array(preds) == np.array(labels)).mean()),
        "f1_macro": float(f1_score(labels, preds, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(labels, preds, average="weighted", zero_division=0)),
    }


def _bootstrap_ci(values: list[float], n_boot: int = 5000, alpha: float = 0.05) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    if len(arr) < 2:
        m = float(arr.mean()) if len(arr) else 0.0
        return m, m
    rng = np.random.default_rng(42)
    boots = [float(rng.choice(arr, size=len(arr), replace=True).mean()) for _ in range(n_boot)]
    lo = float(np.percentile(boots, 100 * alpha / 2))
    hi = float(np.percentile(boots, 100 * (1 - alpha / 2)))
    return lo, hi


def _plot_confusion(cm: list[list[int]], classes: list[str], path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 6))
    cm_arr = np.array(cm)
    im = ax.imshow(cm_arr, cmap="Blues")
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    short = [c.replace("_", "\n") for c in classes]
    ax.set_xticklabels(short, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(short, fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground truth")
    ax.set_title("Stage-2 ResNet-18 Confusion Matrix (best seed, test split)")
    for i in range(cm_arr.shape[0]):
        for j in range(cm_arr.shape[1]):
            ax.text(j, i, str(cm_arr[i, j]), ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def _plot_baselines(rows: list[dict], path: Path) -> None:
    import matplotlib.pyplot as plt

    names = [r["name"] for r in rows]
    f1 = [r["f1_macro"] for r in rows]
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.bar(names, f1, color=["#4477AA", "#EE6677", "#228833", "#CCBB44"][: len(names)])
    ax.set_ylabel("Macro F1")
    ax.set_ylim(0, max(f1) * 1.25 + 0.05)
    ax.set_title("Baseline Comparison (test ROI crops)")
    for i, v in enumerate(f1):
        ax.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def main() -> None:
    t0 = time.time()
    cfg = _load_cfg()
    seeds = [0, 1, 2, 3, 4]
    stage1 = _stage1_metrics(cfg)
    dataset = _dataset_stats(cfg)

    seed_runs = []
    for s in seeds:
        print(f"Training seed {s}...", flush=True)
        seed_runs.append(_train_eval_resnet(cfg, seed=s, epochs=5, lr=1e-3, batch=16))

    best = max(seed_runs, key=lambda r: r["test_f1_macro"])
    long_run = _train_eval_resnet(cfg, seed=0, epochs=15, lr=1e-3, batch=16)
    majority = _majority_baseline(cfg)

    macro_vals = [r["test_f1_macro"] for r in seed_runs]
    acc_vals = [r["test_acc"] for r in seed_runs]
    w_vals = [r["test_f1_weighted"] for r in seed_runs]
    macro_ci = _bootstrap_ci(macro_vals)

    baselines = [
        {"name": "Majority class", "f1_macro": majority["f1_macro"], "test_acc": majority["test_acc"]},
        {
            "name": "ResNet-18 (5 ep, mean)",
            "f1_macro": float(np.mean(macro_vals)),
            "test_acc": float(np.mean(acc_vals)),
        },
        {"name": "ResNet-18 (15 ep)", "f1_macro": long_run["test_f1_macro"], "test_acc": long_run["test_acc"]},
        {"name": "ResNet-18 (best seed)", "f1_macro": best["test_f1_macro"], "test_acc": best["test_acc"]},
    ]

    _plot_confusion(best["confusion_matrix"], best["class_names"], CHARTS / "fig_confusion_matrix.png")
    _plot_baselines(baselines, CHARTS / "fig_baseline_comparison.png")

    out = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_sec": round(time.time() - t0, 1),
        "hardware_note": "CPU-only; Windows; PyTorch ResNet-18 + Ultralytics YOLOv8n stage-1 log",
        "dataset": dataset,
        "stage1": stage1,
        "stage2_seeds": seed_runs,
        "stage2_summary": {
            "n_seeds": len(seed_runs),
            "eval_split": "test",
            "val_acc_mean": float(np.mean([r["val_acc"] for r in seed_runs])),
            "val_acc_std": float(np.std([r["val_acc"] for r in seed_runs], ddof=1)),
            "test_acc_mean": float(np.mean(acc_vals)),
            "test_acc_std": float(np.std(acc_vals, ddof=1)),
            "f1_macro_mean": float(np.mean(macro_vals)),
            "f1_macro_std": float(np.std(macro_vals, ddof=1)),
            "f1_weighted_mean": float(np.mean(w_vals)),
            "f1_weighted_std": float(np.std(w_vals, ddof=1)),
            "f1_macro_bootstrap_ci95": list(macro_ci),
        },
        "baselines": baselines,
        "long_training": long_run,
        "majority_class": majority,
        "best_seed": best["seed"],
        "per_class_best_seed": best["per_class"],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
