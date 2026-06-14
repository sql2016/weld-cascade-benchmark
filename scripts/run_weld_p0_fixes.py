#!/usr/bin/env python3
"""P0 fixes: YOLO test-split metrics, 5-seed detector-routed eval, JSON + paper refresh."""

from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PLAN_B = ROOT / "scripts" / "weld_plan_b"
if str(PLAN_B) not in sys.path:
    sys.path.insert(0, str(PLAN_B))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

OUT = ROOT / "artifacts/rc-weld-two-stage/deliverables/results/priority_experiments.json"
BENCH = ROOT / "artifacts/rc-weld-two-stage/deliverables/results/revision_benchmark.json"
CHARTS = ROOT / "artifacts/rc-weld-two-stage/deliverables/charts"
CKPT_DIR = ROOT / "data/weld_plan_b/prepared/runs/routed_eval_ckpts"


def _load_cfg():
    spec = importlib.util.spec_from_file_location("weld_cfg", PLAN_B / "config.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _yolo_test_metrics(weights: Path, yaml_path: Path) -> dict:
    from ultralytics import YOLO

    model = YOLO(str(weights))
    metrics = model.val(data=str(yaml_path), split="test", workers=0)
    return {
        "eval_split": "test",
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "weights": str(weights),
    }


def _aggregate_routed(seed_runs: list[dict]) -> dict:
    if not seed_runs:
        return {}
    keys = [
        "routed_test_f1_macro",
        "routed_test_f1_weighted",
        "routed_test_acc",
        "detection_recall_on_gt",
    ]
    out = dict(seed_runs[0])
    out["n_seeds"] = len(seed_runs)
    out["seeds"] = seed_runs
    for k in keys:
        vals = [float(r[k]) for r in seed_runs]
        out[k] = float(np.mean(vals))
        out[f"{k}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
    n_correct = [int(r.get("n_matched_correct", 0)) for r in seed_runs]
    n_miscls = [int(r.get("n_matched_misclassified", 0)) for r in seed_runs]
    out["n_matched_correct"] = int(round(np.mean(n_correct)))
    out["n_matched_misclassified"] = int(round(np.mean(n_miscls)))
    return out


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


def main() -> None:
    from run_weld_priority_experiments import _find_yolo_weights, _merge_benchmark, _routed_eval
    from run_weld_revision_benchmark import _train_eval_resnet

    t0 = time.time()
    cfg = _load_cfg()
    bench = json.loads(BENCH.read_text(encoding="utf-8")) if BENCH.exists() else {}
    prio = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}

    weights = _find_yolo_weights(cfg)
    if not weights or not weights.exists():
        raise FileNotFoundError("Stage-1 YOLO weights not found")

    print("P0-1: YOLO metrics on held-out TEST split...", flush=True)
    s1_yaml = cfg.STAGE1_ROOT / "data.yaml"
    mc_yaml = cfg.PREPARED / "stage1_multiclass" / "data.yaml"
    s1_test = _yolo_test_metrics(weights, s1_yaml)
    prio["stage1_retrain"] = {
        **prio.get("stage1_retrain", {}),
        **s1_test,
        "epochs": bench.get("stage1", {}).get("epochs", 10),
        "note": "Stage-1 retrain; metrics on test split",
    }
    bench["stage1"] = {**bench.get("stage1", {}), **s1_test, "note": "Retrained stage-1; test split metrics"}

    mc_weights = cfg.STAGE1_ROOT.parent / "runs" / "stage1_multiclass" / "weights" / "best.pt"
    if mc_weights.exists() and mc_yaml.exists():
        prio["yolo_multiclass_baseline"] = {
            **prio.get("yolo_multiclass_baseline", {}),
            **_yolo_test_metrics(mc_weights, mc_yaml),
            "epochs": prio.get("yolo_multiclass_baseline", {}).get("epochs", 10),
        }
    else:
        print("  (skip multiclass test eval — weights or yaml missing)", flush=True)

    print("P0-2: Detector-routed eval over 5 ResNet-18 seeds...", flush=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    seed_runs: list[dict] = []
    for seed in [0, 1, 2, 3, 4]:
        ckpt = CKPT_DIR / f"seed_{seed}.pt"
        print(f"  Training ResNet-18 seed {seed} (5 ep)...", flush=True)
        _train_eval_resnet(cfg, seed=seed, epochs=5, lr=1e-3, batch=16, device="cpu", ckpt_path=ckpt)
        print(f"  Routed eval seed {seed}...", flush=True)
        r = _routed_eval(cfg, weights, ckpt, split="test")
        r["seed"] = seed
        seed_runs.append(r)
        print(
            f"    seed {seed}: routed F1={r['routed_test_f1_macro']:.4f}, "
            f"acc={r['routed_test_acc']:.4f}",
            flush=True,
        )

    prio["routed_eval"] = _aggregate_routed(seed_runs)
    oracle_f1 = float(bench["stage2_summary"]["f1_macro_mean"])
    _plot_oracle_vs_routed(
        oracle_f1,
        prio["routed_eval"]["routed_test_f1_macro"],
        prio["routed_eval"]["detection_recall_on_gt"],
    )

    prio["generated"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    prio["p0_fixes_elapsed_sec"] = round(time.time() - t0, 1)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(prio, indent=2), encoding="utf-8")
    _merge_benchmark(prio)
    print(f"Wrote {OUT}", flush=True)
    r = prio["routed_eval"]
    print(
        f"Summary: stage-1 test mAP@0.5={s1_test['map50']:.4f}; "
        f"routed F1={r['routed_test_f1_macro']:.4f}±{r['routed_test_f1_macro_std']:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
