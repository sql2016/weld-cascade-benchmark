#!/usr/bin/env python3
"""GDXray external validation and CPU inference latency for WeldCascade / Sensors."""

from __future__ import annotations

import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch
from PIL import Image
from torchvision import models, transforms

ROOT = Path(__file__).resolve().parents[1]
PLAN_B = ROOT / "scripts" / "weld_plan_b"
OUT = ROOT / "artifacts/rc-weld-two-stage/deliverables/results/sensors_extras.json"
BENCH = ROOT / "artifacts/rc-weld-two-stage/deliverables/results/revision_benchmark.json"
PRIORITY = ROOT / "artifacts/rc-weld-two-stage/deliverables/results/priority_experiments.json"


def _load_cfg():
    import importlib.util

    spec = importlib.util.spec_from_file_location("weld_cfg", PLAN_B / "config.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _ensure_gdxray(cfg) -> bool:
    if cfg.RAW_GDXRAY.exists() and any(cfg.RAW_GDXRAY.rglob("*")):
        return True
    print("Downloading GDXray Welds...", flush=True)
    subprocess.run(
        [sys.executable, str(PLAN_B / "download_data.py"), "--gdxray"],
        cwd=str(PLAN_B),
        check=True,
    )
    return cfg.RAW_GDXRAY.exists()


def _gdxray_eval(cfg, yolo_weights: Path) -> dict:
    if not _ensure_gdxray(cfg):
        return {"available": False, "note": f"GDXray not found at {cfg.RAW_GDXRAY}"}
    if not (cfg.GDXRAY_EVAL / "data.yaml").exists():
        subprocess.run([sys.executable, str(PLAN_B / "prepare_gdxray.py")], cwd=str(PLAN_B), check=True)
    from ultralytics import YOLO

    model = YOLO(str(yolo_weights))
    t0 = time.time()
    metrics = model.val(data=str(cfg.GDXRAY_EVAL / "data.yaml"), workers=0, verbose=False)
    elapsed = round(time.time() - t0, 1)
    n_labeled = 0
    lbl_dir = cfg.GDXRAY_EVAL / "labels"
    for lbl in lbl_dir.glob("*.txt"):
        if lbl.read_text(encoding="utf-8").strip():
            n_labeled += 1
    return {
        "available": True,
        "dataset": "GDXray Welds",
        "n_images": len(list((cfg.GDXRAY_EVAL / "images").glob("*"))),
        "n_images_with_boxes": n_labeled,
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "eval_elapsed_sec": elapsed,
        "weights": str(yolo_weights),
        "note": "Zero-shot transfer: Steel Pipe-trained YOLOv8n, binary defect detection on GDXray Welds.",
    }


def _percentile(vals: list[float], p: float) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    k = (len(s) - 1) * p / 100.0
    f = int(k)
    c = min(f + 1, len(s) - 1)
    if f == c:
        return s[f]
    return s[f] + (s[c] - s[f]) * (k - f)


def _latency_benchmark(cfg, yolo_weights: Path, cls_ckpt: Path, n_images: int = 30, warmup: int = 3) -> dict:
    if str(PLAN_B) not in sys.path:
        sys.path.insert(0, str(PLAN_B))
    from ultralytics import YOLO
    from io_utils import find_images

    device = "cpu"
    det = YOLO(str(yolo_weights))
    ckpt = torch.load(cls_ckpt, map_location=device, weights_only=False)
    classes = ckpt["classes"]
    clf = models.resnet18(weights=None)
    clf.fc = torch.nn.Linear(clf.fc.in_features, len(classes))
    clf.load_state_dict(ckpt["model"])
    clf.eval().to(device)
    tf = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.485, 0.456, 0.406]),
    ])

    imgs = find_images(cfg.STAGE1_ROOT / "images" / "test")[:n_images]
    if len(imgs) < 5:
        imgs = find_images(cfg.STAGE1_ROOT / "images" / "val")[:n_images]
    if not imgs:
        raise FileNotFoundError("No stage-1 images for latency benchmark")

    s1_ms: list[float] = []
    s2_ms: list[float] = []
    e2e_ms: list[float] = []
    roi_counts: list[int] = []

    for img_path in imgs:
        for _ in range(warmup):
            det.predict(str(img_path), conf=0.25, verbose=False)
        t0 = time.perf_counter()
        res = det.predict(str(img_path), conf=0.25, verbose=False)[0]
        s1_ms.append((time.perf_counter() - t0) * 1000.0)

        n_roi = max(1, len(res.boxes) if res.boxes is not None else 0)
        roi_counts.append(n_roi)
        img = Image.open(img_path).convert("L")
        iw, ih = img.size
        boxes = []
        if res.boxes is not None and len(res.boxes):
            for box in res.boxes:
                boxes.append(tuple(map(int, box.xyxy[0].tolist())))

        per_roi: list[float] = []
        for x1, y1, x2, y2 in boxes[:10] or [(0, 0, iw, ih)]:
            crop = img.crop((x1, y1, x2, y2))
            tensor = tf(crop).unsqueeze(0).to(device)
            for _ in range(warmup):
                with torch.no_grad():
                    clf(tensor)
            t1 = time.perf_counter()
            with torch.no_grad():
                clf(tensor)
            per_roi.append((time.perf_counter() - t1) * 1000.0)
        s2_ms.append(statistics.mean(per_roi) if per_roi else 0.0)

        t2 = time.perf_counter()
        res2 = det.predict(str(img_path), conf=0.25, verbose=False)[0]
        if res2.boxes is not None and len(res2.boxes):
            for box in res2.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                tensor = tf(img.crop((x1, y1, x2, y2))).unsqueeze(0).to(device)
                with torch.no_grad():
                    clf(tensor)
        else:
            tensor = tf(img).unsqueeze(0).to(device)
            with torch.no_grad():
                clf(tensor)
        e2e_ms.append((time.perf_counter() - t2) * 1000.0)

    def _summ(vals: list[float]) -> dict:
        return {
            "mean_ms": round(statistics.mean(vals), 2),
            "std_ms": round(statistics.stdev(vals), 2) if len(vals) > 1 else 0.0,
            "p50_ms": round(_percentile(vals, 50), 2),
            "p95_ms": round(_percentile(vals, 95), 2),
        }

    mean_roi = statistics.mean(roi_counts) if roi_counts else 1.0
    s2_per_image = [a * b for a, b in zip(s2_ms, [max(1, c) for c in roi_counts])]
    return {
        "hardware": "CPU-only; Windows; PyTorch + Ultralytics YOLOv8n",
        "n_images": len(imgs),
        "warmup_runs": warmup,
        "mean_detections_per_image": round(mean_roi, 2),
        "stage1_yolo": _summ(s1_ms),
        "stage2_resnet18_per_roi": _summ(s2_ms),
        "stage2_resnet18_per_image_est": _summ(s2_per_image),
        "cascade_end_to_end": _summ(e2e_ms),
        "throughput_fps_est": round(1000.0 / statistics.mean(e2e_ms), 2) if e2e_ms else 0.0,
    }


def _merge(out: dict) -> None:
    for path in (BENCH, PRIORITY):
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        pe = data.setdefault("priority_experiments", {})
        if out.get("gdxray_external"):
            pe["gdxray_external"] = out["gdxray_external"]
        if out.get("latency"):
            pe["latency"] = out["latency"]
        data["sensors_extras"] = out
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-gdxray", action="store_true")
    parser.add_argument("--latency-only", action="store_true")
    args = parser.parse_args()

    cfg = _load_cfg()
    weights = Path(
        json.loads(PRIORITY.read_text(encoding="utf-8"))["stage1_retrain"]["weights"]
        if PRIORITY.exists()
        else ""
    )
    if not weights.exists():
        runs = sorted(
            (cfg.STAGE1_ROOT.parent / "runs").glob("stage1_defect*/weights/best.pt"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        weights = runs[0] if runs else None
    if not weights or not weights.exists():
        raise FileNotFoundError("Stage-1 YOLO weights not found")

    cls_ckpt = cfg.STAGE2_ROOT.parent / "runs" / "stage2_cls" / "best.pt"
    if not cls_ckpt.exists():
        subprocess.run(
            [sys.executable, str(PLAN_B / "train_stage2.py"), "--epochs", "5", "--device", "cpu", "--batch", "16"],
            cwd=str(PLAN_B),
            check=True,
        )

    out: dict = {"generated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    if not args.skip_gdxray and not args.latency_only:
        print("GDXray external validation...", flush=True)
        out["gdxray_external"] = _gdxray_eval(cfg, weights)
        print(json.dumps(out["gdxray_external"], indent=2), flush=True)
    elif args.skip_gdxray:
        out["gdxray_external"] = {
            "available": False,
            "note": "Skipped or manual download required; see SENSORS_SUBMISSION.md",
        }

    print("CPU latency benchmark...", flush=True)
    out["latency"] = _latency_benchmark(cfg, weights, cls_ckpt)
    print(json.dumps(out["latency"], indent=2), flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    _merge(out)
    print(f"Wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
