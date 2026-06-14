# WeldCascade CPU Reproduction Checklist

## Environment

- Python 3.11+, PyTorch (CPU), Ultralytics YOLOv8, torchvision, scikit-learn
- Windows 10 or Linux; fixed random seed 42 in `config.py`

## Data

- **Steel Pipe:** https://github.com/huangyebiaoke/steel-pipe-weld-defect-detection  
  Release v1.0 → `steel-tube-dataset-all.zip`

## Expected key metrics (archived P0 run, test split)

| Metric | Value |
|--------|-------|
| Stage-1 mAP@0.5 | 0.918 |
| Stage-1 recall | 0.855 |
| Oracle ROI macro F1 | 0.941 ± 0.022 |
| Detector-routed macro F1 | 0.695 ± 0.073 |
| Single-stage YOLO mAP@0.5 | 0.975 |
| Cascade latency (mean) | 129 ms |

## Script order

```bash
python scripts/prepare_steel_pipe.py
python scripts/prepare_stage1.py
python scripts/prepare_stage2_roi.py
python scripts/run_weld_revision_benchmark.py
python scripts/run_weld_p0_fixes.py
python scripts/run_weld_routed_viz.py
python scripts/run_weld_sensors_extras.py
```

Compare outputs under `results/` with the archived JSON files in this repository.
