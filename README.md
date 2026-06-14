# WeldCascade Benchmark (Sensors)

Reproducible benchmark artefacts for **WeldCascade**: a two-stage YOLOv8n + ResNet-18 cascade for steel-pipe weld radiography (MDPI *Sensors*).

## Release

| Item | URL |
|------|-----|
| GitHub release | [v1.0-sensors](https://github.com/sql2016/weld-cascade-benchmark/releases/tag/v1.0-sensors) |
| Version | `1.0.0-sensors` |
- **Zenodo DOI:** upload `weld-cascade-benchmark-v1.0-sensors.zip` to [Zenodo](https://zenodo.org) and add the DOI to `repository_config.json`.

## Contents

| Path | Description |
|------|-------------|
| `results/revision_benchmark.json` | Oracle stage-2 metrics, seeds, baselines, splits |
| `results/priority_experiments.json` | Stage-1 test metrics, 5-seed detector-routed eval, YOLO multiclass baseline |
| `results/sensors_extras.json` | CPU latency benchmark |
| `results/routed_viz.json` | Routed confusion matrix + failure-case metadata |
| `scripts/` | CPU reproduction scripts (Plan B pipeline) |
| `docs/REPRODUCTION.md` | Step-by-step checklist |
| `docs/GDXRAY_ACCESS_STATEMENT.md` | GDXray Welds access log (cross-dataset limitation) |
| `charts/` | Key manuscript figures |

## Key archived metrics (test split)

| Metric | Value |
|--------|-------|
| Stage-1 mAP@0.5 | 0.918 |
| Stage-1 recall | 0.855 |
| Oracle ROI macro F1 | 0.941 ± 0.022 (5 seeds) |
| Detector-routed macro F1 | 0.695 ± 0.073 (5 seeds) |
| Single-stage YOLO mAP@0.5 | 0.975 |
| Cascade latency (mean) | 129 ms (~7.75 FPS) |

## Quick start

1. Download **Steel Pipe** dataset: https://github.com/huangyebiaoke/steel-pipe-weld-defect-detection (`steel-tube-dataset-all.zip`).
2. `pip install -r requirements.txt` (Python 3.11+, PyTorch CPU, ultralytics, scikit-learn).
3. See `docs/REPRODUCTION.md` for script order.

## Citation

If you use this benchmark, cite the WeldCascade *Sensors* manuscript and Huang et al. (Steel Pipe dataset).

```bibtex
@misc{weld_cascade_benchmark_2026,
  title = {WeldCascade reproducible benchmark artefacts (Sensors)},
  author = {Zhu, Yangpeng},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/sql2016/weld-cascade-benchmark/releases/tag/v1.0-sensors}
}
```

## License

MIT License for code and JSON artefacts in this repository. The Steel Pipe radiograph dataset remains under its original terms (Huang et al., GitHub).

## Contact

Yangpeng Zhu — zyp@xsyu.edu.cn — ORCID [0009-0005-9991-4164](https://orcid.org/0009-0005-9991-4164)
