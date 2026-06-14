"""Paths and class taxonomy for Plan B (binary detect + ROI classify)."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT / "data" / "weld_plan_b"

RAW = DATA_ROOT / "raw"
PREPARED = DATA_ROOT / "prepared"

RAW_STEEL = RAW / "steel_pipe"
RAW_SWRD = RAW / "swrd"
RAW_GDXRAY = RAW / "gdxray" / "Welds"

STAGE1_ROOT = PREPARED / "stage1_binary"
STAGE2_ROOT = PREPARED / "stage2_roi"
GDXRAY_EVAL = PREPARED / "gdxray_eval"

# Steel Pipe (huangyebiaoke) — label indices 0..7
STEEL_PIPE_NAMES = [
    "air_hole",
    "bite_edge",
    "broken_arc",
    "crack",
    "hollow_bead",
    "overlap",
    "slag_inclusion",
    "unfused",
]

# Unified taxonomy for stage-2 (merge near-duplicates across sources)
UNIFIED_CLASSES = [
    "porosity",       # air_hole, hollow_bead, porosity
    "crack",
    "slag_inclusion",
    "unfused",
    "bite_edge",
    "broken_arc",
    "overlap",
    "inclusion_other",  # SWRD inclusions / misc
]

STEEL_TO_UNIFIED = {
    0: "porosity",
    1: "bite_edge",
    2: "broken_arc",
    3: "crack",
    4: "porosity",
    5: "overlap",
    6: "slag_inclusion",
    7: "unfused",
}

# SWRD class names as in paper (adjust indices after inspecting data.yaml)
SWRD_NAME_TO_UNIFIED = {
    "porosity": "porosity",
    "pore": "porosity",
    "crack": "crack",
    "cracks": "crack",
    "inclusion": "slag_inclusion",
    "inclusions": "slag_inclusion",
    "slag": "slag_inclusion",
    "slag_inclusion": "slag_inclusion",
    "lack_of_fusion": "unfused",
    "unfused": "unfused",
    "lack_of_penetration": "unfused",
    "undercut": "bite_edge",
    "bite_edge": "bite_edge",
}

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1
RANDOM_SEED = 42
ROI_PAD = 0.15
MIN_ROI_PX = 32

STEEL_PIPE_ZIP_URL = (
    "https://github.com/huangyebiaoke/steel-pipe-weld-defect-detection/"
    "releases/download/1.0/steel-tube-dataset-all.zip"
)
GDXRAY_WELDS_URL = (
    "https://www.dropbox.com/scl/fi/im896nbhllnbnol585fsq/Welds.zip"
    "?rlkey=u584im2jtrdxzmhrg2lcqtavv&dl=1"
)
