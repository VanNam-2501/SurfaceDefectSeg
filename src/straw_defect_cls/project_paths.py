"""Canonical project paths and default configuration for Straw Defect Classification."""

from pathlib import Path

SOURCE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = SOURCE_ROOT.parent.parent
DATASET_ROOT = REPO_ROOT / "data" / "juice_box_straw_defects"
ARTIFACTS_ROOT = REPO_ROOT / "artifacts"
CHECKPOINT_ROOT = ARTIFACTS_ROOT / "checkpoints"

# Ensure directories exist
ARTIFACTS_ROOT.mkdir(parents=True, exist_ok=True)
CHECKPOINT_ROOT.mkdir(parents=True, exist_ok=True)
