"""Canonical local paths for the maintained repository layout."""

from pathlib import Path

SOURCE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = SOURCE_ROOT.parents[1]
DATASET_ROOT = REPO_ROOT / "data" / "3cad_ani"
ARTIFACTS_ROOT = REPO_ROOT / "artifacts"
CHECKPOINT_ROOT = ARTIFACTS_ROOT / "checkpoints" / "final"
DECISION_EXPERIMENT_ROOT = ARTIFACTS_ROOT / "experiments" / "decision"
FINAL_REPORT_ROOT = ARTIFACTS_ROOT / "reports" / "final"

