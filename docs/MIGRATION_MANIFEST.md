# Migration manifest

Date: 2026-08-19

This manifest records the one-time migration from the exploratory thesis workspace
to the maintained project layout. Historical result files are preserved as
artifacts; generated paths inside those historical files are not rewritten.

| Legacy path | Maintained path |
| --- | --- |
| `Aluminum_Surface_Defect_Segmentation_Bundle/Aluminum_Surface_Defect_Segmentation` | `src/threecad_segmentation` |
| `Aluminum_New_Ipad` / embedded dataset | `data/3cad_ani` |
| `dataset_review_tool` | `apps/dataset_review` |
| `web_demo` | `apps/web_demo` |
| `kaggle_upload/all3_eval_weights_20260818` | `artifacts/checkpoints/final` |
| `kaggle_upload/mamba-wheel` | `runtime/wheels/vmamba` |
| `kaggle_upload/*.zip` | `artifacts/packages/kaggle` |
| `decision_workspace` | `artifacts/experiments/decision` |
| `adaptive_component_workspace` | `artifacts/experiments/adaptive_component` |
| learned_verifier_workspace | archive/learned_verifier/artifacts/experiments/learned_verifier |
| `final_thesis_deliverables` | `artifacts/reports/final` |
| `cleanup_quarantine` | `archive/cleanup_quarantine` |
| root experiment scripts | `scripts/experiments` |
| root reporting scripts | `scripts/reporting` |
| root notebooks | `experiments/notebooks` |

An interrupted first move split the bundled VMamba repository between the legacy
and maintained paths. Before continuing, all 1,764 remaining files were copied to
the maintained path and verified with SHA-256 (`HASH_DIFF=0`). The legacy fragment
is retained under `archive/migration_legacy` until final acceptance.
