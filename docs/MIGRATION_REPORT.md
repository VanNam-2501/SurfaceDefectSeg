# Migration report

Date: 2026-08-19

## Outcome

The exploratory workspace was converted into one maintained repository with
separate source, applications, data, artifacts, notebooks, documentation,
runtime binaries, submission evidence and recoverable archive areas. All active
launchers use repository-relative paths; no active default depends on drive `E:`.

## Data-safety checks

- The interrupted VMamba vendor move left 1,764 files in the legacy location.
  They were merged into `src/threecad_segmentation/third_party/VMamba` and every
  file was verified using SHA-256: `MISSING=0`, `HASH_DIFF=0`.
- The legacy fragment was copied to
  `archive/migration_legacy/aluminum_surface_defect_segmentation_bundle_partial`
  and verified again before the duplicate active source was removed.
- The actual dataset was identified before cleanup (18,342 files,
  810,943,017 bytes) and moved to `data/3cad_ani`.
- The report directory was copied to `artifacts/reports/final`; all 5,027 files
  were verified using SHA-256 before the source copy was removed.
- Nested Git metadata from VMamba was moved to `archive`; the vendor source and
  license remain active.

## Verification after migration

- PowerShell parser: 0 errors.
- Python `compileall`: PASS.
- Notebook JSON: 0 errors.
- Dataset protocol: PASS; Train 5,733, Validation 718, Test 717.
- ML unit tests: 10/10 PASS.
- Dataset Review Studio tests: 5/5 PASS.
- Web production build and rendered/client tests: 2/2 PASS.
- FastAPI configuration: all three checkpoints and the frozen spatial rule-based policy resolve successfully.
- Learned Verifier and Learned Hybrid are preserved under archive/learned_verifier and are no longer active.
- Git baseline: `d91f6fd` on branch `main`.

Run the same maintained check with:

```powershell
.\verify.ps1 -IncludeWeb
```

## Recoverability

No user dataset, checkpoint, final result or Kaggle package was discarded.
Superseded material remains under `archive/`, which is ignored by Git. Do not
delete that directory until the new source release has been tested on a clean
machine and the supervisor has accepted the migrated structure.
