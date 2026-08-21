# Current project audit

Date: 2026-08-21. This document supersedes the path layout in
`PROJECT_AUDIT.md`, which is retained as the pre-migration snapshot.

## Current inventory

| Area | Files | Bytes | Role |
|---|---:|---:|---|
| `apps/` | 21,227 | 420,200,450 | Review app and web app; count includes ignored `node_modules`/build output |
| `src/` | 38 | 579,091 | Core ML and minimal vendored VMamba runtime |
| `scripts/` | 37 | 416,263 | Train/evaluate/experiment/report/setup/verify entry points |
| `tests/` | 10 | 21,500 | ML tests and test bootstrap/cache |
| `docs/` | 10 | 41,305 | Maintained guides, cleanup and migration records |
| `experiments/` | 2 | 875,542 | Dataset-audit and Kaggle notebooks |
| `submission/` | 20 | 55,517 | Submission and AI-compliance records |
| `data/` | 18,343 | 810,943,301 | 3CAD-ANI data plus local README |
| `artifacts/` | 7,962 | 1,948,545,153 | Checkpoints, caches, reports and Kaggle package |
| `runtime/` | 3 | 50,102,486 | VMamba wheel and runtime README |
| `archive/` | 15,961 | 8,435,510,784 | Recoverable cleanup/migration material, including review candidates |

## Source-control boundary

Git tracks source, tests, scripts, documentation, notebooks and submission
records. It ignores the dataset, generated artifacts, runtime wheel, archive,
virtual environment, frontend dependencies/builds and review state/exports.
The repository uses one root `.git` on branch `main`; nested Git metadata was
removed from active app/vendor paths.

The 2026-08-21 cleanup removed Colab-only files, retained Kaggle workflows and
moved 1,786 unused or out-of-scope files into organized review groups under
`archive/review_candidates/2026-08-21/`. See
`docs/cleanup/unused_candidates_2026-08-21.md` for the rationale and restore
instructions.

## Requirements still missing

The code workspace is organized and verified, but the following products still
require the student's real content or external confirmation:

1. Report in the faculty's official DOCX/PDF template.
2. Defense slide deck and PDF export.
3. Backup demo video.
4. Complete internship and development logs with real durations.
5. Exact Codex model/build and raw task exports for the AI archive.
6. Company/internship-unit signed confirmation.
7. Clean-machine source-release installation test.
8. Explicit dataset redistribution/license evidence if data will be shared.
