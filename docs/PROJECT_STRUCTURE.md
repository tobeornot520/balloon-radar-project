# Project structure

## Active source

| Path | Responsibility |
|---|---|
| `datasets/` | Manifest-driven detection, polarimetric, and ROI datasets |
| `features/` | RD, gated polarimetric, and ROI feature construction |
| `models/` | FCN, DPG, background calibrator, and ROI refiner models |
| `training/` | Formal model training implementations |
| `scripts/` | Experiment orchestration, audits, summaries, and packaging |
| `evaluation/` | CFAR evaluation, metric analysis, and reports |
| `baselines/` | Classical detection baselines |
| `configs/` | Versioned experiment configuration |
| `tests/` | Automated tests collected by pytest |

## Evidence and documentation

| Path | Responsibility |
|---|---|
| `results/data_audit/` | Manifests, alignment audits, and compact result tables |
| `results/final_evidence/` | Frozen paper evidence and precomputed assets |
| `docs/` | Stable conclusions, preregistration, and operating instructions |

Large checkpoints, raw radar data, experiment runs, and generated distributions
are excluded from Git.

## Supporting and historical material

| Path | Responsibility |
|---|---|
| `tools/` | Diagnostics, maintenance, and data-layout utilities |
| `notes/development_history/` | Local raw transcripts and manuscript drafts; ignored by Git |
| `_cleanup_archive/` | Local recovery material; new content ignored by Git |
| `payload/` | Previously committed delivery snapshot; not an active entry point |

The repository has no package-level `main.py`. Use the versioned entry points
listed in the root README.
