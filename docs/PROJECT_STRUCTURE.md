# Project structure

## Active source

| Path | Responsibility |
|---|---|
| `datasets/` | Manifest-driven detection, polarimetric, and ROI datasets |
| `features/` | RD, gated polarimetric, and ROI feature construction |
| `models/` | FCN, DPG, background calibrator, and ROI refiner models |
| `training/` | Reusable formal training implementations; no duplicate CLI copies |
| `scripts/` | User-invoked orchestration, audits, summaries, plots, and packaging |
| `evaluation/` | Importable metrics, postprocessing, analysis, and report logic |
| `baselines/` | Classical detection baselines |
| `configs/` | Versioned experiment configuration |
| `tests/` | Automated pytest tests only; standalone audits belong in `scripts/` |
| `utils/` | Shared provenance and small infrastructure helpers |

## Evidence and documentation

| Path | Responsibility |
|---|---|
| `results/data_audit/` | Manifests, alignment audits, and compact result tables |
| `results/final_evidence/` | Frozen paper evidence and precomputed assets |
| `data/metadata/` | Versioned manifests and external dataset/artifact provenance registries; no raw signals |
| `data/raw/external/` | Local official downloads and per-file integrity manifests; ignored by Git and excluded from sharing |
| `docs/` | Stable conclusions, preregistration, and operating instructions |

Directory-level indexes are maintained in `scripts/README.md`, `configs/README.md`,
`results/README.md`, and `docs/INDEX.md`.

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

## Root boundary

The project PPT, application PDF, paper PDF, `方向.md`, and `先改.md` are user-owned
research inputs. They remain at the root until the user explicitly approves a
reference-material migration. They are never treated as source entry points.

Generated checkpoints and raw data stay local. Empty `losses/`, `metrics/`,
`postprocess/`, `radar_processing/`, `checkpoints/`, `logs/`, and `notebooks/`
directories are reserved local placeholders, not active modules.

## Enforcement

`scripts/check_project_health.py` checks:

1. syntax of active Python files;
2. required workflow files;
3. absence of exact duplicate Python implementations in active directories;
4. absence of root-level Python entry points;
5. optional availability of frozen six-fold joint inputs.
