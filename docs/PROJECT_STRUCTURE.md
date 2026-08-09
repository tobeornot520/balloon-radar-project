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

`dist/` is the local delivery area. It may contain the single current onboarding
package and the independent algorithm-code review package; generated ZIPs and
staging directories are ignored by Git. Delivery packages must be built from a
clean commit and audited before old package artifacts are removed.

## Four logical zones

The root-level layout is organized into four research zones, with a prominent
control panel outside the code modules:

| Zone | Path | Responsibility |
|---|---|---|
| Project control | `PROJECT_CONTROL/` | Long-term handbook, task board, roadmap, log, source inputs and team review material |
| Data | `data/` | Raw/processed signals, manifests, splits and provenance metadata |
| Code | `datasets/`, `features/`, `models/`, `training/`, `evaluation/`, `scripts/`, `tests/`, `utils/`, `baselines/`, `tools/` | Active implementation and executable entry points; `code/README.md` is the map |
| Results | `results/` | Audits, experiment runs, figures, tables and frozen evidence |
| Paper | `paper/` | Local references, drafts, figures, tables and archival manuscript material |

The Python directories remain at the root for import compatibility. They are
one logical code zone, not separate projects.

## Supporting and historical material

| Path | Responsibility |
|---|---|
| `tools/` | Diagnostics, maintenance, and data-layout utilities |
| `PROJECT_CONTROL/logs/` | Consolidated local development exports and historical chat records; ignored by Git |
| `PROJECT_CONTROL/archive/` | Optional local recovery material; ignored by Git |
| `dist/` | Current delivery packages only; generated and ignored by Git |

Python bytecode (`__pycache__/`, `*.pyc`) and test caches (`.pytest_cache/`) are
reproducible local artifacts. They are never source, evidence, or delivery
inputs and may be removed during housekeeping.

The repository has no package-level `main.py`. Use the versioned entry points
listed in the root README.

## Root boundary

User-owned research inputs are kept under `PROJECT_CONTROL/source_inputs/` and
`paper/references/`. They are never treated as source entry points.

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
