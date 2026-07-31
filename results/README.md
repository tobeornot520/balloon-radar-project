# Results layout

| Path | Meaning | Git policy |
|---|---|---|
| `data_audit/` | Manifests, preflight checks and compact diagnostic evidence | Keep selected evidence |
| `experiment_ledger/` | Local run provenance and summary snapshots | Local/ignored by default |
| `experiments/` | Checkpoints, histories and per-run outputs | Local/ignored |
| `final_evidence/` | Frozen tables, reports and paper assets | Keep |
| `analysis/` | Generated exploratory analysis | Local/ignored |
| `reproduction/` | Reproduction smoke and protocol outputs | Local/ignored by default |

An experiment directory is not authoritative by itself. Its ledger row, frozen configuration,
test-access policy and retained summary define whether it may be cited.

Compact outputs are promoted into versioned `data_audit/` or `final_evidence/` directories only
after their provenance and claim boundary have been reviewed.
