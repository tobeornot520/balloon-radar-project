# Detection acquisition-order audit

## Decision

**Formal causal-training gate: CLOSED.** The current files do not contain a verified within-scan sample acquisition order.

The audit covers 1148 samples in 71 target and 6 background scan groups. The timestamp encoded in each sample ID is identical for every sample in its scan group, so it cannot order samples within that group.

## Evidence

- MAT files contain H/V IQ arrays but no timestamp-like variable.
- MAT v5 `Created on` values are at least 49.1 days after the filename acquisition second and track filesystem mtime within 3.0 seconds. They are conversion/save times.
- Filesystem mtime belongs to later file handling and is not acquisition metadata.
- `(beam_layer, azimuth_deg, sample_id)` is unique and deterministic within each group, but no hardware log verifies that execution order.

## Allowed use

The inferred beam/azimuth order may be used only for interface smoke tests that are explicitly labelled development-only. It must not be used for model/window selection, formal performance claims, or deployment.

## Reopening the gate

Provide a per-sample acquisition timestamp or monotonic hardware sequence number, document clock resolution and reset behavior, and verify one-to-one alignment with sample IDs before causal model selection begins.

See `order_source_summary.csv` and `group_order_coverage.csv` for the machine-readable findings.
