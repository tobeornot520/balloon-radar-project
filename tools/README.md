# Auxiliary tools

This directory contains utilities that support the research pipeline but are not
training or evaluation entry points.

- `diagnostics/`: read-only data and context inspection tools.
- `maintenance/`: recovery tools retained for reproducibility.
- `radar_data_reader/`: standalone data-layout preparation package.

Run project experiments from `scripts/` and `training/`. Historical patch
installers and one-off repair artifacts are kept locally under
`_cleanup_archive/` and are not part of the active source tree.
