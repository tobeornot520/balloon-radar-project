#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python scripts/preflight_roi_stage4_selected_sixfold_v1.py
MISSING=$(python - <<'PYINNER'
import json
from pathlib import Path
d=json.loads(Path('results/data_audit/roi_stage4_selected_sixfold_v1/preflight_status.json').read_text());print(' '.join(map(str,d['missing_power2_folds'])))
PYINNER
)
if [[ -n "$MISSING" ]]; then
 python scripts/run_polarimetric_representation_benchmark_v2.py --folds $MISSING --modes power2 --formal 2>&1 | tee roi_stage4_missing_power2_formal_terminal_v1.log
fi
python scripts/run_roi_stage4_selected_sixfold_v1.py --folds 2 3 5 6 --modes power2_baseline power2_roi_power_control power2_roi_ri4 --smoke 2>&1 | tee roi_stage4_selected_sixfold_smoke_terminal_v1.log
python scripts/run_roi_stage4_selected_sixfold_v1.py --folds 1 2 3 4 5 6 --modes power2_baseline power2_roi_power_control power2_roi_ri4 --formal 2>&1 | tee roi_stage4_selected_sixfold_formal_terminal_v1.log
python scripts/audit_roi_stage4_selected_sixfold_v1.py --folds 1 2 3 4 5 6 --scope formal
python scripts/generate_roi_stage4_paper_assets_v1.py --scope formal
python scripts/package_roi_stage4_selected_sixfold_acceptance_v1.py --scope formal --folds 1 2 3 4 5 6 --terminal-log roi_stage4_selected_sixfold_formal_terminal_v1.log
