#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${1:-$HOME/projects/balloon_radar_project}"

python "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/apply_fix.py"

cd "$PROJECT_DIR"

python -m compileall   models/background_calibrated_dpg_fcn.py   training/train_background_calibrator.py

echo
echo "修复完成。请重新运行："
echo "python scripts/run_bc_dpg_fold14.py --folds 1 --formal --overwrite"
