#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${1:-$HOME/projects/balloon_radar_project}"
PATCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET="$PROJECT_DIR/scripts/evaluate_bc_dpg_v2_global_threshold.py"
STAMP="$(date +%Y%m%d_%H%M%S)"

mkdir -p "$PROJECT_DIR/scripts" "$PROJECT_DIR/backups"

if [[ -f "$TARGET" ]]; then
  cp -a "$TARGET" \
    "$PROJECT_DIR/backups/evaluate_bc_dpg_v2_global_threshold_${STAMP}.py"
fi

cp -a \
  "$PATCH_DIR/scripts/evaluate_bc_dpg_v2_global_threshold.py" \
  "$TARGET"

cd "$PROJECT_DIR"

python -m compileall \
  scripts/evaluate_bc_dpg_v2_global_threshold.py

echo
echo "安装完成。运行："
echo "python scripts/evaluate_bc_dpg_v2_global_threshold.py --folds 1 2 3 4 5 6"
