#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${1:-$HOME/projects/balloon_radar_project}"
PATCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET="$PROJECT_DIR/scripts/compare_bc_dpg_v2.py"
STAMP="$(date +%Y%m%d_%H%M%S)"

mkdir -p "$PROJECT_DIR/scripts" "$PROJECT_DIR/backups"

if [[ -f "$TARGET" ]]; then
  cp -a "$TARGET" \
    "$PROJECT_DIR/backups/compare_bc_dpg_v2_${STAMP}.py"
fi

cp -a "$PATCH_DIR/scripts/compare_bc_dpg_v2.py" "$TARGET"

cd "$PROJECT_DIR"
python -m compileall scripts/compare_bc_dpg_v2.py

echo
echo "对比脚本修复完成。运行："
echo "python scripts/compare_bc_dpg_v2.py --folds 1 4"
