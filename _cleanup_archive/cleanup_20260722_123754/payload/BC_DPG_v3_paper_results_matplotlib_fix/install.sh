#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${1:-$HOME/projects/balloon_radar_project}"
PATCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET="$PROJECT_DIR/scripts/build_bc_dpg_v3_paper_results.py"
STAMP="$(date +%Y%m%d_%H%M%S)"

if [[ ! -d "$PROJECT_DIR/.git" ]]; then
  echo "错误：目标目录不是 Git 项目：$PROJECT_DIR"
  exit 1
fi

mkdir -p "$PROJECT_DIR/scripts" "$PROJECT_DIR/backups"

if [[ -f "$TARGET" ]]; then
  cp -a "$TARGET" \
    "$PROJECT_DIR/backups/build_bc_dpg_v3_paper_results_${STAMP}.py"
fi

cp -a \
  "$PATCH_DIR/scripts/build_bc_dpg_v3_paper_results.py" \
  "$TARGET"

cd "$PROJECT_DIR"

python -m compileall \
  scripts/build_bc_dpg_v3_paper_results.py

echo
echo "Matplotlib 兼容修复已安装。"
echo
echo "重新运行："
echo "python scripts/build_bc_dpg_v3_paper_results.py \\"
echo "  --folds 1 2 3 4 5 6"
