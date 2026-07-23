#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${1:-$HOME/projects/balloon_radar_project}"
PATCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET="$PROJECT_DIR/scripts/audit_detection_split_leakage.py"
STAMP="$(date +%Y%m%d_%H%M%S)"

if [[ ! -d "$PROJECT_DIR/.git" ]]; then
  echo "错误：目标目录不是 Git 项目：$PROJECT_DIR"
  exit 1
fi

mkdir -p "$PROJECT_DIR/scripts" "$PROJECT_DIR/backups"

if [[ -f "$TARGET" ]]; then
  cp -a "$TARGET" \
    "$PROJECT_DIR/backups/audit_detection_split_leakage_${STAMP}.py"
fi

cp -a \
  "$PATCH_DIR/scripts/audit_detection_split_leakage.py" \
  "$TARGET"

cd "$PROJECT_DIR"

python -m compileall \
  scripts/audit_detection_split_leakage.py

echo
echo "审计脚本安装完成。"
echo
echo "先运行快速审计："
echo "python scripts/audit_detection_split_leakage.py \\"
echo "  --folds 1 2 3 4 5 6 \\"
echo "  --test-feedback-known"
