#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${1:-$HOME/projects/balloon_radar_project}"
PACKAGE_ROOT="$(cd "$(dirname "$0")" && pwd)"
STAMP="$(date +%Y%m%d_%H%M%S)"
BACKUP_ROOT="$PROJECT_ROOT/backups/radar_data_reader_$STAMP"

mkdir -p "$PROJECT_ROOT/datasets" "$PROJECT_ROOT/scripts" "$BACKUP_ROOT"

backup_if_exists() {
  local dst="$1"
  if [[ -e "$dst" ]]; then
    mkdir -p "$BACKUP_ROOT/$(dirname "${dst#$PROJECT_ROOT/}")"
    cp -a "$dst" "$BACKUP_ROOT/${dst#$PROJECT_ROOT/}"
  fi
}

for rel in \
  datasets/radar_tasks.py \
  scripts/prepare_radar_data_layout.py \
  scripts/audit_and_smoke_test.py; do
  backup_if_exists "$PROJECT_ROOT/$rel"
  cp "$PACKAGE_ROOT/$rel" "$PROJECT_ROOT/$rel"
done
chmod +x "$PROJECT_ROOT/scripts/prepare_radar_data_layout.py" "$PROJECT_ROOT/scripts/audit_and_smoke_test.py"

python -m py_compile \
  "$PROJECT_ROOT/datasets/radar_tasks.py" \
  "$PROJECT_ROOT/scripts/prepare_radar_data_layout.py" \
  "$PROJECT_ROOT/scripts/audit_and_smoke_test.py"

echo "安装完成：$PROJECT_ROOT"
echo "如有旧文件，备份位置：$BACKUP_ROOT"
echo "没有覆盖 datasets/__init__.py，也没有改动原 data/。"
