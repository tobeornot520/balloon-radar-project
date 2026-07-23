#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${1:-$HOME/projects/balloon_radar_project}"
PATCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STAMP="$(date +%Y%m%d_%H%M%S)"
BACKUP_DIR="$PROJECT_DIR/backups/bc_dpg_fcn_v1_integration_$STAMP"

if [[ ! -d "$PROJECT_DIR/.git" ]]; then
  echo "错误：目标目录不是 Git 项目：$PROJECT_DIR"
  exit 1
fi

mkdir -p \
  "$BACKUP_DIR" \
  "$PROJECT_DIR/models" \
  "$PROJECT_DIR/scripts" \
  "$PROJECT_DIR/configs"

install_one() {
  local relative="$1"
  local source="$PATCH_DIR/$relative"
  local target="$PROJECT_DIR/$relative"

  if [[ ! -f "$source" ]]; then
    echo "错误：补丁缺少 $relative"
    exit 1
  fi

  if [[ -f "$target" ]]; then
    mkdir -p "$BACKUP_DIR/$(dirname "$relative")"
    cp -a "$target" "$BACKUP_DIR/$relative"
  fi

  cp -a "$source" "$target"
  echo "installed: $relative"
}

install_one "models/background_calibrated_dpg_fcn.py"
install_one "scripts/test_bc_dpg_real_checkpoint.py"
install_one "scripts/inspect_detection_dataset_interface.py"
install_one "configs/bc_dpg_fcn_v1.yaml"

echo
echo "安装完成"
echo "备份目录：$BACKUP_DIR"
