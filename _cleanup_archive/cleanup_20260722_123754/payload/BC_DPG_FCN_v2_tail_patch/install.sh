#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${1:-$HOME/projects/balloon_radar_project}"
PATCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STAMP="$(date +%Y%m%d_%H%M%S)"
BACKUP_DIR="$PROJECT_DIR/backups/bc_dpg_v2_tail_$STAMP"

if [[ ! -d "$PROJECT_DIR/.git" ]]; then
  echo "错误：目标目录不是 Git 项目：$PROJECT_DIR"
  exit 1
fi

mkdir -p \
  "$BACKUP_DIR" \
  "$PROJECT_DIR/models" \
  "$PROJECT_DIR/training" \
  "$PROJECT_DIR/scripts" \
  "$PROJECT_DIR/configs"

install_one() {
  local relative="$1"
  local source="$PATCH_DIR/$relative"
  local target="$PROJECT_DIR/$relative"

  if [[ -f "$target" ]]; then
    mkdir -p "$BACKUP_DIR/$(dirname "$relative")"
    cp -a "$target" "$BACKUP_DIR/$relative"
  fi

  cp -a "$source" "$target"
  echo "installed: $relative"
}

install_one "models/background_tail_calibrated_dpg_fcn.py"
install_one "training/train_background_tail_calibrator.py"
install_one "scripts/run_bc_dpg_v2_tail.py"
install_one "scripts/compare_bc_dpg_v2.py"
install_one "configs/bc_dpg_fcn_v2_tail.yaml"

cd "$PROJECT_DIR"
python -m compileall \
  models/background_tail_calibrated_dpg_fcn.py \
  training/train_background_tail_calibrator.py \
  scripts/run_bc_dpg_v2_tail.py \
  scripts/compare_bc_dpg_v2.py

echo
echo "BC-DPG-FCN v2 tail patch installed."
echo "Backup directory: $BACKUP_DIR"
