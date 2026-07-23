#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${1:-$HOME/projects/balloon_radar_project}"
PATCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STAMP="$(date +%Y%m%d_%H%M%S)"
BACKUP_DIR="$PROJECT_DIR/backups/bc_dpg_v3_$STAMP"

if [[ ! -d "$PROJECT_DIR/.git" ]]; then
  echo "错误：目标目录不是 Git 项目：$PROJECT_DIR"
  exit 1
fi

if [[ ! -f \
  "$PROJECT_DIR/models/background_tail_calibrated_dpg_fcn.py" \
 ]]; then
  echo "错误：缺少 v2 特征提取文件："
  echo "$PROJECT_DIR/models/background_tail_calibrated_dpg_fcn.py"
  echo "请保留已经安装的 BC-DPG-FCN v2 文件。"
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

install_one "models/target_protected_scan_calibrator.py"
install_one "training/train_target_protected_scan_calibrator.py"
install_one "scripts/run_bc_dpg_v3.py"
install_one "scripts/compare_bc_dpg_v3.py"
install_one "configs/bc_dpg_fcn_v3_scan_target.yaml"

cd "$PROJECT_DIR"

python -m compileall \
  models/target_protected_scan_calibrator.py \
  training/train_target_protected_scan_calibrator.py \
  scripts/run_bc_dpg_v3.py \
  scripts/compare_bc_dpg_v3.py

echo
echo "BC-DPG-FCN v3 安装完成。"
echo "备份目录：$BACKUP_DIR"
echo
echo "先运行 Fold2 烟雾测试："
echo "python scripts/run_bc_dpg_v3.py --folds 2 --smoke --overwrite"
