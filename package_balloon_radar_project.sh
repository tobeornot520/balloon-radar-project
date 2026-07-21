#!/usr/bin/env bash
set -euo pipefail

# 用法：
#   ./package_balloon_radar_project.sh research
#   ./package_balloon_radar_project.sh minimal
#
# research：代码 + 配置 + metadata + 数据审计结果
# minimal ：最小代码运行包

PROFILE="${1:-research}"
PROJECT_DIR="${PROJECT_DIR:-$(pwd)}"

case "$PROFILE" in
  minimal|research) ;;
  *)
    echo "错误：参数只能是 minimal 或 research"
    exit 1
    ;;
esac

cd "$PROJECT_DIR"

if [[ ! -d .git ]]; then
  echo "错误：当前目录不是 Git 仓库：$PROJECT_DIR"
  exit 1
fi

if ! command -v rsync >/dev/null 2>&1; then
  echo "错误：缺少 rsync。请先执行："
  echo "sudo apt update && sudo apt install -y rsync"
  exit 1
fi

DATE_TAG="$(date +%Y%m%d)"
GIT_COMMIT="$(git rev-parse --short HEAD)"
PACKAGE_NAME="balloon_radar_project_${PROFILE}_${DATE_TAG}_${GIT_COMMIT}"
DIST_DIR="$PROJECT_DIR/dist"
STAGE_DIR="$DIST_DIR/$PACKAGE_NAME"
ARCHIVE_TGZ="$DIST_DIR/${PACKAGE_NAME}.tar.gz"

rm -rf "$STAGE_DIR"
mkdir -p "$STAGE_DIR" "$DIST_DIR"

RSYNC_EXCLUDES=(
  --exclude='.git/'
  --exclude='dist/'
  --exclude='__pycache__/'
  --exclude='*.py[cod]'
  --exclude='.pytest_cache/'
  --exclude='.mypy_cache/'
  --exclude='.ipynb_checkpoints/'
  --exclude='data/raw/'
  --exclude='backups/'
  --exclude='checkpoints/'
  --exclude='weights/'
  --exclude='logs/'
  --exclude='runs/'
  --exclude='wandb/'
  --exclude='tensorboard/'
  --exclude='results/experiments/'
  --exclude='results/analysis/'
  --exclude='results/**/predictions/'
  --exclude='results/**/visualizations/'
  --exclude='results/**/checkpoints/'
  --exclude='detection_ablation_analysis_v2_package/'
  --exclude='detection_diagnostics_v3_package/'
  --exclude='detection_group_split_v1_package/'
  --exclude='dpg_fcn_v1_patch/'
  --exclude='hv_late_fusion_diagnostic_v1/'
  --exclude='current_structure_reader_package/'
  --exclude='full_detection_baseline_v2_package/'
  --exclude='detection_visualization_hotfix_v3_1/'
  --exclude='*.mat'
  --exclude='*.h5'
  --exclude='*.hdf5'
  --exclude='*.npy'
  --exclude='*.npz'
  --exclude='*.bin'
  --exclude='*.pt'
  --exclude='*.pth'
  --exclude='*.ckpt'
  --exclude='*.onnx'
  --exclude='*.zip'
  --exclude='*.rar'
  --exclude='*.7z'
  --exclude='*.tar'
  --exclude='*.tar.gz'
)

copy_item() {
  local item="$1"
  if [[ -e "$item" ]]; then
    rsync -a "${RSYNC_EXCLUDES[@]}" "$item" "$STAGE_DIR/"
  fi
}

# 核心代码与配置
for item in \
  baselines \
  configs \
  datasets \
  evaluation \
  losses \
  metrics \
  models \
  postprocess \
  radar_processing \
  scripts \
  training \
  utils
do
  copy_item "$item"
done

# 环境与说明文件
for item in \
  environment.yml \
  requirements-lock.txt \
  requirements.txt \
  pyproject.toml \
  setup.py \
  setup.cfg \
  README.md \
  LICENSE \
  .gitignore
do
  copy_item "$item"
done

# research 模式附加复现资料
if [[ "$PROFILE" == "research" ]]; then
  copy_item "docs"
  copy_item "tests"

  mkdir -p "$STAGE_DIR/data" "$STAGE_DIR/results"

  if [[ -d data/metadata ]]; then
    rsync -a "${RSYNC_EXCLUDES[@]}" data/metadata "$STAGE_DIR/data/"
  fi

  if [[ -d results/data_audit ]]; then
    rsync -a "${RSYNC_EXCLUDES[@]}" results/data_audit "$STAGE_DIR/results/"
  fi

  if [[ -d results/data_audit_current ]]; then
    rsync -a "${RSYNC_EXCLUDES[@]}" results/data_audit_current "$STAGE_DIR/results/"
  fi
fi

# 创建空目录骨架
mkdir -p \
  "$STAGE_DIR/data/raw/detection_dataset" \
  "$STAGE_DIR/data/raw/classification_dataset" \
  "$STAGE_DIR/data/raw/legacy_uav_positive" \
  "$STAGE_DIR/checkpoints" \
  "$STAGE_DIR/logs" \
  "$STAGE_DIR/results/experiments"

cat > "$STAGE_DIR/DATA_LAYOUT.md" <<'EOF'
# 外部数据放置说明

本压缩包不包含原始雷达数据、模型权重或大规模训练输出。

请将数据放置到：

```text
data/
├── metadata/
└── raw/
    ├── detection_dataset/
    ├── classification_dataset/
    └── legacy_uav_positive/
```

模型权重建议放置到：

```text
checkpoints/
```
EOF

cat > "$STAGE_DIR/INSTALL_AND_RUN.md" <<'EOF'
# 安装与运行

## Conda 环境

```bash
conda env create -f environment.yml
conda activate radar-torch
```

## 基础语法检查

```bash
python -m compileall \
  baselines datasets evaluation models scripts training utils
```

## 常用入口

```bash
python scripts/train_detection_baseline_v2.py --help
python scripts/evaluate_detection_baseline_v2.py --help
python training/train_dual_branch_gated.py --help
```
EOF

cat > "$STAGE_DIR/PACKAGE_INFO.txt" <<EOF
Package profile : $PROFILE
Created at      : $(date -Iseconds)
Git commit      : $(git rev-parse HEAD)
Git branch      : $(git branch --show-current)
Source project  : $PROJECT_DIR
EOF

# Python 语法检查
python -m compileall -q \
  "$STAGE_DIR/baselines" \
  "$STAGE_DIR/datasets" \
  "$STAGE_DIR/evaluation" \
  "$STAGE_DIR/models" \
  "$STAGE_DIR/scripts" \
  "$STAGE_DIR/training" \
  "$STAGE_DIR/utils"

# 删除编译检查生成的 Python 缓存
find "$STAGE_DIR" -type d -name "__pycache__" -prune -exec rm -rf {} +
find "$STAGE_DIR" -type f \( -name "*.pyc" -o -name "*.pyo" \) -delete

# 删除编译检查生成的 Python 缓存
find "$STAGE_DIR" -type d -name "__pycache__" -prune -exec rm -rf {} +
find "$STAGE_DIR" -type f \( -name "*.pyc" -o -name "*.pyo" \) -delete

# 生成压缩包
tar -C "$DIST_DIR" -czf "$ARCHIVE_TGZ" "$PACKAGE_NAME"

# 安全检查：仅检查文件，不检查空目录
FORBIDDEN_PATTERN='/(data/raw|backups|checkpoints|logs|runs|wandb)/|[.](mat|h5|hdf5|npy|npz|bin|pt|pth|ckpt|onnx)$'

if tar -tzf "$ARCHIVE_TGZ" \
  | grep -v '/$' \
  | grep -E "$FORBIDDEN_PATTERN" \
  > "$DIST_DIR/${PACKAGE_NAME}_forbidden_files.txt"; then

  echo "错误：压缩包中发现不应包含的文件："
  head -50 "$DIST_DIR/${PACKAGE_NAME}_forbidden_files.txt"
  exit 1
else
  rm -f "$DIST_DIR/${PACKAGE_NAME}_forbidden_files.txt"
fi

echo
echo "打包完成："
echo "$ARCHIVE_TGZ"
echo
echo "文件大小："
du -h "$ARCHIVE_TGZ"
echo
echo "包内文件数量："
tar -tzf "$ARCHIVE_TGZ" | wc -l
