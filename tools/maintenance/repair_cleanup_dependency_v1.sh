#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

echo "============================================================"
echo "恢复被误归档的 detection_dataset_v2.py"
echo "============================================================"

SOURCE_FILE="$(
    find _cleanup_archive \
        -type f \
        -path "*/datasets/detection_dataset_v2.py" \
        -printf '%T@|%p\n' 2>/dev/null |
    sort -t'|' -k1,1nr |
    head -n 1 |
    cut -d'|' -f2-
)"

if [[ -z "${SOURCE_FILE:-}" || ! -f "$SOURCE_FILE" ]]; then
    echo "[ERROR] 未在 _cleanup_archive 中找到 detection_dataset_v2.py"
    exit 1
fi

echo "[FOUND] $SOURCE_FILE"

mkdir -p datasets

if [[ -f datasets/detection_dataset_v2.py ]]; then
    cp -a \
        datasets/detection_dataset_v2.py \
        "datasets/detection_dataset_v2.py.before_repair_$(date +%Y%m%d_%H%M%S)"
fi

# 使用复制而不是移动，保留清理归档作为回滚副本
cp -a "$SOURCE_FILE" datasets/detection_dataset_v2.py

echo "[RESTORED] datasets/detection_dataset_v2.py"

echo
echo "=== 语法检查 ==="

python -m py_compile \
    datasets/detection_dataset_v2.py \
    datasets/polarimetric_detection_dataset_v2.py \
    models/polarimetric_representation_fcn.py \
    training/train_polarimetric_representation_fcn_v2.py

echo "[PASS] Python语法检查"

echo
echo "=== 导入检查 ==="

python - <<'PY'
from datasets.detection_dataset_v2 import (
    DetectionGeometry,
    _generate_heatmap,
    _load_iq_pair,
)
from datasets.polarimetric_detection_dataset_v2 import (
    PolarimetricDetectionDatasetV2,
    representation_channels,
)
from models.polarimetric_representation_fcn import (
    PolarimetricRepresentationFCN,
)

print("DetectionGeometry                  : PASS")
print("_generate_heatmap                  : PASS")
print("_load_iq_pair                      : PASS")
print("PolarimetricDetectionDatasetV2     : PASS")
print("PolarimetricRepresentationFCN      : PASS")
print(
    "polar6_gated channels              :",
    representation_channels("polar6_gated"),
)
print(
    "ri8_gated channels                 :",
    representation_channels("ri8_gated"),
)
PY

echo
echo "=== Stage 3门控极化测试 ==="

python scripts/test_polarimetric_gated_pipeline_v2.py

echo
echo "=== Stage 4 ROI测试 ==="

python scripts/test_roi_polarimetric_stage4_v1.py

echo
echo "=== 检查其他被归档模块是否仍有活动引用 ==="

grep -R \
    "datasets\.radar_dataset\|datasets\.detection_dataset_v2\|datasets\.polarimetric_detection_dataset_v1\|training\.train_background_calibrator\|training\.train_background_tail_calibrator" \
    datasets features models training scripts evaluation tests \
    --include="*.py" \
    --exclude-dir="__pycache__" \
    -n 2>/dev/null || true

echo
echo "============================================================"
echo "依赖恢复和核心测试完成"
echo "暂时不要删除 _cleanup_archive"
echo "============================================================"
