# BC-DPG-FCN v1 正式训练补丁

该补丁基于已经验证通过的真实接口：

```text
input: [B, 2, 128, 100]
base output: fusion_logits, h_logits, v_logits, gate_weights
checkpoint: model_state_dict
dataset: DetectionRadarDatasetV3
```

现有 DPG-FCN 全部冻结，只训练约千级参数的背景校准头。

## 1. 安装

```bash
cd BC_DPG_FCN_v1_training_patch
bash install.sh ~/projects/balloon_radar_project
```

## 2. 语法检查

```bash
cd ~/projects/balloon_radar_project
conda activate radar-torch

python -m compileall \
  models/background_calibrated_dpg_fcn.py \
  training/train_background_calibrator.py \
  scripts/evaluate_background_calibrator.py \
  scripts/run_bc_dpg_fold14.py \
  scripts/compare_bc_dpg_results.py

python training/train_background_calibrator.py --help
```

## 3. Fold1 小样本烟雾测试

```bash
python scripts/run_bc_dpg_fold14.py \
  --folds 1 \
  --smoke \
  --overwrite
```

对应配置：

```text
每类 8 个样本
3 epochs
batch size 4
Fold1
```

输出目录：

```text
results/experiments/bc_dpg_fcn_v4_fold01_seed42_smoke/
```

## 4. 查看烟雾测试摘要

```bash
python - <<'PY'
import json
from pathlib import Path

path = Path(
    "results/experiments/"
    "bc_dpg_fcn_v4_fold01_seed42_smoke/"
    "tables/summary.json"
)
summary = json.loads(path.read_text(encoding="utf-8"))

for key in (
    "best_epoch",
    "validation_threshold",
    "test_metrics",
    "raw_test_metrics",
    "fixed_test_metrics",
    "raw_fixed_test_metrics",
    "argmax_preserved_validation",
    "argmax_preserved_test",
):
    print(f"\n[{key}]")
    print(summary[key])
PY
```

## 5. Fold1、Fold4 正式实验

烟雾测试通过后：

```bash
python scripts/run_bc_dpg_fold14.py \
  --folds 1 4 \
  --formal \
  --overwrite
```

## 6. 对比正式结果

```bash
python scripts/compare_bc_dpg_results.py \
  --folds 1 4
```

## 设计说明

训练使用与现有评价一致的样本分数：

```text
score = sigmoid(calibrated_logits).amax()
```

训练损失在对应 max logit 上使用 BCEWithLogits，并加入温度和偏置正则。

每个样本只预测一个正温度 T 和一个偏置 b：

```text
calibrated_logits = (raw_logits - b) / T
```

因此不会改变距离—速度热图的空间 argmax。训练和评价过程中会逐样本检查这一性质。
