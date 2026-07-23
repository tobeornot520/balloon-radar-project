# BC-DPG-FCN v1 预检补丁

该补丁不修改现有 DPG-FCN，也不覆盖已有 checkpoint。

## 安装

```bash
cd BC_DPG_FCN_v1_preflight_patch
bash install.sh ~/projects/balloon_radar_project
```

## 模型骨架烟雾测试

```bash
cd ~/projects/balloon_radar_project
conda activate radar-torch
python scripts/test_background_calibrator.py
```

## Fold1 兼容性报告

```bash
python scripts/inspect_bc_dpg_compat.py \
  --checkpoint results/experiments/dpg_fcn_v4_fold01_seed42/checkpoints/best.pt \
  --manifest results/data_audit/dataset_v4_multifold/fold_01_manifest.csv \
  --output results/data_audit/bc_dpg_preflight/fold01_compat.json
```

## Fold4 兼容性报告

```bash
python scripts/inspect_bc_dpg_compat.py \
  --checkpoint results/experiments/dpg_fcn_v4_fold04_seed42/checkpoints/best.pt \
  --manifest results/data_audit/dataset_v4_multifold/fold_04_manifest.csv \
  --output results/data_audit/bc_dpg_preflight/fold04_compat.json
```

下一步将依据真实 checkpoint、模型构造函数、forward 返回值和 manifest 列，
生成完全匹配当前工程的训练与评估脚本。
