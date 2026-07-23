# BC-DPG-FCN v1 真实接口集成补丁

该版本根据当前工程真实接口修正：

```text
DualBranchGatedFCN.forward(input_tensor)
input_tensor: [B, 2, 128, 100]
output["fusion_logits"]
output["gate_weights"]
```

## 安装

```bash
cd BC_DPG_FCN_v1_integration_patch
bash install.sh ~/projects/balloon_radar_project
```

## Fold1 真实 checkpoint 集成测试

```bash
cd ~/projects/balloon_radar_project
conda activate radar-torch

python scripts/test_bc_dpg_real_checkpoint.py \
  --checkpoint results/experiments/dpg_fcn_v4_fold01_seed42/checkpoints/best.pt
```

## Fold4 真实 checkpoint 集成测试

```bash
python scripts/test_bc_dpg_real_checkpoint.py \
  --checkpoint results/experiments/dpg_fcn_v4_fold04_seed42/checkpoints/best.pt
```

## 提取数据集和训练接口

```bash
python scripts/inspect_detection_dataset_interface.py
```

报告位置：

```text
results/data_audit/bc_dpg_preflight/dataset_training_interface.json
```
