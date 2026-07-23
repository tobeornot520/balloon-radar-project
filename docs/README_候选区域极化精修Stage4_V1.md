# 候选区域引导极化精修 Stage 4 V1

## 研究定位

本阶段不重新训练或修改已冻结的 BC-DPG-FCN v3，也不让显式极化分支重新决定全图目标位置。每一折先加载已经完成的 Power2 正式 checkpoint，冻结其参数并产生候选峰；随后只在候选峰周围裁剪局部 ROI，用小型样本独立网络判断该候选是否更像真实目标或背景伪峰。

Stage 4 的主评价继续使用 Power2 checkpoint 中保存的原始部署阈值。ROI 网络采用 suppression-only 设计，只能降低候选分数，不能提高分数，因此在原始固定阈值下不会制造新的虚警；代价是可能压低真实目标，需要通过验证集 Pd 约束和 target-protection loss 控制。

## 第一轮五组对照

1. `power2_baseline`
2. `power2_roi_power_control`
3. `power2_roi_ri4`
4. `power2_roi_polar6_gated`
5. `power2_roi_ri4_polar6_gated`

所有 ROI 模式统一为 8 通道输入和相同网络容量。第一轮只运行 Fold 1 与 Fold 4。

## 文件说明

- `features/roi_polarimetric_refinement.py`：共享十通道源、ROI裁剪和模式选择。
- `datasets/roi_polarimetric_refinement_dataset.py`：完整帧源数据集与候选缓存数据集。
- `models/roi_polarimetric_refiner.py`：局部ROI suppression-only精修器。
- `scripts/build_roi_polarimetric_cache_v1.py`：使用冻结Power2 checkpoint建立候选ROI缓存。
- `training/train_roi_polarimetric_refiner_v1.py`：训练、验证和逐样本预测导出。
- `scripts/run_roi_polarimetric_stage4_v1.py`：Fold 1/4批量入口。
- `scripts/summarize_roi_polarimetric_stage4_v1.py`：固定阈值、低FPR、迁移和救回/退化汇总。
- `scripts/test_roi_polarimetric_stage4_v1.py`：无原始数据的合成接口测试。
- `scripts/package_roi_polarimetric_stage4_acceptance_v1.py`：生成单一验收ZIP，不包含权重、缓存张量或原始MAT。

## 安全边界

- 不修改旧Power2、RI4、Polar6-gated、RI8-gated实验。
- 不修改BC-DPG-FCN v3。
- 新结果统一写入`results/experiments/roi_polar_stage4_v1_*`。
- cache写入`results/data_audit/roi_polarimetric_stage4_v1/cache/`。
- smoke仅验证接口，不作科学结论。
- 显式极化量仍使用`relative_ZDR_like`、`local_rho_HV`和`relative phase`等谨慎命名。
