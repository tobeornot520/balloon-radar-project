# Polarimetric Twofold Diagnostics Collection V1

生成时间：2026-07-22T08:09:11.682325+00:00

## 状态

- experiments requested：8
- experiments found：8
- prediction candidate tables：24
- usable prediction tables：16
- low-FPR rows：16
- threshold-transfer rows：8
- low-FPR status：COMPLETE
- acceptance ZIP：polarimetric_twofold_diagnostics_acceptance_v1.zip

## 说明

本次工具只读取已有 Fold 1 / Fold 4 正式实验，不重新训练、不修改权重。
若状态为 `COMPLETE`，可直接使用本包中的低FPR、分数分位数、阈值迁移和组级统计。
若状态为 `NEED_PREDICTION_EXPORT`，说明现有实验结果未保存可识别的逐样本标签与分数表；
本包已同时收集真实训练入口、Dataset、特征构造、配置、checkpoint键名和全部CSV字段，
可据此生成与当前工程零猜测兼容的预测导出补丁。
