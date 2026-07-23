# 极化门控两折诊断采集 V1

## 目的

本工具用于补齐 `polarimetric_gated_twofold_formal_acceptance.zip` 中缺少的诊断材料，重点获取：

- 验证集和测试集逐样本预测表；
- ROC-AUC；
- partial AUC@5% FPR；
- TPR@1% FPR；
- TPR@5% FPR；
- 目标与背景分数分位数；
- 验证阈值在测试背景中的迁移情况；
- source_file / scan_group 级分数漂移；
- 实验目录、CSV字段、checkpoint键名和相关源码接口。

工具只读取现有结果，不重新训练模型，不修改权重，不覆盖正式结果，也不修改已冻结的 BC-DPG-FCN v3。

## 安装

把补丁 ZIP 放到项目根目录：

```bash
cd ~/projects/balloon_radar_project
unzip -o BC_DPG_polarimetric_twofold_diagnostics_collection_v1.zip -d .
python apply_polarimetric_twofold_diagnostics_collection_v1.py
```

## 运行并自动打包

```bash
cd ~/projects/balloon_radar_project
conda activate radar-torch
set -o pipefail

python scripts/collect_polarimetric_twofold_diagnostics_v1.py \
  --folds 1 4 \
  2>&1 | tee polarimetric_twofold_diagnostics_terminal_v1.log
```

脚本会自动生成：

```text
results/data_audit/polarimetric_twofold_diagnostics_v1/
├── collection_status.json
├── experiment_file_inventory.csv
├── csv_schema_inventory.csv
├── prediction_candidate_inventory.csv
├── checkpoint_inventory.csv
├── score_distribution_quantiles.csv
├── low_fpr_metrics.csv
├── threshold_transfer_metrics.csv
├── group_score_stratification.csv
├── missing_requirements.csv
├── README_diagnostics_collection.md
├── source_context/
└── collected_outputs/
```

同时在项目根目录自动生成：

```text
polarimetric_twofold_diagnostics_acceptance_v1.zip
```

请直接上传这个 ZIP，不需要再手工挑文件。

## 正常状态

理想情况下终端应显示：

```text
experiments requested : 8
experiments found     : 8
prediction tables     : 16 或更多
low-FPR status        : COMPLETE
acceptance zip        : .../polarimetric_twofold_diagnostics_acceptance_v1.zip
```

若显示：

```text
low-FPR status : NEED_PREDICTION_EXPORT
```

说明现有正式实验目录没有保存逐样本预测表。工具仍会把真实训练脚本、Dataset接口、checkpoint键名和结果目录清单打包；上传验收ZIP后即可据此生成与当前工程精确兼容的预测导出补丁，不再猜测接口。
