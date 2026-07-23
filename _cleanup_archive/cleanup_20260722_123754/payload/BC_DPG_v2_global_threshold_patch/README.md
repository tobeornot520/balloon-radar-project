# BC-DPG v2 六折全局阈值评估

该脚本不会重新训练模型。

它会：

1. 合并六折验证集预测；
2. 为 Raw DPG 选择一个全局阈值；
3. 为 BC-DPG v2 选择一个全局阈值；
4. 将阈值原样应用到六折测试集；
5. 输出总体和逐 Fold 的 Pd、Pfa、AUC；
6. 按扫描时间分组分析背景虚警。

默认同时计算两种阈值策略：

```text
two_per_fold：每折允许 2 个验证虚警，六折共 12 个
pfa05：合并验证集 Pfa 不超过 5%
```

终端主要显示 `two_per_fold`。

## 安装

```bash
cd BC_DPG_v2_global_threshold_patch
bash install.sh ~/projects/balloon_radar_project
```

## 运行

```bash
cd ~/projects/balloon_radar_project
conda activate radar-torch

python scripts/evaluate_bc_dpg_v2_global_threshold.py \
  --folds 1 2 3 4 5 6
```

## 输出目录

```text
results/data_audit/bc_dpg_global_threshold/
```

主要文件：

```text
global_thresholds.csv
global_threshold_metrics.csv
global_threshold_comparison.csv
scan_group_false_alarm_analysis.csv
global_validation_threshold_curves.csv
summary.json
```

## 注意

六折对应六套 DPG/校准模型。该实验验证的是：

```text
六套模型的输出经过 v2 后，
是否比原始输出更适合使用统一数值阈值。
```

它不是“单个模型跨六个测试域”的最终部署实验。
