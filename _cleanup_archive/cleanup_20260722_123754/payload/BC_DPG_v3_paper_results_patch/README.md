# BC-DPG-FCN v3 六折扫描组级统计与论文结果包

该脚本不会训练模型，也不会修改现有 checkpoint。

它读取每折 v3 实验目录中的：

```text
tables/summary.json
tables/base_threshold_test_predictions.csv
tables/raw_base_threshold_test_predictions.csv
```

所有比较均使用每折冻结的原始 DPG 阈值。

## 安装

```bash
cd BC_DPG_v3_paper_results_patch
bash install.sh ~/projects/balloon_radar_project
```

## 运行

```bash
cd ~/projects/balloon_radar_project
conda activate radar-torch

python scripts/build_bc_dpg_v3_paper_results.py \
  --folds 1 2 3 4 5 6
```

## 输出目录

```text
results/data_audit/bc_dpg_v3_paper_results/
```

主要表格：

```text
six_fold_sample_metrics.csv
six_fold_pooled_summary.csv
six_fold_scan_group_metrics.csv
scan_group_macro_summary.csv
hardest_scan_groups.csv
fold1_fold4_scan_group_analysis.csv
paper_main_table.csv
paper_main_table.tex
paper_results_summary.md
summary.json
```

高清图片：

```text
figures/fold_pfa_comparison.png
figures/scan_group_pfa_scatter.png
figures/hardest_scan_groups.png
figures/shift_background_vs_target.png
figures/background_probability_separation.png
```

## 统计解释

### 样本级 pooled Pfa

把六折所有背景样本合并后计算：

```text
总虚警数 / 总背景样本数
```

### 扫描组宏平均 Pfa

先为每个独立 scan_group 计算 Pfa，再对所有扫描组等权平均。这样一个包含150个样本的大型扫描组不会获得150倍权重。

脚本还会进行扫描组配对 bootstrap，给出：

```text
Raw Pfa - v3 Pfa 的平均差
95% bootstrap confidence interval
```

### 论文使用原则

主结论应基于：

```text
沿用原始 DPG 阈值
Raw 与 v3 使用同一工作点
Pd保持
Pfa下降
```

不要把重新选择的 v3 低阈值作为主结果。
