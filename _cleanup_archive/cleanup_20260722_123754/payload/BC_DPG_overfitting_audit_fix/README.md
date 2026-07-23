# 修复说明

本修复版优先使用 manifest 中的 `new_split` 字段作为当前六折划分。

`original_split` 仅代表原始划分来源，不用于本次泄漏审计。

# DPG / BC-DPG 数据泄漏与过拟合审计

该工具不会重新训练任何模型，也不会修改现有结果。

它检查四类风险：

1. **Fold 内直接泄漏**
   - 同一样本同时出现在 train/val/test；
   - 同一 MAT、标签文件或源文件跨 split；
   - 可选 SHA256 内容哈希，发现改名复制的相同文件。

2. **扫描环境泄漏**
   - 同一个时间戳扫描组同时出现在 train/val/test；
   - 这种情况会让模型在训练时见到测试环境的其他 beam/azimuth。

3. **六折轮转异常**
   - 对完整六折，理想情况下每个样本或扫描组：
     - train 4 次；
     - val 1 次；
     - test 1 次。

4. **环境有效样本量与测试反馈过拟合**
   - 每个 split 有多少独立扫描时间；
   - 是否只有一两个环境却包含大量 beam/azimuth 样本；
   - 记录当前 Fold 测试结果已经参与 v1/v2/v3 设计。

## 安装

```bash
cd BC_DPG_overfitting_audit_patch
bash install.sh ~/projects/balloon_radar_project
```

## 第一步：快速审计

```bash
cd ~/projects/balloon_radar_project
conda activate radar-torch

python scripts/audit_detection_split_leakage.py \
  --folds 1 2 3 4 5 6 \
  --test-feedback-known
```

## 第二步：文件内容哈希审计

快速审计通过后，再运行较慢但更严格的文件哈希检查：

```bash
python scripts/audit_detection_split_leakage.py \
  --folds 1 2 3 4 5 6 \
  --test-feedback-known \
  --hash-files
```

## 输出目录

```text
results/data_audit/overfitting_audit/
```

主要文件：

```text
summary.json
within_fold_split_overlaps.csv
within_split_duplicates.csv
split_environment_summary.csv
cross_fold_sample_rotation.csv
cross_fold_scan_group_rotation.csv
overfitting_risk_register.csv
normalized_manifest_rows.csv
```

## 关键解释

### `within_fold_split_overlaps.csv`

- `CRITICAL`：相同样本、相同文件路径或相同文件内容跨 split；
- `HIGH`：同一扫描环境跨 split；
- `MEDIUM`：文件名或 stem 相同，需要人工核查。

### `split_environment_summary.csv`

重点看：

```text
unique_scan_group_count
largest_scan_group_size
```

如果一个 split 有数百个样本，但只有一个扫描组，统计意义上的独立环境样本量仍接近 1。

### 测试反馈过拟合

本项目的 Fold1/Fold4 和六折测试结果已经影响 v2/v3 设计，因此即使文件划分完全无泄漏，当前测试指标也应视为开发阶段估计。最终论文需要：

```text
冻结 v3
→ 新采集扫描组一次性盲测
```

若无法增加数据，则使用严格的嵌套留一扫描组交叉验证。
