# BC-DPG-FCN v3：目标保护型扫描组感知校准

v2 已经证明可以显著降低 Fold1/Fold4 的虚警，但六折统一阈值下：

```text
Pfa：0.0313 → 0.0060
Pd ：0.9120 → 0.8553
```

问题是目标分数也被过度压低。

v3 不再直接预测一个 shift，而是拆分为：

```text
p_background：当前峰是背景伪峰的概率
suppression：最大抑制幅度

shift = p_background × suppression
calibrated_logit = raw_logit - shift
```

## 新增扫描组信息

根据 `sample_id` 的时间戳，将同一次扫描的 beam/azimuth 聚合，生成：

```text
扫描样本数
Raw 分数均值、标准差、中位数
75%、90%分位数与最大值
高于0.3、0.5、原始阈值的比例
H/V分支差异均值
H/V峰位置分歧均值
```

## 目标保护

- 真实目标 shift 超过 0.10 logit 后受到强惩罚；
- 背景 shift 应显著大于目标 shift；
- 最佳 checkpoint 优先满足：

```text
BC validation Pd ≥ Raw validation Pd - 0.01
```

只有满足 Pd 约束后，才优先降低 Pfa。

## 安装

```bash
cd BC_DPG_FCN_v3_scan_target_patch
bash install.sh ~/projects/balloon_radar_project
```

## 第一步：Fold2 烟雾测试

```bash
cd ~/projects/balloon_radar_project
conda activate radar-torch

python scripts/run_bc_dpg_v3.py \
  --folds 2 \
  --smoke \
  --overwrite
```

烟雾测试会先运行冻结 DPG 一次，预计算特征；后续 epoch 只训练小型校准器。

## 第二步：Fold2 正式实验

```bash
python scripts/run_bc_dpg_v3.py \
  --folds 2 \
  --formal \
  --overwrite
```

## 第三步：Fold4 正式实验

Fold2 的目标保护有效后再运行：

```bash
python scripts/run_bc_dpg_v3.py \
  --folds 4 \
  --formal \
  --overwrite
```

## 对比

```bash
python scripts/compare_bc_dpg_v3.py \
  --folds 2 4
```

## 首轮通过标准

Fold2：

```text
v3 Pd ≥ Raw Pd - 0.01
```

Fold4：

```text
v3 Pd ≥ Raw Pd - 0.01
Pfa 明显低于 Raw
background shift > target shift
p(background|background) > p(background|target)
```
