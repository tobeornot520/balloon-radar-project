# BC-DPG-FCN v2：背景高分尾部抑制

v1 的诊断表明：

- 中等分数背景可以被压低；
- 但 `T < 1` 会放大极高分背景；
- 验证阈值迁移到测试背景时仍不稳定。

v2 改为：

```text
calibrated_logits = raw_logits - shift
shift >= 0
```

因此任何样本的分数都只能下降，不能高于原始 DPG。

新增 24 维特征，包括：

- H/V 输入统计；
- gate 权重；
- fusion 均值、方差、top-k；
- top1-top2 峰值间隔；
- 高分单元比例；
- 熵与局部峰值对比；
- H/V 分支峰值和峰位置分歧。

损失包含：

```text
背景高分尾部损失
+ 目标分数下限损失
+ 目标/背景配对排序损失
+ shift 正则
```

## 安装

```bash
cd BC_DPG_FCN_v2_tail_patch
bash install.sh ~/projects/balloon_radar_project
```

## 先跑 Fold1 烟雾测试

```bash
cd ~/projects/balloon_radar_project
conda activate radar-torch

python scripts/run_bc_dpg_v2_tail.py \
  --folds 1 \
  --smoke \
  --overwrite
```

## 烟雾测试通过后跑 Fold1 正式实验

```bash
python scripts/run_bc_dpg_v2_tail.py \
  --folds 1 \
  --formal \
  --overwrite
```

## 对比

```bash
python scripts/compare_bc_dpg_v2.py \
  --folds 1
```
