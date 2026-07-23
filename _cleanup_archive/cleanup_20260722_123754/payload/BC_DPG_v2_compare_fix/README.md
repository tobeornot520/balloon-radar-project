# BC-DPG v2 对比脚本修复

旧脚本错误地强制读取：

```text
results/experiments/bc_dpg_fcn_v4_fold04_seed42/tables/summary.json
```

这是 BC-DPG v1 的 Fold4 结果，但你没有运行 v1 Fold4，因此报错。

新脚本只比较：

```text
冻结的原始 DPG 输出
vs
BC-DPG-FCN v2 输出
```

这些指标都已经保存在 v2 的 `summary.json` 中，不再依赖 v1。

安装：

```bash
cd BC_DPG_v2_compare_fix
bash install.sh ~/projects/balloon_radar_project
```

运行：

```bash
cd ~/projects/balloon_radar_project
conda activate radar-torch

python scripts/compare_bc_dpg_v2.py \
  --folds 1 4
```
