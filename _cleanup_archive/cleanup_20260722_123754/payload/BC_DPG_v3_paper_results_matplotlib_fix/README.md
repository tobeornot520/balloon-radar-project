# BC-DPG v3 论文结果脚本 Matplotlib 兼容修复

修复错误：

```text
TypeError: Axes.boxplot() got an unexpected keyword argument 'labels'
```

原因是当前 Matplotlib 版本已不再接受 `labels=`。

本修复版不再向 `boxplot()` 传递 `labels` 或 `tick_labels`，而是在绘图后单独设置横坐标标签，因此兼容新旧 Matplotlib。

## 安装

```bash
cd BC_DPG_v3_paper_results_matplotlib_fix
bash install.sh ~/projects/balloon_radar_project
```

## 重新运行

```bash
cd ~/projects/balloon_radar_project
conda activate radar-torch

python scripts/build_bc_dpg_v3_paper_results.py \
  --folds 1 2 3 4 5 6
```
