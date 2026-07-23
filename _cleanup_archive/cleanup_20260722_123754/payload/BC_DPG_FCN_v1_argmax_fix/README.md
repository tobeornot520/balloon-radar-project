# BC-DPG-FCN v1 Argmax / AMP 修复

问题来自 AMP/FP16 数值舍入，而不是模型结构错误。

修复内容：

- 冻结 DPG-FCN 仍可使用 AMP；
- 校准特征、温度、偏置和仿射变换强制使用 FP32；
- 峰值索引直接从 FP32 logits 提取；
- 距离和速度定位显式继承原始 DPG-FCN 峰值；
- 数值 argmax 差异仅作为诊断记录，不再终止训练。

安装：

```bash
cd BC_DPG_FCN_v1_argmax_fix
bash install.sh ~/projects/balloon_radar_project
```

重跑：

```bash
cd ~/projects/balloon_radar_project
conda activate radar-torch

python scripts/run_bc_dpg_fold14.py   --folds 1   --formal   --overwrite
```
