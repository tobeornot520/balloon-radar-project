# Stage 4 下一阶段完整命令

## 安装

```bash
cd ~/projects/balloon_radar_project
unzip -o BC_DPG_stage4_next_all_in_one_v1.zip -d .
python apply_stage4_next_all_in_one_v1.py
```

## 预检

```bash
conda activate radar-torch
python scripts/preflight_roi_stage4_selected_sixfold_v1.py
```

预检允许 Power2 checkpoint 缺失；全自动脚本会补齐缺失折。manifest 或核心脚本缺失时才会停止。

## 全自动执行（推荐）

```bash
cd ~/projects/balloon_radar_project
conda activate radar-torch
set -o pipefail
bash scripts/run_stage4_next_all_v1.sh 2>&1 | tee stage4_next_all_terminal_v1.log
```

脚本依次完成：

1. 补齐缺失的 Fold 2/3/5/6 Power2 正式 checkpoint；
2. 在新增折运行三模式 smoke；
3. 在六折运行 Power2 baseline、ROI power control、ROI RI4；
4. 运行不变量审计；
5. 生成六折论文图表；
6. 打包验收材料。

正常最终输出：

```text
roi_stage4_selected_sixfold_formal_acceptance_v1.zip
```

## 六折完成后的联合接口采集

```bash
python scripts/collect_roi_bc_dpg_joint_context_v1.py
```

生成：

```text
roi_bc_dpg_joint_context_acceptance_v1.zip
```

该包用于下一阶段 ROI 与 BC-DPG 顺序组合审计，不包含原始 MAT 和 checkpoint 二进制。
