# BC-DPG-FCN v3 最终冻结与论文证据补丁 V1

本补丁不重新训练模型，也不会覆盖现有正式实验。它读取已经完成的六折部署对照、九项消融和 shift regularization 正式扫描结果，生成当前模型的最终冻结证据包。

## 最终固定决策

- 模型：BC-DPG-FCN v3
- shift_regularization：0.01
- 完整模型定位：离线扫描组上下文增强的背景条件校准器
- v3.1 折内自适应权重：仅作为探索结果，不替代当前固定权重模型

## 生成内容

- 原始 DPG、样本独立 BC、扫描上下文 BC 的六折统一对照
- 九项消融汇总与逐折明细
- 正则权重验证选择与“不升级 v3.1”的决策记录
- 论文结果中文草稿、可/不可声称边界
- 六折数据 manifest、源代码、配置和模型 checkpoint 的 SHA256
- 4 张结果图和 12 张结构化 CSV 表
- 项目根目录中的 `bc_dpg_v3_final_evidence.zip`

## 安装

把本压缩包放在项目根目录后执行：

```bash
cd ~/projects/balloon_radar_project
unzip -o BC_DPG_v3_final_freeze_paper_evidence_patch_v1.zip -d .
python apply_bc_dpg_v3_final_freeze_paper_evidence_v1.py
```

## 正式运行

```bash
cd ~/projects/balloon_radar_project
set -o pipefail
python scripts/build_bc_dpg_v3_final_evidence.py   --include-checkpoints   --require-all   --package   2>&1 | tee bc_dpg_v3_final_evidence_terminal.log
```

正常结束应包含：

```text
final model            : BC-DPG-FCN v3
shift regularization   : 0.01
checkpoint hashes      : 12
missing hash inputs    : 0
status                 : PASS
```

## 验收材料打包

```bash
cd ~/projects/balloon_radar_project

zip -j bc_dpg_v3_final_freeze_acceptance.zip   bc_dpg_v3_final_evidence_terminal.log   bc_dpg_v3_final_evidence.zip   results/final_evidence/bc_dpg_v3_final/FINAL_EVIDENCE_REPORT.md   results/final_evidence/bc_dpg_v3_final/PAPER_RESULTS_DRAFT_ZH.md   results/final_evidence/bc_dpg_v3_final/final_model_spec.json   results/final_evidence/bc_dpg_v3_final/final_evidence_audit.json   results/final_evidence/bc_dpg_v3_final/tables/table_01_main_model_comparison.csv   results/final_evidence/bc_dpg_v3_final/tables/table_03_ablation_summary.csv   results/final_evidence/bc_dpg_v3_final/tables/table_06_regularization_selected_by_fold.csv   results/final_evidence/bc_dpg_v3_final/tables/table_09_final_model_decision.csv   results/final_evidence/bc_dpg_v3_final/tables/table_10_claim_boundaries.csv   results/final_evidence/bc_dpg_v3_final/tables/table_12_checkpoint_hashes.csv   results/final_evidence/bc_dpg_v3_final/SHA256SUMS.txt
```

检查：

```bash
unzip -l bc_dpg_v3_final_freeze_acceptance.zip
```

最终上传项目根目录中的：

```text
bc_dpg_v3_final_freeze_acceptance.zip
```

若运行失败，请将完整终端 traceback、`bc_dpg_v3_final_evidence_terminal.log` 以及已经生成的 `results/final_evidence/bc_dpg_v3_final/final_evidence_audit.json` 一并打包返回。
