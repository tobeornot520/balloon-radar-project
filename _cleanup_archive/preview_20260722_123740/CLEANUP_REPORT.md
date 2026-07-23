# Balloon Radar Project 根目录清理报告

- 模式：`preview`
- 候选项目数：**66**
- 候选总大小：**4.03 MB**
- 项目根目录：`/home/tobeornot8259748/projects/balloon_radar_project`

## 分类统计

| 分类 | 数量 | 大小 |
|---|---:|---:|
| acceptance_zip | 10 | 497.75 KB |
| cache_dir | 11 | 1.65 MB |
| evidence_zip | 1 | 290.93 KB |
| installer | 6 | 401.88 KB |
| legacy_audit_dir | 3 | 99.07 KB |
| legacy_diagnostic_dir | 1 | 27.31 KB |
| legacy_package_dir | 6 | 200.45 KB |
| legacy_paper_dir | 2 | 74.61 KB |
| legacy_patch_dir | 9 | 306.22 KB |
| patch_zip | 4 | 86.70 KB |
| stage_readme | 4 | 8.26 KB |
| terminal_log | 9 | 447.09 KB |

## 永久保留范围

以下顶层目录不会被自动移动或删除：

```text
.git
.github
.vscode
_cleanup_archive
backups
baselines
checkpoints
configs
data
datasets
dist
docs
evaluation
features
logs
losses
metrics
models
notebooks
results
scripts
training
```

## 候选清单

| 路径 | 类型 | 分类 | 建议 | 大小 | 原因 |
|---|---|---|---|---:|---|
| `BC_DPG_FCN_v1_argmax_fix` | directory | legacy_patch_dir | archive | 10.91 KB | 早期 BC-DPG-FCN v1 补丁目录，功能已集成 |
| `BC_DPG_FCN_v1_integration_patch` | directory | legacy_patch_dir | archive | 22.12 KB | 早期 BC-DPG-FCN v1 补丁目录，功能已集成 |
| `BC_DPG_FCN_v1_preflight_patch` | directory | legacy_patch_dir | archive | 20.99 KB | 早期 BC-DPG-FCN v1 补丁目录，功能已集成 |
| `BC_DPG_FCN_v1_training_patch` | directory | legacy_patch_dir | archive | 66.63 KB | 早期 BC-DPG-FCN v1 补丁目录，功能已集成 |
| `BC_DPG_FCN_v2_tail_patch` | directory | legacy_patch_dir | archive | 51.16 KB | 早期 BC-DPG-FCN v2 补丁目录，功能已集成 |
| `BC_DPG_FCN_v3_scan_target_patch` | directory | legacy_patch_dir | archive | 60.59 KB | v3 扫描目标补丁目录，功能已集成 |
| `BC_DPG_current_model_stage2_patch_v2.zip` | file | patch_zip | archive | 20.56 KB | 已安装的 Stage 2 补丁压缩包 |
| `BC_DPG_explicit_polarimetric_representation_stage2_patch_v1.zip` | file | patch_zip | archive | 22.32 KB | 已安装的极化 Stage 2 补丁压缩包 |
| `BC_DPG_overfitting_audit_fix` | directory | legacy_audit_dir | archive | 32.84 KB | 旧过拟合审计补丁目录 |
| `BC_DPG_overfitting_audit_patch` | directory | legacy_audit_dir | archive | 32.64 KB | 旧过拟合审计补丁目录 |
| `BC_DPG_overfitting_audit_rotation_fix` | directory | legacy_audit_dir | archive | 33.59 KB | 旧过拟合审计补丁目录 |
| `BC_DPG_polarimetric_gated_diagnostic_stage3_patch_v1.zip` | file | patch_zip | archive | 26.33 KB | 已安装的极化 Stage 3 补丁压缩包 |
| `BC_DPG_v2_compare_fix` | directory | legacy_patch_dir | archive | 6.06 KB | 旧 v2 比较/阈值补丁目录 |
| `BC_DPG_v2_global_threshold_patch` | directory | legacy_patch_dir | archive | 21.14 KB | 旧 v2 比较/阈值补丁目录 |
| `BC_DPG_v3_final_freeze_paper_evidence_patch_v1.zip` | file | patch_zip | archive | 17.49 KB | 已安装的 v3 最终冻结补丁压缩包 |
| `BC_DPG_v3_paper_results_matplotlib_fix` | directory | legacy_paper_dir | archive | 36.84 KB | 旧论文结果生成补丁目录 |
| `BC_DPG_v3_paper_results_patch` | directory | legacy_paper_dir | archive | 37.77 KB | 旧论文结果生成补丁目录 |
| `README_BC_DPG_v3最终冻结与论文证据_V1.md` | file | stage_readme | archive | 2.89 KB | 阶段安装说明，正式报告已在 results/final_evidence |
| `README_BC_DPG当前模型Stage2_V2.md` | file | stage_readme | archive | 3.52 KB | 阶段安装说明 |
| `README_显式极化表征基准Stage2.md` | file | stage_readme | archive | 1.08 KB | 阶段安装说明，正式文档已在项目目录内 |
| `README_显式极化门控与分数迁移诊断_v2.md` | file | stage_readme | archive | 791.00 B | 阶段安装说明，正式文档已在项目目录内 |
| `apply_bc_dpg_current_model_stage2_v2.py` | file | installer | archive | 62.54 KB | 已执行的当前模型 Stage 2 安装器 |
| `apply_bc_dpg_v3_ablation_polar_stage1.py` | file | installer | archive | 128.01 KB | 已执行的 v3 消融安装器 |
| `apply_bc_dpg_v3_ablation_runner_fix_v2.py` | file | installer | archive | 17.73 KB | 已执行的 v3 消融安装器 |
| `apply_bc_dpg_v3_final_freeze_paper_evidence_v1.py` | file | installer | archive | 42.81 KB | 已执行的最终冻结安装器 |
| `apply_polarimetric_gated_representation_v2.py` | file | installer | archive | 84.56 KB | 已执行的极化门控安装器 |
| `apply_polarimetric_representation_benchmark_v1.py` | file | installer | archive | 66.23 KB | 已执行的极化表征安装器 |
| `baselines/__pycache__` | directory | cache_dir | delete_cache | 7.11 KB | Python/测试/Notebook 可再生缓存 |
| `bc_dpg_current_model_stage2_smoke_acceptance.zip` | file | acceptance_zip | archive | 13.74 KB | Stage 2 验收包，结果已进入 results |
| `bc_dpg_current_model_stage2_smoke_terminal.log` | file | terminal_log | archive | 27.38 KB | Stage 2 根目录过程日志 |
| `bc_dpg_v31_shift_reg_formal_acceptance.zip` | file | acceptance_zip | archive | 13.50 KB | 正则扫描验收包，结果已进入 results |
| `bc_dpg_v31_shift_reg_formal_acceptance_v2.zip` | file | acceptance_zip | archive | 19.51 KB | 正则扫描验收包，结果已进入 results |
| `bc_dpg_v31_shift_reg_formal_terminal.log` | file | terminal_log | archive | 142.32 KB | 正则扫描根目录过程日志 |
| `bc_dpg_v3_ablation_extended_formal_acceptance.zip` | file | acceptance_zip | archive | 11.67 KB | v3 消融验收包，结果已进入 results |
| `bc_dpg_v3_ablation_extended_formal_terminal.log` | file | terminal_log | archive | 52.59 KB | v3 消融根目录过程日志 |
| `bc_dpg_v3_ablation_formal_acceptance.zip` | file | acceptance_zip | archive | 3.97 KB | v3 消融验收包，结果已进入 results |
| `bc_dpg_v3_ablation_formal_terminal.log` | file | terminal_log | archive | 4.81 KB | v3 消融根目录过程日志 |
| `bc_dpg_v3_ablation_sixfold_formal_acceptance.zip` | file | acceptance_zip | archive | 18.39 KB | v3 消融验收包，结果已进入 results |
| `bc_dpg_v3_ablation_sixfold_formal_terminal.log` | file | terminal_log | archive | 110.69 KB | v3 消融根目录过程日志 |
| `bc_dpg_v3_final_evidence.zip` | file | evidence_zip | archive | 290.93 KB | 根目录证据副本，正式文件保存在 results/final_evidence |
| `bc_dpg_v3_final_evidence_terminal.log` | file | terminal_log | archive | 685.00 B | 最终证据根目录过程日志 |
| `bc_dpg_v3_final_freeze_acceptance.zip` | file | acceptance_zip | archive | 302.24 KB | 最终冻结验收包 |
| `current_structure_reader_package` | directory | legacy_package_dir | archive | 25.56 KB | 一次性项目结构读取包 |
| `datasets/__pycache__` | directory | cache_dir | delete_cache | 77.01 KB | Python/测试/Notebook 可再生缓存 |
| `detection_ablation_analysis_v2_package` | directory | legacy_package_dir | archive | 19.04 KB | 旧检测消融分析包 |
| `detection_diagnostics_v3_package` | directory | legacy_package_dir | archive | 32.16 KB | 旧检测诊断包 |
| `detection_group_split_v1_package` | directory | legacy_package_dir | archive | 64.94 KB | 旧分组划分包 |
| `detection_group_split_v1_package/scripts/__pycache__` | directory | cache_dir | delete_cache | 36.16 KB | Python/测试/Notebook 可再生缓存 |
| `detection_visualization_hotfix_v3_1` | directory | legacy_package_dir | archive | 16.36 KB | 旧可视化热修复目录 |
| `dpg_fcn_v1_patch` | directory | legacy_patch_dir | archive | 46.63 KB | 早期 DPG-FCN v1 补丁目录 |
| `evaluation/__pycache__` | directory | cache_dir | delete_cache | 142.00 B | Python/测试/Notebook 可再生缓存 |
| `features/__pycache__` | directory | cache_dir | delete_cache | 21.73 KB | Python/测试/Notebook 可再生缓存 |
| `full_detection_baseline_v2_package` | directory | legacy_package_dir | archive | 42.39 KB | 旧完整检测基线包 |
| `hv_late_fusion_diagnostic_v1` | directory | legacy_diagnostic_dir | archive | 27.31 KB | 旧 H/V 后融合诊断目录 |
| `models/__pycache__` | directory | cache_dir | delete_cache | 75.42 KB | Python/测试/Notebook 可再生缓存 |
| `polarimetric_gated_stage3_smoke_acceptance.zip` | file | acceptance_zip | archive | 92.05 KB | 极化 Stage 3 smoke 验收包 |
| `polarimetric_gated_stage3_smoke_terminal.log` | file | terminal_log | archive | 19.72 KB | 极化 Stage 3 根目录过程日志 |
| `polarimetric_representation_stage2_smoke_acceptance.zip` | file | acceptance_zip | archive | 6.77 KB | 极化 Stage 2 smoke 验收包 |
| `polarimetric_representation_stage2_smoke_terminal.log` | file | terminal_log | archive | 16.80 KB | 极化 Stage 2 根目录过程日志 |
| `polarimetric_representation_twofold_formal_acceptance.zip` | file | acceptance_zip | archive | 15.92 KB | 极化两折正式验收包 |
| `polarimetric_representation_twofold_formal_terminal.log` | file | terminal_log | archive | 72.12 KB | 极化两折正式根目录过程日志 |
| `radar_data_reader_package/datasets/__pycache__` | directory | cache_dir | delete_cache | 22.18 KB | Python/测试/Notebook 可再生缓存 |
| `radar_data_reader_package/scripts/__pycache__` | directory | cache_dir | delete_cache | 15.07 KB | Python/测试/Notebook 可再生缓存 |
| `scripts/__pycache__` | directory | cache_dir | delete_cache | 1.12 MB | Python/测试/Notebook 可再生缓存 |
| `training/__pycache__` | directory | cache_dir | delete_cache | 281.75 KB | Python/测试/Notebook 可再生缓存 |
| `utils/__pycache__` | directory | cache_dir | delete_cache | 2.13 KB | Python/测试/Notebook 可再生缓存 |
