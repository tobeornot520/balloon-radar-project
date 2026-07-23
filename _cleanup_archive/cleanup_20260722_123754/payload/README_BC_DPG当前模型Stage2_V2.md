# BC-DPG-FCN 当前模型 Stage 2（V2）

本补丁只推进当前 BC-DPG-FCN v3 研究线，不把显式极化、时频/微多普勒或长期历史记忆并入当前模型。

## 本阶段目标

1. 冻结当前 v3 六折证据，记录源码、配置、manifest 和 summary 的 SHA256；
2. 将当前模型按部署条件拆分为：
   - 原始 DPG：样本独立；
   - `no_scan_context`：样本独立校准器；
   - `full`：使用完整扫描组统计的离线 scan-aware 校准器；
3. 对 `shift_regularization` 做候选扫描；
4. 每折只使用验证集指标选择权重，选择完成后才读取对应测试指标。

## 新增脚本

- `scripts/build_bc_dpg_v3_deployment_comparison.py`
- `scripts/run_bc_dpg_v31_shift_reg_sweep.py`
- `scripts/select_bc_dpg_v31_shift_reg.py`
- `scripts/freeze_bc_dpg_v3_evidence.py`

## 安装

将 `apply_bc_dpg_current_model_stage2_v2.py` 放到项目根目录后执行：

```bash
cd ~/projects/balloon_radar_project
python apply_bc_dpg_current_model_stage2_v2.py
```

## 第一轮：部署对照、证据冻结与困难折 smoke

```bash
cd ~/projects/balloon_radar_project
set -o pipefail

{
  echo "===== deployment comparison ====="
  python scripts/build_bc_dpg_v3_deployment_comparison.py --require-all

  echo "===== evidence freeze ====="
  python scripts/freeze_bc_dpg_v3_evidence.py

  echo "===== shift regularization smoke sweep ====="
  python scripts/run_bc_dpg_v31_shift_reg_sweep.py     --folds 1 4     --regularizations 0.01 0.005 0.0025 0.001 0     --smoke
} 2>&1 | tee bc_dpg_current_model_stage2_smoke_terminal.log
```

预期权重扫描最后应显示：

```text
candidate rows  : 10
selected folds  : 2
missing         : 0
```

smoke 只做接口验收，不形成论文结论。

## 验收材料一键打包

```bash
cd ~/projects/balloon_radar_project

FREEZE_DIR=$(find results/model_freeze -maxdepth 1 -type d   -name 'bc_dpg_v3_freeze_*' | sort | tail -n 1)

test -n "$FREEZE_DIR" || { echo "未找到 freeze 目录"; exit 1; }

zip -j bc_dpg_current_model_stage2_smoke_acceptance.zip   bc_dpg_current_model_stage2_smoke_terminal.log   results/data_audit/bc_dpg_v3_deployment_comparison/deployment_comparison_detail.csv   results/data_audit/bc_dpg_v3_deployment_comparison/deployment_comparison_aggregate.csv   results/data_audit/bc_dpg_v3_deployment_comparison/README_deployment_comparison.md   results/data_audit/bc_dpg_v3_deployment_comparison/deployment_comparison_audit.json   results/data_audit/bc_dpg_v31_shift_reg/candidate_validation_metrics_smoke.csv   results/data_audit/bc_dpg_v31_shift_reg/selected_by_fold_smoke.csv   results/data_audit/bc_dpg_v31_shift_reg/selected_test_metrics_smoke.csv   results/data_audit/bc_dpg_v31_shift_reg/aggregate_selected_test_metrics_smoke.csv   results/data_audit/bc_dpg_v31_shift_reg/selection_audit_smoke.json   results/data_audit/bc_dpg_v31_shift_reg/README_shift_regularization_selection_smoke.md   results/data_audit/bc_dpg_v31_shift_reg/latest_run_plan.json   results/data_audit/bc_dpg_v31_shift_reg/latest_run_status.json   "$FREEZE_DIR/freeze_manifest.json"   "$FREEZE_DIR/SHA256SUMS.txt"   "$FREEZE_DIR/README_freeze.md"

unzip -l bc_dpg_current_model_stage2_smoke_acceptance.zip
```

最终只上传：

```text
bc_dpg_current_model_stage2_smoke_acceptance.zip
```

## 研究边界

- 当前 `full` 使用完整扫描组统计，只能解释为离线 scan-aware 校准器；
- `no_scan_context` 是当前样本独立校准对照；
- 权重扫描仍属于现有数据上的内部开发验证；
- 未来自主采集的新日期、新环境盲测不可替代。
