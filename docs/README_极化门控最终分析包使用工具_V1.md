# 极化门控最终分析包使用工具 V1

## 这个分析包是什么

`polarimetric_twofold_diagnostics_final_analysis_v1.zip` 是结果与证据归档包，不是模型训练补丁。它用于：

- 冻结 Stage 3 的正式结论；
- 保存论文可复核表格；
- 为候选区域引导的极化精修确定下一阶段接口；
- 防止后续把旧结果与新实验混淆。

它不会直接改变模型性能，也不应被解压到 `models/` 或 `training/`。

## 安装后的位置

```text
results/data_audit/polarimetric_twofold_diagnostics_final_analysis_v1/
docs/polarimetric_stage3/STAGE3_FROZEN_CONCLUSION.md
docs/polarimetric_stage3/NEXT_STAGE_ROI_POLARIMETRIC_INTERFACE.md
```

## 运行方式

将以下两个ZIP放入项目根目录：

```text
polarimetric_twofold_diagnostics_final_analysis_v1.zip
BC_DPG_polarimetric_final_analysis_usage_tools_v1.zip
```

先安装工具，再运行：

```bash
cd ~/projects/balloon_radar_project
unzip -o BC_DPG_polarimetric_final_analysis_usage_tools_v1.zip -d .
python apply_polarimetric_final_analysis_usage_tools_v1.py

set -o pipefail
python scripts/use_polarimetric_final_analysis_v1.py \
  --analysis-zip polarimetric_twofold_diagnostics_final_analysis_v1.zip \
  2>&1 | tee polarimetric_final_analysis_usage_terminal_v1.log
```

正常状态：

```text
required : 11
found    : 11
status   : PASS
```

自动生成验收包：

```text
polarimetric_final_analysis_usage_acceptance_v1.zip
```

## 只重新校验

分析包已经安装后，可运行：

```bash
python scripts/use_polarimetric_final_analysis_v1.py --verify-only
```

## 注意

- 如果目标目录已经存在，脚本会先移动到 `backups/`，不会直接覆盖。
- 不会修改 checkpoint、模型代码、数据 manifest 或 BC-DPG-FCN v3。
- 该包用于归档和决策，不会启动下一阶段训练。
