# Balloon Radar Project

本仓库是基于 H/V 双极化雷达 IQ 数据的 UAV 检测、距离-速度定位与背景虚警抑制研究工程。当前成果属于检测定位前端，不代表空飘球载荷分类或跨场景泛化结论。

## 当前主线

项目按以下顺序演进：

1. Power2 FCN：使用 H/V 功率距离-多普勒图进行整图候选检测与定位。
2. BC-DPG-FCN v3：在冻结 DPG 输出上使用扫描背景上下文进行目标保护的分数校准。
3. 显式极化 Stage 3：比较 Power2、RI4、Polar6-gated 和 RI8-gated；结论是 Power2 仍为主检测表征。
4. ROI Stage 4：冻结 Power2 候选位置，仅使用局部 ROI 极化特征进行 suppression-only 精修。
5. 联合审计：对齐 BC-DPG 与 ROI 六折逐样本预测，分析虚警与正确检测的互补性，不在测试集重新选择阈值。

冻结结论和研究边界见：

- [当前研究状态](docs/CURRENT_STATUS.md)
- [项目结构说明](docs/PROJECT_STRUCTURE.md)
- [项目阶段说明](docs/PROJECT_STAGE_20260719.md)
- [Stage 3 冻结结论](docs/polarimetric_stage3/STAGE3_FROZEN_CONCLUSION.md)
- [Stage 4 预注册](docs/STAGE4_SIXFOLD_PREREGISTRATION.md)
- [ROI 与 BC-DPG 联合设计](docs/ROI_BC_DPG_JOINT_NEXT_DESIGN.md)
- [ROI/BC-DPG 固定阈值联合证据](results/final_evidence/roi_bc_dpg_joint_fixed_threshold/JOINT_AUDIT_REPORT.md)

## 目录结构

| 目录 | 职责 |
|---|---|
| `data/` | 原始 MAT、标签与数据清单；`data/raw/` 不进入 Git |
| `datasets/` | V2/V3、极化和 ROI 数据接口 |
| `features/` | RD、复数极化、门控和 ROI 特征构造 |
| `models/` | FCN、双分支、BC-DPG 和 ROI refiner |
| `training/` | 正式训练实现 |
| `scripts/` | 实验编排、审计、汇总、绘图和打包入口 |
| `configs/` | YAML/JSON 实验配置与冻结参数 |
| `evaluation/` | CFAR、指标分析和报告生成 |
| `results/data_audit/` | 数据划分、审计表和冻结证据 |
| `results/final_evidence/` | 论文用冻结结果、表格和图片 |
| `results/experiments/` | checkpoint、逐样本预测和训练输出 |
| `docs/` | 阶段结论、预注册与使用说明 |
| `tools/` | 数据布局、诊断和恢复辅助工具 |
| `notes/` | 本地聊天记录与草稿；原始内容不进入 Git |
| `_cleanup_archive/` | 本地历史清理与恢复材料，不属于活动源码 |
| `payload/` | 历史补丁交付内容，不作为运行入口 |

`losses/`、`metrics/`、`postprocess/` 和 `radar_processing/` 目前是保留目录，不是独立实现模块。

## 环境

正式环境定义在 `environment.yml`，环境名为 `radar-torch`，Python 版本为 3.11。

```bash
conda env create -f environment.yml
conda activate radar-torch
```

首次运行或合并补丁后，先执行不依赖 GPU 的结构检查：

```bash
python scripts/check_project_health.py
```

自动化单元测试只从 `tests/` 收集；`scripts/test_*.py` 是需要显式运行的历史 smoke 工具。

```bash
python -m pytest
```

## 主要入口

六折 BC-DPG v3：

```bash
python scripts/run_bc_dpg_v3.py --help
```

显式极化表征：

```bash
python scripts/run_polarimetric_representation_benchmark_v2.py --help
```

ROI Stage 4 六折：

```bash
python scripts/run_roi_stage4_selected_sixfold_v1.py --help
```

重新构建 BC-DPG 与 ROI 联合审计时，建议先写入新目录并与冻结输出比较：

```bash
python scripts/build_final_roi_bc_dpg_joint_audit.py \
  --output-dir results/data_audit/final_roi_bc_dpg_joint_rebuild
```

脚本默认拒绝覆盖非空输出目录。确认结果后才能显式使用 `--overwrite`。

当前有效的联合审计是
`results/data_audit/final_roi_bc_dpg_joint_v2_base_threshold/`。早期同名输出使用了
错误的 BC 判决来源，已移入本地恢复归档，不应用于论文或后续选型。

从冻结联合审计生成正式论文表格、PNG/PDF 图件、报告和 SHA256 清单：

```bash
python scripts/build_roi_bc_dpg_joint_paper_assets.py
```

正式输出位于 `results/final_evidence/roi_bc_dpg_joint_fixed_threshold/`。脚本会在
写入前复核 1,148 行逐样本对齐、六折来源和固定阈值状态，并默认拒绝覆盖非空目录。
AND/OR 结果只作为诊断统计，不代表已训练或已选定的联合模型。

## 实验纪律

- 阈值、模型选择和后处理规则只能由训练集或验证集确定。
- smoke 结果只验证接口，不作为性能结论。
- 不修改已冻结 checkpoint，不用测试集重新调阈值。
- 样本独立实验与扫描上下文实验必须分别报告。
- `results/experiments/`、原始数据、大权重、原始聊天记录和生成分发包不应提交到 Git。
- 当前分类数据只覆盖 UAV，长期空飘球、载荷和状态分类仍需要补充真实数据。

## 版本说明

带明确版本后缀的文件是当前可信入口，例如 `*_v1.py`、`*_v2.py` 和 `*_v3.py`。无版本后缀的早期 ROI 骨架以及 `_cleanup_archive/`、`payload/` 中的副本只用于历史追踪。
