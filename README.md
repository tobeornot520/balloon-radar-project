# 面向未来空飘球辨识的 H/V 双极化 UAV 检测定位前端研究

本仓库是基于 H/V 双极化雷达 IQ 数据的 UAV 检测、距离-速度定位与背景虚警抑制研究工程。当前成果属于内部开发阶段的检测定位前端，不代表空飘球载荷分类、跨日期/跨场景泛化或严格实时部署结论。

## 当前主线

项目按以下顺序演进：

1. Power2 FCN：使用 H/V 功率距离-多普勒图进行整图候选检测与定位。
2. BC-DPG-FCN v3：样本独立版本是部署导向基准；完整扫描上下文版本是非因果离线性能上限。
3. 显式极化 Stage 3：比较 Power2、RI4、Polar6-gated 和 RI8-gated；结论是 Power2 仍为主检测表征。
4. ROI Stage 4：冻结 Power2 候选位置，仅使用局部 ROI 极化特征进行 suppression-only 精修。
5. 联合审计：对齐 BC-DPG 与 ROI 六折逐样本预测，分析虚警与正确检测的互补性，不在测试集重新选择阈值。
6. 因果上下文敏感性审计：冻结完整模型和阈值，对比 complete-scan、leave-one-out 与假定顺序 past-only 上下文，不把后验窗口结果用于选型。
7. 因果训练就绪审计：检查文件名、MAT 元数据和文件时间，确认当前没有可验证的组内采集顺序；仅开放 validation-only 小样本 smoke。
8. 冻结定位证据：从六折 base-threshold 预测聚合距离/速度误差、条件口径和物理量分层，不重新训练、推理或调阈值。
9. 新数据合同：以 capture、causal、locked-evaluation 三档预检固化真实顺序、同日类别对照、事件时长和外层隔离要求。

冻结结论和研究边界见：

- [当前研究状态](docs/CURRENT_STATUS.md)
- [当前检测数据卡](docs/DATA_CARD.md)
- [评价指标定义](docs/METRIC_DEFINITIONS.md)
- [模型选择台账](docs/MODEL_SELECTION_LEDGER.md)
- [项目结构说明](docs/PROJECT_STRUCTURE.md)
- [项目阶段说明](docs/PROJECT_STAGE_20260719.md)
- [Stage 3 冻结结论](docs/polarimetric_stage3/STAGE3_FROZEN_CONCLUSION.md)
- [Stage 4 预注册](docs/STAGE4_SIXFOLD_PREREGISTRATION.md)
- [ROI 与 BC-DPG 联合设计](docs/ROI_BC_DPG_JOINT_NEXT_DESIGN.md)
- [ROI/BC-DPG 固定阈值联合证据](results/final_evidence/roi_bc_dpg_joint_fixed_threshold/JOINT_AUDIT_REPORT.md)
- [BC-DPG 因果上下文敏感性审计](results/data_audit/bc_dpg_v3_causal_context_audit/CAUSAL_CONTEXT_AUDIT.md)
- [采集顺序审计](results/data_audit/detection_acquisition_order/ACQUISITION_ORDER_AUDIT.md)
- [BC-DPG 因果训练协议](docs/BC_DPG_CAUSAL_TRAINING_PROTOCOL.md)
- [BC-DPG 冻结定位证据](results/final_evidence/bc_dpg_localization/LOCALIZATION_EVIDENCE_REPORT.md)
- [新数据采集与锁定评价协议](docs/NEW_DATA_COLLECTION_PROTOCOL.md)
- [当前数据合同缺口报告](results/data_audit/data_collection_readiness_v1/PRECHECK_REPORT.md)
- [项目对外分享材料](docs/share/README_SHARE_ZH.md)

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
生成资产同时报告 pooled、macro、median/IQR、worst-fold、Wilson 区间、扫描组
bootstrap 和 McNemar 配对诊断。AND/OR 结果只作为测试后诊断统计，不代表已训练或
已选定的联合模型。

冻结完整 BC-DPG checkpoint，在不训练、不改阈值的条件下重放 complete-scan、
leave-one-out 和假定 beam/azimuth 顺序的 past-only 上下文：

```bash
python scripts/audit_bc_dpg_v3_causal_context.py --overwrite
```

正式审计位于 `results/data_audit/bc_dpg_v3_causal_context_audit/`。其中 past-only
结果属于完整模型的上下文替换敏感性，不是经过因果上下文训练和验证集选型的新模型。

检查真实采集顺序是否足以支持因果训练：

```bash
python scripts/audit_detection_acquisition_order.py --overwrite
```

当前正式因果训练门禁关闭。开发接口可运行一个受限、validation-only 的小样本 smoke：

```bash
python scripts/run_bc_dpg_causal_smoke.py
```

该 smoke 使用未经时间戳验证的 beam/azimuth 推断顺序，不加载测试 split，结果不用于模型或窗口选择。

从六折冻结 base-threshold 预测构建距离-速度定位汇总、图件和 SHA256 清单：

```bash
python scripts/build_bc_dpg_localization_evidence.py --overwrite
```

该构建器会逐折验证 raw DPG 与 BC-DPG 的定位坐标一致，只输出聚合证据，不复制逐样本预测。

新数据落盘后的第一入口是清单合同预检：

```bash
python scripts/validate_data_collection_manifest.py \
  path/to/collection_manifest.csv \
  --profile capture \
  --output-dir results/data_audit/new_collection_capture
```

采集完整性通过后再依次使用 `causal` 和 `locked_evaluation`。空白模板位于 `configs/data_collection_manifest_template_v1.csv`；旧 `new_split=test` 不得直接改名为锁定测试。

生成不含原始数据、权重、逐样本预测和开发聊天记录的对外分享包：

```bash
python scripts/build_project_share_package.py
```

目录版和 ZIP 默认生成在 `dist/`，该目录属于本地分发产物，不进入 Git。分享包是
可追溯、可校验的冻结结果摘录；完整复现仍需要内部代码、数据、逐样本预测和
checkpoint。

## 实验纪律

- 阈值、模型选择和后处理规则只能由训练集或验证集确定。
- smoke 结果只验证接口，不作为性能结论。
- 不修改已冻结 checkpoint，不用测试集重新调阈值。
- 样本独立实验与扫描上下文实验必须分别报告。
- 完整扫描上下文可能使用未来样本，只能表述为离线扫描感知上限。
- 同时报告 pooled、折间分布和最差折；当前 56 个 BC-DPG 虚警全部集中于 Fold 1/4。
- Stage 4 的 Fold 1/4 参与过模式筛选，最终六折 ROI 汇总不是独立盲测估计。
- 当前目标和背景来自不同采集日期，不能声称跨日期泛化。
- past-only 顺序目前由 beam、azimuth 和 sample ID 推断，不能替代采集时间戳。
- `results/experiments/`、原始数据、大权重、原始聊天记录和生成分发包不应提交到 Git。
- 当前分类数据只覆盖 UAV，长期空飘球、载荷和状态分类仍需要补充真实数据。

## 版本说明

带明确版本后缀的文件是当前可信入口，例如 `*_v1.py`、`*_v2.py` 和 `*_v3.py`。无版本后缀的早期 ROI 骨架以及 `_cleanup_archive/`、`payload/` 中的副本只用于历史追踪。
