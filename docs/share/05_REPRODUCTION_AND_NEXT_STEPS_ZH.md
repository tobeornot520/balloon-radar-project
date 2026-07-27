# 复核入口与后续计划

## 1. 分享包能做什么

本分享包是可追溯、可校验的冻结结果摘录，不包含完整源码、原始数据、逐样本预测或模型权重。阅读 Markdown、CSV 和图片不需要安装 Python 环境。

`MANIFEST.json` 记录分享版本、来源提交、证据角色和文件哈希；`SHA256SUMS.txt` 用于检查文件是否在传输中损坏。哈希校验不等同于重新运行指标计算。完整复现需要内部代码、数据、冻结预测和 checkpoint。

## 2. 完整仓库环境

内部工程使用 Conda 环境 `radar-torch`，Python 3.11。仓库根目录提供 `environment.yml` 和锁定依赖文件。

```bash
conda env create -f environment.yml
conda activate radar-torch
python scripts/check_project_health.py --require-joint-inputs
python -m pytest
```

原始数据、checkpoint 和大规模训练输出不在 Git 中，需要由数据持有方按项目目录约定单独准备。

## 3. 主要内部入口

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

重新构建联合审计时应写入新目录，不直接覆盖冻结证据：

```bash
python scripts/build_final_roi_bc_dpg_joint_audit.py \
  --output-dir results/data_audit/final_roi_bc_dpg_joint_rebuild
```

从冻结预测重建证据资产与脱敏分享包：

```bash
python scripts/build_roi_bc_dpg_joint_paper_assets.py --overwrite
python scripts/build_project_share_package.py --overwrite
```

## 4. 当前证据与治理文档

- BC-DPG v3：`results/final_evidence/bc_dpg_v3_final/`
- Stage 3：`docs/polarimetric_stage3/STAGE3_FROZEN_CONCLUSION.md`
- Stage 4：`results/data_audit/roi_stage4_selected_sixfold_v1/`
- 最终联合审计：`results/data_audit/final_roi_bc_dpg_joint_v2_base_threshold/`
- 联合证据资产：`results/final_evidence/roi_bc_dpg_joint_fixed_threshold/`
- 数据卡：`docs/DATA_CARD.md`
- 指标定义：`docs/METRIC_DEFINITIONS.md`
- 模型选择台账：`docs/MODEL_SELECTION_LEDGER.md`

早期 `final_roi_bc_dpg_joint` 使用了错误的 BC 判决来源，已经移出活动证据，不得引用。

## 5. 下一阶段优先级

### 优先级 1：同日多类数据与锁定外部评价

- 每个日期和场地同时采集目标与无目标背景；
- 增加空飘球、带载空飘球、鸟类和不同背景类型；
- 以日期、场地或飞行架次为外层隔离单位；
- 在盲测前冻结结构、阈值、后处理、主指标和失败判据。

### 优先级 2：BC-DPG 因果在线化

分别评估样本独立、leave-one-sample-out、past-only 不同历史窗口和完整扫描离线上限，禁止把完整扫描结果混写成在线性能。

### 优先级 3：部署级虚警与定位指标

补齐时间戳、扫描时长和事件边界，报告每扫描、每小时和事件级虚警；同时补充距离/速度 MAE、中位数、90% 分位数和分层结果。

### 优先级 4：嵌套选择与学习型联合模型

使用嵌套扫描组交叉验证，或独立开发集选择 BC 分数、ROI 分数、背景状态和质量指标的门控/融合规则；外层测试只评价一次。

### 优先级 5：多域细粒度分类

在真实空飘球数据齐备后，引入长慢时间时频/微多普勒、极化、轨迹和行为特征，逐级开展目标类别、有载/无载、载荷类型和运动状态识别。

## 6. 继续开发时必须保持的纪律

- 不依据外层测试结果重新选择阈值、容差、模型或组合逻辑；
- 不覆盖冻结 checkpoint 和正式证据目录；
- smoke 结果只验证接口，不作为性能结论；
- 两折筛选、六折内部开发评价和外部盲测必须分开报告；
- 样本独立、因果上下文和完整扫描离线模型必须分开命名；
- 同时报告 pooled、macro、median/IQR 和 worst-fold 指标；
- 新结论必须附带数据范围、划分方式、选择来源、指标定义和完整复现清单。
