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

公开 LAT-MRICD 原始数据同样不包含在分享包中。按官方来源下载并放到项目约定目录后，先运行：

```bash
python scripts/audit_lat_mricd_dataset_v1.py --overwrite
```

该数据禁止随机拆行；第一轮只放行 batch-grouped 的 HRRP/归一化微动大类基线。

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
python scripts/build_bc_dpg_localization_evidence.py --overwrite
python scripts/build_project_share_package.py --overwrite
```

冻结 BC-DPG 上下文敏感性审计：

```bash
python scripts/audit_bc_dpg_v3_causal_context.py --overwrite
```

该命令不训练模型或重选阈值；正式结果中的 leave-one-out 与 past-only 行都是冻结测试后的上下文替换诊断。

检查采集顺序并运行受限接口 smoke：

```bash
python scripts/audit_detection_acquisition_order.py --overwrite
python scripts/run_bc_dpg_causal_smoke.py
```

当前审计结论是正式训练门禁关闭。smoke 只加载 train/val 小样本，不读取测试 split，也不提供性能或窗口选择证据。

新数据采集完成后先运行分级合同预检：

```bash
python scripts/validate_data_collection_manifest.py \
  path/to/collection_manifest.csv \
  --profile capture \
  --output-dir results/data_audit/new_collection_capture
```

只有 `causal` 报告打开因果训练门禁后才能正式训练；只有 `locked_evaluation` 报告通过后才能进入外部锁定评价。

## 4. 当前证据与治理文档

- BC-DPG v3：`results/final_evidence/bc_dpg_v3_final/`
- Stage 3：`docs/polarimetric_stage3/STAGE3_FROZEN_CONCLUSION.md`
- Stage 4：`results/data_audit/roi_stage4_selected_sixfold_v1/`
- 最终联合审计：`results/data_audit/final_roi_bc_dpg_joint_v2_base_threshold/`
- 联合证据资产：`results/final_evidence/roi_bc_dpg_joint_fixed_threshold/`
- BC-DPG 因果上下文敏感性审计：`results/data_audit/bc_dpg_v3_causal_context_audit/`
- 采集顺序就绪审计：`results/data_audit/detection_acquisition_order/`
- BC-DPG 冻结定位证据：`results/final_evidence/bc_dpg_localization/`
- 新数据采集协议：`docs/NEW_DATA_COLLECTION_PROTOCOL.md`
- 当前数据合同缺口：`results/data_audit/data_collection_readiness_v1/`
- 数据卡：`docs/DATA_CARD.md`
- 指标定义：`docs/METRIC_DEFINITIONS.md`
- 模型选择台账：`docs/MODEL_SELECTION_LEDGER.md`
- 外部公开数据审计：`docs/EXTERNAL_PUBLIC_DATA_AUDIT_20260803.md`

早期 `final_roi_bc_dpg_joint` 使用了错误的 BC 判决来源，已经移出活动证据，不得引用。

## 5. 下一阶段优先级

### 优先级 1：同日多类数据与锁定外部评价

- 每个日期和场地同时采集目标与无目标背景；
- 增加空飘球、带载空飘球、鸟类和不同背景类型；
- 以日期、场地或飞行架次为外层隔离单位；
- 在盲测前冻结结构、阈值、后处理、主指标和失败判据。
- 使用 v1 空白模板采集全部 40 个字段，并在数据进入训练前通过三档合同预检。

### 优先级 2：BC-DPG 因果在线化

冻结 checkpoint 的 leave-one-out 和 past-only 敏感性审计已经完成。leave-one-out 为 54/830 个虚警、289/318 个正确检测，但仍使用未来样本；past-only all-history 为 93/830、288/318，但顺序由 `(beam_layer, azimuth_deg, sample_id)` 推断，未由时间戳验证。两者都不是重新训练的因果模型，也不得用于从测试集选择 all-history 窗口。

下一步应补齐逐样本真实时间戳，按因果上下文重新训练 BC-DPG，只在训练/验证集比较冷启动处理和历史窗口，然后用锁定外部测试集评价一次。完整扫描结果继续只作为离线上限，样本独立 BC 继续作为在线导向参照。

### 优先级 3：部署级虚警与连续物理定位指标

冻结网格定位证据已经补齐全部目标、过阈值目标、联合成功目标的距离/速度 MAE、中位数、P90 和分层结果。下一步需补齐时间戳、扫描时长和事件边界，报告每扫描、每小时和事件级虚警；同时增加 SNR、环境分层、雷达标定信息和未量化连续物理真值，才能评价真实测距测速精度。

### 优先级 4：嵌套选择与学习型联合模型

使用嵌套扫描组交叉验证，或独立开发集选择 BC 分数、ROI 分数、背景状态和质量指标的门控/融合规则；外层测试只评价一次。

### 优先级 5：多域细粒度分类

在真实空飘球数据齐备后，引入长慢时间时频/微多普勒、极化、轨迹和行为特征，逐级开展目标类别、有载/无载、载荷类型和运动状态识别。

在等待新数据期间，LAT-MRICD 可作为独立支线先完成 Narrow-X、HRRP-X 的 batch-grouped
UAV/鸟/气象特征基线和 band-held-out 迁移。它不与当前 H/V 六折结果合并计分，也不解除
极化、PRF、连续时序和空飘球标签门禁。

## 6. 继续开发时必须保持的纪律

- 不依据外层测试结果重新选择阈值、容差、模型或组合逻辑；
- 不覆盖冻结 checkpoint 和正式证据目录；
- smoke 结果只验证接口，不作为性能结论；
- 两折筛选、六折内部开发评价和外部盲测必须分开报告；
- 样本独立、因果上下文和完整扫描离线模型必须分开命名；
- 不用当前 4/16/64/all-history 测试结果选择下一版因果历史窗口；
- 同时报告 pooled、macro、median/IQR 和 worst-fold 指标；
- 新结论必须附带数据范围、划分方式、选择来源、指标定义和完整复现清单。
