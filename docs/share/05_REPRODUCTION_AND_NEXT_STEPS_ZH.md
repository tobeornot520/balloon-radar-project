# 复现入口与后续计划

## 1. 分享包的使用方式

本分享包是说明和证据摘录，不包含完整源码、原始数据或模型权重。阅读 Markdown、CSV 和图片不需要安装 Python 环境。

`MANIFEST.json` 记录分享版本、来源提交和文件哈希；`SHA256SUMS.txt` 可用于检查文件是否在传输中损坏。

## 2. 完整仓库环境

完整工程使用 Conda 环境 `radar-torch`，Python 3.11。仓库根目录提供 `environment.yml` 和锁定依赖文件。

```bash
conda env create -f environment.yml
conda activate radar-torch
python scripts/check_project_health.py
python -m pytest
```

原始数据、checkpoint 和大规模训练输出不在 Git 中，需要由数据持有方按项目目录约定单独准备。

## 3. 主要入口

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

生成联合审计论文资产：

```bash
python scripts/build_roi_bc_dpg_joint_paper_assets.py
```

## 4. 当前证据位置

- BC-DPG v3：`results/final_evidence/bc_dpg_v3_final/`
- Stage 3：`docs/polarimetric_stage3/STAGE3_FROZEN_CONCLUSION.md`
- Stage 4：`results/data_audit/roi_stage4_selected_sixfold_v1/`
- 最终联合审计：`results/data_audit/final_roi_bc_dpg_joint_v2_base_threshold/`
- 联合论文资产：`results/final_evidence/roi_bc_dpg_joint_fixed_threshold/`

早期 `final_roi_bc_dpg_joint` 使用了错误的 BC 判决来源，已经移出活动证据，不得引用。

## 5. 下一阶段优先级

### 优先级 1：补充真实目标和干扰数据

- 空飘球、带载空飘球和不同载荷构型；
- 静稳、摆动、旋转及耦合运动状态；
- 鸟类、地物、气象和不同硬件状态背景；
- 不同日期、距离、方位、天气和场地；
- 保留连续 H/V 复数 IQ、时间顺序和扫描元数据。

### 优先级 2：建立锁定外部评价

按日期、场地或采集批次预先隔离训练、验证和最终盲测。所有阈值、模型和后处理规则在盲测前冻结。

### 优先级 3：BC-DPG 因果在线化

将完整扫描统计替换为只使用过去样本的历史状态，并分别报告样本独立、因果上下文和完整扫描离线增强三种条件。

### 优先级 4：学习型联合模型

仅在训练/验证集使用 BC 分数、ROI 分数、背景状态和质量指标选择门控或融合规则；规则冻结后只评估一次测试集。

### 优先级 5：多域细粒度分类

在真实空飘球数据齐备后，引入长慢时间时频/微多普勒、极化、轨迹和行为特征，逐级开展目标类别、有载/无载、载荷类型和运动状态识别。

## 6. 继续开发时必须保持的纪律

- 不依据测试集重新选择阈值、容差、模型或组合逻辑；
- 不覆盖冻结 checkpoint 和正式证据目录；
- smoke 结果只验证接口，不作为性能结论；
- 两折诊断、六折内部验证和外部盲测必须分开报告；
- 样本独立、完整扫描上下文和因果在线模型必须分开命名；
- 新结论必须附带数据范围、划分方式、决策来源和可复现清单。
