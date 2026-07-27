# 项目整体介绍

## 1. 研究定位

项目长期目标是建立面向空飘球及其载荷的多域雷达智能辨识链路：

> 目标检测定位 → 目标类别识别 → 空飘球有载/无载判断 → 载荷类型识别 → 静稳、摆动、旋转等状态识别

当前完成的是这条链路的前端：基于 UAV 与背景数据验证 H/V 双极化雷达的目标检测、距离—速度定位和背景虚警抑制。UAV 数据用于建立方法和工程基础，不能替代未来真实空飘球、载荷和状态数据。

## 2. 当前输入、输出与评价

- 输入：H/V 两路复数 IQ 数据。
- 基础表征：距离—多普勒图与 H/V 功率表征 Power2。
- 输出：目标存在性分数、目标距离门和速度单元位置。
- 冻结评价：830 个背景样本、318 个目标样本；6 个背景扫描组、71 个目标扫描组。
- 正确检测：目标分数达到冻结阈值，且距离误差不超过 2 gates、速度误差不超过 3 bins。
- 主要评价：虚警数、Pfa、joint Pd、折间分布、扫描组 bootstrap，以及距离和速度定位误差。

六折按扫描组隔离，但当前目标与背景的采集日期完全耦合，因此结果是内部开发估计，不是跨日期泛化证据。阈值和模型选择原则上只使用训练集或验证集；现有 Stage 4 六折扩展仍复用了曾参与模式筛选的 Fold 1 和 Fold 4，这一限制已记录在模型选择台账中。

## 3. 当前技术路线

```text
H/V complex IQ
      |
      +--> RD / Power2 representation
              |
              +--> DPG-FCN candidate detection and localization
              |         |
              |         +--> sample-independent BC (online-oriented baseline)
              |         +--> complete-scan BC-DPG v3 (offline upper bound)
              |
              +--> frozen candidate locations
                        |
                        +--> local ROI polarimetric suppression study

Frozen BC-DPG decisions + frozen ROI decisions
              |
              +--> sample-aligned post-test audit; no joint rule selected
```

### DPG-FCN

以 H/V 功率距离—多普勒图为主要输入，完成整图候选检测和距离—速度定位。它是后续背景校准和 ROI 精修的共同前端。

### BC-DPG-FCN v3

在冻结 DPG 输出上加入目标保护的背景条件校准。样本独立 BC 不使用扫描上下文，更接近在线条件；完整版本使用同一扫描的整体统计上下文，可能包含未来样本，只能定位为离线扫描感知性能上限。

### 显式极化与 ROI Stage 4

项目比较了 RI4、Polar6-gated、RI8-gated 等显式极化表征。密集极化输入没有稳定替代 Power2，因此后续保留 Power2 候选位置，只在局部 ROI 内使用极化信息做 suppression-only 精修。Power control 与 RI4 是先在 Fold 1/4 筛选，再扩展到六折。

### 联合审计

BC-DPG 与 ROI 使用不同的抑制信息。最终审计按 fold、sample ID、标签和 MAT 路径对齐 1,148 条测试记录，在冻结阈值下分析重叠与互补；它不训练联合模型，也不从测试结果中选择 AND/OR 规则。

## 4. 工程构成

| 模块 | 职责 |
|---|---|
| `datasets/` | 清单驱动的数据读取、分组划分、极化和 ROI 数据接口 |
| `features/` | RD、显式极化、门控和 ROI 特征构造 |
| `models/` | FCN、DPG、背景校准器和 ROI refiner |
| `training/` | 正式训练实现 |
| `scripts/` | 实验编排、审计、汇总、绘图和证据打包 |
| `results/data_audit/` | 数据划分、对齐审计和冻结汇总表 |
| `results/final_evidence/` | 报告、表格、图件和哈希清单 |
| `tests/` | 自动化接口与证据生成测试 |

完整仓库没有单一 `main.py`。各阶段通过带版本号的训练、评估和审计脚本运行。

## 5. 当前完成度

已经完成：

- H/V IQ 到 RD 输入、标签和物理坐标映射；
- UAV/Background 联合检测和距离—速度定位；
- H、V、HV 与 DPG-FCN 基线；
- 分组数据审计与六折内部评价；
- BC-DPG-FCN v3 背景虚警校准及消融；
- 显式极化 Stage 3 与局部 ROI Stage 4；
- BC-DPG/ROI 冻结阈值逐样本联合审计；
- 折间分布、样本级区间、扫描组 bootstrap 和 McNemar 配对诊断；
- 数据卡、指标定义、模型选择台账和哈希清单。

尚未完成：

- 真实空飘球有载/无载和载荷类别数据集；
- 鸟类等关键干扰类别的充分覆盖；
- 长慢时间时频、微多普勒和轨迹建模；
- 同日目标/背景对照及跨日期、跨场地、跨天气的锁定盲测；
- 按观测时长和事件定义统计的部署级虚警指标；
- past-only 和 leave-one-sample-out 扫描上下文控制实验；
- 在训练/验证集上选定并一次性测试的学习型联合模型。
