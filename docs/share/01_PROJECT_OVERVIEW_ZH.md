# 项目整体介绍

## 1. 研究定位

项目长期目标是建立面向空飘球及其载荷的多域雷达智能辨识链路：

> 目标检测定位 → 目标类别识别 → 空飘球有载/无载判断 → 载荷类型识别 → 静稳、摆动、旋转等状态识别

当前完成的是这条链路的前端：基于 UAV 与背景数据验证 H/V 双极化雷达的目标检测、距离—速度定位和背景虚警抑制。UAV 数据用于建立方法和工程基础，不能替代未来真实空飘球、载荷和状态数据。

## 2. 当前输入与任务

- 输入：H/V 两路复数 IQ 数据。
- 基础表征：距离—多普勒图与 H/V 功率表征 Power2。
- 输出：目标存在性分数、目标距离门和速度单元位置。
- 主要评价：虚警数、Pfa、Pd、ROC-AUC，以及距离和速度定位误差。
- 评价纪律：阈值和模型选择只能使用训练集或验证集；测试集只用于冻结规则的一次性评价。

## 3. 当前技术路线

```text
H/V complex IQ
      |
      +--> RD / Power2 representation
              |
              +--> DPG-FCN candidate detection and localization
              |         |
              |         +--> BC-DPG-FCN v3 scan-aware score calibration
              |
              +--> frozen candidate locations
                        |
                        +--> local ROI polarimetric suppression study

BC-DPG decisions + ROI decisions
              |
              +--> sample-aligned fixed-threshold joint audit
```

### DPG-FCN

以 H/V 功率距离—多普勒图为主要输入，完成整图候选检测和距离—速度定位。它是后续背景校准和 ROI 精修的共同前端。

### BC-DPG-FCN v3

在冻结 DPG 输出上加入目标保护的背景条件校准。完整版本使用同一扫描的统计上下文，主要解决验证集与测试集之间的背景分数迁移和集中虚警问题。

### 显式极化与 ROI Stage 4

项目比较了 RI4、Polar6-gated、RI8-gated 等显式极化表征。密集极化输入没有稳定替代 Power2，因此后续改为保留 Power2 的候选位置，只在局部 ROI 内使用极化信息做 suppression-only 精修。

### 联合审计

BC-DPG 与 ROI 使用不同的抑制信息。最终审计按 fold、sample ID、标签和 MAT 路径对齐 1,148 条测试记录，在固定阈值下分析二者的重叠与互补，不训练联合模型，也不在测试集上选择 AND/OR 规则。

## 4. 工程构成

| 模块 | 职责 |
|---|---|
| `datasets/` | 清单驱动的数据读取、分组划分、极化和 ROI 数据接口 |
| `features/` | RD、显式极化、门控和 ROI 特征构造 |
| `models/` | FCN、DPG、背景校准器和 ROI refiner |
| `training/` | 正式训练实现 |
| `scripts/` | 实验编排、审计、汇总、绘图和证据打包 |
| `results/data_audit/` | 数据划分、对齐审计和冻结汇总表 |
| `results/final_evidence/` | 论文用报告、表格、图件和哈希清单 |
| `tests/` | 自动化接口与证据生成测试 |

完整仓库没有单一 `main.py`。各阶段通过带版本号的训练、评估和审计脚本运行。

## 5. 当前完成度

已经完成：

- H/V IQ 到 RD 输入、标签和物理坐标映射；
- UAV/Background 联合检测和距离—速度定位；
- H、V、HV 与 DPG-FCN 基线；
- 分组数据审计与六折评价；
- BC-DPG-FCN v3 背景虚警校准及消融；
- 显式极化 Stage 3 与局部 ROI Stage 4；
- BC-DPG/ROI 固定阈值逐样本联合审计；
- 正式论文证据、图表、哈希和项目健康检查。

尚未完成：

- 真实空飘球有载/无载和载荷类别数据集；
- 鸟类等关键干扰类别的充分覆盖；
- 长慢时间时频、微多普勒和轨迹建模；
- 跨日期、跨场地、跨天气的锁定盲测；
- 只使用过去样本的严格因果在线背景状态；
- 在训练/验证集上选定并一次性测试的学习型联合模型。
