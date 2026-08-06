# 零多普勒 residual 目标安全审计 V1

更新时间：2026-08-06

状态：`COMPLETE_AS_DEVELOPMENT_AUDIT`

## 1. 为什么补这项审计

既有 V2 结果正确报告了 joint hit `290 -> 290`，但 joint hit 是“超过检测阈值且定位误差满足
容差”的复合指标。复合指标不变，并不意味着目标分数、原始检测状态和预测峰位置完全不变。

本审计不训练模型、不修改阈值、不改变 V2 的冻结选择，只对六折 318 个目标预测进行逐样本
配对，用于披露 residual 的目标侧行为和风险边界。

## 2. 冻结结果

| 指标 | fixed-notch | fixed + residual | 配对变化 |
|---|---:|---:|---:|
| 原始 detected | 302 | 301 | 丢失 1，新增 0 |
| localization ok | 298 | 298 | 丢失 0，新增 0 |
| joint success | 290 | 290 | 丢失 0，新增 0 |

唯一的 raw-detection 损失案例在 fixed 和 residual 下都不满足定位容差，因此它不是原有 joint
success，joint Pd 没有下降。这不等于该变化可以忽略：它说明 residual 的“目标保护”只在当前
冻结 joint 判据下通过，不代表所有目标行为完全保持。

其他配对事实：

- 292 个目标分数下降，26 个在 `1e-12` 容差内不变，0 个上升；
- 平均分数变化为约 `-0.0010521`，最大下降约 `-0.0500557`；
- 312 个目标峰位置不变，4 个最多移动 1 bin，2 个至少一个轴移动超过 10 bins；
- 6 个峰移动案例中，4 个前后均为 joint success，2 个前后均非 joint success；
- 距离误差 3 个变差、1 个改善；速度误差 5 个变差、1 个改善；其余不变。

大位移集中在本来就失败的样本，其中一个同时对应 raw-detection 损失。由于这是已消费开发
证据，只能用于定位风险，不允许据此修改阈值、loss 权重或 residual 结构后继续在六折确认。

## 3. 本地层与分享层

本地 `target_case_library_local.csv` 保存 318 行 sample ID、来源标识、分数、峰位置、误差和
状态转移，用于未来新数据对照；它不进入 Git 或分享包。

可分享层只包含：

- `fold_target_safety_summary.csv`：折级 detected/localization/joint、分数和误差变化；
- `peak_shift_histogram.csv`：全局峰移动分箱，不含逐样本标识；
- `score_delta_quantiles.csv`：全局分数下降分位数；
- `summary.json`：输入哈希、冻结计数、检测损失上下文和禁止声明。

## 4. 可以和不可以怎样表述

可以表述：residual 在已消费六折上没有改变 joint-success 数量，也没有增加背景虚警；但它
使 1 个原本定位失败的目标失去 raw-detected 状态，并使 6 个目标峰发生移动。

不可以表述：

- “residual 对所有目标完全无影响”；
- “目标保护已经得到部署安全验证”；
- “290 -> 290 证明新场景 joint Pd 一定不下降”；
- “大位移样本已经被识别为某种物理机制”；
- 根据本审计重新调参后仍把同六折称为独立验证。

真正的确认性验收仍需要新同条件目标/背景扫描、冻结阈值和一次性 locked evaluation。

## 5. 复现

```bash
conda run -n radar-torch python scripts/audit_zero_doppler_target_safety_v1.py --overwrite
conda run -n radar-torch python -m pytest -q tests/test_zero_doppler_target_safety.py
```

该流程只读取既有预测 CSV，不读取原始 MAT，不联网、不使用 GPU，也不训练模型。
