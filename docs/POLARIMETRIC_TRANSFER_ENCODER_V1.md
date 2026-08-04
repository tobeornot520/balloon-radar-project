# 可迁移极化表征编码器 V1

日期：2026-07-30

## 定位

当前可以先搭极化模型，但只能把它定义为可迁移表征编码器，不能称为空飘球载荷识别
模型。现有内部 H/V 主任务数据只有 UAV 与背景，并且 H/V 相干和绝对幅相标定尚无设备证据。

冻结的 Stage 3 已证明：Power2 检测和定位最好，RI4 的固定阈值背景迁移较小，密集
显式极化通道没有稳定独立增益。Stage 4 又证明局部候选 ROI 比全图极化替换更合理。
因此 V1 不重做热力图检测器，而是编码候选附近的 H/V 局部结构。

## 输入与结构

输入沿用 `build_roi_source` 的 10 通道合同：

| 分支 | 通道 | 作用 |
|---|---|---|
| Power | H/V 归一化功率 | 保留当前最可靠的检测能量结构 |
| Complex | H/V RD 实部与虚部 | 保留可迁移的复数散射信息 |
| Explicit | 门控相对 ZDR、rho、相位 cos/sin | 低功率区域抑制后的相对极化描述 |

三个分支在空间域分别编码，再融合并输出固定长度 embedding。网络支持任意 ROI 尺寸，
默认输出 128 维 embedding。任务分类头独立存在，未来可以替换而不修改编码器。

`channel_validity` 可以关闭不可信通道。例如设备尚未证明 H/V 相干时，应关闭相位
cos/sin；这不会把未标定相位伪装成可信物理特征。

## 当前能做什么

- 使用现有 train/validation 候选 ROI 做接口和表征稳定性检查；
- 设计 target/background 辅助预训练，但必须按采集组隔离且不得根据 test 选模型；
- 比较 embedding 是否主要编码日期、采集源或距离速度先验；
- 保存可迁移 encoder 权重，不冻结当前 Stage 3/4 结论。

当前尚未授权正式预训练。必须先预登记任务、正负样本、组隔离、增强、模型选择指标和
迁移验收标准。

## 未来如何复用到空飘球

1. 用已标定、同步的空飘球 H/V ROI 载入 encoder；
2. 先冻结 encoder，只训练新的载荷分类头；
3. 验证集稳定后逐层解冻，避免小样本直接破坏已有表征；
4. 将极化 embedding 与时域 embedding、微多普勒 embedding 和轨迹特征在融合层组合；
5. 在同日期背景、无载球、有载稳定和有载运动控制下评价；
6. 最终只在锁定测试集评价一次。

## 当前边界

- `relative_ZDR_like` 和相对相位不是绝对极化参数；
- 当前 UAV/背景标签不能替代空飘球载荷标签；
- 当前数据日期与类别混杂，不能用分类准确率证明载荷可迁移性；
- 编码器实现完成不等于预训练 checkpoint 已存在。

对应实现：

- `models/polarimetric_transfer_encoder.py`
- `features/roi_polarimetric_refinement.py`
- `configs/polarimetric_transfer_encoder_v1.yaml`
- `tests/test_polarimetric_transfer_encoder.py`
