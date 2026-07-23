# 显式极化门控与分数迁移诊断 V2

本补丁不修改已冻结的 BC-DPG-FCN v3，也不覆盖既有 Polarimetric Representation V1 正式结果。

新增两类工作：

1. 复用 V1 的 `val_predictions.csv` 与 `test_predictions.csv`，诊断验证集阈值跨环境失效、低 FPR 性能和 source-file 分层分数漂移；
2. 新增 `polar6_gated` 与 `ri8_gated`，使用局部 H/V 联合功率置信度抑制低功率区域中不可靠的 ZDR 类、相关性和相位通道。

四种 V2 模式仍统一为 8 通道、相同网络结构：

- `power2`
- `ri4`
- `polar6_gated`
- `ri8_gated`

主实验继续保持样本独立推理，不使用扫描上下文。门控量为样本内功率置信度，只用于抑制低功率区域，不能解释为绝对极化标定。
