# 候选区域引导极化精修：下一阶段固定接口

## 1. 研究目标

保留 Power2 的高检测率和定位能力，只在 Power2 候选峰周围提取局部极化特征，判断候选是真目标还是背景伪峰。极化分支不重新决定全图峰位置。

## 2. 第一轮对照

1. `power2_baseline`：原始 Power2。
2. `power2_roi_power_control`：只使用候选ROI功率统计，作为ROI机制控制组。
3. `power2_roi_ri4`：ROI内 H/V 实部和虚部。
4. `power2_roi_polar6_gated`：ROI内门控 relative_ZDR_like、local_rho_HV 与相对相位。
5. `power2_roi_ri4_polar6_gated`：复数信息与门控极化联合。

第一轮仅运行 Fold 1 与 Fold 4；保持相同 manifest、seed、Power2 checkpoint、阈值选择规则和测试评价规则。

## 3. 建议输入输出

候选输入至少包含：

```text
sample_id
fold
raw_power2_score
power2_pred_range_index
power2_pred_velocity_index
roi_tensor
roi_valid_mask
```

模型统一输出：

```python
{
    "raw_power2_score": ...,
    "refined_score": ...,
    "roi_quality": ...,
    "polarimetric_confidence": ...,
    "score_shift": ...,
    "pred_range_index": ...,   # 默认沿用Power2
    "pred_velocity_index": ... # 默认沿用Power2
}
```

## 4. 评价指标

除固定阈值 Pd、Pfa、AUC、距离和速度MAE外，必须报告：

```text
partial AUC@5% FPR
TPR@1% FPR
TPR@5% FPR
验证—测试背景分数漂移
Power2虚警救回/新增虚警
Power2目标救回/退化
```

## 5. 约束

- 样本独立推理，不使用完整扫描上下文。
- 不修改已冻结的 BC-DPG-FCN v3。
- 不覆盖现有 Power2、RI4、Polar6-gated、RI8-gated 正式结果。
- smoke只验证接口，不用于模型选择。
- 如果ROI极化在两个困难折均无稳定增益，应冻结当前预实验，把重点转向未来严格极化标定与连续时序采集。
