# BC-DPG-FCN v3 最终冻结与论文证据报告

生成时间：2026-07-22T10:54:55

## 1. 最终模型决策

- 最终模型：**BC-DPG-FCN v3**
- 固定 `shift_regularization`：**0.01**
- 当前定位：带完整扫描组统计的离线背景条件校准器
- 不采用折内自适应 v3.1 作为最终部署模型

## 2. 六折主结果

| 模型名称 | false_alarms_sum | 相对原始虚警降幅 | pfa_mean | pd_mean | auc_mean | target_shift_mean |
|---|---|---|---|---|---|---|
| 原始DPG-FCN | 186.0000 | 0.0000 | 0.2067 | 0.9059 | 0.9912 | 0.0000 |
| 样本独立BC校准 | 122.0000 | 0.3441 | 0.1356 | 0.9059 | 0.9919 | 0.0355 |
| 扫描上下文BC-DPG-FCN v3 | 56.0000 | 0.6989 | 0.0622 | 0.9059 | 0.9993 | 0.0028 |

原始 DPG 的六折虚警为 186 个；样本独立校准器降至 122 个；完整扫描上下文 v3 降至 56 个。完整 v3 的平均 Pd 由 0.9059 保持为 0.9059。

## 3. 消融结论

| mode | 模块角色 | calibrated_false_alarms_sum | calibrated_test_pfa_mean | calibrated_test_pd_mean | target_shift_mean_mean |
|---|---|---|---|---|---|
| full | 最终完整模型 | 56.0000 | 0.0622 | 0.9059 | 0.0028 |
| no_scan_context | 结构消融 | 122.0000 | 0.1356 | 0.9059 | 0.0355 |
| no_background_classification | 损失消融 | 92.0000 | 0.1022 | 0.9059 | 0.0042 |
| no_background_tail | 损失消融 | 116.0000 | 0.1289 | 0.9059 | 0.0033 |
| no_target_protection | 安全约束消融 | 50.0000 | 0.0556 | 0.9059 | 0.0103 |
| no_target_keep | 安全约束消融 | 59.0000 | 0.0656 | 0.9059 | 0.0060 |
| no_pairwise | 排序约束消融 | 71.0000 | 0.0789 | 0.9059 | 0.0010 |
| no_shift_selectivity | 安全约束消融 | 44.0000 | 0.0489 | 0.9059 | 0.0087 |
| no_shift_regularization | 强度约束消融 | 35.0000 | 0.0389 | 0.9059 | 0.0078 |

扫描上下文、背景分类和高分背景尾部损失均有直接消融支持。目标保护、target keep 与 shift selectivity 主要限制真实目标的非必要抑制，不能只用当前 Pd 是否下降来判断其价值。

## 4. 正则权重扫描决策

各折验证集选择的权重为：Fold 1=0, Fold 2=0.01, Fold 3=0, Fold 4=0.0025, Fold 5=0.01, Fold 6=0

折内选择结果在测试侧共剩余 35 个虚警，但权重跨折不一致，且较弱正则会提高部分折的目标 shift。因此该结果只作为方法探索，不替代固定权重 0.01 的当前 v3。

## 5. 论文表述边界

- 可以表述为六折扫描组内部验证。
- 不能表述为跨日期、跨场地或跨环境独立盲测。
- 完整扫描上下文可能使用同一扫描后续样本，因此只能作为离线增强结果。
- 样本独立校准器是未来样本级多域主模型更合适的对照基础。

## 6. 文件冻结

- 已记录哈希文件：34
- 缺失文件：0
- checkpoint 哈希数量：12

## 7. 生成图

- `figures/fig1_deployment_false_alarms.png`
- `figures/fig2_ablation_false_alarms.png`
- `figures/fig3_false_alarm_target_shift_tradeoff.png`
- `figures/fig4_selected_regularization_by_fold.png`
