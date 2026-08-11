# 评价指标定义

版本：2026-07-28

## 1. 基本单位

- 样本：一条进入检测模型的 H/V 距离—多普勒输入记录。
- 目标样本：`target_present = 1`。
- 背景样本：`target_present = 0`。
- 扫描组：由 `scan_group` 标识的相关样本集合。当前联合审计包含 71 个目标扫描组和 6 个背景扫描组。
- 决策阈值：各 fold 在训练/验证流程中冻结的阈值；联合审计不在测试集重新调阈值。

## 2. 检测与定位判定

### Detected

当模型分数严格大于该 fold 的冻结决策阈值时，记为 detected。当前代码和正式评价统一采用
`score > threshold`；分数恰好等于阈值不计为 detected。

### Localization OK

仅对目标样本定义。当前冻结评价要求：

- 预测距离门与真实距离门的绝对误差不超过 2 gates；
- 预测速度单元与真实速度单元的绝对误差不超过 3 bins。

BC-DPG 和 ROI 均冻结原始候选位置，只调整候选分数，因此校准不会修改定位坐标。

### Correct detection

`correct_detection = target_present AND detected AND localization_ok`

因此，文档中的“正确检测”是检测与定位联合成功，不只是分数超过阈值。

### False alarm

`false_alarm = background sample AND detected`

## 3. 样本级指标

### Score Pd

目标样本中分数严格超过阈值的比例，不要求定位正确。

`score_pd = detected_target_count / target_count`

### Joint Pd

目标样本中检测与定位联合成功的比例。

`joint_pd = correct_detection_count / target_count`

### Pfa

背景样本中超过阈值的比例。

`pfa = false_alarm_count / background_count`

### Pooled / micro 指标

先合并六折计数，再计算比例。例如联合审计的 pooled joint Pd 为 `289 / 318 = 0.9088`，pooled Pfa 为 `56 / 830 = 0.0675`。

### Macro 指标

先在每个 fold 或扫描组内计算比例，再对各单位等权平均。BC-DPG v3 的六折 macro Pfa 为 0.0622，与 pooled Pfa 0.0675 统计口径不同。

### Worst-fold、median 和 IQR

- worst-fold Pfa：六折 Pfa 最大值；
- worst-fold joint Pd：六折 joint Pd 最小值；
- median：六折中位数；
- IQR：六折第 25 至第 75 百分位范围。

这些指标用于显示折间异质性，不能被 pooled 指标替代。

## 4. 派生可读性指标

联合证据表中的派生 precision 和 F1 使用 joint-success contingency：

- TP：correct detection；
- FP：背景 false alarm；
- FN：目标样本减 correct detection；
- TN：背景样本减 false alarm。

`joint_precision = TP / (TP + FP)`

`joint_f1 = 2 * joint_precision * joint_pd / (joint_precision + joint_pd)`

`specificity = TN / (TN + FP) = 1 - Pfa`

由于 TP 要求定位正确，这里的 precision/F1 是“检测定位联合成功”口径，不应与纯分类 precision/F1 混写。

## 5. 不确定性与配对统计

### Wilson interval

对 pooled 二项比例给出 Wilson 95% 区间。连续雷达样本存在相关性，因此该区间仅作为样本级参考，可能偏窄。

### Stratified scan-group bootstrap

分别在目标扫描组和背景扫描组内有放回重采样，保留组内全部样本，重新计算 pooled joint Pd 和 Pfa。当前背景仅有 6 个独立扫描组，区间应结合这一小样本限制解读。

### McNemar exact diagnostic

对同一样本上的两个冻结方法，仅使用不一致对的计数进行双侧精确二项检验。该检验描述配对差异，不负责选择新模型或组合规则。

## 6. ROI 和校准专用指标

- `target shift`：校准前后目标 logit 或分数的抑制变化，用于检查是否通过压低目标换取低虚警；
- `background shift`：背景抑制变化；
- `pAUC@5% FPR`：只关注低 FPR 区间的局部 ROC 面积；
- `test background exceed validation Q99`：测试背景分数超过验证背景第 99 百分位的比例，用于诊断背景迁移。

## 7. 当前尚不可报告的指标

以下指标缺少观测时长或事件定义，当前不能从样本计数可靠换算：

- false alarms per hour；
- event-level false alarm rate；
- 连续虚警持续时间；
- 单位面积虚警率。

可以报告每个已评价背景扫描组的虚警数，但必须同时给出扫描组数量、样本数和最大值，不能把它等同于每小时虚警率。

## 8. 扫描上下文敏感性口径

- complete-scan：统计同一扫描的全部样本，包含当前样本并可能包含未来样本；
- leave-one-out：排除当前样本，但仍可使用扫描内未来样本，只用于检查 self-inclusion；
- past-only：只统计确定性顺序中位于当前样本之前的样本；
- history window：past-only 模式最多保留的最近历史样本数；
- zero-context / cold start：当前样本之前没有历史，12 维上下文全部置零。

当前 past-only 顺序按 `(beam_layer, azimuth_deg, sample_id)` 推断，未由逐样本时间戳验证。冻结完整模型原本使用 complete-scan 上下文训练，因此将其输入替换为 leave-one-out 或 past-only 属于 out-of-distribution 敏感性审计。该结果不能用于从测试集选择窗口，也不能命名为已训练的因果模型。

## 9. 定位误差汇总口径

冻结定位证据同时报告三种样本范围：

- `all_targets`：全部目标样本，属于无条件定位误差；
- `score_detected_targets`：分数达到各 fold 冻结阈值的目标，用于描述检测后的条件定位误差；
- `joint_success_targets`：同时达到阈值且在 2 gates / 3 bins 容差内的目标，只描述成功样本内部误差。

条件定位 MAE 不能替代无条件误差或 joint Pd。正式表格同时给出 MAE、中位数、P90、P95、P99 和最大值，以避免少量大误差被单一均值或中位数掩盖。

当前距离换算使用 `30 m/gate`，速度换算使用 `0.183 m/s/bin`。这些数值是离散网格下标误差的等价换算，不是相对于未量化连续真值的物理测量误差。BC-DPG 只校准候选分数，冻结审计已验证六折 BC-DPG 与 raw DPG 的预测距离/速度坐标逐样本一致。
