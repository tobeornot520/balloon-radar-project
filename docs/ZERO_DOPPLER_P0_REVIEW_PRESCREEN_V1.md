# 零多普勒 P0 人工复核预筛 V1

版本：2026-08-03
范围：fixed-notch 与 target-protected residual V2 配对比较中，Fold 4 被 residual 移除的
11 个背景误警。  
证据角色：人工复核前的结构化排序，不是物理背景标签，也不是新的模型性能评价。

## 已知配对事实

- 11 个案例的 fixed-notch 预测峰都位于零多普勒参考 bin 64 的 1–7 bins 内，中位距离为
  4 bins；
- residual 对 11 个案例的分数均严格降低，平均变化为 -0.0773，范围为 -0.2287 至
  -0.0009；
- 5 个案例的分数下降至少 0.1，另外 6 个案例下降小于 0.03；
- 7 个案例的固定/residual 预测峰位置完全相同，4 个案例发生 1–5 个 range gates 或
  2–3 个 Doppler bins 的峰位变化；
- 候选局部零多普勒能量占比的中位数为 0.9763；9/11 不低于 0.95，10/11 不低于 0.92。

这些数值只刻画模型候选附近的相对 RD 结构和配对分数变化。速度 bin 不是已由 PRF
确认的物理速度；H/V 相对量不是绝对极化标定量。

## 推荐人工复核顺序

1. 先看分数下降至少 0.1 的 5 个案例，记录是否能观察到窄近零频脊、多峰、宽结构、边缘峰
   或无明确结构；
2. 再看分数下降小于 0.03 的 6 个案例，比较其可见结构是否与强抑制组不同；
3. 对发生峰位变化的 4 个案例，描述“fixed/residual 标记相对可见结构的位置变化”，不要
   仅据移动方向推断目标或杂波机制；
4. 若图像不足以判断，使用 `needs_more_context` 或 `unavailable`，并写明需要的上下文；
5. 没有独立场景记录时，`physical_class` 必须保持 `unknown`。

## 不可得出的结论

本预筛不能证明这些案例是建筑、地物、鸟类、直流泄漏或任何特定干扰；也不能证明 residual
已具有跨日期、跨场地或部署泛化能力。当前 11 个案例全部来自同一 Fold 4 背景扫描，且六折
均已在开发中使用。

## 复核闭环

1. 运行 `build_zero_doppler_human_review_queue_v1.py`、
   `build_zero_doppler_review_atlas_v1.py` 和
   `build_zero_doppler_review_workbench_v1.py` 得到本地队列、图册与离线工作台；
2. 直接打开图册目录内的 `review_workbench.html`，先按上表顺序审 11 个 P0；
3. 工作台通过 `localStorage` 保存进度；校验通过后导出带日期的新 CSV；
4. 运行 `audit_zero_doppler_human_review_v1.py --reviewed-queue <导出CSV>`；
5. 将审计摘要、证据来源与不确定项回填 D04；
6. 新同条件数据到位后，才可将此处的结构假设用于一次锁定外层验证。

## 2026-08-03 人工复核结果

独立导出的 11 行 CSV 已通过 `audit_zero_doppler_human_review_v1.py`：

| 项目 | 结果 |
|---|---:|
| 已复核 | 11/11 |
| `near_zero_doppler_peak` | 9 |
| `broad_structure` | 2 |
| 其他可见模式 | 0 |
| `physical_class=unknown` | 11 |
| 非空复核备注 | 11 |

审计状态为 `COMPLETE`，导出 CSV SHA256 为
`f71bc142bbea5c445cd9d89a178988dcbbe0157cbbe2b393f47b99ec1fb32680`。该哈希只用于
本地逐样本复核记录的版本追踪；分享包不包含逐样本 CSV。结果支持“11 个被 residual V2
移除的开发集虚警都呈近零频相关可见结构，其中 2 个更适合描述为宽结构”，但没有独立
现场证据把它们命名为具体物理类别。

相关的逐样本图册、工作台、队列和复核输出只保留在受控本地目录，不进入分享包。本文件仅保留
不含样本 ID 的聚合结论。
