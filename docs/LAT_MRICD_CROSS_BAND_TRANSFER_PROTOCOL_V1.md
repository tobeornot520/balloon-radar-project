# LAT-MRICD 跨频段迁移预登记协议 V1

版本日期：2026-08-03

状态：`PREREGISTERED_NOT_RUN`

冻结配置：`configs/lat_mricd_cross_band_transfer_v1.json`

## 1. 预登记时点和研究问题

本协议在任何 S/Ku 目标频段迁移性能被运行或查看之前冻结。X 波段分组基线结果已知，
因此任何以 X 为测试频段的方向均不进入本次正式配置。所有冻结方向必须一次密封运行，
不得在查看某一方向后新增方向、修改特征、调参或重选模型。

研究问题是：在 LAT-MRICD-1.0 同一公开发布内，仅用源频段拟合的固定可解释特征分类器，
对另一个完全留出的发布频段中 UAV 与气象两类记录有何表现？该实验只是
dataset-internal band-held-out transfer，不是外部盲测。

## 2. 先验依据与放行范围

已冻结的 X 波段基线中，Narrow 逻辑回归的 batch-class macro accuracy 为
`0.7999`，最差折 balanced accuracy 为 `0.7204`，允许进入跨频段预登记。HRRP 对应
数值为 `0.6617` 和 `0.4946`，折间异质性明显，所以 HRRP 迁移始终只是 exploratory，
不能开放深度模型门。

鸟类只在 X 波段发布，本次所有方向统一为 `UAV vs weather`，类别码仅允许
`[1, 3]`。禁止将 X 的三分类模型直接在缺少鸟类的 S/Ku 上评分。

## 3. 输入、可比子集和覆盖

正式实现只能读取配置中绑定 SHA256 的 5 个聚合 MAT，不得读取任何同目录明细 MAT，
也不得同时加载聚合文件和明细文件，否则会重复记录。前四列仅用于类别筛选、
覆盖审计和 batch 分组；`band_code`、`category_code`、`model_code` 和 `batch_code`
均不得作为模型特征。

Narrow 主分析仅保留三个频段共有的 UAV 型号 `1=Mavic 2`、`2=Phantom 4`、
`3=Air 3S` 以及 `9=weather clutter`：

| 表征/频段 | 分析记录 | UAV 记录 / batch | weather 记录 / batch | 唯一 batch |
|---|---:|---:|---:|---:|
| Narrow-S | 4,038 | 1,531 / 10 | 2,507 / 406 | 412 |
| Narrow-X | 6,451 | 868 / 5 | 5,583 / 357 | 358 |
| Narrow-Ku | 3,319 | 2,598 / 8 | 721 / 41 | 47 |
| HRRP-X | 2,496 | 1,411 / 12 | 1,085 / 22 | 33 |
| HRRP-Ku | 3,471 | 1,765 / 10 | 1,706 / 41 | 50 |

batch-class 数可因一个 batch 同时包含两类而大于唯一 batch 数。HRRP-Ku 的 UAV 全部为
`model_code=10 (unspecified UAV)`，而 HRRP-X UAV 为型号 1--6，无法建立型号对齐子集。
上述筛选后总记录数、逐类记录数、逐类 batch 数和唯一 batch 数已经逐源写入冻结配置的
`expected_analysis_coverage`，正式实现必须逐项相等后才允许拟合，不能只检查原始 MAT 行列数。

## 4. 冻结迁移方向

| transfer_id | 源频段 | 目标频段 | 角色 |
|---|---|---|---|
| `narrow_x_to_s_shared_binary` | X | S | locked primary |
| `narrow_x_to_ku_shared_binary` | X | Ku | locked primary |
| `narrow_s_to_ku_shared_binary` | S | Ku | secondary |
| `narrow_ku_to_s_shared_binary` | Ku | S | secondary |
| `narrow_x_ku_to_s_shared_binary` | X + Ku | S | secondary |
| `narrow_x_s_to_ku_shared_binary` | X + S | Ku | secondary |
| `hrrp_x_to_ku_binary` | X | Ku | exploratory |

配置中不得出现 target X。两个 locked primary 共同构成继续门；secondary 只描述方向
不对称性和多源训练的压力；HRRP X->Ku 因型号标注不对齐，且 X 聚合为 `float64`、
Ku 聚合为 `uint16`，不能将差异归因为频段物理效应。

## 5. 特征、模型和源侧拟合

特征必须逐列复用 grouped baseline V1 的 `extract_narrow_features`、`extract_hrrp_features`
和 `reconstruct_narrow_iq`，不允许增删特征、PCA 或数据驱动筛选。Narrow 频率仅使用
`cycles/sample`；HRRP 位置仅是归一化样点坐标。

“复用 V1”不是只复用一个可变文件名：冻结来源提交为
`a102ea0c81925a3e0686bccc763a1856d6da319e`，冻结 grouped runner SHA256 为
`e5cdf9342492196b8999c9be0c4434cfe49fe8cbdeb0514b50862baaf367dfcd`。配置同时冻结
Narrow 27 项和 HRRP 18 项特征的完整名称与顺序。正式运行必须同时校验文件哈希、
实际提取列名、列顺序和列数；任何一项变化都必须建立新协议版本，不能沿用 V1。

三个模型和超参与 V1 完全一致：

- `dummy_prior`；
- `logistic_batch_balanced`：StandardScaler + L2 logistic，`C=1`、`lbfgs`、`max_iter=5000`；
- `random_forest_batch_balanced`：500 棵树、`max_depth=8`、`min_samples_leaf=5`、
  `max_features=sqrt`、`n_jobs=1`。

不扫参，不根据目标频段结果选模型，三者全部保留。预登记主模型是低复杂度的
`logistic_batch_balanced`，随机森林仅作固定敏感性模型。StandardScaler、分类器、概率校准
和任何阈值只能在源频段拟合；目标频段的信号、特征统计和标签不得参与拟合、
选择或校准。逐记录归一化可以使用该记录自身，但不能使用跨记录的目标统计。
目标元数据标签已用于预登记类别交集和 batch 覆盖资格检查，并将在密封运行末尾用于最终指标；
不得把这一事实写成“目标标签从未使用”。禁止的是用目标标签拟合、选阈值、校准概率、选模型或调参。

不做概率校准，三个模型都按 `model.classes_` 的升序类别码执行 argmax，完全平局时固定选
较小类别码。`dummy_prior` 也只在 source 上使用与 LR/RF 完全相同的样本权重拟合，
不得使用 target prior。二分类 argmax 等价于固定 0.5 决策，不进行 target 阈值调整。

训练权重按三层等权：类别 -> 源频段 -> 该频段该类别内的 batch-class cell。对记录
`i` 的未归一化权重为：

```text
1 / (该类别存在的源频段数
     * 该源频段该类别的 batch 数
     * 该 band-batch-category cell 的记录数)
```

最后归一化为平均权重 1。多源时 batch cell 必须使用 `(band_code, batch_code,
category_code)`，不能把不同频段的同号 batch 合并。

## 6. raw batch-code overlap 审计

正式分组键仍是 `(representation, band_code, batch_code)`，因为 raw batch code 是否在频段间
具有全局采集语义未被独立证实。但每个方向都必须记录 raw code 重号数并执行预登记的
disjoint sensitivity。

Narrow 全发布元数据中 S-X 重号为 `177`；应用共享 UAV 型号子集后为 `176`。
这两个数不得混用。共享型号子集的 S-Ku 为 `32`、X-Ku 为 `33`；HRRP X-Ku 为
`0`。

disjoint sensitivity 先应用类别资格和型号筛选，再保持目标频段完整不变，仅从 source 中删除 raw
`batch_code` 与 target 任一 batch 重号的记录。不允许为增加覆盖而删除 target、重新编码
batch 或降低最小覆盖门。去重后 source 每类少于 3 个 batch 时，只输出
`NOT_IDENTIFIABLE` 和覆盖，不填充性能值或 CI。

- X->S 去重后 X 仅剩 2 个 UAV batch，固定为 `NOT_IDENTIFIABLE`；
- X->Ku 去重后 X 仍有 UAV 5 batch、weather 324 batch，可报告；
- X+S->Ku 去重后 pooled source 仍有两类，但 S 源只剩 2 个 UAV batch，未通过“每个
  source-band x class 至少 3 batch”的门，因此仅记为 partial-support `NOT_IDENTIFIABLE`，不报性能；
- 其他 secondary 方向依配置中的冻结状态报告，不用 disjoint 结果替代主分析。

该敏感性仍不能证明 session 隔离：相同 raw code 可能只是重复编号，不同 raw code 也可能
来自同一未记录事件。

## 7. 指标、不确定性和最差目标频段

预登记主指标为 `target_batch_class_macro_accuracy`：先在每个 target batch-category cell 内计算
逐行正确率，再在每类内对 cell 等权平均，最后对 UAV/weather 等权平均。

必须同时报告 pooled accuracy、balanced accuracy、macro-F1、binary log-loss、ROC-AUC、
target batch 等权正确率、P10 和最差 batch、每类 batch 等权召回、batch-class cell P10/
最差值与其记录数，以及逐行和 batch-class 等权的两种混淆矩阵。最差单记录 cell
不能单独支持方法结论。

ROC-AUC 固定以 `category_code==1 (UAV)` 为正类并使用 class-1 概率；log-loss 的概率列顺序
固定为 `[1, 3]`；P10 使用 linear 分位数插值。batch-class 等权混淆矩阵先在每个真类内让
target batch 等权，再累计预测类别质量。以上定义只影响辅助指标，不改变继续门。

不确定性使用固定种子的 2,000 次 target-batch cluster bootstrap，重采样单位为
`(representation, target_band_code, batch_code)`。必须报告每个模型的主指标 95% CI，以及
`logistic_batch_balanced - dummy_prior` 的配对 95% CI。配对比较的每次重采样必须使用
完全相同的 target batches，一次抽到某个 raw batch code 时必须重采该 target batch 的全部类别和
全部记录，不得分别重采 batch-class cell。有效重复数不得少于 1,900。该 CI 是在每个
已固定 source fit 条件下的 target-batch 不确定性，不包含重新拟合 source 所带来的变动。
区间固定使用 percentile method；若一次重采样缺少任一类，则废弃该次并报告有效重复数。

每次重复有放回抽取的次数严格等于该 transfer/scope 中 target 唯一 batch 数；同一 batch
重复抽中时必须按抽中次数计权，且每次抽中带入该 batch 的全部记录和全部类别。指标仍按
“batch-class 内逐行正确率 -> 类内对抽中 batch 等权 -> 两类等权”计算。分位数固定使用
linear 插值。每个 transfer/scope 的随机种子由
`SHA256("random_state|analysis_scope|transfer_id")` 前 8 个十六进制字符转换为 uint32，
不得依赖循环执行顺序。

CI 适用于 7 个 `band_qualified_primary` 方向，以及 2 个实际通过覆盖门的
`raw_batch_code_disjoint_sensitivity` 方向；每个可报告 transfer/scope 都必须包含 3 个固定模型
的主指标 CI 和 1 个 LR-minus-dummy 配对 CI。`NOT_IDENTIFIABLE` sensitivity 不产生性能或
CI。继续门只读取两个 locked primary 的 `band_qualified_primary` 配对区间。

最差目标频段指标是两个 locked primary 的
`target_batch_class_macro_accuracy` 中的较小值。S 和 Ku 必须分开报告，不得只报 pooled 结果。

## 8. 继续门和停止规则

两个 locked primary target 必须同时满足下列全部条件：

1. `logistic_batch_balanced` 的 `target_batch_class_macro_accuracy > 0.60`；
2. UAV 和 weather 的 target batch 等权召回均严格 `> 0.50`；
3. `logistic - dummy` 主指标差值的配对 95% CI 下界严格 `> 0`。

上述三项均必须使用未四舍五入数值判定，显示表中的格式化小数不得用于开关。

任一 target 任一条失败，即冻结负结果，停止基于本次数据的 CNN、域适配、特征扩展和
调参。HRRP exploratory 结果无论多高都不能单独开门。

即使全部通过，也只允许启动另行预登记的方法工程研发，不授权在本次已消费的 S/Ku
target 上重新调参、选模型后宣称 confirmatory improvement。后续正式比较必须取得新的 locked
数据，或使用在本次运行前已独立保留且尚未查看的测试集。
当前没有这样的独立保留集。

## 9. 工程验收与执行顺序

正式运行前必须机器校验：配置 schema，5 个聚合路径/哈希/行列数，类别与共享型号覆盖，
正式方向无 target X，target 不参与拟合，多源权重的三层等权，二分类 bootstrap 和
LR-dummy 配对重采样，以及 `NOT_IDENTIFIABLE` 方向不生成伪指标。

只允许用合成数据或不输出目标性能的接口检查做 smoke。正式入口始终拒绝脏 Git 工作树，
不提供 overwrite、output 路径覆写或跳过提交绑定的函数参数/CLI 选项；output 只能取冻结配置中的
`results/experiments/lat_mricd_cross_band_transfer_v1`。该路径不得与原始数据根目录或 `.git`
存在自身、祖先或后代关系，也不得是项目根目录或其祖先。测试只能 monkeypatch 外部 Git 校验、
正式 output/消费记录路径解析和数据加载接口。实现、测试、配置和冻结特征实现进入同一 pre-result
Git 提交后，才能执行一次密封正式运行。正式 output 目录必须完全不存在，禁止覆盖任何已有结果。

在加载任一 target 聚合文件前，runner 必须确认冻结的外部消费记录
`results/final_evidence/lat_mricd_cross_band_transfer_v1.run_consumed.json` 不存在，并在 clean Git/
commit 绑定通过后以 exclusive-create 写入 `RESERVED`。创建记录或原子替换记录后必须 fsync
其父目录。任一后续异常必须保留记录并更新为
`FAILED_OR_INTERRUPTED_RUN_CONSUMED`；成功写完 15 个正式输出后更新为
`COMPLETED_AND_TARGETS_CONSUMED`。消费记录绑定 experiment id、pre-result commit、配置与 runner
SHA256、S/Ku target 列表以及成功 summary 的 SHA256，不得包含原始或绝对路径。已有消费记录
表示本次 target 已被保守视为消费，禁止再次运行。

结果必须记录 pre-result commit，并保留实现、配置、冻结特征实现、5 个聚合文件和输出表的
SHA256，明确记录未执行搜索、目标拟合或结果驱动修改。冻结正式配置必须产生 27 个 source fit
记录和 36 行 bootstrap interval；数量不符即失败并消费本次运行。

正式 runner 根目录输出精确为以下 15 个文件，不得多、不得少：

```text
transfer_coverage.csv
raw_batch_overlap_audit.csv
training_weight_audit.csv
aggregate_metrics.csv
target_batch_class_metrics.csv
bootstrap_intervals.csv
confusion_matrices.csv
feature_definitions.csv
feature_importance.csv
disjoint_sensitivity.csv
claim_boundaries.csv
model_fit_manifest.json
gate_decision.json
summary.json
REPORT.md
```

禁止生成或发布逐样本预测、OOF 预测、逐样本权重、原始数据、模型 checkpoint、原始/绝对
文件路径。证据构建器必须再次校验精确文件集合、字段、跨表关系、哈希和确定性。
证据构建器的 CLI 同样冻结 source、配置、实现和
`results/final_evidence/lat_mricd_cross_band_transfer_v1` 输出路径，不提供路径覆写或 overwrite。
证据目录必须完全不存在；构建器只删除自身创建的临时 staging 目录，不递归删除已有目的地。

## 10. 允许与禁止的声明

唯一允许的核心表述是：

> 在 LAT-MRICD-1.0 内，用一个或多个发布源频段训练的固定可解释特征分类器，
> 在完全留出的发布目标频段上评价了共同 UAV/weather 任务。

禁止声明：

- 物理频率不变性，或 Hz、速度、旋翼转速等物理微多普勒；
- 同事件配对跨频融合或样本级多频特征比较；
- 独立 session、跨场景、外部盲测或部署泛化；
- 未见 UAV 型号泛化，尤其不得将 HRRP-Ku 的 `unspecified UAV` 当作未见型号证据；
- H/V 极化、绝对距离/RCS、因果时序或真实连续微动；
- 空飘球、载荷类型、载荷状态或运动状态识别；
- Tian 2024 精确复现。

跨频表现受型号组成、batch 语义、采集条件、存储类型和未记录处理链共同混杂，
不得把任何方向差异单独归因为雷达频段。
