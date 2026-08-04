# 当前成果与证据

## 1. 冻结阈值六折联合审计

最终联合审计精确对齐六折 1,148 条记录，其中背景 830 条、目标 318 条。BC 判决来自冻结的 `base_threshold_test_predictions.csv`，ROI 判决来自冻结的 `refined_fixed_*` 列，测试阈值没有重新调整。

| 方法 | 虚警数 | pooled Pfa | 正确检测 | pooled joint Pd |
|---|---:|---:|---:|---:|
| 完整扫描上下文 BC-DPG-FCN v3 | 56 | 0.0675 | 289/318 | 0.9088 |
| Power2 baseline | 300 | 0.3614 | 268/318 | 0.8428 |
| ROI power control | 237 | 0.2855 | 268/318 | 0.8428 |
| ROI RI4 | 196 | 0.2361 | 268/318 | 0.8428 |

“正确检测”要求分数达到冻结阈值，且距离误差不超过 2 gates、速度误差不超过 3 bins。因此这里的 Pd 是检测与定位联合成功率，不是仅按分数计算的 score Pd。

完整扫描上下文 BC-DPG-FCN v3 是当前六折内部开发评价中表现最好的离线扫描感知方案。它可能使用同一扫描中晚于当前样本的信息，因此代表离线条件下的性能上限，不代表严格实时部署性能。

![六折检测权衡](../assets/figures/joint_pooled_detection_tradeoff.png)

## 2. 冻结距离-速度定位证据

六折 raw DPG 与 BC-DPG 的预测距离/速度坐标逐样本完全一致，说明 BC 只改变候选分数，不改变定位位置。在 318 个目标样本中：

| 条件 | 数量 | 比例 |
|---|---:|---:|
| 达到各折冻结分数阈值 | 302/318 | 0.9497 |
| 不考虑分数、定位在 2 gates / 3 bins 容差内 | 297/318 | 0.9340 |
| 同时达到阈值且定位正确 | 289/318 | 0.9088 |
| 已过阈值目标中的定位正确率 | 289/302 | 0.9570 |

全部目标的距离误差 MAE 为 1.418 gates、中位数为 1、P90 为 2；速度误差 MAE 为 1.154 bins、中位数为 0、P90 为 1。最大误差分别达到 39 gates 和 40 bins，说明少量灾难性定位错误会明显影响均值，不能只报告 MAE 或只报告成功样本。

按离散网格换算，1 gate 对应 30 m，1 bin 对应 0.183 m/s。这只是网格下标误差的等价值，不是相对于未量化连续真值的物理测量误差。详细逐折、距离分层和速度分层结果见[冻结定位证据](../evidence/07_BC_DPG_LOCALIZATION_EVIDENCE.md)。

![冻结定位误差 CDF](../assets/figures/bc_dpg_localization_error_cdf.png)

## 3. 折间异质性

| BC-DPG 指标 | 结果 |
|---|---:|
| pooled Pfa | 0.0675 |
| 六折 macro Pfa | 0.0622 |
| 六折 median Pfa | 0 |
| Pfa IQR | 0 至 0.0700 |
| 最差折 Pfa | 0.2800，Fold 1 |
| 六折 macro joint Pd | 0.9059 |
| 六折 median joint Pd | 0.9142 |
| joint Pd IQR | 0.8750 至 0.9436 |
| 最差折 joint Pd | 0.7917，Fold 6 |

56 个 BC-DPG 虚警全部集中在 Fold 1 的 42 个和 Fold 4 的 14 个，其他四折为 0。合并 Pfa 不能替代折间分布，困难背景鲁棒性仍是主要瓶颈。

![六折异质性](../assets/figures/joint_fold_heterogeneity.png)

## 4. 派生指标与不确定性

按“正确检测”为 TP 的检测定位联合口径，BC-DPG 的 joint precision 为 0.8377、joint F1 为 0.8718、specificity 为 0.9325。

| 区间 | joint Pd 95% | Pfa 95% | 解释 |
|---|---:|---:|---|
| 样本级 Wilson | 0.8721–0.9358 | 0.0523–0.0866 | 忽略扫描内相关性，仅作参考 |
| 分层扫描组 bootstrap | 0.8599–0.9521 | 0–0.1618 | 以 71 个目标和 6 个背景扫描组重采样 |

背景只有 6 个独立扫描组，扫描组区间仍不稳定。当前缺少每个样本的观测时长和事件合并定义，不能把 56 个样本级虚警换算为每小时或事件级虚警率。

## 5. 部署条件分层

BC-DPG 独立冻结证据比较了三个条件：

| 方法 | 六折虚警 | 六折平均 Pd | 合理定位 |
|---|---:|---:|---|
| 原始 DPG-FCN | 186 | 0.9059 | 样本级基础检测定位模型 |
| 样本独立 BC | 122 | 0.9059 | 更接近在线条件的部署导向基准 |
| 完整扫描上下文 BC-DPG v3 | 56 | 0.9059 | 离线扫描感知性能上限 |

![BC-DPG 部署条件比较](../assets/figures/bc_dpg_deployment_false_alarms.png)

这里的“六折平均 Pd”是各折指标的算术平均；第 1 节的 pooled joint Pd 是合并 318 个目标后的计数比例，两者不能混写。

## 6. 冻结 BC-DPG 上下文敏感性

冻结完整模型的上下文替换审计还得到以下结果：

| 上下文模式 | 背景误警 | 正确检测 | 证据角色 |
|---|---:|---:|---|
| complete-scan | 56/830 | 289/318 | 与训练上下文匹配的离线上限；六折判决零重放差异 |
| leave-one-out | 54/830 | 289/318 | 测试后 self-inclusion 敏感性；仍使用未来样本 |
| past-only，最近 4/16/64 个 | 148/138/105 | 均为 288/318 | 测试后假定顺序与窗口敏感性 |
| past-only，全部历史 | 93/830 | 288/318 | 测试后假定顺序敏感性，不能据此选择窗口 |

完整 checkpoint 使用 complete-scan 上下文训练，所以 leave-one-out 和 past-only 均属于推理时上下文替换的 OOD 诊断。past-only 顺序由 `(beam_layer, azimuth_deg, sample_id)` 推断，没有逐样本采集时间戳验证；71 个目标扫描组和 6 个背景扫描组各有一个零历史冷启动样本。`54/830` 和 `93/830` 不能描述为部署性能。下一模型必须使用真实时间顺序的因果上下文训练，只在训练/验证集选择窗口，再进行锁定测试。

详细定义和聚合结论见[因果上下文敏感性审计](../evidence/05_BC_DPG_V3_CAUSAL_CONTEXT_AUDIT.md)；包内同时提供聚合指标、成对变化、历史覆盖和逐折重放检查 CSV。

后续[采集顺序就绪审计](../evidence/06_DETECTION_ACQUISITION_ORDER_AUDIT.md)检查了全部 1,148 个 MAT 文件。文件只含 H/V IQ 数组，没有时间戳变量；MAT 头创建时间至少晚于文件名时间 49.1 天，并与文件 mtime 相差不超过 3 秒，属于后期保存或复制时间。正式因果训练门禁因此保持关闭。已经通过的 Fold 1 validation-only 小样本 smoke 不加载测试 split，只证明代码接口贯通，不增加性能证据。

为避免下一批数据重复出现同类缺口，项目已经建立 40 列版本化采集合同和三档预检。当前 V4 1,148 行清单在最严格的 `locked_evaluation` 档为 FAIL，缺少 33 个合同字段；schema 失败后，行完整性、事件时间、因果顺序、通道、外层隔离和同条件类别对照门禁全部标为 BLOCKED。详见[当前数据合同缺口报告](../evidence/08_CURRENT_DATA_COLLECTION_READINESS.md)。

## 7. BC-DPG 与 ROI RI4 的互补性

| 结果分区 | 数量 |
|---|---:|
| 共同虚警 / 仅 BC / 仅 ROI | 36 / 20 / 160 |
| 虚警并集 | 216 |
| 共同正确 / 仅 BC / 仅 ROI | 263 / 26 / 5 |
| 正确目标并集 / 两者均未正确 | 294 / 24 |

![互补性分区](../assets/figures/joint_complementarity.png)

逻辑 AND 可把虚警降至 36，但正确检测降至 263；逻辑 OR 可把正确检测提高到 294，但虚警升至 216。McNemar 配对诊断的背景双侧精确 p 值为 `2.61e-28`，目标联合成功 p 值为 `0.000192`。这些都是查看冻结测试结果后的描述性诊断，不负责选择新模型或组合规则。

## 8. 选择复用与数据限制

Stage 3 使用 Fold 1 和 Fold 4 做表征诊断。Stage 4 同样先用这两个开发折筛选 Power2 baseline、ROI power control 和 ROI RI4，再扩展到六折；最终六折 ROI 汇总仍包含 Fold 1/4。因此结果不是六个完全未参与模式选择的独立外层折。

冻结审计的 71 个目标扫描组全部来自 `20260202`，6 个背景扫描组全部来自 `20260204`，类别与采集日期完全耦合。当前结果不能排除日期、环境、硬件状态或采集流程差异，也不能支持跨日期泛化结论。

## 9. LAT-MRICD 分组基线与跨频段冻结负结果

D17-NX/HX 已完成并冻结 X 波段 batch-code-held-out 三类基线。Narrow-X 的固定 LR/RF
batch-class macro accuracy 为 0.7999/0.7872，HRRP-X 为 0.6617/0.6481；两项 RF-LR
配对差值 CI 均跨 0，因此不选模型胜者。

D17-XBAND 随后按预登记协议执行了唯一一次密封运行。StandardScaler、固定模型、加权 source
prior 和 argmax 判决都只由 source band 拟合，S/Ku target 不参与缩放、校准、阈值、特征或
模型选择。locked primary 结果为：

| 迁移 | 固定主模型 | target batch-class macro accuracy | UAV batch recall | weather batch recall | target 门 |
|---|---|---:|---:|---:|---|
| Narrow X->S | batch-balanced logistic | 0.6517 | 0.4433 | 0.8600 | 失败：UAV recall 未严格大于 0.50 |
| Narrow X->Ku | batch-balanced logistic | 0.8400 | 0.8493 | 0.8307 | 通过 |

虽然 X->Ku 的全部门条件通过，X->S 的 UAV batch recall 失败意味着两个 locked target 未
同时通过，因此整体决策是 `FAIL_STOP`，不是“部分成功后可继续调优”。S/Ku 都已经消费，
禁止在同一 target 上重跑确认性模型比较、调 CNN、做域适配、改阈值或结果驱动扩特征。
队员可以复核冻结表格、哈希、门判定和声明边界，也可以用完全合成数据运行接口/合同测试，
但不能再次消费真实 S/Ku 来生成新的确认性结论。

完整冻结报告见[LAT-MRICD 跨频段迁移](../evidence/24_LAT_MRICD_CROSS_BAND_TRANSFER.md)。
允许的唯一解释是“在 LAT-MRICD-1.0 同一公开发布内，用固定可解释特征完成了 released-band-
held-out UAV/weather 评价，且预登记总体继续门失败”。它不证明物理频率不变性、同事件多频
融合、未见型号、独立场景、H/V 极化、空飘球识别、因果部署或 Tian 复现。该公开数据支线
也不改变主 UAV 完成门，主方向仍为 `4/6`、`BLOCKED_EXTERNAL`。

## 10. 新公开数据的完整性证据与边界

LSS-DAUR-1.0 V3 的 314 个官方文件和 148,763,512 bytes 与官方发布元数据一致，并与本地
保留的 314 文件下载清单复核一致。全量只读
审计覆盖 308 个 MAT：154 canonical 与 154 `backup_original` 是同一 77 个逻辑观测的
MATLAB v5/v7.3 存储视图，共享数值完全相等，不能把 backup 当额外样本。77 对 TD/TR 的
公共时间、帧号、载频和逐帧结构完全对齐，共 11,366 帧、7,728,640 个有限复数 DPL 值。
但有一组 2 个 recording 的 TD/TR 内容完全重复，只剩 76 个唯一内容对；另有 11 对记录
共享内部帧，字段与内容保守连通后得到 39 个候选 source-session 组。

该审计没有放行训练。全部 77 条都有重复时间戳，894 个重复位置后只剩 10,472 个唯一时间
位置；13 条帧号不连续，6 条文件名/`File_head` 日期冲突。58 条为 512 bins、19 条为
1024 bins，而官方绘图脚本固定 512；Bird/UAV 在三种候选 session 定义下均零重叠，24 个
日期/配置 scene 中 20 个类别纯，存在明显域捷径。`V` 字段全零。状态为
`PASS_SCHEMA_PAIRING_BLOCKED_GROUPING_AND_PHYSICAL_AXIS`；证据见
[DAUR 只读审计](../evidence/26_LSS_DAUR_READ_ONLY_AUDIT.md)。

LSS-HSR-L 的 ScienceDB V2 正式包已经下载，大小为 237,020,946 bytes，SHA256 为
`fea8a21354110a96fb9644dc1c69649b6dc6d1a1b6da512498d9c2d74d839540`，ZIP 完整性通过。
V2 有 1,561 个 ZIP entries，期刊包有 1,478 个；V2 额外包含 `overflow/air_routes`，同名
样本长度也存在差异，因此两包已确认不等价，不能拼接或混合划分；期刊包只保留为独立血统
对照。V2 的 `CC-BY-NC-4.0` 许可记录来自 2026-08-04 ScienceDB 页面访问，不是 ZIP 内嵌许可文本。

V2 全量只读审计冻结 1,530 MAT、63,148 个真实帧和 865 条 `air_route_x`。train 为
1,269 MAT/51,789 帧/723 routes，按发布默认参数精确得到 45,366 窗口；validation 为
250/10,655/131/9,336。另有 11 MAT/704 帧/11 routes/529 窗口的 overflow，发布说明和
`Dataset.py` 主入口均未赋予其训练、验证或测试角色，因此保持隔离。

全部 MAT 都是有限 `float64 T×512` DPL 与对齐 `T×5` 轨迹。轨迹五列单位已验证为径向速度
m/s（正值远离）、距离 km、方位角 degree、高度 m、距离归一化 SNR dB；DPL 512 bins
到 Hz/速度的物理映射未知。每个 MAT 恰被一个 route 引用，route 不跨 published split，
但缺少 route 以上 site/date/weather/sensor-run/physical-target/source-session 键，不能称
session-disjoint 或独立场景外测。状态为
`PASS_SCHEMA_BLOCKED_SOURCE_PROVENANCE_AND_PHYSICAL_AXIS`，`model_training_allowed=false`。
V2 自带 `Dataset.py` 是只读懒加载器；移动原件警告只适用于不等价期刊历史包的旧脚本。

DroneRFc-MM V1 的数据 DOI 为 `10.57760/sciencedb.j00173.00094`，许可为
`CC-BY-SA-4.0`。完整发布共 113 files、75,612,067,287 bytes；本地没有下载完整发布，只选择
28 个雷达相关文件：1 个 README、9 个 mmRadar PCD ZIP、9 个 GT CSV、6 个 labels 和
3 个 code，共 47,366,902 bytes。选择性子集 manifest SHA256 为
`6b0c2ed1a075aa9164a516af001b630a9f775fddc9f399223c1aeeb6e7047b2b`；9 个 ZIP 完整。
只读审计进一步对 30,717 个 PCD 和 639,527 个点完成固定 15 列、finite、POINTS 行数及
嵌入/文件名时间戳核对，字段包括 doppler、power、snr 和 timestamp。

该子集只有 9 个 recordings、6 个 UAV models，且同日同场景；717 个派生 5 秒窗口不是独立
采集单元，禁止随机 frame/window split。它不是 ADC/IQ、不是 H/V，也没有鸟、天气或空飘球
标签，只放行点云/轨迹接口和按 recording 分组的时序算法审计，不能替代主数据、构成独立
外部验证或宣布识别性能。8 条 recording 的 radar/GT 时间范围重叠；B1 雷达在同名 GT
开始前约 8 分钟已经结束，零重叠，因此整体状态为
`PASS_SCHEMA_BLOCKED_TIMESTAMP_ALIGNMENT`，B1 禁止监督对齐。

两个更小的样本只用于接口 smoke。Ku UAV 群官方包为 3,996,753 bytes，5 个 MAT 按连续
屏号聚合后共 275 屏、171,309 个有限 XYZ 点，但只来自 3 个物理实验，没有目标 ID、类别、
Doppler、IQ 或极化。NEXRAD 只取一个 395,379-byte KTLX Level II 体扫，实际含 Z、V、SW、
ZDR、PhiDP、RhoHV；它没有 UAV/气球标签，也不是本项目雷达的原始 H/V IQ。两者不能产生
识别指标，分别只核验航迹和双极化读取接口。

## 11. 证据等级

| 证据 | 范围 | 当前角色 |
|---|---|---|
| H/V/HV 历史基线 | 早期固定划分 | 历史开发基线 |
| Stage 3 显式极化 | Fold 1/4 | 开发折表征筛选 |
| Stage 4 两折 | Fold 1/4 | 六折扩展模式筛选 |
| BC-DPG v3 | 六折分组评价 | 内部开发估计；完整扫描版本为离线上限 |
| BC-DPG 冻结定位汇总 | 六折、318 个测试目标 | 测试后固定结果聚合；不训练、不推理、不调阈值 |
| BC-DPG 上下文替换审计 | 六折冻结测试特征 | 测试后敏感性诊断；未训练或选择因果模型 |
| 因果训练就绪审计与 smoke | 1,148 个文件元数据；Fold 1 小样本 train/val | 正式门禁关闭；smoke 仅验证接口，不提供性能证据 |
| 新数据合同基线 | 当前 V4 1,148 行；40 列 v1 合同 | locked_evaluation 为 FAIL，缺 33 列；下游门禁 BLOCKED |
| 最终联合审计 | 六折、1,148 行对齐 | 测试后固定结果互补性诊断 |
| LAT-MRICD X 波段分组基线 | 同一公开发布、batch-code-held-out | 冻结的公开数据内部三类基线；不是独立外部验证 |
| LAT-MRICD D17-XBAND | 同一公开发布、S/Ku 一次性 locked target | `FAIL_STOP` 冻结负结果；target 已消费，禁止同 target 确认性复用 |
| LSS-DAUR V3 | 77 个逻辑配对观测、76 个唯一内容对、39 个保守候选组 | schema/配对/数值等价性通过；分组/时间/1024-bin 轴阻塞；无模型性能证据 |
| HSR ScienceDB V2/期刊包 | V2 的 1,530 MAT/63,148 帧/865 routes；不等价历史包 | V2 schema/route 通过但 source provenance/512-bin 轴阻塞；overflow 隔离、禁止训练和混包，无模型性能证据 |
| DroneRFc-MM radar subset | 完整发布中的 28 个选择性文件、9 recordings | PCD schema 通过但 B1 同步阻塞；不是独立外部性能证据 |
| Ku UAV 群 / NEXRAD | 3 个物理实验 / 1 个完整体扫 | 航迹与双极化 loader smoke；不是识别训练或性能证据 |
| 外部锁定盲测 | 尚无 | 不能声称 |

## 12. 不能从当前结果推出的内容

- 不能声称已经完成空飘球有载/无载或载荷类型识别；
- 不能声称已经通过跨日期、跨场地或跨天气独立盲测；
- 不能把完整扫描上下文 BC-DPG 描述为严格实时因果模型；
- 不能把 54 个 leave-one-out 或 93 个 past-only all-history 虚警描述为已训练模型的部署结果，也不能据此选择历史窗口；
- 不能把 AND、OR 或串联逻辑描述为已训练、已选择或已部署的联合模型；
- 不能用两折诊断结果或 pooled 指标替代完整六折分布；
- 不能把样本虚警数直接解释为每小时或事件级虚警率。
- 不能把 30 m/gate、0.183 m/s/bin 的网格等价误差描述为连续物理真值测量精度。
- 不能把旧 `new_split=test` 改名为锁定测试，也不能用文件名或 beam/azimuth 推断值补写已验证采集顺序。
- 不能把 D17-XBAND 的 X->Ku 通过写成整体迁移成功，也不能隐去 X->S UAV recall 失败；
- 不能在已消费的 S/Ku target 上重跑确认性比较、训练 CNN、做域适配或根据结果改特征。
- 不能混合 HSR V2 与期刊包，不能使用 overflow、随机拆 HSR MAT/frame/window、把 route
  称为 session-disjoint、把 512 bins 写成物理 Hz/速度或启动 HSR 模型训练；也不能把
  DroneRFc-MM 的随机 frame/window 拆分写成独立泛化，B1 不能在没有更正 GT/可归因偏移时做监督对齐；
- 不能把 DroneRFc-MM 点云子集描述为 ADC/IQ、H/V 极化、鸟/天气/空飘球或主任务性能证据。
- 不能把 Ku 群目标的列/点或 NEXRAD 的 gate/ray/patch 随机拆成独立样本，也不能把天气
  雷达双极化矩描述为本项目已标定 H/V 通道证据。
