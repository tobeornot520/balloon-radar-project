# 当前检测数据卡

版本：2026-08-05

## 1. 适用范围

本数据卡描述当前冻结六折联合审计覆盖的 H/V UAV 与背景检测数据。它只对应检测、距离—速度定位和虚警抑制前端，不对应空飘球有载/无载、载荷类型或运动状态分类。

## 2. 冻结评价样本

| Fold | 背景样本 | 目标样本 | 合计 | 背景扫描组 | 目标扫描组 |
|---:|---:|---:|---:|---:|---:|
| 1 | 150 | 53 | 203 | 1 | 11 |
| 2 | 150 | 53 | 203 | 1 | 12 |
| 3 | 115 | 52 | 167 | 1 | 12 |
| 4 | 150 | 52 | 202 | 1 | 12 |
| 5 | 150 | 60 | 210 | 1 | 14 |
| 6 | 115 | 48 | 163 | 1 | 10 |
| 合计 | 830 | 318 | 1,148 | 6 | 71 |

六折测试记录按 fold、sample ID、标签、scan group 和 MAT 路径精确对齐。目标扫描组每组包含 3 至 5 个样本，中位数为 5；背景扫描组每组包含 115 或 150 个样本。

## 3. 已知数据字段

- H/V 两路复数 IQ；
- 距离—多普勒输入和模型分数；
- `sample_id`、`scan_group`、目标存在标签；
- beam layer 与 azimuth 字段，可用于构造确定性顺序，但不是已验证采集时间戳；
- 预测与真实距离门、速度单元；
- fold、split 和源文件路径；
- 冻结阈值下的 detected、false alarm 和 correct detection 标记。

原始 MAT/IQ 数据、大型训练输出和 checkpoint 不进入 Git 或对外分享包。

## 4. 关键混杂与独立性限制

冻结联合审计中的 318 个目标样本及 71 个目标扫描组全部标记为 `20260202`；830 个背景样本及 6 个背景扫描组全部标记为 `20260204`。因此，目标/背景类别与采集日期在当前评价数据中完全耦合。

这意味着：

- 六折隔离了当前清单中的扫描组，但没有形成同日期内同时包含目标和背景的类别对照；
- 模型收益可能包含日期、硬件状态、环境或采集流程差异，无法仅凭当前数据完全排除；
- 6 个背景扫描组是背景鲁棒性统计的主要独立单位，样本级 Wilson 区间会低估扫描相关性；
- 当前结果必须称为内部开发阶段分组评价，不能称为跨日期、跨场地或外部盲测。

## 5. 当前缺失的论文级数据说明

现有冻结材料无法完整确定以下信息，正式论文或新数据采集必须补齐：

- UAV 平台类型、平台数量和独立飞行架次；
- 场地、天气、风况、雷达工作参数及硬件状态；
- 距离、速度、方位、SNR 和近零多普勒分布；
- 背景类型构成及各类型独立扫描数；
- 单个样本和扫描组对应的物理观测时长；
- 扫描内逐样本真实采集时间与硬件执行顺序；
- H/V 通道幅相校准、同步、缺失值和异常样本处理；
- 相邻虚警合并为事件的规则和总观测时长。

在这些信息缺失时，不能把样本虚警数换算成每小时或事件级虚警率。

## 6. 推荐的新数据设计

1. 每个采集日期和场地同时采集 UAV、空飘球、鸟类等目标及无目标背景。
2. 以日期、场地或飞行架次为最外层隔离单位，预留从未参与结构、阈值和叙事选择的锁定盲测集。
3. 保留连续 H/V 复数 IQ、时间戳、扫描顺序和完整雷达配置。
4. 为背景建立地物、气象、系统干扰和近零多普勒等分层标签。
5. 记录观测时长和事件边界，支持每扫描、每小时和事件级虚警评价。

## 7. 可接受的当前表述

> 当前数据支持基于扫描组隔离的 H/V UAV 检测定位内部开发评价。由于目标与背景采集日期耦合、背景独立扫描组数量有限，最终泛化能力仍需通过同日类别对照和跨日期、跨场地锁定盲测验证。

## 8. 下一批数据的强制合同

下一批数据必须使用 `configs/data_collection_manifest_template_v1.csv`，并通过 `scripts/validate_data_collection_manifest.py` 的分级预检。字段和跨行规则见 [NEW_DATA_COLLECTION_PROTOCOL.md](NEW_DATA_COLLECTION_PROTOCOL.md)。

当前 V4 清单在 `locked_evaluation` 档为 FAIL：40 个合同字段中缺少 33 个，后续完整性、因果顺序和锁定评价检查均被 schema 门禁阻断。该结果位于 `results/data_audit/data_collection_readiness_v1/`。

## 9. 外部公开数据补充：LAT-MRICD-1.0

2026-08-03 从期刊官方补充材料入口取得 LAT-MRICD-1.0。其 33 个 MAT 文件包含 7,119 条
X/Ku HRRP 和 16,072 条 S/X/Ku 窄带 I/Q 记录，可用于 UAV/鸟/气象的大类分组基线、
归一化频率微动特征和跨频段迁移预研。

该数据集与当前冻结 H/V 检测数据是两个独立证据对象：它没有 H/V 极化通道、空飘球标签，
也没有在当前材料中确认 PRF、时间戳、连续 session 或同事件跨频配对。不得用它补写当前
H/V 数据的设备事实，或报告物理 Hz 微多普勒、极化和空飘球识别结论。

批次审计发现 batch 编号存在跨类别/型号碰撞，因此禁止随机拆行。第一轮算法必须按
`(representation, band_code, batch_code)` 保守分组。完整来源、哈希、规模与使用边界见
[外部公开数据核验](EXTERNAL_PUBLIC_DATA_AUDIT_20260803.md)。

D17-XBAND V1 已完成同一发布内、预登记的 band-held-out transfer，并以 `FAIL_STOP` 冻结为
负结果。正式 locked target 仅为 Narrow-S 与 Narrow-Ku 的共同 UAV/weather 二分类；bird
不进入该任务，HRRP 仅作预先声明的压力分析。主模型 batch-balanced LR 的结果如下：

| 迁移 | target batch-class macro accuracy | UAV batch recall | weather batch recall | LR-dummy 95% CI 下界 |
|---|---:|---:|---:|---:|
| X->S | 0.6516798767 | 0.4433201701 | 0.8600395832 | 0.0839896354 |
| X->Ku | 0.8399853939 | 0.8493090645 | 0.8306617233 | 0.2670982858 |

八项预登记条件中仅 S 频段 UAV recall 未通过。S/Ku target 已在唯一一次密封评价中消费，
不得再用同一 target 做 CNN、域适配、特征扩展、结果驱动调参或新的确认性模型比较。完整
证据位于 `results/final_evidence/lat_mricd_cross_band_transfer_v1/`。该实验仍不提供 H/V、
真实 PRF、同事件跨频配对、空飘球或 Tian 复现证据，也不改变主 UAV 方向 4/6
`BLOCKED_EXTERNAL` 的状态。

2026-08-04 新取得的 LSS-DAUR-1.0、LSS-FMCWR-2.0、LSS-HSR-L V2、DroneRFc-MM
选择性雷达子集、Ku 波段 UAV 群目标小包和单个 NEXRAD Level II 体扫，均不自动进入已放行
建模数据。HSR-L 的 237,020,946-byte ScienceDB V2
与 209,569,478-byte 期刊历史包已确认不等价，后者不得与 V2 混合。V2 的
`CC-BY-NC-4.0` 来自 2026-08-04 ScienceDB 页面访问记录，不是 ZIP 内嵌许可文本。V2
全量只读审计冻结
`1530` 个 MAT、`63148` 个真实帧和 `865` 条 `air_route_x`：train 为
`1269 MAT / 51789 帧 / 723 routes / 45366` 个官方默认窗口，validation 为
`250 / 10655 / 131 / 9336`；另有 `11 / 704 / 11 / 529` 的 overflow，因发布用途未说明而
隔离，不当作 train、validation 或 test。全部 MAT 均为唯一变量 `Trace_DPL_Data` 的 1×2
cell，包含有限 `float64` 的 `T×512` DPL 和时间对齐的 `T×5` 轨迹，且 T 与文件名后缀一致。
轨迹五列由官方说明明确为径向速度（m/s，正值远离）、距离（km）、方位角（°）、高度（m）
和距离归一化 SNR（dB）。

V2 的每个 MAT 恰被一个 route 引用，route 不跨 published split 或类别，也未发现原始字节或
解码内容重复；但 `route_id` 只是当前最低可用分组，发布中没有场地、日期、天气、雷达运行、
物理目标或 source-session 键，不能证明 train/validation 在这些来源层面独立。512 列 DPL
也没有可核验的 bin 顺序、零频位置及到 Hz/速度的映射。因此状态为
`PASS_SCHEMA_BLOCKED_SOURCE_PROVENANCE_AND_PHYSICAL_AXIS`，`model_training_allowed=false`；
只放行只读 schema/route 审计、归一化 bin 和轨迹方法设计，不放行模型训练、随机
track/frame/window split 或物理微多普勒
结论。V2 自带 `Dataset.py` 是只读懒加载器，首帧填充只在内存构窗；“会移动原件”的警告
只适用于不等价的期刊历史包旧脚本。

LSS-FMCWR-2.0 V4 的只读 RAR/MAT 审计也已完成。六个 RAR 内实际有 90 个 MAT，其中
84 个 MATLAB v5、6 个 v7.3；64 个 K 频段 `channelA` 为有限复数，26 个 L 频段
`channelA` 为有限实数，而 `channelB` 在 90 个文件中全部为空，因此不是 H/V 极化数据。
原始和解码数值都只有 71 个唯一 payload：11 个精确重复组覆盖 30 个 MAT；66 个候选
recording stems 经精确重复边连通后为 48 个保守候选组，但发布方没有确认其 session 含义。
另有 1 个目录角度/文件名角度冲突，仿真鸟也只能作为仿真数据。该集状态为
`PASS_ARCHIVE_SCHEMA_BLOCKED_GROUPING_PROVENANCE_AND_PHYSICAL_AXIS`，禁止随机 MAT/frame/
window split、模型训练、物理 Hz/速度和自然鸟结论。详见
[FMCWR-2.0 审计](LSS_FMCWR_2_READ_ONLY_AUDIT_20260805.md)。归一化轴的单记录 FFT/STFT
合同和合成 smoke 已完成，但只输出 index/cycles-per-sample 轴；它不解除任何分组、物理轴、
H/V、训练或性能门禁，见
[归一化处理合同](LSS_FMCWR_2_NORMALIZED_PROCESSING_CONTRACT_20260805.md)。

DroneRFc-MM 子集包含 9 个同日 recording 的毫米波 PCD 点云、同步飞行真值和派生标签；
它不是 ADC/IQ，也没有
鸟、天气、背景、H/V 或空飘球标签。其 30,717 帧/639,527 点的 schema、有限值和时间戳已
通过只读审计，但 B1 radar 与同名 GT 时间范围零重叠，因此同步总门阻塞，B1 不得进入监督
对齐。UAV 群小包的 5 个 MAT 仅来自 3 个物理实验，只用于
量测/航迹接口 smoke；NEXRAD 体扫虽含 ZDR、PhiDP、RhoHV 等双极化矩，却没有 UAV/气球
标签，且不是原始 H/V IQ，只用于 loader 和公式 smoke。上述对象在各自 schema、标签、配对、
同步及分组门通过前不进入本数据卡的已放行建模样本。来源、许可、状态和校验回执见
`data/metadata/external_public_datasets_v1.csv` 与
`data/metadata/external_public_artifacts_v1.csv`。

DAUR V3 的只读审计已覆盖 308 个 MAT。154 个 canonical 文件与 154 个
`backup_original` 文件只是同一 77 个逻辑记录 ID 的两种存储视图，共享数值逐元素相等；
backup 不计额外样本。其中 1 组 2 个 recording 的 TD/TR 内容完全重复，因此只有 76 个
唯一内容对；另有 11 对 recording 共享内部帧，保守连通后为 39 个候选 source-session
组。77 对 TD/TR 在时间、帧号、载频和逐帧行数上完全对齐，但全部轨迹
都有重复时间戳，6 条文件名日期与 `File_head` 日期冲突，且 45/40 个候选 session 定义均
未获官方确认。58 条为 512 Doppler bins，19 条为 1024 bins，官方固定 512 的绘图脚本不能
解释后者；宽度还与类别混杂。状态冻结为
`PASS_SCHEMA_PAIRING_BLOCKED_GROUPING_AND_PHYSICAL_AXIS`，随机 frame/window split、把
backup 当新样本、静默修日期、物理 Hz 微多普勒和任何模型训练均未放行。
