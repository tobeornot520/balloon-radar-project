# 近期成果、失败机制与算法判断

版本：2026-08-06 V12

## 本版新增：零多普勒可审计虚警库

V11 将六折 fixed-notch 与 fixed-residual 的既有预测逐样本配对，但不重新训练、不重新调阈值。
830 个背景样本中，fixed-notch 有 120 个虚警，residual 有 109 个：Fold 4 移除 11 个、没有
新增虚警，Fold 1 的 53 个全部保留。人工复核覆盖 11/120：9 例为近零多普勒峰、2 例为
宽结构，物理类别均为 `unknown`。

逐样本 ID、源文件标识和人工备注只在本地库保存；包内只提供折级、匿名扫描级和可见模式
聚合表。该结果是已消费开发证据的审计，不是外部盲测，也不能把 109/830 当作部署 Pfa。

目标侧补充审计发现：joint success 保持 290→290，但 raw detected 从 302 降到 301；唯一损失
案例在两种方法下都定位失败。318 个目标中有 6 个峰移动，其中 2 个失败案例至少一个轴移动
超过 10 bins。准确表述应是“冻结 joint 判据下无 joint-success 损失”，而不是“目标行为完全
不变”。这项事后审计不授权在同六折重新调参。

本文补充 7 月 28 日以后完成的多域特征挖掘、零多普勒机制、Tian FCN 复现、极化
迁移架构、LAT-MRICD 冻结实验、外部公开数据审计和外场准备。旧冻结 BC-DPG、ROI
与定位结果仍然有效，但以下新结果的证据等级不同，不能直接拼成一张“模型排行榜”。

## 1. 三套数字为什么不同

| 结果族 | 基线误警 | 数据/推理来源 | 证据角色 |
|---|---:|---|---|
| 冻结 BC-DPG 正式证据 | 186 | 既有六折 GPU 冻结预测 | 历史开发基线 |
| 候选 veto 审计 | 186 | 同一批冻结 OOF 候选表 | 测试后机制诊断 |
| 零多普勒统一比较 | 187 | 从 checkpoint 在 CPU 重新推理完整热力图 | 开发机制比较 |

CPU 重推理基线多出的 1 个误警来自半精度 GPU 与 CPU 数值漂移的判决边界。它不应
被解释为数据变化，也不能把 187 偷换回 186。比较 notch 时必须在同一 CPU 重推理
口径内比较 187 与 120；描述冻结历史时继续使用 186。

## 2. 多域特征目录

项目对 1,148 个检测样本、77 个扫描组和 256 个长窗 UAV 样本建立了 56 个特征的
目录。检测局部特征锚定冻结 OOF DPG 候选，不使用标签真值位置构造输入。

当前最显著的特征包括：

| 特征 | 方向无关 AUC | 当前解释 |
|---|---:|---|
| 候选局部零多普勒能量占比 | 0.9687 | 背景候选更靠近零频区域 |
| 候选局部峰值占比 | 0.9619 | 目标候选局部结构更集中 |
| 主带/旁带能量占比 | 0.9327 | 指向候选附近的谱形差异 |
| ROI ZDR-like IQR | 0.8567 | 相对 H/V 幅度结构，不是绝对 ZDR |
| ROI 相位集中度 | 0.8378 | 未标定相对量，可能受采集源影响 |

零多普勒占比在六个背景扫描组压力检查中最差 AUC 仍为 0.9535，是目前最稳定的单
特征线索。但目标全部来自 `20260202`，背景全部来自 `20260204`，因此这些 AUC 同时
包含日期、设备状态和场景差异，不能作为物理因果结论。

时域峰度、零频能量、距离位置和相位类特征对背景扫描组或 UAV 文件具有较高依赖度。
这说明直接训练多域分类器很容易学习“采集文件/日期”，所以现阶段先做特征稳定性、
组泄漏和可辨识性审计，而不是追求一个看起来很高的分类准确率。

完整目录见[多域特征审计](../evidence/09_MULTIDOMAIN_FEATURE_CATALOG.md)。

## 3. 零多普勒候选 veto 诊断

在冻结候选位置上，按预测速度单元到零频中心的距离做离线 veto：

| 半宽/bins | 误警 | 联合命中 | 相对基线损失 |
|---:|---:|---:|---:|
| 不 veto | 186 | 289 | 0 |
| 4 | 88 | 289 | 0 |
| 6 | 47 | 289 | 0 |
| 7 | 19 | 289 | 0 |
| 8 | 14 | 288 | 1 |
| 9 | 13 | 282 | 7 |

半宽 7 到 9 之间出现明显目标损失拐点。这个结果说明“接近零频”是虚警的重要机制，
但也说明不能简单扩大硬 notch：真实低径向速度目标正处于相同区域。veto 使用了完整
开发结果观察半宽，不是可部署规则，也没有寻找第二候选峰。

原报告与完整曲线见[候选 veto 审计](../evidence/14_ZERO_DOPPLER_CANDIDATE_VETO.md)
和 `assets/tables/zero_doppler_candidate_veto_tradeoff.csv`。

## 4. 固定 soft notch 与学习抑制

统一机制接口保持每折 DPG 阈值冻结，比较四种模式：原基线、固定 soft notch、
dense-negative 训练和只允许降低 logit 的 clutter-aware head。

### 六折冻结对照

| 模式 | 误警 | pooled Pfa | 最差折 Pfa | 联合命中 |
|---|---:|---:|---:|---:|
| CPU 基线 | 187 | 0.2253 | 0.6667 | 289 |
| 固定 soft notch | 120 | 0.1446 | 0.4467 | 290 |

固定 notch 是安全参考，因为它行为确定、从不提高 logit，而且在同口径重推理中降低
误警并保留目标。但它仍由当前开发数据启发，不能直接冻结为外部部署规则。

### Fold 1/4 学习机制门槛

| Fold | 模式 | 选中 epoch | 测试误警 | 测试 joint Pd | 固定 notch 参考 |
|---|---|---:|---:|---:|---|
| 1 | dense negative | 1 | 71 | 1.0000 | 53 / 1.0000 |
| 1 | clutter aware | 12 | 85 | 1.0000 | 53 / 1.0000 |
| 4 | dense negative | 0 | 100 | 0.9038 | 67 / 0.9231 |
| 4 | clutter aware | 12 | 100 | 0.9038 | 67 / 0.9231 |

两种学习机制都没有超过固定 notch。Fold 4 dense-negative 选择 epoch 0 的含义是：
在包含未训练起点的合法模型选择中，所有训练后 epoch 都没有改善验证规则。clutter
head 虽满足“校准后 logit 不增加”的结构合同，但学到的抑制过弱或依赖训练扫描，未
泛化到困难背景。

当前决策是停止这两个设置的六折扩展。下一版只考虑“固定 notch + 小幅学习残差”，
并要求在 Fold 1 和 Fold 4 同时超过固定参考后才开放六折。详见
[六折固定对照](../evidence/15_ZERO_DOPPLER_FROZEN_SIXFOLD.md)、
[Fold 1/4 四模式比较](../evidence/16_ZERO_DOPPLER_FOLD01_04_COMPARISON.md)和
[阶段决策](../evidence/17_ZERO_DOPPLER_MECHANISM_CONCLUSION.md)。

## 5. Tian 2024 FCN 复现状态

项目已独立实现模型、分类/回归目标、PIR/MDP 后处理、验证阈值选择、单折训练器和
六折编排器。第一轮 H-only 六折迁移得到零联合检测，随后审计发现旧实现还使用了非
论文 L1 距离及错误的 `d_min/d_5/d_avg` 定义，因此这轮只能作为失败迁移记录。

修正指标与后处理后，Fold 1 扩展 GT 的 validation joint Pd 仍为 0。预登记的
point-GT 单一救援只改变分类目标，不加载 test，得到：

| 指标 | 结果 |
|---|---:|
| 验证目标/背景 | 53 / 150 |
| 联合成功 | 22/53，joint Pd 0.4151 |
| 背景误警 | 2/150，Pfa 0.0133 |
| 责任单元被 MDP 选中 | 8/53 |
| 速度 MAE | 18.57 bins |

进一步机制审计发现所有 53 张目标图都产生两条约 `12x1` 的多普勒条带，目标概率图
与公共模板平均相关系数为 0.99818。加入 16 个随机负样本把 joint Pd 降到 0.2453；
同距离列密集负监督进一步降到 0.1132，模板相关仍约 0.998。两项均已拒绝。

本地 Fold 1 的 318 个目标映射后全部位于同一个距离输出列，71 个采集源中 47 个只
对应一个速度值。当前无法区分论文实现差异、训练条件差异和本地数据可辨识性不足。
因此不继续扫学习率、PIR 阈值、V/HV 或六折，而是先向学长核对输入、采样和原始配置。

截至 2026-08-03，上述原始条件和 H/V/IQ/PRF/坐标事实被报告为目前无法取得。精确复现
因此冻结，底层技术事实仍保持 unknown；替代路线转为 DPG-FCN 零多普勒主线、公开
数据 schema/group 审计和 Tian 合成合同测试。详细说明见分享包
`docs/18_TIAN_REPRODUCTION_FAILURE_AND_ALTERNATIVES_20260803_ZH.md`。

详见[诊断结论](../evidence/10_TIAN_FCN_FOLD1_DIAGNOSTIC_CONCLUSION.md)、
[PIR/MDP 机制分析](../evidence/11_TIAN_FCN_FOLD1_COMPONENT_MECHANISM.md)和
[复现条件请求](../evidence/13_TIAN_FCN_REPRODUCTION_CONDITIONS_REQUEST.md)。

### 5.1 LAT-MRICD 分组可解释基线

D17-NX/HX 已按信号特征提取前冻结的 metadata-only 五折划分完成。整组键为
`(representation, band_code, batch_code)`，模型固定为 dummy、batch-balanced 逻辑回归
和随机森林，不根据 held-out 结果选模型。

| 任务 | 模型 | batch-class macro accuracy | batch-code cluster 95% CI | 最差折 balanced accuracy |
|---|---|---:|---:|---:|
| Narrow-X | logistic | 0.7999 | 0.7659–0.8313 | 0.7204 |
| Narrow-X | random forest | 0.7872 | 0.7373–0.8340 | 0.6973 |
| HRRP-X | logistic | 0.6617 | 0.5826–0.7404 | 0.4946 |
| HRRP-X | random forest | 0.6481 | 0.5764–0.7240 | 0.4934 |

RF-LR 配对差值在 Narrow-X 为 -0.0127（95% CI -0.0511–0.0246），在 HRRP-X 为
-0.0136（-0.0775–0.0487），均跨 0，因此不宣布模型胜者。HRRP-X 和 Narrow-X 属于同一
公开发布，batch 语义未独立验证，子型号也已在开发折出现；这些结果只能称
batch-code-held-out 三类基线，不能称 unseen-model、独立外部、极化、空飘球或 Tian 复现证据。完整报告见
[LAT-MRICD 分组基线](../evidence/23_LAT_MRICD_GROUPED_BASELINES.md)。

### 5.2 D17-XBAND 跨频段冻结负结果

在查看 S/Ku 性能前，项目已冻结 Narrow X source、共同 UAV/weather 二分类、固定可解释
特征、batch-balanced logistic 主模型、dummy 对照、target batch 等权指标和停止规则。正式
运行只执行一次，target 不参与缩放、校准、阈值、选模或扩特征。

| locked target | LR target batch-class macro accuracy | UAV batch recall | weather batch recall | 门判定 |
|---|---:|---:|---:|---|
| X->S | 0.6517 | 0.4433 | 0.8600 | 失败：UAV recall 未严格大于 0.50 |
| X->Ku | 0.8400 | 0.8493 | 0.8307 | 通过 |

X->Ku 的通过不能覆盖 X->S 的失败。预登记条件要求两个 locked target 全部通过，因此整体
决策为 `FAIL_STOP`，作为跨频段不稳定的冻结负结果保留。S/Ku 已消费，禁止同 target 重跑、
CNN、域适配、阈值调整或结果驱动特征扩展；新成员只能复核证据或用合成数据跑合同测试。
完整证据见[跨频段迁移冻结报告](../evidence/24_LAT_MRICD_CROSS_BAND_TRANSFER.md)。

这项结果只属于 LAT-MRICD-1.0 同一公开发布内的 released-band-held-out UAV/weather 评价，
不支持物理频率不变性、未见型号、独立场景、同事件多频融合、H/V 极化、空飘球或 Tian
复现结论，也不改变主 UAV `4/6 BLOCKED_EXTERNAL` 状态。

### 5.3 新增公开数据审计状态与下一项

下一步不再消费 LAT-MRICD 的 S/Ku target，而是处理已取得的新官方数据：

1. DAUR：全量只读审计已完成。77 个逻辑观测的 schema、TD/TR 配对及 canonical/backup
   数值等价性通过，但只有 76 个唯一内容对；11 对记录共享内部帧，连通后形成 39 个未经
   作者确认的候选 source-session 组。严格时间、日期冲突和 1024-bin 物理轴阻塞，禁止训练；
2. HSR：V2 全量只读审计已完成。1,530 MAT/63,148 真实帧/865 routes 和官方
   45,366/9,336 个 train/validation 默认窗口均已冻结；11 MAT/704 帧/529 窗口的 overflow
   隔离。route 以上来源和 512-bin DPL 物理轴未知，禁止训练；
3. FMCWR-2.0：6 个 RAR/90 MAT 已完成只读审计；64 K/26 L、71 个唯一 payload、11 个
   重复组/30 个成员、48 个非权威候选组和 B 通道全空已冻结。归一化轴合成/单记录处理
   合同和合成 smoke 已完成；session/物理轴仍未确认，不训练；
4. DroneRFc-MM：全量 PCD schema 已核验；8 条 radar/GT 时间范围重叠，B1 零重叠并冻结
   `BLOCKED`。更正材料到位前不做 B1 监督对齐，其余 recording 也需另行预登记。

HSR ScienceDB V2 正式 ZIP 为 237,020,946 bytes，SHA256 为
`fea8a21354110a96fb9644dc1c69649b6dc6d1a1b6da512498d9c2d74d839540`，完整性通过。
V2 与期刊包分别有 1,561/1,478 个 ZIP entries；V2 额外含 `overflow/air_routes`，同名样本
长度也存在差异，因此已确认两者不等价、不可混合。V2 的 `CC-BY-NC-4.0` 许可来自
2026-08-04 ScienceDB 页面访问记录，不是 ZIP 内嵌许可文本。全部 MAT schema 统一为有限
`float64 T×512` DPL 与对齐 `T×5` 轨迹；轨迹五列的径向速度 m/s、距离 km、方位角 degree、
高度 m、距离归一化 SNR dB 单位已核验。每个 MAT 恰被一个 route 引用且 route 不跨
published split，但没有 site/date/weather/sensor-run/physical-target/source-session 键，
不能把 route-disjoint 改写成 session-disjoint。512 bins 也不能换算为未经说明的 Hz/速度。
状态为 `PASS_SCHEMA_BLOCKED_SOURCE_PROVENANCE_AND_PHYSICAL_AXIS`，
`model_training_allowed=false`。V2 `Dataset.py` 是只读懒加载器；移动原件警告只属于历史包。

DroneRFc-MM V1 的数据 DOI 为 `10.57760/sciencedb.j00173.00094`，许可为
`CC-BY-SA-4.0`。完整发布为 113 files、75,612,067,287 bytes，本地只选择下载 28 个文件：
1 README、9 mmRadar PCD ZIP、9 GT CSV、6 labels 和 3 code，共 47,366,902 bytes；选择性
subset manifest SHA256 为 `6b0c2ed1a075aa9164a516af001b630a9f775fddc9f399223c1aeeb6e7047b2b`。
9 个 ZIP 均完整；30,717 个 PCD/639,527 个点通过 15 列 schema、finite、POINTS 行数与
嵌入/文件名时间戳核对，字段包括 doppler、power、snr、timestamp。

该子集覆盖 9 recordings、6 UAV models，但全部同日同场景；717 个派生 5 秒 windows 必须
按原始 recording 管理，禁止随机 frame/window split。它不是 ADC/IQ、不是 H/V，也没有鸟、
天气或空飘球，只能用于点云/轨迹接口与时序算法审计，不能替代主数据或宣布模型性能。
其中 B1 radar 在同名 GT 开始约 8 分钟前已结束，零时间重叠，整体状态为
`PASS_SCHEMA_BLOCKED_TIMESTAMP_ALIGNMENT`。其余数据同样在 schema/group 门通过前不训练，
FMCWR-2.0 的仿真飞鸟不能写成真实自然鸟证据。

另外只下载了两个很小的接口样本，而没有继续堆积大数据。Ku UAV 群包约 4 MB，schema
核验得到 3 个物理实验、275 个连续屏和 171,309 个有限 XYZ 点，没有类别、Doppler、IQ 或
极化；NEXRAD 单体扫约 0.4 MB，实际含 Z、V、SW、ZDR、PhiDP、RhoHV，但没有目标标签。
前者只验证群目标航迹接口，后者只验证双极化 loader/公式。42.4/23.46/30.36 GB 的气球、
S 波段 UAV、三频 UAV/真鸟主包均已登记但暂缓，避免在 split 和研究缺口不清楚时浪费网络。

## 6. 极化特征现在做到什么程度

现有 Stage 3/4 结论支持“Power2 负责候选检测，局部 ROI 编码极化信息”的路线。
已实现的迁移 encoder 分为 H/V 功率、H/V 复数 RD、门控显式极化三支，输出可替换
任务头的 embedding。`channel_validity` 可关闭未验证的相位通道。

当前没有正式预训练 checkpoint。下一步若使用现有 UAV/背景做辅助预训练，主验收
指标应是按采集组隔离后的表示稳定性和来源泄漏，而不是目标/背景准确率。未来获得
空飘球数据后先冻结 encoder 训练新任务头，再逐层解冻并与时域、微多普勒和轨迹
embedding 融合。详见[极化迁移编码器](../evidence/18_POLARIMETRIC_TRANSFER_ENCODER.md)。

## 7. 外场采集为什么是当前主线

真正的时域、物理微多普勒和极化研究依赖当前文件没有提供的条件：连续逐脉冲关系、
真实 PRF、硬件时间戳、H/V 同步方式、幅相标定、事件起止和连续真值。

项目已把 2026 年 8-9 月工作分成五道门：

1. capability：确认连续 H/V 复数 IQ 和 300 秒稳定写盘；
2. synchronization：雷达、视频和真值时间映射；
3. polarimetric calibration：H/V 通道映射、幅相重复性和标定 ID；
4. dry run：至少 600 秒纯背景、丢帧和完整回放检查；
5. Pilot：背景、无载球、有载稳定、有载运动四场景，各至少 3 次独立 session。

门禁代码和模板已完成，但没有真实设备证据时状态必须是 BLOCKED。完整 SOP 见
[外场采集 SOP](../evidence/19_FIELD_COLLECTION_SOP.md)。

## 8. 当前算法优先级判断

| 优先级 | 工作 | 启动条件 |
|---:|---|---|
| 1 | 复核 FMCWR-2.0 归一化轴合同；维护 DAUR/HSR/FMCWR/DroneRFc 阻塞证据 | 合同与合成 smoke 已完成，但 grouping/provenance/axis 仍阻塞；其余数据门也关闭；不训练 |
| 2 | 向学长核对 Tian 数据与原始实现 | 获得样例、配置或明确答复即可 |
| 3 | 完成设备能力、同步和标定摸底 | 设备负责人提供实测证据 |
| 4 | 固定 notch + 目标保护残差 | Fold 1/4 预登记，禁止直接六折 |
| 5 | 极化辅助预训练 | 明确组隔离、通道 validity 和泄漏验收 |
| 6 | 时域/微多普勒融合 | 获得连续慢时间、PRF 与状态时间标签 |
| 7 | 空飘球载荷分类 | 至少有同条件背景、无载、有载与锁定外层数据 |

现阶段继续无约束跑网络的边际价值较低。最有价值的工作是获得正确数据条件、建立
可逐层对齐的复现样例，并用困难折门槛拒绝无效机制。

公开数据支线可在不改变主 UAV `4/6 BLOCKED_EXTERNAL` 门槛的前提下推进四项只读数据审计。
D17-XBAND 已以 `FAIL_STOP` 结束，不能继续使用已消费的 S/Ku target 做确认性模型开发；
DAUR 的 schema/pairing 门已完成但 grouping/physical-axis 门未放行；HSR 的 schema/route
门已完成但 source-provenance/physical-axis 门未放行；FMCWR 的 archive/schema/重复门已
完成但 grouping/provenance/physical-axis 门未放行；归一化处理合同已完成，当前只做接口复核。DroneRFc-MM
的 B1 必须先取得更正 GT/可归因偏移，其余
8 条也要新预登记，之后才决定算法任务。
