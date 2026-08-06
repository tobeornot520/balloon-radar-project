# 一页式成果摘要与请教说明

版本：2026-08-06 V14

## 项目在做什么

长期目标是利用雷达的时域、极化、微多普勒、距离—速度和轨迹信息，识别空飘球、
有载/无载状态及载荷运动状态。当前已有数据主要覆盖 UAV 与背景，因此现阶段完成的
是检测定位和虚警抑制前端，以及为后续空飘球数据准备的多域特征、模型接口和采集
协议，尚未形成空飘球载荷分类结果。

## 已经得到的主要结果

| 工作 | 当前结果 | 应如何解释 |
|---|---|---|
| 完整扫描 BC-DPG v3 | 56/830 背景误警，289/318 联合检测定位成功 | 当前六折内部开发中的离线扫描感知上限，可能使用未来样本 |
| 样本独立 BC | 122/830 背景误警，289/318 联合成功 | 更接近在线条件的对照，仍不是外部盲测 |
| 冻结定位汇总 | 距离误差中位数/P90 为 1/2 gates，速度为 0/1 bins | 有少量 39 gates、40 bins 的大误差长尾 |
| 多域特征目录 | 1,148 个检测样本提取 56 个特征 | 零多普勒能量和局部峰值最突出，但类别与日期混杂 |
| 固定 soft notch | CPU 重推理下由 187 降至 120 个误警，联合成功 289 增至 290 | 开发期安全参照，不是已冻结部署规则 |
| 两种学习抑制 | Fold 1/4 均未超过固定 notch | 当前设置停止扩展六折 |
| fixed notch + residual V2 | 120 降至 109 个误警，联合成功保持 290/318 | 受约束开发候选；11 个移除误警均在 Fold 4 |
| Tian 2024 FCN | 原迁移失败；point-GT 验证集 joint Pd 0.4151 | 方法诊断和本地消融，不是成功复现 |
| 极化迁移编码器 | 三分支 ROI encoder 接口与测试完成 | 尚无预训练权重，未标定相位可通过 validity mask 关闭 |
| 外场准备 | 能力、同步、标定、dry run、Pilot 五道门已固化 | 真实证据未提供前全部保持关闭 |
| 公开多频段分组基线 | D17-NX/HX 已完成；Narrow-X LR/RF batch-class macro 0.7999/0.7872，HRRP-X 0.6617/0.6481，均有 batch-code cluster CI | 同一公开发布内的 batch-code-held-out 基线；不是 unseen-model、独立外部、H/V、空飘球或 Tian 证据 |
| D17-XBAND 密封迁移 | X->S LR batch-class macro 0.6517，但 UAV recall 0.4433 未过门；X->Ku batch-class macro 0.8400 且该 target 过门；整体 `FAIL_STOP` | 两个 locked target 未同时通过，冻结为负结果；S/Ku 已消费，不得在同 target 上重跑、调 CNN、做域适配或结果驱动改模 |
| DAUR V3 只读审计 | 77 个逻辑 TD/TR 观测、76 个唯一内容对、39 个保守候选 source-session 组；schema/配对通过 | 分组、严格时间、日期和 1024-bin 物理轴仍阻塞，`model_training_allowed=false` |
| HSR V2 只读审计 | 1,530 MAT、63,148 真实帧、865 routes；官方 train/validation 窗口 45,366/9,336 | route 以上来源和 512-bin DPL 物理轴阻塞；529 个 overflow 窗口隔离，`model_training_allowed=false` |

## 当前最关键的发现

1. 困难背景虚警高度集中在接近零多普勒的候选区域。候选 veto 半宽 7 bins 可把
   186 个误警降至 19 个而不损失当前观察到的 289 个联合命中，但半宽 9 开始出现
   明显目标损失。这证明零频附近是强机制线索，也证明固定硬删除存在风险。
2. 固定 soft notch 比第一版 dense-negative 和 clutter-aware 抑制稳定；后续 residual V2
   冻结 fixed notch，只学习局部额外降分，在同口径下进一步把误警从 120 降至 109，联合
   成功保持 290/318。11 例人工复核中 9 例为近零频峰、2 例为宽结构，物理类别均未知。
3. Tian FCN 的本地输出近似两条固定多普勒带，53 张验证目标图与公共模板的平均相关
   系数约 0.998。问题不只是阈值，而是输入/监督条件与当前数据可辨识性不匹配；原数据、
   配置或对齐样例目前不可获得，因此精确复现冻结，转入明确的替代路线。
4. 当前内部 H/V 主任务分类资料只有 UAV，且 H/V 绝对幅相、PRF 和连续慢时间关系未证实。现阶段只能
   搭可迁移表示，不能声称获得物理微多普勒或绝对极化结论。
5. LAT-MRICD 的五折 metadata-only 分组基线和一次性跨频段迁移均已冻结。跨频段主模型
   在 X->Ku 上通过，但 X->S 的 UAV batch recall 只有 0.4433，低于预登记的严格 `>0.50`
   门，因此整体为 `FAIL_STOP`，不能用 Ku 的正结果掩盖 S 的失败。该结果只支持同一公开
   发布内的 band-held-out UAV/weather 评价，不证明物理频率不变性或外部泛化。
6. HSR ScienceDB V2 已完成全量只读审计：1,530 MAT、63,148 个真实帧和 865 routes 的
   schema/索引均通过，官方 train/validation 默认窗口精确复现为 45,366/9,336。另有 11 MAT、
   704 帧、529 窗口的 overflow，因用途未说明而隔离。route 只是最低 published group，
   source-session/场景来源与 512-bin DPL 物理轴未知，状态为
   `PASS_SCHEMA_BLOCKED_SOURCE_PROVENANCE_AND_PHYSICAL_AXIS`，`model_training_allowed=false`。
   V2 与 1,478-entry 期刊历史包不等价、不可混合；V2 自带 `Dataset.py` 是只读懒加载器，
   移动原件警告只针对历史包旧脚本。`CC-BY-NC-4.0` 来自 2026-08-04 ScienceDB 页面访问
   记录，不是 ZIP 内嵌许可文本。
7. DroneRFc-MM V1 完整发布有 113 files、75,612,067,287 bytes；本地只取得 28 个雷达相关
   文件，共 47,366,902 bytes。全量只读审计通过 30,717 frames/639,527 points 的 schema、
   finite/POINTS/时间戳检查；8 条 radar/GT 时间范围重叠，B1 零重叠，整体同步门阻塞。
8. DAUR 的 308 个 MAT 已完成全量只读审计。154 canonical 与 154 backup 共享数值完全
   相等，只代表 77 个逻辑观测；其中 2 个 recording 的 TD/TR 内容完全重复，只有 76 个
   唯一内容对。全部轨迹有重复时间，6 个日期冲突，58/19 条分别为 512/1024 bins；11 对
   recording 共享内部帧，连通后为 39 个候选组。候选组仍不是作者确认的 session，Bird/UAV
   候选 session 零重叠，训练和物理 Hz 微多普勒结论均未放行。
9. FMCWR-2.0 的 6 个 RAR/90 MAT 已完成只读 schema/重复审计：64 K/26 L，B 通道全空；
   90 个 MAT 仅 71 个唯一 payload，11 个精确重复组覆盖 30 个文件，66 stems 连通为 48 个
   非权威候选组。session 和物理轴未知，状态为
   `PASS_ARCHIVE_SCHEMA_BLOCKED_GROUPING_PROVENANCE_AND_PHYSICAL_AXIS`；归一化轴
   合成/单记录处理合同和合成 smoke 已完成，只继续接口复核与外部参数确认；同时维护
   DAUR、HSR 和 DroneRFc 阻塞，各自门禁通过前禁止训练。
10. 公开检索只额外落地两个小型接口样本：Ku UAV 群包约 4 MB、仅 3 个物理实验；NEXRAD
   单体扫约 0.4 MB，含 ZDR/PhiDP/RhoHV 等矩但没有目标标签。它们分别验证航迹和双极化
   读取接口，不扩大训练集。42.4 GB 气球、23.46 GB S 波段 UAV 和 30.36 GB 三频 UAV/真鸟
   主包均因域差异、体量或 split 未审计而暂缓，避免无目的网络与存储开销。

## 最希望请教学长的五件事

1. Tian 论文推荐复现数据的输入轴、预处理、GT 构造、负样本采样和 PIR/MDP 细节；
2. 能否提供一份样例输入、对应标签、网络输出或 checkpoint，用于逐层数值对齐；
3. 学长数据是否覆盖多个距离/速度输出网格，并具有同条件背景和固定划分；
4. 对零多普勒抑制，应优先采用 MTI/杂波图/自适应滤波，还是固定 notch 加学习残差；
5. 雷达能否导出连续 H/V 复数 IQ、真实 PRF、硬件时间戳和可验证的通道相干/标定信息。

详细问题见[向学长请教清单](10_QUESTIONS_FOR_SENIOR_ZH.md)，希望获得的数据格式见
[数据需求清单](11_DATA_REQUEST_CHECKLIST_ZH.md)。

## 必须保留的边界

- 目标与背景来自不同日期，不能声称跨日期泛化；
- 六个开发折都已被研究过程使用，不是六个独立盲测；
- 完整扫描模型不是因果在线模型；
- 候选 veto、soft notch 和 Tian point-GT 都属于开发诊断；
- 没有真实空飘球载荷标签，不能报告空飘球分类准确率。
- LAT-MRICD 结果不支持 unseen-model、独立外部、极化、空飘球或 Tian 复现结论；
- D17-XBAND 的 S/Ku locked target 已消费；队员只能复核冻结证据或运行不接触真实 target
  的合成合同测试，不能重跑密封 target，也不能据此调 CNN、域适配、阈值或特征；
- HSR V2 与期刊包不得混合；HSR 不得随机拆 MAT/frame/window、使用 overflow 训练/测试、
  把 route 写成 session-disjoint，或把 512 bins 写成 Hz/速度微多普勒；DroneRFc-MM 不是 ADC/IQ、H/V 或鸟/天气/空飘球数据，不能
  替代主数据，也不能把接口审计写成模型性能；B1 不得监督对齐；
- Ku UAV 群的帧/拆分 MAT、NEXRAD 的 gate/ray/patch 都不是独立样本；两个小包只作接口
  smoke，不作识别指标，也不把 NEXRAD 双极化矩冒充本项目已标定 H/V；
- 公开数据支线完成不改变主 UAV 完成门，当前仍为 `4/6`、`BLOCKED_EXTERNAL`。
