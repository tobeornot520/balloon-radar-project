# 面向未来空飘球辨识的 H/V 双极化 UAV 检测定位前端研究

## 一句话介绍

本项目利用 H/V 双极化复数 IQ 雷达数据，研究低慢小目标的检测、距离—速度定位与背景虚警抑制。当前形成的是 UAV 检测定位前端及其可追溯、可校验的冻结结果摘录；完整复现仍需要内部源码、数据、逐样本预测和 checkpoint。空飘球有载/无载、载荷类型及运动状态识别属于后续目标，尚未由当前数据证明。

本 V13 包已合并成果材料、入组执行手册、成员资格与分工验收规则、LAT-MRICD 分组基线与
跨频段冻结负结果、DAUR V3、HSR-L V2 与 FMCWR-2.0 V4 全量只读审计，以及 Tian 复现受阻说明。
准备继续参与下一阶段的成员应先完成 `TEAM_START_HERE.md` 和成员资格办法规定的验收，再
认领模型、数据或文档任务。

## 建议阅读顺序

0. [新组员从这里开始](TEAM_START_HERE.md)：48 小时入组验收、全部任务、红线、分工和交付标准。
1. [一页式成果摘要](docs/00_ONE_PAGE_SUMMARY_ZH.md)：可直接转发的项目结论和五个核心问题。
2. [项目整体介绍](docs/01_PROJECT_OVERVIEW_ZH.md)：任务定位、技术路线和工程构成。
3. [开发历史简述](docs/02_DEVELOPMENT_HISTORY_ZH.md)：从基础定位到背景校准和联合审计的主要转折。
4. [项目历史重建与认知修订](docs/02A_HISTORICAL_PROJECT_RECONSTRUCTION_ZH.md)：对 14 份历史导出的逐阶段审计，以及早期结论与当前证据的差异。
5. [当前成果与证据](docs/03_RESULTS_AND_EVIDENCE_ZH.md)：冻结主结果、折间差异、不确定性和结论边界。
6. [近期成果与失败分析](docs/09_RECENT_PROGRESS_AND_FAILURE_ANALYSIS_ZH.md)：多域特征、零频抑制、Tian 复现和极化准备。
7. [向学长请教清单](docs/10_QUESTIONS_FOR_SENIOR_ZH.md)和[数据需求清单](docs/11_DATA_REQUEST_CHECKLIST_ZH.md)：可直接用于交流。
8. [对外分享提纲](docs/04_SHARING_TALK_TRACK_ZH.md)与[复核入口和后续计划](docs/05_REPRODUCTION_AND_NEXT_STEPS_ZH.md)。
9. [数据卡](docs/06_DATA_CARD_ZH.md)、[指标定义](docs/07_METRIC_DEFINITIONS_ZH.md)、[模型选择台账](docs/08_MODEL_SELECTION_LEDGER_ZH.md)和[新数据采集协议](docs/NEW_DATA_COLLECTION_PROTOCOL.md)：评价口径与门禁。
10. [多域特征候选放行门](docs/14_MULTIDOMAIN_FEATURE_GATE_V1_ZH.md)：当前特征的允许用途、禁止用途和下一次训练的条件。
11. [当前论文主线与写作边界](docs/15_PAPER_MAINLINE_V1_ZH.md)：当前可写贡献、主表和不能跨越的结论边界。
12. [外部事实确认可转发消息](docs/16_EXTERNAL_FACT_REQUEST_MESSAGE_V1_ZH.md)：向学长和设备方索取最小关键条件的短消息。
13. [零多普勒 P0 人工复核预筛](evidence/21_ZERO_DOPPLER_P0_REVIEW_PRESCREEN.md)：聚合结构事实、复核顺序和结论边界。
14. [下一阶段研究与数据计划](docs/17_NEXT_STAGE_PLAN_20260803_ZH.md)：任务分工、数据规格、算法放行门和停止规则。
15. [项目任务台账](docs/12_PROJECT_TASK_LEDGER_ZH.md)：持续更新的问题、分工、验收标准和下一动作。
16. [队员复现指南](docs/13_TEAM_REPRODUCTION_GUIDE_ZH.md)：按材料完整程度区分分享包复核、冻结重放和重新训练。
17. [外部公开数据核验](docs/EXTERNAL_PUBLIC_DATA_AUDIT_20260803.md)：LAT-MRICD/LSS 的来源、规模、划分风险与可用边界。
18. [LAT-MRICD 分组基线协议](docs/LAT_MRICD_GROUPED_BASELINE_PROTOCOL_V1.md)：冻结划分、特征、模型、指标和停止规则。
19. [LAT-MRICD 分组基线冻结报告](evidence/23_LAT_MRICD_GROUPED_BASELINES.md)：Narrow-X/HRRP-X 五折结果、CI、分组纪律和结论边界。
20. [LAT-MRICD 跨频段迁移冻结报告](evidence/24_LAT_MRICD_CROSS_BAND_TRANSFER.md)：一次性 S/Ku locked target 结果、`FAIL_STOP` 与禁止复用规则。
21. [DroneRFc-MM 只读审计](evidence/25_DRONERFC_MM_READ_ONLY_AUDIT.md)：PCD 完整性、6 个 base-family 分组、B1 时间错位和禁止训练边界。
22. [LSS-DAUR V3 只读审计](evidence/26_LSS_DAUR_READ_ONLY_AUDIT.md)：77 个逻辑观测、数值等价视图、候选 session、重复内容和物理轴阻塞。
23. [Tian 复现受阻说明与替代路线](docs/18_TIAN_REPRODUCTION_FAILURE_AND_ALTERNATIVES_20260803_ZH.md)：失败证据、当前不可获得条件、替代方案和重开门槛。
24. [成员资格、角色与数据权限验收办法](docs/19_TEAM_QUALIFICATION_AND_ROLE_SCREENING_ZH.md)：统一时限、五问、试做、评分、补验、角色和沟通说辞。
25. [推荐论文与下载登记](docs/20_RECOMMENDED_PAPERS_20260805.md)：Tian 前作、微多普勒、极化、载荷和有限数据检测的官方来源与阅读导引。
26. [FMCWR 只读审计与归一化处理合同](docs/LSS_FMCWR_2_READ_ONLY_AUDIT_20260805.md)：当前公开数据的事实边界、允许的单记录接口和禁止声明。
27. [零多普勒可审计虚警库 V1](docs/ZERO_DOPPLER_FALSE_ALARM_LIBRARY_V1.md)：830 个背景案例的本地/分享分层、120→109 配对变化和物理标签边界。
28. [零多普勒目标安全审计 V1](docs/ZERO_DOPPLER_TARGET_SAFETY_AUDIT_V1.md)：318 个目标的 detected/localization/joint、峰移动和分数下降配对边界。
29. [外场 H/V 复数 IQ 完整性探针 V1](docs/FIELD_IQ_INTEGRITY_PROBE_V1.md)：最小 MAT 到位后的 v5/v7.3 内容检查、设备形状门和禁止声明。

只需快速了解时，阅读一页摘要和近期失败分析；准备与学长交流时，再打开问题清单。

## 2026-07-31 至 2026-08-05 新增进展

- 完成 56 个时域、极化、时频和 RD 特征的候选锚定目录及组依赖审计；
- 定位零多普勒附近的集中虚警机制，并完成 candidate veto、固定 soft notch、
  dense-negative 和 clutter-aware 四类对照；
- 固定 notch 是确定性安全参考；第一版两种学习设置被拒绝，受约束 residual V2 后续通过
  开发门并把同口径误警从 120 降至 109，联合命中保持 290/318；
- Tian 2024 FCN 的论文指标和后处理已修正，但本地输出仍退化为固定速度模板；原条件
  目前不可获得，精确复现冻结并转入 DPG/LAT-MRICD/合同测试替代路线；
- 完成可屏蔽未标定通道的极化 ROI 迁移 encoder 接口；
- 固化 8-9 月 capability、同步、极化标定、dry run 和 Pilot 五道外场门禁。
- 增加零多普勒虚警的本地人工复核队列、P0 RD 图册、离线工作台和结果校验流程；逐样本材料
  不进入分享包，复核不构成新的模型性能结论；
- 完成 11 例 P0 人工复核审计：9 例近零多普勒峰、2 例宽结构，物理类别全部保持 unknown；
- 根据外部成果评议冻结下一阶段任务、所需数据、极化特征使用条件和算法停止规则。
- 完成 FMCWR-2.0 V4 的归一化轴/单记录处理合同：仅输出 index/normalized-frequency 接口和
  合成 smoke，不生成物理 Hz、速度、训练集或性能结论。
- 从官方补充材料取得并校验 LAT-MRICD-1.0，完成 23,191 条 HRRP/窄带 I/Q 的 schema、标签和 batch 混杂审计；原始数据不进入分享包。
- 完成 D17-NX/HX 五折 batch-code-held-out 固定基线：Narrow-X 的 batch-class macro 为
  LR 0.7999（95% CI 0.7659–0.8313）、RF 0.7872（0.7373–0.8340）；HRRP-X 分别为
  0.6617（0.5826–0.7404）和 0.6481（0.5764–0.7240）。配对差值 CI 均跨 0，不选胜者；
- 完成唯一一次 D17-XBAND 密封运行：X->S 的 LR target batch-class macro accuracy 为
  0.6517，但 UAV batch recall 仅 0.4433，未通过严格大于 0.50 的门；X->Ku 的同一指标为
  0.8400，且该 target 的全部门条件通过。由于两个 locked target 未同时通过，整体决策为
  `FAIL_STOP`。S/Ku 均已消费，禁止在同一 target 上重跑确认性比较、调 CNN、做域适配或
  结果驱动扩特征；
- 已从官方来源下载并校验 LSS-DAUR-1.0 V3、LSS-FMCWR-2.0 V4，以及 LSS-HSR-L 的期刊包
  和 ScienceDB V2 正式包。HSR V2 为 237,020,946 bytes，SHA256 为
  `fea8a21354110a96fb9644dc1c69649b6dc6d1a1b6da512498d9c2d74d839540`，ZIP 完整；
  V2/期刊包分别有 1,561/1,478 个 ZIP entries，V2 额外含 `overflow/air_routes`，且同名
  样本长度存在差异，已确认两者不等价、不可混合；
- 完成 HSR V2 的 1,530 MAT/63,148 真实帧/865 routes 全量只读审计，官方
  train/validation 默认窗口精确复现为 45,366/9,336；11 MAT/704 帧/529 窗口的 overflow
  用途未说明，保持隔离。route 只是最低 published group，source-session/场景来源和
  512-bin DPL 物理轴未知；轨迹五列单位已验证。状态为
  `PASS_SCHEMA_BLOCKED_SOURCE_PROVENANCE_AND_PHYSICAL_AXIS`，`model_training_allowed=false`。
  V2 `Dataset.py` 是只读懒加载器，移动原件警告只适用于历史期刊包；V2 的
  `CC-BY-NC-4.0` 来自 2026-08-04 ScienceDB 页面访问记录，不是 ZIP 内嵌文本；
- 完成 DAUR V3 的 308 个 MAT 全量只读审计：154 canonical 与 154 backup 是同一 77 个
  逻辑观测的不同存储视图，共享数值完全相等；其中一组两个 recording 的 TD/TR 内容完全
  重复，因此只有 76 个唯一内容对。77 对 TD/TR 对齐，但全部轨迹有重复时间，6 个日期
  冲突，58/19 条分别为 512/1024 bins；11 对 recording 共享内部帧，保守连通后只有 39 个
  候选 source-session 组。状态为
  `PASS_SCHEMA_PAIRING_BLOCKED_GROUPING_AND_PHYSICAL_AXIS`，未放行训练；
- 完成 FMCWR-2.0 V4 的 6 个 RAR/90 MAT 全量只读审计：64 K/26 L，B 通道全空；原始与
  解码数值均只有 71 个唯一 payload，11 个精确重复组覆盖 30 文件，66 stems 连通为 48 个
  非权威候选组。状态为
  `PASS_ARCHIVE_SCHEMA_BLOCKED_GROUPING_PROVENANCE_AND_PHYSICAL_AXIS`；session、Fs/PRF、
  载频、零频和物理轴未确认，禁止训练、随机切窗、H/V/自然鸟和物理 Hz/速度声明；
- 新增 DroneRFc-MM V1 的选择性 radar subset：数据 DOI `10.57760/sciencedb.j00173.00094`，
  `CC-BY-SA-4.0`。完整发布为 113 files、75,612,067,287 bytes，本地只下载 28 个雷达相关
  文件，共 47,366,902 bytes；9 个 mmRadar PCD ZIP 均完整。全量只读审计验证了 30,717 个
  PCD、639,527 个有限点和时间戳；8 条 radar/GT 时间范围重叠，B1 零重叠并被冻结；
- 只增加两个低成本接口样本：3,996,753-byte Ku 波段 UAV 群目标包只含 3 个物理实验，
  用于量测/航迹 smoke；395,379-byte KTLX NEXRAD Level II 单体扫实际含 Z、V、SW、ZDR、
  PhiDP、RhoHV，用于双极化 loader/公式 smoke。两者都不进入识别训练；
- 已登记但暂缓 42.4 GB 室内铝箔数字气球、23.46 GB S 波段野外 UAV 和 30.36 GB
  24/94/207 GHz UAV/真实鸟主数据；暂缓原因、许可、分组单位和禁止声明均写入来源台账；
- DAUR、HSR 与 FMCWR 当前只读审计均完成但训练阻塞；FMCWR 归一化轴合成/单记录处理
  合同和合成 smoke 已完成，下一步只维护 grouping/provenance/axis 门并等待可归因参数。DroneRFc-MM 保持
  `PASS_SCHEMA_BLOCKED_TIMESTAMP_ALIGNMENT`，B1 等待更正 GT/可归因偏移；审计通过前不训练。

## 当前最重要的结论

| 方法 | 六折虚警数 | 正确检测 | 当前定位 |
|---|---:|---:|---|
| 完整扫描上下文 BC-DPG-FCN v3 | 56/830 | 289/318 | 当前六折内部开发评价中表现最好的离线扫描感知方案 |
| 样本独立 BC | 122/830 | 289/318 | 单独训练、更接近在线条件的部署导向基准 |
| 完整模型 + leave-one-out | 54/830 | 289/318 | 测试后 self-inclusion 敏感性；仍使用未来样本 |
| 完整模型 + past-only all-history | 93/830 | 288/318 | 测试后假定顺序敏感性；不能据此选窗口 |
| Power2 baseline | 300/830 | 268/318 | ROI 候选基线 |
| ROI RI4 | 196/830 | 268/318 | 独立极化抑制研究 |

完整扫描上下文 BC-DPG 可使用同一扫描中晚于当前样本的信息，因此 56 个虚警的结果应视为离线扫描感知上限，不是严格实时部署性能。其 56 个虚警全部集中在 Fold 1 和 Fold 4；最差折 Pfa 为 0.280，最差折 joint Pd 为 0.7917，困难背景鲁棒性仍是主要瓶颈。

leave-one-out 与 past-only 数字来自冻结完整 checkpoint 的推理上下文替换，没有因果重训练。past-only 顺序由 `(beam_layer, azimuth_deg, sample_id)` 推断，未由逐样本采集时间戳验证；`54/830` 和 `93/830` 只能作为测试后敏感性诊断。下一版需按真实时间顺序重新训练，并只在训练/验证集选择历史窗口。

后续就绪审计检查了全部 1,148 个 MAT 文件：没有时间戳变量，MAT 头时间至少晚于文件名时间 49.1 天并贴近文件 mtime，因此正式因果训练门禁仍关闭。当前只完成了一个不加载测试 split 的单折小样本接口 smoke，不构成性能证据。

冻结定位汇总进一步区分了分数检测与定位：318 个目标中 302 个达到冻结阈值，297 个不考虑分数时位于 2 gates / 3 bins 容差内，289 个同时满足两项要求。距离/速度误差的中位数与 P90 较低，但最大值达到 39 gates / 40 bins，说明仍有少量大误差长尾。

![冻结定位误差 CDF](assets/figures/bc_dpg_localization_error_cdf.png)

下一批数据已经有统一进入标准：40 列采集合同覆盖真实 UTC 时间、硬件序号、时钟重置、事件时长、SNR、雷达/标定版本、同条件目标/背景和外层隔离。当前 V4 清单缺少其中 33 列，在 `locked_evaluation` 档明确为 FAIL；旧测试集和推断顺序不会被重新命名为锁定证据。

BC-DPG 与 ROI RI4 的 OR/union 可得到 294/318 个正确目标，但虚警升至 216；AND/intersection 可把虚警降至 36，但正确检测降至 263。两者只用于测试后的互补性诊断，没有作为部署规则。

![固定阈值检测权衡](assets/figures/joint_pooled_detection_tradeoff.png)

## 必须同时阅读的证据限制

- Stage 4 使用 Fold 1 和 Fold 4 筛选扩展模式，最终六折 ROI 汇总又包含这两个折，因此不是独立盲测估计。
- 冻结审计的目标扫描组全部来自 `20260202`，背景扫描组全部来自 `20260204`，类别与采集日期完全耦合。
- 当前只有 6 个背景扫描组；扫描组 bootstrap 的 BC-DPG Pfa 95% 区间为 0 至 0.1618，背景不确定性较大。
- 因果上下文审计测试了 4/16/64/all-history 多个窗口，结果不得反过来用于选择部署窗口。
- “测试阈值未重调”不等于结构、损失、表征和研究叙事未受开发结果影响。

## 包内内容

- `docs/`：一页摘要、项目历史、完整结果、近期失败分析、请教与数据需求清单。
- `assets/figures/`：用于介绍的精选 PNG 和 PDF 图件。
- `assets/tables/`：冻结主结果、折间分布、定位误差、区间估计、配对诊断、上下文敏感性和采集顺序来源 CSV。
- `assets/contracts/`、`assets/templates/`：新数据采集合同、FMCWR 归一化处理合同和空白清单模板。
- `scripts/process_lss_fmcwr_normalized_v1.py`：不接触 RAR/MAT 的归一化 FFT/STFT 合成 smoke 工具；
- `scripts/build_zero_doppler_false_alarm_library_v1.py`：重建本地逐样本库并输出脱敏聚合证据；
- `scripts/audit_zero_doppler_target_safety_v1.py`：审计 residual 的目标检测、定位与峰移动风险；
- `scripts/audit_field_iq_integrity_v1.py`：只读检查现场 MAT 的 H/V 复数、有限值、I/Q 变化与设备形状；
  它只验证接口，不产生模型性能。
- `TEAM_START_HERE.md`：给零项目背景组员的完整执行手册；组员应先读此文件再认领任务。
- `docs/19_TEAM_QUALIFICATION_AND_ROLE_SCREENING_ZH.md`：成员筛选、分工、权限和沟通的统一规则。
- `assets/templates/team_qualification_scorecard_template_v1.csv`：逐人评分与决定记录。
- `evidence/`：四个阶段的冻结结论或正式报告，以及上下文敏感性、因果训练就绪、定位证据、LAT-MRICD 审计/分组基线/跨频段冻结负结果、DroneRFc-MM、DAUR、HSR-L 与 FMCWR-2.0 只读审计和当前数据合同缺口审计。
- `MANIFEST.json`：版本、范围、源文件及 SHA256 哈希。
- `SHA256SUMS.txt`：包内文件完整性校验值。

## 分享边界

本包不包含原始 MAT/IQ/PCD、外部数据压缩包、标签明细、逐样本预测、checkpoint、训练日志、开发聊天记录、个人路径或访问凭据。哈希只能校验包内文件是否变化，不能替代从源码和数据重新计算结果。LAT-MRICD 数字只属于同一公开发布内的 batch-code-held-out 基线或 band-held-out 迁移，不是 unseen-model、独立外部、极化、空飘球或 Tian 复现证据。D17-XBAND 的 S/Ku target 已消费，分享包接收者只能复核证据或运行不接触真实 target 的合成合同测试，不能把重跑、CNN、域适配或调参称为新的确认性结果。DAUR 的 canonical/backup 不得倍增，TD/TR 和保守连接组不得跨 split；随机 frame/window/MAT 拆分、静默修日期、把 1024-bin 转成未经说明的物理 Hz 轴以及任何 DAUR 模型性能均禁止。FMCWR-2.0 的 90 MAT 只有 71 个唯一 payload，48 个候选组不是发布方确认的 session，B 通道全空；成员级哈希不在包内，禁止随机 MAT/frame/window、模型性能、H/V、自然鸟和物理 Hz/速度声明。DroneRFc-MM 子集不是 ADC/IQ 或 H/V 数据，没有鸟、天气或空飘球对照，只能用于点云/轨迹接口和时序算法审计，不能替代主数据或宣布识别性能；B1 radar/GT 零重叠，禁止监督对齐。UAV 群和 NEXRAD 小样本也只用于接口 smoke，不能按帧/gate/patch 随机拆分或写成识别结果；三个大体量候选没有进入本包或本地训练。主 UAV 方向仍为 `4/6`、`BLOCKED_EXTERNAL`；其余包内数字属于当前数据上的内部开发评价或明确标注的诊断证据，不代表跨日期、跨场地盲测或严格实时部署。

分享包版本：`2026-08-06 V13`
