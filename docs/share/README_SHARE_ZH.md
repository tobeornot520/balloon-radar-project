# 面向未来空飘球辨识的 H/V 双极化 UAV 检测定位前端研究

## 一句话介绍

本项目利用 H/V 双极化复数 IQ 雷达数据，研究低慢小目标的检测、距离—速度定位与背景虚警抑制。当前形成的是 UAV 检测定位前端及其可追溯、可校验的冻结结果摘录；完整复现仍需要内部源码、数据、逐样本预测和 checkpoint。空飘球有载/无载、载荷类型及运动状态识别属于后续目标，尚未由当前数据证明。

## 建议阅读顺序

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

只需快速了解时，阅读一页摘要和近期失败分析；准备与学长交流时，再打开问题清单。

## 2026-07-31 至 2026-08-03 新增进展

- 完成 56 个时域、极化、时频和 RD 特征的候选锚定目录及组依赖审计；
- 定位零多普勒附近的集中虚警机制，并完成 candidate veto、固定 soft notch、
  dense-negative 和 clutter-aware 四类对照；
- 固定 notch 是当前开发安全参考，两种学习设置未通过 Fold 1/4 门槛，不扩展六折；
- Tian 2024 FCN 的论文指标和后处理已修正，但本地输出仍退化为固定速度模板，当前
  转向向学长核对数据与复现条件；
- 完成可屏蔽未标定通道的极化 ROI 迁移 encoder 接口；
- 固化 8-9 月 capability、同步、极化标定、dry run 和 Pilot 五道外场门禁。
- 增加零多普勒虚警的本地人工复核队列、P0 RD 图册、离线工作台和结果校验流程；逐样本材料
  不进入分享包，复核不构成新的模型性能结论；
- 根据外部成果评议冻结下一阶段任务、所需数据、极化特征使用条件和算法停止规则。
- 从官方补充材料取得并校验 LAT-MRICD-1.0，完成 23,191 条 HRRP/窄带 I/Q 的 schema、标签和 batch 混杂审计；原始数据不进入分享包。

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
- `assets/contracts/`、`assets/templates/`：新数据采集合同和空白清单模板。
- `evidence/`：四个阶段的冻结结论或正式报告，以及上下文敏感性、因果训练就绪、定位证据和当前数据合同缺口审计。
- `MANIFEST.json`：版本、范围、源文件及 SHA256 哈希。
- `SHA256SUMS.txt`：包内文件完整性校验值。

## 分享边界

本包不包含原始 MAT/IQ 数据、标签明细、逐样本预测、checkpoint、训练日志、开发聊天记录、个人路径或访问凭据。哈希只能校验包内文件是否变化，不能替代从源码和数据重新计算结果。包内数字属于当前数据上的内部开发评价或明确标注的两折诊断证据，不代表跨日期、跨场地盲测、严格实时部署或空飘球载荷分类性能。

分享包版本：`2026-08-03`
