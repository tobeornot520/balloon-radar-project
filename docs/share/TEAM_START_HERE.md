# 新组员从这里开始：入组、认领任务与交付手册

版本：2026-08-06 V8（随 V13 分享包发布）

适用对象：此前没有参与过项目开发的新组员，以及准备继续参与下一阶段的现有成员

负责人：项目负责人指定

执行原则：先通过入组验收，再认领任务；先确认数据和评价边界，再运行模型。

## 1. 完成本手册后应达到什么状态

新组员不能只做到“看过分享包”。正式认领任务前，应能独立完成以下事项：

1. 用三分钟说明项目的长期目标、当前任务和当前不能声称的结果；
2. 区分当前 H/V UAV 数据、LAT-MRICD、HSR/DAUR/FMCWR/DroneRFc-MM 公开数据和未来
   空飘球数据；
3. 解释为什么当前 56 个虚警结果不是严格实时或外部盲测结论；
4. 解释为什么没有 PRF 时不能报告有 Hz 单位的物理微多普勒；
5. 完成分享包校验，具备源码权限时还要通过环境和测试验收；
6. 填写任务认领表，明确输入、输出、分组方式、测试访问规则和停止条件；
7. 按统一周报模板汇报成功、失败和阻塞，不只汇报最好的数字。

没有完成以上验收，不直接分配正式训练、阈值选择或论文数字整理工作。

成员角色、评分、补验和数据权限以
[成员资格与分工验收办法](docs/19_TEAM_QUALIFICATION_AND_ROLE_SCREENING_ZH.md)为准。

## 2. 先建立统一项目认知

### 2.1 长期目标

项目长期希望融合时域、极化、微多普勒、距离-速度、轨迹和行为信息，逐级完成：

```text
目标/背景检测
  -> 空飘球/其他低空目标区分
  -> 空飘球有载/无载识别
  -> 载荷类型或运动状态识别
```

### 2.2 当前真正完成的工作

当前数据主要覆盖 UAV 和背景。现阶段形成的是：

- H/V 双通道 UAV 检测和距离-速度定位前端；
- 样本独立与扫描感知背景校准；
- 候选 ROI 极化精修和联合审计；
- 零多普勒虚警机制诊断与目标保护残差开发参考；
- Tian 2024 FCN 方法迁移、失败诊断和复现条件清单；
- 新数据合同、外场能力、同步、标定、dry run 和 Pilot 门禁；
- LAT-MRICD 公开 HRRP/窄带 I/Q 数据的正式 schema/batch 审计、D17-NX/HX 五折分组基线，
  以及 D17-XBAND 唯一一次密封迁移与 `FAIL_STOP` 冻结负结果；
- DAUR V3 的 308 个 MAT 全量只读审计：77 个逻辑 TD/TR 观测只有 76 个唯一内容对，
  canonical/backup 数值等价；39 个保守候选 source-session 组、严格时间、日期和 1024-bin
  轴仍阻塞，禁止训练；
- FMCWR-2.0 V4 的 6 个 RAR/90 MAT 已完成只读 schema/重复审计：64 K/26 L，B 通道全空；
  71 个唯一 payload、11 个重复组/30 个成员、48 个非权威候选组已冻结，session/物理轴
  阻塞。HSR 两包已确认不等价；V2 的 1,530 MAT/63,148 帧/865 routes 已完成只读审计，
  但 source provenance/512-bin 物理轴阻塞。DroneRFc PCD schema 已通过但 B1 radar/GT
  零重叠，均不能因“已下载”或“schema 通过”写成“已可训练”。

### 2.3 不同证据对象禁止混用

| 证据对象 | 当前可做什么 | 当前不能证明什么 |
|---|---|---|
| 现有 H/V UAV/背景数据 | 检测、网格定位、内部虚警机制、相对极化工程特征 | 跨日期泛化、严格实时部署、空飘球分类、绝对极化物理量 |
| LAT-MRICD-1.0 | 同一公开发布内的 HRRP、归一化频率微动、batch-code-held-out UAV/鸟/气象基线与已冻结 band-held-out UAV/weather 评价 | unseen-model/独立外部泛化、H/V 极化、物理 Hz 微多普勒、空飘球状态、同事件跨频融合；已消费 S/Ku 不能重跑确认性比较 |
| DAUR V3 | 复核 schema/配对/等价性与 39 个保守候选组；设计不使用标签的未来分组方案 | 当前模型性能、随机 MAT/frame/window、backup 倍增、1024-bin 物理 Hz、独立 session 结论 |
| FMCWR-2.0 | 复核 90 MAT/71 唯一 payload/48 候选组和已完成的归一化轴合成/单记录处理合同 | 模型性能、随机 MAT/frame/window、把候选组当 session、物理 Hz/速度、H/V 极化、仿真鸟的自然鸟结论 |
| HSR ScienceDB V2/期刊包 | V2 的 schema/route 与官方窗口计数已审计；按归一化 bin/轨迹复核接口 | 混包、使用 overflow、随机 MAT/frame/window split、把 route 当 session、物理 Hz/速度轴、训练或独立外部性能 |
| DroneRFc-MM radar subset | 复核全量 PCD schema 和 8 条时间重叠 recording；另行预登记后做点云/轨迹接口 | B1 监督对齐；ADC/IQ、H/V、鸟/天气/空飘球；随机 frame/window split；宣布性能 |
| Ku UAV 群 / NEXRAD smoke | 分别验证量测/航迹接口和 Z/V/SW/ZDR/PhiDP/RhoHV 读取/公式 | 把 3 个实验或 1 个体扫扩成独立训练样本；UAV/气球识别性能；把天气雷达矩冒充本项目 H/V IQ |
| 未来同步空飘球数据 | 通过合同后开展极化、时域、微多普勒和分层分类 | 在真实数据到位和锁定评价前不能预报性能 |

任何报告都要写明使用的是哪一个证据对象，不允许把三者的结论拼成一个看似完整的模型结果。

## 3. 当前状态必须会复述

- 完整扫描 BC-DPG：`56/830` 背景误警、`289/318` 联合检测定位成功；它可以使用同一扫描
  后续样本，是离线扫描感知上限。
- 样本独立 BC：`122/830` 背景误警、`289/318` 联合成功；它更接近在线导向基线。
- fixed notch + target-protected residual V2：CPU 重推理下 `109` 个误警、`290/318`
  联合成功；它是开发参考，不是盲测或部署结论。
- D17-NX/HX：Narrow-X 的 batch-class macro 为 LR `0.7999`、RF `0.7872`，HRRP-X 为
  `0.6617`、`0.6481`；四项均有 batch-code cluster 95% CI，两个配对差值 CI 均跨 0。
- D17-XBAND：X->S LR target batch-class macro accuracy 为 `0.6517`，但 UAV batch recall
  `0.4433` 未通过严格 `>0.50` 门；X->Ku 的同一指标为 `0.8400` 且该 target 过门。整体为
  `FAIL_STOP`，S/Ku 已消费，禁止同 target 重跑、CNN、域适配或结果驱动改模。
- Tian 本地迁移没有成功复现论文。point-GT 只是 Fold 1 train/validation 上的本地消融。
- 当前目标与背景日期完全耦合，六个折都参与过开发，不能称外部盲测。
- 当前方向完成门为 `4/6`，状态是 `BLOCKED_EXTERNAL`；剩余两项是外部阻塞，没有待成员
  通过猜测补齐的物理事实。

只在检查器输出 `COMPLETE` 后，负责人才能宣布“当前方向任务已全部完成”。

## 4. 48 小时基础入组流程

本节完成基础阅读与操作检查；随后还需按成员资格办法完成第 7 天试做和角色确认。

### 阶段 A：收到包后的前 30 分钟

1. 解压分享包，不移动包内相对目录；
2. 在包含 `SHA256SUMS.txt` 的目录执行：

```bash
sha256sum -c SHA256SUMS.txt
```

3. 所有文件必须显示 `OK`；若有失败，停止使用该包并联系负责人；
4. 阅读根目录 `README.md` 和本手册，不先翻模型代码。

### 阶段 B：第一轮阅读

按以下顺序阅读，不要求一次看完全部 96 个以上的文件：

| 顺序 | 文件 | 需要回答的问题 |
|---:|---|---|
| 1 | `docs/00_ONE_PAGE_SUMMARY_ZH.md` | 项目做什么、做到哪一步？ |
| 2 | `docs/03_RESULTS_AND_EVIDENCE_ZH.md` | 当前主要数字是什么、证据角色是什么？ |
| 3 | `docs/09_RECENT_PROGRESS_AND_FAILURE_ANALYSIS_ZH.md` | 哪些尝试失败，为什么停止？ |
| 4 | `docs/17_NEXT_STAGE_PLAN_20260803_ZH.md` | 下一阶段按什么顺序推进？ |
| 5 | `docs/12_PROJECT_TASK_LEDGER_ZH.md` | 当前有哪些问题、谁负责、怎样验收？ |
| 6 | `docs/13_TEAM_REPRODUCTION_GUIDE_ZH.md` | 自己手里的材料允许做到哪种复现等级？ |
| 7 | `docs/EXTERNAL_PUBLIC_DATA_AUDIT_20260803.md` | LAT-MRICD 为什么不能随机拆行？ |
| 8 | `evidence/23_LAT_MRICD_GROUPED_BASELINES.md` | D17-NX/HX 得到什么、CI 和结论边界是什么？ |
| 9 | `evidence/24_LAT_MRICD_CROSS_BAND_TRANSFER.md` | 为什么 Ku 过门仍是整体 `FAIL_STOP`，且 S/Ku 不能重跑？ |
| 10 | `evidence/25_DRONERFC_MM_READ_ONLY_AUDIT.md` | 为什么 B1 被阻塞、为什么最低按 6 个 base family 分组且当前禁止训练？ |
| 11 | `evidence/26_LSS_DAUR_READ_ONLY_AUDIT.md` | 为什么 308 个 MAT 只能算 77 个逻辑观测、为何 schema 通过仍禁止训练？ |
| 12 | `docs/18_TIAN_REPRODUCTION_FAILURE_AND_ALTERNATIVES_20260803_ZH.md` | Tian 为什么冻结、替代路线是什么？ |
| 13 | `docs/19_TEAM_QUALIFICATION_AND_ROLE_SCREENING_ZH.md` | 如何验收、评分、补验和分配权限？ |

### 阶段 C：口头验收

每人用约八分钟回答：

1. 为什么 `56/830` 不能写成严格实时部署性能？
2. Tian 2024 为什么不能称为复现成功？
3. 为什么当前 H/V 不能直接称为完整极化特征？
4. 为什么 LAT-MRICD 不能随机拆行，D17-XBAND 又为什么不能重跑 S/Ku？
5. 当前还有什么能做、什么不能做，自己准备认领什么任务？

答不清楚时返回对应文档，不通过猜测补答案。

### 阶段 D：操作验收

只有分享包权限的成员：完成 SHA256 校验并填写个人入组清单。

具有完整仓库权限的成员，在项目根目录执行：

```bash
conda env create -f environment.yml
conda activate radar-torch
python scripts/check_project_health.py
python -m pytest
git status --short
```

已有 `radar-torch` 环境时不要重复创建。验收要求：健康检查通过、测试通过、没有来源不明的
工作区改动。失败时记录完整命令和错误，不自行替换依赖版本掩盖问题。

## 5. 数据访问等级

| 等级 | 可获得材料 | 允许工作 | 禁止工作 |
|---|---|---|---|
| A | 合并分享包 | 阅读、核对哈希、审阅表格和结论边界 | 重新训练、宣称复现指标 |
| B | 完整 Git，无原始数据 | 测试、接口审阅、代码和文档任务 | 报告模型性能 |
| C | Git + 指定 manifest/冻结预测 | 重建审计、图表和汇总 | 改测试阈值后覆盖冻结证据 |
| D | Git + 受控原始数据 + 配置 | 预登记后训练或重新训练 | 把新训练数字冒充历史冻结结果 |
| E | 设备/外场权限 | 能力核验、采集、同步、标定和 Pilot | 未通过门禁直接正式采集或论文评价 |

负责人按最小必要原则授予数据权限。原始 MAT/IQ、checkpoint、日志和包含个人路径的清单不在
聊天群、公共网盘或分享包中传播。

权限等级 D 也不授权重跑 D17-XBAND：S/Ku locked target 已在唯一一次密封运行中消费。
成员对该实验只能复核冻结证据或运行完全合成的接口/门禁测试，不能通过复制文件、改路径、
改实验 ID、CNN 或域适配规避停止规则。

## 6. 任务认领方法

使用 [任务认领表](assets/templates/team_task_claim_template_v1.csv)。每个任务必须具备：

- 唯一 `task_id`；
- 一名 owner 和一名 reviewer；
- 明确输入路径或 manifest；
- 明确输出路径，不覆盖冻结目录；
- 分组键和测试集访问规则；
- 主指标、完成门和停止条件；
- 禁止表述；
- 目标日期、阻塞项和下一动作。

没有 reviewer、划分方式或完成标准的任务不开始。一个人同时最多承担一个主任务和一个维护
任务，避免所有任务都处于“进行中”。

## 7. 当前可立即认领的任务

| 任务 ID | 优先级 | 当前状态 | 推荐角色 | 必须交付 |
|---|---:|---|---|---|
| ONB-01 | P0 | 立即可做 | 所有新人 | 完整入组清单、口头验收和任务认领表 |
| D04-P0 | P0 | 已完成 | 雷达/数据分析 | 11 例 P0 复核 CSV 已通过审计；聚合结论已冻结 |
| D01-D02 | P0 | 外部阻塞；当前不可获得 | 外部数据/设备方 | 有来源时提供 H/V、IQ、PRF、坐标事实；无来源保持 unknown |
| D10 | P0 | 精确复现冻结 | 模型复现 | 只维护方法级负结果与合同测试；满足重开条件前不扫参 |
| D17-NX | P0 | 已完成 | 信号处理/ML | Narrow-X 五折分组结果、CI、特征和边界已冻结到证据 23 |
| D17-HX | P1 | 已完成 | 雷达识别 | HRRP-X 五折分组结果、CI、子型号压力和边界已冻结到证据 23 |
| D17-XBAND | P1 | 已完成；`FAIL_STOP` | 证据复核 | 核对证据 24、门判定和消费记录；禁止重跑真实 S/Ku target |
| PUB-DAUR | P0 | 只读审计完成；grouping/axis 阻塞 | 数据/信号处理 | 复核证据 26、验证禁止训练边界，或提出不使用标签的保守 group 方案；不得重新包装成开放训练任务 |
| PUB-HSR | P1 | V2 只读审计完成；provenance/axis 阻塞 | 数据工程/证据复核 | 复跑 1,530 MAT/63,148 帧/865 routes 审计并核对 overflow 隔离、五列单位和禁止训练边界；禁止混包 |
| PUB-FMCWR2 | P1 | 只读审计与归一化合同完成；grouping/provenance/axis 阻塞 | 雷达/数据工程 | 复核证据 28 的 90 MAT/71 payload/48 候选组、v5/v7.3 读取和归一化 smoke；不训练 |
| PUB-DRONERFC | P1 | schema 已完成；B1 同步阻塞 | 证据/点云 | 复核证据 25、B1 零重叠和 base-family group；无更正材料不做 B1 监督对齐 |
| D15 | P0 | 持续 | 工程/复现 | 实验台账、配置哈希、提交号和结果摘要检查 |
| D16 | P1 | 持续 | 文档/工程 | Git 清洁、包内敏感信息审计、最新和上一版分享包 |

### 7.1 D04-P0：11 例人工复核

输入：本地 `results/data_audit/zero_doppler_review_atlas_v1/review_workbench.html`。

执行要求：

1. 依次查看 11 张 RD 图；
2. 只记录 `near_zero_doppler_peak`、`broad_structure`、`multiple_peaks`、`edge_peak` 或
   `no_clear_pattern` 等可见结构；
3. 没有独立现场记录时 `physical_class` 必须保持 `unknown`；
4. 每例填写中性备注，例如“零参考线附近有窄峰，仅凭图像不能确定物理来源”；
5. 导出独立 CSV，不覆盖源队列；
6. 运行人工复核审计并把摘要交给负责人。

完成标准：11 例均不再是 `pending`，导出 CSV 通过校验；不要求强行给出物理类别。

2026-08-03 状态：11/11 已通过审计，其中 9 例为 `near_zero_doppler_peak`、2 例为
`broad_structure`，物理类别全部保持 `unknown`。逐样本 CSV 只留在受控本地目录。

### 7.2 D17-NX：窄带公开数据基线

输入：`data/raw/external/LAT-MRICD-1.0/`。正式 ZIP SHA256 应为：

```text
2fe0d5e89016382c7c980172d67ba640179d6e2724edc735bcdf65c66b533bc0
```

开始前执行：

```bash
python scripts/audit_lat_mricd_dataset_v1.py --overwrite
```

冻结状态：

- 已完成 Narrow-X 的 UAV/鸟/气象大类五折评价；
- 按 `(representation, band_code, batch_code)` 分组，禁止随机拆行；
- 特征使用归一化包络、相位增量、自相关、归一化谱、谱熵和谱展宽等；
- 无 PRF 时频率只写 cycles/sample 或 normalized frequency；
- 固定 batch-balanced LR/RF 的 batch-class macro 为 0.7999/0.7872，对应 95% CI 为
  0.7659–0.8313/0.7373–0.8340；配对差值 CI 跨 0，不选胜者；
- macro、每类、最差组、混淆矩阵、batch 覆盖和 CI 已进入冻结聚合证据。

阅读入口：`evidence/23_LAT_MRICD_GROUPED_BASELINES.md`。该结果只属于同一公开发布内、
已见子型号的 batch-code-held-out 基线。

### 7.3 D17-HX：HRRP 公开数据基线

该任务已按与 D17-NX 相同的分组纪律完成。归一化幅度几何、熵、粗糙度和自相关特征上的固定
LR/RF batch-class macro 为 0.6617/0.6481，对应 95% CI 为 0.5826–0.7404/
0.5764–0.7240；配对差值 CI 跨 0。样点不解释为已标定绝对距离或 RCS，HRRP-X 也不是
Narrow-X 的独立外部验证。

### 7.4 D17-XBAND：跨频段冻结负结果

该任务已经按提交绑定的预登记协议执行唯一一次正式运行：

- Narrow X->S 的固定 LR target batch-class macro accuracy 为 0.6517，UAV/weather batch
  recall 为 0.4433/0.8600；UAV recall 未严格大于 0.50；
- Narrow X->Ku 的 macro 为 0.8400，UAV/weather recall 为 0.8493/0.8307，该 target
  全部门条件通过；
- 两个 target 未同时通过，整体 gate 为 `FAIL_STOP`；S/Ku 均已消费。

组员交付只能是证据 24 的数字、哈希、门判定、模型拟合范围和结论边界复核，或以下不接触
真实 target 的合成合同测试：

```bash
python -m pytest tests/test_lat_mricd_cross_band_transfer.py \
  tests/test_lat_mricd_cross_band_evidence.py
```

不得重跑正式 runner，不得用同一 S/Ku 调 CNN、域适配、阈值或特征，也不得把 X->Ku 的通过
单独写成“跨频段迁移成功”。阅读入口：`evidence/24_LAT_MRICD_CROSS_BAND_TRANSFER.md`。

### 7.5 D10：Tian 复现条件对齐

优先获取：一个输入、网络前张量、标签图、网络输出、PIR/MDP 输出或 checkpoint。收到材料
后先对齐一个样本，不直接跑六折。没有取得论文同域数据或明确配置前，禁止继续扫随机负样本、
PIR 阈值、V/HV 输入或更多 epoch。

截至 2026-08-03，所需外部材料被报告为目前无法取得。精确复现保持冻结，执行 DPG-FCN
零多普勒主线、公开数据 schema/group 审计和 Tian 合同测试。完整失败链、替代路线及唯一重开条件
见 `docs/18_TIAN_REPRODUCTION_FAILURE_AND_ALTERNATIVES_20260803_ZH.md`。

### 7.6 新公开数据审计

DAUR V3 已冻结为 `PASS_SCHEMA_PAIRING_BLOCKED_GROUPING_AND_PHYSICAL_AXIS`。154 canonical
与 154 backup 只代表 77 个逻辑观测，共享数值完全相等；一组 2 个 recording 的 TD/TR
内容完全重复，只剩 76 个唯一内容对。11 对 recording 共享内部帧，45/40 个字段候选组
保守连通后得到 39 个 source-session 候选组，但没有作者确认。全部轨迹含重复时间，另有
6 个日期冲突和 58/19 条 512/1024-bin 混宽。成员可以复核证据、测试 loader 或设计保守
分组，不得随机拆分、倍增 backup、静默修日期、声称物理 Hz 轴或训练模型。

HSR ScienceDB V2 正式包为 237,020,946 bytes，SHA256 为
`fea8a21354110a96fb9644dc1c69649b6dc6d1a1b6da512498d9c2d74d839540`，ZIP 完整。
V2/期刊包分别有 1,561/1,478 个 entries；V2 额外有 `overflow/air_routes`，同名样本长度也
不同，因此两包不可混合。V2 全量审计冻结 1,530 MAT、63,148 真实帧、865 routes；train/
validation 为 45,366/9,336 个官方默认窗口，11 MAT/704 帧/529 窗口的 overflow 保持隔离。
全部 MAT 的 DPL/轨迹 schema 通过，轨迹五列单位已验证；但 route 仅是最低 published group，
没有权威 source-session/场景来源，512-bin DPL 也缺物理 Hz/速度轴。状态为
`PASS_SCHEMA_BLOCKED_SOURCE_PROVENANCE_AND_PHYSICAL_AXIS`，`model_training_allowed=false`。
V2 自带 `Dataset.py` 是只读懒加载器；会移动原件的是不等价期刊历史包的旧 `dataset.py`，
该警告不能错写到 V2。`CC-BY-NC-4.0` 来自 2026-08-04 ScienceDB 页面访问记录，不是 ZIP
内嵌文本。

DroneRFc-MM V1 数据 DOI 为 `10.57760/sciencedb.j00173.00094`，许可为
`CC-BY-SA-4.0`。完整发布是 113 files、75,612,067,287 bytes，本地只下载 28 个雷达相关
文件（1 README、9 PCD ZIP、9 GT CSV、6 labels、3 code），共 47,366,902 bytes；subset
manifest SHA256 为 `6b0c2ed1a075aa9164a516af001b630a9f775fddc9f399223c1aeeb6e7047b2b`。
9 个 ZIP 均完整；30,717 PCD frames、639,527 points 全部通过 15 列 schema、finite、POINTS
行数和嵌入/文件名时间戳检查，字段有 doppler、power、snr、timestamp。

该 subset 只有 9 recordings、6 UAV models，且同日同场景。717 个派生 5 秒 windows 必须
回连原始 recording，不能随机拆 frame/window。它不是 ADC/IQ 或 H/V，没有鸟、天气、
空飘球。8 条 radar/GT 时间范围重叠；B1 雷达结束约 8 分钟后同名 GT 才开始，零重叠，
所以状态为 `PASS_SCHEMA_BLOCKED_TIMESTAMP_ALIGNMENT`。B1 等待更正 GT/可归因偏移；其余
8 条也只有另行预登记后才允许点云/轨迹接口研究，不允许替代主数据或宣布模型性能。

## 8. 数据或设备到位后才能启动的任务

| 任务 ID | 开启条件 | 主要工作 | 完成标准 |
|---|---|---|---|
| D04-LIB | 11 例 P0 审计完成 | 合并模型变化、结构、证据来源，建立虚警库 | 命名类别可追溯，unknown 比例保留 |
| D05-D07 | 新同条件 H/V 数据与标定通过 | 相对极化统计、最小 ROI 增强 | 困难验证组 Pfa 降低且目标保护门通过 |
| D08 | 真实硬件顺序/时间戳可用 | past-only 因果背景校准 | 不读未来样本，窗口只由 train/val 选择 |
| D11 | 连续慢时间、PRF、事件标签可用 | 物理 STFT、脊线、周期和时序模型 | 时间/频率单位、连续性和回放验证通过 |
| D12 | 同步和极化标定证据通过 | 相位、相关和绝对极化候选 | capability/sync/calibration/dry run/Pilot 全通过 |
| D13 | 空飘球分层数据合同通过 | 球/其他、有载/无载、状态分层识别 | session 隔离，locked test 只评价一次 |
| FIELD-01 | 设备可操作 | 300 秒写盘、同步、标定、四场景 Pilot | 五道外场门均有真实证据文件 |

门禁未通过时可以完善接口、测试、模板和预登记，不能用模拟成功代替真实证据。

## 9. 每项任务的标准执行流程

### 9.1 认领

负责人在任务认领表中批准 owner、reviewer、范围和目标日期。成员确认自己具备所需数据等级。

### 9.2 预检

```bash
git status --short
git rev-parse HEAD
python scripts/check_project_health.py
python -m pytest
```

记录输入 manifest、数据哈希、配置和已有证据。发现工作区存在他人改动时不得删除或覆盖。

### 9.3 预登记

在读取 locked test 前写明：研究问题、输入特征、分组单位、训练/验证/测试划分、主指标、容差、
停止条件、允许尝试次数和禁止表述。

已经消费且触发停止规则的 locked target 不能通过“重新预登记”恢复为未见数据。D17-XBAND
S/Ku 只允许证据复核和合成合同测试。

### 9.4 实现

- 可复用逻辑进入 `datasets/`、`features/`、`models/`、`training/`、`evaluation/` 或 `utils/`；
- 用户入口进入 `scripts/`；
- 自动测试进入 `tests/`；
- 新正式输出使用新目录和新实验 ID；
- 不复制一份旧脚本改名后长期并存。

### 9.5 验证

先运行最小合成测试和 smoke，再运行正式实验。smoke 只证明接口能走通，不写成性能结论。
所有正式指标同时报告分母、每折/每组、macro、最差组和配对变化。

### 9.6 复核与合入

reviewer 检查数据范围、测试访问、代码、日志、结果、失败记录和结论边界。负责人批准后才能
进入主分支、论文表格或分享包。

## 10. 实验记录最低要求

每一轮正式实验都要记录：

| 类别 | 必填字段 |
|---|---|
| 身份 | experiment ID、任务 ID、owner、reviewer、日期 |
| 代码 | Git commit、branch、工作区是否干净 |
| 环境 | Python、PyTorch、CUDA、GPU/CPU、依赖锁定文件 |
| 数据 | manifest、SHA256、样本数、batch/scan/session 分组键 |
| 配置 | 配置路径、哈希、随机种子、epoch、优化器和损失 |
| 选择 | 哪些只看 train、哪些可看 val、test 是否加载 |
| 指标 | 分母、pooled、macro、每组、最差组、区间和混淆矩阵 |
| 配对 | 新增/移除误警、目标损失、定位变化 |
| 输出 | checkpoint、日志、预测、表格和报告的新目录 |
| 决策 | 继续、停止、降级、失败原因和禁止表述 |

只记录“最好一轮”视为记录不合格。已经删除的历史轮次不能凭印象补数字。

## 11. 全体成员必须遵守的红线

1. 不随机拆分同一 batch、scan、session 或连续事件中的切片；
2. 不根据 test 选择模型、阈值、特征、窗口、notch 半宽或组合规则；
3. 不覆盖 `results/final_evidence/`、冻结 checkpoint 或历史正式输出；
4. 不把 complete-scan 模型写成 causal/real-time；
5. 不把当前六折写成独立外部盲测；
6. 不把 UAV/背景结果写成空飘球载荷识别；
7. 不把无 PRF 的归一化谱写成物理 Hz 微多普勒；
8. 不把未经同步和幅相标定的相对 H/V 量写成绝对极化参数；
9. 不根据 RD 图或文件名猜建筑、鸟、地物等物理类别；无证据时写 `unknown`；
10. 不把 Tian point-GT 本地消融写成论文成功复现；
11. 不提交原始 MAT/IQ、checkpoint、访问凭据、个人绝对路径或开发聊天记录；
12. 不隐去失败实验、目标损失、最差折或数据混杂，只展示总体 AUC。
13. 不重跑 D17-XBAND 已消费的 S/Ku target，不用新实验 ID、CNN、域适配或复制数据规避
    `FAIL_STOP`。
14. 不混合 HSR ScienceDB V2 和期刊包；不运行期刊历史包中会移动原件的旧 loader；不把
    V2 route 称为 session-disjoint，不随机拆 MAT/frame/window，不使用 overflow 训练/测试；
15. 不随机拆分 DroneRFc-MM 的 frame 或派生 5 秒 window，不把点云 subset 写成 IQ、极化或
    独立外部性能证据。

违反红线的结果不进入论文、答辩、分享包或下一阶段选型。

## 12. Git 与文件要求

- 每个任务使用独立分支，例如 `task/PUB-DAUR-review`；
- 开始和交付前都运行 `git status --short`；
- 提交只包含本任务文件，不顺手整理无关目录；
- 原始数据放 `data/raw/`，正式生成结果放约定的 ignored output 目录；
- 小型、可追溯、经过审核的证据才能提交 Git；
- commit 信息说明做了什么，不写“update”“try again”等无意义标题；
- 不直接重写他人的历史提交；需要恢复时先联系负责人；
- 当前仓库没有默认远端时，先本地提交，由负责人决定共享或合并方式。

## 13. 周报和沟通要求

使用 [统一周报模板](assets/templates/TEAM_WEEKLY_REPORT_TEMPLATE_ZH.md)。每周只回答八件事：

1. 本周要解决的问题；
2. 使用的数据、版本和分组；
3. 实际完成内容；
4. 验证命令和结果；
5. 成功结果；
6. 失败、异常和最差组；
7. 当前允许/禁止结论；
8. 下周唯一优先动作和需要谁协助。

发现数据泄漏、test 被提前读取、原始数据损坏或结论边界错误时立即报告，不等周会。

## 14. 推荐人员分工

### 可分配任务线

| 角色 | 主任务 | 维护任务 |
|---|---|---|
| A：公开数据与信号 | PUB-HSR V2 审计复核；DAUR 证据 26 复核 | route/source-session、物理轴、分组与单位边界 |
| B：公开数据工程 | PUB-FMCWR2 证据 28 复核/归一化处理合同 | 原件只读、v5/v7.3、频段/角度/重复分组、物理轴门禁 |
| C：复现与工程 | Tian 合同测试、D15 | Git、测试、哈希和实验记录 |
| D：证据与质量保证 | D17-XBAND 证据复核、D16 | 周报、消费门禁、敏感信息和证据边界 |
| E：点云与时序 | PUB-DRONERFC 证据复核/B1 更正条件清单 | 禁止强行对齐 B1、随机 frame/window split 或把接口结果冒充性能 |

### 四至五名组员的合并方式

可把公开数据拆成 DAUR 证据复核、HSR/FMCWR、DroneRFc-MM 三个范围，把 Tian/工程拆成复现与质量
保证两人。负责人继续掌握
研究问题、外部沟通、locked test 开放和最终结论审批，不把这些决定完全下放。

## 15. 第一周建议安排

| 时间 | 全员任务 | 可并行任务 |
|---|---|---|
| 第 1 天 | 分享包校验、阅读一页摘要和本手册 | 无 |
| 第 2 天 | 口头验收、填写个人清单 | 选择任务方向 |
| 第 3 天 | 环境/测试验收、提交任务认领表 | 分享包证据核验、D17-XBAND 合成合同测试或 Tian 合同核对 |
| 第 4-5 天 | reviewer 审核任务边界 | DAUR 证据 26/grouping 阻塞复核、HSR 证据 27/provenance/axis 阻塞复核、FMCWR 证据 28/grouping/axis 阻塞复核或 DroneRFc-MM 证据 25/B1 阻塞复核 |
| 第 6-7 天 | 第一次周报 | 冻结下一周唯一主动作 |

第一周不以“训练出一个准确率”为目标，以所有成员理解证据边界、能复跑审计并形成合格任务卡
为完成标准。

## 16. 任务完成定义

一个任务只有同时满足以下条件才算完成：

- 认领表中的范围和交付物全部具备；
- 输入数据、分组和代码版本可追溯；
- 测试和审计通过；
- 主要指标、最差组和失败结果齐全；
- reviewer 已签字或留下可追溯审核结论；
- 文档写清允许和禁止表述；
- Git 工作区没有本任务造成的未说明改动；
- 后续动作已回填任务台账。

“代码能跑”“模型有数字”“做了 PPT”都不能单独作为完成标准。

## 17. 项目负责人的职责

负责人需要：

- 主持新人五问验收并批准任务认领表；
- 控制原始数据、checkpoint 和 locked test 权限；
- 指定每项任务 reviewer；
- 决定研究问题、主指标、停止规则和是否进入下一阶段；
- 有现实渠道时对外索取 H/V、PRF、坐标和 Tian 条件；不可获得时批准 unknown 降级并保持门禁；
- 审核每周的失败、最差组和结论边界；
- 只有完成检查器输出 `COMPLETE` 时宣布当前方向结束。

## 18. 随包模板

- `assets/templates/team_onboarding_checklist_template_v1.csv`：每位成员一行的入组验收表；
- `assets/templates/team_qualification_scorecard_template_v1.csv`：成员评分、补验、角色和权限记录；
- `assets/templates/team_task_claim_template_v1.csv`：每项任务一行的认领与门禁表；
- `assets/templates/TEAM_WEEKLY_REPORT_TEMPLATE_ZH.md`：每人每周复制一份填写；
- `assets/contracts/current_direction_completion_v1.json`：当前方向完成规则；
- `docs/12_PROJECT_TASK_LEDGER_ZH.md`：项目问题和状态总台账。

所有模板填写后保存在负责人指定的协作位置。包含个人信息、内部路径或数据权限的信息不要放入
对外分享包。
