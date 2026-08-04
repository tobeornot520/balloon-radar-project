# 复核入口与后续计划

## 1. 分享包能做什么

本分享包是可追溯、可校验的冻结结果摘录，不包含完整源码、原始数据、逐样本预测或模型权重。阅读 Markdown、CSV 和图片不需要安装 Python 环境。

`MANIFEST.json` 记录分享版本、来源提交、证据角色和文件哈希；`SHA256SUMS.txt` 用于检查文件是否在传输中损坏。哈希校验不等同于重新运行指标计算。完整复现需要内部代码、数据、冻结预测和 checkpoint。

## 2. 完整仓库环境

内部工程使用 Conda 环境 `radar-torch`，Python 3.11。仓库根目录提供 `environment.yml` 和锁定依赖文件。

```bash
conda env create -f environment.yml
conda activate radar-torch
python scripts/check_project_health.py --require-joint-inputs
python -m pytest
```

原始数据、checkpoint 和大规模训练输出不在 Git 中，需要由数据持有方按项目目录约定单独准备。

公开 LAT-MRICD 原始数据同样不包含在分享包中。按官方来源下载并放到项目约定目录后，先运行：

```bash
python scripts/audit_lat_mricd_dataset_v1.py --overwrite
```

该数据禁止随机拆行。D17-NX/HX 的五折 batch-code-held-out 大类基线已完成并冻结。
D17-XBAND 的唯一一次密封运行也已完成：X->S 的固定 LR target batch-class macro accuracy
为 0.6517，但 UAV batch recall 为 0.4433，未通过严格 `>0.50` 门；X->Ku 的同一指标为
0.8400，且该 target 的全部门条件通过。两个 locked target 未同时通过，整体决策为
`FAIL_STOP`。S/Ku 已消费，不能再次运行真实 target 来做确认性比较，也不能据此调 CNN、
域适配、阈值或特征。

完整仓库中的冻结结果重放顺序如下。分享包本身不含这些脚本和原始 MAT，因此只能做证据
复核，不能单独执行重放。

```bash
python scripts/freeze_lat_mricd_grouped_split_v1.py --overwrite
git diff --exit-code -- data/splits/lat_mricd_x_batch_grouped_v1.csv \
  data/splits/lat_mricd_x_batch_grouped_v1.json
python scripts/run_lat_mricd_grouped_baseline_v1.py \
  --output-dir /tmp/lat_mricd_grouped_baseline_replay --overwrite
python scripts/build_lat_mricd_grouped_evidence_v1.py \
  --source-dir /tmp/lat_mricd_grouped_baseline_replay \
  --output-dir /tmp/lat_mricd_grouped_evidence_replay --overwrite
diff -rq /tmp/lat_mricd_grouped_evidence_replay/tables \
  results/final_evidence/lat_mricd_grouped_baselines_v1/tables
```

先阅读 `docs/LAT_MRICD_GROUPED_BASELINE_PROTOCOL_V1.md`。预期冻结划分无 Git 差异且聚合表
一致；报告和证据 manifest 因记录当前提交，不要求逐字节相同。

D17-XBAND 不提供真实 target 重放入口。组员只允许阅读
`evidence/24_LAT_MRICD_CROSS_BAND_TRANSFER.md` 及配套聚合表、消费记录和 manifest，或在
完整仓库中运行不读取真实 S/Ku 的合成合同测试：

```bash
python -m pytest tests/test_lat_mricd_cross_band_transfer.py \
  tests/test_lat_mricd_cross_band_evidence.py
```

这些测试只验证实现和门禁合同，不产生新的性能证据。不得删除消费记录、复制真实 target 到
其他路径或改实验 ID 来规避密封规则。

## 3. 主要内部入口

六折 BC-DPG v3：

```bash
python scripts/run_bc_dpg_v3.py --help
```

显式极化表征：

```bash
python scripts/run_polarimetric_representation_benchmark_v2.py --help
```

ROI Stage 4 六折：

```bash
python scripts/run_roi_stage4_selected_sixfold_v1.py --help
```

重新构建联合审计时应写入新目录，不直接覆盖冻结证据：

```bash
python scripts/build_final_roi_bc_dpg_joint_audit.py \
  --output-dir results/data_audit/final_roi_bc_dpg_joint_rebuild
```

从冻结预测重建证据资产与脱敏分享包：

```bash
python scripts/build_roi_bc_dpg_joint_paper_assets.py --overwrite
python scripts/build_bc_dpg_localization_evidence.py --overwrite
python scripts/build_project_share_package.py
```

冻结 BC-DPG 上下文敏感性审计：

```bash
python scripts/audit_bc_dpg_v3_causal_context.py --overwrite
```

该命令不训练模型或重选阈值；正式结果中的 leave-one-out 与 past-only 行都是冻结测试后的上下文替换诊断。

检查采集顺序并运行受限接口 smoke：

```bash
python scripts/audit_detection_acquisition_order.py --overwrite
python scripts/run_bc_dpg_causal_smoke.py
```

当前审计结论是正式训练门禁关闭。smoke 只加载 train/val 小样本，不读取测试 split，也不提供性能或窗口选择证据。

新数据采集完成后先运行分级合同预检：

```bash
python scripts/validate_data_collection_manifest.py \
  path/to/collection_manifest.csv \
  --profile capture \
  --output-dir results/data_audit/new_collection_capture
```

只有 `causal` 报告打开因果训练门禁后才能正式训练；只有 `locked_evaluation` 报告通过后才能进入外部锁定评价。

## 4. 当前证据与治理文档

- BC-DPG v3：`results/final_evidence/bc_dpg_v3_final/`
- Stage 3：`docs/polarimetric_stage3/STAGE3_FROZEN_CONCLUSION.md`
- Stage 4：`results/data_audit/roi_stage4_selected_sixfold_v1/`
- 最终联合审计：`results/data_audit/final_roi_bc_dpg_joint_v2_base_threshold/`
- 联合证据资产：`results/final_evidence/roi_bc_dpg_joint_fixed_threshold/`
- BC-DPG 因果上下文敏感性审计：`results/data_audit/bc_dpg_v3_causal_context_audit/`
- 采集顺序就绪审计：`results/data_audit/detection_acquisition_order/`
- BC-DPG 冻结定位证据：`results/final_evidence/bc_dpg_localization/`
- 新数据采集协议：`docs/NEW_DATA_COLLECTION_PROTOCOL.md`
- 当前数据合同缺口：`results/data_audit/data_collection_readiness_v1/`
- 数据卡：`docs/DATA_CARD.md`
- 指标定义：`docs/METRIC_DEFINITIONS.md`
- 模型选择台账：`docs/MODEL_SELECTION_LEDGER.md`
- 外部公开数据审计：`docs/EXTERNAL_PUBLIC_DATA_AUDIT_20260803.md`
- LAT-MRICD 分组基线协议：`docs/LAT_MRICD_GROUPED_BASELINE_PROTOCOL_V1.md`
- LAT-MRICD 分组基线冻结报告：`evidence/23_LAT_MRICD_GROUPED_BASELINES.md`
- LAT-MRICD 分组指标与 CI：`assets/tables/lat_mricd_grouped_aggregate_metrics.csv`、
  `assets/tables/lat_mricd_grouped_cluster_bootstrap_intervals.csv`
- LAT-MRICD 跨频段冻结负结果：`evidence/24_LAT_MRICD_CROSS_BAND_TRANSFER.md`
- 跨频段门判定与消费记录：`evidence/24_LAT_MRICD_CROSS_BAND_TRANSFER_GATE.json`、
  `evidence/24_LAT_MRICD_CROSS_BAND_TRANSFER_RUN_CONSUMED.json`

早期 `final_roi_bc_dpg_joint` 使用了错误的 BC 判决来源，已经移出活动证据，不得引用。

## 5. 下一阶段优先级

### 优先级 1：同日多类数据与锁定外部评价

- 每个日期和场地同时采集目标与无目标背景；
- 增加空飘球、带载空飘球、鸟类和不同背景类型；
- 以日期、场地或飞行架次为外层隔离单位；
- 在盲测前冻结结构、阈值、后处理、主指标和失败判据。
- 使用 v1 空白模板采集全部 40 个字段，并在数据进入训练前通过三档合同预检。

### 优先级 2：BC-DPG 因果在线化

冻结 checkpoint 的 leave-one-out 和 past-only 敏感性审计已经完成。leave-one-out 为 54/830 个虚警、289/318 个正确检测，但仍使用未来样本；past-only all-history 为 93/830、288/318，但顺序由 `(beam_layer, azimuth_deg, sample_id)` 推断，未由时间戳验证。两者都不是重新训练的因果模型，也不得用于从测试集选择 all-history 窗口。

下一步应补齐逐样本真实时间戳，按因果上下文重新训练 BC-DPG，只在训练/验证集比较冷启动处理和历史窗口，然后用锁定外部测试集评价一次。完整扫描结果继续只作为离线上限，样本独立 BC 继续作为在线导向参照。

### 优先级 3：部署级虚警与连续物理定位指标

冻结网格定位证据已经补齐全部目标、过阈值目标、联合成功目标的距离/速度 MAE、中位数、P90 和分层结果。下一步需补齐时间戳、扫描时长和事件边界，报告每扫描、每小时和事件级虚警；同时增加 SNR、环境分层、雷达标定信息和未量化连续物理真值，才能评价真实测距测速精度。

### 优先级 4：嵌套选择与学习型联合模型

使用嵌套扫描组交叉验证，或独立开发集选择 BC 分数、ROI 分数、背景状态和质量指标的门控/融合规则；外层测试只评价一次。

### 优先级 5：多域细粒度分类

在真实空飘球数据齐备后，引入长慢时间时频/微多普勒、极化、轨迹和行为特征，逐级开展目标类别、有载/无载、载荷类型和运动状态识别。

D17-NX/HX 已完成：Narrow-X 固定 LR/RF 的 batch-class macro 为 0.7999/0.7872，HRRP-X
为 0.6617/0.6481；四项 batch-code cluster 95% CI 和两个配对差值均已冻结，差值 CI 均
跨 0，不选胜者。D17-XBAND 也已结束为 `FAIL_STOP`：X->Ku 过门，但 X->S 的 UAV recall
0.4433 未过门，S/Ku target 均已消费。

下一公开数据工作不再使用 LAT-MRICD 的 S/Ku target，而按以下顺序推进：

1. LSS-DAUR-1.0：全量只读审计已完成并冻结为
   `PASS_SCHEMA_PAIRING_BLOCKED_GROUPING_AND_PHYSICAL_AXIS`。77 个逻辑观测只有 76 个唯一
   TD/TR 内容对；39 个保守候选 source-session 组未经作者确认，严格时间、日期和 1024-bin
   轴仍阻塞。canonical/backup 与 TD/TR 必须始终同组，禁止随机 MAT/frame/window split，
   禁止静默修日期、物理 Hz 声明和模型训练；
2. LSS-HSR-L：当前执行项。以已下载且 ZIP 完整的 ScienceDB V2 正式包为 canonical candidate，
   实现只读 loader 并核对 schema、scene/track 分组；V2 的 1,561 个 entries 与期刊包的
   1,478 个不等价，V2 额外含 `overflow/air_routes`，同名样本长度也不同，两包禁止混合；
3. LSS-FMCWR-2.0：以可审计方式解包 RAR，建立 MAT schema、目标/频段/角度/记录分组和
   潜在重复审计；仿真飞鸟不得写成真实自然鸟；
4. DroneRFc-MM V1：选择性 radar subset 的全量 PCD schema/时间覆盖审计已完成。完整发布为
   113 files、75,612,067,287 bytes；本地仅有 28 个文件、47,366,902 bytes。30,717 frames、
   639,527 points 均通过 schema/finite/POINTS/时间戳检查；8 条 recording 与 GT 时间范围重叠，
   B1 零重叠，故整体同步门阻塞。717 个派生 5 秒 windows 禁止随机拆分。

DAUR 虽已通过 schema 与配对门，但 grouping/physical-axis 总门仍关闭；其余三项也只在各自
schema、配对和 group 门通过后再预登记算法，不以“已经下载”代替数据可用。
DroneRFc-MM 数据 DOI 为 `10.57760/sciencedb.j00173.00094`、许可为 `CC-BY-SA-4.0`；它不是
ADC/IQ 或 H/V，没有鸟、天气、空飘球对照，只能用于点云/轨迹接口和时序算法审计；B1
只有取得更正 GT 或可归因时间偏移后才可重开监督对齐。

辅助 smoke 已最小落地：Ku UAV 群包只按 `{Exp1}`、`{Exp2_1,Exp2_2}`、
`{Exp3_1,Exp3_2}` 三个物理实验管理，用 275 个连续屏的 XYZ 量测验证航迹接口；NEXRAD
只用一个完整 KTLX 体扫验证 Z/V/SW/ZDR/PhiDP/RhoHV 读取。两者不训练识别器。42.4 GB
室内气球、23.46 GB S 波段 UAV 和 30.36 GB 三频 UAV/真鸟主包已登记但暂缓，只有现有
数据审计暴露明确缺口时才重新评估下载。

LAT-MRICD 只构成同一公开发布内、已见子型号的 batch-code-held-out 证据，不是 unseen-model
或独立外部验证。它不与当前 H/V 六折结果合并计分，也不解除极化、PRF、连续时序、空飘球
标签或 Tian 复现门禁；主 UAV 方向仍为 `4/6`、`BLOCKED_EXTERNAL`。

## 6. 继续开发时必须保持的纪律

- 不依据外层测试结果重新选择阈值、容差、模型或组合逻辑；
- 不覆盖冻结 checkpoint 和正式证据目录；
- 不重跑已消费的 D17-XBAND S/Ku target，不以新实验 ID、CNN 或域适配规避 `FAIL_STOP`；
- 不混合 HSR ScienceDB V2 与期刊包；只读 loader 以 V2 为 canonical candidate；
- 不随机拆分 DroneRFc-MM 的 PCD frame 或派生 5 秒 window，最低按原始 recording 分组；
- 不把 DroneRFc-MM B1 与当前同名 GT 强行平移对齐；更正材料到位前保持 blocked；
- 不随机拆分 Ku 群目标列/点或 NEXRAD gate/ray/patch；两个小样本保持 smoke-only；
- smoke 结果只验证接口，不作为性能结论；
- 两折筛选、六折内部开发评价和外部盲测必须分开报告；
- 样本独立、因果上下文和完整扫描离线模型必须分开命名；
- 不用当前 4/16/64/all-history 测试结果选择下一版因果历史窗口；
- 同时报告 pooled、macro、median/IQR 和 worst-fold 指标；
- 新结论必须附带数据范围、划分方式、选择来源、指标定义和完整复现清单。
