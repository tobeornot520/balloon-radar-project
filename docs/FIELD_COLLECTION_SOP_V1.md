# 2026 年 8-9 月双极化雷达外场准备与 Pilot SOP V1

版本：2026-07-28

## 1. 目的与边界

本 SOP 把 2026 年 8-9 月任务冻结为五道连续门禁：设备与连续采集能力、时间同步、
H/V 极化标定、无球 dry run、四场景 Pilot。它服务于未来时域、极化、微多普勒和
轨迹数据采集，不替代逐样本数据合同，也不产生任何模型性能结论。

Pilot 数据全部标记为 `development`。Pilot 用来发现采集问题，不能预先命名为
`locked_test`，也不能因为结果看起来好就升级为盲测数据。

机器可审计资产：

- `configs/field_readiness_checklist_v1.json`
- `configs/field_readiness_evidence_template_v1.csv`
- `configs/pilot_scenario_matrix_v1.csv`
- `configs/pilot_session_log_template_v1.csv`
- `configs/data_collection_manifest_template_v1.csv`
- `scripts/audit_field_readiness_v1.py`
- `scripts/validate_data_collection_manifest.py`

设备负责人联系表和答复模板：

- `docs/FIELD_CAPABILITY_REQUEST_V1.md`
- `configs/field_capability_response_template_v1.csv`

## 2. 8-9 月时间表

| 日期 | 阶段 | 必须交付 | 放行条件 |
|---|---|---|---|
| 7 月 28-31 日 | 文档冻结 | SOP、门禁、场景表、空白证据表 | 本仓库自动测试通过 |
| 8 月 1-10 日 | 能力摸底 | 雷达配置、H/V 原始 IQ、扫描关系、连续写盘报告 | `capability=PASS` |
| 8 月 11-20 日 | 同步台架 | 雷达、视频、记录表的 UTC 映射和至少 5 次同步事件 | `synchronization=PASS` |
| 8 月 21-31 日 | 极化标定 | 参考目标、空场、H/V 幅相重复性、饱和与 calibration_id | `polarimetric_calibration=PASS` |
| 9 月 1-10 日 | 场地 dry run | 纯背景、同步视频、manifest、完整回放和双份备份 | `dry_run=PASS` |
| 9 月 11-20 日 | Pilot 试采 | 四场景各 3 次，每次 2-5 分钟 | 现场完整性检查通过 |
| 9 月 21-30 日 | Pilot 质检 | 接受/拒绝清单、同步误差、通道质量、数据卡草案 | `pilot=PASS` 或形成整改清单 |

任何前置门失败，后续门保持 `BLOCKED`。不得为了赶日期跳过同步或标定。

## 3. 第一次团队确认会

以下问题必须由设备或外场负责人给出可追溯答案，不能从旧数据猜测：

1. 雷达型号、序列号、固件、频段、PRF、带宽、脉冲数和距离门数是什么；
2. 能否同时保存 H/V 原始复数 IQ，是否经过通道切换或非相干处理；
3. 一帧、一个波束、一次扫描和一个文件之间是什么关系；
4. 是否输出硬件序号、真实采集时间和时钟重置状态；
5. 连续写盘的实际数据率、最长稳定时长和存储接口是什么；
6. 视频、GPS/RTK、测距仪、风速仪和统一时间源有哪些；
7. 极化通道过去如何做幅相校准，能否布设稳定参考目标；
8. 场地可用距离、方位、高度、遮挡、电磁背景和安全边界是什么；
9. 雷达操作、目标操作、真值记录、数据备份和安全分别由谁负责；
10. 8 月台架、9 月 dry run 和 Pilot 的可用日期是什么。

回答和附件作为 readiness evidence 保存，不写入聊天记录后即丢失。

## 4. 五道门的现场含义

### 4.1 Capability

必须实际读取一段新采数据，确认 H/V 都是复数数组、尺寸和数据类型符合配置、没有
NaN/Inf，并能通过硬件序号恢复帧顺序。连续采集至少 300 秒，写盘吞吐量至少为实测
原始流速的 1.5 倍。只看到处理后的 RD 图片不算具备原始极化和微多普勒采集能力。

### 4.2 Synchronization

雷达、视频和真值记录统一到 UTC；不能直接统一时钟时，必须保留偏移模型和估计误差。
连续记录至少 5 次清楚的同步事件。Pilot 放行线为绝对同步误差 P95 不超过 50 ms，
最大值不超过 100 ms。状态切换标签应记录开始、结束和不确定区间，不能只给整个文件
写一个“摆动”标签。

### 4.3 Polarimetric calibration

先核对 H/V 线缆和软件映射，再在同一雷达配置下采固定参考目标和空场。报告 H/V
相对幅度和相对相位的重复性，不把未标定的相对量写成绝对散射参数。Pilot 放行线：

- H/V 相对幅度重复性标准差不超过 1.0 dB；
- H/V 相对相位圆标准差不超过 10 度；
- 两通道饱和复样本比例均不超过 0.001；
- 标定方法、文件、配置哈希和有效期归入唯一 `calibration_id`。

若设备不支持相干 H/V，必须关闭需要相对相位的极化研究；仍可保留 H/V 功率特征，
但要更改研究声明和数据字段。

### 4.4 Dry run

不放飞气球，完整走一遍架设、启动、视频同步、纯背景、可确认运动目标、现场检查、
备份和实验室回放。纯背景连续时间至少 600 秒，丢帧率不超过 0.001，不能有未解释的
时钟重置。dry run manifest 必须分别通过 `capture` 和 `causal` 档，并检查实际文件。

### 4.5 Pilot

只执行四个场景：纯背景、无载球、同一球体有载稳定、有载摆动或自然运动。每个场景
至少 3 个独立重启的 session，每段连续 2-5 分钟，全程视频。目标场景必须有同日期、
同场地、同 radar_config_id 和 calibration_id 的背景对照。

Pilot 不同时改变球体、载荷、悬挂、距离和运动状态。第一轮只使用一个固定 Pilot
载荷，先判断链路能否支持可重复差异，再扩展轻载和重载。

## 5. 人员与口令

最低角色：实验负责人、雷达操作员、目标操作员、真值记录员、数据与安全员。可以一人
兼任，但雷达操作和目标操作不应由同一人在采集段内同时承担。实验负责人拥有停止权。

每段开始前口头报出并写入同一个 `session_id`：

```text
YYYYMMDD_SITE_SCENARIO_RNN
```

雷达文件、视频、真值、照片、session log 和 manifest 都引用该 ID。重新启动采集必须
增加 repetition，不覆盖原文件。

## 6. 单个 session 操作顺序

1. 核对场景 ID、人员、安全区、气象、球体、载荷和悬挂；
2. 核对 `radar_config_id`、`calibration_id`、时钟状态和剩余存储；
3. 同时启动真值记录、视频和雷达，口头报告 `session_id`；
4. 执行可见同步事件并记录 UTC；
5. 保留至少 20 秒状态前段，再执行目标状态；
6. 连续采集计划时长，中途不改雷达参数；
7. 状态结束后保留至少 20 秒衰减或恢复段；
8. 停止采集，立即核对 H/V、文件大小、时长、序号和视频；
9. 填写 session log，任何异常都保留原始文件并标记拒绝原因；
10. 复制到第二存储设备并生成 SHA256，不在现场做破坏性清理。

## 7. 数据目录与只读边界

建议受控存储结构：

```text
collection_root/
  raw_iq/
  video/
  truth/
  radar_configs/
  calibration/
  weather/
  manifests/
  session_logs/
  hashes/
```

manifest 中所有路径相对于 `collection_root`，不得保存个人绝对路径。原始 IQ、原始视频
和原始真值首次落盘后设为只读；纠错通过新版本 manifest 或派生标签完成，不覆盖原件。

## 8. 现场停止规则

出现以下任一情况，停止正式场景并保留诊断数据：

- H 或 V 缺失、通道映射不确定、复数 IQ 被处理成仅功率；
- 硬件序号不连续且无法解释，或时钟发生未记录重置；
- 同步误差超过门限，视频无法看清目标或状态起止；
- 持续饱和、数据率超过写盘能力、文件无法现场读取；
- 目标超出安全边界或雷达可观测区域；
- 场地、气象、人员或设备不满足安全要求；
- 第一份或第二份备份无法校验。

停止后不得用手工猜测补齐时间、顺序、位置、SNR、标定或状态标签。

## 9. 证据与命令

先从冻结 checklist 生成完整 pending 表：

```bash
python scripts/initialize_field_readiness_evidence.py \
  work/field_readiness_evidence_v1.csv
```

逐项填写真实证据后，按阶段审计，例如 8 月能力摸底：

```bash
python scripts/audit_field_readiness_v1.py \
  work/field_readiness_evidence_v1.csv \
  --through capability \
  --evidence-root /controlled/field_evidence \
  --check-files \
  --output-dir results/data_audit/field_readiness_capability_v1
```

dry run 与 Pilot manifest 还必须通过已有数据合同：

```bash
python scripts/validate_data_collection_manifest.py \
  /controlled/collection/manifests/dry_run_v1.csv \
  --profile causal \
  --data-root /controlled/collection \
  --check-files \
  --output-dir results/data_audit/dry_run_causal_v1
```

最后的 Pilot readiness 审计：

```bash
python scripts/audit_field_readiness_v1.py \
  work/field_readiness_evidence_v1.csv \
  --through pilot \
  --evidence-root /controlled/field_evidence \
  --check-files \
  --output-dir results/data_audit/field_readiness_pilot_v1
```

只有输出 `formal_pilot_gate_open=true` 才表示 Pilot 采集链验收完成。它不表示数据已经
支持载荷分类，也不打开 locked evaluation 门。

## 10. Pilot 后决策

Pilot 质量报告只做三选一：

1. `PASS`：四场景、同步、极化、标签、manifest 和备份全部可靠，进入多日期正式采集；
2. `REPEAT`：研究问题不变，但某个可修复环节失败，修复后重新采 Pilot；
3. `REDESIGN`：设备不支持相干 H/V、连续 IQ 或可靠时序，收缩研究范围并重写数据方案。

模型准确率不能覆盖数据链失败。Pilot 数据质量不过门时，停止复杂融合和分类实验。
