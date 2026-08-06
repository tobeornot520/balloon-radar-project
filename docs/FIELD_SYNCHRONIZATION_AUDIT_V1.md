# 雷达、视频与真值同步事件审计 V1

更新时间：2026-08-06

## 1. 目的

外场 readiness 要求至少 5 个同步事件、radar-video 绝对误差 P95 不超过 50 ms、最大值
不超过 100 ms。过去这些数值只能人工填入证据表，缺少从事件记录到结论的可重复计算。

本审计读取同一物理事件在雷达、视频和真值记录中的 UTC 时间，自动计算三组配对误差，
并输出可回填 `SYNC_EVENT_REPEATS`、`SYNC_P95_ERROR`、`SYNC_MAX_ERROR` 的数值表。

## 2. 记录同步事件

从 `configs/field_sync_event_template_v1.csv` 建立受控记录。每个事件需要：

- 同一 session 内从 0 连续增加的 `event_index` 和全局唯一 `event_id`；
- radar、video、truth 三个 UTC 时间戳；
- 三个时钟 ID 和一个 `timestamp_mapping_id`；
- 事件方法和人工/检测不确定度；
- 接受/拒绝状态，拒绝时必须说明原因。

V1 允许可见标记、主动应答器、人工触发和有文档的其他方法。事件不确定度上限冻结为
20 ms；超过该值的事件不能用于通过数值门。每个参与的 session 至少有一个有效事件，
全表至少有 5 个有效事件。

## 3. 命令

```bash
conda run -n radar-torch python scripts/audit_field_synchronization_v1.py \
  /controlled/field_evidence/sync_events_v1.csv \
  --output-dir results/data_audit/field_synchronization_device_v1
```

## 4. 输出

- `event_audit_local.csv`：逐事件 ID、时间和误差，只留本地；
- `pair_metric_summary.csv`：radar-video、radar-truth、video-truth 聚合误差；
- `session_sync_summary.csv`：稳定匿名 session 的事件数、时长、P95 和最大误差；
- `readiness_measurements.csv`：三个可回填的 readiness 数值；
- `summary.json`：状态、问题、输入哈希和声明边界。

状态 `PASS_NUMERIC_LIMITS_ONLY` 只表示有效事件数量和 radar-video 数值限通过。它不能单独
把整个 synchronization gate 标为 PASS。

## 5. 仍需独立证据

以下 readiness 项不能由五个事件的误差统计替代：

- `SYNC_COMMON_TIMEBASE`：UTC 来源或偏移映射方法、版本、残差和有效期；
- `SYNC_RADAR_TIMESTAMP`：每个雷达样本/帧的采集时间与硬件序号确实保存；
- `SYNC_VIDEO_TIMESTAMP`：视频逐帧时间可以可靠映射到 UTC；
- 时钟重启、缓存溢出和跨文件连续性记录；
- 事件识别延迟是否已经纳入 `event_uncertainty_ms`。

因此 summary 永远固定 `formal_synchronization_gate_open=false`。只有三个非数值证据项也在
readiness 表中通过，且引用文件存在，`audit_field_readiness_v1.py` 才能放行同步门。
