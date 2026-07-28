# 新数据采集与锁定评价协议

版本：2026-07-28

## 1. 目的

本协议把下一阶段的数据要求固化为机器可检查的清单合同。它服务于三个独立目标：保留可审计的原始采集元数据、为 BC-DPG 因果上下文提供真实顺序、为跨日期/跨场地锁定评价建立外层隔离。

合同和空白模板：

- `configs/data_collection_contract_v1.json`
- `configs/data_collection_manifest_template_v1.csv`

模板只有表头，不包含虚构样本。原始 IQ、标签、标定文件和包含敏感位置的信息仍由数据持有方在受控存储中管理，不进入 Git。

## 2. 三档门禁

### Capture

检查逐行字段、UTC 时间格式、相对存储路径、目标/背景标签一致性、事件边界和基本物理元数据。该档通过只表示数据记录完整，不表示可以训练因果模型或进行锁定评价。

### Causal

在 Capture 基础上，要求每个扫描具有从 0 开始的连续 `scan_sequence`、严格递增且唯一的 `hardware_sequence`、非递减 UTC 时间、明确的时钟重置计数和 `order_verified=true`。H/V 通道都必须有效。

### Locked Evaluation

在 Causal 基础上，要求存在 `locked_test`，`session_id` 与 `outer_group_id` 不跨 development/validation/locked_test 分区，同一日期、场地、雷达配置和标定条件下同时包含目标与背景，锁定测试自身也同时包含两类；目标 SNR 必须完整。

## 3. 使用方式

采集落盘后先复制空白模板并逐行填写，再运行：

```bash
python scripts/validate_data_collection_manifest.py \
  path/to/collection_manifest.csv \
  --profile capture \
  --output-dir results/data_audit/new_collection_capture
```

确认采集字段完整后依次提高门禁：

```bash
python scripts/validate_data_collection_manifest.py \
  path/to/collection_manifest.csv \
  --profile causal \
  --output-dir results/data_audit/new_collection_causal

python scripts/validate_data_collection_manifest.py \
  path/to/collection_manifest.csv \
  --profile locked_evaluation \
  --output-dir results/data_audit/new_collection_locked
```

只有当正式输入在 `locked_evaluation` 档通过，且模型结构、窗口、阈值、事件合并规则和主指标已经冻结后，才能打开外部锁定评价。

## 4. 路径与隐私

`iq_path` 和 `label_path` 必须是相对于受控数据根目录的路径，禁止绝对个人路径和 `..` 跳转。需要检查文件是否存在时显式提供数据根目录：

```bash
python scripts/validate_data_collection_manifest.py \
  path/to/collection_manifest.csv \
  --profile capture \
  --data-root /controlled/storage/root \
  --check-files
```

预检报告不复制逐样本清单内容，只记录字段覆盖、汇总计数和有限的问题示例。

## 5. 当前数据状态

当前 V4 清单不满足本合同。它缺少逐样本真实 UTC 时间、硬件序号、时钟信息、事件与观测时长、SNR、雷达/标定版本、场地/会话/飞行、H/V 有效性以及外层锁定分区。已有 beam/azimuth 推断顺序不能替代这些字段，因此正式因果训练和锁定外部评价门禁继续关闭。

## 6. 旧清单迁移边界

以下字段可以做机械重命名或受控标准化：

- `mat_path` 可迁移为相对数据根目录的 `iq_path`，但必须先移除个人绝对路径；
- `class_name` 可结合 `target_present` 标准化为合同枚举中的 `target_class`；
- `source_file` 可保留为旧扫描组参考，但不能自动宣称为经过硬件验证的 `scan_id`；
- `beam_layer`、`azimuth_deg`、`distance_m` 和 `velocity_mps` 可按原值迁移并重新校验。

以下信息不能从旧字段推断或补写：

- `source_file` 文件名秒级时间不能生成逐样本 `acquisition_timestamp_utc`；
- beam/azimuth 排序不能生成 `hardware_sequence` 或 `order_verified=true`；
- 旧 `new_split=test` 不能改名为 `locked_test`，因为这些样本已参与模型与叙事开发；
- MAT 头时间和文件 mtime 不能生成采集时钟、事件边界或观测时长；
- 缺失的 SNR、场地、天气、标定、飞行和平台信息必须回到原始采集记录补齐，不能估造。
