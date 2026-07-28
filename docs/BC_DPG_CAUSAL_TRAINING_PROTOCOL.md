# BC-DPG 因果训练协议

版本：2026-07-28

## 1. 当前门禁状态

正式因果训练门禁为 **关闭**。现有 1,148 个检测样本只提供扫描组级文件名时间，组内样本没有经过验证的采集时间戳或硬件序号。

MAT 文件仅包含 H/V IQ 数组。MAT v5 头部 `Created on` 和文件系统 mtime 比文件名采集时间晚约 49 至 50 天，二者基本同步，属于后期转换、保存或复制时间，不能作为采集顺序。

`(beam_layer, azimuth_deg, sample_id)` 在每个扫描组内唯一，可构造确定性顺序，但没有硬件执行日志证明该顺序。因此它只能用于接口 smoke，不得用于正式训练、窗口选择或性能结论。

权威审计：`results/data_audit/detection_acquisition_order/`

## 2. 当前允许的工作

- 使用单个 fold、每类每 split 最多 32 个样本、最多 4 个 epoch 做开发接口 smoke；
- smoke 必须显式传入 `--allow-inferred-order`；
- smoke 必须使用 validation-only，测试 split 不得加载、计算或输出；
- smoke 输出必须标记 `development_only_inferred_order` 和 `order_verified_by_timestamp=false`；
- smoke 只检查上下文构造、冷启动、训练、checkpoint 和结果接口是否贯通；
- smoke 的验证数字不得用于选择窗口、模型或对外报告。

当前固定的首个接口 smoke 使用 Fold 1、最近 4 个历史样本、2 个 epoch、每类每 split 12 个样本。窗口 4 只用于缩小接口测试，不代表候选窗口选择。

```bash
python scripts/run_bc_dpg_causal_smoke.py
```

## 3. 正式训练前必须补齐

1. 每个样本具有真实采集时间戳或单调硬件序号；
2. 记录时间分辨率、时钟重置、丢帧、重复序号和扫描边界规则；
3. 建立 sample ID 与时间/序号的一对一对齐审计；
4. 以物理时间或明确状态记忆长度预注册候选窗口，不复用当前测试后 4/16/64/all-history 结果做选择；
5. 训练集拟合模型，验证集选择窗口、冷启动策略、checkpoint 和阈值；
6. 外层锁定测试集只在全部选择冻结后评价一次；
7. 完整扫描离线上限、样本独立 BC 和正式因果模型分开命名与报告。

新数据还必须先通过版本化数据合同的 causal 档：

```bash
python scripts/validate_data_collection_manifest.py \
  path/to/collection_manifest.csv \
  --profile causal \
  --output-dir results/data_audit/new_collection_causal
```

合同定义见 `docs/NEW_DATA_COLLECTION_PROTOCOL.md`。只有该报告的 `formal_causal_training_gate_open=true` 才能解除本协议的训练停止条件；beam/azimuth 推断顺序和人工补写字段不能打开门禁。

## 4. 正式评价单位

- pooled、macro、median/IQR 和 worst-fold 检测定位指标；
- 以扫描组、日期或飞行为单位的不确定性；
- 冷启动与稳定历史阶段分层结果；
- 每扫描、每小时和事件级虚警，需要同步补齐观测时长和事件合并规则；
- 同日目标/背景和跨日期、跨场地锁定测试。

## 5. 停止条件

在真实组内顺序尚未恢复时，不启动六折因果训练，不比较多个窗口，不把推断顺序 smoke 写入正式证据或分享包主结果。
