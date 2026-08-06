# LSS-FMCWR-2.0 归一化轴/单记录处理合同

版本：`v1`（2026-08-05）

本合同是对 LSS-FMCWR-2.0 V4 的**接口和合成 smoke**。它不读取真实 RAR/MAT，不产生训练集，也不报告 Pd、Pfa、AUC 或任何性能结论。真实数据的 archive/schema 审计见 [`LSS_FMCWR_2_READ_ONLY_AUDIT_20260805.md`](LSS_FMCWR_2_READ_ONLY_AUDIT_20260805.md)。

## 输入约定

接口接收单条二维 `echoes.channelA` 数组，调用者需要按以下**处理约定**提供方向：

* 行：`slow_time_index`；
* 列：`fast_time_index`。

Python 库函数 `process_record()` 始终只接收上述 `slow-fast` 顺序，不会猜测或自动转置。
官方 `DistancePeriodicGraph.m` 的单个示例把原始 `channelA` 第 1 维记作 `Nr`（距离/快时间
处理维）、第 2 维记作 `Na`（周期/慢时间维），即该示例的原始矩阵是 `fast-slow`。因此从
原始 MAT 导出 `.npy` 后，CLI 必须显式提供 `--input-axis-order fast-slow` 才会在调用库函数前
转置；已经整理成合同顺序的数组使用 `--input-axis-order slow-fast`。程序不会仅凭形状猜方向。

这两个名字只是数组处理索引，不是代码对 chirp、PRF、ADC 采样率或脉冲语义的推断。K 频段样例可以是复数 IQ，L 频段样例可以是实数单通道；代码不会把这两个形式拼成 H/V。审计中的 `channelB` 全为空，因此空 `channelB` 永远只记录为“空/未解释”，不能称 H/V 极化。

## 处理链

1. 对每条记录分别计算幅度 RMS 和幅度第 99 百分位；各自除以独立尺度，保留两个归一化分支。尺度只来自当前记录，不跨记录估计。
2. 沿列（快时间）加 Hann 窗并对 K 复数、L 实数都做完整 FFT（不把 L 实数伪装成 IQ），得到 `[slow_index, fast_bin]` 的非负功率谱。
3. 对每个快时间频率 bin 沿行（慢时间）做不居中的 STFT，得到 `[slow_frame, slow_bin, fast_bin]` 的功率谱；记录过短时使用一个零填充帧。长度为 1 或 2 的退化 Hann 情形使用矩形极限，避免把有效非零输入静默乘成全零；更长窗口仍使用原有 Hann 定义。
4. 对快时间 bin 汇总得到慢时间谱 `[slow_frame, slow_bin]`。输出同时保留 RMS 和 percentile 两个分支。

`fast_bin_axis` 和 `slow_bin_axis` 是 `fftshift` 后的归一化 cycles/sample 轴；`slow_index_axis`、`fast_index_axis` 和 `slow_frame_index_axis` 是整数索引轴。没有真实 Fs、PRF、载频、波长和零频定义时，代码不生成 Hz、m/s、转速或物理微多普勒标签。

## 门禁和边界

接口拒绝非二维、空数组、NaN/Inf、超过元素/维度/幅度上限的数据，并在 STFT 展开前检查每个分支的输出元素上限（避免长记录把内存耗尽）。输出元数据固定包含 `model_training_allowed=false`、`physical_axis=false`、`physical_doppler_hz_axis_available=false`、`h_v_polarimetry_available=false` 和 `performance_reporting_allowed=false`。该结果可用于检查维度、窗口、归一化和绘图接口；不能把每个 STFT patch 当独立样本，不能随机拆分真实 MAT，也不能据此宣称公开数据模型性能。

通过 CLI 处理 `.npy` 时，`metadata.json` 还会记录 `cli_input.declared_axis_order`、原始
`source_shape`、是否执行了转置，以及库函数实际采用的 `contract_axis_order=slow-fast`。

运行合成 smoke（使用项目环境）：

```bash
conda run -n radar-torch python scripts/process_lss_fmcwr_normalized_v1.py --smoke --output-dir /tmp/lss_fmcwr_normalized_smoke
```

该命令只生成内存合成信号和 `.npz`/`metadata.json` 输出，不访问 `data/raw/`。

处理由官方示例顺序导出的二维 `.npy`：

```bash
conda run -n radar-torch python scripts/process_lss_fmcwr_normalized_v1.py \
  --input-npy /path/to/channelA.npy \
  --input-axis-order fast-slow \
  --band K \
  --output-dir /tmp/lss_fmcwr_single_record
```

`--input-axis-order` 对 `.npy` 是必填项；缺失时程序直接拒绝执行。这里的转置只统一数组
处理方向，并不补出未知的 Fs、PRF、载频或物理轴。
