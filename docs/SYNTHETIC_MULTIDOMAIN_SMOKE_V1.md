# 多域特征合成接口 Smoke V1

版本：2026-08-06

## 目的

当前缺少新的 H/V 外场 IQ，不能再用真实数据产生新的泛化结论。但时域、极化和
STFT 接口仍可用确定性的合成信号做工程验收。本 smoke 将同一套信号贯通：

1. 生成 `[slow_time, range_gate]` 复数 H/V IQ；
2. 提取现有多域标量特征；
3. 按冻结顺序打包成 `quality/time/rd/polar/tf` 五个向量；
4. 送入 validity-mask 融合编码器，确认无效极化域权重严格为零。

它不读取 `data/`、不训练、不使用标签，也不计算 Pd、Pfa、AUC。

## 合成信号

`features/synthetic_radar.py` 提供两个仅用于接口测试的场景：

- `tone`：固定归一化慢时间频率，用于检查 H/V 相干和相对功率比；
- `sweep`：归一化频率从负值连续扫到正值，用于检查 STFT 主脊线的变化。

频率单位是“每个慢时间采样的周期数”，不是 Hz。没有 PRF 时，任何输出都不能解释
为物理微多普勒或旋翼频率。合成信号也不是目标电磁散射模型，不用于模型选择。

## 冻结的打包顺序

`features.multidomain_radar_features.MULTIDOMAIN_FEATURE_NAMES` 是标量映射到融合向量
的唯一顺序来源。`split_multidomain_features()` 只做顺序整理和有限值检查，不做归一化、
插值或缺失值填充。归一化统计量仍必须只在训练采集组上拟合。

当前维度固定为：

| 域 | 维度 |
|---|---:|
| quality | 3 |
| time | 11 |
| rd | 22 |
| polar | 8 |
| tf | 12 |

## 运行

```bash
conda run -n radar-torch python scripts/run_multidomain_feature_smoke_v1.py
```

如需保存机器可读摘要，可额外指定 `--output-json /tmp/multidomain_smoke.json`。输出中的
`status=PASS` 只表示接口贯通，不能替代真实数据审计或外场门禁。

## 通过条件

- 合成 H/V 固定相位关系的 coherence 接近 1；
- 相对 H/V 功率比与配置一致；
- `sweep` 的归一化 STFT 主脊线跨度显著高于 `tone`；
- 五个域的向量维度与融合模型一致且全为有限值；
- 将极化域标为无效后，其融合权重为 0。

## 边界

该 smoke 不能证明：

- H/V 硬件同步或绝对幅相标定；
- PRF、距离/速度物理轴或物理微多普勒频率；
- 数据划分、标签质量、检测性能或跨场景泛化；
- 空飘球、无人机或任何具体载荷的识别能力。

