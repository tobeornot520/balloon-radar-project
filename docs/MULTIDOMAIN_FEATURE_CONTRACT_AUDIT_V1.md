# 多域特征契约审查 V1

版本：2026-08-06

## 目的

`configs/multidomain_feature_contract_v1.yaml` 现在由可执行审查脚本检查，不再只是
说明文档。脚本在没有真实 IQ、标签或训练的情况下，核对配置与实现是否仍然一致，
为以后接入新采集数据提供一个低成本的漂移门。

## 审查内容

- 物理解释域及顺序：`time`、`polarimetric`、`range_doppler`、`time_frequency`、
  `trajectory`、`wind_dynamics`；
- 模型融合域及顺序：`quality`、`time`、`rd`、`polar`、`tf`；
- 固定维度：`3/11/22/8/12`，合计 56 个标量；
- 特征名称映射、融合模型默认维度和 YAML 维度三者一致；
- 缺失域必须使用显式 validity mask，并赋予零融合权重；
- 当前训练状态仍为 `scaffold_only`；
- 物理频率单位仍不可用，极化仍只能作相对量，时频仍只能作归一化频率描述；
- 轨迹和风动力学域继续保持阻塞。

## 运行

```bash
conda run -n radar-torch python scripts/audit_multidomain_feature_contract_v1.py
```

可用 `--output-json path/to/summary.json` 保存机器可读摘要。`status=PASS` 表示
“契约和代码接口一致”，不表示模型训练完成，也不产生 Pd、Pfa、AUC 或物理 Hz 结论。
任何维度、顺序、状态或缺失域策略的意外修改都会以非零退出和明确错误信息拒绝。

日常使用可运行统一入口：

```bash
conda run -n radar-torch python scripts/run_multidomain_preflight_v1.py
```

它会先执行本契约审查，再执行确定性合成 H/V 特征与融合 smoke。

统一入口还包含信号级不变量检查：公共相位旋转、公共幅度缩放的相对量不变性，以及
H/V 交换时功率比反号、相干不变性。该层只保护代码数学语义，不开放任何物理或性能声明。

## 与下一阶段的关系

真实 H/V IQ、PRF、同步和标定资料到位后，先运行本审查和
`run_multidomain_feature_smoke_v1.py`，再进入数据审计。审查通过只是入场条件；
仍需独立采集组划分、物理轴验证、标签审计和预注册训练协议。
