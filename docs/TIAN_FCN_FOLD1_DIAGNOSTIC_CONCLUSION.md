# Tian FCN Fold 1 诊断结论

日期：2026-07-29

## 结论

Tian 2024 FCN 在当前本地 H 通道数据上的第一版迁移失败。原始六折输出不是成功复现，
旧版论文距离指标也不可信。修正实现后，扩展 GT 的 Fold 1 validation joint Pd 仍为 0。

预登记的 point-GT 单一救援在不加载 test 的条件下得到：

| 指标 | 结果 |
|---|---:|
| 验证目标 | 53 |
| 验证背景 | 150 |
| 正确检测与定位 | 22 |
| joint Pd | 0.4151 |
| 背景虚警 | 2 |
| Pfa | 0.0133 |
| 责任单元为分类峰值 | 0.3208 |
| 责任单元被 MDP 选中 | 0.1509 |
| 责任单元 oracle joint Pd | 0.5660 |

point-GT 证明“扩展分类 GT 与单点回归责任单元失配”是本地迁移失败的重要原因，但
没有完全解决定位。实际 MDP 解码的欧氏误差均值为 19.51 个 RD 单元，最大值为
88.46；项目协议下速度 MAE 为 18.57 个单元，存在明显长尾。

## 实现审计

- 补齐区域候选在 PIR 和连通域处理前过滤；
- MDP 先选平均概率最高的连通域，再按 `sqrt(dx^2 + dy^2)` 选最小偏移；
- 论文距离改为欧氏 RD 单元距离；
- `d_min`、`d_5`、`d_avg` 按论文集合定义计算；
- point-GT 只改分类目标模式，学习率、数据折、通道、seed、回归目标保持冻结；
- 两次 Fold 1 诊断均未构造 test。

## 决策

暂不运行新的六折、V-only 或 HV 实验，也不继续扫描学习率。下一次模型实验必须先
形成新的预登记，直接处理多普勒责任单元可辨识性和 MDP 选择问题；在此之前，当前
结果仅用于方法诊断。

后续连通域与概率模板审计确认，point-GT 输出主要是两条固定速度带，而不是逐样本
定位响应；目标概率图与共同模板的平均相关系数为 0.99818。预登记的 16 个随机负
样本下限消融未消除条带，joint Pd 降至 0.2453、速度 MAE 升至 24.92，已判为
`REJECT`。详见 `docs/TIAN_FCN_FOLD1_COMPONENT_MECHANISM.md`。

随后预登记的同距离列密集负监督仍未打破模板：相关系数 0.99817，joint Pd 进一步
降至 0.1132，速度 MAE 升至 32.30。point-GT 分类监督分支到此停止，下一阶段只做
论文/学长数据的输入与采样条件核对，以及本地数据可辨识性审计，不启动新训练。

对应证据：

- `results/experiments/tian_fcn_point_gt_rescue_diagnostic_v1_fold01/`
- `results/data_audit/tian_fcn_v1/fold01_point_gt_validation_audit_v1/`
- `configs/tian_fcn_point_gt_rescue_diagnostic_v1.yaml`
