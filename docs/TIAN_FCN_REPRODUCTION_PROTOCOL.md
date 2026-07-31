# Tian 等 2024 FCN 方法复现协议

版本：2026-07-28

## 1. 复现定位

目标论文为 *Fully Convolutional Network-Based Fast UAV Detection in Pulse
Doppler Radar*，DOI `10.1109/TGRS.2024.3358956`。

论文原始实测数据未随项目提供。论文使用 86 张 `64 x 856` RD 图、6000 Hz
PRF 和 2-3 km 距离内的 UAV；当前项目数据为 H/V 双通道 `128 x 100` IQ、
1450 Hz PRF。当前工作因此定义为：

> 论文方法级复现与学长数据迁移验证，不是论文数值级重复实验。

论文 PDF 是本地阅读材料，不进入自动化运行依赖，也不作为生成结果的一部分。

## 2. 论文方法冻结项

- 整张 RD 图输入，不使用滑窗切片推理；
- 分类头和距离/多普勒归一化偏移回归头；
- 仅第一个卷积层由分类与回归共享；
- 卷积核依次为 `3x5`、`3x5`、`3x5`、`3x3`、`1x1`；
- 两个 `2x4` 池化层，总步长为 `4x16`；
- 理论感受野为 20 个多普勒单元乘 72 个距离单元；
- 分类标签由目标邻域经全一权重网络传播得到；
- 正样本与随机采样负样本按 1:1 计算分类交叉熵；
- 责任网格单元使用 Smooth L1 学习归一化位置偏移；
- 分类、回归和联合微调三阶段训练；
- 使用 PIR 动态阈值；在平均概率最高的连通域内，MDP 选择归一化偏移
  `sqrt(dx^2 + dy^2)` 最小的单元完成单目标定位。

## 3. 已知论文歧义

正文把目标邻域写作 `5 x 7` RD cells，但未在文字中完全明确轴顺序。主协议固定
为 5 个距离单元、7 个多普勒单元，反向轴序只允许作为预注册敏感性实验。

总损失公式说明回归权重为 10，而联合微调段落写为 20。主实验使用 10，20 仅作为
预注册敏感性实验。PIR 主规则使用 `max_probability - 0.1`；文中用于部分方法比较
的 0.9 固定阈值不替换 PIR 主规则。

## 4. 当前数据迁移项

这些项目不是论文原方法，必须单独标注：

1. `128 x 100` 输入右侧补零到 `128 x 112`，以适配 `4 x 16` 总步长；原始区域
   不缩放，落入补齐区的预测直接丢弃。
2. 论文训练图均围绕已标注目标；当前数据包含纯背景。背景图固定随机采样 16 个
   负网格单元，使其参与分类训练。
3. 论文 PIR 动态阈值在任意图上至少保留最大值附近候选，不能拒绝纯背景。项目
   正式评价必须再叠加只由验证集选择的绝对阈值。判决固定为
   `probability > threshold`，从而可以用阈值 1 拒绝饱和 sigmoid 背景。
4. H-only 是最接近论文单通道输入的主复现；V-only 和 HV 是迁移扩展。

## 5. 双协议评价

### 方法对齐协议

报告论文定义的 `Pd`、论文 `Pf`、`d_min`、`d_5`、`d_avg`、单图时间和 FLOPs。
论文 `Pf` 是检测结果中未匹配目标的比例，不等于背景样本虚警概率。
位置误差按论文定义使用 RD 单元上的欧氏距离：`d_min` 是每个 GT 到最近检测的
距离均值，`d_avg` 是每个检测到最近 GT 的距离均值，`d_5` 是距对应 GT 不超过
5 个单元的检测距离均值。导出字段为 `paper_d_min_euclidean_cells`、
`paper_d_5_euclidean_cells` 和 `paper_d_avg_euclidean_cells`。论文 `Pd/Pf` 还必须
注明 `Tdis`，不能与项目的 2 门/3 多普勒单元矩形容差混用。FLOPs 采用每个
卷积 MAC 计 2 FLOPs，且不含池化、激活、补齐和 sigmoid；报告中必须保留该口径。

### 项目可信协议

使用扫描组隔离划分；阈值和后处理只在验证集确定；测试集报告 joint Pd、背景
Pfa、距离/速度误差、折间分布和最差折。当前日期与类别混杂限制继续有效。

## 6. 执行门禁

1. 模型几何、目标构造、损失、冻结策略和 PIR/MDP 单元测试全部通过；
2. validation-only smoke 通过，且不加载 test；
3. 冻结配置、随机种子、评价口径和输出目录；
4. 先运行 H-only，再运行 V-only 和 HV；
5. 只有前述步骤通过后才启动六折正式训练。

本地阶段轮数预注册为分类、回归、联合各 20 epoch；这是论文没有冻结轮数后的工程
设定，不宣称是论文原始超参数。每阶段按验证结果恢复本阶段最佳权重，联合阶段按
`joint Pd`、`Pfa`、loss 依次选择。每折允许 2 个 validation 背景虚警；绝对阈值
冻结后才构造 test dataset，test 上不重新选择阈值。

受限接口 smoke：

```bash
python scripts/run_tian_fcn_reproduction_smoke.py
```

该 smoke 每个 split 每类最多读取两个样本，每个训练阶段只运行一个优化步，不输出
测试指标，不能作为性能证据。

完整训练入口的单折受控 smoke：

```bash
python scripts/run_tian_fcn_sixfold.py --smoke --folds 1 --channels H
```

正式 H-only 六折（不会由测试自动启动）：

```bash
python scripts/run_tian_fcn_sixfold.py --formal --channels H
```

若正式方法对齐配置出现训练退化，只允许使用单折 train/validation 诊断入口提出迁移修复：

```bash
python training/train_tian_fcn.py \
  --name tian_fcn_fold01_validation_diagnostic \
  --scope diagnostic \
  --manifest-path results/data_audit/dataset_v4_multifold/fold_01_manifest.csv \
  --fold-id 1 --channel H
```

`diagnostic` 使用完整 train/validation，但和 smoke 一样不构造 test。诊断候选必须先写明
变更原因和参数，再运行一次；不得遍历多个候选后用已见测试结果选择。

输出包括各阶段 checkpoint、训练历史、validation 阈值曲线、逐样本预测和跨折汇总。
正式 scope 才生成 test 预测；smoke 汇总必须记录 `test_split_loaded=false`。

## 7. 2026-07-28 至 2026-07-29 诊断结论

原始 H-only 六折正式流程已经执行，但随后发现旧版论文指标使用 L1 距离且 `d_min`、
`d_5`、`d_avg` 定义错误。因此这批六折输出只保留为失败迁移诊断，不是可引用的
正式复现结果，不允许用旧汇总器重新发布。

修正补齐区域过滤、欧氏 MDP 和论文指标后，Fold 1 扩展 GT 低学习率诊断仍为
validation joint Pd 0。53/53 个目标责任单元超过 PIR 阈值，但 0/53 被 MDP 选中。

预登记的单一 point-GT 本地迁移救援将 validation joint Pd 提升到 `22/53 =
0.4151`，背景图 Pfa 为 `2/150 = 0.0133`，且未构造 test。该变体改变了论文扩展
GT，必须标记为本地迁移消融，不能称为 Tian 原方法复现。当前仍不满足重新启动六折
测试的条件，原因是 MDP 责任单元选择率仅 0.1509，且多普勒定位存在明显长尾。

后续机制审计发现 53 张目标概率图与共同模板的相关系数均值为 0.99818，PIR 输出是
两个近乎固定的速度带。只提高随机负样本数量的预登记消融使 joint Pd 降至 0.2453，
没有缩小条带，故判为 `REJECT`。禁止继续扫描随机负样本数和 PIR 阈值；下一候选必须
直接约束同距离列内的速度定位，并继续保持 Fold 1 train/validation-only。

同距离列 31 个密集负单元的后续预登记诊断仍失败：模板相关系数为 0.99817，joint
Pd 为 0.1132，速度 MAE 为 32.30。该候选已判为 `REJECT`，且 point-GT 分类监督
分支关闭。恢复任何训练前，必须先取得并核对论文/学长数据的轴顺序、归一化、样本
分布和训练细节，或证明本地数据具备逐样本定位可辨识性。
