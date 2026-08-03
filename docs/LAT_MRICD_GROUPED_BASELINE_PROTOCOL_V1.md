# LAT-MRICD 分组可解释基线协议 V1

版本：2026-08-03

## 1. 研究问题

在公开数据集 LAT-MRICD-1.0 内，检验不依赖物理 PRF、绝对距离或 H/V 极化解释的可解释
特征，能否在 X 波段、`batch_code` 留出条件下区分 UAV、鸟和气象三类记录。

本实验包含两个并列任务：

- `narrow_x_category`：512 点窄带复数 I/Q；
- `hrrp_x_category`：500 点 HRRP 非负幅度序列，用作同一公开发布内的表征互证。

HRRP-X 与 Narrow-X 不是独立外部数据，且存在重复 batch code；两者不得写成跨数据集外部
验证或同事件配对融合。

## 2. 冻结划分

随机按行拆分被禁止。正式运行只读取：

- `Narrow/X波段/data_narrow_X.mat`；
- `HRRP/X波段/data_hrrp_X.mat`。

不同时读取聚合文件和同目录明细文件。分折程序只把前四列元数据中的类别与 batch code
交给优化器，不把 I/Q 或 HRRP 信号列用于划分。每个 batch 整体进入一个 held-out fold。

五折分配使用 `metadata_only_balanced_milp_v1`：

1. 每个 batch 只能分配一次；
2. 每折总 batch 数位于理论均分的向下/向上取整之间；
3. 每折每类 batch-class cell 数位于理论均分的向下/向上取整之间；
4. 在满足上述硬约束后，最小化各类记录数和总记录数相对均分目标的绝对偏差；
5. 求解后按 batch 列表规范化折号，消除求解器对称解造成的编号漂移。

冻结清单位于 `data/splits/lat_mricd_x_batch_grouped_v1.csv`。它是一行一个 batch 的清单，
不含逐样本 ID、原始信号、文件绝对路径或模型输出。JSON 侧车记录配置、清单和两个聚合
MAT 的 SHA256，并明确 `signal_columns_used_for_assignment=false`。

## 3. 特征与单位

HRRP 特征使用每记录归一化后的能量质心、展宽、熵、峰位置、峰功率占比、峰均比、幅度
矩、累计能量分位位置、90%-10% 宽度、粗糙度和固定 lag 自相关。

Narrow 特征先按每记录 RMS 归一化，使用包络矩、峰均比、相位增量圆统计、固定 lag
自相关、周期候选以及加 Hann 窗后的归一化频谱质心、展宽、熵、主峰、零频邻域功率、
正负频率比、平坦度和谱峰度。

所有频率仅允许写成 `cycles/sample`。没有 PRF、连续时间戳和事件边界时，不换算 Hz、速度、
转速或物理微多普勒周期。

## 4. 固定模型和训练权重

保留三个并列结果，不根据 held-out 结果选择模型：

- `dummy_prior`；
- `logistic_batch_balanced`：训练折拟合 StandardScaler，L2，`C=1`，lbfgs；
- `random_forest_batch_balanced`：500 棵树，最大深度 8，叶节点最少 5 条记录，
  `max_features=sqrt`。

每条训练记录的权重为：

```text
1 / (该类别在训练折中的 batch 数 × 该 batch-category cell 的记录数)
```

随后归一化到平均权重为 1。该定义使三类总权重相等，并使同一类别内每个 batch-category
cell 的总权重相等。Scaler 和分类器只在训练折拟合。

不进行特征筛选、超参数搜索、PCA、神经网络训练或 held-out 驱动的阈值选择。

## 5. 评价和不确定性

主要分组指标定义为：

```text
r(g,c) = batch g 中真实类别 c 的逐行正确率
R(c)   = 对所有含类别 c 的 batch 等权平均 r(g,c)
primary_batch_macro = mean_c R(c)
```

同时报告：

- pooled accuracy、balanced accuracy、macro-F1、multiclass log-loss 和 macro OvR AUC；
- 五折 balanced accuracy 及最差折；
- batch 等权 accuracy、P10 和最差 batch；
- 每类 batch-class cell 的均值、最小值、P10、中位数和四分位数；
- 逐行与 batch-class 等权的两种混淆矩阵；
- 以 batch code 为聚类单位的 2,000 次 bootstrap 95% 区间；
- 逻辑回归与随机森林 primary metric 的配对 bootstrap 差值；
- 已见子型号的分层压力表，以及标准化线性系数/森林 impurity importance。

最差 cell 必须同时给出记录数；单条记录组成的 cell 不单独支持方法结论。五折只用于展示
异质性，不视为五次独立实验。

## 6. 工程冒烟与正式证据

冻结配置和元数据划分通过测试后，可以在 `/tmp` 执行一次完整工程冒烟，检查模型能否拟合、
表格能否生成以及数值是否有限。临时输出不进入 Git、分享包或论文证据，也不得据其指标
修改特征、模型或超参数。正式运行必须绑定冒烟之后的实现 Git 提交，并输出实现、配置、
数据聚合文件和冻结划分哈希。

## 7. 结论边界和停止规则

允许表述：

> LAT-MRICD-1.0 内、X 波段、batch-code-held-out 的三类可解释特征基线。

不允许表述：

- 独立 session、跨场景或外部盲测泛化；
- 未见无人机型号泛化；
- 物理微多普勒 Hz、速度、绝对距离或 RCS；
- H/V 极化性能、空飘球载荷识别或 Tian 2024 精确复现。

batch 采集语义仍未独立确认，因此 batch-code 隔离只是保守分组代理。若 grouped 指标明显
退化，保留负结果，不通过扩大网络或随机拆行挽救。正式结果若用于后续选择特征或模型，
必须另设 grouped 内层验证或取得新的 locked test；本次 OOF 结果不能反向成为选择集。
