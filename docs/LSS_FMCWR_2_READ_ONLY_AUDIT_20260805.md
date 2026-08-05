# LSS-FMCWR-2.0 V4 只读 RAR/MAT 审计

更新日期：2026-08-05

状态：`PASS_ARCHIVE_SCHEMA_BLOCKED_GROUPING_PROVENANCE_AND_PHYSICAL_AXIS`

本审计只回答“文件是否完整、MAT 里实际有什么、哪些文件重复、能否建立可信独立分组”。
它没有训练模型，也没有产生分类性能。

## 1. 来源和版本边界

- 数据 DOI：`10.57760/sciencedb.radars.00054`；ScienceDB 版本：V4；
- 论文 DOI：`10.12000/JR25004`；
- 许可登记：`CC-BY-NC-ND-4.0`；原始 RAR、重打包数据和成员级内容不进入 Git 或分享包，
  派生数据公开前还要单独审查 ND 边界；
- ScienceDB V4 的正式 8 个文件是 6 个 RAR、`DistancePeriodicGraph.m` 和微多普勒处理 PDF，
  共 `1,013,535,036` bytes；
- `LSS-FMCWR-2.0_usage_instructions.pdf` 是另从《雷达学报》数据栏目取得的说明，不属于上述
  V4 八文件，不能把两种计数混写；
- 六个 RAR 的官方 MD5、本地 SHA-256 和字节数均已通过。

原始文件保存在 Git 忽略的 `data/raw/external/LSS-FMCWR-2.0/`，本轮没有移动、改名或落盘
解压它们。

## 2. 可重复审计入口

保留未完成开发稿 `scripts/audit_lss_fmcwr_2_v1.py` 的内容不变；正式入口为：

```bash
conda run -n radar-torch python \
  scripts/audit_lss_fmcwr_2_hdf5_v1.py \
  --unrar /path/to/unrar-nonfree \
  --output-dir results/data_audit/lss_fmcwr_2_v1
```

本次实际读取器为 UNRAR `7.20`，Ubuntu 包版本 `1:7.2.4-1`；二进制 SHA-256 为
`b7108a21c3276d0eab35ed1910995c2c5a150ad66582ee99f0800043be564cff`。该临时二进制不进入
仓库。入口固定 `C.UTF-8`，禁用外部配置和 list-file 展开，并使用 option terminator。

处理方式为：

1. `unrar lt` 读取 RAR 5 清单并检查路径、尺寸、CRC 和条目类型；
2. `unrar t` 对六个压缩包做完整性测试；
3. `unrar p` 逐个把 MAT 流入内存，不解压到源目录；
4. MATLAB v5 由 `scipy.io.loadmat` 读取；v7.3 由 `h5py` 读取并恢复 MATLAB 逻辑维度；
5. 对原始 MAT 字节和解码后的 `channelA/channelB` 数值分别计算 SHA-256；
6. 先写事务目录，全部检查通过后才原子替换本地审计结果。

本地成员路径和成员哈希位于 Git 忽略的
`results/data_audit/lss_fmcwr_2_v1/`。可提交文档只保留聚合统计和声明边界。

## 3. 冻结结果

| 项目 | 结果 |
|---|---:|
| RAR | 6 |
| RAR 条目 | 116 |
| MAT | 90 |
| 目录条目 | 26 |
| 解压后 MAT 总字节 | 1,041,307,141 |
| K 频段 MAT | 64 |
| L 频段 MAT | 26 |
| MATLAB v5 / v7.3 | 84 / 6 |
| 去掉 ordinal 后的候选 recording stem | 66 |
| 原始 MAT 唯一内容 | 71 |
| 解码数值唯一内容 | 71 |
| 精确重复组 / 涉及成员 | 11 / 30 |
| stem 与精确重复边连通后的保守候选组 | 48 |
| 跨目标精确重复组 | 0 |
| 路径角度与文件名角度冲突 | 1 |

官方标题/表格写 90 组，实查也有 90 个 MAT；但使用说明第 2 页各分项相加为 86。文件命名
说明还遗漏了实际存在的 `4` ms token。审计保留这些矛盾，不替发布方静默修正。

### 3.1 目标和候选组

| 目标 | MAT | recording stem | 保守候选组 |
|---|---:|---:|---:|
| DJI M350 | 18 | 15 | 10 |
| DJI Inspire 2 | 18 | 15 | 12 |
| DJI Mavic 2 | 18 | 15 | 10 |
| 六旋翼无人机 | 18 | 15 | 10 |
| 仿真飞鸟 | 8 | 2 | 2 |
| AC311 直升机 | 10 | 4 | 4 |

`recording stem` 和 48 个连通组都是基于文件名和精确内容构造的保守工程键，不是发布方确认
的 session、场景、物理目标或独立采集事件。它们只能防止已知重复跨 split，不能证明外层
评价独立。

### 3.2 通道和数组形状

所有 90 个 MAT 都只有公开变量 `echoes`，其字段为 `channelA` 和 `channelB`；全部数值有限。

- K：64 个 `channelA` 均为 `complex128`；
- L：26 个 `channelA` 均为 `float64`；
- `channelB`：90/90 都为空 `(0, 0)`；
- 因此只有单个有效数据通道，不能解释成 H/V 双极化或双接收通道；
- K 与 L 的存储类型不同，不能不加说明地逐元素拼接或共享同一归一化规则。

| 频段 | `channelA` 逻辑形状 | MAT 数 |
|---|---|---:|
| K | `150 x 6000` | 29 |
| K | `500 x 6000` | 16 |
| K | `512 x 6000` | 13 |
| K | `2000 x 6000` | 3 |
| K | `2000 x 5704` | 1 |
| K | `500 x 102400` | 1 |
| K | `500 x 43439` | 1 |
| L | `150 x 6000` | 21 |
| L | `150 x 4000` | 5 |

六个 v7.3 文件在 HDF5 中把维度反向保存，并把复数拆成 `real/imag` compound dtype；兼容
入口将它们恢复到与 v5 相同的 MATLAB 逻辑方向。未恢复维度就统计会把例如
`150 x 6000` 错写成 `6000 x 150`。

### 3.3 角度、时长和重复

文件名/目录中出现的采集角度 token 为 `0, 60, 90, 120, 180`，时长 token 为
`0.3, 1.0, 1.024, 4`。这些角度只能称“采集/雷达夹角 token”，尚不能称为经过核验的目标
姿态角或同一目标的严格配对角度。

唯一冲突位于六旋翼 L 频段：目录为 `120`，文件名内部角度为 `0`。审计采用目录值作
`collection_angle`，同时保留冲突标记；任何分析不得静默当作已确认 120 度样本。

11 个原始字节重复组与 11 个解码数值重复组一致，共涉及 30 个 MAT，说明这不是“数值相同
但容器不同”的近似重复。重复既出现在相同 stem 的 ordinal 内，也出现在不同角度目录之间。
所以 90 个 MAT 绝不能被写成 90 个独立样本，也不能随机拆分 MAT、frame 或 window。

## 4. 物理解释门禁

官方示例脚本只为一个 `Br=100 MHz, 0.3 ms` 示例给出 `Fs=500 kHz`，不能外推到所有目标、
频段、时长和数组形状。当前发布中没有可统一核验的：

- 全局采样率、PRF/慢时间间隔；
- K/L 各自载频和波长；
- 零多普勒位置、FFT/窗函数/重叠率；
- Doppler bin 到 Hz 或径向速度的映射；
- 完整 `md_stft` 实现；
- 采集日期、场地、雷达运行、天气和物理目标 session 键。

因此可以在归一化 bin 或 cycles/sample 上设计处理接口，但暂不能把谱带宽、脊线间隔、周期
或速度写成物理 Hz、m/s 或转速。

## 5. 放行与禁止

当前各门状态：

| 门 | 状态 |
|---|---|
| V4 身份、RAR 5 路径和完整性 | `PASS` |
| MAT schema、有限值、K/L 类型 | `PASS` |
| 精确重复隔离 | `BLOCKED_DUPLICATES_PRESENT` |
| 独立 recording/session 身份 | `BLOCKED_NOT_AVAILABLE` |
| 物理时间/Doppler/速度轴 | `BLOCKED_NOT_AVAILABLE` |
| 自然鸟证据 | `BLOCKED_SIMULATION_ONLY` |
| 模型训练 | `BLOCKED` |

当前允许：

- 复跑只读 RAR/MAT/schema/重复审计；
- 构建不接触标签性能的合成微多普勒处理合同；
- 在本地用归一化轴做单记录读取、可视化和方法接口验证；
- 设计将来按已确认 session 隔离的 K/L、角度和目标任务。

当前禁止：

- 把 90 个 MAT、切窗或 STFT patch 当作独立样本随机拆分；
- 把 48 个候选组写成作者确认的 session；
- 训练分类器或汇报准确率、Pd/Pfa；
- 把不同角度 token 当作同一事件的同步多视角；
- 把仿真飞鸟写成自然鸟外部验证；
- 宣称 H/V 极化、绝对极化散射或空飘球载荷识别；
- 使用示例 `Fs=500 kHz` 给全包建立物理 Hz/速度轴。

## 6. 下一步

FMCWR-2.0 的 archive/schema 审计已经完成，但该数据集的建模方向尚未完成。归一化轴
合成/单记录处理合同及其合成 smoke 已在
[`LSS_FMCWR_2_NORMALIZED_PROCESSING_CONTRACT_20260805.md`](LSS_FMCWR_2_NORMALIZED_PROCESSING_CONTRACT_20260805.md)
完成；它不读取真实 RAR/MAT，也不产生物理轴或性能结论。下一步不是立即训练，而是取得
可归因事实：

1. 向发布方确认 recording/session 对应关系、各频段/时长的采样率、PRF、载频、零频和
   微多普勒处理参数；
2. 若暂时联系不到发布方，只复核现有合成信号和单记录归一化接口，不产生性能声明，并把
   后续模型门保持关闭。

这一结论不改变主项目 `4/6 BLOCKED_EXTERNAL`：现有公开数据仍不能替代同条件 H/V、连续
慢时间、同步标定和空飘球有载/无载真值。
