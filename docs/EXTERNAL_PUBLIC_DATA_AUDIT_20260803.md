# 外部公开雷达数据核验与使用计划

版本：2026-08-04

## 1. 来源核验

| 用户提供入口 | 实际对象 | 本轮状态 | 决策 |
|---|---|---|---|
| 本地《低空目标多频段雷达识别特性数据集》PDF | LAT-MRICD-1.0，DOI `10.12466/xhcl.2026.06.001` | 官方 ScienceDB ZIP 已下载并校验 | 纳入 HRRP 和归一化微动算法预研 |
| `10.12466/xhcl.2025.05.003` | 地杂波背景下雷达低慢小无人机探测数据集 LSS-Ku-1.0 | 官方下载入口已定位，ZIP 为 15,985,618,112 bytes，未下载 | 保留为地杂波检测/Tian 同域性候选，先向学长确认价值 |
| `radars.ac.cn` | 《雷达学报》数据栏目；正确路径含 `/cn/`，正式文件由 ScienceDB 托管 | 已核对 LSS-DAUR、LSS-HSR-L、LSS-FMCWR-1.0/2.0 的 DOI、许可和文件树 | 下载 DAUR V3、HSR-L ScienceDB V2 与期刊历史包、FMCWR-2.0 V4 及两版 FMCWR 说明；原始数据不进 Git/分享包 |
| ScienceDB LSS-DAUR-1.0 | 数据 DOI `10.57760/sciencedb.radars.00076`，V3 | 314/314 文件、148,763,512 bytes 完整；77 对 TD/TR 的 schema、配对与 canonical/backup 数值等价性已通过 | 分组键、严格时间和 1024-bin 物理轴阻塞；不训练，下一项转 HSR-L V2 只读审计 |
| ScienceDB LSS-FMCWR-2.0 | 数据 DOI `10.57760/sciencedb.radars.00054`，V4 | 8/8 文件、1,013,535,036 bytes 已下载；6 个 RAR 的官方 MD5 全通过 | 优先做 K/L、角度和微多普勒 schema 审计；鸟类包只作仿真数据 |
| ScienceDB DroneRFc-MM | 数据 DOI `10.57760/sciencedb.j00173.00094`，V1 | 从 75.6 GB/113 文件的完整发布中选择下载 28 个雷达、真值、标签、说明和代码文件，共 47,366,902 bytes | 用于 77--81 GHz 点云、轨迹和时间同步接口审计；不下载无关相机、音频、LiDAR 与通信 RF |
| ScienceDB UAV 群目标 | 论文 DOI `10.12466/xhcl.2025.05.004`，数据 DOI `10.57760/sciencedb.25500`，V1 | 3,996,753-byte 官方 ZIP 已下载，SHA256、ZIP 安全和 MAT schema 检查通过 | 只做 Ku 波段 XYZ 量测、航迹关联和跟踪接口 smoke；5 个 MAT 只对应 3 个物理实验 |
| NOAA NEXRAD Level II | Open Data Registry 的现行 `unidata-nexrad-level2` 桶 | 只下载一个 395,379-byte KTLX Archive II 体扫并核验实际双极化矩 | 只做 Z/V/SW/ZDR/PhiDP/RhoHV loader 和公式 smoke；无 UAV/气球标签 |
| Zenodo Radar Signature Dataset | 数据 DOI `10.5281/zenodo.7573165`，版本 0.1 | README 和 metadata 已校验；42,402,413,502-byte 主归档未下载 | 仅保留室内铝箔数字气球 range-profile/ISAR 接口候选，不外推户外空飘球载荷 |
| 继续检索到的两个大体量实测候选 | ScienceDB `10.57760/sciencedb.18323`；Zenodo `10.5281/zenodo.18553708` | 23.46 GB S 波段野外 UAV 与 30.36 GB 24/94/207 GHz UAV/真鸟数据均未下载 | 来源和分组风险已登记；先完成现有小数据审计，再决定是否承担传输和重建 split 成本 |

LAT-MRICD 的论文页面标注文章采用 CC BY 3.0；文章许可不自动等同于原始 ZIP 可再次分发，
因此仍按 `NOASSERTION` 保守处理数据文件。ScienceDB 的 DAUR/FMCWR V3/V4 元数据明确标注
`CC BY-NC-ND 4.0`，HSR-L 的 ScienceDB V2 标注 `CC BY-NC 4.0`，DroneRFc-MM V1 标注
`CC BY-SA 4.0`；UAV 群目标为 `CC BY-NC 4.0`；Radar Signature 与三频 UAV/真鸟候选为
`CC BY 4.0`。NEXRAD 属 NOAA 开放数据，未硬套 SPDX：使用时应署名，不得暗示 NOAA
背书，修改后不得冒充 NOAA 未修改原件。HSR-L 的 V2 正式包和较早期刊 bundle 均已取得；两者在大小、条目数、
目录、元数据和同名样本长度上均不同，已确认不等价，禁止混合。期刊 bundle 自身条款仍按
`NOASSERTION` 处理。
原始及重新打包的数据只留在本地受控目录，不进入 Git 或分享包；公开派生数据前必须另做
ND 条款判断。

## 2. 本地原件与完整性

| 项目 | 值 |
|---|---|
| 数据集 | `LAT-MRICD-1.0` |
| ZIP 大小 | 181,367,543 bytes，约 173 MiB |
| ZIP SHA256 | `2fe0d5e89016382c7c980172d67ba640179d6e2724edc735bcdf65c66b533bc0` |
| ZIP 完整性 | 通过 |
| MAT 文件 | 33 个，其中 5 个聚合文件 |
| Git/分享状态 | `data/raw/` 与 ZIP 均已忽略，不提交、不打包 |

正式审计入口：

```bash
python scripts/audit_lat_mricd_dataset_v1.py --overwrite
```

输出位于 `results/data_audit/lat_mricd_v1/`，属于可重建的本地审计结果。

## 3. 数据结构与实测规模

HRRP 每行 504 列：前 4 列依次为频段、类别、型号、批次，后 500 列为非负幅度序列。
窄带数据每行 1,028 列：前 4 列为同一组元数据，后 1,024 列按 I/Q 交替存储，可还原
为 512 个复数样本。33 个文件均通过维度、有限值、整数元数据、频段和类别/型号一致性检查。

| 表征 | 频段 | UAV | 鸟 | 气象 | 合计 |
|---|---|---:|---:|---:|---:|
| HRRP | X | 1,411 | 1,152 | 1,085 | 3,648 |
| HRRP | Ku | 1,765 | 0 | 1,706 | 3,471 |
| Narrow I/Q | S | 1,531 | 0 | 2,507 | 4,038 |
| Narrow I/Q | X | 2,169 | 963 | 5,583 | 8,715 |
| Narrow I/Q | Ku | 2,598 | 0 | 721 | 3,319 |
| 合计 | - | 9,474 | 2,115 | 11,602 | 23,191 |

类别码为 `1=UAV`、`2=bird`、`3=weather`；频段码为 `1=S`、`2=X`、`3=Ku`。
型号覆盖 Mavic 2、Phantom 4、Air 3S、M30T、穿越机、自制 UAV、信鸽、大雁、气象杂波，
以及 Ku-HRRP 中未细分型号的 UAV。

## 4. 划分风险和放行结论

审计得到 911 个 `(representation, band, batch)` 组，其中 46 个 batch 编号同时关联多个
型号或类别。94.95% 的批次组只含单一类别，说明随机按行拆分极易把同批次模式同时放进
训练和测试，并夸大泛化结果。

放行规则：

1. 禁止随机按行划分；最低要求按 `(representation, band_code, batch_code)` 分组。
2. batch 的采集语义尚无独立说明，先采用上述保守分组，不把 batch 当成已验证 session。
3. 12 个“表征-频段-型号”组合不足 3 个 batch，不放行这些组合的细粒度型号随机交叉验证。
4. 每个已发布“表征-频段-大类”至少有 3 个 batch，可开展预登记的大类分组基线。
5. 跨频段实验只能称 band-held-out transfer；没有同一事件配对证据，不能称配对多频融合。

审计状态 `READY_FOR_PREREGISTERED_GROUPED_BASELINE` 是算法运行前的放行结论，不代表
外部泛化。基于该门禁完成的 D17-NX/HX 冻结结果见第 5.1 节。

## 5. 可开展的算法工作

| 分支 | 第一轮输入/特征 | 第一轮任务 | 单位与边界 |
|---|---|---|---|
| HRRP | 归一化幅度、质心、展宽、熵、峰度、峰值/旁瓣和低维投影 | UAV/鸟/气象分组分类、X 到 Ku 迁移 | 样点和相对幅度，不解释为绝对距离/RCS |
| Narrow I/Q | 包络、相位增量、自相关、归一化 Doppler 谱、谱熵、谱展宽和周期候选 | UAV/鸟/气象分组分类、S/X/Ku 留一频段迁移 | 频率仅用 cycles/sample 或归一化频率 |
| 多频段 | 各频段独立训练/评价后比较稳定特征 | band-held-out transfer | 不做样本级拼接，不假设同一事件同步 |

第一轮使用可解释特征加线性/树模型建立泄漏敏感性基线，同时报告 batch 级 macro、最差组和
混淆矩阵。只有分组基线稳定后才考虑一维 CNN；不先用深网掩盖数据划分问题。

### 5.1 已冻结的 X 波段分组基线

D17-NX 与 D17-HX 已完成。正式实验在提取信号特征前冻结元数据划分，只读取 X 波段聚合
文件，按 `(representation, band_code, batch_code)` 整组留出五折；逻辑回归、随机森林及
训练权重均预先固定，不根据 held-out 结果选模型。

| 任务 | 固定模型 | batch-class macro accuracy | batch-code cluster bootstrap 95% CI | 最差折 balanced accuracy |
|---|---|---:|---:|---:|
| Narrow-X | batch-balanced logistic | 0.7999 | 0.7659–0.8313 | 0.7204 |
| Narrow-X | batch-balanced random forest | 0.7872 | 0.7373–0.8340 | 0.6973 |
| HRRP-X | batch-balanced logistic | 0.6617 | 0.5826–0.7404 | 0.4946 |
| HRRP-X | batch-balanced random forest | 0.6481 | 0.5764–0.7240 | 0.4934 |

随机森林减逻辑回归的配对差值在 Narrow-X 为 -0.0127（95% CI -0.0511–0.0246），在
HRRP-X 为 -0.0136（-0.0775–0.0487）；两者均跨 0，因此保留两个固定模型，不宣布胜者。
完整聚合证据位于 `results/final_evidence/lat_mricd_grouped_baselines_v1/`。

这些数字只支持“同一 LAT-MRICD-1.0 公开发布内、X 波段、batch-code-held-out 的三类
基线”。它们不是未见型号、独立 session、跨场景或外部盲测泛化；HRRP-X 与 Narrow-X
也不能称为两个独立外部数据集的重复验证。

### 5.2 已完成并冻结的预登记跨频段负结果

`docs/LAT_MRICD_CROSS_BAND_TRANSFER_PROTOCOL_V1.md` 与
`configs/lat_mricd_cross_band_transfer_v1.json` 已冻结。正式主分析只把 Narrow-X 作为源，
分别把 S 和 Ku 作为一次性 locked target；任务固定为共同 UAV 型号与 weather 二分类，
bird 不进入该任务。HRRP 和其余方向只作预先声明的压力分析。

在查看 target 性能前，dummy、batch-balanced 逻辑回归和固定随机森林均已冻结；正式运行
只执行一次，未用 target 拟合缩放、调参、选模或扩特征。主模型 LR 的结果如下：

| 迁移 | target batch-class macro accuracy | UAV batch recall | weather batch recall | LR-minus-dummy 95% CI 下界 |
|---|---:|---:|---:|---:|
| X->S | 0.6516798767 | 0.4433201701 | 0.8600395832 | 0.0839896354 |
| X->Ku | 0.8399853939 | 0.8493090645 | 0.8306617233 | 0.2670982858 |

继续门要求两个 locked target 同时满足：逻辑回归的 target batch-class macro accuracy
严格大于 0.60；UAV/weather 两类 target batch 等权召回均严格大于 0.50；以及
LR-minus-dummy 配对 95% CI 下界严格大于 0。实际八项条件中仅 S 频段 UAV recall 未通过，
因此总门禁状态为 `FAIL_STOP`，预登记负结果已冻结。S/Ku target 均已消费，禁止在同一
target 上继续 CNN、域适配、特征扩展、结果驱动调参或新的确认性模型比较。最终证据位于
`results/final_evidence/lat_mricd_cross_band_transfer_v1/`。该结果不改变主 UAV 方向 4/6
`BLOCKED_EXTERNAL`。

## 6. 不能由该数据集解决的问题

- 没有 H/V 成对通道，不能用于极化特征、相干相位或绝对标定验证；
- 当前材料未确认 PRF，不能把频谱轴换算成 Hz 或速度；
- 当前材料未确认逐脉冲时间戳、连续 session 和丢帧，不能证明因果时序；
- 没有空飘球、有载/无载、载荷类型和状态标签，不能训练项目最终分类器；
- 不包含 Tian 论文的输入/标签/输出对齐样例，不能解除现有 Tian 复现条件门；
- 不提供现有 H/V UAV 数据的设备、坐标和处理链事实，不能关闭 D01/D02。

因此，LAT-MRICD 使 D11 从“完全没有可用微动原始量”推进到“可做归一化窄带微动算法
预研”，但物理微多普勒、因果时域和空飘球迁移仍然阻塞。

## 7. 下一动作与停止点

1. D17-NX/HX 的五折划分、固定模型、分组指标、CI 和边界已经冻结，不再用这些 OOF 结果
   反向筛特征或调参；
2. D17-XBAND 已完成一次提交绑定的密封运行并冻结为 `FAIL_STOP`；S/Ku target 已消费，不再
   用同一 target 扩模、调参或做新确认性比较；
3. 跨频段只比较各频段独立训练/评价后的迁移表现，不做样本级拼接，不声称同一事件同步；
4. 如需新的跨频确认性结论，必须取得独立、未消费 target 并重新预登记；当前不评估轻量
   一维 CNN 或域适配来挽救该结果。

LSS-Ku-1.0 仅在学长确认它能补 Tian 同域性或地杂波测试缺口后再下载，避免为 14.88 GiB
数据付出传输、存储和审计成本却不能解除当前门禁。

## 8. 2026-08-04 新增公开数据下载回执

| 数据集 | 本地状态 | 完整性结论 | 下一门禁 |
|---|---|---|---|
| LSS-DAUR-1.0 V3 | 314 个官方文件完整下载并完成全量只读审计 | 文件数 `314`、总字节 `148763512`、清单 SHA256 `5febc59a29c42fb7dd8b001afa73fe913767f00c2c060a21a5d95073c2ee1745`；308 MAT 全部可读有限；77 对 TD/TR 对齐；canonical/backup 共享数值完全相等 | `PASS_SCHEMA_PAIRING_BLOCKED_GROUPING_AND_PHYSICAL_AXIS`：每轨重复时间、6 个日期冲突、session key 未确认、19 条 1024-bin 轴无说明；禁止训练与随机切窗 |
| LSS-HSR-L ScienceDB V2 | 正式 ZIP 已下载 | `237020946` bytes；SHA256 `fea8a21354110a96fb9644dc1c69649b6dc6d1a1b6da512498d9c2d74d839540`；1,561 个条目且 ZIP 完整 | 以 V2 为唯一规范审计候选，实现只读 loader；核对 `train/validation/overflow` 与原始场景/轨迹关系 |
| LSS-HSR-L 期刊历史包 | ZIP 已下载 | `209569478` bytes；SHA256 `22112d4225636c5626845a9f0640abbf4503cc70763b592a609287760ab5f4a4`；1,478 个条目且 ZIP 完整 | 已确认不等同于 V2，只保留版本史，不混合建模；禁止直接运行会移动原件的官方 `dataset.py` |
| LSS-FMCWR-2.0 V4 | 8 个官方文件完整下载 | 总字节 `1013535036` 与发布元数据一致；6 个 RAR 的官方 MD5 全通过，另生成 SHA256 | 先配置可审计的 RAR 5 解包工具，再按 MAT 实查官方“90 组”与分项合计 86 的矛盾 |
| LSS-FMCWR-1.0 | 仅下载 1.0 使用说明 PDF | 当前没有原始 RAR | 等 FMCWR-2.0 审计后，仅在能补明确 UAV 型号/双频缺口时下载主体 |
| St Andrews 微多普勒仿真包 | 未下载 | DOI 页面与 CC BY 可核验，但官方文件端两次返回 HTTP 403；没有保留部分文件，也未绕过访问限制 | 只在官方端恢复或数据持有人授权后用于 smoke，不作为真实外部性能证据 |
| DroneRFc-MM V1 雷达子集 | 28 个官方文件完整下载并完成只读 schema/时间覆盖审计 | `47366902` bytes；9 个 ZIP/30,717 个 PCD/639,527 个点均通过 CRC、15 列 schema、有限值、POINTS 行数和文件名时间戳检查 | 总门为 `BLOCKED_TIMESTAMP_ALIGNMENT`：B1 雷达与同名 GT 零重叠；B1 禁止监督对齐，其余 8 条只允许后续预登记同步研究 |
| Radar Signature Dataset 0.1 | 只下载 README 与 metadata | README `125546` bytes、SHA256 `34900026bc5081d45eccaf69df62fac65173cd2c8b0520d8088cc335941b2171`；metadata `27732` bytes、SHA256 `6c58ab862caa82cebe72403aed4c33fd61ae3c6aea71cf45bcfe1dcd0d7224b3`；不执行不可信 pickle | 42.4 GB 主包暂缓；即使以后下载，也只按完整测量序列/物理气球分组做室内接口 smoke |
| 低空 UAV 群目标 V1 | 官方 ZIP 已下载并完成最小 schema 审计 | `3996753` bytes；SHA256 `c08e5c93d59d1012f134a3ffa7521eb4d26fb7cfc7a8bcbc297574273350a76e`；7 个安全条目、5 个 MATLAB v5 文件均可解析 | 按 `{Exp1}`、`{Exp2_1,Exp2_2}`、`{Exp3_1,Exp3_2}` 三组物理实验冻结外层分组，只做量测/航迹 smoke，不训练识别器 |
| NOAA NEXRAD Level II | 只下载一个 KTLX 体扫 | `395379` bytes；SHA256 `e6092212670064ebc4da0e38738b38e9f965425c2f219a512c057693211d5c9b`；`AR2V0006.208`、34 个 bzip2 块，实际含 DREF/DVEL/DSW/DZDR/DPHI/DRHO | 以完整体扫为最小单位做 loader/缺失矩 smoke；后续若扩充，按站点、UTC 日和天气事件隔离 |

DAUR 的 canonical 77 TD + 77 TR 为 MATLAB v5，`backup_original` 的 154 个对应文件为
v7.3/HDF5 并额外保留 `File_head`，所以文件级 bytewise 不同；但所有共享数值经 MATLAB
存储顺序对齐后逐元素完全相等。它们是同一 77 个逻辑记录 ID 的两种视图，不是 154 条观测，
不能扩样或跨 split，也不删除原始 backup。全包共 11,366 帧、7,728,640 个有限复数 DPL
值；77 对 TD/TR 的 `DATA_time`、`GPS_time_in_data`、`Iframecnt` 和 `nDaCf` 完全一致。
其中 1 组 2 个 recording 的 canonical TD/TR 内容完全重复，只留下 76 个唯一内容对；
另有 11 对 recording 共享内部 `(GPS_time_in_data, Iframecnt)` 帧。把相同文件名时间、
`File_head` 日期加文件名时刻、共享内部帧及完全重复内容保守连通后，只形成 39 个候选
source-session 组；该键仍未经发布方确认。

严格时间门未通过：全部 77 条都有重复时间戳，894 个相邻重复只留下 10,472 个唯一时间
位置；13 条还有帧号跳跃或重复。6 条文件名日期与 `File_head` 日期冲突，禁止静默修正及
绝对日期/天气拼接。文件名前缀形成 45 个候选 session，header 日期加文件名时间形成 40
个，但都没有官方定义；两种候选键下 Bird 与 UAV 均零重叠，24 个 header-date/scene 组中
20 个类别纯，存在严重域捷径。58 条为 512 bins，19 条为 1024 bins；官方脚本给出
1.36 GHz、PRI 200 us、PRF 5 kHz 和约 0.1346 m/s，但固定按 512 bins 绘图，不能据此解释
1024-bin 物理轴。`V` 字段全零，不得作为特征。当前仅允许只读 loader、归一化 bin/轨迹
方法设计与泄漏审计；正式模型训练仍为 `BLOCKED`。

FMCWR-2.0 中的 bird 是仿真飞鸟，不得写成真实自然鸟外部验证；官方还提示部分 L 波段记录
可能因采集位置未改变而不变，拆分前必须按目标、角度、记录和潜在重复内容建立分组清单。

DroneRFc-MM 下载子集含 9 个同日 recording、30,717 个 PCD 帧和 717 个由 recording
切出的约 5 秒窗口，覆盖 6 种无人机。只读检查器对全部 639,527 个点验证了固定 15 列、
finite numeric、`POINTS`/数据行一致以及行内 sec/nsec 与文件名一致；ZIP 成员物理顺序是
乱序，只有按连续 `frame_id` 排序后时间才严格递增。PCD 字段含位置、距离、方位、俯仰、
Doppler、功率、SNR 和时间戳，但不是 ADC/复数 IQ。

9 份 GT 的 35,959 行时间均为非递减，但每份都有重复时间戳。8 个 recording 的 radar/GT
时间范围重叠，最近 GT 时间误差 P95 约 0.05--0.073 s；B1 雷达范围为本地时间
15:13:17.922--15:19:22.629，而同名 GT 为 15:27:15.240--15:35:24.054，零重叠。README
所说约 0.3 s 软同步不能覆盖这个约 8 分钟错位，故整体门为
`PASS_SCHEMA_BLOCKED_TIMESTAMP_ALIGNMENT`，B1 必须等待更正 GT 或可归因偏移说明。A/E/G 的
`-2` 段还必须与 base group 同 split。帧/窗口随机拆分、unseen-model、鸟/背景、微多普勒 IQ
或部署结论均禁止。

UAV 群数据的 5 个 MAT 是 `Exp1`、`Exp2_1/2`、`Exp3_1/2`，后四个文件是两个连续实验的
拆分，而不是 5 个独立试验。`Measurement` 为 2×K cell：第一行的非空项是 3×N 有限 XYZ
坐标，第二行是会跨相邻列重复的屏号。全包 2,281 列、171,309 点只形成 275 个连续屏；
没有目标 ID、类别、幅度/RCS、速度、Doppler、IQ 或极化字段，列和点均不能当独立样本。
NEXRAD 的射线、仰角层、range gate 和 patch 也不能随机拆分；
它们至少应随整个体扫进入同一 split，相邻体扫还要按完整天气/迁飞事件隔离。两项新增下载
合计仅 `4,392,132` bytes，目的都是验证读取和分组接口，而不是扩大训练样本数。

## 9. 候选数据集决策

| 候选 | 决策 | 原因 |
|---|---|---|
| LSS-DAUR-1.0 | A1，只读审计完成但训练阻塞 | 77 个逻辑观测仅 76 个唯一 TD/TR 内容对，保守连通为 39 个候选组；schema、配对与 canonical/backup 等价性通过，但 session 未确认、严格时间失败、512/1024 混宽且类别混杂，暂只允许归一化 bin/轨迹方法设计 |
| LSS-HSR-L | A2，V2 已下载待 schema/group 审计 | 多场景 UAV/鸟/旋转地物/汽车的 Doppler waterfall 与轨迹；已确认期刊包与 V2 不等价，以 V2 实现只读 loader，期刊包仅留作版本史 |
| LSS-FMCWR-2.0 | A3，已下载 | K/L 双频、多角度和原始回波，适合微多普勒/跨频/跨角度；需严格区分仿真鸟 |
| DroneRFc-MM V1 | A4，schema 通过但同步总门阻塞 | 全量 PCD 结构/数值/时间戳通过；B1 radar/GT 零重叠，禁止用于监督对齐。其余 8 条可在另行预登记后继续同步/轨迹接口研究；最低按 6 个 base family 分组 |
| 低空 UAV 群目标 V1 | S1，小包 schema 已审计 | 约 4 MB、Ku 波段 XYZ 量测，适合按屏聚合的群目标质心/外形、航迹关联和运动模式接口 smoke；只有 3 个物理实验且没有目标 ID、IQ、Doppler 或负类，不能做识别性能 |
| NOAA NEXRAD Level II | S2，单体扫已下载 | 实际含 Z、V、SW、ZDR、PhiDP、RhoHV，可验证双极化 loader/公式；无目标标签且体制不同，不并入主训练集 |
| Radar Signature Dataset 0.1 | B，说明已下载，主包暂缓 | 与气球材料最接近，但对象是室内手动运动的数字形铝箔气球、输出为 range profile；42.4 GB 成本高且不能回答户外载荷问题 |
| S 波段简单野外 UAV `sciencedb.18323` | B，高价值但暂缓 | 真实 S 波段 LFM、5 型 UAV，可形成 HRRP/RD/时频支线；23.46 GB 且只有 13 个原始采集，先做存储和 acquisition-grouped loader 预案 |
| MathWorks/St Andrews 三频 UAV/真鸟 | B，高价值但暂缓 | 24/94/207 GHz 实测 UAV、真实鸟与 clutter/noise 很有价值；总计 30.36 GB，0.25 s 重叠窗口必须重建物理实验级 split 后才可下载建模 |
| LSS-Ku-1.0 | 暂缓 | 14.88 GiB，当前不能解除 Tian 预处理/对齐门禁；只有学长确认其能补明确同域或地杂波缺口时再下载 |
| LSS-PR-1.0 | B，暂缓 | 只有 9 个 drone，海面域占主导，且文件树与元数据的 bird 描述不完全一致；DAUR 已先覆盖 immediate need |
| LSS-FMCWR-1.0 | B，暂缓主体 | 与 2.0 高度重叠且约 1.03 GiB；先用已下载说明设计透明 Python 处理链 |
| Radar-acoustic drone detection system | 本轮拒绝 | 实际文件树主要是 9 个 AWGN SNR 派生 ZIP 和 19 个 JPG；缺 README、雷达参数、session 和鸟/天气对照，跨 SNR 随机拆分会泄漏 |
| HEAD-1.0 混合极化 SAR | 拒绝 | 约 410 GB，属于星载地表 SLC，不是低空目标体制，不能替代 H/V UAV/空飘球数据 |

机器可读真值位于：

- `data/metadata/external_public_datasets_v1.csv`：数据集版本、DOI、许可、用途、决策和声明边界；
- `data/metadata/external_public_artifacts_v1.csv`：实际下载对象及失败访问尝试、大小、SHA256、官方校验和与本地存储键。

两表不保存临时签名 URL、访问凭据或绝对个人路径。被 `data/raw/` 忽略的本地目录另保留
ScienceDB 原始 URL 文件树和逐文件 SHA256 清单，供完整性复核，不进入 Git 或分享包。
DAUR 清单可在官方 V3 原件目录内按以下固定规则重建，再核对清单文件的 SHA256：

```bash
find . -type f ! -name 'OFFICIAL_DOWNLOAD_URLS_V3.txt' \
  ! -name 'SHA256SUMS_V3.txt' -print0 \
  | LC_ALL=C sort -z | xargs -0 sha256sum > SHA256SUMS_V3.txt
sha256sum SHA256SUMS_V3.txt
```

预期为 `5febc59a29c42fb7dd8b001afa73fe913767f00c2c060a21a5d95073c2ee1745`。

DroneRFc-MM 雷达子集在其本地原件目录按同样的“排序逐文件 SHA256”规则生成 28 行清单，
清单自身的预期 SHA256 为
`6b0c2ed1a075aa9164a516af001b630a9f775fddc9f399223c1aeeb6e7047b2b`。该值只证明本地
选择性子集身份，不代表完整 75.6 GB/113 文件发布已下载。

DroneRFc-MM 只读复核入口：

```bash
python scripts/audit_dronerfc_mm_v1.py --overwrite
```

预期状态为 `PASS_SCHEMA_BLOCKED_TIMESTAMP_ALIGNMENT`，`blocked_recordings=["B1"]`。该命令
不解压或修改源 ZIP/CSV，不生成逐帧输出，也不训练模型。
