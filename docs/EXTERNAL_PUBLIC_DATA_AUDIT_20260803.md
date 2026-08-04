# 外部公开雷达数据核验与使用计划

版本：2026-08-04

## 1. 来源核验

| 用户提供入口 | 实际对象 | 本轮状态 | 决策 |
|---|---|---|---|
| 本地《低空目标多频段雷达识别特性数据集》PDF | LAT-MRICD-1.0，DOI `10.12466/xhcl.2026.06.001` | 官方 ScienceDB ZIP 已下载并校验 | 纳入 HRRP 和归一化微动算法预研 |
| `10.12466/xhcl.2025.05.003` | 地杂波背景下雷达低慢小无人机探测数据集 LSS-Ku-1.0 | 官方下载入口已定位，ZIP 为 15,985,618,112 bytes，未下载 | 保留为地杂波检测/Tian 同域性候选，先向学长确认价值 |
| `radars.ac.cn` | 《雷达学报》数据栏目；正确路径含 `/cn/`，正式文件由 ScienceDB 托管 | 已核对 LSS-DAUR、LSS-HSR-L、LSS-FMCWR-1.0/2.0 的 DOI、许可和文件树 | 下载 DAUR V3、HSR-L 官方包、FMCWR-2.0 V4 及两版 FMCWR 说明；原始数据不进 Git/分享包 |
| ScienceDB LSS-DAUR-1.0 | 数据 DOI `10.57760/sciencedb.radars.00076`，V3 | 314/314 文件、148,763,512 bytes 已下载并生成逐文件 SHA256 | 优先做真实轨迹分组的时域/微多普勒与 UAV-vs-bird 审计 |
| ScienceDB LSS-FMCWR-2.0 | 数据 DOI `10.57760/sciencedb.radars.00054`，V4 | 8/8 文件、1,013,535,036 bytes 已下载；6 个 RAR 的官方 MD5 全通过 | 优先做 K/L、角度和微多普勒 schema 审计；鸟类包只作仿真数据 |

LAT-MRICD 的论文页面标注文章采用 CC BY 3.0；文章许可不自动等同于原始 ZIP 可再次分发，
因此仍按 `NOASSERTION` 保守处理数据文件。ScienceDB 的 DAUR/FMCWR V3/V4 元数据明确标注
`CC BY-NC-ND 4.0`，HSR-L 的 ScienceDB V2 标注 `CC BY-NC 4.0`。当前取得的是大小不同的
HSR 期刊 bundle，其与 V2 的内容等价性和 bundle 自身条款尚未建立，不能直接套用 V2 状态。
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

### 5.2 已预登记、尚未运行的跨频段迁移

`docs/LAT_MRICD_CROSS_BAND_TRANSFER_PROTOCOL_V1.md` 与
`configs/lat_mricd_cross_band_transfer_v1.json` 已冻结。正式主分析只把 Narrow-X 作为源，
分别把 S 和 Ku 作为一次性 locked target；任务固定为共同 UAV 型号与 weather 二分类，
bird 不进入该任务。HRRP 和其余方向只作预先声明的压力分析。

截至预登记提交前，S/Ku target 性能尚未运行或查看，正式结果目录、外部消费记录和最终证据
目录均不存在。正式运行只允许一次；失败或中断也视为 target 已消费。dummy、batch-balanced
逻辑回归和固定随机森林均预先保留，不允许用 target 拟合缩放、调参、选模或扩特征。

继续门要求两个 locked target 同时满足：逻辑回归的 target batch-class macro accuracy
严格大于 0.60；UAV/weather 两类 target batch 等权召回均严格大于 0.50；以及
LR-minus-dummy 配对 95% CI 下界严格大于 0。任一项失败即冻结负结果，并停止基于已消费
target 的 CNN、域适配和结果驱动调参。

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
2. D17-XBAND 的类别交集、source/target band、分组指标和失败标准已经预登记；下一步只执行
   一次提交绑定的密封运行，不在运行前查看 S/Ku 性能；
3. 跨频段只比较各频段独立训练/评价后的迁移表现，不做样本级拼接，不声称同一事件同步；
4. 若迁移结果失效，记录负结果并停止扩模型；只有门禁稳定后才评估轻量一维 CNN。

LSS-Ku-1.0 仅在学长确认它能补 Tian 同域性或地杂波测试缺口后再下载，避免为 14.88 GiB
数据付出传输、存储和审计成本却不能解除当前门禁。

## 8. 2026-08-04 新增公开数据下载回执

| 数据集 | 本地状态 | 完整性结论 | 下一门禁 |
|---|---|---|---|
| LSS-DAUR-1.0 V3 | 314 个官方文件完整下载 | 文件数 `314`、总字节 `148763512` 与发布元数据一致；排序逐文件 SHA256 清单的 SHA256 为 `5febc59a29c42fb7dd8b001afa73fe913767f00c2c060a21a5d95073c2ee1745` | 审计全部 MAT schema、TD/TR 1:1 配对、时间单调性、有限值和按原始 track 分组；未通过前不训练 |
| LSS-HSR-L 官方期刊包 | ZIP 已下载 | `209569478` bytes；SHA256 `22112d4225636c5626845a9f0640abbf4503cc70763b592a609287760ab5f4a4`；ZIP 完整性通过 | 实现只读 loader，禁止直接运行会移动原件的官方 `dataset.py`；核对场景/轨迹分组 |
| LSS-FMCWR-2.0 V4 | 8 个官方文件完整下载 | 总字节 `1013535036` 与发布元数据一致；6 个 RAR 的官方 MD5 全通过，另生成 SHA256 | 先配置可审计的 RAR 5 解包工具，再按 MAT 实查官方“90 组”与分项合计 86 的矛盾 |
| LSS-FMCWR-1.0 | 仅下载 1.0 使用说明 PDF | 当前没有原始 RAR | 等 FMCWR-2.0 审计后，仅在能补明确 UAV 型号/双频缺口时下载主体 |
| St Andrews 微多普勒仿真包 | 未下载 | DOI 页面与 CC BY 可核验，但官方文件端两次返回 HTTP 403；没有保留部分文件，也未绕过访问限制 | 只在官方端恢复或数据持有人授权后用于 smoke，不作为真实外部性能证据 |

DAUR 的 canonical 77 TD + 77 TR 与 `backup_original` 中的 154 个对应文件全部 bytewise
不同，因此“backup”不能按重复文件删除。规范建模输入暂定 canonical TD/TR；backup 只保留为
处理链对照，最终选择必须由 schema 和说明书审计决定。

FMCWR-2.0 中的 bird 是仿真飞鸟，不得写成真实自然鸟外部验证；官方还提示部分 L 波段记录
可能因采集位置未改变而不变，拆分前必须按目标、角度、记录和潜在重复内容建立分组清单。

## 9. 候选数据集决策

| 候选 | 决策 | 原因 |
|---|---|---|
| LSS-DAUR-1.0 | A1，已下载待审计 | 小体量真实轨迹、512 点 Doppler 序列和官方配套 TR；先审计 1:1 配对、时间轴与 PRF，再决定能否开放物理微多普勒和轨迹融合 |
| LSS-HSR-L | A2，期刊包已下载待等价审计 | 多场景 UAV/鸟/旋转地物/汽车的 Doppler waterfall 与轨迹；先核对期刊包与 ScienceDB V2 差异并实现只读 loader |
| LSS-FMCWR-2.0 | A3，已下载 | K/L 双频、多角度和原始回波，适合微多普勒/跨频/跨角度；需严格区分仿真鸟 |
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
