# 复杂气象、场地杂波与雷达干扰文献包

版本：2026-09-18  
用途：环境方向初步调研、外场实验设计和后续抗干扰方法选型  
全文目录：`../../paper/references/环境与干扰_公开/`（该目录被 Git 忽略）

## 1. 使用边界

本目录是“阅读和方法参考包”，不是当前项目的新实验结果。文献中的雷达波段、体制、目标、天气、场地和指标都可能与本项目不同：

- 不能把论文中的检测率、虚警率或抗干扰增益直接写成项目结果；
- 不能把汽车 FMCW 雷达方法直接假设适用于当前 H/V 脉冲多普勒数据；
- 不能把气象雷达的 PPI/RHI/体扫产品假设为合作设备一定具备；
- 不能把场地背景图在单一场地上的效果写成跨场地泛化；
- 真实外场结论仍需通过项目的 `capture`、`causal` 和 `locked_evaluation` 数据合同。

全文来源为出版社、机构仓储、arXiv 或开放获取平台；许可证和再分发边界并不完全相同。PDF 仅保存在本地忽略目录，项目分享包只应携带题录、官方链接、摘要和本地 SHA-256。

## 2. 建议先读的 8 篇

| 优先级 | 文件 | 主题 | 先回答什么 |
|---:|---|---|---|
| 1 | `CloudRadar_ClutterDiscrimination_2021.pdf` | 云雷达低层云/杂波区分 | 如何把低层环境结构和杂波区分，哪些变量属于观测条件而非目标类别 |
| 2 | `DualPolarization_DualScan_GroundClutter_2016.pdf` | 双极化、双扫描地杂波识别 | 极化和多次扫描如何辅助杂波判定，哪些要求依赖气象雷达体制 |
| 3 | `DigitalEarthSurfaceMaps_RadarGroundClutter_2022.pdf` | 地表地图与雷达地杂波仿真 | 如何把地形/地表先验转为背景仿真，如何避免把地图当作现场真值 |
| 4 | `LearningStrategies_RadarClutterClassification_2020.pdf` | 雷达杂波分类学习策略 | 杂波分类任务如何划分数据、构造负样本和防止背景捷径 |
| 5 | `RadarSlidingWindowDetectors_2017.pdf` | 滑窗检测与 CFAR 基础 | 小样本和非均匀背景下，统计检测基线应如何建立 |
| 6 | `RFI_Mitigation_SAR_Review_2019.pdf` | SAR 射频干扰综述 | 干扰来源、观测表现、传统抑制和学习方法的分类框架 |
| 7 | `RadarInterferenceMitigation_AutomatedDriving_2020.pdf` | 雷达干扰主动/被动抑制 | 干扰压力测试如何与真实接收机和系统约束连接 |
| 8 | `UrbanStreetGeometry_RadarDetectionProbability_2023.pdf` | 城市场景几何对检测概率的影响 | 场地几何如何改变可见性、遮挡和检测概率，如何设计场地对照 |

## 3. 天气、地杂波与场地背景

### 3.1 已下载文件

| 文件 | 题录/来源 | 与本项目的用途 | 主要限制 | SHA-256 |
|---|---|---|---|---|
| `CloudRadar_ClutterDiscrimination_2021.pdf` | A robust low-level cloud and clutter discrimination method for ground-based millimeter-wavelength cloud radar, *Atmospheric Measurement Techniques* 14, 1743 (2021). DOI: `10.5194/amt-14-1743-2021`；[官方 PDF](https://amt.copernicus.org/articles/14/1743/2021/amt-14-1743-2021.pdf) | 学习云/杂波判别、低层环境分层和质量控制 | 云雷达与本项目雷达体制不同；不能直接迁移阈值 | `e016fdd03e2c0ccca7b1ae44fc95b8191698996322dd9bf5d7e022a3143463b7` |
| `DualPolarization_DualScan_GroundClutter_2016.pdf` | Detection of Ground Clutter from Weather Radar Using a Dual-Polarization and Dual-Scan Method, *Atmosphere* 7, 83 (2016). DOI: `10.3390/atmos7060083`；[DOI](https://doi.org/10.3390/atmos7060083) | 研究双极化和扫描差异如何区分地杂波 | 依赖天气雷达的双扫描/极化变量；当前 H/V 物理含义仍待确认 | `e2cfeeb6c4a850fcdb4e0a8fd87c93416066d6d72ea96504262d32f3d98d50c8` |
| `DigitalEarthSurfaceMaps_RadarGroundClutter_2022.pdf` | Digital Earth surface maps for radar ground clutter simulation, DOI: `10.23919/JSEE.2022.000035`；[IEEE PDF](https://ieeexplore.ieee.org/ielx7/5971804/9775056/09775057.pdf) | 参考地表/地形先验与地杂波仿真接口 | 地图和仿真不是现场测量；需要雷达几何、天线和传播参数 | `02e9a69c51f1d491f32c99963ceaacc556eb2d61e2a6c1595b89ccc02830f28d` |
| `WindFarmClutter_ClutterMap_KSVD_2019.pdf` | Wind farm clutter suppression for air surveillance radar based on a combined method of clutter map and K-SVD algorithm, DOI: `10.1049/iet-rsn.2019.0399`；[DOI](https://doi.org/10.1049/iet-rsn.2019.0399) | 固定强反射体、背景图和字典学习的思路 | 机构 PDF 受站点保护，当前只登记题录，未成功下载；风场/空情雷达体制不同 | 未下载 |
| `TerrainAdaptive_ClutterMap_WeatherRadar_2014.pdf` | Performance Analysis on Terrain-Adaptive Clutter Map Algorithm for Ground Clutter Rejection of Weather Radar, DOI: `10.5515/kjkiees.2014.25.12.1292`；[DOI](https://doi.org/10.5515/kjkiees.2014.25.12.1292) | 地形自适应背景图和固定场地先验 | 当前网络下载失败；天气雷达地杂波模型不能直接代替本项目场地档案 | 未下载 |
| `GroundClutter_PolarimetricParameters_2008.pdf` | Influence of Ground Clutter Contamination on Polarimetric Radar Parameters, DOI: `10.1175/2008jtecha1092.1`；[DOI](https://doi.org/10.1175/2008jtecha1092.1) | 认识地杂波如何污染极化量，提醒标定和背景污染风险 | AMS PDF 站点要求验证，当前只登记题录；不能据此解释当前 H/V 相对量 | 未下载 |

### 3.2 这组文献对项目的直接启发

先做“场地背景档案”和“环境分层”，再考虑复杂场地模型。最低实验应包括：

1. 同一场地多次纯背景 session，记录固定峰、近零多普勒能量和场地照片；
2. 同日期、同配置的目标与背景配对，避免场地成为类别捷径；
3. 跨日期或跨场地留出数据，检查背景图是否只记住原场地；
4. 先用统计/CFAR 或背景校准作基线，再评估学习模型；
5. 若使用地图或遥感信息，明确它是先验/仿真输入，不是回波真值。

## 4. 统计杂波、检测与背景分类

| 文件 | 题录/来源 | 与本项目的用途 | 主要限制 | SHA-256 |
|---|---|---|---|---|
| `LearningStrategies_RadarClutterClassification_2020.pdf` | Learning Strategies for Radar Clutter Classification, arXiv:`2004.08277`；[arXiv](https://arxiv.org/abs/2004.08277) | 杂波分类、训练策略、背景分组和负样本设计 | arXiv 方法与数据可能是特定雷达场景；不能替代同场地外测 | `d5000ba338e27ba7be3f4616f24a17fe89070323cea075da3bb38c3ae8c22d7d` |
| `RadarSlidingWindowDetectors_2017.pdf` | An Introduction to Radar Sliding Window Detectors, arXiv:`1709.09786`；[arXiv](https://arxiv.org/abs/1709.09786) | 了解滑窗检测、参考单元、保护单元和虚警控制 | 统计假设和窗口尺度需按当前 RD/扫描结构重设 | `d748ca5512fd9bf60c0465364e40aca78e44a6b39143291d770f9490dd54e136` |
| `RadarClutter_RandomProcessModel_2024.pdf` | A Random Process Model Useful for Describing Radar Clutter, arXiv:`2411.10263`；[arXiv](https://arxiv.org/abs/2411.10263) | 建立随机杂波、相关性和合成压力测试的概念 | 统计模型自洽不等于当前场地物理真实；需真实背景验证 | `2d6001ace7b7a88b6565e19ff4309423a3fca4a80b81ec795a77c02974dd232f` |
| `RadarClutter_Covariance_Estimation_2023.pdf` | Radar Clutter Covariance Estimation: A Nonlinear Spectral Shrinkage Approach, arXiv:`2302.02045`；[arXiv](https://arxiv.org/abs/2302.02045) | 参考背景协方差、相关结构和小样本估计 | 方法偏阵列/统计检测；当前项目先做二维 RD 背景统计即可 | `c4dcf1499706a8e7cf88df304661bf7d20d0d584058e0f6b8d5a5c7ba88834e2` |
| `Cognitive_RadarClutter_Classification_TargetDetection_2022.pdf` | Innovative Cognitive Approaches for Joint Radar Clutter Classification and Multiple Target Detection in Heterogeneous Environments, arXiv:`2207.03781`；[arXiv](https://arxiv.org/abs/2207.03781) | 参考异质环境中的“先识别背景、再检测目标”思路 | 认知雷达闭环依赖波形/阵列控制，当前项目未确认具备 | `c2e59f667631d1eb38bb86362a28477121a661ff99c658a969ab614efc5e6473` |

## 5. 雷达干扰与抗干扰

| 文件 | 题录/来源 | 与本项目的用途 | 主要限制 | SHA-256 |
|---|---|---|---|---|
| `RFI_Mitigation_SAR_Review_2019.pdf` | Mitigation of Radio Frequency Interference in Synthetic Aperture Radar Data: Current Status and Future Trends, *Remote Sensing* 11, 2438 (2019). DOI: `10.3390/rs11202438`；[DOI](https://doi.org/10.3390/rs11202438) | 干扰来源、检测、抑制、数据域和评价方式总览 | SAR 干扰与当前脉冲/双通道数据不同，适合做分类框架 | `4fed1411763c3ade4d3aaa06e13b3048460c069533a7d8b625569f841b99c7fc` |
| `RadarInterferenceMitigation_AutomatedDriving_2020.pdf` | Radar Interference Mitigation for Automated Driving: Exploring Proactive Strategies, *IEEE Signal Processing Magazine* (2020). DOI: `10.1109/MSP.2020.2969319`；[机构 PDF](https://research.chalmers.se/publication/521636/file/521636_Fulltext.pdf) | 了解干扰预防、协同和接收端抑制的系统层取舍 | 汽车 FMCW 多雷达共存场景；不能直接当作本项目现场干扰证据 | `e83eee7dd094f7ba7e5024bcf9a7a3aca1b157ec344fb438b689ac6ef4eade56` |
| `FMCW_Interference_STFT_2018.pdf` | An Interference Mitigation Technique for FMCW Radar Using Beat-Frequencies Interpolation in the STFT Domain, DOI: `10.1109/TMTT.2018.2881154`；[TU Delft PDF](https://repository.tudelft.nl/file/File_0d7972bf-a84e-4af6-a07d-c688bf5d10c8) | STFT 域干扰识别/插值和时频压力测试 | 需要 FMCW beat signal；当前 H/V 脉冲数据不能直接套用 | `d1b81e1d376555fd1cb0d7d5100a30e78f9a141f373538622e6a8596471ac01b` |
| `AdaptiveNoiseCanceller_AutomotiveRadar_2019.pdf` | Automotive Radar Interference Mitigation Using Adaptive Noise Canceller, DOI: `10.1109/TVT.2019.2901493`；[arXiv](https://arxiv.org/abs/1911.06372) | 参考自适应噪声抵消和干扰检测前端 | 汽车 FMCW 体制，需重新定义参考通道和干扰观测 | `5c412c244f4557280962070afe2aa1e21ad235e24c4596850ad7743b6322f97c` |
| `SynchronousAsynchronous_RadarInterference_2018.pdf` | Synchronous and Asynchronous Radar Interference Mitigation, *IEEE Access* (2018). DOI: `10.1109/ACCESS.2018.2884637`；[IEEE PDF](https://ieeexplore.ieee.org/ielx7/6287639/8600701/08555539.pdf) | 比较同步/异步干扰形态和抑制思路 | 多雷达互扰设定与合作场地干扰源未必相同 | `7464c487fc91110e218ae4ea7cb998e07416300a766e8804f61bcff90ead94b1` |
| `Autoregressive_SignalReconstruction_RadarInterference_2020.pdf` | Autoregressive Model-Based Signal Reconstruction for Automotive Radar Interference Mitigation, DOI: `10.1109/JSEN.2020.3042061`；[IEEE PDF](https://ieeexplore.ieee.org/ielx7/7361/9347831/09277579.pdf) | 参考模型化重建和信号恢复 | 对采样体制、干扰稀疏性和信号结构有额外假设 | `d40e715e735dedc8f8321c3d35f10e669b42967b8193f84144691f1be0ecc061` |
| `SparseLowRankHankel_FMCW_Interference_2022.pdf` | Interference Mitigation for FMCW Radar With Sparse and Low-Rank Hankel Matrix Decomposition, DOI: `10.1109/TSP.2022.3147863`；[arXiv](https://arxiv.org/abs/2106.06748) | 参考低秩/稀疏分解类干扰压力测试 | 计算量和 FMCW 假设可能不适合本地部署；先做方法阅读 | `54ef2d6f3076beaaee69f6343a888971d1b9d49b2f1d60b5b6e29954571b3fe1` |
| `DeepInterferenceMitigation_RealWorldFMCW_2020.pdf` | Deep Interference Mitigation and Denoising of Real-World FMCW Radar Signals, arXiv:`2012.02529`；[arXiv](https://arxiv.org/abs/2012.02529) | 观察真实干扰数据下深度去噪的训练和评价方式 | FMCW、汽车数据和干扰标签与本项目不同 | `ae22142949cb573d39c06a175b725b2dd2526eed8ac6a4fbe05bbcbaa569378b` |
| `FullyCNN_AutomotiveRadarInterference_2020.pdf` | Fully Convolutional Neural Networks for Automotive Radar Interference Mitigation, arXiv:`2007.11102`；[arXiv](https://arxiv.org/abs/2007.11102) | 参考全卷积干扰抑制网络和端到端评价 | 不是当前目标检测任务；不能只比较去噪图像质量 | `3a97d077f4ae8fcf35f8dc7df7ece0670bfe14aba6f74e83b6fec2d09a2b20d7` |
| `SparseChirplet_AutomotiveFMCWInterference_2021.pdf` | Sparse Reconstruction of Chirplets for Automotive FMCW Radar Interference Mitigation, arXiv:`2106.05594`；[arXiv](https://arxiv.org/abs/2106.05594) | 参考时频稀疏表示和可解释的干扰形态 | 需要 FMCW 时频模型，不能直接等同于天气杂波 | `410ee2478669f8d7bbfbe0e113ed48c8d393ffa8be41b6218dce30de770dd515` |
| `RIMformer_FMCW_Interference_2024.pdf` | RIMformer: An End-to-End Transformer for FMCW Radar Interference Mitigation, arXiv:`2407.11459`；[arXiv](https://arxiv.org/abs/2407.11459) | 了解新近 Transformer 干扰抑制路线和复杂度讨论 | 当前项目不应在真实干扰数据和基线未建立前直接上 Transformer | `a1da84a2461630244758c46a9fb5b61c6424ef408fc84e0ce3b924a1aa1547bc` |

## 6. 城市/固定场景影响

| 文件 | 题录/来源 | 与本项目的用途 | SHA-256 |
|---|---|---|---|
| `UrbanStreetGeometry_RadarDetectionProbability_2023.pdf` | Impact of Urban Street Geometry on the Detection Probability of Automotive Radars, arXiv:`2312.05623`；[arXiv](https://arxiv.org/abs/2312.05623) | 说明建筑几何、遮挡和反射如何改变检测概率；帮助设计开阔地/建筑区对照 | `2d2d946af48ba403aaff4cc0197b7bd0fbe0a9436a90478ecafc291de7c60c43` |

它适合帮助你建立“场地因素不仅是一个 site_id”的意识：场地需要拆成遮挡、强反射体、朝向、可见扇区和背景运动等可解释变量。汽车雷达的城市街道几何不能直接代替合作场地测绘。

## 7. 与当前项目的对应关系

| 当前任务 | 应优先参考 | 近期产物 |
|---|---|---|
| D03：日期/场地是否混杂 | Clutter classification、urban geometry、terrain/clutter map | 环境--目标--背景分组矩阵 |
| D08：背景校准是否因果 | Sliding-window detectors、clutter random process、interference mitigation | 在线可用变量清单和 past-only 边界 |
| D09：虚警抑制能否保护目标 | Ground clutter、clutter map、dual-pol clutter | 目标保护/Pfa/Joint Pd 评价表 |
| D11：时频/干扰特征 | FMCW STFT、deep denoising、chirplet | 干扰目录和时频压力测试方案 |
| D12：H/V 与极化可信度 | Dual-polarization clutter、极化污染论文 | 通道有效性和标定问题单 |
| D13：空飘球环境条件 | 场地几何、天气/云雷达、目标运动记录 | 空飘球目标/背景/风况 Pilot 矩阵 |

## 8. 建议阅读顺序与记录模板

每篇文献只记以下 8 项，避免变成泛读：

1. 雷达体制、波段和数据级别；
2. 目标和背景/杂波定义；
3. 环境变量和采样方式；
4. 是否有同条件目标/背景；
5. 数据划分的独立单位；
6. 使用的基线和主指标；
7. 哪个方法可以迁移到当前项目；
8. 哪个假设在当前设备上尚未满足。

推荐顺序：

1. `CloudRadar_ClutterDiscrimination_2021.pdf`；
2. `DualPolarization_DualScan_GroundClutter_2016.pdf`；
3. `DigitalEarthSurfaceMaps_RadarGroundClutter_2022.pdf`；
4. `LearningStrategies_RadarClutterClassification_2020.pdf`；
5. `RadarSlidingWindowDetectors_2017.pdf`；
6. `RFI_Mitigation_SAR_Review_2019.pdf`；
7. `RadarInterferenceMitigation_AutomatedDriving_2020.pdf`；
8. 再按项目设备能力选择统计模型、场地图或深度干扰抑制论文。

## 9. 下载重试记录

2026-09-18 再次尝试下载以下三篇：

| 文献 | 官方站点返回 | 当前处理 |
|---|---|---|
| `GroundClutter_PolarimetricParameters_2008.pdf` | AMS 页面返回 WAF/403 | 保留 DOI 和官方链接，未使用镜像 |
| `WindFarmClutter_ClutterMap_KSVD_2019.pdf` | Wiley/IET 页面返回 403 | 保留 DOI 和机构入口，未使用镜像 |
| `TerrainAdaptive_ClutterMap_WeatherRadar_2014.pdf` | KoreaScience 连接被重置 | 保留 DOI 和官方文章入口，未使用镜像 |

这三篇不影响当前阅读包的主体使用；等网络或机构访问条件具备后，可以用登记文件中的官方链接补下。当前目录没有把错误页或空文件当作论文保存。

## 10. 文件完整性

下载目录中的 20 个 PDF 均已检查为 PDF 文件并计算 SHA-256。未成功下载的 3 篇 AMS/IET/KoreaScience 论文保留在题录清单中，没有用来源不明的镜像替代。

本目录不进入 Git；本登记文件与项目现有 `RECOMMENDED_PAPERS_20260805.md` 一样，只维护题录、官方链接、用途、边界和哈希。
