# 推荐论文与下载登记

版本：2026-08-05
用途：给项目成员一份经过来源、主题和许可筛选的补充阅读清单。
范围：无人机雷达检测、微多普勒、极化、有限数据检测和网络感受野。

## 1. 使用规则

这不是“把网上搜到的论文都保存下来”的目录，而是一个小型、可审计的阅读集。每篇论文都回答三个问题：

1. 它解决的任务与本项目哪一层相同？
2. 它能为当前代码或未来空飘球数据提供什么可迁移方法？
3. 哪些实验条件不同，不能直接拿来作本项目证据？

PDF 下载到本机的目录为 `paper/references/推荐论文_公开/`。该目录被 `.gitignore` 忽略，原因是论文全文的再分发许可并不统一；Git 中应提交本文件的题录、来源、哈希和阅读结论，而不是自动提交全文。

访问日期：2026-08-05。哈希算法：SHA-256。

## 2. 已下载论文登记

| 编号 | 文件 | 题录与 DOI | 合法来源 | 许可/分享边界 | SHA-256（完整值） |
|---:|---|---|---|---|---|
| 1 | `Luo_2016_Understanding_the_Effective_Receptive_Field.pdf` | W. Luo, Y. Li, R. Urtasun, R. Zemel, “Understanding the Effective Receptive Field in Deep Convolutional Neural Networks,” NeurIPS 2016；arXiv:1701.04128（2017 版本） | [arXiv abs](https://arxiv.org/abs/1701.04128v2) / [PDF](https://arxiv.org/pdf/1701.04128v2) | arXiv 公开阅读；页面未作为本项目的 CC 再分发授权。分享包保留题录和链接。 | `c00ce41d239b1a2fe098084ae4367a4dea39149d1e75c4ad52e9e84865f8ec8b` |
| 2 | `Rahman_Robertson_2018_Drone_Bird_MicroDoppler.pdf` | S. Rahman, D. A. Robertson, “Radar micro-Doppler signatures of drones and birds at K-band and W-band,” *Scientific Reports*, 8, 17396 (2018). DOI: `10.1038/s41598-018-35880-9` | [Nature 官方 PDF](https://www.nature.com/articles/s41598-018-35880-9.pdf) | Gold OA，文章页面标注 CC BY；引用作者和 DOI。 | `8eb49db322f4ca69450136e8198ac6b99e5d7a43e92a9c63aa9e35d7fd494e2d` |
| 3 | `Torvik_Olsen_Griffiths_2016_Birds_UAVs_Radar_Polarimetry_AAM.pdf` | B. Torvik, K. E. Olsen, H. Griffiths, “Classification of Birds and UAVs Based on Radar Polarimetry,” *IEEE GRSL*, 13(9), 1305–1309 (2016). DOI: `10.1109/LGRS.2016.2582538` | [UCL Discovery 记录](https://discovery.ucl.ac.uk/id/eprint/1533719/) / [作者接受稿 PDF](https://discovery.ucl.ac.uk/1533719/1/Griffiths-H_classification%20of%20birds%20and%20UAVs_aam.pdf) | Green OA 作者接受稿；仓储未给出 CC 授权，不能把它重新包装成 CC-BY 全文。 | `838683a6b720e7f7c5363a106cdce60b07396e37b41c822046ead6731d8809a5` |
| 4 | `Moore_Robertson_Rahman_2023_Drone_MicroDoppler_Simulation.pdf` | M. Moore, D. A. Robertson, S. Rahman, “A new simulation methodology for generating accurate drone micro-Doppler with experimental validation,” *IET Radar, Sonar & Navigation*, 18(3), 477–492 (issue 2024; online first 2023). DOI: `10.1049/rsn2.12494` | [DOI/出版社入口](https://doi.org/10.1049/rsn2.12494) / [St Andrews 作者库 PDF](https://research-repository.st-andrews.ac.uk/bitstream/10023/28555/1/Moore_2023_IETRSN_A_new_simulation_CCBY.pdf) | Gold OA，作者库文件标注 CC BY；保留署名和来源。 | `b44fdee21bb8685331e5fc6aaa234a452001ceedb9134740643ff3e44edf8ffb` |
| 5 | `Wang_2021_Radar_Target_Detection_Limited_Data_IEEE_Access.pdf` | F. Wang, P. Wang, X. Zhang, H. Li, B. Himed, “An Overview of Parametric Modeling and Methods for Radar Target Detection With Limited Data,” *IEEE Access* (2021). DOI: `10.1109/ACCESS.2021.3074063` | [IEEE 官方 OA PDF](https://ieeexplore.ieee.org/ielx7/6287639/9312710/09406810.pdf?arnumber=9406810) | OpenAlex/Semantic Scholar 标为 Gold OA、CC BY-NC-ND；本地只做个人研究，不改编或重新发布全文。 | `3bddac2a7d3fe9ac2241a11a1cb80cff45959a06f82e0dd65d55df01bacfcbad` |
| 6 | `Sethuraman_2022_UAV_Payload_Classification_Polarimetric_Radar_VoR.pdf` | H. V. Sethuraman, A. Yarovoy, F. Fioranelli, “Classification of Unmanned Aerial Vehicles (UAVs) Carrying Payloads with Polarimetric Radar,” *18th European Radar Conference*, pp. 365–368 (2022). DOI: `10.23919/EuRAD50154.2022.9784557` | [TU Delft 机构记录](https://resolver.tudelft.nl/uuid:345185cc-8259-4faa-bf27-e25851224a7b) / [机构文件](https://repository.tudelft.nl/file/File_b3abd8c5-ab3e-4615-8c06-1bc6e8e33d51) | 本地 PDF 元数据标为 VoR（出版社正式版本）；记录未给出 CC 许可，按个人研究使用处理，不放入对外全文分享包。 | `f296757a69962a90730e36fa75d79d9ff64c6725ddf57e57a68b10a3d4ddbb20` |

> 注：第 6 项的 2021 是“第 18 届会议”在部分数据库中的届次标识；TU Delft 记录、正式 DOI 记录和 PDF 题录均显示出版年份为 2022。项目引用统一写 2022。

## 3. 逐篇说明：读什么、能用什么、不能说什么

### 3.1 Luo 等：有效感受野

论文区分理论感受野（结构上可能影响输出的全部输入区域）和有效感受野（实际梯度贡献显著的中心区域）。作者通过梯度分析说明有效区域常近似高斯分布，并且只占理论感受野的一部分；下采样、非线性、残差连接等会改变这种分布。

对本项目最直接的用途是理解 Tian 论文为什么不能只看“卷积核尺寸”：Tian 以原滑窗 `9 x 33` 为参照，再用理论 `20 x 72` 近似期望的有效覆盖范围。`20 x 72` 是网络设计选择，不是雷达物理常数，也不是 Luo 论文给出的普适比例。当前项目应把它当作可审计的结构假设，必要时用输入遮挡或梯度图验证。

不能把视觉网络上的“约 30%”直接当作本项目的定理。RD 图的两个轴单位不同、杂波结构不同，真正的有效覆盖范围要在本项目数据和归一化方式下重新检查。

### 3.2 Rahman 与 Robertson：鸟和无人机的微多普勒

作者用 K 波段 24 GHz 和 W 波段 94 GHz 雷达测量三种无人机与四种鸟。旋翼转动和鸟翼拍动都在主体多普勒之外产生微多普勒结构；相干雷达能够观察到周期脊线、谐波和展宽。W 波段样本的 SNR 通常更高，但 K 波段也能保留可辨识的微动信息。

它适合给组员建立“微多普勒不是一条神秘纹理”的直觉：先有局部散射点运动，再在慢时间相位中形成调制，最后由 STFT 或 Doppler 谱显现。项目未来采集时应记录载频、PRF、CPI、窗函数和目标姿态，不能只保存一张彩图。

这篇文章的鸟、无人机、波段、距离和设备都不同于当前 H/V 数据；它不能提供当前设备的 PRF，也不能证明空飘球载荷一定有相同的谱线。

### 3.3 Torvik 等：极化区分鸟与 UAV

该文在 S 波段用极化参数区分体型相近的鸟与 UAV，强调短驻留时间、空间分辨率有限时，极化仍可提供非微动判别信息。简单近邻分类器已经能利用共极化/交叉极化相关和去极化相关参数获得分离。

可迁移到项目的是“先构造可解释的相对极化参数，再做轻量基线”：例如通道功率比、交叉/共极化相对能量、相关性和稳定性，而不是一上来把所有复数通道塞进深网。它也提醒我们，极化优势依赖标定、目标姿态和雷达体制；当前 H/V 定义和幅相标定不完整时只能写 relative H/V features，不能写绝对散射矩。

### 3.4 Moore 等：用物理模型生成微多普勒

作者从无人机三维部件和旋翼运动参数生成微多普勒，再用专门的实验雷达验证单旋翼谱，显示仿真和实测在受控条件下可以较好吻合。价值在于提供“可控接口测试”：改变旋翼长度、转速、姿态、载频，观察谱线如何变化。

它可以用于本项目的合成数据单元测试、微多普勒特征解释和小样本预研，但不应替代真实空飘球采集。仿真模型若没有真实材料、姿态、散射中心和接收机链路，生成的漂亮谱图只说明模型自洽，不说明外场泛化。

### 3.5 Wang 等：有限数据下的参数化检测

这是一篇关于参数化自回归模型和自适应检测的综述，重点讨论训练数据很少、背景非均匀时如何用有限参数描述相关结构，并与非参数方法比较。它不是深度网络论文，但正好对应当前项目“样本少、背景扫描组少、不能靠随机拆帧制造独立样本”的现实。

对项目的启发有三点：先建立 CFAR/统计基线；用背景相关性和协方差诊断数据是否同质；在小样本时考虑参数化或混合模型，而不是盲目加深网络。它不能直接给出当前 H/V 数据的最优检测器，因为雷达体制、阵列和统计假设不同。

### 3.6 Sethuraman 等：带载荷 UAV 的极化分类

这是与未来“载荷识别”最接近的短文之一。PARSAX S 波段全极化 FMCW 雷达观测 DJI M200（空载/1 kg）和 M600（空载/2.35 kg），包含悬停、往返和矩形航线。作者从 VV/VH/HV/HH 单通道及组合通道提取 SVD、谱质心/带宽和均值、标准差、偏度、峰度等统计特征，再比较单通道决策融合与极化特征集成。摘要报告初始约 90%–95%，分场景结果并不完全相同，矩形航线、悬停和合并场景有明显差别。

这篇文章可以指导项目设计“极化特征编码器 + 可解释统计基线 + 场景独立测试”，但不能把其准确率搬到空飘球。它的目标是 UAV 载荷、波段和距离不同，样本也来自特定航线；而且论文自己指出极化特征的电磁解释仍需研究。

## 4. 推荐但暂未下载的补充文献

这些文献值得读，但目前没有把版权不明或访问受限的 PDF 放入目录：

| 文献 | 作用 | 状态 |
|---|---|---|
| C. Wang, J. Tian, J. Cao, X. Wang, “Deep Learning-Based UAV Detection in Pulse-Doppler Radar,” *IEEE TGRS* 60 (2022), DOI `10.1109/TGRS.2021.3104907` | Tian 2024 的直接前作，理解滑窗 DCNN、数据和速度比较 | IEEE 版本未确认开放下载；项目已有本地题录和用户提供的 2024 论文，不用不明镜像 |
| V. C. Chen et al., “Micro-doppler effect in radar: phenomenon, model, and simulation study,” *IEEE TAES* 42 (2006), DOI `10.1109/TAES.2006.1603402` | 微多普勒建模经典 | 出版社版本闭合，未下载 |
| A. Hanif et al., “Micro-Doppler Based Target Recognition With Radars: A Review,” *IEEE Sensors Journal* 22(4), 2948–2961 (2022), DOI `10.1109/JSEN.2022.3141213` | 微多普勒特征、时频分析和识别方法综述 | 期刊版闭合；TechRxiv 预印本 DOI `10.36227/techrxiv.17133611` 标注 CC BY 4.0，但本次网络环境未能稳定取得 PDF，暂只登记链接 |
| H. V. Sethuraman et al., “Classification of UAVs Carrying Payloads with Polarimetric Radar,” DOI `10.23919/EuRAD50154.2022.9784557` | 与项目载荷极化方向高度相关 | 已下载机构记录中的 VoR 文件，见第 6 项；无明确 CC 再分发授权 |

不使用 Sci-Hub、来路不明的转载站或去除版权信息的镜像。若以后需要把某篇全文放入公开分享包，应先核对明确的开放许可；否则分享 DOI、官方链接和自己的摘要即可。

## 5. 建议阅读顺序

1. 先读本项目本地的 Tian 2024 精读文档，再读 Luo，理解 RF/ERF、FCN 和本地结构差异。
2. 读 Rahman，建立微多普勒的时域到时频图直觉。
3. 读 Torvik，再读 Sethuraman，把极化特征从“公式”连接到鸟机和载荷任务。
4. 读 Moore，做可控仿真和接口测试时参考，不把仿真当外场证据。
5. 最后读 Wang 有限数据综述，反思当前的背景分组、CFAR 基线和小样本风险。

## 6. 文件完整性自检

在项目根目录执行：

```bash
sha256sum paper/references/推荐论文_公开/*.pdf
file paper/references/推荐论文_公开/*.pdf
```

若哈希改变，先检查是否下载到了 HTML 错误页或仓储封面，再更新本登记；不要静默替换来源。
