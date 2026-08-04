# 队员复现指南

版本：2026-08-04
适用提交：以分享包 `MANIFEST.json` 中的 `source_commit` 为准。

## 1. 先说清楚：什么叫“复现”

本项目有四种不同等级，不能混为一谈：

| 等级 | 你手里的材料 | 可以做什么 | 不能声称什么 |
|---|---|---|---|
| A：分享包复核 | 最新 ZIP | 阅读结论、检查 CSV/图、验证 SHA256、核对指标定义和边界 | 不能重新训练、不能从 ZIP 单独得到模型预测 |
| B：源码接口复核 | 完整 Git 仓库 + 环境，无原始数据 | 运行语法检查、单元测试、查看入口 `--help`、验证不加载测试集的 smoke | 不能报告性能，不构成论文复现 |
| C：冻结结果重放 | 完整仓库 + 对应 manifest + 原始 MAT/IQ + 对应 checkpoint/冻结预测 | 重放正式表格、审计上下文、重建图表和分享包 | 不能把内部开发评价写成外部盲测 |
| D：重新训练 | 完整仓库 + 原始数据 + manifest + 配置，checkpoint 可缺失 | 按同一划分重新训练并产生新实验 | 新训练结果不是历史冻结数字，必须新建实验 ID 并重新审计 |

当前脱敏分享包属于 **A 级**。它是可追溯、可校验的成果摘录，不是自包含的训练包。

LAT-MRICD-1.0 是另一个公开数据证据对象。队员可自行从期刊官方补充材料取得原始 ZIP，
但项目分享包不会代为分发；下载后必须先核对 ZIP SHA256 和批次审计，再运行算法。

## 2. 队员需要向项目负责人申请的材料

不要把原始数据和权重直接放进公共分享包。队员在受控的项目目录中申请：

1. 完整仓库及指定 Git 提交；
2. 对应的 `results/data_audit/dataset_v4_multifold/` manifest；
3. 与 manifest 路径一致的原始 MAT/IQ 和标签文件；
4. 需要重放的 fold checkpoint，或已经冻结的逐样本预测表；
5. 该实验的配置文件、随机种子和实验台账行；
6. 运行机器、PyTorch/CUDA 版本和是否允许 GPU 的说明。

如果只拿到新数据而没有历史 checkpoint，必须走 D 级重新训练流程，不能称为“复现历史结果”。

## 3. 环境和仓库检查

以下命令在项目根目录执行。不要把个人绝对路径写入 manifest、日志或提交。

```bash
# 从分享包 MANIFEST.json 的 source_commit 取得精确提交，不使用文档里的旧哈希。
git checkout <source_commit>
conda env create -f environment.yml
conda activate radar-torch
python scripts/check_project_health.py --require-joint-inputs
python -m pytest
```

如果环境已经存在，只需确认：

```bash
python --version
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
git rev-parse HEAD
git status --short
```

预期：Python 3.11、项目健康检查通过、测试全部通过、工作区没有未说明的代码改动。

公开 LAT-MRICD 数据放到 `data/raw/external/LAT-MRICD-1.0/` 后运行：

```bash
sha256sum data/raw/external/LAT-MRICD-1.0.zip
python scripts/audit_lat_mricd_dataset_v1.py --overwrite
```

预期 ZIP SHA256 为
`2fe0d5e89016382c7c980172d67ba640179d6e2724edc735bcdf65c66b533bc0`，审计状态为
`READY_FOR_PREREGISTERED_GROUPED_BASELINE`。任何训练划分都必须按 batch 分组，不能随机拆行。

### 3.1 LAT-MRICD D17-NX/HX 冻结结果重放

先阅读 `docs/LAT_MRICD_GROUPED_BASELINE_PROTOCOL_V1.md`。V7 分享包只支持 A 级证据复核；
下面的完整重放还需要 Git 仓库、`radar-torch` 环境和上述公开原始数据。不得同时读取聚合
MAT 与同目录明细 MAT，否则会重复计入记录。

```bash
python scripts/freeze_lat_mricd_grouped_split_v1.py --overwrite
git diff --exit-code -- data/splits/lat_mricd_x_batch_grouped_v1.csv \
  data/splits/lat_mricd_x_batch_grouped_v1.json

python scripts/run_lat_mricd_grouped_baseline_v1.py \
  --output-dir /tmp/lat_mricd_grouped_baseline_replay \
  --overwrite

python scripts/build_lat_mricd_grouped_evidence_v1.py \
  --source-dir /tmp/lat_mricd_grouped_baseline_replay \
  --output-dir /tmp/lat_mricd_grouped_evidence_replay \
  --overwrite

diff -rq /tmp/lat_mricd_grouped_evidence_replay/tables \
  results/final_evidence/lat_mricd_grouped_baselines_v1/tables
```

预期：冻结划分没有 Git 差异，聚合表逐文件一致。`REPORT.md` 和
`evidence_manifest.json` 会记录重放时的当前提交，因此不要求与历史文件逐字节相同。重放
不得据 OOF 结果重新选特征、调参或宣布模型胜者。

### 3.2 LAT-MRICD D17-XBAND 只允许证据复核

D17-XBAND 已完成唯一一次提交绑定的密封运行。X->S 的 UAV batch recall 为 `0.4433`，
未通过预登记门；X->Ku 虽然通过该 target 的全部门槛，两个 locked target 仍未同时通过，
所以总决策为 `FAIL_STOP`。S/Ku target 均已消费。

队员只能核对 `evidence/24_LAT_MRICD_CROSS_BAND_TRANSFER.md`、消费记录、门判定、聚合表和
哈希，或运行不接触真实 target 的合成合同测试：

```bash
python -m pytest tests/test_lat_mricd_cross_band_transfer.py \
  tests/test_lat_mricd_cross_band_evidence.py
```

不得再次运行正式 runner，不得用复制数据、修改路径或新实验 ID 绕过消费记录，也不得在
同一 S/Ku target 上训练 CNN、做域适配、扩特征、调阈值或建立新的确认性比较。如需新的
跨频确认性结论，必须换独立且未消费的 target，并在查看结果前建立新协议。

### 3.3 LSS-DAUR V3 只读审计复核

V7 分享包可直接做 A 级复核：阅读 `evidence/26_LSS_DAUR_READ_ONLY_AUDIT.md`、核对
JSON 和聚合表。完整仓库中的 B/C 级复核还需从官方 ScienceDB V3 自行取得原始发布，
放在 `data/raw/external/LSS-DAUR-1.0/`；受 `CC-BY-NC-ND-4.0` 约束，原始或重打包数据
不得从项目分享包转发。

```bash
conda run -n radar-torch python scripts/audit_lss_daur_v1.py --overwrite
conda run -n radar-torch python -m pytest -q tests/test_audit_lss_daur_v1.py
```

预期状态为 `PASS_SCHEMA_PAIRING_BLOCKED_GROUPING_AND_PHYSICAL_AXIS`，`paired_track_count=77`、
`unique_signal_trajectory_content_count=76`、`candidate_source_session_group_count=39`、
`frame_count=11366` 且 `model_training_allowed=false`。审计器只写入被忽略的本地审计目录，
不修改原始文件，也不训练模型。V7 仅纳入报告、summary 与 39/6/3 行三张聚合表，不纳入
逐 recording 的日期、路径、时间戳或 payload 哈希明细。

复核纪律：canonical/backup 是等价视图，不能倍增；TD/TR、完全重复记录和保守连通组不得
跨 split；不能随机拆 MAT/frame/window、静默修正 6 个日期冲突，或把 19 条 1024-bin
记录映射到未经说明的物理 Hz/速度轴。得到相同阻塞结论才算复核成功，不能把它改写成
模型训练或识别性能结果。

## 4. 分享包的独立复核

解压最新包后，在包含 `SHA256SUMS.txt` 的目录执行：

```bash
sha256sum -c SHA256SUMS.txt
```

然后按以下顺序阅读：

1. `README.md`；
2. `docs/00_ONE_PAGE_SUMMARY_ZH.md`；
3. `docs/02A_HISTORICAL_PROJECT_RECONSTRUCTION_ZH.md`；
4. `docs/03_RESULTS_AND_EVIDENCE_ZH.md`；
5. `docs/09_RECENT_PROGRESS_AND_FAILURE_ANALYSIS_ZH.md`；
6. `docs/12_PROJECT_TASK_LEDGER_ZH.md`；
7. `evidence/` 中与问题对应的冻结报告。

队员可以独立核对：

- 表格的分母和指标定义；
- 六折、两折、测试后诊断和外部盲测的区别；
- 完整扫描 BC-DPG 的离线条件；
- 日期混杂、时间顺序缺失和标定缺口；
- SHA256 和 Markdown 链接完整性。

## 5. 有完整数据和 checkpoint 时的重放顺序

先做只读预检，不要直接覆盖冻结目录：

```bash
python scripts/check_project_health.py --require-joint-inputs
python scripts/audit_detection_acquisition_order.py --overwrite
python scripts/audit_detection_group_leakage_v1.py --overwrite
```

确认 manifest、checkpoint 和配置都来自同一个实验台账行后，再查看入口参数：

```bash
python scripts/run_bc_dpg_v3.py --help
python scripts/run_polarimetric_representation_benchmark_v2.py --help
python scripts/run_roi_stage4_selected_sixfold_v1.py --help
python scripts/run_tian_fcn_sixfold.py --help
```

冻结证据重建使用新输出目录，不能覆盖正式证据：

```bash
python scripts/build_final_roi_bc_dpg_joint_audit.py \
  --output-dir results/data_audit/team_rebuild_YYYYMMDD
```

已有冻结预测时，可以重建图表和脱敏包：

```bash
python scripts/build_roi_bc_dpg_joint_paper_assets.py --overwrite
python scripts/build_bc_dpg_localization_evidence.py --overwrite
python scripts/build_project_share_package.py
```

这些命令的前提是内部证据目录、原始数据和 checkpoint 已按项目约定准备好。分享包本身
没有这些二进制输入，因此在分享包解压目录中运行会失败，这是有意的保护，而不是代码错误。

## 6. 结果对齐时必须使用的口径

### BC-DPG v3

当前六折内部开发证据的参考值是：

- 完整扫描上下文 BC-DPG：`56/830` 背景误警、`289/318` 联合检测定位成功；
- 样本独立 BC：`122/830` 背景误警、`289/318` 联合成功；
- 完整扫描版本必须标为 `offline scan-aware upper bound`，不能标为严格实时模型。

### 零多普勒开发比较

在同一 CPU 重推理口径下：

- CPU baseline：`187` 个误警；
- 固定 soft notch：`120` 个误警、`290/318` 联合成功；
- fixed notch + target-protected residual V2：`109` 个误警、`290/318` 联合成功。

V2 是开发参考，不是盲测或部署结果。不能把 CPU 重推理的 `187` 与冻结 GPU 表中的 `186`
混为同一个基线。

### 零多普勒人工复核

这是对既有预测的人工审计，不训练模型、不选择阈值，也不生成新的性能结论。前提是仓库中
已有对应六折的 fixed-notch/residual 预测、特征目录和原始 MAT/IQ。先构建原始队列和 P0 图册：

```bash
python scripts/build_zero_doppler_human_review_queue_v1.py --overwrite
python scripts/build_zero_doppler_review_atlas_v1.py --overwrite
python scripts/build_zero_doppler_review_workbench_v1.py --overwrite
```

直接在浏览器打开
`results/data_audit/zero_doppler_review_atlas_v1/review_workbench.html`，只先复核其中 11 个
`P0_removed_by_residual` 条目。工作台逐例显示图册和结构指标，进度保存在浏览器
`localStorage`；导出的 CSV 保留审计所需原始列，不会覆盖源队列。

`reviewed` 必须填写可见结构和备注；没有独立场景记录时 `physical_class` 必须保持
`unknown`。如果填写具体类别，必须把 `evidence_source` 写为
`independent_scene_record` 并在备注中说明依据。工作台会在导出前执行相同规则，但导出后
仍须用命令行审计：

```bash
python scripts/audit_zero_doppler_human_review_v1.py \
  --reviewed-queue results/data_audit/zero_doppler_human_review_v1/review_queue_reviewer_YYYYMMDD.csv \
  --output-dir results/data_audit/zero_doppler_human_review_summary_reviewer_YYYYMMDD
```

图册、工作台、逐样本队列和复核汇总都只在受控本地目录保存，不进入分享包。审计输出为
`INCOMPLETE` 仅表示还有待复核项，并不是失败或物理结论。

### Tian FCN

Tian 线是方法级迁移诊断，不是成功复现。point-GT 分支只在 Fold 1 的 train/validation
设置上得到 `22/53` 联合成功和 `2/150` 背景误警，不能当作论文复现或六折部署结果。

## 7. 重放结果的验收表

队员完成一次重放后，必须提交下面这些信息：

| 项目 | 必填内容 |
|---|---|
| 实验 ID | 新建唯一编号，不复用历史 ID |
| Git | commit、工作区状态 |
| 环境 | Python、PyTorch、CUDA/GPU 或 CPU |
| 数据 | manifest 路径、SHA256、样本数、fold 和 split |
| 权重 | checkpoint 路径、SHA256、训练 epoch |
| 配置 | YAML/JSON 路径和哈希 |
| 测试访问 | 是否读取 test；若读取，是否为一次性冻结评价 |
| 指标 | pooled、macro、每折、最差折、Pd、Pfa、AUC、定位误差 |
| 配对变化 | 新增/移除误警、目标损失、joint Pd 变化 |
| 结论 | 复现、数值漂移、接口 smoke 或失败诊断 |
| 输出位置 | 新结果目录和汇总 Markdown |

结果只有在这些信息齐全，并通过项目负责人审核后，才可进入论文或分享包。

## 8. 常见错误

- 只下载 ZIP 就声称“复现了模型”；
- 用 `sample_id`、文件名或 beam/azimuth 推断真实采集时间；
- 用测试集重新选阈值、loss、notch 半宽、ROI 组合或历史窗口；
- 把没有绝对标定的 H/V 比值称作绝对极化参数；
- 把单帧 RD 的速度纹理称作物理微多普勒；
- 把两折筛选结果当作六折独立盲测；
- 把 UAV/背景结果写成空飘球载荷分类结果；
- 把 LAT-MRICD 的归一化频谱写成有 Hz 单位的物理微多普勒，或随机拆行报告高精度；
- 直接覆盖 `results/final_evidence/` 或历史 checkpoint；
- 没有实验台账就把新训练数字追加到旧表格。

## 9. 队员分工建议

队员可以从三个独立方向参与：

1. **数据与物理复核**：读 MAT/TXT、核对 H/V、画 RD 图、检查标签和困难样本；
2. **算法与审计**：运行 baseline、极化特征压力测试、错误分析和结果表构建；
3. **工程与文档**：维护环境、测试、实验台账、复现日志和分享包。

每个方向都必须先在 `docs/PROJECT_TASK_LEDGER.md` 中领取任务 ID，再创建对应实验目录。

## 10. 一句话结论

分享包可以让队员理解项目、核对证据并知道如何进入完整复现流程；真正重现模型指标还需要
完整源码、原始数据、manifest、checkpoint 和实验条件。缺少其中任何一项，都应如实标成
接口复核、冻结结果审计或重新训练，而不是“成功复现”。
