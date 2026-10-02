# 项目长期日志

本文件只记录事实、决策和证据入口，不复制聊天记录。新记录追加在文件末尾，旧记录不改写；若发现错误，用“更正记录”明确说明。

## 更新规则

- 每次正式实验记录日期、任务 ID、代码提交、数据 manifest、配置、结果目录和结论边界。
- 失败实验必须记录原因、是否消耗外层数据、是否允许重开。
- 读数、图表和人工判断分开写；没有来源的物理解释保持 `unknown`。
- 外部资料或新数据进入项目时，先登记来源、版本、许可和 SHA-256。

## 2026-08-09

- 将项目整理为控制区、数据区、代码区、结果区和论文区的逻辑结构。
- 建立 `PROJECT_CONTROL/` 作为长期入口，新增技术手册、任务台账、路线图和本日志。
- 合并“当前开发过程”和历史记录：逐字节重复的 10 个开发文件只保留一份，14 份 saas-nexus 导出单独保留。
- 参考资料迁移到 `paper/references/`；旧根目录资料路径不再作为代码或文档入口。
- 将 4 个仍被早期工作流导入的兼容源码恢复到活动 `datasets/` 和 `training/` 目录，算法分享包不再依赖本地清理归档。
- 删除已过时的 `payload/` 旧交付快照、清理归档和修复脚本；内容可从 Git 历史恢复。
- 当前研究边界未改变：已有数据支持 H/V UAV 检测、定位和虚警机制开发，不支持空飘球载荷识别或物理微多普勒结论。

## 2026-09-24 / FIELD-RSTM-20250425-ST001

- 本次目标：接收、整理并审计学姐既有实验的 RSTM 数据，建立可供本项目预研使用的开发模型接口。
- 使用数据与分组：`data/raw/field_collection_20250425_st001/`；五个目标来源目录共 472 个 IQ，背景 IQ 3 个，背景 `.bin.bz2` 产品 291 个；该数据不是本项目定向采集，源目录标签暂未升级为物理真值。
- 代码与清单：`scripts/intake_rstm_field_collection.py`、`scripts/train_rstm_target_baseline.py`；`data/metadata/field_collection_20250425_st001_manifest_hashed.csv`（766 行、766 个唯一 SHA-256）。
- 实际完成：完成根目录数据入库、RSTM 头部读取、文件名/头时间并列记录、RHI/PPI/RPI 分类；完成每文件一个样本的鲁棒统计特征基线训练。
- 结果目录：`results/experiments/rstm_target_baseline_20250425/`；鲁棒特征修正后的五类来源分类 Accuracy 0.565、Balanced Accuracy 0.503、Macro F1 0.473；气球 vs 其他目标开发基线 Accuracy 0.957、Macro F1 0.940。
- 失败、阻塞或数据消耗：RSTM 记录块、H/V 通道、物理轴和真实 session 尚未由设备方确认；类别按同日时间连续采集，时间/场景混杂；背景产品尚未进入模型训练。训练结果不能称独立盲测、跨日期泛化或实时部署。
- 当前允许结论：学姐数据为本项目提供了真实 RSTM、多目标来源、站点背景和 RHI/PPI 接口预研材料；当前仅允许作为格式读取、特征接口和探索性模型验证，不构成本项目正式实验结果。
- 下一步与放行条件：先获取 RSTM 格式、通道/轴和标签说明，再按本项目需求重新设计并实施目标/背景、天气、场地和同步采集；只有本项目数据配对完成并具备多日期或多场地 locked test 后，才重建正式 Power2/RD/STFT 检测与多域模型。

## 2026-09-25 / ENVIRONMENT-PLANNING-V1

- 本次目标：明确当前气象与场地方向的阻塞事项，提出补充数据采集方案，并建立影响因素验证的理论模型和字段契约。
- 使用数据与分组：未使用新的项目定向外场数据；学姐 RSTM 数据继续仅作格式、读取器和探索性预研材料，不作为环境模型训练或正式评价依据。
- 实际完成：新增 `PROJECT_CONTROL/ENVIRONMENTAL_BLOCKERS_AND_SUPPLEMENTAL_COLLECTION_PLAN_ZH.md`，列出 B1-B11 阻塞、E0-E3 分阶段采集、目标/背景配对、同步、气象、场地和锁定评价要求；新增 `PROJECT_CONTROL/environment/WEATHER_SCENE_FACTOR_MODEL_PLAN_V1_ZH.md`，定义快变天气/慢变场地观测模型、F0-F4 验证路线和 S0-S5 模型比较；新增 `configs/environment_context_observation_schema_v1.json`，冻结字段、单位、质量码和正式模型门禁。
- 当前允许结论：已完成研究设计和工程接口准备；可以进行字段审计、采集干运行和 Pilot 准备，不能声称天气/场地因素已经被验证，也不能报告正式环境模型增益。
- 当前阻塞：本项目定向雷达原始数据、逐样本时间与同步、目标/背景真值、气象观测、场地档案和独立 locked test 尚未齐备；正式模型门为 `blocked_until_project_directed_field_collection`。
- 下一步与放行条件：先由导师/场地方确认雷达格式、H/V/轴、可用场地、气象设备和安全条件；随后执行 E0 能力与同步、E1 Pilot。只有 capture/causal 合同、同条件目标/背景配对以及多日期或多场地 locked test 通过后，才启动 F1-F4 环境影响验证和 S3-S5 增量模型。

## 2026-09-26 / ENVIRONMENT-PLANNING-VALIDATION

- `python3 -m json.tool configs/environment_context_observation_schema_v1.json`：通过。
- `PYTHONPATH=. conda run --no-capture-output -n radar-torch pytest -q tests/test_project_contracts.py`：`2 passed`。
- `python scripts/check_current_direction_completion_v1.py --overwrite`（`radar-torch` 环境）：输出 `IN_PROGRESS`，当前完成项为 `3/6`；其中 2 项为外部阻塞，另 1 项因既有分享包 manifest/zip 证据路径缺失而未通过。该状态与 D19 的环境设计完成不是同一个完成门，不能互相替代。
- 当前结论：本轮没有训练正式气象/场地模型；只完成字段契约、理论模型和采集/验证设计，正式环境模型继续保持关闭。

## 2026-09-26 / ENVIRONMENT-LOCAL-PREFLIGHT-V1

- 本次目标：完成不依赖外场数据的本地准备工作，把环境字段、场地档案、因素分析配置和质量审计工具串成可执行流程。
- 实际完成：新增 `configs/environment_context_observation_template_v1.csv`、`configs/site_scene_profile_template_v1.csv`、`configs/environment_factor_analysis_config_v1.json`、`scripts/validate_environment_context_v1.py`、`tests/test_environment_context.py` 和 `PROJECT_CONTROL/environment/LOCAL_PREPARATION_RUNBOOK_V1_ZH.md`。
- 本地验收：环境审计器允许空白模板以 `EMPTY_TEMPLATE` 通过；环境、readiness、同步和项目合同测试共 `21 passed`；空模板不被误报为正式环境数据。
- 当前允许结论：本地字段和审计工具已经可以接收第一批真实天气/场地观测；可以执行 E0/E1 的准备和质量检查，但尚未产生环境因素或模型性能结论。
- 下一步与放行条件：外场数据到位后先填场地档案和观测表，依次通过环境字段审计、同步审计、capture/causal manifest 和 readiness 门，再开始 F0 分层基线；S3-S5 仍保持关闭。

## 记录模板

### YYYY-MM-DD / TASK-ID

- 本次目标：
- 使用数据与分组：
- 代码提交：
- 配置与命令：
- 结果目录/证据：
- 实际完成：
- 失败、阻塞或数据消耗：
- 当前允许结论：
- 下一步与放行条件：
