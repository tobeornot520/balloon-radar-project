# 复杂气象、场地与干扰方向

这是本项目环境方向的统一入口，负责把气象条件、固定场地背景、射频/系统干扰和后续部署设计放在同一条研究线上。

## 先看什么

| 顺序 | 文件 | 用途 |
|---:|---|---|
| 1 | [ENVIRONMENT_PROJECT_DESIGN_REPORT_ZH.md](ENVIRONMENT_PROJECT_DESIGN_REPORT_ZH.md) | 明天组会介绍、研究假设、数据需求、Pilot 和算法路线 |
| 2 | [ENVIRONMENT_RESEARCH_STARTER_ZH.md](ENVIRONMENT_RESEARCH_STARTER_ZH.md) | 初步调研路线、变量字典、场地档案和最小 Pilot |
| 3 | [ENVIRONMENT_INTERFERENCE_KNOWLEDGE_BASE_ZH.md](ENVIRONMENT_INTERFERENCE_KNOWLEDGE_BASE_ZH.md) | 20 篇论文的逐篇提炼、方法阶梯和术语表 |
| 4 | [ENVIRONMENT_INTERFERENCE_LITERATURE_20260918.md](ENVIRONMENT_INTERFERENCE_LITERATURE_20260918.md) | 题录、官方链接、下载状态、文件哈希和阅读顺序 |
| 5 | [RESEARCH_STATUS_DOMESTIC_INTERNATIONAL_ZH.md](RESEARCH_STATUS_DOMESTIC_INTERNATIONAL_ZH.md) | 国内外研究现状、主要不足和本项目切入点 |
| 6 | [WEATHER_SCENE_RESEARCH_STATUS_ZH.md](WEATHER_SCENE_RESEARCH_STATUS_ZH.md) | 专门聚焦气象条件、场景杂波、场地先验和 DEM/DSM 建模的研究现状与不足 |
| 7 | [../ENVIRONMENTAL_BLOCKERS_AND_SUPPLEMENTAL_COLLECTION_PLAN_ZH.md](../ENVIRONMENTAL_BLOCKERS_AND_SUPPLEMENTAL_COLLECTION_PLAN_ZH.md) | 当前阻塞事项、补充采集需求、分阶段 Pilot 和放行条件 |
| 8 | [WEATHER_SCENE_FACTOR_MODEL_PLAN_V1_ZH.md](WEATHER_SCENE_FACTOR_MODEL_PLAN_V1_ZH.md) | 气象与场地因素的理论观测模型、统计验证和环境增量模型 |
| 9 | [../../configs/environment_context_observation_schema_v1.json](../../configs/environment_context_observation_schema_v1.json) | 气象/场地字段、单位、质量码和正式模型门禁 |
| 10 | [LOCAL_PREPARATION_RUNBOOK_V1_ZH.md](LOCAL_PREPARATION_RUNBOOK_V1_ZH.md) | 本地准备流程、模板、命令和停止条件 |
| 11 | [../../configs/environment_context_observation_template_v1.csv](../../configs/environment_context_observation_template_v1.csv) | 可填写的天气/场地观测表 |
| 12 | [../../configs/site_scene_profile_template_v1.csv](../../configs/site_scene_profile_template_v1.csv) | 固定站点场景档案表 |
| 13 | [../../configs/environment_factor_analysis_config_v1.json](../../configs/environment_factor_analysis_config_v1.json) | S0--S5 因素分析配置和评价约束 |

## 关联但不搬入本目录的文件

以下文件属于全项目通用规范，仍保留在原位置：

- [实验准备与数据需求](../EXPERIMENT_PREPARATION_DATA_REQUIREMENTS_ZH.md)：外场数据合同、采集阶段、字段和放行门；
- [外场采集 SOP](../../docs/FIELD_COLLECTION_SOP_V1.md)：设备同步、Dry Run、Pilot 和停止条件；
- [新数据采集协议](../../docs/NEW_DATA_COLLECTION_PROTOCOL.md)：`capture`、`causal`、`locked_evaluation` 合同；
- [任务板](../TASK_BOARD.md)：正式任务状态的唯一来源。

原始论文 PDF 保留在 [paper/references/环境与干扰_公开/](../../paper/references/环境与干扰_公开/)。该目录被 Git 忽略，避免把大文件和许可证边界不一致的全文混入项目文档提交；本目录的文献登记文件负责管理题录、官方来源和哈希。

## 研究边界

当前优先顺序是：

```text
记录和分层
  -> 同条件目标/背景对照
  -> 简单统计与因果校准
  -> 经验场地先验
  -> 真实干扰确认后的恢复方法
  -> 新日期/新场地锁定评价
```

本目录中的方案是研究设计，不代表设备能力已经确认、外场已经采集或环境模型已经完成。正式状态仍以 [TASK_BOARD.md](../TASK_BOARD.md) 和项目证据目录为准。
