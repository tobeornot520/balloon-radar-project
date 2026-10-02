# 气象与场地本地准备运行手册 V1

版本：V1.0  
日期：2026-09-26  
状态：本地准备完成；正式外场数据仍未到位

## 1. 这份手册解决什么问题

本手册把当前不依赖外场数据的工作收口为一条固定流程：

```text
字段和配置冻结
  -> 空白模板检查
  -> 现场 readiness / 同步 / manifest 工具准备
  -> E0 干运行
  -> E1 Pilot
  -> F0 环境字段质量审计
```

当前不能用历史学姐 RSTM 数据代替 E0/E1，也不能用模拟数据制造天气或场地性能结论。

## 2. 已准备好的本地资产

| 资产 | 用途 |
|---|---|
| `configs/environment_context_observation_schema_v1.json` | 气象、场地、背景先验字段和单位契约 |
| `configs/environment_context_observation_template_v1.csv` | 每条天气/场地观测的可填写模板 |
| `configs/site_scene_profile_template_v1.csv` | 每个固定站点的几何、遮挡和资料来源档案 |
| `configs/environment_factor_analysis_config_v1.json` | S0--S5 模型比较、分组、指标和禁止事项 |
| `scripts/validate_environment_context_v1.py` | 对天气/场地观测做字段、时间、范围和背景来源审计 |
| `configs/field_readiness_checklist_v1.json` | capability 到 pilot 的五道 readiness 门 |
| `configs/field_sync_event_template_v1.csv` | 雷达、视频、真值同步事件记录 |
| `configs/data_collection_manifest_template_v1.csv` | capture/causal/locked-evaluation 数据合同 |
| `scripts/audit_field_readiness_v1.py` | readiness 门审计 |
| `scripts/audit_field_synchronization_v1.py` | 同步事件数值审计 |
| `scripts/validate_data_collection_manifest.py` | 采集 manifest 合同审计 |

## 3. 现在即可执行的本地检查

### 3.1 检查空白环境模板

```bash
conda run --no-capture-output -n radar-torch \
  python scripts/validate_environment_context_v1.py \
  configs/environment_context_observation_template_v1.csv \
  --allow-empty \
  --output-dir results/data_audit/environment_context_template_v1 \
  --overwrite
```

预期状态是 `EMPTY_TEMPLATE`，它只说明模板结构正确，不代表环境数据可用。

### 3.2 初始化 readiness 证据表

```bash
python scripts/initialize_field_readiness_evidence.py \
  work/field_readiness_evidence_v1.csv \
  --overwrite
```

空白证据表应保持 `pending`。没有真实设备证据时，不要把项目状态手工改成 `pass`。

### 3.3 运行已有合同测试

```bash
PYTHONPATH=. conda run --no-capture-output -n radar-torch \
  pytest -q tests/test_environment_context.py \
    tests/test_field_readiness.py \
    tests/test_field_synchronization.py \
    tests/test_project_contracts.py
```

本地准备阶段的验收标准是测试通过；它不打开任何外场或模型门禁。

## 4. 第一次外场数据到位后的顺序

1. 先填写 `site_scene_profile_template_v1.csv`，给每个 `site_id` 建立版本化档案。
2. 复制 `environment_context_observation_template_v1.csv`，填入 UTC 时间、传感器、天气和场地观测。
3. 用 `validate_environment_context_v1.py` 检查时间、单位、质量码、缺失和背景图来源。
4. 对同步事件运行 `audit_field_synchronization_v1.py`。
5. 对每个采集 manifest 依次运行 `capture`、`causal`，不要直接提交 `locked_evaluation`。
6. 只有 E0/E1 通过后，才开始 F0 字段覆盖、缺失图和混杂矩阵。

## 5. 明确不做的事情

- 不用文件名时间、mtime 或扫描顺序猜测真实天气/雷达时间；
- 不把缺失天气写成 0；
- 不把 DEM/DSM 直接当作回波强度；
- 不把场地先验更新到 locked test 后再报告测试结果；
- 不在正式项目数据缺失时训练 S3--S5 并报告增益；
- 不把 `EMPTY_TEMPLATE`、`PASS_NUMERIC_LIMITS_ONLY` 或 readiness 代码测试写成实验通过。

## 6. 当前停止点

本地工具链已经可以使用，但正式工作仍停在外部事实门：需要项目定向雷达采集、真实同步、目标/背景配对、天气记录、场地档案和新日期或新场地 locked test。相关边界见 [阻塞事项与补充采集方案](../ENVIRONMENTAL_BLOCKERS_AND_SUPPLEMENTAL_COLLECTION_PLAN_ZH.md) 和 [气象与场地因素模型规划](WEATHER_SCENE_FACTOR_MODEL_PLAN_V1_ZH.md)。
