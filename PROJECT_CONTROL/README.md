# 项目控制面板

这是项目的长期入口。先看本目录，再进入代码、数据、结果或论文区。

## 现在看什么

| 目的 | 文件 |
|---|---|
| 了解雷达物理、信号处理、特征、模型和代码 | `TECHNICAL_HANDBOOK_ZH.md` |
| 查看待解决问题、负责人、验收门和下一动作 | `TASK_BOARD.md` |
| 查看短期、中期、长期路线及停止规则 | `ROADMAP.md` |
| 查看每次正式推进的事实记录和决策 | `PROJECT_LOG.md` |
| 查看原始方向输入 | `source_inputs/` |
| 查看验收材料 | `team_review/` |
| 查看会议决策与规划备忘 | `meetings/` |
| 查看历史开发导出 | `logs/` |

## 四区结构

- `data/`：原始信号、处理数据、划分、manifest 和数据来源登记。
- 代码区：根级 `datasets/`、`features/`、`models/`、`training/`、`evaluation/`、`scripts/`、`tests/`、`utils/`、`baselines/`、`tools/`；入口说明见 `code/README.md`。
- `results/`：审计结果、实验运行、图表、表格和冻结证据。
- `paper/`：参考资料、论文草稿、图表和归档稿件。

Python 代码暂不物理搬到 `code/`，因为既有导入路径和测试依赖根级模块；这是一种有意的逻辑分区。

## 每次更新规则

1. 先在 `TASK_BOARD.md` 认领任务 ID，再修改代码或运行正式实验。
2. 正式实验必须记录代码提交、数据 manifest、配置、结果目录、指标和结论边界。
3. 失败实验也要写入 `PROJECT_LOG.md`，不得只保留最好结果。
4. 原始数据、checkpoint、完整日志、论文全文和成员材料不进入 Git 或分享包。
5. 每次阶段收口后运行项目健康检查和阶段完成检查，再更新本控制面板。

常用检查：

```bash
python scripts/check_project_health.py
python -m pytest -q
python scripts/check_current_direction_completion_v1.py --overwrite
```

当前主 UAV 方向仍以阶段检查结果为准；`BLOCKED_EXTERNAL` 不得写成方向已完成。
