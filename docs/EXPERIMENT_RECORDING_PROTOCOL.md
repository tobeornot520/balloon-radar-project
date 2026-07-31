# 实验记录协议

版本：2026-07-29

## 记录粒度

台账按一次独立运行记录，不按 epoch 手工记录。只要模型、数据划分、随机种子、通道、
训练目标、超参数或后处理规则发生改变，就视为新的 experiment。每个 epoch 的 loss
和验证指标继续保存在实验目录的 `training_history.csv`。

大型 checkpoint、缓存和图像可以留在被 Git 忽略的 `results/experiments/`。总台账
`results/experiment_ledger/experiments.csv` 及自动生成的
`results/experiment_ledger/summaries/*.json` 必须进入 Git，确保实验目录被清理后仍能
回答运行目的、配置、数据、测试集访问、关键结果和保留决策。

## 标准运行

新训练优先通过统一包装器执行：

```bash
python3 scripts/run_recorded_experiment.py \
  --experiment-id example_fold01_seed42 \
  --purpose "验证预登记的单一模型改动" \
  --evidence-role validation_only_diagnostic \
  --data-manifest results/data_audit/dataset_v4_multifold/fold_01_manifest.csv \
  --split-scope diagnostic \
  --test-policy forbidden \
  --config-path configs/example.yaml \
  --summary-path results/experiments/example_fold01_seed42/tables/summary.json \
  --artifact-dir results/experiments/example_fold01_seed42 \
  --seed 42 --fold 1 --channel H \
  -- python3 training/example.py --name example_fold01_seed42
```

包装器在命令启动前写入 `RUNNING`，结束后写入退出码和 `COMPLETED`、`FAILED` 或
`ABORTED`。若声明 test forbidden，但 summary 显示加载了 test，则标记为
`POLICY_VIOLATION`。

训练完成并审阅结果后，必须补充决策：

```bash
python3 scripts/manage_experiment_ledger.py update \
  --experiment-id example_fold01_seed42 \
  --decision-status DIAGNOSTIC_ONLY \
  --notes "validation 改善，但尚未开放 test"
```

## 历史与导入

已有 summary 可导入；导入器会自动保存一份小型、可追踪的 summary 快照。不得用
当前 Git 状态冒充当时状态，无法从 summary 证明的字段保持 `unknown`。已经删除且
无法追溯的探索实验只登记为 `LOST`，不补造参数和指标。

校验台账：

```bash
python3 scripts/manage_experiment_ledger.py verify
```

若包装进程被外部会话终止而遗留 `RUNNING`，先确认训练进程确实不存在，再显式关闭记录：

```bash
python3 scripts/manage_experiment_ledger.py mark-aborted \
  --experiment-id example_fold01_seed42 \
  --notes "确认进程不存在；外部会话中断"
```

## 保留规则

- 必须长期保留：台账、冻结配置、summary、数据 manifest 或其哈希、逐样本正式预测、
  模型选择与阈值来源。
- 每个正式或候选模型至少保留最佳 checkpoint；中间 epoch checkpoint 可清理。
- 失败实验保留台账和失败原因，产物可在确认无独有诊断价值后清理。
- 原始数据、唯一标注、已被报告引用的结果和唯一最佳 checkpoint 不得按临时产物删除。
