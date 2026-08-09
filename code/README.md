# 代码区说明

代码区采用逻辑划分，活动 Python 包暂保留在项目根目录，以保持现有导入路径和命令兼容。

| 子区 | 内容 | 典型入口 |
|---|---|---|
| `datasets/` | manifest 驱动的数据读取、标签和几何转换 | `detection_dataset_v3.py`、`polarimetric_detection_dataset_v2.py` |
| `features/` | RD、H/V 相对极化、ROI 和扫描上下文特征 | `polarimetric_rd.py`、`scan_context.py` |
| `models/` | FCN、DPG、背景校准和 ROI 精修模型 | `tian_fcn.py`、`target_protected_scan_calibrator.py` |
| `training/` | 可复用训练实现和损失 | `train_tian_fcn.py`、`tian_fcn_objective.py` |
| `evaluation/` | 后处理、指标和报告逻辑 | `tian_fcn_metrics.py`、`tian_fcn_postprocess.py` |
| `scripts/` | 用户调用的审计、实验编排、打包和检查 | `run_bc_dpg_v3.py`、`check_project_health.py` |
| `tests/` | 自动化合同测试 | `python -m pytest -q` |
| `baselines/` | CFAR 等经典基线 | 见 `scripts/README.md` |
| `tools/` | 诊断和维护工具 | 见 `tools/README.md` |

运行脚本前使用 `python <path> --help`。脚本存在不代表数据门已经打开；放行状态记录在 `PROJECT_CONTROL/TASK_BOARD.md` 和 `PROJECT_CONTROL/ROADMAP.md`。
