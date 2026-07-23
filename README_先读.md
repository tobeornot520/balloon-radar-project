# BC-DPG 极化门控两折诊断采集补丁 V1

本补丁仿照项目此前的“补丁ZIP → 根目录安装 → 一条命令运行 → 自动生成单一验收ZIP”流程。

## 三条命令

```bash
cd ~/projects/balloon_radar_project
unzip -o BC_DPG_polarimetric_twofold_diagnostics_collection_v1.zip -d .
python apply_polarimetric_twofold_diagnostics_collection_v1.py
```

```bash
conda activate radar-torch
set -o pipefail
python scripts/collect_polarimetric_twofold_diagnostics_v1.py \
  --folds 1 4 \
  2>&1 | tee polarimetric_twofold_diagnostics_terminal_v1.log
```

完成后上传项目根目录中的：

```text
polarimetric_twofold_diagnostics_acceptance_v1.zip
```

工具不会重新训练，不会覆盖正式实验，不会复制原始数据或大权重。
