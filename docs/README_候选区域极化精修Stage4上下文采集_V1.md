# 候选区域引导极化精修 Stage 4：工程上下文采集 V1

## 目的

Stage 4将保留Power2的候选峰与定位，只在局部ROI内使用功率、RI4和门控极化特征进行候选确认。为了与现有工程零猜测兼容，本工具先采集真实模型、Dataset、训练入口、预测CSV及checkpoint结构。

## 安全边界

- 不训练模型；
- 不修改现有代码逻辑；
- 不覆盖任何正式实验；
- 不复制`.mat`原始数据；
- 不复制`.pt/.pth/.ckpt`权重，只读取键名、张量形状和SHA256；
- 所有输出放入独立的`results/data_audit/roi_polarimetric_stage4_context_v1/`。

## 安装

```bash
cd ~/projects/balloon_radar_project
unzip -o BC_DPG_roi_polarimetric_stage4_context_collection_v1.zip -d .
python apply_roi_polarimetric_stage4_context_collection_v1.py
```

## 运行

```bash
cd ~/projects/balloon_radar_project
conda activate radar-torch
set -o pipefail

python scripts/collect_roi_polarimetric_stage4_context_v1.py \
  --folds 1 4 \
  2>&1 | tee roi_polarimetric_stage4_context_terminal_v1.log
```

正常应显示：

```text
Stage 4 ROI极化精修上下文采集完成
raw data copied  : false
checkpoint copied: false
```

## 上传

上传项目根目录自动生成的：

```text
roi_polarimetric_stage4_context_acceptance_v1.zip
```

不要上传原始数据、完整checkpoint或整个工程。
