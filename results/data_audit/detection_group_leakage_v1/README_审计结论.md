# 检测数据分组泄漏审计结论

- 数据根目录：`/home/tobeornot8259748/projects/balloon_radar_project/data/raw/detection_dataset`
- 样本总数：1148

## Exact scan/source group

- background：6个exact group，其中6个跨split。
- uav：71个exact group，其中0个跨split。

## Session敏感性

- gap=30s：background=1, uav=2；跨split session=3。
- gap=60s：background=1, uav=2；跨split session=3。
- gap=120s：background=1, uav=2；跨split session=3。
- gap=300s：background=1, uav=1；跨split session=2。

## 自动警告

- background在gap=120s定义下仅1个session，无法进行严格session级三划分；需要补采独立时段数据
- background在gap=300s定义下仅1个session，无法进行严格session级三划分；需要补采独立时段数据
- background在gap=30s定义下仅1个session，无法进行严格session级三划分；需要补采独立时段数据
- background在gap=60s定义下仅1个session，无法进行严格session级三划分；需要补采独立时段数据
- uav在gap=120s定义下仅2个session，无法进行严格session级三划分；需要补采独立时段数据
- uav在gap=300s定义下仅1个session，无法进行严格session级三划分；需要补采独立时段数据
- uav在gap=30s定义下仅2个session，无法进行严格session级三划分；需要补采独立时段数据
- uav在gap=60s定义下仅2个session，无法进行严格session级三划分；需要补采独立时段数据

## 下一步

1. 先审阅本目录中的exact/session overlap表。
2. 根据真实采集流程确定UAV连续session时间间隔，不能仅凭模型结果选择。
3. 若某一类别严格session少于3组，不能声称跨session独立测试，应补采数据。
4. 确定分组级别后，再运行`propose_detection_group_split_v1.py`，不要直接移动原始数据。