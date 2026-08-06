# 零多普勒可审计虚警库 V1

更新时间：2026-08-06
状态：`COMPLETE_AS_DEVELOPMENT_AUDIT`

## 1. 目的

本模块把 fixed-notch 与 fixed-residual 六折预测逐样本配对，建立可追溯的背景案例登记，回答：

1. residual 具体移除了哪些已有虚警，是否引入新虚警；
2. 变化集中在哪一折、哪一个扫描来源；
3. 人工复核记录能支持“可见结构”还是“物理类别”；
4. 哪些内容可以分享，哪些内容只能留在本地。

它不是一次新实验，不重新训练模型，不重新选择阈值，也不把已消费测试结果升级为外部盲测。

## 2. 冻结输入与结果

配置：`configs/zero_doppler_false_alarm_library_v1.json`。配置冻结了六折 fixed-notch / residual
预测、原始 120 条复核队列和 11 条已复核记录的 SHA-256。

| 项目 | 数量 |
|---|---:|
| 六折测试样本 | 1,148 |
| 目标样本 | 318 |
| 背景样本 | 830 |
| 背景扫描来源 | 6 |
| fixed-notch 虚警 | 120 |
| residual 虚警 | 109 |
| residual 移除 | 11 |
| residual 新增 | 0 |
| 两者共同保留 | 109 |
| 已人工复核 | 11 / 120 |

折级变化为：Fold 1 从 53 个虚警到 53 个；Fold 4 从 67 个到 56 个；其余四折均为
0 到 0。11 个被移除案例全部来自 Fold 4，也全部进入人工复核。人工可见模式为
`near_zero_doppler_peak` 9 例、`broad_structure` 2 例；11 例的物理类别均为 `unknown`。

## 3. 两层数据边界

### 本地逐样本层

`case_library_local.csv` 覆盖全部 830 个背景样本，保留 sample ID、源文件标识、两种方法的
分数与峰位置、变化类型、人工复核状态和备注。它用于后续逐图核验和现场记录回填，保持在
`results/data_audit/zero_doppler_false_alarm_library_v1/`，不进入 Git 和分享包。

### 可提交聚合层

- `fold_transition_summary.csv`：六折背景数、虚警数、转移数和折内 Pfa；
- `scan_transition_summary.csv`：覆盖六个背景来源，但只发布稳定哈希别名，不发布别名映射；
- `review_pattern_summary.csv`：只统计复核状态、可见模式和证据来源，不含备注；
- `summary.json`：输入哈希、冻结计数、声明边界和共享边界。

这些表禁止出现 `sample_id`、`source_file`、`review_note` 和本机路径。脚本与测试会执行该门禁。

## 4. 结论边界

当前结果只支持：在已消费的开发证据中，固定 residual 保持“不新增虚警”，并从 Fold 4
移除 11 个与近零多普勒/宽结构相关的可见案例。

当前结果不支持：

- residual 已在独立外部数据上泛化；
- 109 / 830 是部署环境 Pfa；
- 可见水平亮带已经被识别为建筑、地杂波或某种具体物体；
- relative H/V 统计是经过标定的 ZDR、相关系数或极化散射参数；
- 通过继续查看这 830 个样本重新调阈值，再把结果称作盲测。

完整物理类别仍需独立现场记录、设备日志或同条件复测。没有独立证据时，`physical_class`
必须保持 `unknown`。

## 5. 复现

```bash
conda run -n radar-torch python scripts/build_zero_doppler_false_alarm_library_v1.py --overwrite
conda run -n radar-torch python -m pytest -q tests/test_zero_doppler_false_alarm_library.py
```

脚本运行只读取既有预测与复核 CSV，不读取原始 MAT，不调用网络，不使用 GPU，也不训练模型。
