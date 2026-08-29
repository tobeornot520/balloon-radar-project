# 项目组员从零开始

日期：2026-08-29

这份说明给“手里只有学长提供的初始数据、还没有项目代码和文档”的组员。你不需要先学会 Git；先按压缩包阅读和做数据检查，之后再决定是否从 GitHub 同步更新。

## 先拿哪几个包

请向负责人收齐以下四个 ZIP：

1. `00_组员启动与目录说明`：本文件、项目摘要、目录地图和安装说明；
2. `01_项目资料与研究证据`：项目历史、当前结论、失败分析、论文边界和聚合结果；
3. `02_工程源码与配置`：模型、特征、数据接口、训练/评价入口和测试；
4. `03_外场数据合同与执行模板`：设备确认、采集 SOP、同步、标定和新数据记录表。

四个包合计很小，适合微信分开发送。原始雷达数据、checkpoint 和参考书不会放在包里：初始数据由负责人或学长单独交付，参考书按许可和需要另行阅读。

## 解压后的第一小时

1. 先读 `README.md`、`docs/share/TEAM_SYNC_BRIEF_20260826.md` 和 `PROJECT_CONTROL/TASK_BOARD.md`；
2. 再读 `docs/CURRENT_STATUS.md` 和 `docs/PROJECT_LEARNING_COURSE_ZH.md` 的课程目录；
3. 对照 `docs/PROJECT_STRUCTURE.md` 找到代码、数据、结果和论文区；
4. 不要在没有完成数据审计前启动训练，也不要把内部开发数字写成空飘球识别或严格实时结论。

## 初始数据放在哪里

拿到学长数据后，不要改动原文件名或直接覆盖项目文件。先放到本地受控目录，再按
`configs/data_collection_contract_v1.json`、`docs/DATA_CARD.md` 和数据 manifest 模板登记。
只有通过格式、通道、时间、分组和标签检查后，才允许进入训练或评价流程。数据目录默认被 Git 忽略，不会因 `git add -A` 上传。

## 不会 Git 也没关系

短期协作：以四个 ZIP 为准，按 `README.md` 和课程提交物推进。

长期更新：安装 Git 后，在项目目录执行：

```bash
git clone https://github.com/tobeornot520/balloon-radar-project.git
cd balloon-radar-project
git pull --rebase origin master
```

GitHub 版本只包含可公开同步的代码、文档、配置和聚合证据，不包含原始数据、参考书和 checkpoint；本次四个 ZIP 是当前日期的离线备份，内容可能比尚未推送的 GitHub 提交更新。

如果不熟悉命令行，先把你当前 ZIP 的版本日期、已完成的课程/任务和遇到的错误发给负责人，由负责人统一合并。

## 你应当记住的边界

- 当前正式证据是 H/V UAV/背景内部开发结果，不是空飘球载荷识别完成证明；
- `complete-scan` BC-DPG 是离线上限，不等于严格实时；
- 当前缺少跨日期、跨场地的 locked blind test；
- 未确认绝对幅相标定前，不把 H/V 写成完整全极化物理结论；
- 任何新实验都要记录数据版本、分组、配置、代码版本、阈值和结果。

遇到疑问时，优先在 `PROJECT_CONTROL/TASK_BOARD.md` 找任务 ID，再在组内提出问题；不要通过修改旧结果来“修正”项目状态。
