# Tian 2024 论文复现专题

这个文件夹只保存 Tian 2024 复现专题的索引和独有结果包。正式研究文档统一维护在
`docs/`，原论文统一保存在 `paper/references/`，不在专题目录重复复制。

专题结论：当前完成的是方法级实现、本地迁移和失败机制诊断，不是论文数值级复现成功。

## 推荐阅读顺序

1. [`Tian 等 - 2024 - Fully Convolutional Network-Based Fast UAV Detection in Pulse Doppler Radar.pdf`](../references/Tian%20等%20-%202024%20-%20Fully%20Convolutional%20Network-Based%20Fast%20UAV%20Detection%20in%20Pulse%20Doppler%20Radar.pdf)
   - 原论文。先看摘要、方法总图、Table I、PIR/MDP 和实验结论。
2. [`TIAN_2024_PAPER_GUIDE_AND_ORAL_EXAM_ZH.md`](../../docs/TIAN_2024_PAPER_GUIDE_AND_ORAL_EXAM_ZH.md)
   - 按页解释论文、公式、网络结构、指标和论文中的歧义。
3. [`TIAN_FCN_REPRODUCTION_PROTOCOL.md`](../../docs/TIAN_FCN_REPRODUCTION_PROTOCOL.md)
   - 说明哪些内容属于论文方法，哪些内容属于本地迁移，以及当前复现门禁。
4. [`TIAN_FCN_REPRODUCTION_CONDITIONS_REQUEST.md`](../../docs/TIAN_FCN_REPRODUCTION_CONDITIONS_REQUEST.md)
   - 列出需要向学长确认或索取的最小复现材料。
5. [`TIAN_REPRODUCTION_FAILURE_AND_ALTERNATIVES_20260803.md`](../../docs/TIAN_REPRODUCTION_FAILURE_AND_ALTERNATIVES_20260803.md)
   - 当前正式失败结论、缺失条件、证据边界和替代路线。
6. [`TIAN_FCN_FOLD1_DIAGNOSTIC_CONCLUSION.md`](../../docs/TIAN_FCN_FOLD1_DIAGNOSTIC_CONCLUSION.md)
   - Fold 1 的 point-GT 诊断结果。
7. [`TIAN_FCN_FOLD1_COMPONENT_MECHANISM.md`](../../docs/TIAN_FCN_FOLD1_COMPONENT_MECHANISM.md)
   - 固定速度模板、MDP 选偏和两项负监督消融的详细证据。
8. [`TIAN_REPRODUCTION_FAILURE_SENIOR_DEFENSE_ZH.md`](../../docs/TIAN_REPRODUCTION_FAILURE_SENIOR_DEFENSE_ZH.md)
   - 面向学长交流的完整问答稿，包含大一学生版表达。
9. [`07_QUESTIONS_FOR_SENIOR_ZH.md`](../../docs/share/07_QUESTIONS_FOR_SENIOR_ZH.md)
   - 可以直接发送或照着提问的问题清单。
10. [`08_DATA_REQUEST_CHECKLIST_ZH.md`](../../docs/share/08_DATA_REQUEST_CHECKLIST_ZH.md)
    - 输入、预处理张量、标签、输出、配置和数据划分的最小需求。
11. `reproduction_results_package/README.md`
    - 复现结果、四张实际结果图、完整预测表、关键证据和简单失败原因说明的轻量整合包。
## 维护规则

- 论文、教材等全文只放在 `paper/references/`；
- 当前结论、协议和问答只在 `docs/` 维护；
- 本目录仅接收该专题独有的结果整理，不保存上述文件的副本；
- `reproduction_results_package/` 中的逐样本表属于内部诊断材料，不进入普通成员分享包。
