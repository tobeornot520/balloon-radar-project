# Stage 4六折扩展预注册说明

固定问题：候选区域约束能否在六个扫描组折中稳定降低Power2虚警并保持联合Pd？局部RI4是否在ROI power control之外提供增量？

固定设置：seed=42；ROI=11×9；候选来自冻结Power2预测；suppression-only；位置冻结；样本独立；无扫描上下文；主评价使用原始Power2阈值；禁止依据六折测试集重新调参。
