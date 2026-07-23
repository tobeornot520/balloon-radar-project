# 显式极化特征审计 V1

- Manifest：`/home/tobeornot8259748/projects/balloon_radar_project/results/data_audit/dataset_v4_multifold/fold_01_manifest.csv`
- 选取样本：120
- 成功处理：120
- 错误：0
- IQ形状：{'(128, 100)': 120}

## 已计算的显式极化量

- H/V复数RD与功率；
- 局部功率比 `ZDR-like`；
- 邻域估计的共极化相关系数幅值 `rho_HV`；
- 相对差分相位的正弦/余弦表示；
- 归一化 Stokes-like S1/S2/S3。

注意：尚未提供通道幅相标定信息，因此只能使用“relative ZDR-like”和“relative differential phase”命名，不能直接宣称为绝对气象雷达ZDR或PhiDP。

## 单特征区分度前10项

| feature                |   background_mean |   target_mean |   cohens_d_target_minus_background |   auc_target_high |   orientation_free_auc |
|:-----------------------|------------------:|--------------:|-----------------------------------:|------------------:|-----------------------:|
| peak_rho_median        |            0.9638 |        0.9933 |                             0.4400 |            0.7969 |                 0.7969 |
| peak_rho_mean          |            0.9581 |        0.9907 |                             0.4839 |            0.7403 |                 0.7403 |
| peak_rho_p10           |            0.9135 |        0.9752 |                             0.4378 |            0.7281 |                 0.7281 |
| peak_phase_resultant   |            0.9560 |        0.9981 |                             0.5224 |            0.7272 |                 0.7272 |
| global_phase_resultant |            0.2615 |        0.0773 |                            -1.3705 |            0.3003 |                 0.6997 |
| peak_zdr_std           |            0.7121 |        0.3813 |                            -0.2527 |            0.3036 |                 0.6964 |
| peak_stokes_s2_mean    |            0.5004 |        0.6961 |                             0.6989 |            0.6675 |                 0.6675 |
| global_rho_median      |            0.5238 |        0.3352 |                            -1.3539 |            0.3325 |                 0.6675 |
| global_rho_mean        |            0.5262 |        0.3721 |                            -1.4699 |            0.3558 |                 0.6442 |
| peak_phase_cos_mean    |            0.5198 |        0.7170 |                             0.6328 |            0.6328 |                 0.6328 |

## 多域路线判定

1. 当前DPG/BC-DPG只使用H/V功率RD与网络隐特征，没有显式极化分支；
2. 分类数据集中的旧`polar5`只是早期逐点特征，其`corr`实质更接近相位差余弦，不是真正邻域相关系数；
3. 本审计通过后，可依次比较 Power2、RI4、Polar6、RI8；
4. 时频/微多普勒分支不能从单帧128脉冲数据直接宣称完成，需要连续慢时间序列、PRF及目标距离门对齐。
