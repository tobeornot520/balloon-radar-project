# 外场 H/V 复数 IQ 完整性探针 V1

更新时间：2026-08-06

## 1. 解决什么问题

`validate_data_collection_manifest.py --check-files` 能确认 manifest 引用的文件存在，但不会打开
MAT 验证内容。外场 capability 与 calibration 门要求我们进一步确认：H/V 变量确实存在、均为
复数 IQ、形状符合设备配置、没有 NaN/Inf，并且 I/Q 分量不是常数。

`audit_field_iq_integrity_v1.py` 补上这一只读检查。它支持 scipy 可读的 MATLAB v5 文件和
HDF5/MATLAB v7.3 文件；v7.3 会恢复 MATLAB 逻辑轴顺序，不直接拿 HDF5 反向维度与设备
形状比较。脚本不修改原始数据，也不复制 IQ 到输出目录。

## 2. 使用前先填写设备合同

仓库提供
`configs/field_iq_probe_contract_template_v1.json`。先复制到受控证据目录，再根据设备说明填写：

- `h_variable`、`v_variable`：真实 H/V 变量名；
- `expected_ndim`：数组维数；
- `expected_shape`：设备配置下每个文件的期望形状；
- 如设备量化或输出约定不同，再经负责人审核后建立新版本合同。

模板故意把 `expected_shape` 留为 `null`。在该状态下，即使文件确实包含可变的复数 H/V，
结果也只能是 `BLOCKED_EXPECTED_SHAPE`，不能直接通过 `CAL_COMPLEX_IQ`。

## 3. 命令

```bash
conda run -n radar-torch python scripts/audit_field_iq_integrity_v1.py \
  /controlled/collection/manifests/capability_sample.csv \
  --data-root /controlled/collection \
  --contract /controlled/field_evidence/field_iq_probe_contract_device_v1.json \
  --output-dir results/data_audit/field_iq_integrity_device_v1
```

manifest 至少需要 `iq_path` 列，路径必须相对 `data-root`，不能包含 `..`，V1 只接受 `.mat`。
正式 dry run/Pilot 仍须使用完整 40 列 manifest 再通过 capture/causal 合同。

## 4. 输出与判定

- `file_audit_local.csv`：逐文件相对路径、SHA、形状、dtype、I/Q 标准差和错误码；仅本地；
- `shape_dtype_summary.csv`：不含文件名的聚合形状与 dtype 表；
- `summary.json`：门状态、文件计数、输入合同哈希和声明边界；
- `README.md`：结果说明。

状态含义：

| 状态 | 含义 |
|---|---|
| `FAIL` | 至少一个文件缺变量、非复数、非有限、无 I/Q 变化、形状不符或不可读 |
| `BLOCKED_EXPECTED_SHAPE` | 基础复数内容通过，但设备期望形状尚未配置 |
| `PASS_FILE_CONTENT_ONLY` | 所有引用文件通过内容和已配置形状检查 |

只有 `PASS_FILE_CONTENT_ONLY` 才能作为 `CAL_COMPLEX_IQ` 的一部分证据；它仍不是整个
calibration gate 的 PASS。

## 5. 不能由该探针证明的事实

即使文件内容全部通过，以下项目仍保持阻塞：

- H/V 变量是否与真实天线通道映射一致；
- H/V 是否同时采样或保持相干时序；
- 相对幅度、相对相位是否经过参考目标标定；
- PRF、载频、距离门和 Doppler bin 的物理映射；
- 文件之间是否属于独立 session；
- 数据是否足以训练模型或建立部署性能。

这些事实必须分别由设备说明、同步记录、参考目标测量、配置导出和采集日志提供，不能从
两个数组“看起来相关”反推。
