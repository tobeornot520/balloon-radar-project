from pathlib import Path
import re

import numpy as np
import pandas as pd


# ==============================
# 文件路径
# ==============================

IQ_DIR = Path("data/raw/IQ_Data")
LABEL_DIR = Path("data/raw/Labels")

AUDIT_OUTPUT = Path("results/tables/label_data_audit.csv")
SAMPLES_OUTPUT = Path("data/metadata/samples.csv")


# ==============================
# 与 MATLAB 脚本保持一致的雷达参数
# ==============================

C = 3.0e8
FC = 9300.0e6
WAVELENGTH = C / FC

PRF = 2900.0 / 2.0

NUM_PULSES = 128
NUM_GATES = 100
RANGE_RESOLUTION_M = 30.0

DOPPLER_BINS = np.arange(
    -NUM_PULSES // 2,
    NUM_PULSES // 2,
)

VELOCITY_AXIS = (
    -DOPPLER_BINS
    * WAVELENGTH
    * PRF
    / (2.0 * NUM_PULSES)
)

RANGE_AXIS = (
    np.arange(1, NUM_GATES + 1)
    * RANGE_RESOLUTION_M
)


def read_text_file(path: Path) -> tuple[str, str]:
    """尝试使用常见编码读取标签文件。"""

    encodings = (
        "utf-8-sig",
        "utf-8",
        "gb18030",
    )

    for encoding in encodings:
        try:
            return path.read_text(encoding=encoding), encoding
        except UnicodeDecodeError:
            continue

    raise UnicodeDecodeError(
        "unknown",
        b"",
        0,
        1,
        f"无法识别文件编码：{path}",
    )


def parse_label_file(label_path: Path) -> dict:
    """读取并检查一个标签文件。"""

    result = {
        "sample_id": label_path.stem,
        "label_filename": label_path.name,
        "label_path": str(label_path),
        "mat_path": "",
        "read_success": False,
        "encoding": "",
        "source_file": "",
        "beam_layer": np.nan,
        "azimuth_deg": np.nan,
        "distance_m": np.nan,
        "velocity_mps": np.nan,
        "filename_source": "",
        "filename_beam": np.nan,
        "mat_exists": False,
        "source_matches_filename": False,
        "beam_matches_filename": False,
        "distance_in_axis": False,
        "velocity_in_axis": False,
        "range_index_0": np.nan,
        "range_gate_1": np.nan,
        "range_axis_value_m": np.nan,
        "range_quantization_error_m": np.nan,
        "velocity_index_0": np.nan,
        "doppler_bin": np.nan,
        "velocity_axis_value_mps": np.nan,
        "velocity_quantization_error_mps": np.nan,
        "all_valid": False,
        "error": "",
    }

    errors = []

    # 从文件名提取时间和波束层
    filename_match = re.fullmatch(
        r"(?P<source>\d{8}_\d{6})_beam(?P<beam>\d+)",
        label_path.stem,
    )

    if filename_match is None:
        errors.append("标签文件名格式不正确")
    else:
        result["filename_source"] = filename_match.group("source")
        result["filename_beam"] = int(filename_match.group("beam"))

    # 对应的 MAT 文件
    mat_path = IQ_DIR / f"{label_path.stem}.mat"
    result["mat_path"] = str(mat_path)
    result["mat_exists"] = mat_path.exists()

    if not result["mat_exists"]:
        errors.append("缺少对应的 MAT 文件")

    # 读取标签正文
    try:
        text, encoding = read_text_file(label_path)
        result["read_success"] = True
        result["encoding"] = encoding
    except Exception as exc:
        errors.append(f"读取失败：{exc}")
        result["error"] = "；".join(errors)
        return result

    fields = {}

    for raw_line in text.splitlines():
        line = raw_line.strip()

        if not line or ":" not in line:
            continue

        key, value = line.split(":", 1)
        fields[key.strip()] = value.strip()

    required_fields = (
        "Source_File",
        "Beam_Layer",
        "Azimuth(deg)",
        "Distance(m)",
        "Velocity(m/s)",
    )

    missing_fields = [
        field
        for field in required_fields
        if field not in fields
    ]

    if missing_fields:
        errors.append(
            "缺少字段：" + "、".join(missing_fields)
        )

    try:
        if "Source_File" in fields:
            result["source_file"] = fields["Source_File"]

        if "Beam_Layer" in fields:
            result["beam_layer"] = int(fields["Beam_Layer"])

        if "Azimuth(deg)" in fields:
            result["azimuth_deg"] = float(fields["Azimuth(deg)"])

        if "Distance(m)" in fields:
            result["distance_m"] = float(fields["Distance(m)"])

        if "Velocity(m/s)" in fields:
            result["velocity_mps"] = float(fields["Velocity(m/s)"])

    except ValueError as exc:
        errors.append(f"字段数值格式错误：{exc}")

    # 检查文件名和标签内容是否一致
    if filename_match is not None:
        result["source_matches_filename"] = (
            result["source_file"]
            == result["filename_source"]
        )

        if not result["source_matches_filename"]:
            errors.append("Source_File 与文件名不一致")

        if not pd.isna(result["beam_layer"]):
            result["beam_matches_filename"] = (
                int(result["beam_layer"])
                == int(result["filename_beam"])
            )

            if not result["beam_matches_filename"]:
                errors.append("Beam_Layer 与文件名不一致")

    # 检查距离并映射到距离门
    if not pd.isna(result["distance_m"]):
        distance = float(result["distance_m"])

        result["distance_in_axis"] = bool(
            RANGE_AXIS.min()
            <= distance
            <= RANGE_AXIS.max()
        )

        if result["distance_in_axis"]:
            range_index = int(
                np.argmin(
                    np.abs(RANGE_AXIS - distance)
                )
            )

            exact_range = float(RANGE_AXIS[range_index])

            result["range_index_0"] = range_index
            result["range_gate_1"] = range_index + 1
            result["range_axis_value_m"] = exact_range

            result["range_quantization_error_m"] = (
                exact_range - distance
            )
        else:
            errors.append("目标距离超出距离轴范围")

    # 检查速度并映射到多普勒单元
    if not pd.isna(result["velocity_mps"]):
        velocity = float(result["velocity_mps"])

        result["velocity_in_axis"] = bool(
            VELOCITY_AXIS.min()
            <= velocity
            <= VELOCITY_AXIS.max()
        )

        if result["velocity_in_axis"]:
            velocity_index = int(
                np.argmin(
                    np.abs(VELOCITY_AXIS - velocity)
                )
            )

            exact_velocity = float(
                VELOCITY_AXIS[velocity_index]
            )

            result["velocity_index_0"] = velocity_index
            result["doppler_bin"] = int(
                DOPPLER_BINS[velocity_index]
            )

            result["velocity_axis_value_mps"] = (
                exact_velocity
            )

            result["velocity_quantization_error_mps"] = (
                exact_velocity - velocity
            )
        else:
            errors.append("目标速度超出速度轴范围")

    result["all_valid"] = len(errors) == 0
    result["error"] = "；".join(errors)

    return result


def build_samples_table(
    audit_df: pd.DataFrame,
) -> pd.DataFrame:
    """建立后续统一使用的样本清单。"""

    valid_df = audit_df.loc[
        audit_df["all_valid"]
    ].copy()

    samples_df = pd.DataFrame(
        {
            "sample_id": valid_df["sample_id"],
            "session_id": valid_df["source_file"],
            "timestamp": valid_df["source_file"],
            "mat_path": valid_df["mat_path"],
            "label_path": valid_df["label_path"],
            "target_present": 1,
            "target_class": "uav",
            "payload_present": "unknown",
            "payload_mass_g": np.nan,
            "payload_state": "unknown",
            "distance_m": valid_df["distance_m"],
            "velocity_mps": valid_df["velocity_mps"],
            "azimuth_deg": valid_df["azimuth_deg"],
            "beam_layer": valid_df["beam_layer"],
            "range_index_0": valid_df["range_index_0"],
            "range_gate_1": valid_df["range_gate_1"],
            "velocity_index_0": valid_df["velocity_index_0"],
            "doppler_bin": valid_df["doppler_bin"],
            "split": "",
        }
    )

    return samples_df


def main() -> None:
    if not IQ_DIR.exists():
        raise FileNotFoundError(
            f"IQ目录不存在：{IQ_DIR.resolve()}"
        )

    if not LABEL_DIR.exists():
        raise FileNotFoundError(
            f"标签目录不存在：{LABEL_DIR.resolve()}"
        )

    label_files = sorted(LABEL_DIR.glob("*.txt"))

    if not label_files:
        raise FileNotFoundError(
            f"没有找到标签文件：{LABEL_DIR.resolve()}"
        )

    print(f"发现 {len(label_files)} 个标签文件。")
    print(f"距离轴范围：{RANGE_AXIS.min():.1f}～"
          f"{RANGE_AXIS.max():.1f} m")
    print(f"速度轴范围：{VELOCITY_AXIS.min():.4f}～"
          f"{VELOCITY_AXIS.max():.4f} m/s")
    print(
        "速度分辨率："
        f"{abs(VELOCITY_AXIS[1] - VELOCITY_AXIS[0]):.6f} m/s"
    )
    print()

    records = []

    for index, label_path in enumerate(
        label_files,
        start=1,
    ):
        result = parse_label_file(label_path)
        records.append(result)

        status = (
            "通过"
            if result["all_valid"]
            else "异常"
        )

        print(
            f"[{index:03d}/{len(label_files):03d}] "
            f"{status}：{label_path.name}"
        )

    audit_df = pd.DataFrame(records)

    AUDIT_OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    audit_df.to_csv(
        AUDIT_OUTPUT,
        index=False,
        encoding="utf-8-sig",
    )

    samples_df = build_samples_table(audit_df)

    SAMPLES_OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    samples_df.to_csv(
        SAMPLES_OUTPUT,
        index=False,
        encoding="utf-8-sig",
    )

    total_count = len(audit_df)
    valid_count = int(audit_df["all_valid"].sum())
    invalid_count = total_count - valid_count

    print("\n========== 标签检查结果 ==========")
    print(f"标签文件总数：{total_count}")
    print(f"正常标签：{valid_count}")
    print(f"异常标签：{invalid_count}")
    print(f"标签检查表：{AUDIT_OUTPUT.resolve()}")
    print(f"样本总表：{SAMPLES_OUTPUT.resolve()}")

    if invalid_count > 0:
        print("\n异常文件：")

        invalid_df = audit_df.loc[
            ~audit_df["all_valid"],
            [
                "label_filename",
                "error",
            ],
        ]

        print(
            invalid_df.to_string(index=False)
        )

    if not samples_df.empty:
        first_sample = samples_df.iloc[0]

        print("\n========== 第一个样本的坐标映射 ==========")
        print(f"样本：{first_sample['sample_id']}")
        print(
            f"距离：{first_sample['distance_m']} m"
            f" → 第 {int(first_sample['range_gate_1'])} 个距离门"
            f" → Python下标 {int(first_sample['range_index_0'])}"
        )
        print(
            f"速度：{first_sample['velocity_mps']} m/s"
            f" → Python下标 {int(first_sample['velocity_index_0'])}"
            f" → 多普勒单元 {int(first_sample['doppler_bin'])}"
        )


if __name__ == "__main__":
    main()