from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat


# 项目中的原始 IQ 数据目录
DATA_DIR = Path("data/raw/IQ_Data")

# 检查结果保存位置
OUTPUT_DIR = Path("results/tables")
OUTPUT_FILE = OUTPUT_DIR / "mat_data_audit.csv"


def inspect_mat_file(mat_path: Path) -> dict:
    """检查一个 MAT 文件中的 H/V 复数 IQ 数据。"""

    result = {
        "filename": mat_path.name,
        "filepath": str(mat_path),
        "read_success": False,
        "has_h": False,
        "has_v": False,
        "h_shape": "",
        "v_shape": "",
        "h_dtype": "",
        "v_dtype": "",
        "h_is_complex": False,
        "v_is_complex": False,
        "h_has_nan": False,
        "v_has_nan": False,
        "h_has_inf": False,
        "v_has_inf": False,
        "shape_valid": False,
        "data_valid": False,
        "error": "",
    }

    try:
        mat_data = loadmat(mat_path)
        result["read_success"] = True

        result["has_h"] = "local_data_H" in mat_data
        result["has_v"] = "local_data_V" in mat_data

        if not result["has_h"] or not result["has_v"]:
            result["error"] = "缺少 local_data_H 或 local_data_V"
            return result

        h_data = mat_data["local_data_H"]
        v_data = mat_data["local_data_V"]

        result["h_shape"] = str(h_data.shape)
        result["v_shape"] = str(v_data.shape)

        result["h_dtype"] = str(h_data.dtype)
        result["v_dtype"] = str(v_data.dtype)

        result["h_is_complex"] = bool(np.iscomplexobj(h_data))
        result["v_is_complex"] = bool(np.iscomplexobj(v_data))

        result["h_has_nan"] = bool(np.isnan(h_data).any())
        result["v_has_nan"] = bool(np.isnan(v_data).any())

        result["h_has_inf"] = bool(np.isinf(h_data).any())
        result["v_has_inf"] = bool(np.isinf(v_data).any())

        result["shape_valid"] = (
            h_data.shape == (128, 100)
            and v_data.shape == (128, 100)
        )

        result["data_valid"] = (
            result["shape_valid"]
            and result["h_is_complex"]
            and result["v_is_complex"]
            and not result["h_has_nan"]
            and not result["v_has_nan"]
            and not result["h_has_inf"]
            and not result["v_has_inf"]
        )

        if not result["data_valid"]:
            result["error"] = "尺寸、复数类型或数值有效性检查未通过"

    except Exception as exc:
        result["error"] = str(exc)

    return result


def main() -> None:
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"数据目录不存在：{DATA_DIR.resolve()}")

    mat_files = sorted(DATA_DIR.glob("*.mat"))

    if not mat_files:
        raise FileNotFoundError(f"目录中没有找到 MAT 文件：{DATA_DIR.resolve()}")

    print(f"共发现 {len(mat_files)} 个 MAT 文件，开始检查。")

    records = []

    for index, mat_path in enumerate(mat_files, start=1):
        result = inspect_mat_file(mat_path)
        records.append(result)

        status = "通过" if result["data_valid"] else "异常"

        print(
            f"[{index:03d}/{len(mat_files):03d}] "
            f"{status}：{mat_path.name}"
        )

    output_df = pd.DataFrame(records)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    total_count = len(output_df)
    valid_count = int(output_df["data_valid"].sum())
    invalid_count = total_count - valid_count

    print("\n========== 检查结果 ==========")
    print(f"文件总数：{total_count}")
    print(f"正常文件：{valid_count}")
    print(f"异常文件：{invalid_count}")
    print(f"结果表格：{OUTPUT_FILE.resolve()}")

    if invalid_count > 0:
        print("\n异常文件：")
        invalid_df = output_df.loc[
            ~output_df["data_valid"],
            ["filename", "error"],
        ]
        print(invalid_df.to_string(index=False))


if __name__ == "__main__":
    main()