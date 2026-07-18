from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd
from scipy.io import loadmat


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_METADATA = PROJECT_ROOT / "data" / "metadata" / "samples_split.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "analysis" / "hv_raw_peak_offsets"

NUM_PULSES = 128
NUM_GATES = 100
CHANNEL_VARIABLES = {
    "H": "local_data_H",
    "V": "local_data_V",
}


def configure_chinese_font() -> None:
    """尽量启用中文字体；找不到时仍可继续生成表格和图片。"""
    candidate_paths = [
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
        Path("/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc"),
        Path("/usr/share/fonts/truetype/arphic/uming.ttc"),
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/simhei.ttf"),
    ]

    for font_path in candidate_paths:
        if not font_path.exists():
            continue
        try:
            font_manager.fontManager.addfont(str(font_path))
            font_name = font_manager.FontProperties(
                fname=str(font_path)
            ).get_name()
            plt.rcParams["font.sans-serif"] = [font_name]
            plt.rcParams["axes.unicode_minus"] = False
            return
        except (RuntimeError, OSError):
            continue

    plt.rcParams["axes.unicode_minus"] = False


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "统计 H/V 原始 RD 峰值相对于标签的距离和速度偏移，"
            "并按波束、session、距离区间和速度区间汇总。"
        )
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=DEFAULT_METADATA,
        help="数据清单 CSV，默认 data/metadata/samples_split.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="输出目录",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=["train", "val", "test", "all"],
        default=["all"],
        help="分析哪些划分，默认全部样本",
    )
    parser.add_argument(
        "--local-range-radius",
        type=int,
        default=8,
        help="标签邻域峰搜索的距离半径，默认 8 门",
    )
    parser.add_argument(
        "--local-velocity-radius",
        type=int,
        default=3,
        help="标签邻域峰搜索的速度半径，默认 3 单元",
    )
    parser.add_argument(
        "--distance-bin-width",
        type=float,
        default=300.0,
        help="距离分组宽度，单位 m，默认 300",
    )
    parser.add_argument(
        "--velocity-bin-width",
        type=float,
        default=2.0,
        help="速度分组宽度，单位 m/s，默认 2",
    )
    parser.add_argument(
        "--demean",
        action="store_true",
        help="在多普勒 FFT 前沿慢时间去均值；默认关闭以匹配当前 Dataset",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="只分析前 N 个样本，便于快速测试",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="只输出表格，不生成图片",
    )
    return parser.parse_args()


def resolve_project_path(path: Path) -> Path:
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def validate_arguments(args: argparse.Namespace) -> None:
    if args.local_range_radius < 0:
        raise ValueError("local-range-radius 不能小于 0")
    if args.local_velocity_radius < 0:
        raise ValueError("local-velocity-radius 不能小于 0")
    if args.distance_bin_width <= 0:
        raise ValueError("distance-bin-width 必须大于 0")
    if args.velocity_bin_width <= 0:
        raise ValueError("velocity-bin-width 必须大于 0")
    if args.max_samples is not None and args.max_samples <= 0:
        raise ValueError("max-samples 必须大于 0")


def load_metadata(args: argparse.Namespace) -> pd.DataFrame:
    metadata_path = resolve_project_path(args.metadata)
    if not metadata_path.exists():
        raise FileNotFoundError(f"找不到数据清单：{metadata_path}")

    dataframe = pd.read_csv(metadata_path)
    required_columns = {
        "sample_id",
        "session_id",
        "mat_path",
        "distance_m",
        "velocity_mps",
        "beam_layer",
        "range_index_0",
        "velocity_index_0",
        "split",
    }
    missing_columns = required_columns - set(dataframe.columns)
    if missing_columns:
        raise KeyError(
            "数据清单缺少字段：" + ", ".join(sorted(missing_columns))
        )

    if "all" not in args.splits:
        dataframe = dataframe.loc[
            dataframe["split"].isin(args.splits)
        ].copy()

    dataframe = dataframe.sort_values(
        ["session_id", "beam_layer", "sample_id"]
    ).reset_index(drop=True)

    if args.max_samples is not None:
        dataframe = dataframe.iloc[: args.max_samples].copy()

    if dataframe.empty:
        raise ValueError("筛选后没有可分析样本")

    return dataframe


def load_complex_iq(mat_path: Path, variable_name: str) -> np.ndarray:
    if not mat_path.exists():
        raise FileNotFoundError(f"找不到 MAT 文件：{mat_path}")

    data = loadmat(mat_path, variable_names=[variable_name])
    if variable_name not in data:
        raise KeyError(f"{mat_path.name} 中没有变量 {variable_name}")

    iq = np.asarray(data[variable_name])
    if iq.shape != (NUM_PULSES, NUM_GATES):
        raise ValueError(
            f"{mat_path.name} 中 {variable_name} 尺寸为 {iq.shape}，"
            f"期望 {(NUM_PULSES, NUM_GATES)}"
        )
    if not np.iscomplexobj(iq):
        raise ValueError(f"{mat_path.name} 中 {variable_name} 不是复数 IQ")
    if not np.isfinite(iq).all():
        raise ValueError(f"{mat_path.name} 中 {variable_name} 存在 NaN 或 Inf")

    return iq


def calculate_rd_power(iq: np.ndarray, demean: bool) -> np.ndarray:
    working = np.asarray(iq)
    if demean:
        working = working - np.mean(working, axis=0, keepdims=True)

    window = np.hanning(NUM_PULSES).astype(np.float32)[:, None]
    rd_complex = np.fft.fftshift(
        np.fft.fft(working * window, axis=0),
        axes=0,
    )
    return (np.abs(rd_complex) ** 2).astype(np.float32, copy=False)


def power_to_db(power: np.ndarray) -> np.ndarray:
    return (10.0 * np.log10(np.asarray(power) + 1e-12)).astype(
        np.float32,
        copy=False,
    )


def peak_from_full_map(rd_db: np.ndarray) -> tuple[int, int, float]:
    flat_index = int(np.argmax(rd_db))
    velocity_index, range_index = np.unravel_index(
        flat_index,
        rd_db.shape,
    )
    return (
        int(velocity_index),
        int(range_index),
        float(rd_db[velocity_index, range_index]),
    )


def peak_from_label_neighborhood(
    rd_db: np.ndarray,
    true_velocity_index: int,
    true_range_index: int,
    velocity_radius: int,
    range_radius: int,
) -> tuple[int, int, float]:
    velocity_start = max(0, true_velocity_index - velocity_radius)
    velocity_stop = min(
        rd_db.shape[0],
        true_velocity_index + velocity_radius + 1,
    )
    range_start = max(0, true_range_index - range_radius)
    range_stop = min(
        rd_db.shape[1],
        true_range_index + range_radius + 1,
    )

    patch = rd_db[
        velocity_start:velocity_stop,
        range_start:range_stop,
    ]
    if patch.size == 0:
        raise RuntimeError("标签邻域搜索窗口为空")

    flat_index = int(np.argmax(patch))
    local_velocity, local_range = np.unravel_index(
        flat_index,
        patch.shape,
    )
    velocity_index = velocity_start + int(local_velocity)
    range_index = range_start + int(local_range)

    return (
        velocity_index,
        range_index,
        float(rd_db[velocity_index, range_index]),
    )


def analyze_channel(
    rd_db: np.ndarray,
    channel: str,
    true_velocity_index: int,
    true_range_index: int,
    velocity_radius: int,
    range_radius: int,
) -> dict[str, float | int | str]:
    (
        global_velocity,
        global_range,
        global_peak_db,
    ) = peak_from_full_map(rd_db)

    (
        local_velocity,
        local_range,
        local_peak_db,
    ) = peak_from_label_neighborhood(
        rd_db=rd_db,
        true_velocity_index=true_velocity_index,
        true_range_index=true_range_index,
        velocity_radius=velocity_radius,
        range_radius=range_radius,
    )

    label_db = float(rd_db[true_velocity_index, true_range_index])

    return {
        f"{channel}_global_velocity_index": global_velocity,
        f"{channel}_global_range_index": global_range,
        f"{channel}_global_velocity_offset": (
            global_velocity - true_velocity_index
        ),
        f"{channel}_global_range_offset": global_range - true_range_index,
        f"{channel}_global_peak_db": global_peak_db,
        f"{channel}_local_velocity_index": local_velocity,
        f"{channel}_local_range_index": local_range,
        f"{channel}_local_velocity_offset": local_velocity - true_velocity_index,
        f"{channel}_local_range_offset": local_range - true_range_index,
        f"{channel}_local_abs_velocity_offset": abs(
            local_velocity - true_velocity_index
        ),
        f"{channel}_local_abs_range_offset": abs(
            local_range - true_range_index
        ),
        f"{channel}_local_peak_db": local_peak_db,
        f"{channel}_label_db": label_db,
        f"{channel}_local_peak_minus_label_db": local_peak_db - label_db,
    }


def make_bin_edges(values: pd.Series, width: float) -> np.ndarray:
    minimum = float(values.min())
    maximum = float(values.max())
    start = math.floor(minimum / width) * width
    stop = math.ceil(maximum / width) * width
    if stop <= maximum:
        stop += width
    if stop <= start:
        stop = start + width
    return np.arange(start, stop + 0.5 * width, width, dtype=float)


def mode_value(series: pd.Series) -> float:
    mode = series.mode(dropna=True)
    if mode.empty:
        return float("nan")
    return float(mode.iloc[0])


def summarize_channel(
    dataframe: pd.DataFrame,
    channel: str,
    group_type: str,
    group_value: str,
) -> dict[str, object]:
    range_signed = dataframe[f"{channel}_local_range_offset"].astype(float)
    velocity_signed = dataframe[
        f"{channel}_local_velocity_offset"
    ].astype(float)
    range_absolute = range_signed.abs()
    velocity_absolute = velocity_signed.abs()

    return {
        "group_type": group_type,
        "group_value": group_value,
        "channel": channel,
        "sample_count": int(len(dataframe)),
        "range_signed_mean": float(range_signed.mean()),
        "range_signed_median": float(range_signed.median()),
        "range_signed_mode": mode_value(range_signed),
        "range_signed_std": float(range_signed.std(ddof=0)),
        "range_mae": float(range_absolute.mean()),
        "range_rmse": float(np.sqrt(np.mean(range_signed**2))),
        "range_within_0_rate": float((range_absolute <= 0).mean()),
        "range_within_1_rate": float((range_absolute <= 1).mean()),
        "range_within_2_rate": float((range_absolute <= 2).mean()),
        "range_within_3_rate": float((range_absolute <= 3).mean()),
        "velocity_signed_mean": float(velocity_signed.mean()),
        "velocity_signed_median": float(velocity_signed.median()),
        "velocity_signed_mode": mode_value(velocity_signed),
        "velocity_signed_std": float(velocity_signed.std(ddof=0)),
        "velocity_mae": float(velocity_absolute.mean()),
        "velocity_rmse": float(np.sqrt(np.mean(velocity_signed**2))),
        "velocity_within_0_rate": float((velocity_absolute <= 0).mean()),
        "velocity_within_1_rate": float((velocity_absolute <= 1).mean()),
        "velocity_within_2_rate": float((velocity_absolute <= 2).mean()),
        "velocity_within_3_rate": float((velocity_absolute <= 3).mean()),
    }


def build_group_summary(
    dataframe: pd.DataFrame,
    group_column: str | None,
    group_type: str,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []

    if group_column is None:
        groups: Iterable[tuple[object, pd.DataFrame]] = [
            ("ALL", dataframe)
        ]
    else:
        groups = dataframe.groupby(group_column, observed=True, dropna=False)

    for group_value, group in groups:
        for channel in ["H", "V"]:
            records.append(
                summarize_channel(
                    dataframe=group,
                    channel=channel,
                    group_type=group_type,
                    group_value=str(group_value),
                )
            )

    return pd.DataFrame(records)


def save_histogram(
    dataframe: pd.DataFrame,
    column_h: str,
    column_v: str,
    xlabel: str,
    title: str,
    output_path: Path,
) -> None:
    values = np.concatenate(
        [
            dataframe[column_h].to_numpy(dtype=float),
            dataframe[column_v].to_numpy(dtype=float),
        ]
    )
    minimum = int(np.floor(np.min(values)))
    maximum = int(np.ceil(np.max(values)))
    bins = np.arange(minimum - 0.5, maximum + 1.5, 1.0)

    plt.figure(figsize=(10, 6))
    plt.hist(
        dataframe[column_h],
        bins=bins,
        alpha=0.65,
        label="H",
    )
    plt.hist(
        dataframe[column_v],
        bins=bins,
        alpha=0.65,
        label="V",
    )
    plt.axvline(0, linestyle="--", linewidth=1)
    plt.xlabel(xlabel)
    plt.ylabel("样本数")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_channel_difference_histogram(
    dataframe: pd.DataFrame,
    output_path: Path,
) -> None:
    values = dataframe["H_minus_V_local_range_index"].to_numpy(dtype=float)
    minimum = int(np.floor(np.min(values)))
    maximum = int(np.ceil(np.max(values)))
    bins = np.arange(minimum - 0.5, maximum + 1.5, 1.0)

    plt.figure(figsize=(10, 6))
    plt.hist(values, bins=bins)
    plt.axvline(0, linestyle="--", linewidth=1)
    plt.xlabel("H 局部峰距离下标 - V 局部峰距离下标（门）")
    plt.ylabel("样本数")
    plt.title("H/V 目标邻域峰距离差分布")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_beam_mean_plot(
    beam_summary: pd.DataFrame,
    output_path: Path,
) -> None:
    pivot = beam_summary.pivot(
        index="group_value",
        columns="channel",
        values="range_signed_mean",
    ).sort_index()

    positions = np.arange(len(pivot.index), dtype=float)
    width = 0.36

    plt.figure(figsize=(10, 6))
    plt.bar(positions - width / 2, pivot["H"], width=width, label="H")
    plt.bar(positions + width / 2, pivot["V"], width=width, label="V")
    plt.axhline(0, linestyle="--", linewidth=1)
    plt.xticks(positions, pivot.index)
    plt.xlabel("波束层")
    plt.ylabel("平均有符号距离偏移（门）")
    plt.title("不同波束的 H/V 目标邻域峰距离偏移")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_scatter_plot(
    dataframe: pd.DataFrame,
    output_path: Path,
) -> None:
    x = dataframe["V_local_range_offset"].to_numpy(dtype=float)
    y = dataframe["H_local_range_offset"].to_numpy(dtype=float)
    lower = float(min(x.min(), y.min())) - 0.5
    upper = float(max(x.max(), y.max())) + 0.5

    plt.figure(figsize=(7, 7))
    plt.scatter(x, y, alpha=0.65)
    plt.plot([lower, upper], [lower, upper], linestyle="--", linewidth=1)
    plt.axhline(0, linestyle=":", linewidth=1)
    plt.axvline(0, linestyle=":", linewidth=1)
    plt.xlim(lower, upper)
    plt.ylim(lower, upper)
    plt.xlabel("V 目标邻域峰距离偏移（门）")
    plt.ylabel("H 目标邻域峰距离偏移（门）")
    plt.title("逐样本 H/V 距离偏移配对比较")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def print_terminal_summary(
    detail_df: pd.DataFrame,
    overall_summary: pd.DataFrame,
    output_dir: Path,
) -> None:
    print("\n" + "=" * 78)
    print("H/V 原始 RD 目标邻域峰偏移汇总")
    print("=" * 78)
    print(f"样本数：{len(detail_df)}")

    for channel in ["H", "V"]:
        row = overall_summary.loc[
            overall_summary["channel"] == channel
        ].iloc[0]
        print(f"\n{channel} 通道")
        print(f"  平均有符号距离偏移：{row['range_signed_mean']:.3f} 门")
        print(f"  距离偏移中位数：{row['range_signed_median']:.3f} 门")
        print(f"  距离偏移众数：{row['range_signed_mode']:.0f} 门")
        print(f"  距离 MAE：{row['range_mae']:.3f} 门")
        print(f"  距离 ±1 门比例：{row['range_within_1_rate']:.2%}")
        print(f"  平均有符号速度偏移：{row['velocity_signed_mean']:.3f} 单元")
        print(f"  速度 MAE：{row['velocity_mae']:.3f} 单元")
        print(f"  速度 ±1 单元比例：{row['velocity_within_1_rate']:.2%}")

    hv_difference = detail_df["H_minus_V_local_range_index"]
    print("\nH/V 通道间局部峰距离差")
    print(f"  平均值：{hv_difference.mean():.3f} 门")
    print(f"  中位数：{hv_difference.median():.3f} 门")
    print(f"  众数：{mode_value(hv_difference):.0f} 门")
    print(f"  H 与 V 同门比例：{(hv_difference == 0).mean():.2%}")
    print(f"  |H-V| ≤ 1 门比例：{(hv_difference.abs() <= 1).mean():.2%}")
    print(f"\n输出目录：{output_dir}")


def main() -> None:
    args = parse_arguments()
    validate_arguments(args)
    configure_chinese_font()

    metadata_df = load_metadata(args)
    output_dir = resolve_project_path(args.output_dir)
    table_dir = output_dir / "tables"
    figure_dir = output_dir / "figures"
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, object]] = []
    error_records: list[dict[str, str]] = []

    print("=" * 78)
    print("开始分析 H/V 原始 RD 峰值偏移")
    print(f"样本数：{len(metadata_df)}")
    print(
        "标签邻域搜索窗口："
        f"速度 ±{args.local_velocity_radius}，"
        f"距离 ±{args.local_range_radius}"
    )
    print(f"慢时间去均值：{'开启' if args.demean else '关闭'}")
    print("=" * 78)

    for index, row in metadata_df.iterrows():
        sample_id = str(row["sample_id"])
        try:
            mat_path = resolve_project_path(Path(str(row["mat_path"])))
            true_range_index = int(row["range_index_0"])
            true_velocity_index = int(row["velocity_index_0"])

            if not 0 <= true_range_index < NUM_GATES:
                raise IndexError(f"标签距离下标越界：{true_range_index}")
            if not 0 <= true_velocity_index < NUM_PULSES:
                raise IndexError(f"标签速度下标越界：{true_velocity_index}")

            record: dict[str, object] = {
                "sample_id": sample_id,
                "session_id": str(row["session_id"]),
                "split": str(row["split"]),
                "beam_layer": int(row["beam_layer"]),
                "distance_m": float(row["distance_m"]),
                "velocity_mps": float(row["velocity_mps"]),
                "true_range_index": true_range_index,
                "true_velocity_index": true_velocity_index,
                "mat_path": str(row["mat_path"]),
            }

            for channel, variable_name in CHANNEL_VARIABLES.items():
                iq = load_complex_iq(mat_path, variable_name)
                rd_power = calculate_rd_power(iq, demean=args.demean)
                rd_db = power_to_db(rd_power)
                record.update(
                    analyze_channel(
                        rd_db=rd_db,
                        channel=channel,
                        true_velocity_index=true_velocity_index,
                        true_range_index=true_range_index,
                        velocity_radius=args.local_velocity_radius,
                        range_radius=args.local_range_radius,
                    )
                )

            record["H_minus_V_local_range_index"] = (
                int(record["H_local_range_index"])
                - int(record["V_local_range_index"])
            )
            record["H_minus_V_local_velocity_index"] = (
                int(record["H_local_velocity_index"])
                - int(record["V_local_velocity_index"])
            )
            record["H_minus_V_local_peak_db"] = (
                float(record["H_local_peak_db"])
                - float(record["V_local_peak_db"])
            )
            records.append(record)

        except Exception as error:  # noqa: BLE001
            error_records.append(
                {
                    "sample_id": sample_id,
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                }
            )

        if (index + 1) % 50 == 0 or index + 1 == len(metadata_df):
            print(
                f"已处理 {index + 1}/{len(metadata_df)}，"
                f"成功 {len(records)}，失败 {len(error_records)}"
            )

    if not records:
        raise RuntimeError("没有任何样本成功完成分析")

    detail_df = pd.DataFrame(records)

    distance_edges = make_bin_edges(
        detail_df["distance_m"],
        args.distance_bin_width,
    )
    velocity_edges = make_bin_edges(
        detail_df["velocity_mps"],
        args.velocity_bin_width,
    )
    detail_df["distance_bin"] = pd.cut(
        detail_df["distance_m"],
        bins=distance_edges,
        right=False,
        include_lowest=True,
    ).astype(str)
    detail_df["velocity_bin"] = pd.cut(
        detail_df["velocity_mps"],
        bins=velocity_edges,
        right=False,
        include_lowest=True,
    ).astype(str)

    details_path = table_dir / "hv_raw_peak_offsets.csv"
    errors_path = table_dir / "hv_raw_peak_errors.csv"
    detail_df.to_csv(details_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(error_records).to_csv(
        errors_path,
        index=False,
        encoding="utf-8-sig",
    )

    summaries = {
        "overall": build_group_summary(detail_df, None, "overall"),
        "split": build_group_summary(detail_df, "split", "split"),
        "beam": build_group_summary(
            detail_df,
            "beam_layer",
            "beam_layer",
        ),
        "session": build_group_summary(
            detail_df,
            "session_id",
            "session_id",
        ),
        "distance_bin": build_group_summary(
            detail_df,
            "distance_bin",
            "distance_bin",
        ),
        "velocity_bin": build_group_summary(
            detail_df,
            "velocity_bin",
            "velocity_bin",
        ),
    }

    for name, summary_df in summaries.items():
        summary_df.to_csv(
            table_dir / f"summary_by_{name}.csv",
            index=False,
            encoding="utf-8-sig",
        )

    hv_difference = detail_df["H_minus_V_local_range_index"]
    summary_json = {
        "configuration": {
            "metadata": str(resolve_project_path(args.metadata)),
            "sample_count": int(len(detail_df)),
            "failed_sample_count": int(len(error_records)),
            "local_range_radius": args.local_range_radius,
            "local_velocity_radius": args.local_velocity_radius,
            "demean": bool(args.demean),
            "distance_bin_width_m": args.distance_bin_width,
            "velocity_bin_width_mps": args.velocity_bin_width,
            "splits": args.splits,
        },
        "channel_difference": {
            "H_minus_V_range_mean": float(hv_difference.mean()),
            "H_minus_V_range_median": float(hv_difference.median()),
            "H_minus_V_range_mode": mode_value(hv_difference),
            "same_range_gate_rate": float((hv_difference == 0).mean()),
            "within_1_range_gate_rate": float(
                (hv_difference.abs() <= 1).mean()
            ),
            "within_2_range_gate_rate": float(
                (hv_difference.abs() <= 2).mean()
            ),
        },
        "overall": summaries["overall"].to_dict(orient="records"),
    }
    (table_dir / "summary.json").write_text(
        json.dumps(summary_json, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if not args.no_plots:
        save_histogram(
            dataframe=detail_df,
            column_h="H_local_range_offset",
            column_v="V_local_range_offset",
            xlabel="目标邻域峰距离下标 - 标签距离下标（门）",
            title="H/V 原始 RD 目标邻域峰距离偏移",
            output_path=figure_dir / "local_range_offset_hist.png",
        )
        save_histogram(
            dataframe=detail_df,
            column_h="H_local_velocity_offset",
            column_v="V_local_velocity_offset",
            xlabel="目标邻域峰速度下标 - 标签速度下标（单元）",
            title="H/V 原始 RD 目标邻域峰速度偏移",
            output_path=figure_dir / "local_velocity_offset_hist.png",
        )
        save_channel_difference_histogram(
            detail_df,
            figure_dir / "H_minus_V_range_difference_hist.png",
        )
        save_beam_mean_plot(
            summaries["beam"],
            figure_dir / "beam_mean_range_offset.png",
        )
        save_scatter_plot(
            detail_df,
            figure_dir / "H_V_range_offset_scatter.png",
        )

    print_terminal_summary(
        detail_df=detail_df,
        overall_summary=summaries["overall"],
        output_dir=output_dir,
    )

    if error_records:
        print(
            f"\n警告：有 {len(error_records)} 个样本失败，"
            f"详情见 {errors_path}"
        )


if __name__ == "__main__":
    main()
