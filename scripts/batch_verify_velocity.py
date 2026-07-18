import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from plot_rd_with_label import (
    PROJECT_ROOT,
    MAT_DIR,
    LABEL_DIR,
    load_config,
    parse_label,
    load_iq,
    generate_rd,
    build_physical_rd,
    plot_rd,
)


BATCH_FIGURE_DIR = (
    PROJECT_ROOT
    / "results"
    / "figures"
    / "velocity_direction_check"
)

SUMMARY_PATH = (
    PROJECT_ROOT
    / "results"
    / "tables"
    / "velocity_direction_check.csv"
)


def extract_timestamp(stem):
    """20260202_144238_beam1 -> 20260202_144238"""
    if "_beam" not in stem:
        raise ValueError(f"无法解析样本名：{stem}")

    return stem.rsplit("_beam", 1)[0]


def extract_beam(stem):
    """20260202_144238_beam1 -> 1"""
    return int(stem.rsplit("_beam", 1)[1])


def choose_evenly(items, count):
    """
    从排序后的列表中均匀抽取。

    count <= 0 表示全部选择。
    """
    items = list(items)

    if count <= 0 or count >= len(items):
        return items

    raw_indices = np.linspace(
        0,
        len(items) - 1,
        count,
    )

    indices = sorted(
        set(int(round(value)) for value in raw_indices)
    )

    return [items[index] for index in indices]


def local_top_score(
    rd_map,
    velocity_index,
    range_index,
    velocity_radius=2,
    range_radius=2,
):
    """
    在目标附近的小区域内取最强5个像素的平均dB值。

    相比只取单个最大值，对一格坐标误差更稳健。
    """
    v0 = max(0, velocity_index - velocity_radius)
    v1 = min(rd_map.shape[0], velocity_index + velocity_radius + 1)

    r0 = max(0, range_index - range_radius)
    r1 = min(rd_map.shape[1], range_index + range_radius + 1)

    patch = rd_map[v0:v1, r0:r1].ravel()

    if patch.size == 0:
        return float("nan")

    count = min(5, patch.size)
    strongest = np.sort(patch)[-count:]

    return float(np.mean(strongest))


def classify_vote(
    label_velocity,
    velocity_resolution,
    score_difference,
):
    """
    score_difference =
        标签速度位置得分 - 镜像速度位置得分
    """
    if abs(label_velocity) <= velocity_resolution:
        return "near_zero"

    if score_difference >= 3.0:
        return "supports_current_sign"

    if score_difference <= -3.0:
        return "supports_opposite_sign"

    return "uncertain"


def main():
    parser = argparse.ArgumentParser(
        description="批量验证多普勒速度方向"
    )
    parser.add_argument(
        "--num-times",
        type=int,
        default=10,
        help="抽取时间点数量；0表示全部时间点",
    )
    parser.add_argument(
        "--beams-per-time",
        type=int,
        default=2,
        help="每个时间点抽取波束数；0表示全部波束",
    )
    parser.add_argument(
        "--channels",
        nargs="+",
        choices=["H", "V"],
        default=["H", "V"],
        help="需要检查的通道，默认H和V",
    )
    parser.add_argument(
        "--no-demean",
        action="store_true",
        help="关闭慢时间去均值",
    )
    args = parser.parse_args()

    config = load_config()

    mat_files = sorted(MAT_DIR.glob("*.mat"))

    if not mat_files:
        raise FileNotFoundError(f"没有找到MAT文件：{MAT_DIR}")

    groups = defaultdict(list)

    for mat_path in mat_files:
        label_path = LABEL_DIR / f"{mat_path.stem}.txt"

        if not label_path.exists():
            print(f"⚠️ 缺少标签，跳过：{mat_path.name}")
            continue

        timestamp = extract_timestamp(mat_path.stem)
        groups[timestamp].append(mat_path)

    timestamps = sorted(groups)

    selected_timestamps = choose_evenly(
        timestamps,
        args.num_times,
    )

    selected_samples = []

    for timestamp in selected_timestamps:
        samples = sorted(
            groups[timestamp],
            key=lambda path: extract_beam(path.stem),
        )

        selected_samples.extend(
            choose_evenly(
                samples,
                args.beams_per_time,
            )
        )

    clutter_method = (
        "none"
        if args.no_demean
        else config["clutter_suppression"]
    )

    BATCH_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)

    records = []

    print("=" * 72)
    print(f"抽取时间点：{len(selected_timestamps)}")
    print(f"抽取MAT样本：{len(selected_samples)}")
    print(f"检查通道：{args.channels}")
    print(
        f"预计生成图片："
        f"{len(selected_samples) * len(args.channels)} 张"
    )
    print("=" * 72)

    for sample_number, mat_path in enumerate(
        selected_samples,
        start=1,
    ):
        label_path = LABEL_DIR / f"{mat_path.stem}.txt"
        label_info = parse_label(label_path)

        for channel in args.channels:
            iq = load_iq(
                mat_path,
                channel,
                config["n_pulses"],
                config["n_range"],
            )

            rd_db = generate_rd(
                iq,
                clutter_method,
            )

            rd_physical, velocity_axis = build_physical_rd(
                rd_db,
                config["fc"],
                config["prf"],
            )

            output_path = (
                BATCH_FIGURE_DIR
                / f"{mat_path.stem}_{channel}_rd.png"
            )

            plot_rd(
                rd_physical,
                velocity_axis,
                label_info,
                config["range_res"],
                output_path,
                channel,
            )

            range_centers = (
                np.arange(config["n_range"])
                * config["range_res"]
            )

            range_index = int(
                np.argmin(
                    np.abs(
                        range_centers
                        - label_info["distance"]
                    )
                )
            )

            label_velocity_index = int(
                np.argmin(
                    np.abs(
                        velocity_axis
                        - label_info["velocity"]
                    )
                )
            )

            mirrored_velocity = -label_info["velocity"]

            mirror_velocity_index = int(
                np.argmin(
                    np.abs(
                        velocity_axis
                        - mirrored_velocity
                    )
                )
            )

            label_score = local_top_score(
                rd_physical,
                label_velocity_index,
                range_index,
            )

            mirror_score = local_top_score(
                rd_physical,
                mirror_velocity_index,
                range_index,
            )

            score_difference = label_score - mirror_score

            velocity_resolution = float(
                np.median(np.diff(velocity_axis))
            )

            vote = classify_vote(
                label_info["velocity"],
                abs(velocity_resolution),
                score_difference,
            )

            records.append(
                {
                    "sample": mat_path.stem,
                    "timestamp": extract_timestamp(
                        mat_path.stem
                    ),
                    "beam": extract_beam(mat_path.stem),
                    "channel": channel,
                    "distance_m": label_info["distance"],
                    "velocity_mps": label_info["velocity"],
                    "range_index": range_index,
                    "velocity_index": label_velocity_index,
                    "label_score_db": round(label_score, 3),
                    "mirror_score_db": round(mirror_score, 3),
                    "delta_db": round(score_difference, 3),
                    "vote": vote,
                    "figure": str(
                        output_path.relative_to(PROJECT_ROOT)
                    ),
                }
            )

        print(
            f"[{sample_number}/{len(selected_samples)}] "
            f"完成 {mat_path.stem}"
        )

    dataframe = pd.DataFrame(records)
    dataframe.to_csv(
        SUMMARY_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    print()
    print("=" * 72)
    print("速度方向验证统计")
    print("=" * 72)

    counts = dataframe["vote"].value_counts()

    for name in [
        "supports_current_sign",
        "uncertain",
        "supports_opposite_sign",
        "near_zero",
    ]:
        print(f"{name:28s}: {int(counts.get(name, 0))}")

    print()
    print(f"图片目录：{BATCH_FIGURE_DIR}")
    print(f"汇总表格：{SUMMARY_PATH}")
    print()
    print("判定含义：")
    print("supports_current_sign  当前速度方向的能量更强")
    print("supports_opposite_sign 镜像速度方向的能量更强")
    print("uncertain              两边差异不足3 dB")
    print("near_zero              标签过于接近零速度，无法判断符号")
    print("=" * 72)


if __name__ == "__main__":
    main()
