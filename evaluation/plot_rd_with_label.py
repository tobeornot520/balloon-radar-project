import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import yaml
from scipy.io import loadmat


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "radar_config.yaml"
MAT_DIR = PROJECT_ROOT / "data" / "raw" / "IQ_Data"
LABEL_DIR = PROJECT_ROOT / "data" / "raw" / "Labels"
OUTPUT_DIR = PROJECT_ROOT / "results" / "figures"

# 已通过样本确认：FFT默认符号与标签速度定义相反。
# 这里只修改速度坐标和RD行排列，绝不修改标签值。
DOPPLER_SIGN = -1.0

# 批量比较时使用统一色条，避免不同图片自动缩放造成误判。
DISPLAY_DB_MIN = -60.0
DISPLAY_DB_MAX = 30.0


def load_config():
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"配置文件不存在：{CONFIG_PATH}")

    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    radar = config["radar"]
    processing = config.get("processing", {})

    return {
        "fc": float(radar["carrier_frequency_hz"]),
        "prf": float(radar["prf_hz"]),
        "n_pulses": int(radar["num_pulses"]),
        "n_range": int(radar["num_range_gates"]),
        "range_res": float(radar["range_resolution_m"]),
        "clutter_suppression": str(
            processing.get("clutter_suppression", "demean")
        ).lower(),
    }


def parse_label(label_path):
    if not label_path.exists():
        raise FileNotFoundError(f"标签不存在：{label_path}")

    info = {}

    with label_path.open("r", encoding="utf-8") as file:
        for line in file:
            if ":" not in line:
                continue

            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip()

            if key == "Source_File":
                info["source_file"] = value
            elif key == "Beam_Layer":
                info["beam"] = int(value)
            elif key == "Azimuth(deg)":
                info["azimuth"] = float(value)
            elif key == "Distance(m)":
                info["distance"] = float(value)
            elif key == "Velocity(m/s)":
                info["velocity"] = float(value)

    required = {"distance", "velocity"}
    missing = required - set(info)

    if missing:
        raise ValueError(f"{label_path.name} 缺少标签字段：{missing}")

    return info


def load_iq(mat_path, channel, n_pulses, n_range):
    mat = loadmat(mat_path)

    variable_name = {
        "H": "local_data_H",
        "V": "local_data_V",
    }[channel]

    if variable_name not in mat:
        raise KeyError(
            f"{mat_path.name} 中不存在 {variable_name}，"
            f"实际变量为：{list(mat.keys())}"
        )

    iq = np.asarray(mat[variable_name])
    expected_shape = (n_pulses, n_range)

    if iq.shape == expected_shape[::-1]:
        print(f"⚠️ 自动转置 IQ：{iq.shape} -> {expected_shape}")
        iq = iq.T
    elif iq.shape != expected_shape:
        raise ValueError(
            f"{mat_path.name} 的IQ尺寸为 {iq.shape}，"
            f"期望尺寸为 {expected_shape}"
        )

    return iq


def generate_rd(iq_data, clutter_suppression="demean"):
    iq_processed = iq_data.astype(np.complex64, copy=True)

    if clutter_suppression == "demean":
        # 沿慢时间去均值，压制零多普勒静止杂波。
        iq_processed -= iq_processed.mean(axis=0, keepdims=True)

    window = np.hanning(iq_processed.shape[0]).astype(np.float32)[:, None]
    iq_windowed = iq_processed * window

    rd_complex = np.fft.fft(iq_windowed, axis=0)
    rd_complex = np.fft.fftshift(rd_complex, axes=0)

    return 20.0 * np.log10(np.abs(rd_complex) + 1e-8)


def build_physical_rd(rd_db, fc, prf):
    """
    把FFT行映射到与标签一致的真实速度坐标。

    返回：
        rd_physical: 速度从负到正排列的RD图
        velocity_axis: 每一行对应的速度中心
    """
    n_pulses = rd_db.shape[0]
    wavelength = 3.0e8 / fc

    doppler_frequency = np.fft.fftshift(
        np.fft.fftfreq(n_pulses, d=1.0 / prf)
    )

    velocity_raw = (
        DOPPLER_SIGN
        * wavelength
        * doppler_frequency
        / 2.0
    )

    order = np.argsort(velocity_raw)

    velocity_axis = velocity_raw[order]
    rd_physical = rd_db[order, :]

    return rd_physical, velocity_axis


def plot_rd(
    rd_physical,
    velocity_axis,
    label_info,
    range_res,
    output_path,
    channel,
):
    n_velocity, n_range = rd_physical.shape

    velocity_step = float(np.median(np.diff(velocity_axis)))
    velocity_bottom = velocity_axis[0] - velocity_step / 2.0
    velocity_top = velocity_axis[-1] + velocity_step / 2.0

    # 假设第0距离门中心对应0 m，第68门中心对应2040 m。
    range_left = -range_res / 2.0
    range_right = (n_range - 0.5) * range_res

    fig, ax = plt.subplots(figsize=(12, 7))

    image = ax.imshow(
        rd_physical,
        aspect="auto",
        origin="lower",
        cmap="jet",
        extent=[
            range_left,
            range_right,
            velocity_bottom,
            velocity_top,
        ],
        vmin=DISPLAY_DB_MIN,
        vmax=DISPLAY_DB_MAX,
    )

    distance = label_info["distance"]
    velocity = label_info["velocity"]

    # 标签保持TXT中的原始值。
    ax.scatter(
        distance,
        velocity,
        marker="x",
        color="red",
        s=180,
        linewidths=3,
        zorder=5,
        label=f"Label: {distance:.1f} m, {velocity:.2f} m/s",
    )

    ax.axvline(
        distance,
        color="red",
        linestyle="--",
        alpha=0.55,
    )
    ax.axhline(
        velocity,
        color="red",
        linestyle="--",
        alpha=0.55,
    )
    ax.axhline(
        0.0,
        color="white",
        linestyle=":",
        linewidth=1.2,
        alpha=0.8,
    )

    ax.set_xlim(0.0, n_range * range_res)
    ax.set_ylim(velocity_bottom, velocity_top)
    ax.set_xlabel("Range (m)")
    ax.set_ylabel("Doppler Velocity (m/s)")
    ax.set_title(f"RD Map with Target Label | Channel {channel}")
    ax.legend(loc="upper right")

    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("Magnitude (dB)")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"✅ 已保存：{output_path}")


def find_sample(sample_name):
    mat_files = sorted(MAT_DIR.glob("*.mat"))

    if not mat_files:
        raise FileNotFoundError(f"目录中没有MAT文件：{MAT_DIR}")

    if sample_name is None:
        return mat_files[0]

    stem = sample_name.removesuffix(".mat")
    mat_path = MAT_DIR / f"{stem}.mat"

    if not mat_path.exists():
        raise FileNotFoundError(f"样本不存在：{mat_path}")

    return mat_path


def main():
    parser = argparse.ArgumentParser(
        description="生成速度方向已校正的RD图并叠加标签"
    )
    parser.add_argument(
        "--sample",
        type=str,
        default=None,
        help="样本名，可带或不带.mat；默认使用第一个样本",
    )
    parser.add_argument(
        "--channel",
        choices=["H", "V"],
        default="H",
        help="极化通道，默认H",
    )
    parser.add_argument(
        "--no-demean",
        action="store_true",
        help="关闭慢时间去均值",
    )
    args = parser.parse_args()

    config = load_config()
    mat_path = find_sample(args.sample)
    label_path = LABEL_DIR / f"{mat_path.stem}.txt"

    iq = load_iq(
        mat_path,
        args.channel,
        config["n_pulses"],
        config["n_range"],
    )
    label_info = parse_label(label_path)

    clutter_method = (
        "none"
        if args.no_demean
        else config["clutter_suppression"]
    )

    rd_db = generate_rd(iq, clutter_method)
    rd_physical, velocity_axis = build_physical_rd(
        rd_db,
        config["fc"],
        config["prf"],
    )

    output_path = (
        OUTPUT_DIR
        / f"{mat_path.stem}_{args.channel}_rd.png"
    )

    print(f"样本：{mat_path.name}")
    print(f"通道：{args.channel}")
    print(f"标签：{label_info}")
    print(
        f"速度范围：{velocity_axis[0]:.3f} ~ "
        f"{velocity_axis[-1]:.3f} m/s"
    )

    plot_rd(
        rd_physical,
        velocity_axis,
        label_info,
        config["range_res"],
        output_path,
        args.channel,
    )


if __name__ == "__main__":
    main()
