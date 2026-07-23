from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from datasets.radar_dataset import (
    RadarDataset,
)


OUTPUT_DIR = Path(
    "results/figures/dataset_check"
)

OUTPUT_FIGURE = (
    OUTPUT_DIR
    / "radar_dataset_check.png"
)


def configure_chinese_font() -> None:
    font_paths = [
        Path(
            "/usr/share/fonts/opentype/noto/"
            "NotoSansCJK-Regular.ttc"
        ),
        Path(
            "/usr/share/fonts/truetype/wqy/"
            "wqy-zenhei.ttc"
        ),
    ]

    for font_path in font_paths:
        if not font_path.exists():
            continue

        font_property = (
            font_manager.FontProperties(
                fname=str(font_path)
            )
        )

        font_name = (
            font_property.get_name()
        )

        plt.rcParams[
            "font.family"
        ] = font_name

        plt.rcParams[
            "axes.unicode_minus"
        ] = False

        return


def check_single_sample(
    sample: dict,
) -> None:
    input_tensor = sample[
        "input"
    ]

    target_tensor = sample[
        "target"
    ]

    if input_tensor.shape != (
        2,
        128,
        100,
    ):
        raise ValueError(
            "输入形状错误："
            f"{input_tensor.shape}"
        )

    if target_tensor.shape != (
        1,
        128,
        100,
    ):
        raise ValueError(
            "标签形状错误："
            f"{target_tensor.shape}"
        )

    if not torch.isfinite(
        input_tensor
    ).all():
        raise ValueError(
            "输入中存在无效数值"
        )

    if not torch.isfinite(
        target_tensor
    ).all():
        raise ValueError(
            "标签中存在无效数值"
        )

    heatmap = (
        target_tensor[0]
        .numpy()
    )

    peak_flat_index = int(
        np.argmax(heatmap)
    )

    peak_velocity_index, peak_range_index = (
        np.unravel_index(
            peak_flat_index,
            heatmap.shape,
        )
    )

    expected_velocity_index = int(
        sample["velocity_index"]
    )

    expected_range_index = int(
        sample["range_index"]
    )

    if (
        peak_velocity_index
        != expected_velocity_index
    ):
        raise ValueError(
            "热力图速度峰值与标签不一致："
            f"{peak_velocity_index}，"
            f"期望{expected_velocity_index}"
        )

    if (
        peak_range_index
        != expected_range_index
    ):
        raise ValueError(
            "热力图距离峰值与标签不一致："
            f"{peak_range_index}，"
            f"期望{expected_range_index}"
        )

    print(
        "\n========== 单样本检查 =========="
    )

    print(
        f"样本编号：{sample['sample_id']}"
    )

    print(
        f"波束层：{sample['beam_layer']}"
    )

    print(
        f"输入形状：{tuple(input_tensor.shape)}"
    )

    print(
        f"标签形状：{tuple(target_tensor.shape)}"
    )

    print(
        "输入最小值："
        f"{input_tensor.min().item():.6f}"
    )

    print(
        "输入最大值："
        f"{input_tensor.max().item():.6f}"
    )

    print(
        "标签最小值："
        f"{target_tensor.min().item():.6f}"
    )

    print(
        "标签最大值："
        f"{target_tensor.max().item():.6f}"
    )

    print(
        "标签坐标："
        f"速度下标={expected_velocity_index}，"
        f"距离下标={expected_range_index}"
    )

    print(
        "热力图峰值坐标："
        f"速度下标={peak_velocity_index}，"
        f"距离下标={peak_range_index}"
    )


def save_check_figure(
    sample: dict,
) -> None:
    configure_chinese_font()

    input_array = (
        sample["input"]
        .numpy()
    )

    target_array = (
        sample["target"][0]
        .numpy()
    )

    range_index = int(
        sample["range_index"]
    )

    velocity_index = int(
        sample["velocity_index"]
    )

    figure, axes = plt.subplots(
        1,
        3,
        figsize=(16, 5),
    )

    h_image = axes[0].imshow(
        input_array[0],
        aspect="auto",
        cmap="jet",
    )

    axes[0].plot(
        range_index,
        velocity_index,
        marker="+",
        color="red",
        markersize=14,
        markeredgewidth=2,
    )

    axes[0].set_title(
        "H通道归一化RD图"
    )

    axes[0].set_xlabel(
        "距离门下标"
    )

    axes[0].set_ylabel(
        "速度单元下标"
    )

    figure.colorbar(
        h_image,
        ax=axes[0],
    )

    v_image = axes[1].imshow(
        input_array[1],
        aspect="auto",
        cmap="jet",
    )

    axes[1].plot(
        range_index,
        velocity_index,
        marker="+",
        color="red",
        markersize=14,
        markeredgewidth=2,
    )

    axes[1].set_title(
        "V通道归一化RD图"
    )

    axes[1].set_xlabel(
        "距离门下标"
    )

    axes[1].set_ylabel(
        "速度单元下标"
    )

    figure.colorbar(
        v_image,
        ax=axes[1],
    )

    target_image = axes[2].imshow(
        target_array,
        aspect="auto",
        cmap="jet",
        vmin=0.0,
        vmax=1.0,
    )

    axes[2].plot(
        range_index,
        velocity_index,
        marker="+",
        color="white",
        markersize=14,
        markeredgewidth=2,
    )

    axes[2].set_title(
        "目标高斯热力图"
    )

    axes[2].set_xlabel(
        "距离门下标"
    )

    axes[2].set_ylabel(
        "速度单元下标"
    )

    figure.colorbar(
        target_image,
        ax=axes[2],
    )

    figure.suptitle(
        (
            f"样本：{sample['sample_id']}　"
            f"波束层：{sample['beam_layer']}"
        ),
        fontsize=14,
    )

    figure.tight_layout()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure.savefig(
        OUTPUT_FIGURE,
        dpi=220,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )


def main() -> None:
    train_dataset = RadarDataset(
        split="train",
        range_sigma=2.0,
        velocity_sigma=1.0,
    )

    val_dataset = RadarDataset(
        split="val",
        range_sigma=2.0,
        velocity_sigma=1.0,
    )

    test_dataset = RadarDataset(
        split="test",
        range_sigma=2.0,
        velocity_sigma=1.0,
    )

    print(
        "========== 数据集规模 =========="
    )

    print(
        f"训练集：{len(train_dataset)}"
    )

    print(
        f"验证集：{len(val_dataset)}"
    )

    print(
        f"测试集：{len(test_dataset)}"
    )

    first_sample = train_dataset[0]

    check_single_sample(
        first_sample
    )

    data_loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0,
        drop_last=False,
    )

    first_batch = next(
        iter(data_loader)
    )

    print(
        "\n========== 批量读取检查 =========="
    )

    print(
        "批量输入形状："
        f"{tuple(first_batch['input'].shape)}"
    )

    print(
        "批量标签形状："
        f"{tuple(first_batch['target'].shape)}"
    )

    print(
        "批量输入数据类型："
        f"{first_batch['input'].dtype}"
    )

    print(
        "批量标签数据类型："
        f"{first_batch['target'].dtype}"
    )

    save_check_figure(
        first_sample
    )

    print(
        "\n========== 检查完成 =========="
    )

    print(
        "数据集读取、批量加载和热力图"
        "坐标检查均已通过。"
    )

    print(
        f"检查图片：{OUTPUT_FIGURE.resolve()}"
    )


if __name__ == "__main__":
    main()