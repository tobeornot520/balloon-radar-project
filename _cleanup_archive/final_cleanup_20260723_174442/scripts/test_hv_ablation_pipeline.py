from __future__ import annotations

import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from datasets.radar_dataset import RadarDataset
from models.simple_fcn import SimpleRadarFCN
from training.train_dual_fcn import (
    DualViewFCN,
    SoftNotchDataset,
)


def main() -> None:
    print("=" * 72)
    print("H/V/HV消融管线检查")
    print("=" * 72)

    for channel in ["H", "V", "HV"]:
        input_channels = (
            2 if channel == "HV" else 1
        )
        dataset = RadarDataset(
            split="train",
            channel_mode=channel,
            max_samples=1,
        )
        sample = dataset[0]
        input_tensor = sample["input"].unsqueeze(0)

        expected_input_shape = (
            1,
            input_channels,
            128,
            100,
        )
        if tuple(input_tensor.shape) != expected_input_shape:
            raise RuntimeError(
                f"{channel}输入尺寸错误："
                f"{tuple(input_tensor.shape)}"
            )

        base_model = SimpleRadarFCN(
            in_channels=input_channels,
        )
        base_model.eval()

        with torch.no_grad():
            base_output = base_model(input_tensor)

        if tuple(base_output.shape) != (
            1,
            1,
            128,
            100,
        ):
            raise RuntimeError(
                f"{channel} Simple FCN输出尺寸错误："
                f"{tuple(base_output.shape)}"
            )

        wrapped_dataset = SoftNotchDataset(
            dataset
        )
        wrapped_sample = wrapped_dataset[0]

        dual_model = DualViewFCN(
            base_model.state_dict(),
            in_channels=input_channels,
        )
        dual_model.eval()

        with torch.no_grad():
            outputs = dual_model(
                wrapped_sample[
                    "raw_input"
                ].unsqueeze(0),
                wrapped_sample[
                    "notch_input"
                ].unsqueeze(0),
            )

        for output in outputs:
            if tuple(output.shape) != (
                1,
                1,
                128,
                100,
            ):
                raise RuntimeError(
                    f"{channel} Dual FCN输出尺寸错误："
                    f"{tuple(output.shape)}"
                )

        print(
            f"{channel}：通过｜"
            f"输入={tuple(input_tensor.shape)}｜"
            f"输出={tuple(outputs[-1].shape)}"
        )

    print("=" * 72)
    print("Dataset、Simple FCN、软陷波和Dual FCN均已通过。")


if __name__ == "__main__":
    main()
