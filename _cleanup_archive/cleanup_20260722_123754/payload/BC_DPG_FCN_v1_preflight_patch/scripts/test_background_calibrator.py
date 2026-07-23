#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from models.background_calibrated_dpg_fcn import (  # noqa: E402
    BackgroundCalibratedDPGFCN,
    calibration_loss,
)


class DummyDPG(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=1)

    def forward(self, h_input, v_input):
        heatmap = self.conv(torch.cat([h_input, v_input], dim=1))
        gate = torch.softmax(
            torch.stack(
                [
                    h_input.flatten(1).mean(dim=1),
                    v_input.flatten(1).mean(dim=1),
                ],
                dim=1,
            ),
            dim=1,
        )
        return {"heatmap": heatmap, "gate_weights": gate}


def main():
    torch.manual_seed(42)

    base = DummyDPG()
    model = BackgroundCalibratedDPGFCN(base, freeze_base=True)

    h_input = torch.randn(4, 1, 32, 24)
    v_input = torch.randn(4, 1, 32, 24)
    labels = torch.tensor([0.0, 1.0, 0.0, 1.0])

    output = model(h_input, v_input)

    raw_argmax = output["raw_heatmap"].flatten(1).argmax(dim=1)
    calibrated_argmax = output["calibrated_heatmap"].flatten(1).argmax(dim=1)
    assert torch.equal(raw_argmax, calibrated_argmax)

    sample_logits = model.sample_score(output["calibrated_logits"])
    loss, parts = calibration_loss(
        sample_logits,
        labels,
        output["temperature"],
        output["bias"],
    )
    loss.backward()

    assert not any(p.requires_grad for p in base.parameters())
    assert all(
        p.grad is not None
        for p in model.calibrator.parameters()
        if p.requires_grad
    )

    print("BC-DPG-FCN model smoke test passed")
    print("spatial argmax preserved: yes")
    print("base model frozen       : yes")
    print("calibrator gradients    : yes")
    print(f"feature shape           : {tuple(output['calibration_features'].shape)}")
    print(f"temperature mean        : {output['temperature'].mean().item():.6f}")
    print(f"bias mean               : {output['bias'].mean().item():.6f}")
    print(f"loss                    : {loss.item():.6f}")
    for key, value in parts.items():
        print(f"{key:24s}: {value.item():.6f}")


if __name__ == "__main__":
    main()
