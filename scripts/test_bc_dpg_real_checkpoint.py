#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from models.dual_branch_gated_fcn import DualBranchGatedFCN
from models.background_calibrated_dpg_fcn import (
    BackgroundCalibratedDPGFCN,
    calibration_loss,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Real-checkpoint integration smoke test for BC-DPG-FCN"
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()

    if not args.checkpoint.exists():
        raise FileNotFoundError(args.checkpoint)

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    config = checkpoint.get("config", {})
    gate_hidden_dim = int(config.get("gate_hidden_dim", 16))

    base_model = DualBranchGatedFCN(
        gate_hidden_dim=gate_hidden_dim
    )
    load_result = base_model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True,
    )

    model = BackgroundCalibratedDPGFCN(
        base_model=base_model,
        input_is_probability=False,
        topk=16,
        hidden_dims=(32, 16),
        min_temperature=0.05,
        freeze_base=True,
    ).to(device)

    model.train()

    input_tensor = torch.randn(
        args.batch_size,
        2,
        128,
        100,
        device=device,
    )
    labels = torch.tensor(
        [float(i % 2) for i in range(args.batch_size)],
        device=device,
    )

    output = model(input_tensor)

    raw_argmax = output["raw_logits"].flatten(1).argmax(dim=1)
    calibrated_argmax = (
        output["calibrated_logits"].flatten(1).argmax(dim=1)
    )
    if not torch.equal(raw_argmax, calibrated_argmax):
        raise AssertionError("Calibration changed spatial argmax")

    sample_logits = model.sample_score(
        output["calibrated_logits"]
    )
    loss, loss_parts = calibration_loss(
        sample_logits,
        labels,
        output["temperature"],
        output["bias"],
    )
    loss.backward()

    if any(p.requires_grad for p in model.base_model.parameters()):
        raise AssertionError("Base DPG-FCN is not frozen")

    base_grads = [
        p.grad
        for p in model.base_model.parameters()
        if p.grad is not None
    ]
    if base_grads:
        raise AssertionError("Frozen DPG-FCN received gradients")

    calibrator_grads = [
        p.grad
        for p in model.calibrator.parameters()
        if p.requires_grad
    ]
    if not calibrator_grads or not all(
        grad is not None for grad in calibrator_grads
    ):
        raise AssertionError("Calibration head did not receive gradients")

    print("Real DPG checkpoint integration test passed")
    print(f"device                    : {device}")
    print(f"checkpoint                : {args.checkpoint}")
    print(f"checkpoint epoch          : {checkpoint.get('epoch')}")
    print(f"checkpoint stage          : {checkpoint.get('stage')}")
    print(f"gate_hidden_dim           : {gate_hidden_dim}")
    print(f"missing keys              : {load_result.missing_keys}")
    print(f"unexpected keys           : {load_result.unexpected_keys}")
    print(f"raw logits shape          : {tuple(output['raw_logits'].shape)}")
    print(f"gate weights shape        : {tuple(output['gate_weights'].shape)}")
    print(f"calibration features      : {tuple(output['calibration_features'].shape)}")
    print(f"temperature mean          : {output['temperature'].mean().item():.6f}")
    print(f"bias mean                 : {output['bias'].mean().item():.6f}")
    print(f"spatial argmax preserved  : yes")
    print(f"base model frozen         : yes")
    print(f"calibrator gradients      : yes")
    print(f"loss                      : {loss.item():.6f}")
    for key, value in loss_parts.items():
        print(f"{key:27s}: {value.item():.6f}")


if __name__ == "__main__":
    main()
