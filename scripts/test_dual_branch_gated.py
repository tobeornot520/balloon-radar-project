#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.dual_branch_gated_fcn import (
    DualBranchGatedFCN,
    count_total_parameters,
    load_single_branch_checkpoint,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="检查DPG-FCN模型、前向传播及单通道checkpoint加载")
    parser.add_argument("--h-checkpoint", default="results/experiments/detection_h_baseline_v2/checkpoints/best.pt")
    parser.add_argument("--v-checkpoint", default="results/experiments/detection_v_baseline_v2/checkpoints/best.pt")
    parser.add_argument("--skip-checkpoints", action="store_true")
    args = parser.parse_args()

    model = DualBranchGatedFCN(gate_hidden_dim=16)
    if not args.skip_checkpoints:
        h_path = (PROJECT_ROOT / args.h_checkpoint).resolve() if not Path(args.h_checkpoint).is_absolute() else Path(args.h_checkpoint)
        v_path = (PROJECT_ROOT / args.v_checkpoint).resolve() if not Path(args.v_checkpoint).is_absolute() else Path(args.v_checkpoint)
        print("H加载：", load_single_branch_checkpoint(model.h_branch, h_path))
        print("V加载：", load_single_branch_checkpoint(model.v_branch, v_path))

    model.eval()
    with torch.no_grad():
        output = model(torch.randn(2, 2, 128, 100))
    assert output["fusion_logits"].shape == (2, 1, 128, 100)
    assert output["h_logits"].shape == (2, 1, 128, 100)
    assert output["v_logits"].shape == (2, 1, 128, 100)
    assert output["gate_weights"].shape == (2, 2)
    assert torch.allclose(output["gate_weights"].sum(dim=1), torch.ones(2), atol=1e-6)

    for stage in ("warmup", "partial", "full"):
        model.set_branch_trainability(stage)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"stage={stage}: trainable={trainable:,}")
    print(f"总参数量：{count_total_parameters(model):,}")
    print("DPG-FCN模型、门控权重与前向尺寸检查全部通过。")


if __name__ == "__main__":
    main()
