#!/usr/bin/env python3
from __future__ import annotations

import argparse
import inspect
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from datasets.detection_dataset_v3 import DetectionRadarDatasetV3


def source_or_error(obj):
    try:
        return inspect.getsource(obj)
    except Exception as exc:
        return f"<source unavailable: {exc}>"


def find_context(path: Path, terms, radius: int = 8):
    lines = path.read_text(encoding="utf-8").splitlines()
    selected = set()

    for index, line in enumerate(lines):
        if any(term in line for term in terms):
            start = max(0, index - radius)
            end = min(len(lines), index + radius + 1)
            selected.update(range(start, end))

    blocks = []
    current = []
    previous = None

    for index in sorted(selected):
        if previous is not None and index != previous + 1:
            if current:
                blocks.append(current)
            current = []

        current.append(
            {
                "line_number": index + 1,
                "text": lines[index],
            }
        )
        previous = index

    if current:
        blocks.append(current)

    return blocks


def main():
    parser = argparse.ArgumentParser(
        description="Inspect dataset and current training integration"
    )
    parser.add_argument(
        "--training-file",
        type=Path,
        default=Path("training/train_dual_branch_gated.py"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "results/data_audit/bc_dpg_preflight/"
            "dataset_training_interface.json"
        ),
    )
    args = parser.parse_args()

    report = {
        "dataset_class": {
            "name": DetectionRadarDatasetV3.__name__,
            "signature": str(inspect.signature(DetectionRadarDatasetV3)),
            "init_signature": str(
                inspect.signature(DetectionRadarDatasetV3.__init__)
            ),
            "getitem_signature": str(
                inspect.signature(DetectionRadarDatasetV3.__getitem__)
            ),
            "init_source": source_or_error(
                DetectionRadarDatasetV3.__init__
            ),
            "getitem_source": source_or_error(
                DetectionRadarDatasetV3.__getitem__
            ),
        },
        "training_file": str(args.training_file.resolve()),
        "training_context": find_context(
            args.training_file,
            terms=[
                "DetectionRadarDatasetV3",
                "DataLoader",
                "target_present",
                "target_heatmap",
                "fusion_logits",
                "select_threshold",
                "evaluate",
                "validation_metrics",
            ],
            radius=10,
        ),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Interface report written to: {args.output}")
    print()
    print("Dataset signature:")
    print(report["dataset_class"]["signature"])
    print()
    print("__getitem__ signature:")
    print(report["dataset_class"]["getitem_signature"])
    print()
    print("Training context blocks:")
    print(len(report["training_context"]))


if __name__ == "__main__":
    main()
