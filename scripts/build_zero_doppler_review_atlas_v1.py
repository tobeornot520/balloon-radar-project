#!/usr/bin/env python3
"""Render local RD review sheets for selected zero-Doppler false alarms."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
DEFAULT_QUEUE = PROJECT_ROOT / "results/data_audit/zero_doppler_human_review_v1/review_queue.csv"
DEFAULT_MANIFEST_ROOT = PROJECT_ROOT / "results/data_audit/dataset_v4_multifold"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results/data_audit/zero_doppler_review_atlas_v1"
ZERO_VELOCITY_INDEX = 64


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render local RD review sheets from the zero-Doppler queue."
    )
    parser.add_argument("--queue", type=Path, default=DEFAULT_QUEUE)
    parser.add_argument("--manifest-root", type=Path, default=DEFAULT_MANIFEST_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--priorities",
        nargs="+",
        default=["P0_removed_by_residual"],
        help="Queue priorities to render, in their existing queue order.",
    )
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_cases(
    queue: pd.DataFrame, priorities: list[str], max_cases: int
) -> pd.DataFrame:
    required = {
        "fold",
        "sample_id",
        "review_priority",
        "pred_range_index_fixed",
        "pred_velocity_index_fixed",
        "pred_range_index_residual",
        "pred_velocity_index_residual",
    }
    missing = required - set(queue.columns)
    if missing:
        raise ValueError(f"review queue missing columns: {sorted(missing)}")
    if max_cases < 0:
        raise ValueError("max-cases must be zero or positive")
    selected = queue[queue["review_priority"].isin(priorities)].copy()
    if selected.empty:
        raise ValueError("no queue rows match the requested priorities")
    if max_cases:
        selected = selected.iloc[:max_cases].copy()
    return selected.reset_index(drop=True)


def manifest_record(
    manifest_root: Path, fold: int, sample_id: str) -> pd.Series:
    path = manifest_root / f"fold_{int(fold):02d}_manifest.csv"
    if not path.is_file():
        raise FileNotFoundError(f"missing fold manifest: {path}")
    frame = pd.read_csv(path, encoding="utf-8-sig")
    required = {"sample_id", "new_split", "mat_path"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path.name} missing columns: {sorted(missing)}")
    matches = frame[
        frame["sample_id"].astype(str).eq(str(sample_id))
        & frame["new_split"].astype(str).eq("test")
    ]
    if len(matches) != 1:
        raise ValueError(
            f"fold {fold} must contain exactly one test record for {sample_id}, got {len(matches)}"
        )
    return matches.iloc[0]


def relative_hv_db(h_power: np.ndarray, v_power: np.ndarray) -> np.ndarray:
    ratio = 10.0 * np.log10((h_power + 1e-12) / (v_power + 1e-12))
    return np.clip(ratio, -20.0, 20.0).astype(np.float32, copy=False)


def draw_case(
    row: pd.Series,
    mat_path: Path,
    output: Path,
) -> None:
    # Keep the plot implementation local to make headless batch rendering reliable.
    temp_config = Path(tempfile.gettempdir()) / "balloon-radar-matplotlib"
    temp_config.mkdir(parents=True, exist_ok=True)
    import os

    os.environ.setdefault("MPLCONFIGDIR", str(temp_config))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from datasets.detection_dataset_v2 import _calculate_rd_power, _load_iq_pair, _normalize_power_db

    h_iq, v_iq = _load_iq_pair(mat_path)
    h_power = _calculate_rd_power(h_iq)
    v_power = _calculate_rd_power(v_iq)
    panels = (
        ("H normalized RD", _normalize_power_db(h_power), "viridis", 0.0, 1.0),
        ("V normalized RD", _normalize_power_db(v_power), "viridis", 0.0, 1.0),
        (
            "Combined H+V normalized RD",
            _normalize_power_db(h_power + v_power),
            "viridis",
            0.0,
            1.0,
        ),
        (
            "Relative H/V power (dB, uncalibrated)",
            relative_hv_db(h_power, v_power),
            "coolwarm",
            -20.0,
            20.0,
        ),
    )
    fixed = (float(row["pred_range_index_fixed"]), float(row["pred_velocity_index_fixed"]))
    residual = (
        float(row["pred_range_index_residual"]),
        float(row["pred_velocity_index_residual"]),
    )

    figure, axes = plt.subplots(2, 2, figsize=(15, 11), constrained_layout=True)
    for axis, (title, image, cmap, vmin, vmax) in zip(axes.ravel(), panels):
        rendered = axis.imshow(
            image, origin="lower", aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax
        )
        axis.axhline(
            ZERO_VELOCITY_INDEX,
            color="white",
            linestyle=":",
            linewidth=1.2,
            alpha=0.9,
        )
        axis.scatter(
            fixed[0], fixed[1], marker="x", color="white", s=76, linewidths=1.8,
            label="Fixed-notch peak",
        )
        axis.scatter(
            residual[0], residual[1], marker="o", facecolors="none", edgecolors="cyan",
            s=76, linewidths=1.5, label="Residual peak",
        )
        axis.set_title(title)
        axis.set_xlabel("Range gate index")
        axis.set_ylabel("Doppler bin index")
        axis.legend(loc="upper right", fontsize=8)
        figure.colorbar(rendered, ax=axis, fraction=0.046, pad=0.04)

    figure.suptitle(
        " | ".join(
            [
                f"Fold {int(row['fold'])}",
                str(row["review_priority"]),
                str(row["sample_id"]),
                "zero-Doppler reference: bin 64",
            ]
        ),
        fontsize=12,
    )
    figure.savefig(output, dpi=180)
    plt.close(figure)


def make_readme(cases: pd.DataFrame) -> str:
    return f"""# Zero-Doppler Review Atlas V1

This local atlas contains {len(cases)} queue entries selected from the paired
fixed-notch/residual development diagnostic. It is a visual aid for human
review, not a model result and not a physical-background labeling system.

## Reading rules

- The horizontal and vertical axes are range-gate and Doppler-bin indices.
  No PRF-backed physical velocity is claimed here.
- The dotted horizontal line is bin 64, the assumed zero-Doppler reference used
  consistently by the existing diagnostic.
- The H/V panel is an uncalibrated relative power ratio. It is not absolute ZDR.
- Markers show the fixed-notch and residual predicted peaks. They do not prove a
  target or background mechanism.
- Record only visible structure in the review queue. Keep `physical_class` as
  `unknown` unless an independent scene record supports a specific label.

## Files

- `cases.csv`: selected queue rows and image filenames;
- `images/`: one full-resolution RD sheet per selected row;
- `manifest.json`: reproducibility metadata without raw-data paths.
"""


def build_atlas(
    *,
    queue_path: Path,
    manifest_root: Path,
    output_dir: Path,
    priorities: list[str],
    max_cases: int,
    overwrite: bool,
) -> dict[str, Any]:
    queue_path = resolve_path(queue_path)
    manifest_root = resolve_path(manifest_root)
    output_dir = resolve_path(output_dir)
    if not queue_path.is_file():
        raise FileNotFoundError(f"missing review queue: {queue_path}")
    if not manifest_root.is_dir():
        raise FileNotFoundError(f"missing manifest directory: {manifest_root}")
    if output_dir == PROJECT_ROOT:
        raise ValueError("output directory must not be the project root")
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(f"output directory is nonempty: {output_dir}")
        shutil.rmtree(output_dir)
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    queue = pd.read_csv(queue_path, encoding="utf-8-sig")
    cases = select_cases(queue, priorities, max_cases)
    filenames: list[str] = []
    manifests: dict[str, dict[str, str]] = {}
    for rank, (_, row) in enumerate(cases.iterrows(), start=1):
        record = manifest_record(manifest_root, int(row["fold"]), str(row["sample_id"]))
        from datasets.polarimetric_detection_dataset_v2 import _recover_data_path

        mat_path = _recover_data_path(str(record["mat_path"]))
        filename = f"case_{rank:03d}_{row['sample_id']}.png"
        draw_case(row, mat_path, image_dir / filename)
        filenames.append(filename)
        manifest_path = manifest_root / f"fold_{int(row['fold']):02d}_manifest.csv"
        manifests[manifest_path.name] = {"sha256": sha256_file(manifest_path)}
    cases = cases.copy()
    cases.insert(0, "atlas_rank", range(1, len(cases) + 1))
    cases["image_file"] = [f"images/{name}" for name in filenames]
    cases.to_csv(output_dir / "cases.csv", index=False, encoding="utf-8-sig")
    (output_dir / "README.md").write_text(make_readme(cases), encoding="utf-8")
    result = {
        "schema_version": 1,
        "role": "local_human_review_visual_aid_not_model_evidence",
        "queue_sha256": sha256_file(queue_path),
        "priority_filter": priorities,
        "case_count": int(len(cases)),
        "zero_velocity_index": ZERO_VELOCITY_INDEX,
        "fold_manifests": manifests,
        "claim_boundary": (
            "RD bin visualizations and uncalibrated relative H/V power only; no "
            "physical background labels, calibrated polarimetry, or blind-test "
            "claim is established"
        ),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return result


def main() -> int:
    args = parse_args()
    result = build_atlas(
        queue_path=args.queue,
        manifest_root=args.manifest_root,
        output_dir=args.output_dir,
        priorities=list(args.priorities),
        max_cases=args.max_cases,
        overwrite=args.overwrite,
    )
    print("Zero-Doppler review atlas: PASS")
    print(f"case_count={result['case_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
