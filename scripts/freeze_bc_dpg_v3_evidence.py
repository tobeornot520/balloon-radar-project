#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FILES = (
    "models/target_protected_scan_calibrator.py",
    "training/train_target_protected_scan_calibrator.py",
    "training/train_target_protected_scan_calibrator_ablation.py",
    "scripts/run_bc_dpg_v3_ablation.py",
    "scripts/summarize_bc_dpg_v3_ablation.py",
    "scripts/build_bc_dpg_v3_deployment_comparison.py",
    "scripts/run_bc_dpg_v31_shift_reg_sweep.py",
    "scripts/select_bc_dpg_v31_shift_reg.py",
    "configs/bc_dpg_fcn_v3_scan_target.yaml",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze hashes and metadata for the current BC-DPG-FCN v3 evidence."
    )
    parser.add_argument(
        "--output-root",
        default="results/model_freeze",
    )
    parser.add_argument(
        "--include-checkpoints",
        action="store_true",
        help="Also hash base DPG and full-v3 best checkpoints. This can take longer.",
    )
    return parser.parse_args()


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def record_file(path: Path, category: str) -> dict[str, Any]:
    stat = path.stat()
    return {
        "category": category,
        "relative_path": str(path.relative_to(PROJECT_ROOT)),
        "size_bytes": stat.st_size,
        "modified_time": datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
        "sha256": sha256_file(path),
    }


def main() -> None:
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = resolve_path(args.output_root) / f"bc_dpg_v3_freeze_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=False)

    records: list[dict[str, Any]] = []
    missing: list[str] = []

    for relative in DEFAULT_FILES:
        path = PROJECT_ROOT / relative
        if path.is_file():
            records.append(record_file(path, "source_or_config"))
        else:
            missing.append(relative)

    for fold in range(1, 7):
        manifest = (
            PROJECT_ROOT
            / "results"
            / "data_audit"
            / "dataset_v4_multifold"
            / f"fold_{fold:02d}_manifest.csv"
        )
        if manifest.is_file():
            records.append(record_file(manifest, "fold_manifest"))
        else:
            missing.append(str(manifest.relative_to(PROJECT_ROOT)))

        full_summary = (
            PROJECT_ROOT
            / "results"
            / "experiments"
            / f"bc_dpg_v3_ablation_full_v4_fold{fold:02d}_seed42"
            / "tables"
            / "summary.json"
        )
        if full_summary.is_file():
            records.append(record_file(full_summary, "full_v3_summary"))
        else:
            missing.append(str(full_summary.relative_to(PROJECT_ROOT)))

        if args.include_checkpoints:
            for category, checkpoint in (
                (
                    "base_dpg_checkpoint",
                    PROJECT_ROOT
                    / "results"
                    / "experiments"
                    / f"dpg_fcn_v4_fold{fold:02d}_seed42"
                    / "checkpoints"
                    / "best.pt",
                ),
                (
                    "full_v3_checkpoint",
                    PROJECT_ROOT
                    / "results"
                    / "experiments"
                    / f"bc_dpg_v3_ablation_full_v4_fold{fold:02d}_seed42"
                    / "checkpoints"
                    / "best.pt",
                ),
            ):
                if checkpoint.is_file():
                    records.append(record_file(checkpoint, category))
                else:
                    missing.append(str(checkpoint.relative_to(PROJECT_ROOT)))

    manifest = {
        "freeze_created_at": datetime.now().isoformat(timespec="seconds"),
        "project_root": str(PROJECT_ROOT),
        "include_checkpoints": bool(args.include_checkpoints),
        "file_count": len(records),
        "missing_count": len(missing),
        "missing_files": missing,
        "files": records,
    }
    manifest_path = output_dir / "freeze_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    sums_path = output_dir / "SHA256SUMS.txt"
    sums_path.write_text(
        "\n".join(f"{item['sha256']}  {item['relative_path']}" for item in records) + "\n",
        encoding="utf-8",
    )

    readme = [
        "# BC-DPG-FCN v3 evidence freeze",
        "",
        f"Created: {manifest['freeze_created_at']}",
        f"Files hashed: {len(records)}",
        f"Missing files: {len(missing)}",
        f"Checkpoint hashes included: {bool(args.include_checkpoints)}",
        "",
        "This directory records file identities only. It does not copy raw radar data "
        "or model checkpoints.",
        "",
        "The current v3 evidence is an internal six-fold scan-group evaluation and "
        "must not be described as a new-date or new-environment blind test.",
    ]
    if missing:
        readme += ["", "## Missing", ""]
        readme.extend(f"- `{item}`" for item in missing)
    (output_dir / "README_freeze.md").write_text("\n".join(readme) + "\n", encoding="utf-8")

    print("=" * 82)
    print("BC-DPG-FCN v3 evidence freeze complete")
    print(f"directory : {output_dir}")
    print(f"manifest  : {manifest_path}")
    print(f"hashes    : {sums_path}")
    print(f"files     : {len(records)}")
    print(f"missing   : {len(missing)}")
    print("=" * 82)


if __name__ == "__main__":
    main()
