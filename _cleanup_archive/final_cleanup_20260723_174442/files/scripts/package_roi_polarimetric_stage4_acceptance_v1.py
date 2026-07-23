#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import zipfile
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODES = (
    "power2_baseline",
    "power2_roi_power_control",
    "power2_roi_ri4",
    "power2_roi_polar6_gated",
    "power2_roi_ri4_polar6_gated",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", choices=["smoke", "formal"], required=True)
    parser.add_argument("--folds", nargs="+", type=int, default=[1, 4])
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--terminal-log", default=None)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def experiment_name(mode: str, fold: int, scope: str) -> str:
    return f"roi_polar_stage4_v1_{mode}_v4_fold{fold:02d}_seed42_{scope}"


def main() -> None:
    args = parse_args()
    audit = PROJECT_ROOT / "results/data_audit/roi_polarimetric_stage4_v1"
    package = PROJECT_ROOT / f"roi_polarimetric_stage4_{args.scope}_acceptance_v1.zip"
    files: list[Path] = []
    required_audit = [
        audit / "pipeline_test.json",
        audit / f"latest_run_plan_{args.scope}.json",
        audit / f"latest_run_status_{args.scope}.json",
        audit / f"stage4_detail_{args.scope}.csv",
        audit / f"stage4_aggregate_{args.scope}.csv",
        audit / f"stage4_threshold_transfer_{args.scope}.csv",
        audit / f"stage4_rescue_regression_{args.scope}.csv",
        audit / f"README_stage4_{args.scope}.md",
    ]
    files.extend(path for path in required_audit if path.is_file())
    for fold in args.folds:
        cache = audit / "cache" / f"fold_{fold:02d}_{args.scope}"
        files.extend(path for path in [
            cache / "cache_status.json",
            cache / "train_candidate_inventory.csv",
            cache / "val_candidate_inventory.csv",
            cache / "test_candidate_inventory.csv",
        ] if path.is_file())
        for mode in args.modes:
            root = PROJECT_ROOT / "results/experiments" / experiment_name(mode, fold, args.scope)
            files.extend(path for path in [
                root / "tables/summary.json",
                root / "tables/val_predictions.csv",
                root / "tables/test_predictions.csv",
                root / "tables/training_history.csv",
            ] if path.is_file())
    source_files = [
        PROJECT_ROOT / "features/roi_polarimetric_refinement.py",
        PROJECT_ROOT / "datasets/roi_polarimetric_refinement_dataset.py",
        PROJECT_ROOT / "models/roi_polarimetric_refiner.py",
        PROJECT_ROOT / "training/train_roi_polarimetric_refiner_v1.py",
        PROJECT_ROOT / "scripts/build_roi_polarimetric_cache_v1.py",
        PROJECT_ROOT / "scripts/run_roi_polarimetric_stage4_v1.py",
        PROJECT_ROOT / "scripts/summarize_roi_polarimetric_stage4_v1.py",
        PROJECT_ROOT / "scripts/test_roi_polarimetric_stage4_v1.py",
        PROJECT_ROOT / "configs/roi_polarimetric_stage4_v1.yaml",
        PROJECT_ROOT / "docs/README_候选区域极化精修Stage4_V1.md",
    ]
    files.extend(path for path in source_files if path.is_file())
    if args.terminal_log:
        log_path = Path(args.terminal_log).expanduser()
        if not log_path.is_absolute():
            log_path = PROJECT_ROOT / log_path
        if log_path.is_file():
            files.append(log_path.resolve())

    checkpoint_rows = []
    missing_experiments = []
    for fold in args.folds:
        for mode in args.modes:
            name = experiment_name(mode, fold, args.scope)
            checkpoint = PROJECT_ROOT / "results/experiments" / name / "checkpoints/best.pt"
            if not checkpoint.is_file():
                missing_experiments.append(name)
                continue
            obj = torch.load(checkpoint, map_location="cpu", weights_only=False)
            checkpoint_rows.append({
                "experiment_name": name,
                "checkpoint_path": str(checkpoint.relative_to(PROJECT_ROOT)),
                "size_bytes": checkpoint.stat().st_size,
                "sha256": sha256(checkpoint),
                "top_level_keys": "|".join(sorted(str(key) for key in obj.keys())) if isinstance(obj, dict) else type(obj).__name__,
                "epoch": obj.get("epoch") if isinstance(obj, dict) else None,
                "mode": obj.get("mode") if isinstance(obj, dict) else None,
                "fold_id": obj.get("fold_id") if isinstance(obj, dict) else None,
                "base_threshold": obj.get("base_threshold") if isinstance(obj, dict) else None,
                "state_parameter_tensors": len(obj.get("model_state_dict", {})) if isinstance(obj, dict) else None,
            })

    inventory = audit / f"checkpoint_inventory_{args.scope}.csv"
    inventory.parent.mkdir(parents=True, exist_ok=True)
    with inventory.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(checkpoint_rows[0].keys()) if checkpoint_rows else ["experiment_name"])
        writer.writeheader()
        writer.writerows(checkpoint_rows)
    files.append(inventory)

    status = {
        "status": "PASS" if not missing_experiments else "INCOMPLETE",
        "scope": args.scope,
        "requested_experiments": len(args.folds) * len(args.modes),
        "checkpoint_metadata_found": len(checkpoint_rows),
        "missing_experiments": missing_experiments,
        "checkpoint_bytes_included": False,
        "cache_tensor_bytes_included": False,
        "raw_mat_bytes_included": False,
        "files_included": len(set(files)),
    }
    status_path = audit / f"acceptance_package_status_{args.scope}.json"
    status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    files.append(status_path)

    unique = sorted(set(path.resolve() for path in files if path.is_file()))
    with zipfile.ZipFile(package, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in unique:
            try:
                arcname = path.relative_to(PROJECT_ROOT)
            except ValueError:
                arcname = Path("external") / path.name
            if path.suffix.lower() in {".pt", ".pth", ".ckpt", ".mat"}:
                continue
            archive.write(path, arcname.as_posix())
    print("=" * 88)
    print(f"Stage 4 acceptance package: {package}")
    print(f"status                    : {status['status']}")
    print(f"experiments               : {len(checkpoint_rows)}/{status['requested_experiments']}")
    print(f"files                     : {len(unique)}")
    print("checkpoint/cache/raw bytes: excluded")
    print("=" * 88)
    if missing_experiments:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
