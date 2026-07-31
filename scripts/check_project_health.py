#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import hashlib
import sys
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

EXCLUDED_PARTS = {
    ".git",
    "_cleanup_archive",
    "data",
    "dist",
    "payload",
    "results",
    "notes",
}

REQUIRED_FILES = (
    "environment.yml",
    "datasets/detection_dataset_v3.py",
    "datasets/polarimetric_detection_dataset_v2.py",
    "features/polarimetric_rd.py",
    "features/roi_polarimetric_refinement.py",
    "features/scan_context.py",
    "models/target_protected_scan_calibrator.py",
    "models/tian_fcn.py",
    "models/roi_polarimetric_refiner.py",
    "training/train_target_protected_scan_calibrator.py",
    "training/tian_fcn_objective.py",
    "training/train_tian_fcn.py",
    "training/train_roi_polarimetric_refiner_v1.py",
    "scripts/build_roi_bc_dpg_joint_tables_v1.py",
    "scripts/build_final_roi_bc_dpg_joint_audit.py",
    "scripts/build_roi_bc_dpg_joint_paper_assets.py",
    "scripts/build_project_share_package.py",
    "scripts/audit_bc_dpg_v3_causal_context.py",
    "scripts/audit_detection_acquisition_order.py",
    "scripts/run_bc_dpg_causal_smoke.py",
    "scripts/build_bc_dpg_localization_evidence.py",
    "scripts/validate_data_collection_manifest.py",
    "scripts/audit_field_readiness_v1.py",
    "scripts/initialize_field_readiness_evidence.py",
    "scripts/run_recorded_experiment.py",
    "scripts/manage_experiment_ledger.py",
    "scripts/run_tian_fcn_reproduction_smoke.py",
    "scripts/run_tian_fcn_sixfold.py",
    "scripts/summarize_tian_fcn_sixfold.py",
    "evaluation/tian_fcn_metrics.py",
    "evaluation/tian_fcn_postprocess.py",
    "configs/tian_fcn_reproduction_v1.yaml",
    "configs/data_collection_contract_v1.json",
    "configs/data_collection_manifest_template_v1.csv",
    "configs/field_readiness_checklist_v1.json",
    "configs/field_readiness_evidence_template_v1.csv",
    "configs/pilot_scenario_matrix_v1.csv",
    "configs/pilot_session_log_template_v1.csv",
    "docs/NEW_DATA_COLLECTION_PROTOCOL.md",
    "docs/FIELD_COLLECTION_SOP_V1.md",
    "docs/TIAN_FCN_REPRODUCTION_PROTOCOL.md",
    "docs/EXPERIMENT_RECORDING_PROTOCOL.md",
    "utils/experiment_ledger.py",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check active Python syntax and required project files."
    )
    parser.add_argument(
        "--require-joint-inputs",
        action="store_true",
        help="Also require all frozen six-fold BC-DPG and ROI prediction tables.",
    )
    return parser.parse_args()


def active_python_files() -> list[Path]:
    return sorted(
        path
        for path in ROOT.rglob("*.py")
        if not EXCLUDED_PARTS.intersection(path.relative_to(ROOT).parts)
        and not path.name.endswith(".syntax_error_backup")
    )


def syntax_errors(paths: list[Path]) -> list[str]:
    errors: list[str] = []
    for path in paths:
        try:
            source = path.read_text(encoding="utf-8-sig")
            ast.parse(source, filename=str(path))
        except (SyntaxError, UnicodeError) as exc:
            relative = path.relative_to(ROOT)
            line = getattr(exc, "lineno", None)
            errors.append(f"{relative}:{line}: {exc}")
    return errors


def exact_duplicate_python_groups(paths: list[Path]) -> list[list[Path]]:
    by_digest: dict[str, list[Path]] = defaultdict(list)
    for path in paths:
        if path.name == "__init__.py":
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        by_digest[digest].append(path)
    return [group for group in by_digest.values() if len(group) > 1]


def root_python_files() -> list[Path]:
    return sorted(ROOT.glob("*.py"))


def missing_required_files() -> list[str]:
    return [name for name in REQUIRED_FILES if not (ROOT / name).is_file()]


def joint_input_paths() -> list[Path]:
    paths: list[Path] = []
    modes = (
        "power2_baseline",
        "power2_roi_power_control",
        "power2_roi_ri4",
    )
    for fold in range(1, 7):
        fold_tag = f"fold{fold:02d}"
        paths.append(
            ROOT
            / "results"
            / "experiments"
            / f"bc_dpg_v3_scan_target_v4_{fold_tag}_seed42"
            / "tables"
            / "base_threshold_test_predictions.csv"
        )
        for mode in modes:
            paths.append(
                ROOT
                / "results"
                / "experiments"
                / f"roi_polar_stage4_v1_{mode}_v4_{fold_tag}_seed42_formal"
                / "tables"
                / "test_predictions.csv"
            )
    return paths


def print_result(label: str, ok: bool, detail: str) -> None:
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {label}: {detail}")


def main() -> int:
    args = parse_args()
    python_files = active_python_files()
    errors = syntax_errors(python_files)
    duplicate_groups = exact_duplicate_python_groups(python_files)
    root_entrypoints = root_python_files()
    missing = missing_required_files()

    print_result(
        "active Python syntax",
        not errors,
        f"{len(python_files)} files checked, {len(errors)} errors",
    )
    for error in errors:
        print(f"  {error}")

    print_result(
        "active Python uniqueness",
        not duplicate_groups,
        f"{len(duplicate_groups)} exact duplicate groups",
    )
    for group in duplicate_groups:
        names = ", ".join(str(path.relative_to(ROOT)) for path in group)
        print(f"  duplicate: {names}")

    print_result(
        "root Python entrypoints",
        not root_entrypoints,
        f"{len(root_entrypoints)} root-level Python files",
    )
    for path in root_entrypoints:
        print(f"  move into package or scripts/: {path.name}")

    print_result(
        "required project files",
        not missing,
        f"{len(REQUIRED_FILES) - len(missing)}/{len(REQUIRED_FILES)} present",
    )
    for name in missing:
        print(f"  missing: {name}")

    joint_missing: list[Path] = []
    if args.require_joint_inputs:
        joint_paths = joint_input_paths()
        joint_missing = [path for path in joint_paths if not path.is_file()]
        print_result(
            "six-fold joint inputs",
            not joint_missing,
            f"{len(joint_paths) - len(joint_missing)}/{len(joint_paths)} present",
        )
        for path in joint_missing:
            print(f"  missing: {path.relative_to(ROOT)}")

    return 1 if (
        errors
        or duplicate_groups
        or root_entrypoints
        or missing
        or joint_missing
    ) else 0


if __name__ == "__main__":
    sys.exit(main())
