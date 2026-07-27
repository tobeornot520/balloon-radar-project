#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_NAME = "balloon_radar_project_share_20260727"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "dist" / PACKAGE_NAME
DEFAULT_ZIP_PATH = PROJECT_ROOT / "dist" / f"{PACKAGE_NAME}.zip"
PACKAGE_DATE = "2026-07-27"
ZIP_TIMESTAMP = (2026, 7, 27, 0, 0, 0)

ALLOWED_SUFFIXES = {".md", ".csv", ".png", ".pdf", ".json", ".txt"}
FORBIDDEN_SUFFIXES = {
    ".mat",
    ".h5",
    ".hdf5",
    ".npy",
    ".npz",
    ".pt",
    ".pth",
    ".ckpt",
    ".onnx",
    ".log",
    ".doc",
    ".docx",
    ".zip",
    ".rar",
    ".7z",
    ".tar",
    ".gz",
}
FORBIDDEN_NAME_PARTS = {
    ".git",
    "raw_transcript",
    "chat_transcript",
    "聊天记录",
}
SENSITIVE_TEXT_MARKERS = (
    "/home/",
    "tobeornot8259748",
    "C:\\Users\\",
    "BEGIN PRIVATE KEY",
    "api_key=",
    "password=",
    "secret=",
)
TEXT_SUFFIXES = {".md", ".csv", ".json", ".txt"}
MARKDOWN_LINK_PATTERN = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")


@dataclass(frozen=True)
class PackageFile:
    source: str
    destination: str
    category: str


PACKAGE_FILES = (
    PackageFile("docs/share/README_SHARE_ZH.md", "README.md", "share_document"),
    PackageFile(
        "docs/share/01_PROJECT_OVERVIEW_ZH.md",
        "docs/01_PROJECT_OVERVIEW_ZH.md",
        "share_document",
    ),
    PackageFile(
        "docs/share/02_DEVELOPMENT_HISTORY_ZH.md",
        "docs/02_DEVELOPMENT_HISTORY_ZH.md",
        "share_document",
    ),
    PackageFile(
        "docs/share/03_RESULTS_AND_EVIDENCE_ZH.md",
        "docs/03_RESULTS_AND_EVIDENCE_ZH.md",
        "share_document",
    ),
    PackageFile(
        "docs/share/04_SHARING_TALK_TRACK_ZH.md",
        "docs/04_SHARING_TALK_TRACK_ZH.md",
        "share_document",
    ),
    PackageFile(
        "docs/share/05_REPRODUCTION_AND_NEXT_STEPS_ZH.md",
        "docs/05_REPRODUCTION_AND_NEXT_STEPS_ZH.md",
        "share_document",
    ),
    PackageFile(
        "docs/DATA_CARD.md",
        "docs/06_DATA_CARD_ZH.md",
        "governance_document",
    ),
    PackageFile(
        "docs/METRIC_DEFINITIONS.md",
        "docs/07_METRIC_DEFINITIONS_ZH.md",
        "governance_document",
    ),
    PackageFile(
        "docs/MODEL_SELECTION_LEDGER.md",
        "docs/08_MODEL_SELECTION_LEDGER_ZH.md",
        "governance_document",
    ),
    PackageFile(
        "results/final_evidence/bc_dpg_v3_final/FINAL_EVIDENCE_REPORT.md",
        "evidence/01_BC_DPG_V3_FINAL_REPORT.md",
        "frozen_report",
    ),
    PackageFile(
        "docs/polarimetric_stage3/STAGE3_FROZEN_CONCLUSION.md",
        "evidence/02_POLARIMETRIC_STAGE3_FROZEN_CONCLUSION.md",
        "frozen_report",
    ),
    PackageFile(
        "results/final_evidence/roi_stage4_twofold/STAGE4_TWOFOLD_FROZEN_ANALYSIS.md",
        "evidence/03_ROI_STAGE4_TWOFOLD_FROZEN_ANALYSIS.md",
        "frozen_report",
    ),
    PackageFile(
        "results/final_evidence/roi_bc_dpg_joint_fixed_threshold/JOINT_AUDIT_REPORT.md",
        "evidence/04_ROI_BC_DPG_JOINT_AUDIT_REPORT.md",
        "frozen_report",
    ),
    PackageFile(
        "results/data_audit/bc_dpg_v3_causal_context_audit/CAUSAL_CONTEXT_AUDIT.md",
        "evidence/05_BC_DPG_V3_CAUSAL_CONTEXT_AUDIT.md",
        "post_test_sensitivity_report",
    ),
    PackageFile(
        "results/final_evidence/bc_dpg_v3_final/figures/fig1_deployment_false_alarms.png",
        "assets/figures/bc_dpg_deployment_false_alarms.png",
        "figure",
    ),
    PackageFile(
        "results/final_evidence/bc_dpg_v3_final/figures/fig2_ablation_false_alarms.png",
        "assets/figures/bc_dpg_ablation_false_alarms.png",
        "figure",
    ),
    PackageFile(
        "results/final_evidence/bc_dpg_v3_final/figures/fig3_false_alarm_target_shift_tradeoff.png",
        "assets/figures/bc_dpg_false_alarm_target_shift_tradeoff.png",
        "figure",
    ),
    PackageFile(
        "results/final_evidence/roi_bc_dpg_joint_fixed_threshold/figures/fig1_pooled_detection_tradeoff.png",
        "assets/figures/joint_pooled_detection_tradeoff.png",
        "figure",
    ),
    PackageFile(
        "results/final_evidence/roi_bc_dpg_joint_fixed_threshold/figures/fig1_pooled_detection_tradeoff.pdf",
        "assets/figures/joint_pooled_detection_tradeoff.pdf",
        "figure",
    ),
    PackageFile(
        "results/final_evidence/roi_bc_dpg_joint_fixed_threshold/figures/fig2_fold_heterogeneity.png",
        "assets/figures/joint_fold_heterogeneity.png",
        "figure",
    ),
    PackageFile(
        "results/final_evidence/roi_bc_dpg_joint_fixed_threshold/figures/fig2_fold_heterogeneity.pdf",
        "assets/figures/joint_fold_heterogeneity.pdf",
        "figure",
    ),
    PackageFile(
        "results/final_evidence/roi_bc_dpg_joint_fixed_threshold/figures/fig3_complementarity.png",
        "assets/figures/joint_complementarity.png",
        "figure",
    ),
    PackageFile(
        "results/final_evidence/roi_bc_dpg_joint_fixed_threshold/figures/fig3_complementarity.pdf",
        "assets/figures/joint_complementarity.pdf",
        "figure",
    ),
    PackageFile(
        "results/final_evidence/bc_dpg_v3_final/tables/table_01_main_model_comparison.csv",
        "assets/tables/bc_dpg_main_model_comparison.csv",
        "table",
    ),
    PackageFile(
        "results/final_evidence/bc_dpg_v3_final/tables/table_03_ablation_summary.csv",
        "assets/tables/bc_dpg_ablation_summary.csv",
        "table",
    ),
    PackageFile(
        "results/final_evidence/roi_stage4_twofold/paper_table_stage4_twofold_main.csv",
        "assets/tables/roi_stage4_twofold_main.csv",
        "table",
    ),
    PackageFile(
        "results/final_evidence/roi_bc_dpg_joint_fixed_threshold/tables/table_01_pooled_detection.csv",
        "assets/tables/joint_pooled_detection.csv",
        "table",
    ),
    PackageFile(
        "results/final_evidence/roi_bc_dpg_joint_fixed_threshold/tables/table_02_fold_detection.csv",
        "assets/tables/joint_fold_detection.csv",
        "table",
    ),
    PackageFile(
        "results/final_evidence/roi_bc_dpg_joint_fixed_threshold/tables/table_03_complementarity.csv",
        "assets/tables/joint_complementarity.csv",
        "table",
    ),
    PackageFile(
        "results/final_evidence/roi_bc_dpg_joint_fixed_threshold/tables/table_04_simple_combination_diagnostics.csv",
        "assets/tables/joint_combination_diagnostics.csv",
        "table",
    ),
    PackageFile(
        "results/final_evidence/roi_bc_dpg_joint_fixed_threshold/tables/table_05_claim_boundaries.csv",
        "assets/tables/joint_claim_boundaries.csv",
        "table",
    ),
    PackageFile(
        "results/final_evidence/roi_bc_dpg_joint_fixed_threshold/tables/table_06_fold_distribution_summary.csv",
        "assets/tables/joint_fold_distribution_summary.csv",
        "table",
    ),
    PackageFile(
        "results/final_evidence/roi_bc_dpg_joint_fixed_threshold/tables/table_07_derived_metrics_and_wilson_ci.csv",
        "assets/tables/joint_derived_metrics_and_wilson_ci.csv",
        "table",
    ),
    PackageFile(
        "results/final_evidence/roi_bc_dpg_joint_fixed_threshold/tables/table_08_scan_group_bootstrap.csv",
        "assets/tables/joint_scan_group_bootstrap.csv",
        "table",
    ),
    PackageFile(
        "results/final_evidence/roi_bc_dpg_joint_fixed_threshold/tables/table_09_paired_mcnemar_diagnostics.csv",
        "assets/tables/joint_paired_mcnemar_diagnostics.csv",
        "table",
    ),
    PackageFile(
        "results/data_audit/bc_dpg_v3_causal_context_audit/context_metrics_aggregate.csv",
        "assets/tables/bc_dpg_causal_context_aggregate.csv",
        "post_test_sensitivity_table",
    ),
    PackageFile(
        "results/data_audit/bc_dpg_v3_causal_context_audit/paired_deltas_vs_complete_scan.csv",
        "assets/tables/bc_dpg_causal_context_paired_deltas.csv",
        "post_test_sensitivity_table",
    ),
    PackageFile(
        "results/data_audit/bc_dpg_v3_causal_context_audit/complete_replay_validation.csv",
        "assets/tables/bc_dpg_causal_context_replay_validation.csv",
        "replay_validation_table",
    ),
    PackageFile(
        "results/data_audit/bc_dpg_v3_causal_context_audit/history_coverage_by_fold.csv",
        "assets/tables/bc_dpg_causal_context_history_coverage.csv",
        "post_test_sensitivity_table",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a sanitized project-overview share package without raw data, "
            "checkpoints, predictions, logs, or development transcripts."
        )
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--zip-path", type=Path, default=DEFAULT_ZIP_PATH)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing package directory and ZIP after a successful build.",
    )
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    path = path.expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def current_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def validate_source_map(files: Iterable[PackageFile] = PACKAGE_FILES) -> None:
    files = tuple(files)
    destinations = [item.destination for item in files]
    if len(destinations) != len(set(destinations)):
        raise ValueError("Package file map contains duplicate destinations")

    missing: list[str] = []
    errors: list[str] = []
    for item in files:
        source = PROJECT_ROOT / item.source
        destination = Path(item.destination)
        if not source.is_file():
            missing.append(item.source)
        if destination.is_absolute() or ".." in destination.parts:
            errors.append(f"unsafe destination: {item.destination}")
        if destination.suffix.lower() not in ALLOWED_SUFFIXES:
            errors.append(f"unsupported destination type: {item.destination}")
        if destination.suffix.lower() in FORBIDDEN_SUFFIXES:
            errors.append(f"forbidden destination type: {item.destination}")
    if missing:
        raise FileNotFoundError(f"Missing share-package sources: {missing}")
    if errors:
        raise ValueError("Invalid share-package mapping: " + "; ".join(errors))


def ensure_output_available(output_dir: Path, zip_path: Path, overwrite: bool) -> None:
    if output_dir.exists() and not output_dir.is_dir():
        raise FileExistsError(f"Package output is not a directory: {output_dir}")
    if output_dir.is_dir() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Package directory is nonempty: {output_dir}. Use --overwrite to replace it."
        )
    if zip_path.exists() and not overwrite:
        raise FileExistsError(
            f"Package ZIP already exists: {zip_path}. Use --overwrite to replace it."
        )


def validate_output_paths(output_dir: Path, zip_path: Path) -> None:
    output_dir = output_dir.resolve()
    zip_path = zip_path.resolve()
    if output_dir == PROJECT_ROOT or output_dir in PROJECT_ROOT.parents:
        raise ValueError("Package output cannot be the repository or its parent")
    if zip_path == PROJECT_ROOT or zip_path.is_dir():
        raise ValueError("ZIP output must be a file path outside the package directory")
    if zip_path == output_dir or output_dir in zip_path.parents:
        raise ValueError("ZIP output must not be inside the package directory")


def copy_package_files(staging_dir: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for item in PACKAGE_FILES:
        source = PROJECT_ROOT / item.source
        destination = staging_dir / item.destination
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        records.append(
            {
                "category": item.category,
                "packaged_path": item.destination,
                "repository_source": item.source,
                "size_bytes": destination.stat().st_size,
                "sha256": sha256_file(destination),
            }
        )
    return records


def write_manifest(staging_dir: Path, records: list[dict[str, object]]) -> None:
    manifest = {
        "schema_version": 1,
        "package_name": PACKAGE_NAME,
        "package_date": PACKAGE_DATE,
        "language": "zh-CN",
        "purpose": (
            "sanitized, traceable, and hash-verifiable frozen-result excerpts; "
            "not a self-contained reproduction package"
        ),
        "source_commit": current_commit(),
        "current_scope": "H/V UAV detection, localization, and false-alarm suppression front end",
        "long_term_scope": "balloon payload and motion-state recognition after new data collection",
        "evidence_rules": {
            "test_threshold_retuning": False,
            "joint_model_trained": False,
            "and_or_rules_selected": False,
            "complete_scan_bc_is_causal": False,
            "causal_context_audit_role": "post-hoc frozen-checkpoint sensitivity",
            "causal_context_retraining_performed": False,
            "causal_history_window_selected": False,
            "leave_one_out_is_causal": False,
            "past_only_order_verified_by_timestamp": False,
            "past_only_order_columns": ["beam_layer", "azimuth_deg", "sample_id"],
            "evaluation_role": "internal development estimate",
            "stage4_development_folds_reused_in_sixfold": [1, 4],
            "class_and_acquisition_date_confounded": True,
        },
        "full_reproduction_requires": [
            "internal source code",
            "raw data and manifests",
            "sample-level frozen predictions",
            "model checkpoints",
        ],
        "excluded_content": [
            "raw MAT or IQ data",
            "sample labels and sample-level predictions",
            "model checkpoints and training logs",
            "development transcripts and local recovery archives",
            "personal paths and credentials",
        ],
        "files": sorted(records, key=lambda record: str(record["packaged_path"])),
    }
    (staging_dir / "MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_checksums(staging_dir: Path) -> None:
    files = sorted(
        path
        for path in staging_dir.rglob("*")
        if path.is_file() and path.name != "SHA256SUMS.txt"
    )
    lines = [
        f"{sha256_file(path)}  {path.relative_to(staging_dir).as_posix()}"
        for path in files
    ]
    (staging_dir / "SHA256SUMS.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def audit_package_directory(package_dir: Path) -> dict[str, object]:
    errors: list[str] = []
    files = sorted(path for path in package_dir.rglob("*") if path.is_file())
    for path in package_dir.rglob("*"):
        relative = path.relative_to(package_dir).as_posix()
        lower = relative.lower()
        if path.is_symlink():
            errors.append(f"symlink is not allowed: {relative}")
        if any(part.lower() in lower for part in FORBIDDEN_NAME_PARTS):
            errors.append(f"forbidden name in package: {relative}")
        if path.is_file():
            suffix = path.suffix.lower()
            if suffix in FORBIDDEN_SUFFIXES or suffix not in ALLOWED_SUFFIXES:
                errors.append(f"forbidden or unsupported file type: {relative}")
            if path.stat().st_size > 10 * 1024 * 1024:
                errors.append(f"unexpected large file: {relative}")
            if suffix in TEXT_SUFFIXES:
                text = path.read_text(encoding="utf-8-sig")
                for marker in SENSITIVE_TEXT_MARKERS:
                    if marker.lower() in text.lower():
                        errors.append(f"sensitive marker {marker!r} in {relative}")
    if errors:
        raise ValueError("Share-package audit failed: " + "; ".join(errors))
    link_count = audit_markdown_links(package_dir)
    return {
        "status": "PASS",
        "file_count": len(files),
        "size_bytes": sum(path.stat().st_size for path in files),
        "raw_data_included": False,
        "checkpoints_included": False,
        "sample_predictions_included": False,
        "development_transcripts_included": False,
        "sensitive_markers_found": 0,
        "markdown_links_checked": link_count,
    }


def audit_markdown_links(package_dir: Path) -> int:
    errors: list[str] = []
    checked = 0
    package_dir = package_dir.resolve()
    for markdown in sorted(package_dir.rglob("*.md")):
        text = markdown.read_text(encoding="utf-8-sig")
        for match in MARKDOWN_LINK_PATTERN.finditer(text):
            target = match.group(1).strip().strip("<>")
            if not target or target.startswith(("#", "http://", "https://", "mailto:")):
                continue
            target = target.split("#", 1)[0]
            resolved = (markdown.parent / target).resolve()
            checked += 1
            try:
                resolved.relative_to(package_dir)
            except ValueError:
                errors.append(
                    f"link escapes package: {markdown.relative_to(package_dir)} -> {target}"
                )
                continue
            if not resolved.is_file():
                errors.append(
                    f"missing link target: {markdown.relative_to(package_dir)} -> {target}"
                )
    if errors:
        raise ValueError("Share-package Markdown link audit failed: " + "; ".join(errors))
    return checked


def write_deterministic_zip(package_dir: Path, zip_path: Path) -> None:
    with zipfile.ZipFile(
        zip_path,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as archive:
        for path in sorted(item for item in package_dir.rglob("*") if item.is_file()):
            relative = path.relative_to(package_dir).as_posix()
            info = zipfile.ZipInfo(f"{package_dir.name}/{relative}", ZIP_TIMESTAMP)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = (0o100644 & 0xFFFF) << 16
            archive.writestr(info, path.read_bytes(), compress_type=zipfile.ZIP_DEFLATED)


def audit_zip(zip_path: Path, package_name: str) -> None:
    with zipfile.ZipFile(zip_path) as archive:
        bad_file = archive.testzip()
        if bad_file is not None:
            raise ValueError(f"ZIP integrity check failed at {bad_file}")
        names = archive.namelist()
        expected_prefix = f"{package_name}/"
        if not names or any(
            not name.startswith(expected_prefix)
            or Path(name).is_absolute()
            or ".." in Path(name).parts
            for name in names
        ):
            raise ValueError("ZIP contains an unsafe or unexpected archive path")


def build_share_package(
    output_dir: Path,
    zip_path: Path,
    overwrite: bool = False,
) -> tuple[Path, Path, dict[str, object]]:
    output_dir = resolve_path(output_dir)
    zip_path = resolve_path(zip_path)
    validate_source_map()
    validate_output_paths(output_dir, zip_path)
    ensure_output_available(output_dir, zip_path, overwrite)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    staging_parent = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.build-", dir=output_dir.parent)
    )
    staging_dir = staging_parent / output_dir.name
    staging_zip = staging_parent / zip_path.name
    staging_dir.mkdir()
    try:
        records = copy_package_files(staging_dir)
        write_manifest(staging_dir, records)
        write_checksums(staging_dir)
        audit = audit_package_directory(staging_dir)
        write_deterministic_zip(staging_dir, staging_zip)
        audit_zip(staging_zip, staging_dir.name)

        if output_dir.exists():
            shutil.rmtree(output_dir)
        if zip_path.exists():
            zip_path.unlink()
        staging_dir.replace(output_dir)
        staging_zip.replace(zip_path)
    except Exception:
        shutil.rmtree(staging_parent, ignore_errors=True)
        raise
    shutil.rmtree(staging_parent, ignore_errors=True)
    return output_dir, zip_path, audit


def main() -> int:
    args = parse_args()
    try:
        output_dir, zip_path, audit = build_share_package(
            args.output_dir, args.zip_path, args.overwrite
        )
    except (FileNotFoundError, FileExistsError, ValueError, OSError) as exc:
        print(f"[ERROR] {exc}")
        return 1

    print("Project share package: PASS")
    print(f"directory={output_dir}")
    print(f"zip={zip_path}")
    print(f"files={audit['file_count']}")
    print(f"size_bytes={audit['size_bytes']}")
    print("raw_data_included=False")
    print("checkpoints_included=False")
    print("development_transcripts_included=False")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
