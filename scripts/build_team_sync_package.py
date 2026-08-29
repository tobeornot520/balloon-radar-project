#!/usr/bin/env python3
"""Build the complete data-free project synchronization package for team members."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_DATE = "20260829"
OUTPUT_NAME = f"team_member_full_sync_{PACKAGE_DATE}"
ZIP_TIMESTAMP = (2026, 8, 29, 0, 0, 0)
MAX_PACKAGE_BYTES = 20 * 1024 * 1024
TEXT_SUFFIXES = {".md", ".json", ".yaml", ".yml", ".csv", ".txt", ".py"}
FORBIDDEN_SUFFIXES = {
    ".mat", ".h5", ".hdf5", ".npy", ".npz", ".pcd", ".pt", ".pth", ".ckpt",
    ".onnx", ".zip", ".rar", ".7z", ".tar", ".gz", ".pdf", ".pptx", ".doc", ".docx",
}
SENSITIVE_MARKERS = (
    b"/home/", b"tobeornot8259748", b"BEGIN PRIVATE KEY", b"api_key=",
    b"password=", b"secret=",
)
ABSOLUTE_HOME_PATH = re.compile(rb"/home/[^\"'\s]+")
EXCLUDED_RELATIVE_PATHS = {
    "scripts/build_project_share_package.py",
    "scripts/build_algorithm_code_share_package.py",
    "scripts/build_team_sync_package.py",
    "tests/test_share_package.py",
    "tests/test_algorithm_code_share_package.py",
}


@dataclass(frozen=True)
class Package:
    name: str
    title: str
    description: str
    files: tuple[str, ...]


def collect_tree(root: str, suffixes: set[str]) -> tuple[str, ...]:
    source_root = PROJECT_ROOT / root
    return tuple(
        path.relative_to(PROJECT_ROOT).as_posix()
        for path in sorted(source_root.rglob("*"))
        if path.is_file()
        and path.suffix.lower() in suffixes
        and path.relative_to(PROJECT_ROOT).as_posix() not in EXCLUDED_RELATIVE_PATHS
    )


def collect_member_evidence() -> tuple[str, ...]:
    """Return reports and aggregate evidence, excluding provenance-only internals."""
    evidence_root = PROJECT_ROOT / "results/final_evidence"
    excluded_names = {
        "SHA256SUMS.txt",
        "localization_manifest.json",
        "final_evidence_audit.json",
        "evidence_manifest.json",
        "model_fit_manifest.json",
        "gate_decision.json",
        "lat_mricd_cross_band_transfer_v1.run_consumed.json",
        "table_05_regularization_validation_candidates.csv",
        "table_06_regularization_selected_by_fold.csv",
        "table_06_source_files.csv",
        "table_11_file_hashes.csv",
        "table_12_checkpoint_hashes.csv",
    }
    return tuple(
        path.relative_to(PROJECT_ROOT).as_posix()
        for path in sorted(evidence_root.rglob("*"))
        if path.is_file()
        and path.suffix.lower() in {".md", ".csv", ".png"}
        and "source_evidence" not in path.relative_to(evidence_root).parts
        and path.name not in excluded_names
    )


DOCUMENT_FILES = (
    "README.md",
    "PROJECT_CONTROL/README.md",
    "PROJECT_CONTROL/PROJECT_LOG.md",
    "PROJECT_CONTROL/ROADMAP.md",
    "PROJECT_CONTROL/TASK_BOARD.md",
    "PROJECT_CONTROL/TECHNICAL_HANDBOOK_ZH.md",
    *collect_tree("docs", {".md"}),
    "paper/README.md",
    "paper/TIAN_2024_reproduction_topic/README.md",
    "paper/TIAN_2024_reproduction_topic/reproduction_results_package/README.md",
    "paper/TIAN_2024_reproduction_topic/reproduction_results_package/ACTUAL_RESULTS_FOR_SENIOR_ZH.md",
    "paper/TIAN_2024_reproduction_topic/reproduction_results_package/FAILURE_REASON_SIMPLE_ZH.md",
    "paper/TIAN_2024_reproduction_topic/reproduction_results_package/RESULT_SUMMARY_ZH.md",
    *collect_member_evidence(),
)

ENGINE_FILES = (
    "README.md",
    "environment.yml",
    "requirements-lock.txt",
    "pytest.ini",
    "code/README.md",
    "configs/README.md",
    "scripts/README.md",
    *collect_tree("baselines", {".py"}),
    *collect_tree("configs", {".json", ".yaml", ".yml", ".csv", ".md"}),
    *collect_tree("datasets", {".py"}),
    *collect_tree("evaluation", {".py"}),
    *collect_tree("features", {".py"}),
    *collect_tree("models", {".py"}),
    *collect_tree("training", {".py"}),
    *collect_tree("utils", {".py"}),
    *collect_tree("scripts", {".py", ".sh"}),
    *collect_tree("tests", {".py"}),
)

FIELD_FILES = (
    "docs/FIELD_CAPABILITY_REQUEST_V1.md",
    "docs/FIELD_COLLECTION_SOP_V1.md",
    "docs/FIELD_IQ_INTEGRITY_PROBE_V1.md",
    "docs/FIELD_SYNCHRONIZATION_AUDIT_V1.md",
    "docs/NEW_DATA_COLLECTION_PROTOCOL.md",
    "docs/EXTERNAL_FACT_REQUEST_MESSAGE_V1.md",
    "PROJECT_CONTROL/meetings/2026-08-26_PROJECT_PLANNING_MEMO.md",
    "configs/data_collection_contract_v1.json",
    "configs/data_collection_manifest_template_v1.csv",
    "configs/field_capability_response_template_v1.csv",
    "configs/field_iq_probe_contract_template_v1.json",
    "configs/field_readiness_checklist_v1.json",
    "configs/field_readiness_evidence_template_v1.csv",
    "configs/field_sync_event_contract_v1.json",
    "configs/field_sync_event_template_v1.csv",
    "configs/pilot_scenario_matrix_v1.csv",
    "configs/pilot_session_log_template_v1.csv",
)

PACKAGES = (
    Package(
        "01_项目资料与研究证据",
        "项目资料与研究证据",
        "完整项目叙事、文档、课程、任务台账、论文边界和冻结聚合证据。",
        tuple(dict.fromkeys(DOCUMENT_FILES)),
    ),
    Package(
        "02_工程源码与配置",
        "工程源码与配置",
        "当前核心工程源码、配置、训练/评价入口和自动测试。",
        tuple(dict.fromkeys(ENGINE_FILES)),
    ),
    Package(
        "03_外场数据合同与执行模板",
        "外场数据合同与执行模板",
        "设备确认、同步、标定、Pilot、新数据清单及现场分工材料。",
        FIELD_FILES,
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "dist")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def git_head() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
        ).strip()
    except subprocess.CalledProcessError:
        return None


def package_bytes(relative: str) -> tuple[Path, bytes, str]:
    pure = PurePosixPath(relative)
    if pure.is_absolute() or ".." in pure.parts:
        raise ValueError(f"Unsafe source path: {relative}")
    source = PROJECT_ROOT / pure
    if not source.is_file():
        raise FileNotFoundError(f"Missing package source: {relative}")
    if source.suffix.lower() in FORBIDDEN_SUFFIXES:
        raise ValueError(f"Forbidden package source type: {relative}")
    data = source.read_bytes()
    if source.suffix.lower() not in TEXT_SUFFIXES:
        if any(marker in data for marker in SENSITIVE_MARKERS):
            raise ValueError(f"Sensitive marker in binary source: {relative}")
        return source, data, "copied"

    sanitized = ABSOLUTE_HOME_PATH.sub(b"<controlled-path>", data)
    sanitized = sanitized.replace(b"tobeornot8259748", b"<redacted-user>")
    if b"BEGIN PRIVATE KEY" in sanitized:
        raise ValueError(f"Private key marker in source: {relative}")
    for marker in (b"api_key=", b"password=", b"secret="):
        if marker in sanitized:
            raise ValueError(f"Potential credential marker in source: {relative}")
    transformation = "sanitized_local_paths" if sanitized != data else "copied"
    return source, sanitized, transformation


def readme(package: Package) -> bytes:
    return f"""# {package.title}\n\n{package.description}\n\n## 使用边界\n\n- 这是全员项目同步材料，不含原始雷达数据、视频、权重、逐样本预测和合作方受控资料。\n- 当前模型结果主要是 H/V UAV/背景的内部开发证据；空飘球识别、严格实时和外部泛化仍待新数据验证。\n- 解压后先阅读根 `README.md`、`docs/share/TEAM_SYNC_BRIEF_20260826.md` 和 `PROJECT_CONTROL/TASK_BOARD.md`。\n- 完整训练或受控数据重放需要另行申请相应数据和 checkpoint。\n""".encode("utf-8")


def write_zip(package_dir: Path, zip_path: Path) -> None:
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for source in sorted(path for path in package_dir.rglob("*") if path.is_file()):
            relative = source.relative_to(package_dir.parent).as_posix()
            info = zipfile.ZipInfo(relative, ZIP_TIMESTAMP)
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, source.read_bytes())


def build_package(package: Package, output_root: Path, head: str | None) -> dict[str, object]:
    package_dir = output_root / package.name
    package_dir.mkdir(parents=True, exist_ok=False)
    records: list[dict[str, object]] = []
    for relative in package.files:
        source, data, transformation = package_bytes(relative)
        destination = package_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
        records.append(
            {
                "path": relative,
                "source_sha256": sha256_bytes(source.read_bytes()),
                "packaged_sha256": sha256_bytes(data),
                "size_bytes": len(data),
                "transformation": transformation,
            }
        )
    readme_data = readme(package)
    (package_dir / "README.md").write_bytes(readme_data)
    records.append({"path": "README.md", "size_bytes": len(readme_data), "sha256": sha256_bytes(readme_data)})
    manifest = {
        "package_name": package.name,
        "package_date": PACKAGE_DATE,
        "scope": "complete data-free project synchronization for current team members",
        "source_snapshot": "working_tree_snapshot; file hashes are authoritative",
        "git_head_at_build": head,
        "files": sorted(records, key=lambda item: str(item["path"])),
    }
    (package_dir / "MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    checksums = [
        f"{sha256_bytes(path.read_bytes())}  {path.relative_to(package_dir).as_posix()}"
        for path in sorted(path for path in package_dir.rglob("*") if path.is_file())
    ]
    (package_dir / "SHA256SUMS.txt").write_text("\n".join(checksums) + "\n", encoding="utf-8")
    zip_path = output_root / f"{package.name}_{PACKAGE_DATE}.zip"
    write_zip(package_dir, zip_path)
    shutil.rmtree(package_dir)
    if zip_path.stat().st_size > MAX_PACKAGE_BYTES:
        raise ValueError(f"Package exceeds 20 MiB: {zip_path.name}")
    return {"name": package.name, "zip": zip_path.name, "zip_size_bytes": zip_path.stat().st_size, "file_count": len(records)}


def main() -> None:
    args = parse_args()
    output_root = args.output_root.resolve() / OUTPUT_NAME
    if output_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output already exists: {output_root}; pass --overwrite")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True)
    head = git_head()
    rows = [build_package(package, output_root, head) for package in PACKAGES]
    guide, guide_data, _ = package_bytes("docs/share/TEAM_MEMBER_FULL_SYNC_GUIDE_20260829.md")
    del guide
    (output_root / "README_全员同步包说明.md").write_bytes(guide_data)
    (output_root / "CATALOG.json").write_text(
        json.dumps(
            {
                "package_date": PACKAGE_DATE,
                "scope": "complete data-free project synchronization",
                "source_snapshot": "working_tree_snapshot; file hashes are authoritative",
                "git_head_at_build": head,
                "packages": rows,
                "excluded": ["raw data", "model weights", "private field materials", "historical chat exports", "reference PDFs/PPTs"],
            },
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    checksums = [
        f"{sha256_bytes(path.read_bytes())}  {path.name}"
        for path in sorted(output_root.glob("*.zip"))
    ]
    (output_root / "SHA256SUMS.txt").write_text("\n".join(checksums) + "\n", encoding="utf-8")
    print(f"Full team synchronization package: PASS\nOutput: {output_root}")
    for row in rows:
        print(f"- {row['zip']}: {row['zip_size_bytes']} bytes, {row['file_count']} files")


if __name__ == "__main__":
    main()
