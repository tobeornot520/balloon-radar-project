#!/usr/bin/env python3
"""Build a provenance-bound, data-free algorithm code review package."""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import io
import json
import shutil
import subprocess
import tempfile
import zipfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_NAME = "balloon_radar_algorithm_code_review_20260804_v1"

ROOT_FILES = {
    "README.md",
    "environment.yml",
    "pytest.ini",
    "requirements-lock.txt",
}

ACTIVE_PREFIXES = (
    "baselines/",
    "configs/",
    "datasets/",
    "evaluation/",
    "features/",
    "models/",
    "scripts/",
    "tests/",
    "training/",
    "utils/",
)

TOOLS_SUFFIXES = {".md", ".py", ".sh", ".txt"}

REVIEW_DOCUMENTS = {
    "docs/share/00_ONE_PAGE_SUMMARY_ZH.md": "review/00_ONE_PAGE_SUMMARY_ZH.md",
    "docs/share/03_RESULTS_AND_EVIDENCE_ZH.md": "review/01_RESULTS_AND_EVIDENCE_ZH.md",
    "docs/share/04_SHARING_TALK_TRACK_ZH.md": "review/02_SHARING_TALK_TRACK_ZH.md",
    "docs/share/07_QUESTIONS_FOR_SENIOR_ZH.md": "review/03_QUESTIONS_FOR_SENIOR_ZH.md",
    "docs/share/08_DATA_REQUEST_CHECKLIST_ZH.md": "review/04_DATA_REQUEST_CHECKLIST_ZH.md",
}

# These tracked recovery copies fill imports used only by early workflows. They
# remain provenance-labelled compatibility sources, not silently promoted code.
COMPATIBILITY_SOURCES = {
    "_cleanup_archive/final_cleanup_20260723_174442/datasets/radar_dataset.py": (
        "project/datasets/radar_dataset.py"
    ),
    "_cleanup_archive/final_cleanup_20260723_174442/datasets/"
    "polarimetric_detection_dataset_v1.py": (
        "project/datasets/polarimetric_detection_dataset_v1.py"
    ),
    "_cleanup_archive/final_cleanup_20260723_174442/training/"
    "train_background_calibrator.py": (
        "project/training/train_background_calibrator.py"
    ),
    "_cleanup_archive/final_cleanup_20260723_174442/training/"
    "train_background_tail_calibrator.py": (
        "project/training/train_background_tail_calibrator.py"
    ),
}

FORBIDDEN_SOURCE_PREFIXES = (
    "data/",
    "results/",
    "checkpoints/",
    "logs/",
    "dist/",
    "payload/",
    "notes/development_history/",
    "\u53c2\u8003\u8d44\u6599/",
    "\u5f53\u524d\u5f00\u53d1\u8fc7\u7a0b/",
    "\u6709\u7528\uff1f/",
)

FORBIDDEN_PACKAGED_SUFFIXES = {
    ".bin",
    ".ckpt",
    ".h5",
    ".mat",
    ".npy",
    ".npz",
    ".pcd",
    ".pt",
    ".pth",
    ".rar",
    ".tar",
}

ACTUAL_PRIVATE_MARKERS = (
    b"/home/tobeornot8259748/",
    b"C:\\Users\\tobeornot8259748\\",
)

INTERNAL_TOP_LEVEL = {
    "baselines",
    "datasets",
    "evaluation",
    "features",
    "models",
    "scripts",
    "training",
    "utils",
}

SMOKE_TESTS = (
    "tests/test_multidomain_radar_features.py",
    "tests/test_multidomain_feature_fusion.py",
    "tests/test_polarimetric_transfer_encoder.py",
    "tests/test_scan_context.py",
    "tests/test_thesis_tian2024_adapter.py",
    "tests/test_tian_fcn_reproduction.py",
    "tests/test_zero_doppler_mechanisms.py",
)


@dataclass(frozen=True)
class AlgorithmRecord:
    family: str
    status: str
    primary_entry: str
    core_files: str
    required_input: str
    evidence_boundary: str


@dataclass(frozen=True)
class FileRecord:
    packaged_path: str
    source_path: str | None
    category: str
    size_bytes: int
    sha256: str
    transformation: str


ALGORITHMS = (
    AlgorithmRecord(
        "CA-CFAR",
        "classical baseline",
        "project/scripts/evaluate_cfar.py",
        "baselines/ca_cfar.py",
        "manifest plus radar arrays",
        "baseline only; current package has no raw data",
    ),
    AlgorithmRecord(
        "DPG-FCN detection/localization",
        "active base model",
        "project/scripts/train_detection_baseline_v2.py",
        "models/simple_fcn.py",
        "H/V IQ and grouped detection manifest",
        "internal UAV/background development evidence",
    ),
    AlgorithmRecord(
        "BC-DPG v1/v2 calibration",
        "historical active comparison",
        "project/scripts/run_bc_dpg_fold14.py",
        "models/background_calibrated_dpg_fcn.py; models/background_tail_calibrated_dpg_fcn.py",
        "frozen DPG checkpoint and grouped manifest",
        "earlier calibration stages; not the current main claim",
    ),
    AlgorithmRecord(
        "BC-DPG v3 scan-aware calibration",
        "current frozen offline main result",
        "project/scripts/run_bc_dpg_v3.py",
        "models/target_protected_scan_calibrator.py; features/scan_context.py",
        "frozen DPG checkpoint and complete scan groups",
        "offline upper bound; not verified causal deployment",
    ),
    AlgorithmRecord(
        "H/V dual-branch gated FCN",
        "implemented experimental branch",
        "project/training/train_dual_branch_gated.py",
        "models/dual_branch_gated_fcn.py",
        "paired H/V IQ",
        "no independent external blind result",
    ),
    AlgorithmRecord(
        "Polarimetric representation benchmark",
        "two-fold development study",
        "project/scripts/run_polarimetric_representation_benchmark_v2.py",
        "features/polarimetric_rd.py; features/polarimetric_gated_rd.py; models/polarimetric_representation_fcn.py",
        "paired H/V IQ",
        "relative representations only; absolute calibration unverified",
    ),
    AlgorithmRecord(
        "ROI polarimetric refinement",
        "six-fold internal study",
        "project/scripts/run_roi_stage4_selected_sixfold_v1.py",
        "features/roi_polarimetric_refinement.py; models/roi_polarimetric_refiner.py",
        "base candidates, H/V IQ, grouped manifests",
        "mode selection reused development folds",
    ),
    AlgorithmRecord(
        "Tian 2024 FCN reproduction",
        "implemented but exact reproduction blocked",
        "project/scripts/run_tian_fcn_sixfold.py",
        "models/tian_fcn.py; evaluation/tian_fcn_postprocess.py",
        "paper-compatible DPL/GT contract and unavailable author details",
        "current point-GT result is a local diagnostic, not successful reproduction",
    ),
    AlgorithmRecord(
        "Zero-Doppler mechanisms",
        "learned development candidate",
        "project/scripts/run_zero_doppler_mechanism_v1.py",
        "models/zero_doppler_mechanisms.py; training/train_zero_doppler_mechanism.py",
        "frozen DPG scores and grouped H/V data",
        "outer folds consumed; needs new locked data",
    ),
    AlgorithmRecord(
        "Multi-domain feature extraction/fusion",
        "implemented algorithm scaffold",
        "project/scripts/build_multidomain_feature_catalog_v1.py",
        "features/multidomain_radar_features.py; models/multidomain_feature_fusion.py",
        "time-domain, H/V and timing-aware micro-Doppler inputs",
        "current data do not verify physical micro-Doppler axis",
    ),
    AlgorithmRecord(
        "Polarimetric transfer encoder",
        "architecture/interface ready",
        "project/models/polarimetric_transfer_encoder.py",
        "models/polarimetric_transfer_encoder.py",
        "calibrated or validity-masked H/V representations",
        "no pretrained checkpoint or balloon result",
    ),
    AlgorithmRecord(
        "LAT-MRICD grouped baselines",
        "frozen within-release result",
        "project/scripts/run_lat_mricd_grouped_baseline_v1.py",
        "scripts/run_lat_mricd_grouped_baseline_v1.py",
        "official LAT-MRICD archive",
        "batch-code-held-out, not unseen-model evidence",
    ),
    AlgorithmRecord(
        "LAT-MRICD cross-band transfer",
        "frozen negative result",
        "project/scripts/run_lat_mricd_cross_band_transfer_v1.py",
        "scripts/run_lat_mricd_cross_band_transfer_v1.py",
        "official LAT-MRICD archive",
        "S/Ku targets consumed; no confirmatory retuning",
    ),
    AlgorithmRecord(
        "Dataset/readiness auditors",
        "supporting reproducibility code",
        "project/scripts/check_project_health.py",
        "scripts/audit_*.py; scripts/validate_data_collection_manifest.py",
        "dataset-specific local sources",
        "audits do not constitute model performance",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the data-free algorithm code review package."
    )
    parser.add_argument("--source-ref", default="HEAD")
    parser.add_argument("--output-root", default=str(PROJECT_ROOT / "dist"))
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def run_git(*args: str, binary: bool = False) -> bytes | str:
    completed = subprocess.run(
        ["git", *args],
        cwd=PROJECT_ROOT,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if binary:
        return completed.stdout
    return completed.stdout.decode("utf-8", errors="strict").strip()


def resolve_commit(source_ref: str) -> str:
    value = str(run_git("rev-parse", "--verify", f"{source_ref}^{{commit}}"))
    if len(value) != 40 or any(char not in "0123456789abcdef" for char in value):
        raise RuntimeError(f"Unexpected source commit: {value!r}")
    return value


def tracked_paths(commit: str) -> list[str]:
    raw = run_git("ls-tree", "-r", "--name-only", "-z", commit, binary=True)
    assert isinstance(raw, bytes)
    return sorted(
        item.decode("utf-8", errors="strict")
        for item in raw.split(b"\0")
        if item
    )


def blob(commit: str, source_path: str) -> bytes:
    result = run_git("show", f"{commit}:{source_path}", binary=True)
    assert isinstance(result, bytes)
    return result


def is_active_source(path: str) -> bool:
    if path in ROOT_FILES:
        return True
    if path.startswith(ACTIVE_PREFIXES):
        return True
    if path.startswith("docs/") and not path.startswith("docs/share/"):
        return True
    if path.startswith("tools/") and PurePosixPath(path).suffix.lower() in TOOLS_SUFFIXES:
        return True
    return False


def validate_source_selection(paths: Iterable[str]) -> None:
    for path in paths:
        if path.startswith(FORBIDDEN_SOURCE_PREFIXES):
            raise RuntimeError(f"Forbidden source selected: {path}")
        if path.startswith("_cleanup_archive/"):
            raise RuntimeError(f"Recovery source selected as active code: {path}")
        if PurePosixPath(path).suffix.lower() in FORBIDDEN_PACKAGED_SUFFIXES:
            raise RuntimeError(f"Forbidden artifact selected: {path}")


def sanitize_blob(source_path: str, payload: bytes) -> tuple[bytes, str]:
    if source_path != "requirements-lock.txt":
        return payload, "none"
    text = payload.decode("utf-8")
    old = (
        "packaging @ file:///home/conda/feedstock_root/build_artifacts/"
        "bld/rattler-build_packaging_1777103621/work"
    )
    if old not in text:
        raise RuntimeError("Frozen requirements local-build URI changed unexpectedly")
    return text.replace(old, "packaging==26.2").encode("utf-8"), "local_build_uri_removed"


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def write_payload(
    package_root: Path,
    packaged_path: str,
    payload: bytes,
    *,
    source_path: str | None,
    category: str,
    transformation: str = "none",
) -> FileRecord:
    pure = PurePosixPath(packaged_path)
    if pure.is_absolute() or ".." in pure.parts:
        raise RuntimeError(f"Unsafe packaged path: {packaged_path}")
    if pure.suffix.lower() in FORBIDDEN_PACKAGED_SUFFIXES:
        raise RuntimeError(f"Forbidden packaged artifact: {packaged_path}")
    for marker in ACTUAL_PRIVATE_MARKERS:
        if marker in payload:
            raise RuntimeError(f"Private path marker in {packaged_path}")
    destination = package_root / pure
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(payload)
    return FileRecord(
        packaged_path=pure.as_posix(),
        source_path=source_path,
        category=category,
        size_bytes=len(payload),
        sha256=sha256_bytes(payload),
        transformation=transformation,
    )


def algorithm_index_bytes() -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(asdict(ALGORITHMS[0]).keys()))
    writer.writeheader()
    for record in ALGORITHMS:
        writer.writerow(asdict(record))
    return stream.getvalue().encode("utf-8-sig")


def package_readme(commit: str) -> bytes:
    tests = " ".join(SMOKE_TESTS)
    text = f"""# \u96f7\u8fbe\u9879\u76ee\u7b97\u6cd5\u4ee3\u7801\u8bc4\u5ba1\u5305

\u7248\u672c\uff1a2026-08-04 V1
\u6e90 Git commit\uff1a`{commit}`

## \u8fd9\u4e2a\u5305\u7528\u6765\u505a\u4ec0\u4e48

\u8fd9\u662f\u7ed9\u5b66\u957f\u6216\u5408\u4f5c\u8005\u505a\u65b9\u6cd5\u5ba1\u67e5\u7684\u4ee3\u7801\u5305\u3002`project/` \u5305\u542b\u6307\u5b9a Git \u63d0\u4ea4\u4e2d\u7684\u5f53\u524d\u7b97\u6cd5\u3001\u914d\u7f6e\u3001\u8bad\u7ec3/\u8bc4\u4ef7\u5165\u53e3\u548c\u81ea\u52a8\u5316\u6d4b\u8bd5\uff1b`review/` \u5305\u542b\u9879\u76ee\u6458\u8981\u3001\u7ed3\u679c\u8fb9\u754c\u548c\u8bf7\u6559\u95ee\u9898\u3002

\u5305\u4e2d\u6ca1\u6709\u539f\u59cb MAT/IQ/PCD\u3001\u6807\u7b7e\u660e\u7ec6\u3001\u9010\u6837\u672c\u9884\u6d4b\u3001checkpoint\u3001\u8bad\u7ec3\u65e5\u5fd7\u3001\u53c2\u8003\u8bba\u6587\u6216\u51ed\u636e\u3002\u672a\u63d0\u4ea4\u7684\u5de5\u4f5c\u533a\u5185\u5bb9\u4e5f\u6ca1\u6709\u8fdb\u5165\u672c\u5305\u3002

## \u5148\u770b\u4ec0\u4e48

1. `ALGORITHM_INDEX.csv`\uff1a\u6bcf\u6761\u7b97\u6cd5\u7684\u5165\u53e3\u3001\u6838\u5fc3\u6587\u4ef6\u3001\u6240\u9700\u6570\u636e\u548c\u53ef\u58f0\u660e\u8fb9\u754c\u3002
2. `review/00_ONE_PAGE_SUMMARY_ZH.md`\uff1a\u9879\u76ee\u4e00\u9875\u6982\u89c8\u3002
3. `review/01_RESULTS_AND_EVIDENCE_ZH.md`\uff1a\u5df2\u51bb\u7ed3\u7ed3\u679c\u548c\u9650\u5236\u3002
4. `review/03_QUESTIONS_FOR_SENIOR_ZH.md`\uff1a\u5efa\u8bae\u8bf7\u6559\u7684\u95ee\u9898\u3002
5. `INTERNAL_IMPORT_AUDIT.csv` \u548c `MANIFEST.json`\uff1a\u5b8c\u6574\u6027\u4e0e\u6765\u6e90\u6838\u5bf9\u3002

## \u5f53\u524d\u4e3b\u7ed3\u679c\u53e3\u5f84

\u5b8c\u6574\u626b\u63cf\u4e0a\u4e0b\u6587 BC-DPG-FCN v3 \u7684\u516d\u6298\u5185\u90e8\u5f00\u53d1\u6c47\u603b\u4e3a 56/830 \u4e2a\u80cc\u666f\u865a\u8b66\uff08pooled `P_FA=6.75%`\uff09\u548c 289/318 \u4e2a\u68c0\u6d4b\u5b9a\u4f4d\u8054\u5408\u6210\u529f\uff08joint `P_D=90.88%`\uff09\u3002\u5b83\u4f7f\u7528\u5b8c\u6574\u626b\u63cf\u4e0a\u4e0b\u6587\uff0c\u662f\u79bb\u7ebf\u6027\u80fd\u4e0a\u9650\uff0c\u4e0d\u662f\u5df2\u9a8c\u8bc1\u5b9e\u65f6\u90e8\u7f72\u7ed3\u679c\u3002

\u5728\u540c\u4e00\u516d\u6298\u6d4b\u8bd5\u5206\u6570\u4e0a\u4e8b\u540e\u8bfb\u53d6 `P_FA<=1%` \u7684 pooled ROC \u70b9\uff0c\u5f97\u5230 8/830 \u865a\u8b66\uff08`0.964%`\uff09\u3001307/318 score \u68c0\u6d4b\uff08`96.54%`\uff09\u548c 295/318 \u8054\u5408\u6210\u529f\uff08`92.77%`\uff09\u3002\u8fd9\u53ea\u80fd\u4f5c\u4e3a\u4f7f\u7528\u540c\u4e00\u6d4b\u8bd5\u5206\u6570\u7684\u5185\u90e8 ROC \u8bca\u65ad\uff0c\u4e0d\u80fd\u66ff\u4ee3\u201c\u9a8c\u8bc1\u96c6\u5b9a\u9608\u503c + \u65b0\u76f2\u6d4b\u96c6\u201d\u3002

## \u73af\u5883\u548c\u65e0\u6570\u636e\u68c0\u67e5

```bash
cd project
conda env create -f environment.yml
conda activate radar-torch
python scripts/check_project_health.py
python -m pytest -q {tests}
```

\u4e0a\u9762\u7684 pytest \u5217\u8868\u662f\u4e0d\u9700\u539f\u59cb\u6570\u636e\u7684\u6838\u5fc3\u7b97\u6cd5\u5408\u540c\u6d4b\u8bd5\u3002\u4e0d\u8981\u76f4\u63a5\u8fd0\u884c\u5168\u90e8\u8bad\u7ec3\u5165\u53e3\uff1a\u5b83\u4eec\u9700\u8981\u672c\u5305\u6545\u610f\u6392\u9664\u7684\u539f\u59cb\u6570\u636e\u3001manifest \u548c checkpoint\u3002

## \u5386\u53f2\u517c\u5bb9\u6e90\u7801

4 \u4e2a\u65e9\u671f\u5165\u53e3\u4ecd\u4f9d\u8d56\u6e05\u7406\u65f6\u5f52\u6863\u7684\u6a21\u5757\u3002\u6784\u5efa\u5668\u6309 Git \u4e2d\u7684\u5df2\u8ddf\u8e2a\u539f\u4ef6\u5c06\u5b83\u4eec\u8865\u5230 `project/` \u7684\u539f\u5bfc\u5165\u4f4d\u7f6e\uff0c\u5e76\u5728 `MANIFEST.json` \u4e2d\u6807\u8bb0 `legacy_compatibility_source`\u3002\u8fd9\u4e0d\u8868\u793a\u5b83\u4eec\u5df2\u88ab\u91cd\u65b0\u9009\u4e3a\u5f53\u524d\u65b9\u6cd5\u3002

## \u91cd\u8981\u9650\u5236

- \u5f53\u524d 318 \u4e2a UAV \u6b63\u6837\u672c\u5168\u90e8\u6765\u81ea 2026-02-02\uff0c830 \u4e2a\u80cc\u666f\u8d1f\u6837\u672c\u5168\u90e8\u6765\u81ea 2026-02-04\uff0c\u7c7b\u522b\u4e0e\u65e5\u671f\u5b8c\u5168\u6df7\u6742\u3002
- \u5f53\u524d\u7ed3\u679c\u4e0d\u662f\u8de8\u573a\u5730\u3001\u8de8\u65e5\u671f\u6216\u5916\u90e8\u76f2\u6d4b\u3002
- \u5f53\u524d\u6570\u636e\u4e0d\u652f\u6301\u7a7a\u98d8\u7403\u8f7d\u8377\u5206\u7c7b\u7ed3\u8bba\u3002
- \u672a\u9a8c\u8bc1 PRF/CPI/\u7269\u7406\u9891\u7387\u8f74\u65f6\uff0c\u4e0d\u5f97\u628a\u5f52\u4e00\u5316 Doppler bin \u89e3\u91ca\u4e3a\u7269\u7406\u5fae\u591a\u666e\u52d2 Hz\u3002
- \u4ed3\u5e93\u76ee\u524d\u6ca1\u6709\u9879\u76ee\u7ea7\u5f00\u6e90\u8bb8\u53ef\u8bc1\uff1b\u672c\u5305\u53ea\u662f\u79c1\u4e0b\u5b66\u672f\u8bc4\u5ba1\u6750\u6599\uff0c\u4e0d\u81ea\u52a8\u6388\u4e88\u516c\u5f00\u518d\u5206\u53d1\u6743\u3002
"""
    return text.encode("utf-8")


def sharing_notice() -> bytes:
    return """# Sharing notice

This archive is prepared for private academic review. The source repository
does not currently contain a project-level open-source licence, so receipt of
this archive does not grant permission for public redistribution or commercial
reuse. Dataset licences remain separate from the code and no dataset is
included here. Ask the project owner before forwarding the archive or reusing
substantial source code.
""".encode("utf-8")


def import_audit(package_root: Path) -> tuple[bytes, int]:
    project = package_root / "project"
    module_paths: dict[str, str] = {}
    python_files = sorted(project.rglob("*.py"))
    for path in python_files:
        relative = path.relative_to(project)
        parts = list(relative.with_suffix("").parts)
        module = ".".join(parts)
        module_paths[module] = relative.as_posix()
        if parts[-1] == "__init__":
            module_paths[".".join(parts[:-1])] = relative.as_posix()
        if tuple(parts[:3]) == ("tools", "radar_data_reader", "datasets"):
            installed_module = ".".join(("datasets", *parts[3:]))
            module_paths[installed_module] = relative.as_posix()

    rows: list[dict[str, str]] = []
    unresolved = 0
    for path in python_files:
        relative = path.relative_to(project).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=relative)
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                if node.level == 0 and node.module.split(".")[0] in INTERNAL_TOP_LEVEL:
                    imported.add(node.module)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.split(".")[0] in INTERNAL_TOP_LEVEL:
                        imported.add(alias.name)
        for module in sorted(imported):
            resolved = module in module_paths
            unresolved += int(not resolved)
            rows.append(
                {
                    "source_file": relative,
                    "imported_module": module,
                    "resolved": str(resolved).lower(),
                    "resolved_path": module_paths.get(module, ""),
                }
            )

    stream = io.StringIO(newline="")
    fields = ["source_file", "imported_module", "resolved", "resolved_path"]
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8-sig"), unresolved


def validate_package_tree(package_root: Path) -> None:
    files = sorted(path for path in package_root.rglob("*") if path.is_file())
    if not files:
        raise RuntimeError("Package is empty")
    for path in files:
        if path.is_symlink():
            raise RuntimeError(f"Symlink forbidden in package: {path}")
        relative = path.relative_to(package_root).as_posix()
        suffix = path.suffix.lower()
        if suffix in FORBIDDEN_PACKAGED_SUFFIXES:
            raise RuntimeError(f"Forbidden artifact in package: {relative}")
        payload = path.read_bytes()
        for marker in ACTUAL_PRIVATE_MARKERS:
            if marker in payload:
                raise RuntimeError(f"Private path marker in package: {relative}")


def sha256sums(records: Iterable[FileRecord]) -> bytes:
    lines = [
        f"{record.sha256}  {record.packaged_path}"
        for record in sorted(records, key=lambda item: item.packaged_path)
    ]
    return ("\n".join(lines) + "\n").encode("utf-8")


def deterministic_zip(package_root: Path, zip_path: Path) -> None:
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(item for item in package_root.rglob("*") if item.is_file()):
            relative = path.relative_to(package_root.parent).as_posix()
            info = zipfile.ZipInfo(relative, date_time=(2026, 8, 4, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, path.read_bytes(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)


def verify_zip(package_root: Path, zip_path: Path) -> None:
    expected = {
        path.relative_to(package_root.parent).as_posix(): path.read_bytes()
        for path in package_root.rglob("*")
        if path.is_file()
    }
    with zipfile.ZipFile(zip_path) as archive:
        if archive.testzip() is not None:
            raise RuntimeError("ZIP CRC validation failed")
        actual_names = set(archive.namelist())
        if actual_names != set(expected):
            raise RuntimeError("ZIP file set differs from package directory")
        for name, payload in expected.items():
            if archive.read(name) != payload:
                raise RuntimeError(f"ZIP payload mismatch: {name}")


def build_package(source_ref: str, output_root: Path, overwrite: bool) -> tuple[Path, Path]:
    commit = resolve_commit(source_ref)
    all_paths = tracked_paths(commit)
    selected = [path for path in all_paths if is_active_source(path)]
    validate_source_selection(selected)

    missing_review = sorted(set(REVIEW_DOCUMENTS) - set(all_paths))
    missing_compat = sorted(set(COMPATIBILITY_SOURCES) - set(all_paths))
    if missing_review or missing_compat:
        raise RuntimeError(
            f"Missing frozen package inputs: review={missing_review}, compatibility={missing_compat}"
        )

    output_root = output_root.resolve()
    package_root = output_root / PACKAGE_NAME
    zip_path = output_root / f"{PACKAGE_NAME}.zip"
    output_root.mkdir(parents=True, exist_ok=True)
    if package_root.exists() or zip_path.exists():
        if not overwrite:
            raise FileExistsError("Package output exists; pass --overwrite")
        if package_root.exists():
            shutil.rmtree(package_root)
        if zip_path.exists():
            zip_path.unlink()

    staging_parent = Path(tempfile.mkdtemp(prefix=f".{PACKAGE_NAME}.", dir=output_root))
    staging_root = staging_parent / PACKAGE_NAME
    staging_root.mkdir()
    records: list[FileRecord] = []

    try:
        for source_path in selected:
            payload, transformation = sanitize_blob(source_path, blob(commit, source_path))
            records.append(
                write_payload(
                    staging_root,
                    f"project/{source_path}",
                    payload,
                    source_path=source_path,
                    category="active_project_source",
                    transformation=transformation,
                )
            )

        for source_path, packaged_path in COMPATIBILITY_SOURCES.items():
            records.append(
                write_payload(
                    staging_root,
                    packaged_path,
                    blob(commit, source_path),
                    source_path=source_path,
                    category="legacy_compatibility_source",
                )
            )

        for source_path, packaged_path in REVIEW_DOCUMENTS.items():
            records.append(
                write_payload(
                    staging_root,
                    packaged_path,
                    blob(commit, source_path),
                    source_path=source_path,
                    category="review_context",
                )
            )

        generated = {
            "README.md": package_readme(commit),
            "SHARING_NOTICE.md": sharing_notice(),
            "ALGORITHM_INDEX.csv": algorithm_index_bytes(),
            "SOURCE_COMMIT.txt": f"{commit}\n".encode("ascii"),
        }
        for packaged_path, payload in generated.items():
            records.append(
                write_payload(
                    staging_root,
                    packaged_path,
                    payload,
                    source_path=None,
                    category="generated_package_document",
                )
            )

        audit_payload, unresolved_imports = import_audit(staging_root)
        if unresolved_imports:
            unresolved_rows = [
                line
                for line in audit_payload.decode("utf-8-sig").splitlines()
                if ",false," in line
            ]
            raise RuntimeError(
                f"Unresolved internal imports: {unresolved_imports}; rows={unresolved_rows}"
            )
        records.append(
            write_payload(
                staging_root,
                "INTERNAL_IMPORT_AUDIT.csv",
                audit_payload,
                source_path=None,
                category="generated_integrity_audit",
            )
        )

        manifest = {
            "package_name": PACKAGE_NAME,
            "package_version": "2026-08-04 V1",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_ref": source_ref,
            "source_commit": commit,
            "source_is_committed_snapshot": True,
            "uncommitted_worktree_content_included": False,
            "algorithm_count": len(ALGORITHMS),
            "internal_imports_unresolved": unresolved_imports,
            "raw_data_included": False,
            "sample_level_outputs_included": False,
            "checkpoints_included": False,
            "training_logs_included": False,
            "reference_papers_included": False,
            "project_open_source_licence_present": False,
            "pfa_1pct_diagnostic": {
                "role": "post-test pooled ROC diagnostic only",
                "background_samples": 830,
                "false_alarms": 8,
                "pfa": 8 / 830,
                "target_samples": 318,
                "score_detections": 307,
                "score_pd": 307 / 318,
                "joint_detections": 295,
                "joint_pd": 295 / 318,
                "formal_locked_claim_allowed": False,
            },
            "data_free_smoke_tests": list(SMOKE_TESTS),
            "excluded_content": [
                "raw radar data and external archives",
                "sample labels, manifests, and sample-level predictions",
                "model checkpoints and training logs",
                "large or sample-level result directories",
                "reference PDFs and development transcripts",
                "uncommitted worktree changes",
                "credentials and personal absolute paths",
            ],
            "files": [
                asdict(record)
                for record in sorted(records, key=lambda item: item.packaged_path)
            ],
        }
        manifest_payload = (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
        records.append(
            write_payload(
                staging_root,
                "MANIFEST.json",
                manifest_payload,
                source_path=None,
                category="generated_manifest",
            )
        )
        sums_payload = sha256sums(records)
        write_payload(
            staging_root,
            "SHA256SUMS.txt",
            sums_payload,
            source_path=None,
            category="generated_checksums",
        )

        validate_package_tree(staging_root)
        staging_root.replace(package_root)
        shutil.rmtree(staging_parent)
        deterministic_zip(package_root, zip_path)
        verify_zip(package_root, zip_path)
    except Exception:
        shutil.rmtree(staging_parent, ignore_errors=True)
        if package_root.exists():
            shutil.rmtree(package_root)
        if zip_path.exists():
            zip_path.unlink()
        raise

    return package_root, zip_path


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root).expanduser()
    if not output_root.is_absolute():
        output_root = PROJECT_ROOT / output_root
    package_root, zip_path = build_package(args.source_ref, output_root, args.overwrite)
    file_count = sum(1 for path in package_root.rglob("*") if path.is_file())
    print(f"Package: {package_root}")
    print(f"ZIP: {zip_path}")
    print(f"Files: {file_count}")
    print(f"ZIP SHA256: {sha256_bytes(zip_path.read_bytes())}")


if __name__ == "__main__":
    main()
