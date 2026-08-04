from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import subprocess
import sys
import zipfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts/build_algorithm_code_share_package.py"


def load_builder():
    spec = importlib.util.spec_from_file_location("algorithm_package_builder", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_algorithm_package_builds_from_committed_snapshot(tmp_path: Path) -> None:
    builder = load_builder()
    package_root, zip_path = builder.build_package("HEAD", tmp_path, False)

    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
    ).strip()
    manifest = json.loads((package_root / "MANIFEST.json").read_text(encoding="utf-8"))

    assert manifest["source_commit"] == head
    assert manifest["source_is_committed_snapshot"] is True
    assert manifest["uncommitted_worktree_content_included"] is False
    assert manifest["algorithm_count"] >= 14
    assert manifest["internal_imports_unresolved"] == 0
    assert manifest["raw_data_included"] is False
    assert manifest["sample_level_outputs_included"] is False
    assert manifest["checkpoints_included"] is False
    assert manifest["project_open_source_licence_present"] is False

    pfa = manifest["pfa_1pct_diagnostic"]
    assert pfa["false_alarms"] == 8
    assert pfa["background_samples"] == 830
    assert pfa["score_detections"] == 307
    assert pfa["joint_detections"] == 295
    assert pfa["formal_locked_claim_allowed"] is False

    assert (package_root / "project/models/tian_fcn.py").is_file()
    assert (package_root / "project/models/zero_doppler_mechanisms.py").is_file()
    assert (package_root / "project/features/multidomain_radar_features.py").is_file()
    assert (package_root / "project/datasets/radar_dataset.py").is_file()
    assert (package_root / "project/datasets/polarimetric_detection_dataset_v1.py").is_file()
    assert (package_root / "project/training/train_background_calibrator.py").is_file()
    assert (package_root / "review/00_ONE_PAGE_SUMMARY_ZH.md").is_file()

    assert not (package_root / "project/data").exists()
    assert not (package_root / "project/results").exists()
    assert not (package_root / "project/checkpoints").exists()
    assert not (package_root / "project/payload").exists()
    assert not (package_root / "project/_cleanup_archive").exists()
    assert not (package_root / "project/scripts/audit_lss_hsr_l_v2.py").exists()

    requirements = (package_root / "project/requirements-lock.txt").read_text(encoding="utf-8")
    assert "packaging==26.2" in requirements
    assert "file:///home/" not in requirements

    import_rows = list(
        csv.DictReader(
            (package_root / "INTERNAL_IMPORT_AUDIT.csv").open(
                encoding="utf-8-sig", newline=""
            )
        )
    )
    assert import_rows
    assert {row["resolved"] for row in import_rows} == {"true"}

    payload = b"\n".join(
        path.read_bytes()
        for path in sorted(package_root.rglob("*"))
        if path.is_file()
    )
    private_marker = b"/home/" + b"tobeornot8259748/"
    assert private_marker not in payload
    assert not any(
        path.suffix.lower() in builder.FORBIDDEN_PACKAGED_SUFFIXES
        for path in package_root.rglob("*")
        if path.is_file()
    )

    with zipfile.ZipFile(zip_path) as archive:
        assert archive.testzip() is None
        expected = {
            path.relative_to(package_root.parent).as_posix()
            for path in package_root.rglob("*")
            if path.is_file()
        }
        assert set(archive.namelist()) == expected


def test_manifest_and_checksums_cover_packaged_payload(tmp_path: Path) -> None:
    builder = load_builder()
    package_root, _ = builder.build_package("HEAD", tmp_path, False)
    manifest = json.loads((package_root / "MANIFEST.json").read_text(encoding="utf-8"))

    manifest_rows = {row["packaged_path"]: row for row in manifest["files"]}
    assert "README.md" in manifest_rows
    assert "project/models/simple_fcn.py" in manifest_rows
    assert manifest_rows["project/datasets/radar_dataset.py"]["category"] == (
        "legacy_compatibility_source"
    )

    for path, row in manifest_rows.items():
        payload = (package_root / path).read_bytes()
        assert len(payload) == row["size_bytes"]
        assert hashlib.sha256(payload).hexdigest() == row["sha256"]

    checksum_rows = {}
    for line in (package_root / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
        digest, relative = line.split("  ", 1)
        checksum_rows[relative] = digest
    assert set(checksum_rows) == set(manifest_rows) | {"MANIFEST.json"}
    for relative, digest in checksum_rows.items():
        assert hashlib.sha256((package_root / relative).read_bytes()).hexdigest() == digest
