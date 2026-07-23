#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path

PATCH_NAME = "BC_DPG_roi_polarimetric_stage4_implementation_v1"
PAYLOAD_DIRNAME = "files"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def looks_like_project(path: Path) -> bool:
    required = ("features", "datasets", "models", "training", "scripts", "results")
    return all((path / name).exists() for name in required)


def find_project_root(start: Path) -> Path:
    candidates = [start.resolve(), *start.resolve().parents]
    for candidate in candidates:
        if looks_like_project(candidate):
            return candidate
    expected = Path.home() / "projects/balloon_radar_project"
    if looks_like_project(expected):
        return expected.resolve()
    raise RuntimeError(
        "Cannot identify balloon_radar_project. Put this patch in the project root and rerun."
    )


def main() -> None:
    installer = Path(__file__).resolve()
    payload = installer.parent / PAYLOAD_DIRNAME
    if not payload.is_dir():
        raise FileNotFoundError(f"Patch payload not found: {payload}")
    project = find_project_root(Path.cwd())
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_root = project / "backups" / f"roi_polarimetric_stage4_v1_install_{timestamp}"
    copied = []
    backups = []

    for source in sorted(path for path in payload.rglob("*") if path.is_file()):
        relative = source.relative_to(payload)
        target = project / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if sha256(target) == sha256(source):
                copied.append({
                    "path": relative.as_posix(),
                    "status": "UNCHANGED",
                    "sha256": sha256(source),
                })
                continue
            backup = backup_root / relative
            backup.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(target, backup)
            backups.append(relative.as_posix())
        shutil.copy2(source, target)
        copied.append({
            "path": relative.as_posix(),
            "status": "INSTALLED",
            "sha256": sha256(target),
        })
        print(f"INSTALLED  {relative.as_posix()}")

    audit = project / "results/data_audit/roi_polarimetric_stage4_v1"
    audit.mkdir(parents=True, exist_ok=True)
    record = {
        "patch": PATCH_NAME,
        "installed_at": datetime.now().isoformat(),
        "project_root": str(project),
        "files": copied,
        "backed_up_files": backups,
        "backup_root": str(backup_root) if backups else None,
        "safety": {
            "modified_existing_experiment_results": False,
            "modified_bc_dpg_v3": False,
            "copied_checkpoints": False,
            "copied_raw_mat": False,
        },
    }
    record_path = audit / "install_record_v1.json"
    record_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    print("=" * 88)
    print(f"Stage 4 implementation installed: {len(copied)} files")
    print(f"project root : {project}")
    print(f"backup root  : {record['backup_root'] or 'none'}")
    print(f"record       : {record_path}")
    print("Next: run scripts/test_roi_polarimetric_stage4_v1.py, then the smoke benchmark.")
    print("=" * 88)


if __name__ == "__main__":
    main()
