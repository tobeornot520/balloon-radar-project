#!/usr/bin/env python3
from pathlib import Path
import datetime
import json
import shutil

PACKAGE_ROOT = Path(__file__).resolve().parent
SOURCE_ROOT = PACKAGE_ROOT / "payload"
PROJECT_ROOT = Path.cwd().resolve()

if not (PROJECT_ROOT / "scripts").is_dir() or not (PROJECT_ROOT / "results").is_dir():
    raise SystemExit("Run this installer from the balloon_radar_project root directory.")

manifest = json.loads((PACKAGE_ROOT / "INSTALL_MANIFEST.json").read_text(encoding="utf-8"))
stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
backup_root = PROJECT_ROOT / "backups" / f"stage4_next_all_v1_{stamp}"
installed = []
backed_up = []

for relative in manifest["files"]:
    source = SOURCE_ROOT / relative
    destination = PROJECT_ROOT / relative
    if not source.is_file():
        raise FileNotFoundError(f"Package payload missing: {source}")
    if destination.exists():
        backup = backup_root / relative
        backup.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(destination, backup)
        backed_up.append(relative)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    if destination.suffix in {".py", ".sh"}:
        destination.chmod(0o755)
    installed.append(relative)

output = PROJECT_ROOT / "results/data_audit/roi_stage4_selected_sixfold_v1"
output.mkdir(parents=True, exist_ok=True)
(output / "install_record.json").write_text(
    json.dumps(
        {
            "installed": installed,
            "backed_up": backed_up,
            "backup_root": str(backup_root) if backed_up else None,
        },
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)
print(f"INSTALLED {len(installed)} files")
print(f"BACKED_UP {len(backed_up)} files")
print("Next: python scripts/preflight_roi_stage4_selected_sixfold_v1.py")
