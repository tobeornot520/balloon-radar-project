#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.audit_field_readiness_v1 import (
    DEFAULT_CHECKLIST,
    load_checklist,
    pending_evidence_frame,
    resolve_path,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a pending evidence table from the field checklist"
    )
    parser.add_argument("output", type=Path)
    parser.add_argument("--checklist", type=Path, default=DEFAULT_CHECKLIST)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    checklist_path = resolve_path(args.checklist)
    output_path = resolve_path(args.output)
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"output exists: {output_path}; use --overwrite")
    checklist = load_checklist(checklist_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pending_evidence_frame(checklist).to_csv(
        output_path, index=False, encoding="utf-8-sig"
    )
    print(f"Wrote {len(checklist['items'])} pending items to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
