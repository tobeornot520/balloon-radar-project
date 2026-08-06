#!/usr/bin/env python3
"""Report whether the current research-direction milestone is actually complete."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/current_direction_completion_v1.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results/data_audit/current_direction_completion_v1"
VALID_STATUSES = {"complete", "pending_user", "in_progress", "blocked_external"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check the explicit completion gate for the current research direction."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def validate_config(payload: dict[str, Any]) -> list[dict[str, Any]]:
    if payload.get("schema_version") != 1:
        raise ValueError("current-direction completion config must use schema_version=1")
    items = payload.get("items")
    if not isinstance(items, list) or not items:
        raise ValueError("current-direction completion config needs nonempty items")
    required = {"id", "ledger_ids", "title", "owner", "status", "evidence", "next_action"}
    ids: set[str] = set()
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("each completion item must be an object")
        missing = required - set(item)
        if missing:
            raise ValueError(f"completion item missing fields: {sorted(missing)}")
        if item["id"] in ids:
            raise ValueError(f"duplicate completion item ID: {item['id']}")
        ids.add(str(item["id"]))
        if item["status"] not in VALID_STATUSES:
            raise ValueError(f"invalid completion status: {item['status']}")
        if not isinstance(item["ledger_ids"], list) or not item["ledger_ids"]:
            raise ValueError(f"{item['id']} must reference at least one ledger ID")
        if not isinstance(item["evidence"], str) or not item["evidence"].strip():
            raise ValueError(f"{item['id']} must reference at least one evidence path")
    return items


def inspect_evidence(
    items: list[dict[str, Any]], *, evidence_root: Path
) -> list[dict[str, Any]]:
    root = evidence_root.resolve()
    inspected: list[dict[str, Any]] = []
    for original in items:
        item = dict(original)
        raw_paths = [part.strip() for part in str(item["evidence"]).split(";")]
        if not raw_paths or any(not path for path in raw_paths):
            raise ValueError(f"{item['id']} has an empty evidence path")

        missing: list[str] = []
        for raw_path in raw_paths:
            relative_path = Path(raw_path)
            if relative_path.is_absolute():
                raise ValueError(f"{item['id']} evidence path must be repository-relative")
            resolved = (root / relative_path).resolve()
            if not resolved.is_relative_to(root):
                raise ValueError(f"{item['id']} evidence path escapes the repository")
            available = resolved.is_file() and resolved.stat().st_size > 0
            if resolved.is_dir():
                available = next(resolved.iterdir(), None) is not None
            if not available:
                missing.append(raw_path)

        item["evidence_path_count"] = len(raw_paths)
        item["missing_evidence_count"] = len(missing)
        item["missing_evidence_paths"] = "; ".join(missing)
        item["evidence_complete"] = not missing
        inspected.append(item)
    return inspected


def summarize_items(items: list[dict[str, Any]]) -> tuple[dict[str, Any], pd.DataFrame]:
    table = pd.DataFrame(items)
    if "evidence_complete" not in table:
        raise ValueError("completion items must be inspected before summarizing")
    table["ledger_ids"] = table["ledger_ids"].apply(
        lambda values: ", ".join(str(value) for value in values)
    )
    table["is_complete"] = table["status"].eq("complete") & table[
        "evidence_complete"
    ]
    blocked = table[table["status"].eq("blocked_external")]
    remaining = table[~table["is_complete"]]
    if remaining.empty:
        status = "COMPLETE"
    elif remaining["status"].eq("blocked_external").all() and remaining[
        "evidence_complete"
    ].all():
        status = "BLOCKED_EXTERNAL"
    else:
        status = "IN_PROGRESS"
    summary = {
        "status": status,
        "milestone_complete": bool(remaining.empty),
        "item_count": int(len(table)),
        "completed_count": int(table["is_complete"].sum()),
        "remaining_count": int(len(remaining)),
        "blocked_external_count": int(len(blocked)),
        "pending_user_count": int(table["status"].eq("pending_user").sum()),
        "missing_evidence_count": int(table["missing_evidence_count"].sum()),
        "configured_complete_but_missing_evidence_count": int(
            (table["status"].eq("complete") & ~table["evidence_complete"]).sum()
        ),
        "completion_notice": (
            "All current-direction items are complete; start the next-data phase only "
            "under the collection contract."
            if remaining.empty
            else "Do not declare the current direction complete. Resolve the listed items first."
        ),
    }
    return summary, table.sort_values("id")


def make_readme(payload: dict[str, Any], summary: dict[str, Any]) -> str:
    return f"""# Current Direction Completion Check V1

Milestone: `{payload['title']}`  
Status: `{summary['status']}`  
Complete: `{summary['completed_count']}/{summary['item_count']}`

`COMPLETE` is emitted only when every explicitly required item is complete.
`BLOCKED_EXTERNAL` means a data, device, or reproduction-condition fact must be
supplied by an external owner; it is not an engineering success or failure.
An item configured as `complete` is counted only when every repository-relative
evidence path exists and is nonempty.

The current blocker list is in `remaining_actions.csv`. Do not start a new
model-training branch merely because a blocker is inconvenient.
"""


def check_completion(
    *, config_path: Path, output_dir: Path, overwrite: bool
) -> dict[str, Any]:
    config_path = resolve_path(config_path)
    output_dir = resolve_path(output_dir)
    if not config_path.is_file():
        raise FileNotFoundError(f"completion config not found: {config_path}")
    if output_dir == PROJECT_ROOT:
        raise ValueError("output directory must not be the project root")
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(f"output directory is nonempty: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = json.loads(config_path.read_text(encoding="utf-8"))
    items = inspect_evidence(
        validate_config(payload), evidence_root=PROJECT_ROOT
    )
    summary, table = summarize_items(items)
    summary["milestone_id"] = payload["milestone_id"]
    summary["completion_rule"] = payload["completion_rule"]
    table.to_csv(output_dir / "completion_matrix.csv", index=False, encoding="utf-8-sig")
    table[~table["is_complete"]].to_csv(
        output_dir / "remaining_actions.csv", index=False, encoding="utf-8-sig"
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "README.md").write_text(
        make_readme(payload, summary), encoding="utf-8"
    )
    return summary


def main() -> int:
    args = parse_args()
    summary = check_completion(
        config_path=args.config,
        output_dir=args.output_dir,
        overwrite=args.overwrite,
    )
    print("Current direction completion check: PASS")
    print(f"status={summary['status']}")
    print(f"complete={summary['completed_count']}/{summary['item_count']}")
    print(summary["completion_notice"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
