#!/usr/bin/env python3
"""Run the data-free multidomain interface preflight.

The preflight combines the frozen YAML contract audit with the deterministic
synthetic H/V feature-and-fusion smoke.  It never reads project data, trains a
model, or reports Pd/Pfa/AUC.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.audit_multidomain_feature_contract_v1 import (
    DEFAULT_CONTRACT,
    audit_contract,
    load_contract,
)
from scripts.run_multidomain_feature_smoke_v1 import run_smoke


def run_preflight(contract_path: str | Path = DEFAULT_CONTRACT) -> dict[str, Any]:
    contract_summary = audit_contract(load_contract(contract_path))
    smoke_summary = run_smoke()
    if contract_summary["status"] != "PASS":
        raise AssertionError("multidomain contract audit did not pass")
    if smoke_summary["status"] != "PASS":
        raise AssertionError("multidomain synthetic smoke did not pass")
    return {
        "status": "PASS",
        "scope": "data_free_multidomain_preflight",
        "model_training": False,
        "performance_metrics": False,
        "contract": contract_summary,
        "smoke": smoke_summary,
        "interpretation": (
            "Interface and claim-boundary preflight only; no real-data, physical-Hz, "
            "Pd/Pfa/AUC, or generalization claim."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_preflight(args.contract)
    rendered = json.dumps(summary, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output_json is not None:
        output = args.output_json.expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
        print(f"summary: {output}")


if __name__ == "__main__":
    main()
