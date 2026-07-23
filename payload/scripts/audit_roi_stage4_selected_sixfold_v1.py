#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODES = ("power2_baseline", "power2_roi_power_control", "power2_roi_ri4")


def experiment_name(mode: str, fold: int, scope: str) -> str:
    return f"roi_polar_stage4_v1_{mode}_v4_fold{fold:02d}_seed42_{scope}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", nargs="+", type=int, default=[1, 2, 3, 4, 5, 6])
    parser.add_argument("--scope", default="formal")
    args = parser.parse_args()
    issues: list[str] = []
    rows: list[dict] = []

    for fold in args.folds:
        tables: dict[str, pd.DataFrame] = {}
        for mode in MODES:
            path = PROJECT_ROOT / "results/experiments" / experiment_name(mode, fold, args.scope) / "tables/test_predictions.csv"
            if not path.is_file():
                issues.append(f"missing {path}")
                continue
            tables[mode] = pd.read_csv(path).set_index("sample_id")
        if len(tables) != len(MODES):
            continue
        common = set.intersection(*[set(frame.index) for frame in tables.values()])
        baseline = tables["power2_baseline"].loc[list(common)]
        for mode in MODES[1:]:
            candidate = tables[mode].loc[list(common)]
            location_equal = bool(
                (candidate["pred_range_index"].to_numpy() == baseline["pred_range_index"].to_numpy()).all()
                and (candidate["pred_velocity_index"].to_numpy() == baseline["pred_velocity_index"].to_numpy()).all()
            )
            raw_equal = bool(
                (abs(candidate["raw_score"].to_numpy() - baseline["raw_score"].to_numpy()) < 1e-6).all()
            )
            suppression_only = bool(
                (candidate["refined_score"].to_numpy() <= candidate["raw_score"].to_numpy() + 1e-7).all()
            )
            new_false_alarms = int((
                ~baseline["raw_fixed_false_alarm"].astype(bool)
                & candidate["refined_fixed_false_alarm"].astype(bool)
            ).sum())
            rows.append({
                "fold": fold,
                "mode": mode,
                "samples": len(common),
                "power2_location_frozen": location_equal,
                "raw_score_identical": raw_equal,
                "suppression_only": suppression_only,
                "new_false_alarms": new_false_alarms,
            })
            if not (location_equal and raw_equal and suppression_only) or new_false_alarms:
                issues.append(f"invariant failed fold={fold} mode={mode}")

    output = PROJECT_ROOT / "results/data_audit/roi_stage4_selected_sixfold_v1"
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(
        output / f"integrity_audit_{args.scope}.csv", index=False, encoding="utf-8-sig"
    )
    status = {"status": "PASS" if not issues else "FAIL", "issues": issues, "rows": len(rows)}
    (output / f"integrity_audit_status_{args.scope}.json").write_text(
        json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(status, ensure_ascii=False, indent=2))
    raise SystemExit(0 if not issues else 2)


if __name__ == "__main__":
    main()
