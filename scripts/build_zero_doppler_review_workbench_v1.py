#!/usr/bin/env python3
"""Build a self-contained local UI for zero-Doppler human review."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.audit_zero_doppler_human_review_v1 import (  # noqa: E402
    INDEPENDENT_EVIDENCE_SOURCE,
    REVIEW_STATUSES,
    VISIBLE_PATTERNS,
    validate_reviews,
)


DEFAULT_ATLAS_DIR = PROJECT_ROOT / "results/data_audit/zero_doppler_review_atlas_v1"
DEFAULT_CASES = DEFAULT_ATLAS_DIR / "cases.csv"
DEFAULT_OUTPUT = DEFAULT_ATLAS_DIR / "review_workbench.html"
REQUIRED_WORKBENCH_COLUMNS = {"atlas_rank", "image_file"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an offline HTML workbench for zero-Doppler review."
    )
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_cases(frame: pd.DataFrame, cases_dir: Path, output_dir: Path) -> pd.DataFrame:
    missing = REQUIRED_WORKBENCH_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"atlas cases missing columns: {sorted(missing)}")
    normalized = validate_reviews(frame)
    normalized["image_file"] = normalized["image_file"].fillna("").astype(str).str.strip()
    if normalized["image_file"].eq("").any():
        raise ValueError("atlas cases contain an empty image_file")

    cases_root = cases_dir.resolve()
    image_hrefs: list[str] = []
    for image_file in normalized["image_file"]:
        source = (cases_root / image_file).resolve()
        try:
            source.relative_to(cases_root)
        except ValueError as exc:
            raise ValueError(f"atlas image must stay inside atlas directory: {image_file}") from exc
        if not source.is_file():
            raise FileNotFoundError(f"atlas image not found: {source}")
        image_hrefs.append(Path(os.path.relpath(source, output_dir)).as_posix())
    normalized = normalized.copy()
    normalized["workbench_image_href"] = image_hrefs
    return normalized


def json_for_script(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":")).replace(
        "</", "<\\/"
    )


def render_workbench(
    cases: pd.DataFrame,
    *,
    source_sha256: str,
    source_columns: list[str],
) -> str:
    records = json.loads(cases.to_json(orient="records", force_ascii=True))
    payload = {
        "schemaVersion": 1,
        "sourceSha256": source_sha256,
        "sourceColumns": source_columns,
        "reviewStatuses": sorted(REVIEW_STATUSES),
        "visiblePatterns": sorted(VISIBLE_PATTERNS),
        "independentEvidenceSource": INDEPENDENT_EVIDENCE_SOURCE,
        "cases": records,
    }
    storage_key = f"zero-doppler-review-workbench-v1:{source_sha256[:16]}"
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Zero-Doppler P0 Review Workbench</title>
  <style>
    :root {{
      color-scheme: light;
      --ink: #17202a;
      --muted: #66717f;
      --line: #d8dee5;
      --surface: #ffffff;
      --canvas: #f3f5f6;
      --accent: #087f8c;
      --accent-dark: #075e68;
      --warning: #a35d00;
      --danger: #b42318;
      --success: #2f7d32;
      font-family: Inter, "Noto Sans SC", "Microsoft YaHei", Arial, sans-serif;
      font-size: 15px;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; background: var(--canvas); color: var(--ink); }}
    button, select, input, textarea {{ font: inherit; letter-spacing: 0; }}
    button, select, input, textarea {{ border: 1px solid #b9c2cc; border-radius: 5px; }}
    button {{ min-height: 38px; padding: 7px 12px; background: var(--surface); cursor: pointer; }}
    button:hover {{ border-color: var(--accent); }}
    button:focus-visible, select:focus-visible, input:focus-visible, textarea:focus-visible {{
      outline: 3px solid rgba(8, 127, 140, 0.22); outline-offset: 1px;
    }}
    button.primary {{ background: var(--accent); border-color: var(--accent); color: white; }}
    button.primary:hover {{ background: var(--accent-dark); }}
    button:disabled {{ cursor: not-allowed; opacity: 0.45; }}
    .topbar {{
      min-height: 64px; display: flex; align-items: center; gap: 16px; padding: 10px 18px;
      background: #1d2935; color: white; border-bottom: 3px solid var(--accent);
    }}
    .topbar h1 {{ margin: 0; font-size: 18px; font-weight: 650; }}
    .topbar .boundary {{ color: #d7e1e8; font-size: 12px; }}
    .progress-wrap {{ margin-left: auto; min-width: 190px; text-align: right; }}
    .progress-label {{ display: block; font-size: 12px; margin-bottom: 5px; }}
    progress {{ width: 190px; height: 8px; accent-color: #35b6a8; }}
    main {{
      display: grid; grid-template-columns: minmax(0, 1fr) minmax(310px, 380px);
      min-height: calc(100vh - 64px);
    }}
    .viewer {{ min-width: 0; padding: 16px; display: flex; flex-direction: column; gap: 12px; }}
    .casebar {{ display: flex; align-items: center; flex-wrap: wrap; gap: 8px; }}
    .case-id {{ min-width: 0; margin-right: auto; }}
    .case-id strong {{ display: block; overflow-wrap: anywhere; }}
    .case-id span {{ color: var(--muted); font-size: 12px; }}
    .image-stage {{
      min-height: 0; flex: 1; display: grid; place-items: center; background: #111820;
      border: 1px solid #263746; border-radius: 6px; overflow: auto;
    }}
    .image-stage img {{ display: block; max-width: 100%; max-height: calc(100vh - 170px); object-fit: contain; }}
    aside {{ background: var(--surface); border-left: 1px solid var(--line); padding: 16px; overflow-y: auto; }}
    aside h2 {{ margin: 0 0 12px; font-size: 16px; }}
    .facts {{
      display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 1px;
      margin-bottom: 16px; background: var(--line); border: 1px solid var(--line); border-radius: 5px; overflow: hidden;
    }}
    .fact {{ min-width: 0; padding: 8px; background: #f8fafb; }}
    .fact dt {{ color: var(--muted); font-size: 11px; }}
    .fact dd {{ margin: 3px 0 0; font-size: 13px; overflow-wrap: anywhere; }}
    .field {{ margin-bottom: 13px; }}
    .field label {{ display: block; margin-bottom: 5px; color: #344054; font-size: 12px; font-weight: 650; }}
    .field select, .field input, .field textarea {{ width: 100%; padding: 8px 9px; background: white; color: var(--ink); }}
    .field textarea {{ min-height: 92px; resize: vertical; }}
    .error {{ min-height: 34px; margin: 3px 0 10px; color: var(--danger); font-size: 12px; white-space: pre-line; }}
    .actions {{ display: flex; flex-wrap: wrap; gap: 8px; }}
    .save-state {{ margin-top: 10px; color: var(--muted); font-size: 12px; }}
    .status-pill {{
      display: inline-block; padding: 3px 7px; border-radius: 999px; background: #e9eef2;
      color: #3b4652; font-size: 11px; font-weight: 650;
    }}
    .status-pill.reviewed {{ background: #e4f4e5; color: var(--success); }}
    .status-pill.needs_more_context, .status-pill.unavailable {{ background: #fff1d6; color: var(--warning); }}
    @media (max-width: 860px) {{
      .topbar {{ align-items: flex-start; flex-wrap: wrap; }}
      .boundary {{ width: 100%; order: 3; }}
      .progress-wrap {{ min-width: 150px; }}
      progress {{ width: 150px; }}
      main {{ display: block; }}
      .image-stage {{ min-height: 48vh; }}
      .image-stage img {{ max-height: none; }}
      aside {{ border-left: 0; border-top: 1px solid var(--line); }}
    }}
  </style>
</head>
<body>
  <header class="topbar">
    <h1>Zero-Doppler P0 Review</h1>
    <div class="boundary">Visible structure only | Physical class needs an independent scene record</div>
    <div class="progress-wrap">
      <span class="progress-label" id="progress-label">0 / 0 complete</span>
      <progress id="progress" max="1" value="0"></progress>
    </div>
  </header>
  <main>
    <section class="viewer" aria-label="Atlas viewer">
      <div class="casebar">
        <div class="case-id">
          <strong id="sample-id"></strong>
          <span id="case-position"></span>
          <span class="status-pill" id="case-status"></span>
        </div>
        <button type="button" id="previous">Previous</button>
        <button type="button" id="next">Next</button>
      </div>
      <div class="image-stage">
        <img id="case-image" alt="RD review atlas sheet">
      </div>
    </section>
    <aside>
      <h2>Evidence and review</h2>
      <dl class="facts" id="facts"></dl>
      <div class="field">
        <label for="review-status">Review status</label>
        <select id="review-status"></select>
      </div>
      <div class="field">
        <label for="visible-pattern">Visible pattern</label>
        <select id="visible-pattern"></select>
      </div>
      <div class="field">
        <label for="physical-class">Physical class</label>
        <input id="physical-class" value="unknown" autocomplete="off">
      </div>
      <div class="field">
        <label for="evidence-source">Evidence source</label>
        <select id="evidence-source">
          <option value="prediction_and_relative_features_only">prediction_and_relative_features_only</option>
          <option value="independent_scene_record">independent_scene_record</option>
          <option value="unavailable">unavailable</option>
        </select>
      </div>
      <div class="field">
        <label for="review-note">Review note</label>
        <textarea id="review-note"></textarea>
      </div>
      <div class="error" id="validation-error" role="alert"></div>
      <div class="actions">
        <button type="button" class="primary" id="save-next">Save and next</button>
        <button type="button" id="export">Validate and export CSV</button>
        <button type="button" id="reset">Reset local progress</button>
      </div>
      <div class="save-state" id="save-state">Local browser storage ready.</div>
    </aside>
  </main>
  <script id="workbench-data" type="application/json">{json_for_script(payload)}</script>
  <script>
    (() => {{
      "use strict";
      const data = JSON.parse(document.getElementById("workbench-data").textContent);
      const storageKey = {json.dumps(storage_key)};
      const editable = ["review_status", "visible_pattern", "physical_class", "evidence_source", "review_note"];
      let records = data.cases.map((row) => ({{...row}}));
      let current = 0;
      const byId = (id) => document.getElementById(id);
      const controls = {{
        review_status: byId("review-status"),
        visible_pattern: byId("visible-pattern"),
        physical_class: byId("physical-class"),
        evidence_source: byId("evidence-source"),
        review_note: byId("review-note")
      }};

      function addOptions(select, values) {{
        values.forEach((value) => {{
          const option = document.createElement("option");
          option.value = value;
          option.textContent = value;
          select.appendChild(option);
        }});
      }}

      function restore() {{
        try {{
          const saved = JSON.parse(localStorage.getItem(storageKey));
          if (!saved || saved.sourceSha256 !== data.sourceSha256 || !saved.annotations) return;
          records.forEach((row) => {{
            const annotation = saved.annotations[`${{row.fold}}::${{row.sample_id}}`];
            if (annotation) editable.forEach((field) => {{ row[field] = annotation[field] ?? row[field]; }});
          }});
          current = Math.min(Math.max(Number(saved.current) || 0, 0), records.length - 1);
        }} catch (error) {{
          byId("save-state").textContent = "Saved progress could not be read; source values loaded.";
        }}
      }}

      function persist() {{
        const annotations = {{}};
        records.forEach((row) => {{
          annotations[`${{row.fold}}::${{row.sample_id}}`] = Object.fromEntries(
            editable.map((field) => [field, row[field] ?? ""])
          );
        }});
        try {{
          localStorage.setItem(storageKey, JSON.stringify({{
            sourceSha256: data.sourceSha256,
            current,
            annotations
          }}));
          byId("save-state").textContent = `Saved locally at ${{new Date().toLocaleTimeString()}}.`;
        }} catch (error) {{
          byId("save-state").textContent = "Browser storage unavailable; progress remains in this tab only.";
        }}
      }}

      function syncControlsToRecord() {{
        const row = records[current];
        editable.forEach((field) => {{ row[field] = controls[field].value.trim(); }});
        persist();
        updateProgress();
      }}

      function validateRow(row) {{
        const errors = [];
        if (!data.reviewStatuses.includes(row.review_status)) errors.push("Invalid review_status.");
        if (!data.visiblePatterns.includes(row.visible_pattern)) errors.push("Invalid visible_pattern.");
        const reviewed = row.review_status === "reviewed";
        const incomplete = ["needs_more_context", "unavailable"].includes(row.review_status);
        const namedPhysical = row.physical_class !== "unknown";
        if (reviewed && row.visible_pattern === "unreviewed") errors.push("Reviewed rows need a visible pattern.");
        if (reviewed && !row.review_note) errors.push("Reviewed rows need a review note.");
        if (incomplete && !row.review_note) errors.push("Incomplete rows need a note explaining the gap.");
        if (incomplete && namedPhysical) errors.push("Incomplete rows must keep physical_class=unknown.");
        if (namedPhysical && row.evidence_source !== data.independentEvidenceSource) errors.push("Named physical classes require an independent scene record.");
        if (namedPhysical && !row.review_note) errors.push("Named physical classes require a review note.");
        if (row.review_status === "pending" && namedPhysical) errors.push("Pending rows must keep physical_class=unknown.");
        return errors;
      }}

      function updateProgress() {{
        const complete = records.filter((row) => row.review_status !== "pending").length;
        byId("progress").max = records.length;
        byId("progress").value = complete;
        byId("progress-label").textContent = `${{complete}} / ${{records.length}} complete`;
        const status = byId("case-status");
        status.textContent = records[current].review_status;
        status.className = `status-pill ${{records[current].review_status}}`;
      }}

      function fact(label, value) {{
        const wrapper = document.createElement("div");
        wrapper.className = "fact";
        const term = document.createElement("dt");
        const detail = document.createElement("dd");
        term.textContent = label;
        detail.textContent = value;
        wrapper.append(term, detail);
        return wrapper;
      }}

      function render() {{
        const row = records[current];
        byId("sample-id").textContent = row.sample_id;
        byId("case-position").textContent = `Case ${{current + 1}} of ${{records.length}} | Fold ${{row.fold}} | ${{row.review_priority}}`;
        byId("case-image").src = row.workbench_image_href;
        byId("case-image").alt = `RD atlas for ${{row.sample_id}}`;
        const facts = byId("facts");
        facts.replaceChildren(
          fact("Fixed score", Number(row.score_fixed).toFixed(4)),
          fact("Residual score", Number(row.score_residual).toFixed(4)),
          fact("Score delta", Number(row.score_delta_residual_minus_fixed).toFixed(4)),
          fact("Zero-bin distance", `${{row.zero_velocity_distance_bins}} bins`),
          fact("Zero-Doppler fraction", Number(row.feature_rd_anchor_zero_doppler_fraction).toFixed(4)),
          fact("Relative H/V IQR", `${{Number(row.feature_polar_roi_zdr_iqr_db).toFixed(2)}} dB`)
        );
        editable.forEach((field) => {{ controls[field].value = row[field] ?? ""; }});
        byId("previous").disabled = current === 0;
        byId("next").disabled = current === records.length - 1;
        byId("validation-error").textContent = validateRow(row).join("\\n");
        updateProgress();
      }}

      function move(delta) {{
        syncControlsToRecord();
        current = Math.min(Math.max(current + delta, 0), records.length - 1);
        persist();
        render();
        window.scrollTo({{top: 0, behavior: "smooth"}});
      }}

      function csvCell(value) {{
        const text = value === null || value === undefined ? "" : String(value);
        return `"${{text.replaceAll('"', '""')}}"`;
      }}

      function exportCsv() {{
        syncControlsToRecord();
        const allErrors = records.flatMap((row, index) =>
          validateRow(row).map((message) => `Case ${{index + 1}} (${{row.sample_id}}): ${{message}}`)
        );
        if (allErrors.length) {{
          byId("validation-error").textContent = allErrors.slice(0, 6).join("\\n") +
            (allErrors.length > 6 ? `\\n... ${{allErrors.length - 6}} more error(s)` : "");
          return;
        }}
        const lines = [data.sourceColumns.map(csvCell).join(",")];
        records.forEach((row) => lines.push(data.sourceColumns.map((column) => csvCell(row[column])).join(",")));
        const blob = new Blob(["\ufeff" + lines.join("\\r\\n") + "\\r\\n"], {{type: "text/csv;charset=utf-8"}});
        const link = document.createElement("a");
        link.href = URL.createObjectURL(blob);
        const stamp = new Date().toISOString().slice(0, 10).replaceAll("-", "");
        link.download = `review_queue_reviewer_${{stamp}}.csv`;
        document.body.appendChild(link);
        link.click();
        link.remove();
        window.setTimeout(() => URL.revokeObjectURL(link.href), 0);
        byId("validation-error").textContent = "";
        byId("save-state").textContent = "Validated CSV exported.";
      }}

      addOptions(controls.review_status, data.reviewStatuses);
      addOptions(controls.visible_pattern, data.visiblePatterns);
      restore();
      render();
      Object.values(controls).forEach((control) => control.addEventListener("change", () => {{
        syncControlsToRecord();
        byId("validation-error").textContent = validateRow(records[current]).join("\\n");
      }}));
      controls.review_note.addEventListener("input", syncControlsToRecord);
      byId("previous").addEventListener("click", () => move(-1));
      byId("next").addEventListener("click", () => move(1));
      byId("save-next").addEventListener("click", () => move(1));
      byId("export").addEventListener("click", exportCsv);
      byId("reset").addEventListener("click", () => {{
        if (!window.confirm("Reset all locally saved review fields for this atlas?")) return;
        try {{ localStorage.removeItem(storageKey); }} catch (error) {{ /* in-memory reset still applies */ }}
        records = data.cases.map((row) => ({{...row}}));
        current = 0;
        render();
        byId("save-state").textContent = "Local progress reset.";
      }});
      document.addEventListener("keydown", (event) => {{
        const typing = ["INPUT", "TEXTAREA", "SELECT"].includes(document.activeElement?.tagName);
        if (typing) return;
        if (event.key === "ArrowLeft") move(-1);
        if (event.key === "ArrowRight") move(1);
      }});
    }})();
  </script>
</body>
</html>
"""


def build_workbench(*, cases_path: Path, output_path: Path, overwrite: bool) -> dict[str, Any]:
    cases_path = resolve_path(cases_path)
    output_path = resolve_path(output_path)
    if not cases_path.is_file():
        raise FileNotFoundError(f"atlas cases not found: {cases_path}")
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"output already exists: {output_path}")
    if output_path.is_dir():
        raise IsADirectoryError(f"output must be an HTML file: {output_path}")

    source = pd.read_csv(cases_path, encoding="utf-8-sig")
    source_columns = list(source.columns)
    source_sha256 = sha256_file(cases_path)
    cases = prepare_cases(source, cases_path.parent, output_path.parent)
    rendered = render_workbench(
        cases, source_sha256=source_sha256, source_columns=source_columns
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(rendered, encoding="utf-8")
    return {
        "status": "READY",
        "case_count": int(len(cases)),
        "source_sha256": source_sha256,
        "output": str(output_path),
    }


def main() -> int:
    args = parse_args()
    result = build_workbench(
        cases_path=args.cases,
        output_path=args.output,
        overwrite=args.overwrite,
    )
    print("Zero-Doppler review workbench: PASS")
    print(f"case_count={result['case_count']}")
    print(f"output={result['output']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
