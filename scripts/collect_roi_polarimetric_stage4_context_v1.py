#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Collect the exact project interfaces needed for Stage 4 ROI polarimetric refinement.

The collector is intentionally read-only with respect to the existing project. It copies
small source/config/result metadata into a temporary acceptance directory and writes one
ZIP. Raw radar files and checkpoint bytes are never copied.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
import traceback
import zipfile
from typing import Any, Iterable

SCRIPT_PATH = Path(__file__).resolve()
DEFAULT_ROOT = SCRIPT_PATH.parents[1]
CONFIG_REL = Path("configs/roi_polarimetric_stage4_context_v1.json")
OUTPUT_NAME = "roi_polarimetric_stage4_context_acceptance_v1.zip"
STAGE_NAME = "roi_polarimetric_stage4_context_v1"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="采集Stage 4候选区域极化精修所需真实工程接口")
    p.add_argument("--project-root", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--folds", type=int, nargs="+", default=[1, 4])
    p.add_argument("--output", type=Path, default=None)
    return p.parse_args()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_rel(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except Exception:
        return path.name


def load_config(root: Path) -> dict[str, Any]:
    path = root / CONFIG_REL
    if not path.is_file():
        raise FileNotFoundError(f"缺少配置文件: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def is_excluded(path: Path, cfg: dict[str, Any]) -> bool:
    return path.suffix.lower() in {x.lower() for x in cfg["excluded_suffixes"]}


def copy_small_file(src: Path, dst_root: Path, root: Path, cfg: dict[str, Any], records: list[dict[str, Any]]) -> bool:
    if not src.is_file() or is_excluded(src, cfg):
        return False
    size = src.stat().st_size
    limit = cfg["max_csv_copy_bytes"] if src.suffix.lower() == ".csv" else cfg["max_text_file_bytes"]
    if size > limit:
        records.append({"path": safe_rel(src, root), "status": "SKIPPED_TOO_LARGE", "size": size})
        return False
    rel = Path(safe_rel(src, root))
    dst = dst_root / "snapshot" / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    records.append({
        "path": rel.as_posix(), "status": "COPIED", "size": size,
        "sha256": sha256_file(src), "snapshot_path": (Path("snapshot") / rel).as_posix(),
    })
    return True


def python_api(path: Path) -> dict[str, Any]:
    out: dict[str, Any] = {"file": str(path), "classes": [], "functions": [], "imports": []}
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
        tree = ast.parse(text)
        for node in tree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                try:
                    out["imports"].append(ast.unparse(node))
                except Exception:
                    pass
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                out["functions"].append({"name": node.name, "signature": signature_from_ast(node)})
            elif isinstance(node, ast.ClassDef):
                methods = []
                for child in node.body:
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        methods.append({"name": child.name, "signature": signature_from_ast(child)})
                out["classes"].append({"name": node.name, "methods": methods})
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"
    return out


def signature_from_ast(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    try:
        args = ast.unparse(node.args)
        ret = f" -> {ast.unparse(node.returns)}" if node.returns else ""
        return f"{node.name}({args}){ret}"
    except Exception:
        return node.name


def csv_schema(path: Path) -> dict[str, Any]:
    out: dict[str, Any] = {"path": str(path), "size": path.stat().st_size}
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.reader(f)
            header = next(reader, [])
            rows = []
            for _, row in zip(range(3), reader):
                rows.append(row)
        out.update({"columns": header, "sample_rows": rows})
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"
    return out


def checkpoint_metadata(path: Path) -> dict[str, Any]:
    out: dict[str, Any] = {"path": str(path), "size": path.stat().st_size, "sha256": sha256_file(path)}
    try:
        import torch  # type: ignore
        obj = torch.load(path, map_location="cpu", weights_only=False)
        out["object_type"] = type(obj).__name__
        if isinstance(obj, dict):
            out["top_level_keys"] = sorted(map(str, obj.keys()))
            state = None
            state_key = None
            for k in ("model_state_dict", "state_dict", "model", "net", "network"):
                v = obj.get(k)
                if isinstance(v, dict):
                    state = v
                    state_key = k
                    break
            if state is None and obj and all(hasattr(v, "shape") for v in obj.values()):
                state = obj
                state_key = "<root_state_dict>"
            out["state_dict_key"] = state_key
            if isinstance(state, dict):
                params = []
                total = 0
                for k, v in state.items():
                    shape = list(v.shape) if hasattr(v, "shape") else None
                    numel = int(v.numel()) if hasattr(v, "numel") else None
                    if numel is not None:
                        total += numel
                    params.append({"key": str(k), "shape": shape, "dtype": str(getattr(v, "dtype", "")), "numel": numel})
                out["state_dict"] = params
                out["total_state_numel"] = total
            scalars = {}
            for k, v in obj.items():
                if isinstance(v, (str, int, float, bool)) or v is None:
                    scalars[str(k)] = v
            out["scalar_metadata"] = scalars
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"
        out["traceback_tail"] = traceback.format_exc().splitlines()[-5:]
    return out


def iter_text_files(root: Path) -> Iterable[Path]:
    allowed = {".py", ".yaml", ".yml", ".json", ".md", ".txt", ".csv"}
    excluded_dirs = {".git", "__pycache__", "PROJECT_CONTROL", "backups", "data", "checkpoints"}
    for base in ["models", "datasets", "features", "training", "scripts", "configs", "evaluation", "losses"]:
        d = root / base
        if not d.is_dir():
            continue
        for p in d.rglob("*"):
            if p.is_file() and p.suffix.lower() in allowed and not any(part in excluded_dirs for part in p.parts):
                yield p


def code_search(root: Path, patterns: list[str]) -> list[dict[str, Any]]:
    compiled = [(p, re.compile(re.escape(p), re.I)) for p in patterns]
    hits: list[dict[str, Any]] = []
    for path in iter_text_files(root):
        try:
            if path.stat().st_size > 2 * 1024 * 1024:
                continue
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
            for lineno, line in enumerate(lines, 1):
                matched = [name for name, rx in compiled if rx.search(line)]
                if matched:
                    hits.append({"file": safe_rel(path, root), "line": lineno, "patterns": matched, "text": line[:500]})
                    if len(hits) >= 2500:
                        return hits
        except Exception:
            continue
    return hits


def discover_files(root: Path, folds: list[int]) -> dict[str, list[Path]]:
    found: dict[str, list[Path]] = {"manifests": [], "prediction_csvs": [], "metric_csvs": [], "configs": [], "checkpoints": [], "source_candidates": []}
    for fold in folds:
        for pat in [
            f"results/data_audit/dataset_v4_multifold/fold_{fold:02d}_manifest.csv",
            f"results/data_audit/dataset_v4_multifold/fold_{fold}_manifest.csv",
        ]:
            p = root / pat
            if p.is_file():
                found["manifests"].append(p)
        ckpt_patterns = [
            f"results/experiments/*fold{fold:02d}*power2*/**/best.pt",
            f"results/experiments/*power2*fold{fold:02d}*/**/best.pt",
            f"results/experiments/*fold{fold:02d}*/**/best.pt",
        ]
        for pat in ckpt_patterns:
            for p in root.glob(pat):
                if p.is_file() and p not in found["checkpoints"]:
                    found["checkpoints"].append(p)
    results = root / "results"
    if results.is_dir():
        for p in results.rglob("*.csv"):
            low = p.name.lower()
            rel_low = safe_rel(p, root).lower()
            if any(x in low for x in ["prediction", "predictions", "detail", "sample"]):
                if any(f"fold{f:02d}" in rel_low or f"fold_{f:02d}" in rel_low for f in folds) or "polarimetric" in rel_low:
                    found["prediction_csvs"].append(p)
            if any(x in low for x in ["metric", "summary", "aggregate", "history", "threshold"]):
                if "polarimetric" in rel_low or any(f"fold{f:02d}" in rel_low for f in folds):
                    found["metric_csvs"].append(p)
        for p in results.rglob("*.json"):
            if "polarimetric" in safe_rel(p, root).lower() or any(f"fold{f:02d}" in safe_rel(p, root).lower() for f in folds):
                found["configs"].append(p)
    return {k: sorted(set(v)) for k, v in found.items()}


def project_tree(root: Path) -> str:
    lines = []
    keep_top = {"models", "datasets", "features", "training", "scripts", "configs", "results", "docs", "evaluation", "losses"}
    for top in sorted(keep_top):
        d = root / top
        if not d.exists():
            continue
        lines.append(f"{top}/")
        count = 0
        for p in sorted(d.rglob("*")):
            if count >= 700:
                lines.append(f"  ... truncated after {count} entries")
                break
            if any(x in p.parts for x in ["__pycache__", ".git", "PROJECT_CONTROL"]):
                continue
            rel = p.relative_to(d)
            depth = len(rel.parts)
            if depth > 5:
                continue
            if p.is_dir():
                continue
            size = p.stat().st_size
            lines.append(f"  {rel.as_posix()}\t{size}")
            count += 1
    return "\n".join(lines) + "\n"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for k in row:
            if k not in keys:
                keys.append(k)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys or ["empty"])
        writer.writeheader()
        if rows:
            writer.writerows(rows)


def main() -> int:
    args = parse_args()
    root = args.project_root.resolve()
    output = (args.output or (root / OUTPUT_NAME)).resolve()
    cfg = load_config(root)
    folds = sorted(set(args.folds))
    if not (root / "scripts").is_dir():
        raise RuntimeError(f"项目根目录判断失败: {root}")

    temp_parent = root / "results" / "data_audit"
    temp_parent.mkdir(parents=True, exist_ok=True)
    stage_dir = temp_parent / STAGE_NAME
    if stage_dir.exists():
        shutil.rmtree(stage_dir)
    stage_dir.mkdir(parents=True)

    records: list[dict[str, Any]] = []
    expected_status = []
    python_apis = []
    csv_schemas = []
    checkpoint_info = []

    # Fixed expected files.
    for rel_s in cfg["expected_files"]:
        p = root / rel_s
        status = "FOUND" if p.is_file() else "MISSING"
        expected_status.append({"path": rel_s, "status": status})
        if p.is_file():
            copy_small_file(p, stage_dir, root, cfg, records)
            if p.suffix.lower() == ".py":
                api = python_api(p)
                api["file"] = rel_s
                python_apis.append(api)

    discovered = discover_files(root, folds)
    for category in ["manifests", "prediction_csvs", "metric_csvs", "configs"]:
        for p in discovered[category]:
            if copy_small_file(p, stage_dir, root, cfg, records) and p.suffix.lower() == ".csv":
                schema = csv_schema(p)
                schema["path"] = safe_rel(p, root)
                csv_schemas.append(schema)

    # Important model/dataset/training files discovered by names and code references.
    candidate_globs = [
        "models/**/*polar*.py", "models/**/*fcn*.py", "models/**/*refin*.py",
        "datasets/**/*detection*.py", "datasets/**/*polar*.py",
        "features/**/*polar*.py", "training/**/*polar*.py", "evaluation/**/*.py",
    ]
    extra_sources = []
    for pat in candidate_globs:
        for p in root.glob(pat):
            if p.is_file() and p not in extra_sources:
                extra_sources.append(p)
    for p in sorted(extra_sources):
        if copy_small_file(p, stage_dir, root, cfg, records):
            api = python_api(p)
            api["file"] = safe_rel(p, root)
            python_apis.append(api)

    # Do not copy checkpoint bytes; only inspect metadata.
    for p in discovered["checkpoints"]:
        checkpoint_info.append(checkpoint_metadata(p))

    patterns = [
        "power2", "polar6_gated", "ri8_gated", "pred_range", "pred_velocity",
        "localization_ok", "test_predictions", "validation_predictions", "best.pt",
        "DetectionRadarDatasetV3", "polarimetric_confidence", "relative_ZDR", "local_rho",
    ]
    hits = code_search(root, patterns)

    (stage_dir / "EXPECTED_INTERFACE_STATUS.json").write_text(json.dumps(expected_status, ensure_ascii=False, indent=2), encoding="utf-8")
    (stage_dir / "PYTHON_API_SIGNATURES.json").write_text(json.dumps(python_apis, ensure_ascii=False, indent=2), encoding="utf-8")
    (stage_dir / "CSV_SCHEMAS.json").write_text(json.dumps(csv_schemas, ensure_ascii=False, indent=2), encoding="utf-8")
    (stage_dir / "CHECKPOINT_METADATA.json").write_text(json.dumps(checkpoint_info, ensure_ascii=False, indent=2), encoding="utf-8")
    (stage_dir / "DISCOVERED_PATHS.json").write_text(json.dumps({k: [safe_rel(p, root) for p in v] for k, v in discovered.items()}, ensure_ascii=False, indent=2), encoding="utf-8")
    (stage_dir / "CODE_SEARCH_RESULTS.json").write_text(json.dumps(hits, ensure_ascii=False, indent=2), encoding="utf-8")
    (stage_dir / "PROJECT_TREE.txt").write_text(project_tree(root), encoding="utf-8")
    write_csv(stage_dir / "COPIED_FILES.csv", records)

    status = {
        "status": "PASS",
        "project_root": str(root),
        "folds": folds,
        "expected_found": sum(x["status"] == "FOUND" for x in expected_status),
        "expected_missing": [x["path"] for x in expected_status if x["status"] == "MISSING"],
        "copied_files": sum(x.get("status") == "COPIED" for x in records),
        "manifest_count": len(discovered["manifests"]),
        "prediction_csv_count": len(discovered["prediction_csvs"]),
        "checkpoint_metadata_count": len(checkpoint_info),
        "code_search_hits": len(hits),
        "raw_data_copied": False,
        "checkpoint_bytes_copied": False,
        "next_action": "Upload roi_polarimetric_stage4_context_acceptance_v1.zip for exact Stage 4 implementation.",
    }
    (stage_dir / "CONTEXT_COLLECTION_STATUS.json").write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")

    readme = f"""# Stage 4 ROI极化精修工程上下文采集结果\n\n- status: {status['status']}\n- folds: {folds}\n- expected interfaces found: {status['expected_found']}\n- manifests: {status['manifest_count']}\n- prediction CSVs: {status['prediction_csv_count']}\n- checkpoint metadata records: {status['checkpoint_metadata_count']}\n- copied source/config/result files: {status['copied_files']}\n\n本包不包含原始`.mat`数据或checkpoint权重，只包含实现Stage 4所需的源码、配置、CSV结构和权重键名/形状。\n"""
    (stage_dir / "README_CONTEXT_COLLECTION.md").write_text(readme, encoding="utf-8")

    if output.exists():
        output.unlink()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(stage_dir.rglob("*")):
            if p.is_file():
                zf.write(p, p.relative_to(stage_dir).as_posix())

    print("=" * 72)
    print("Stage 4 ROI极化精修上下文采集完成")
    print(f"expected found   : {status['expected_found']}")
    print(f"expected missing : {len(status['expected_missing'])}")
    print(f"manifests        : {status['manifest_count']}")
    print(f"prediction CSVs  : {status['prediction_csv_count']}")
    print(f"checkpoint meta  : {status['checkpoint_metadata_count']}")
    print(f"copied files     : {status['copied_files']}")
    print(f"acceptance ZIP   : {output}")
    print("raw data copied  : false")
    print("checkpoint copied: false")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"FATAL: {type(exc).__name__}: {exc}", file=sys.stderr)
        traceback.print_exc()
        raise SystemExit(1)
