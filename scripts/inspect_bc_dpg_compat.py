#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import importlib
import inspect
import json
import sys
from pathlib import Path

import pandas as pd
import torch


def summarize_checkpoint(path):
    checkpoint = torch.load(path, map_location="cpu")
    result = {
        "checkpoint_path": str(path.resolve()),
        "checkpoint_type": type(checkpoint).__name__,
    }

    if isinstance(checkpoint, dict):
        result["top_level_keys"] = list(checkpoint.keys())
        state_dict = None
        source = None

        for key in ("model_state_dict", "state_dict", "model", "net"):
            candidate = checkpoint.get(key)
            if isinstance(candidate, dict) and candidate:
                state_dict = candidate
                source = key
                break

        if state_dict is None and checkpoint:
            if all(
                isinstance(k, str) and isinstance(v, torch.Tensor)
                for k, v in checkpoint.items()
            ):
                state_dict = checkpoint
                source = "<root>"

        if state_dict is not None:
            keys = list(state_dict.keys())
            result["state_dict_source"] = source
            result["state_dict_key_count"] = len(keys)
            result["state_dict_preview"] = [
                {
                    "key": key,
                    "shape": list(state_dict[key].shape)
                    if isinstance(state_dict[key], torch.Tensor)
                    else None,
                }
                for key in keys[:80]
            ]

        scalar_metadata = {}
        for key, value in checkpoint.items():
            if isinstance(value, (str, int, float, bool)) or value is None:
                scalar_metadata[key] = value
        result["scalar_metadata"] = scalar_metadata

        for key in ("config", "args", "model_config", "hparams", "hyper_parameters"):
            if key in checkpoint:
                value = checkpoint[key]
                if hasattr(value, "__dict__"):
                    value = vars(value)
                try:
                    json.dumps(value)
                    result[key] = value
                except Exception:
                    result[key] = repr(value)

    return result


def summarize_manifest(path):
    frame = pd.read_csv(path)
    result = {
        "manifest_path": str(path.resolve()),
        "rows": int(len(frame)),
        "columns": list(frame.columns),
        "dtypes": {c: str(t) for c, t in frame.dtypes.items()},
        "head": frame.head(3).fillna("").to_dict(orient="records"),
    }

    for candidate in ("split", "subset", "role", "class", "label", "target"):
        if candidate in frame.columns:
            counts = frame[candidate].value_counts(dropna=False).to_dict()
            result[f"{candidate}_counts"] = {
                str(k): int(v) for k, v in counts.items()
            }

    hits = []
    for column in frame.select_dtypes(include=["object"]).columns:
        series = frame[column].fillna("").astype(str)
        mask = (
            series.str.startswith("/home/")
            | series.str.startswith("/mnt/")
            | series.str.match(r"^[A-Za-z]:\\")
        )
        if mask.any():
            hits.append(
                {
                    "column": column,
                    "count": int(mask.sum()),
                    "examples": series[mask].head(3).tolist(),
                }
            )
    result["absolute_path_hits"] = hits
    return result


def summarize_module(module_name):
    module = importlib.import_module(module_name)
    classes = []
    functions = []

    for name, obj in inspect.getmembers(module):
        if inspect.isclass(obj) and obj.__module__ == module.__name__:
            try:
                signature = str(inspect.signature(obj))
            except Exception:
                signature = "<unavailable>"
            classes.append({"name": name, "signature": signature})

        if inspect.isfunction(obj) and obj.__module__ == module.__name__:
            try:
                signature = str(inspect.signature(obj))
            except Exception:
                signature = "<unavailable>"
            functions.append({"name": name, "signature": signature})

    return {"module": module_name, "classes": classes, "functions": functions}


def summarize_python_file(path):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return {
        "path": str(path.resolve()),
        "classes": [
            n.name for n in tree.body if isinstance(n, ast.ClassDef)
        ],
        "functions": [
            n.name
            for n in tree.body
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--model-module",
        default="models.dual_branch_gated_fcn",
    )
    parser.add_argument(
        "--training-file",
        type=Path,
        default=Path("training/train_dual_branch_gated.py"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    project_root = Path.cwd().resolve()
    sys.path.insert(0, str(project_root))

    report = {
        "project_root": str(project_root),
        "checkpoint": summarize_checkpoint(args.checkpoint),
        "manifest": summarize_manifest(args.manifest),
        "model_module": summarize_module(args.model_module),
        "training_file": summarize_python_file(args.training_file),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    print(f"Compatibility report written to: {args.output}")
    for item in report["model_module"]["classes"]:
        print(f"model class: {item['name']}{item['signature']}")
    print(
        "state dict source:",
        report["checkpoint"].get("state_dict_source", "<not found>"),
    )
    print(
        "state dict keys:",
        report["checkpoint"].get("state_dict_key_count", 0),
    )
    print("manifest rows:", report["manifest"]["rows"])
    print("absolute path hits:", len(report["manifest"]["absolute_path_hits"]))


if __name__ == "__main__":
    main()
