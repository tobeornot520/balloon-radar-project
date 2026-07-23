#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.roi_polarimetric_refinement_dataset import ROIPolarimetricSourceDataset
from features.roi_polarimetric_refinement import ROIConfig, crop_roi, logit_from_probability
from models.polarimetric_representation_fcn import PolarimetricRepresentationFCN


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build frozen Power2 candidate ROI caches.")
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--manifest-path", required=True)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--debug-per-class", type=int, default=0)
    parser.add_argument("--roi-velocity-radius", type=int, default=5)
    parser.add_argument("--roi-range-radius", type=int, default=4)
    parser.add_argument("--range-tolerance-gates", type=int, default=2)
    parser.add_argument("--velocity-tolerance-bins", type=int, default=3)
    parser.add_argument("--velocity-window", type=int, default=5)
    parser.add_argument("--range-window", type=int, default=3)
    parser.add_argument("--zdr-clip-db", type=float, default=20.0)
    parser.add_argument("--gate-low-percentile", type=float, default=50.0)
    parser.add_argument("--gate-high-percentile", type=float, default=99.0)
    parser.add_argument("--gate-gamma", type=float, default=1.5)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve(path: str) -> Path:
    value = Path(path).expanduser()
    if not value.is_absolute():
        value = PROJECT_ROOT / value
    return value.resolve()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def to_list(value: Any, length: int) -> list[Any]:
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value] * length


def main() -> None:
    args = parse_args()
    if args.fold <= 0:
        raise ValueError("fold must be positive")
    manifest = resolve(args.manifest_path)
    checkpoint_path = resolve(args.base_checkpoint)
    output_dir = resolve(args.output_dir)
    if not manifest.is_file() or not checkpoint_path.is_file():
        raise FileNotFoundError(f"manifest/checkpoint missing: {manifest} / {checkpoint_path}")
    if output_dir.exists() and not args.overwrite:
        status = output_dir / "cache_status.json"
        expected = [output_dir / f"{split}.pt" for split in ("train", "val", "test")]
        if status.is_file() and all(path.is_file() for path in expected):
            print(f"[cache complete] {output_dir}")
            return
        raise FileExistsError(f"Incomplete cache exists: {output_dir}; use --overwrite")
    if output_dir.exists():
        import shutil
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("input_mode") != "power2":
        raise ValueError(f"Expected a Power2 checkpoint, got {checkpoint.get('input_mode')}")
    if "model_state_dict" not in checkpoint or "threshold" not in checkpoint:
        raise ValueError("Power2 checkpoint lacks model_state_dict or threshold")
    model = PolarimetricRepresentationFCN()
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    base_threshold = float(checkpoint["threshold"])
    roi_config = ROIConfig(args.roi_velocity_radius, args.roi_range_radius)

    status_rows: list[dict[str, Any]] = []
    for split in ("train", "val", "test"):
        dataset = ROIPolarimetricSourceDataset(
            manifest_path=manifest,
            split=split,
            velocity_window=args.velocity_window,
            range_window=args.range_window,
            zdr_clip_db=args.zdr_clip_db,
            gate_low_percentile=args.gate_low_percentile,
            gate_high_percentile=args.gate_high_percentile,
            gate_gamma=args.gate_gamma,
            debug_per_class=args.debug_per_class,
        )
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )
        collected: dict[str, list[torch.Tensor]] = {
            "roi_source": [],
            "roi_valid_mask": [],
            "raw_score": [],
            "raw_logit": [],
            "target_present": [],
            "localization_ok": [],
            "pred_range_index": [],
            "pred_velocity_index": [],
            "true_range_index": [],
            "true_velocity_index": [],
            "roi_quality": [],
            "polarimetric_confidence": [],
        }
        metadata: list[dict[str, Any]] = []
        with torch.inference_mode():
            for batch in loader:
                source = batch["roi_source"].float()
                confidence = batch["confidence_map"].float()
                base_input = torch.zeros(
                    (source.shape[0], 8, 128, 100), dtype=source.dtype, device=device
                )
                base_input[:, :2] = source[:, :2].to(device, non_blocking=True)
                logits = model(base_input)
                flat_logits = logits[:, 0].flatten(1)
                raw_logit, flat_index = flat_logits.max(dim=1)
                raw_score = torch.sigmoid(raw_logit)
                pred_velocity = torch.div(flat_index, 100, rounding_mode="floor")
                pred_range = flat_index % 100

                present = batch["target_present"].long()
                true_range = batch["range_index"].long()
                true_velocity = batch["velocity_index"].long()
                loc_ok = (
                    present.bool()
                    & ((pred_range.cpu() - true_range).abs() <= args.range_tolerance_gates)
                    & ((pred_velocity.cpu() - true_velocity).abs() <= args.velocity_tolerance_bins)
                )

                for index in range(source.shape[0]):
                    roi, mask = crop_roi(
                        source[index],
                        int(pred_velocity[index].item()),
                        int(pred_range[index].item()),
                        roi_config,
                    )
                    conf_roi, _ = crop_roi(
                        confidence[index],
                        int(pred_velocity[index].item()),
                        int(pred_range[index].item()),
                        roi_config,
                    )
                    valid_count = mask.sum().clamp_min(1.0)
                    joint_power = roi[0:2].amax(dim=0, keepdim=True)
                    roi_quality = (joint_power * mask).sum() / valid_count
                    polar_conf = (conf_roi * mask).sum() / valid_count
                    collected["roi_source"].append(roi.cpu())
                    collected["roi_valid_mask"].append(mask.cpu())
                    collected["raw_score"].append(raw_score[index].detach().cpu())
                    collected["raw_logit"].append(raw_logit[index].detach().cpu())
                    collected["target_present"].append(present[index].cpu())
                    collected["localization_ok"].append(loc_ok[index].cpu())
                    collected["pred_range_index"].append(pred_range[index].cpu())
                    collected["pred_velocity_index"].append(pred_velocity[index].cpu())
                    collected["true_range_index"].append(true_range[index].cpu())
                    collected["true_velocity_index"].append(true_velocity[index].cpu())
                    collected["roi_quality"].append(roi_quality.cpu())
                    collected["polarimetric_confidence"].append(polar_conf.cpu())

                count = source.shape[0]
                fields = {
                    key: to_list(batch[key], count)
                    for key in (
                        "sample_id", "source_file", "beam_layer", "azimuth_deg",
                        "distance_m", "velocity_mps", "mat_path",
                    )
                }
                for index in range(count):
                    metadata.append({key: fields[key][index] for key in fields})

        payload = {
            key: torch.stack(values) for key, values in collected.items()
        }
        payload.update({
            "metadata": metadata,
            "base_threshold": base_threshold,
            "base_checkpoint": str(checkpoint_path),
            "base_checkpoint_sha256": sha256(checkpoint_path),
            "manifest_path": str(manifest),
            "fold": int(args.fold),
            "split": split,
            "roi_shape": [roi_config.height, roi_config.width],
            "roi_source_channels": 10,
            "debug_per_class": int(args.debug_per_class),
            "sample_independent": True,
            "scan_context": False,
        })
        cache_path = output_dir / f"{split}.pt"
        torch.save(payload, cache_path)

        frame = pd.DataFrame(metadata)
        frame["target_present"] = payload["target_present"].numpy()
        frame["raw_score"] = payload["raw_score"].numpy()
        frame["pred_range_index"] = payload["pred_range_index"].numpy()
        frame["pred_velocity_index"] = payload["pred_velocity_index"].numpy()
        frame["true_range_index"] = payload["true_range_index"].numpy()
        frame["true_velocity_index"] = payload["true_velocity_index"].numpy()
        frame["localization_ok"] = payload["localization_ok"].numpy()
        frame["roi_quality"] = payload["roi_quality"].numpy()
        frame["polarimetric_confidence"] = payload["polarimetric_confidence"].numpy()
        frame.to_csv(output_dir / f"{split}_candidate_inventory.csv", index=False, encoding="utf-8-sig")
        status_rows.append({
            "split": split,
            "samples": len(dataset),
            "targets": int(payload["target_present"].sum().item()),
            "background": int((payload["target_present"] == 0).sum().item()),
            "localized_targets": int(payload["localization_ok"].sum().item()),
        })

    status = {
        "status": "PASS",
        "fold": int(args.fold),
        "manifest_path": str(manifest),
        "base_checkpoint": str(checkpoint_path),
        "base_checkpoint_sha256": sha256(checkpoint_path),
        "base_threshold": base_threshold,
        "base_checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "roi": {
            "velocity_radius": args.roi_velocity_radius,
            "range_radius": args.roi_range_radius,
            "height": roi_config.height,
            "width": roi_config.width,
        },
        "splits": status_rows,
        "sample_independent": True,
        "scan_context": False,
        "raw_data_copied_to_acceptance": False,
    }
    (output_dir / "cache_status.json").write_text(
        json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("=" * 88)
    print("Stage 4 frozen Power2 candidate cache complete")
    print(f"fold / threshold : {args.fold} / {base_threshold:.6f}")
    print(f"ROI shape        : {roi_config.height} x {roi_config.width}")
    for row in status_rows:
        print(
            f"{row['split']:>5}: n={row['samples']} target={row['targets']} "
            f"background={row['background']} localized={row['localized_targets']}"
        )
    print(f"output           : {output_dir}")
    print("=" * 88)


if __name__ == "__main__":
    main()
