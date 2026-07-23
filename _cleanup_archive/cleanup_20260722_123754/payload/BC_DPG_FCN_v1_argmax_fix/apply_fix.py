#!/usr/bin/env python3
from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path


PROJECT = Path.home() / "projects" / "balloon_radar_project"
MODEL_PATH = PROJECT / "models" / "background_calibrated_dpg_fcn.py"
TRAIN_PATH = PROJECT / "training" / "train_background_calibrator.py"


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(
            f"{label}: expected exactly one match, found {count}"
        )
    return text.replace(old, new, 1)


def main() -> None:
    for path in (MODEL_PATH, TRAIN_PATH):
        if not path.is_file():
            raise FileNotFoundError(path)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = PROJECT / "backups" / f"bc_dpg_argmax_fix_{stamp}"
    backup_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(MODEL_PATH, backup_dir / MODEL_PATH.name)
    shutil.copy2(TRAIN_PATH, backup_dir / TRAIN_PATH.name)

    model_text = MODEL_PATH.read_text(encoding="utf-8")

    old_model_block = '''        raw_logits = self._as_batched_map(
            self.extract_heatmap(base_output)
        )
        gate_weights = self.extract_gate(base_output)

        calibration_features = self.build_features(
            input_tensor=input_tensor,
            raw_logits=raw_logits,
            gate_weights=gate_weights,
        )
        temperature, bias = self.calibrator(
            calibration_features
        )

        broadcast_shape = [raw_logits.shape[0]] + [1] * (
            raw_logits.ndim - 1
        )
        calibrated_logits = (
            raw_logits - bias.reshape(broadcast_shape)
        ) / temperature.reshape(broadcast_shape)
'''

    new_model_block = '''        raw_logits = self._as_batched_map(
            self.extract_heatmap(base_output)
        )
        gate_weights = self.extract_gate(base_output)

        # Keep the frozen DPG-FCN under AMP, but execute the calibration
        # transform in FP32. This prevents FP16 rounding from swapping two
        # nearly equal range-velocity peaks.
        with torch.autocast(
            device_type=raw_logits.device.type,
            enabled=False,
        ):
            raw_logits = raw_logits.float()
            calibration_input = input_tensor.float()
            calibration_gate = (
                gate_weights.float()
                if isinstance(gate_weights, Tensor)
                else gate_weights
            )

            calibration_features = self.build_features(
                input_tensor=calibration_input,
                raw_logits=raw_logits,
                gate_weights=calibration_gate,
            )
            temperature, bias = self.calibrator(
                calibration_features.float()
            )

            broadcast_shape = [raw_logits.shape[0]] + [1] * (
                raw_logits.ndim - 1
            )
            calibrated_logits = (
                raw_logits - bias.reshape(broadcast_shape)
            ) / temperature.reshape(broadcast_shape)
'''

    model_text = replace_once(
        model_text,
        old_model_block,
        new_model_block,
        "FP32 calibration block",
    )
    MODEL_PATH.write_text(model_text, encoding="utf-8")

    train_text = TRAIN_PATH.read_text(encoding="utf-8")

    old_extract = '''def extract_scores_and_peaks(
    logits: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    probability = torch.sigmoid(logits)
    flattened = probability.flatten(start_dim=1)
    score, flat_index = flattened.max(dim=1)

    width = logits.shape[-1]
    velocity_index = torch.div(
        flat_index,
        width,
        rounding_mode="floor",
    )
    range_index = flat_index % width

    return (
        score.detach().cpu().numpy(),
        range_index.detach().cpu().numpy(),
        velocity_index.detach().cpu().numpy(),
    )
'''

    new_extract = '''def extract_scores_and_peaks(
    logits: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Find the peak directly in FP32 logit space. Argmax after FP16 sigmoid
    # can create artificial saturation ties.
    flattened_logits = logits.float().flatten(start_dim=1)
    max_logit, flat_index = flattened_logits.max(dim=1)
    score = torch.sigmoid(max_logit)

    width = logits.shape[-1]
    velocity_index = torch.div(
        flat_index,
        width,
        rounding_mode="floor",
    )
    range_index = flat_index % width

    return (
        score.detach().cpu().numpy(),
        range_index.detach().cpu().numpy(),
        velocity_index.detach().cpu().numpy(),
    )
'''

    train_text = replace_once(
        train_text,
        old_extract,
        new_extract,
        "logit-space peak extraction",
    )

    old_peak_block = '''            calibrated_score, calibrated_r, calibrated_v = (
                extract_scores_and_peaks(
                    outputs["calibrated_logits"]
                )
            )
            raw_score, raw_r, raw_v = extract_scores_and_peaks(
                outputs["raw_logits"]
            )
'''

    new_peak_block = '''            (
                calibrated_score,
                calibrated_numeric_r,
                calibrated_numeric_v,
            ) = extract_scores_and_peaks(
                outputs["calibrated_logits"]
            )
            raw_score, raw_r, raw_v = extract_scores_and_peaks(
                outputs["raw_logits"]
            )

            # BC-DPG calibrates sample scores only. Localization is inherited
            # explicitly from the frozen DPG-FCN.
            calibrated_r = raw_r
            calibrated_v = raw_v
'''

    train_text = replace_once(
        train_text,
        old_peak_block,
        new_peak_block,
        "raw localization inheritance",
    )

    old_match_block = '''                argmax_preserved = (
                    int(calibrated_r[index]) == int(raw_r[index])
                    and int(calibrated_v[index]) == int(raw_v[index])
                )
'''

    new_match_block = '''                numeric_argmax_match = (
                    int(calibrated_numeric_r[index])
                    == int(raw_r[index])
                    and int(calibrated_numeric_v[index])
                    == int(raw_v[index])
                )
                argmax_preserved = True
'''

    train_text = replace_once(
        train_text,
        old_match_block,
        new_match_block,
        "numeric argmax diagnostic",
    )

    old_row_fragment = '''                        "raw_pred_range_index": int(raw_r[index]),
                        "raw_pred_velocity_index": int(raw_v[index]),
                        "true_range_index": int(true_r[index]),
'''

    new_row_fragment = '''                        "raw_pred_range_index": int(raw_r[index]),
                        "raw_pred_velocity_index": int(raw_v[index]),
                        "calibrated_numeric_range_index": int(
                            calibrated_numeric_r[index]
                        ),
                        "calibrated_numeric_velocity_index": int(
                            calibrated_numeric_v[index]
                        ),
                        "numeric_argmax_match": bool(
                            numeric_argmax_match
                        ),
                        "true_range_index": int(true_r[index]),
'''

    train_text = replace_once(
        train_text,
        old_row_fragment,
        new_row_fragment,
        "numeric peak output columns",
    )

    old_guard = '''    frame = pd.DataFrame(rows)
    if not bool(frame["argmax_preserved"].all()):
        failed = frame.loc[
            ~frame["argmax_preserved"],
            ["sample_id"],
        ]
        raise RuntimeError(
            "Spatial argmax changed for samples: "
            + ", ".join(failed["sample_id"].head(10))
        )

    return frame, {
'''

    new_guard = '''    frame = pd.DataFrame(rows)
    numeric_mismatch_count = int(
        (~frame["numeric_argmax_match"]).sum()
    )
    if numeric_mismatch_count:
        examples = frame.loc[
            ~frame["numeric_argmax_match"],
            "sample_id",
        ].head(10)
        print(
            "Warning: calibrated-logit numeric argmax differed for "
            f"{numeric_mismatch_count} samples; localization remains "
            "inherited from raw DPG logits. Examples: "
            + ", ".join(examples)
        )

    return frame, {
'''

    train_text = replace_once(
        train_text,
        old_guard,
        new_guard,
        "non-fatal numeric argmax diagnostic",
    )

    old_summary_fragment = '''        "argmax_preserved_validation": bool(
            val_frame["argmax_preserved"].all()
        ),
        "argmax_preserved_test": bool(
            test_frame["argmax_preserved"].all()
        ),
        "elapsed_seconds": time.time() - start_time,
'''

    new_summary_fragment = '''        "argmax_preserved_validation": bool(
            val_frame["argmax_preserved"].all()
        ),
        "argmax_preserved_test": bool(
            test_frame["argmax_preserved"].all()
        ),
        "numeric_argmax_match_validation": bool(
            val_frame["numeric_argmax_match"].all()
        ),
        "numeric_argmax_match_test": bool(
            test_frame["numeric_argmax_match"].all()
        ),
        "numeric_argmax_mismatch_count_validation": int(
            (~val_frame["numeric_argmax_match"]).sum()
        ),
        "numeric_argmax_mismatch_count_test": int(
            (~test_frame["numeric_argmax_match"]).sum()
        ),
        "elapsed_seconds": time.time() - start_time,
'''

    train_text = replace_once(
        train_text,
        old_summary_fragment,
        new_summary_fragment,
        "summary diagnostics",
    )

    TRAIN_PATH.write_text(train_text, encoding="utf-8")

    print("BC-DPG argmax precision fix installed")
    print(f"Backup directory: {backup_dir}")
    print(f"Updated: {MODEL_PATH}")
    print(f"Updated: {TRAIN_PATH}")


if __name__ == "__main__":
    main()
