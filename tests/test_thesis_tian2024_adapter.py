from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from evaluation.evaluate_thesis_tian2024_adapter import compute_frozen_metrics
from evaluation.thesis_tian2024_postprocess import direct_max_detections
from features.polarimetric_rd import (
    PolarimetricConfig,
    explicit_polarimetric_rd,
    make_thesis_tian6,
)
from models.thesis_tian2024_adapter import (
    ThesisTian2024Adapter,
    zscore_rd_batch,
)
from training.thesis_tian2024_objective import (
    ThesisTian2024Objective,
    build_thesis_tian2024_targets,
)
from training.train_thesis_tian2024_adapter import make_seeded_training_subset


def test_six_channel_input_is_finite_and_aligned() -> None:
    rng = np.random.default_rng(42)
    h = rng.normal(size=(128, 100)) + 1j * rng.normal(size=(128, 100))
    v = rng.normal(size=(128, 100)) + 1j * rng.normal(size=(128, 100))
    features = explicit_polarimetric_rd(
        h.astype(np.complex64),
        v.astype(np.complex64),
        PolarimetricConfig(velocity_window=3, range_window=3),
    )
    channels = make_thesis_tian6(features)
    assert channels.shape == (6, 128, 100)
    assert np.isfinite(channels).all()
    assert np.allclose(channels[0], features["rd_h"].real)
    assert np.all((channels[5] >= 0.0) & (channels[5] <= 1.0))


def test_model_has_local_output_geometry_and_finite_values() -> None:
    model = ThesisTian2024Adapter(normalization_scope="sample_channel")
    output = model(torch.randn(2, 6, 128, 100))
    assert output.original_shape == (128, 100)
    assert output.padded_shape == (128, 100)
    assert output.classification_logits.shape == (2, 1, 32, 25)
    assert output.normalized_offsets.shape == (2, 2, 32, 25)
    assert torch.isfinite(output.classification_logits).all()


def test_zscore_scopes_are_explicit() -> None:
    values = torch.arange(2 * 2 * 4 * 5, dtype=torch.float32).reshape(2, 2, 4, 5)
    per_sample = zscore_rd_batch(values, "sample_channel")
    assert torch.allclose(per_sample.mean(dim=(2, 3)), torch.zeros(2, 2), atol=1e-6)
    per_batch = zscore_rd_batch(values, "batch_channel")
    assert torch.allclose(per_batch.mean(dim=(0, 2, 3)), torch.zeros(2), atol=1e-6)
    with pytest.raises(ValueError):
        zscore_rd_batch(values, "unknown")


def test_targets_pool_positive_area_and_keep_responsible_offset() -> None:
    targets = build_thesis_tian2024_targets(
        torch.tensor([1, 0]),
        torch.tensor([53, -1]),
        torch.tensor([68, -1]),
        (128, 100),
    )
    assert targets.classification.shape == (2, 1, 32, 25)
    assert 1 < int(targets.classification[0].sum()) <= 6
    assert int(targets.classification[1].sum()) == 0
    assert targets.regression_mask[0, 0, 13, 17]
    assert int(targets.regression_mask.sum()) == 1
    assert targets.normalized_offsets[0, 0, 13, 17] == 0
    assert targets.normalized_offsets[0, 1, 13, 17] == pytest.approx(0.25)


def test_balanced_objective_backpropagates() -> None:
    model = ThesisTian2024Adapter("sample_channel")
    output = model(torch.randn(2, 6, 128, 100))
    targets = build_thesis_tian2024_targets(
        torch.tensor([1, 0]),
        torch.tensor([53, -1]),
        torch.tensor([68, -1]),
        (128, 100),
    )
    loss = ThesisTian2024Objective(10.0)(
        output.classification_logits,
        output.normalized_offsets,
        targets,
    )
    assert loss.positive_units == loss.sampled_negative_units
    assert loss.regression_units == 1
    assert torch.isfinite(loss.total)
    loss.total.backward()
    assert model.shared_conv1.weight.grad is not None


def test_direct_max_decodes_responsible_cell_and_threshold() -> None:
    logits = torch.full((2, 1, 32, 25), -10.0)
    offsets = torch.zeros((2, 2, 32, 25))
    logits[0, 0, 13, 17] = 10.0
    offsets[0, 0, 13, 17] = 0.0
    offsets[0, 1, 13, 17] = 0.25
    detections = direct_max_detections(logits, offsets, (128, 100), threshold=0.5)
    assert detections[0] is not None
    assert detections[0].range_index == 68
    assert detections[0].velocity_index == 53
    assert detections[1] is None


def test_thesis_training_subset_selection_is_seeded_and_class_limited() -> None:
    class StubDataset:
        records = (
            [{"target_present": 0} for _ in range(10)]
            + [{"target_present": 1} for _ in range(8)]
        )

        def __len__(self) -> int:
            return len(self.records)

    dataset = StubDataset()
    first = make_seeded_training_subset(dataset, 6, 4, seed=42)
    second = make_seeded_training_subset(dataset, 6, 4, seed=42)
    assert first.indices == second.indices
    labels = [dataset.records[index]["target_present"] for index in first.indices]
    assert labels.count(0) == 6
    assert labels.count(1) == 4


def test_frozen_metrics_never_select_or_change_threshold() -> None:
    table = pd.DataFrame(
        [
            {
                "target_present": 1,
                "peak_score": 0.8,
                "peak_grid_x": 2,
                "peak_grid_y": 3,
                "pred_range_index": 9,
                "pred_velocity_index": 13,
                "true_range_index": 8,
                "true_velocity_index": 12,
            },
            {
                "target_present": 0,
                "peak_score": 0.7,
                "peak_grid_x": 1,
                "peak_grid_y": 1,
                "pred_range_index": 4,
                "pred_velocity_index": 4,
                "true_range_index": -1,
                "true_velocity_index": -1,
            },
        ]
    )
    metrics, predictions = compute_frozen_metrics(table, 0.75, 2, 3)
    assert metrics["frozen_threshold"] == 0.75
    assert metrics["joint_pd"] == 1.0
    assert metrics["pfa"] == 0.0
    assert predictions["detected"].tolist() == [True, False]
