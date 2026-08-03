from __future__ import annotations

import numpy as np

from scripts.run_lat_mricd_grouped_baseline_v1 import (
    batch_class_sample_weights,
    build_batch_split_manifest,
    build_grouped_fold_assignments,
    extract_hrrp_features,
    extract_narrow_features,
)


def test_grouped_folds_keep_every_batch_in_one_heldout_fold() -> None:
    labels = np.repeat([1, 2, 3], 12)
    groups = np.repeat(np.arange(12), 3)
    labels = np.tile([1, 2, 3], 12)

    assignments = build_grouped_fold_assignments(
        labels,
        groups,
        n_splits=3,
        random_state=20260803,
    )

    assert set(assignments) == {1, 2, 3}
    for group in np.unique(groups):
        assert len(np.unique(assignments[groups == group])) == 1
    for fold in (1, 2, 3):
        assert set(labels[assignments == fold]) == {1, 2, 3}

    metadata = np.column_stack(
        [np.full(len(labels), 2), labels, labels, groups]
    )
    manifest = build_batch_split_manifest(
        task={
            "task_id": "synthetic",
            "representation": "Narrow",
            "band_code": 2,
            "band": "X",
            "class_codes": [1, 2, 3],
        },
        metadata=metadata,
        assignments=assignments,
    )
    assert len(manifest) == 12
    assert not {"sample_id", "source_row_index", "relative_path"} & set(
        manifest.columns
    )


def test_batch_class_weights_equalize_classes_and_cells() -> None:
    labels = np.asarray([1, 1, 1, 1, 2, 2, 2, 2, 2])
    groups = np.asarray([10, 11, 11, 11, 20, 20, 21, 21, 21])

    weights = batch_class_sample_weights(labels, groups)

    np.testing.assert_allclose(weights[labels == 1].sum(), weights[labels == 2].sum())
    for code in (1, 2):
        cell_totals = [
            weights[(labels == code) & (groups == group)].sum()
            for group in np.unique(groups[labels == code])
        ]
        np.testing.assert_allclose(cell_totals, np.repeat(cell_totals[0], len(cell_totals)))


def test_hrrp_features_are_finite_and_scale_invariant() -> None:
    amplitude = np.asarray(
        [
            [0.0, 1.0, 4.0, 2.0, 0.5, 0.0, 1.0, 0.0] * 4,
            [1.0, 0.0, 0.5, 3.0, 2.0, 1.0, 0.0, 0.5] * 4,
        ],
        dtype=np.float64,
    )

    original = extract_hrrp_features(amplitude)
    scaled = extract_hrrp_features(amplitude * 17.0)

    assert np.isfinite(original.to_numpy()).all()
    np.testing.assert_allclose(original, scaled, rtol=1e-10, atol=1e-10)


def test_narrow_features_ignore_global_scale_and_phase() -> None:
    index = np.arange(128, dtype=np.float64)
    iq = np.stack(
        [
            (1.0 + 0.2 * np.cos(2 * np.pi * index / 17))
            * np.exp(2j * np.pi * 0.08 * index),
            (1.0 + 0.1 * np.sin(2 * np.pi * index / 23))
            * np.exp(-2j * np.pi * 0.13 * index),
        ]
    )

    original = extract_narrow_features(iq)
    transformed = extract_narrow_features(iq * (4.5 * np.exp(1.2j)))

    assert np.isfinite(original.to_numpy()).all()
    np.testing.assert_allclose(original, transformed, rtol=1e-9, atol=1e-9)
