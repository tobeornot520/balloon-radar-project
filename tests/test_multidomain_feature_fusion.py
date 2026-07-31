from __future__ import annotations

import pytest
import torch

from models.multidomain_feature_fusion import (
    DEFAULT_DOMAIN_DIMENSIONS,
    MultiDomainFeatureClassifier,
    MultiDomainFeatureFusion,
)


def make_inputs(batch: int = 3) -> dict[str, torch.Tensor]:
    return {
        name: torch.randn(batch, dimension)
        for name, dimension in DEFAULT_DOMAIN_DIMENSIONS.items()
    }


def test_multidomain_fusion_shapes_and_weights() -> None:
    model = MultiDomainFeatureFusion()
    output = model(make_inputs())
    assert output.embedding.shape == (3, 128)
    assert output.domain_weights.shape == (3, 5)
    assert torch.allclose(output.domain_weights.sum(1), torch.ones(3))
    assert torch.allclose(
        output.normalized_embedding.norm(dim=1), torch.ones(3), atol=1e-5
    )


def test_missing_domain_gets_zero_weight() -> None:
    model = MultiDomainFeatureFusion()
    inputs = make_inputs(batch=2)
    inputs.pop("tf")
    output = model(inputs)
    tf_index = model.domain_names.index("tf")
    assert torch.all(output.domain_weights[:, tf_index] == 0)


def test_per_sample_validity_masks_untrusted_polarimetry() -> None:
    model = MultiDomainFeatureFusion()
    inputs = make_inputs(batch=2)
    validity = torch.ones(2, len(model.domain_names))
    polar_index = model.domain_names.index("polar")
    validity[0, polar_index] = 0
    output = model(inputs, validity)
    assert output.domain_weights[0, polar_index] == 0
    assert output.domain_weights[1, polar_index] > 0


def test_classifier_head_is_replaceable_and_validated() -> None:
    model = MultiDomainFeatureClassifier(num_classes=5)
    output = model(make_inputs(batch=4))
    assert output["logits"].shape == (4, 5)
    with pytest.raises(ValueError):
        MultiDomainFeatureClassifier(num_classes=1)
