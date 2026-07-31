from __future__ import annotations

import pytest
import torch

from models.polarimetric_transfer_encoder import (
    PolarimetricTransferClassifier,
    PolarimetricTransferEncoder,
    count_parameters,
)


def test_encoder_accepts_stage4_roi_contract_and_arbitrary_size() -> None:
    encoder = PolarimetricTransferEncoder(embedding_dim=96)
    output = encoder(torch.randn(3, 10, 11, 9))
    assert output.embedding.shape == (3, 96)
    assert output.normalized_embedding.shape == (3, 96)
    assert output.feature_map.shape == (3, 64, 11, 9)
    assert torch.allclose(
        output.normalized_embedding.norm(dim=1),
        torch.ones(3),
        atol=1e-5,
    )
    assert count_parameters(encoder) > 0


def test_encoder_masks_uncalibrated_phase_channels() -> None:
    torch.manual_seed(7)
    encoder = PolarimetricTransferEncoder().eval()
    inputs = torch.randn(2, 10, 11, 9)
    validity = torch.ones(2, 10)
    validity[:, 8:10] = 0
    changed = inputs.clone()
    changed[:, 8:10] = 1000.0
    with torch.no_grad():
        first = encoder(inputs, validity).embedding
        second = encoder(changed, validity).embedding
    assert torch.allclose(first, second)


def test_transfer_classifier_has_replaceable_task_output() -> None:
    model = PolarimetricTransferClassifier(num_classes=4, embedding_dim=64)
    output = model(torch.randn(2, 10, 13, 7))
    assert output["logits"].shape == (2, 4)
    assert output["embedding"].shape == (2, 64)
    output["logits"].sum().backward()
    assert model.encoder.power_branch.network[0].weight.grad is not None


def test_encoder_rejects_invalid_contracts() -> None:
    encoder = PolarimetricTransferEncoder()
    with pytest.raises(ValueError, match="inputs must be"):
        encoder(torch.zeros(1, 8, 11, 9))
    with pytest.raises(ValueError, match="channel_validity"):
        encoder(torch.zeros(1, 10, 11, 9), torch.ones(1, 8))
    invalid = torch.ones(1, 10)
    invalid[0, 0] = 2
    with pytest.raises(ValueError, match=r"in \[0,1\]"):
        encoder(torch.zeros(1, 10, 11, 9), invalid)
