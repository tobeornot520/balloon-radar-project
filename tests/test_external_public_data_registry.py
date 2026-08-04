from __future__ import annotations

import csv
import re
from pathlib import Path
from urllib.parse import parse_qs, urlparse


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_REGISTRY = (
    PROJECT_ROOT / "data/metadata/external_public_datasets_v1.csv"
)
ARTIFACT_REGISTRY = (
    PROJECT_ROOT / "data/metadata/external_public_artifacts_v1.csv"
)

SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
FORBIDDEN_QUERY_KEYS = {"access_token", "auth", "expires", "signature", "token"}


def read_registry(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        return list(reader.fieldnames or []), rows


def assert_stable_public_url(value: str) -> None:
    parsed = urlparse(value)
    assert parsed.scheme == "https"
    assert parsed.netloc
    query_keys = {key.lower() for key in parse_qs(parsed.query)}
    assert query_keys.isdisjoint(FORBIDDEN_QUERY_KEYS)


def test_dataset_registry_has_unique_versioned_entries_and_source_boundaries() -> None:
    columns, rows = read_registry(DATASET_REGISTRY)
    required = {
        "dataset_id",
        "release_version",
        "official_name",
        "data_doi",
        "official_landing_url",
        "access_date",
        "license_spdx",
        "redistribution_policy",
        "decision_status",
        "allowed_claim_scope",
        "prohibited_claim_scope",
        "audit_evidence",
    }
    assert required.issubset(columns)
    assert rows

    keys = [(row["dataset_id"], row["release_version"]) for row in rows]
    assert len(keys) == len(set(keys))
    assert all(row["license_spdx"] for row in rows)
    assert all(row["redistribution_policy"] for row in rows)
    assert all(row["allowed_claim_scope"] for row in rows)
    assert all(row["prohibited_claim_scope"] for row in rows)
    assert all(row["audit_evidence"] for row in rows)

    for row in rows:
        assert_stable_public_url(row["official_landing_url"])
        if row["official_file_index_url"]:
            assert_stable_public_url(row["official_file_index_url"])


def test_artifact_registry_references_datasets_and_contains_no_private_paths() -> None:
    _, datasets = read_registry(DATASET_REGISTRY)
    columns, artifacts = read_registry(ARTIFACT_REGISTRY)
    required = {
        "dataset_id",
        "release_version",
        "artifact_id",
        "official_source_url",
        "byte_count",
        "sha256",
        "sha256_scope",
        "integrity_status",
        "local_storage_key",
        "git_policy",
        "license_status",
    }
    assert required.issubset(columns)
    assert artifacts

    dataset_keys = {
        (row["dataset_id"], row["release_version"]) for row in datasets
    }
    artifact_keys = [
        (row["dataset_id"], row["release_version"], row["artifact_id"])
        for row in artifacts
    ]
    assert len(artifact_keys) == len(set(artifact_keys))

    for row in artifacts:
        assert (row["dataset_id"], row["release_version"]) in dataset_keys
        if row["byte_count"]:
            assert int(row["byte_count"]) > 0
        else:
            assert row["integrity_status"].startswith("download_blocked_")
        assert_stable_public_url(row["official_source_url"])
        if row["sha256"]:
            assert SHA256_PATTERN.fullmatch(row["sha256"])
        if row["local_storage_key"]:
            storage_key = Path(row["local_storage_key"])
            assert not storage_key.is_absolute()
            assert ".." not in storage_key.parts
            assert row["local_storage_key"].startswith("data/raw/external/")
        assert "/home/" not in "|".join(row.values())
        assert row["git_policy"]
        assert row["license_status"]
