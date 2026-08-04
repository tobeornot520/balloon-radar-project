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


def test_hsr_v2_and_journal_bundle_are_recorded_as_distinct_non_equivalent_releases() -> None:
    _, datasets = read_registry(DATASET_REGISTRY)
    _, artifacts = read_registry(ARTIFACT_REGISTRY)

    by_version = {
        row["release_version"]: row
        for row in datasets
        if row["dataset_id"] == "lss_hsr_l"
    }
    assert set(by_version) == {"V2", "journal_bundle_20260804"}
    assert by_version["V2"]["official_size_bytes"] == "237020946"
    assert "non_equivalent" in by_version["V2"]["decision_status"]
    assert "not_equivalent" in by_version["journal_bundle_20260804"][
        "decision_status"
    ]
    assert "mixing" in by_version["V2"]["prohibited_claim_scope"]

    v2_artifact = next(
        row
        for row in artifacts
        if row["dataset_id"] == "lss_hsr_l"
        and row["release_version"] == "V2"
    )
    assert v2_artifact["byte_count"] == "237020946"
    assert (
        v2_artifact["sha256"]
        == "fea8a21354110a96fb9644dc1c69649b6dc6d1a1b6da512498d9c2d74d839540"
    )
    assert v2_artifact["integrity_status"] == "official_size_and_zip_test_passed"


def test_dronerfc_registry_distinguishes_full_release_from_selected_radar_subset() -> None:
    _, datasets = read_registry(DATASET_REGISTRY)
    _, artifacts = read_registry(ARTIFACT_REGISTRY)

    dataset = next(row for row in datasets if row["dataset_id"] == "dronerfc_mm")
    assert dataset["release_version"] == "V1"
    assert dataset["data_doi"] == "10.57760/sciencedb.j00173.00094"
    assert dataset["license_spdx"] == "CC-BY-SA-4.0"
    assert dataset["official_file_count"] == "113"
    assert dataset["official_size_bytes"] == "75612067287"
    assert dataset["decision_status"] == "schema_verified_but_blocked_timestamp_alignment_b1"
    assert "ADC or IQ" in dataset["prohibited_claim_scope"]
    assert "random frame/window" in dataset["prohibited_claim_scope"]
    assert "B1 supervised alignment" in dataset["prohibited_claim_scope"]
    assert "scripts/audit_dronerfc_mm_v1.py" in dataset["audit_evidence"]

    artifact = next(
        row for row in artifacts if row["dataset_id"] == "dronerfc_mm"
    )
    assert artifact["artifact_role"] == "selected_modality_subset"
    assert artifact["byte_count"] == "47366902"
    assert (
        artifact["sha256"]
        == "6b0c2ed1a075aa9164a516af001b630a9f775fddc9f399223c1aeeb6e7047b2b"
    )
    assert artifact["sha256_scope"] == "sha256_of_sorted_28_file_sha256_manifest"
    assert "full_pcd_schema" in artifact["integrity_status"]
    assert "B1 has zero radar-GT range overlap" in artifact["notes"]


def test_balloon_signature_registry_keeps_documentation_separate_from_large_archive() -> None:
    _, datasets = read_registry(DATASET_REGISTRY)
    _, artifacts = read_registry(ARTIFACT_REGISTRY)

    dataset = next(
        row for row in datasets if row["dataset_id"] == "radar_signature_foil_balloon"
    )
    assert dataset["data_doi"] == "10.5281/zenodo.7573165"
    assert dataset["license_spdx"] == "CC-BY-4.0"
    assert dataset["official_size_bytes"] == "42402566780"
    assert "full_archive_deferred" in dataset["decision_status"]
    assert "Outdoor airborne-balloon" in dataset["prohibited_claim_scope"]

    by_artifact = {
        row["artifact_id"]: row
        for row in artifacts
        if row["dataset_id"] == "radar_signature_foil_balloon"
    }
    assert set(by_artifact) == {
        "radar_signature_readme",
        "radar_signature_metadata_pickle",
        "radar_signature_isar_archive",
    }
    assert by_artifact["radar_signature_readme"]["byte_count"] == "125546"
    assert (
        by_artifact["radar_signature_metadata_pickle"]["integrity_status"]
        == "size_md5_sha256_passed_static_inspection_only"
    )
    assert (
        by_artifact["radar_signature_isar_archive"]["integrity_status"]
        == "not_downloaded_deferred_large_domain_mismatch"
    )


def test_small_public_smoke_downloads_have_exact_receipts_and_group_boundaries() -> None:
    _, datasets = read_registry(DATASET_REGISTRY)
    _, artifacts = read_registry(ARTIFACT_REGISTRY)
    by_dataset = {row["dataset_id"]: row for row in datasets}
    by_artifact = {row["artifact_id"]: row for row in artifacts}

    swarm = by_dataset["low_altitude_uav_swarm_25500"]
    assert swarm["paper_doi"] == "10.12466/xhcl.2025.05.004"
    assert swarm["data_doi"] == "10.57760/sciencedb.25500"
    assert swarm["license_spdx"] == "CC-BY-NC-4.0"
    assert swarm["official_size_bytes"] == "3996753"
    assert "three physical experiments" in swarm["grouping_risk"]
    assert "275 contiguous radar screens" in swarm["grouping_risk"]
    assert swarm["decision_status"] == "downloaded_integrity_and_schema_verified_smoke_only"
    assert "ADC/IQ" in swarm["prohibited_claim_scope"]
    swarm_zip = by_artifact["uav_swarm_v1_release_zip"]
    assert swarm_zip["byte_count"] == "3996753"
    assert (
        swarm_zip["sha256"]
        == "c08e5c93d59d1012f134a3ffa7521eb4d26fb7cfc7a8bcbc297574273350a76e"
    )
    assert "zip_test" in swarm_zip["integrity_status"]

    nexrad = by_dataset["noaa_nexrad_level2_nodd"]
    assert nexrad["official_landing_url"] == "https://registry.opendata.aws/noaa-nexrad/"
    assert nexrad["license_spdx"] == "NOASSERTION"
    assert "complete volume" in nexrad["independent_unit"]
    assert "target-labeled performance" in nexrad["prohibited_claim_scope"]
    volume = by_artifact["ktlx_20200101_000532_v06"]
    assert volume["byte_count"] == "395379"
    assert (
        volume["sha256"]
        == "e6092212670064ebc4da0e38738b38e9f965425c2f219a512c057693211d5c9b"
    )
    assert volume["official_checksum_algorithm"] == "HTTP ETag"
    assert volume["official_checksum"] == "7de6abafdecce9233a77e75687ab7e79"


def test_large_measured_candidates_are_deferred_without_overclaiming() -> None:
    _, datasets = read_registry(DATASET_REGISTRY)
    by_dataset = {row["dataset_id"]: row for row in datasets}

    s_band = by_dataset["simple_field_sband_uav_18323"]
    assert s_band["data_doi"] == "10.57760/sciencedb.18323"
    assert s_band["official_size_bytes"] == "23458116419"
    assert s_band["decision_status"].startswith("deferred_large")
    assert "13 original acquisition files" in s_band["independent_unit"]
    assert "Tian 2024 exact reproduction" in s_band["prohibited_claim_scope"]

    tri_band = by_dataset["mathworks_standrews_radar_drone_classification"]
    assert tri_band["data_doi"] == "10.5281/zenodo.18553708"
    assert tri_band["official_size_bytes"] == "30358190877"
    assert tri_band["band"] == "24;94;207 GHz"
    assert tri_band["decision_status"].startswith("deferred_large")
    assert "real bird" in tri_band["classes"]
    assert "Random image/window split" in tri_band["prohibited_claim_scope"]
