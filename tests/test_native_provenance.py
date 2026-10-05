"""Bind embedded native provenance to its owning commit, never a newer attempt."""
import hashlib
import json
import math
import tarfile

import h5py
import pytest

from qcl_negf_contracts.messages import ContractError
from qcl_negf_results.commits import artifact_row, json_bytes
from qcl_negf_results.export import export_snapshot
from native_result_fixtures import declare_native
from test_series_commit_identity import historical_row, row_and_commit, save_series


def refresh(commit_path, commit):
    for row in commit["artifacts"]:
        source = commit_path.parent / row["path"]
        row.update(sha256=hashlib.sha256(source.read_bytes()).hexdigest(), bytes=source.stat().st_size)
    commit_path.write_bytes(json_bytes(commit))


def metadata_identity(path, name, identity):
    with h5py.File(path, "r+") as handle:
        handle["metadata"][name] = json.dumps(identity)


@pytest.mark.parametrize("file,name", [("analysis.h5", "identity_json"),
                                       ("recovery.h5", "point_identity_json")])
@pytest.mark.parametrize("field,value", [("point_id", "point-2"), ("execution_id", "other"),
    ("attempt", 2), ("plan_fingerprint", "c" * 64)])
def test_embedded_native_identity_cannot_be_relabelled_by_a_valid_commit(tmp_path, file, name, field, value):
    root, output = tmp_path / "run", tmp_path / "exports"
    plan, series, commit_path, commit = row_and_commit(root)
    embedded = {**commit["identity"], field: value}
    metadata_identity(commit_path.parent / file, name, embedded)
    refresh(commit_path, commit)
    save_series(root, series)
    with pytest.raises(ContractError, match="native.*identity"):
        export_snapshot(root, output, plan=json_bytes(plan), profile="full-state")
    assert not list(output.glob("*.tar.xz"))
    assert not list(output.glob("*.json"))


def test_borrowed_checkpoint_is_checked_against_its_original_commit_attempt(tmp_path):
    root, output = tmp_path / "run", tmp_path / "exports"
    plan, series, commit_path, commit = row_and_commit(root)
    metadata_identity(commit_path.parent / "analysis.h5", "identity_json", commit["identity"])
    metadata_identity(commit_path.parent / "recovery.h5", "point_identity_json", commit["identity"])
    refresh(commit_path, commit)
    historical_row(series)
    save_series(root, series)
    receipt = export_snapshot(root, output, plan=json_bytes(plan), profile="full-state")
    with tarfile.open(output / receipt["archive"], "r:xz") as archive:
        manifest = json.load(archive.extractfile("manifest.json"))
        assert manifest["records"][0]["identity"]["attempt"] == 1
        assert all(row["native_provenance"]["identity"]["status"] == "matched"
                   for row in manifest["records"][0]["included"])


def test_absent_native_identity_and_coordinates_are_explicitly_not_available(tmp_path):
    root, output = tmp_path / "run", tmp_path / "exports"
    plan, series, _, _ = row_and_commit(root)
    save_series(root, series)
    receipt = export_snapshot(root, output, plan=json_bytes(plan))
    with tarfile.open(output / receipt["archive"], "r:xz") as archive:
        manifest = json.load(archive.extractfile("manifest.json"))
        provenance = manifest["records"][0]["included"][0]["native_provenance"]
        assert provenance["identity"]["status"] == "not_available"
        assert all(value["status"] == "not_available" for value in provenance["coordinates"].values())


@pytest.mark.parametrize("temperature", [70.0, 71.0])
def test_explicit_native_kelvin_value_matches_the_frozen_plan(tmp_path, temperature):
    root, output = tmp_path / "run", tmp_path / "exports"
    plan, series, commit_path, commit = row_and_commit(root)
    plan["points"][0]["temperature_K"] = 70.0
    plan["points"][0]["effective_inputs"]["temperature_K"] = 70.0
    series["points"][0]["coordinates"]["temperature_K"] = 70.0
    metadata_identity(commit_path.parent / "analysis.h5", "identity_json", commit["identity"])
    with h5py.File(commit_path.parent / "analysis.h5", "r+") as handle:
        handle["metadata"]["temperature_K"] = temperature
    refresh(commit_path, commit)
    save_series(root, series)
    if temperature != 70.0:
        with pytest.raises(ContractError, match="native.*temperature"):
            export_snapshot(root, output, plan=json_bytes(plan))
    else:
        receipt = export_snapshot(root, output, plan=json_bytes(plan))
        with tarfile.open(output / receipt["archive"], "r:xz") as archive:
            manifest = json.load(archive.extractfile("manifest.json"))
            assert manifest["records"][0]["included"][0]["native_provenance"]["coordinates"]["temperature_K"]["status"] == "matched"


@pytest.mark.parametrize("file,path", [("analysis.h5", "metadata/voltage_per_period_V"),
                                       ("recovery.h5", "inputs/V_period_V")])
@pytest.mark.parametrize("roundtrip", [False, True])
def test_reconstructed_voltage_is_exactly_matched_or_explicitly_not_verified(tmp_path, file, path, roundtrip):
    root, output = tmp_path / "run", tmp_path / "exports"
    plan, series, commit_path, commit = row_and_commit(root)
    expected = 0.056
    observed = math.nextafter(expected, math.inf) if roundtrip else expected
    plan["points"][0]["voltage_per_period_V"] = expected
    plan["points"][0]["effective_inputs"]["voltage_per_period_V"] = expected
    series["points"][0]["coordinates"]["voltage_per_period_V"] = expected
    with h5py.File(commit_path.parent / file, "r+") as handle:
        parent, name = path.split("/")
        handle.require_group(parent)[name] = observed
    refresh(commit_path, commit)
    save_series(root, series)
    receipt = export_snapshot(root, output, plan=json_bytes(plan), profile="full-state")
    with tarfile.open(output / receipt["archive"], "r:xz") as archive:
        manifest = json.load(archive.extractfile("manifest.json"))
        included = next(row for row in manifest["records"][0]["included"] if row["source_path"] == file)
        coordinate = included["native_provenance"]["coordinates"]["voltage_per_period_V"]
        assert coordinate["status"] == ("not_verified" if roundtrip else "matched")
        assert coordinate["observed"] == observed and coordinate["expected"] == expected
        assert coordinate["source"] == "/" + path
        if roundtrip:
            assert "roundtrip" in coordinate["reason"]


@pytest.mark.parametrize("source_attempt", [1, 3])
def test_cumulative_history_keeps_earlier_attempts_and_rejects_future_ones(tmp_path, source_attempt):
    root, output = tmp_path / "run", tmp_path / "exports"
    plan, series, commit_path, commit = row_and_commit(root)
    commit["identity"]["attempt"] = 2
    series["points"][0]["attempt"] = 2
    history = commit_path.parent / "history.h5"
    with h5py.File(history, "w") as handle:
        metadata = declare_native(handle, "qcl-negf-scientific-history-v4", "science.history")
        metadata.attrs["representation"] = "lossless consolidation of every local closed history segment"
        metadata["source_segments_json"] = json.dumps([{"identity": {
            **commit["identity"], "attempt": source_attempt}, "sha256": "a" * 64,
            "scba_rows": 0, "outer_rows": 0}])
    commit["artifacts"].append(artifact_row(history, relative="history.h5", role="science.history",
        media_type="application/x-hdf5", schema="qcl-negf-scientific-history-v4"))
    refresh(commit_path, commit)
    save_series(root, series)
    if source_attempt > 2:
        with pytest.raises(ContractError, match="native.*attempt"):
            export_snapshot(root, output, plan=json_bytes(plan), profile="full-state")
    else:
        receipt = export_snapshot(root, output, plan=json_bytes(plan), profile="full-state")
        with tarfile.open(output / receipt["archive"], "r:xz") as archive:
            manifest = json.load(archive.extractfile("manifest.json"))
            included = next(row for row in manifest["records"][0]["included"] if row["role"] == "science.history")
            assert included["native_provenance"]["history_segments"]["status"] == "matched"
