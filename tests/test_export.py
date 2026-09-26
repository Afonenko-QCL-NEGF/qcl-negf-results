from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tarfile

import h5py
import numpy as np
import pytest

from qcl_negf_contracts.artifacts import COMMIT_SCHEMA
from qcl_negf_contracts.messages import ContractError
from qcl_negf_results.commits import artifact_row, atomic_write, json_bytes, publish_pointer
from qcl_negf_results.export import export_snapshot
import qcl_negf_results.export as exporter
from native_result_fixtures import declare_native


def fixture(root: Path, *, status: str = "unconverged", data: bytes | None = None) -> Path:
    generation = root / "point-1" / "artifacts" / "generation-000001"
    generation.mkdir(parents=True)
    path = generation / "analysis.h5"
    if data is None:
        with h5py.File(path, "w") as handle:
            declare_native(handle, "qcl-negf-physics-analysis-v4", "physics.analysis")
            ds = handle.create_dataset("density", data=np.array([1.0, -0.0, np.nan, np.inf]))
            ds.attrs["units"] = "m^-3"
            ds.attrs["axes"] = "z"
    else:
        path.write_bytes(data)
    with h5py.File(generation / "recovery.h5", "w") as handle:
        declare_native(handle, "qcl-negf-checkpoint-v4", "recovery")
        handle.create_dataset("anderson_history", data=np.arange(120, dtype=np.float64))
    rows = [artifact_row(path, relative="analysis.h5", role="physics.analysis", schema="qcl-negf-physics-analysis-v4", media_type="application/x-hdf5"),
        artifact_row(generation / "recovery.h5", relative="recovery.h5", role="recovery", schema="qcl-negf-checkpoint-v4",
                     media_type="application/x-hdf5", profile="full-state")]
    commit = {"schema": COMMIT_SCHEMA, "contract_set": "qcl-negf.results.v1", "identity": {"point_id": "point-1", "execution_id": "e-1"},
        "generation": 1, "scientific_accepted": False, "terminal_status": status,
        "quality": "unconverged", "artifacts": rows}
    payload = json_bytes(commit)
    atomic_write(generation / "commit.json", payload, immutable=True)
    publish_pointer(generation.parent / "current.json", "generation-000001/commit.json", payload, 1)
    series = {"schema": "qcl-negf-series-result-v3", "contract_set": "qcl-negf.results.v1", "points": [{"id": "point-1", "status": status,
        "data": {"result_commit": "point-1/artifacts/generation-000001/commit.json"}}]}
    atomic_write(root / "series_result.json", json_bytes(series))
    return generation


def unpack(receipt: dict, destination: Path) -> tuple[dict, dict[str, bytes]]:
    with tarfile.open(destination / (receipt["sha256"] + ".tar.xz"), "r:xz") as archive:
        members = {item.name: archive.extractfile(item).read() for item in archive if item.isfile()}
    return json.loads(members["manifest.json"]), members


def test_running_science_pins_complete_native_hdf5_without_recovery(tmp_path: Path) -> None:
    root, output = tmp_path / "run", tmp_path / "exports"
    generation = fixture(root)
    receipt = export_snapshot(root, output)
    manifest, members = unpack(receipt, output)
    assert receipt["job_complete"] is False and receipt["snapshot_consistent"] is True
    assert manifest["records"][0]["scientific_accepted"] is False
    included = manifest["records"][0]["included"]
    assert [item["role"] for item in included] == ["physics.analysis"]
    assert members[included[0]["object"]] == (generation / "analysis.h5").read_bytes()
    assert manifest["records"][0]["omitted_by_policy"][0]["role"] == "recovery"
    assert receipt["bytes"] <= 200_000_000


def test_full_state_includes_recovery(tmp_path: Path) -> None:
    root, output = tmp_path / "run", tmp_path / "exports"
    fixture(root)
    receipt = export_snapshot(root, output, profile="full-state", job_status="completed_with_warnings")
    manifest, _ = unpack(receipt, output)
    assert {item["role"] for item in manifest["records"][0]["included"]} == {"physics.analysis", "recovery"}
    assert manifest["complete"] is True


@pytest.mark.parametrize("profile,roles", [
    ("science", ["physics.analysis"]),
    ("full-state", ["physics.analysis", "recovery"]),
])
def test_preview_roles_follow_profile_without_reading_payloads(tmp_path: Path,
        profile: str, roles: list[str]) -> None:
    job = "1234567890abcdef"
    root = tmp_path / "runs" / job
    generation = fixture(root)
    # Preview is metadata-only. Actual export remains responsible for verifying
    # the payload even when a role is available in the pinned descriptor.
    (generation / "analysis.h5").unlink()
    value = exporter.preview(root, profile=profile, job_id=job)
    assert value["selected_roles"] == roles
    assert value["id"] == job and value["complete"] is False
    with pytest.raises(ContractError, match="disappeared"):
        exporter.export_snapshot(root, tmp_path / "exports", profile=profile)


def test_analysis_clock_comes_from_commit_and_unknown_is_not_zero(tmp_path: Path) -> None:
    root, output = tmp_path / "run", tmp_path / "exports"
    generation = fixture(root)
    first = export_snapshot(root, output)
    assert first["freshness"]["records"][0]["analysis_unix"] is None
    value = json.loads((generation / "commit.json").read_bytes())
    value.update(published_unix=100.0, analysis_coordinates={"attempt": 1,
        "last_completed_outer": None, "last_inner": None})
    (generation / "commit.json").write_bytes(json_bytes(value))
    receipt = export_snapshot(root, output)
    frame = receipt["freshness"]["records"][0]
    assert frame["analysis_unix"] == 100.0
    assert frame["analysis_coordinates"]["last_inner"] is None
    assert frame["analysis_age_seconds"] == receipt["captured_unix"] - 100.0




def test_archive_reuses_identical_committed_generation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root, output = tmp_path / "run", tmp_path / "exports"
    fixture(root)
    first = export_snapshot(root, output)
    monkeypatch.setattr(exporter, "_capture", lambda *_: pytest.fail("unchanged export must reuse validated receipt"))
    assert export_snapshot(root, output) == first


def test_progress_reports_frozen_prefix_without_reading_live_recovery(tmp_path: Path) -> None:
    root, output = tmp_path / "run", tmp_path / "exports"
    generation = fixture(root)
    # A science snapshot must not spend time reading or validating recovery bytes.
    (generation / "recovery.h5").write_bytes(b"not selected by science profile")
    updates = []
    advanced = False

    def progress(value: dict) -> None:
        nonlocal advanced
        updates.append(value)
        if value["phase"] == "capturing" and not advanced:
            # The producer advances after collection; this export keeps its
            # previously captured committed index instead of chasing live state.
            advanced = True
            new_generation = generation.parent / "generation-000002"
            new_generation.mkdir()
            new_analysis = new_generation / "analysis.h5"
            with h5py.File(new_analysis, "w") as handle:
                declare_native(handle, "qcl-negf-physics-analysis-v4", "physics.analysis")
                handle.create_dataset("density", data=np.array([99.0]))
            new_commit = json.loads((generation / "commit.json").read_bytes())
            new_commit.update(generation=2, artifacts=[artifact_row(new_analysis,
                relative="analysis.h5", role="physics.analysis", schema="qcl-negf-physics-analysis-v4", media_type="application/x-hdf5")])
            commit_payload = json_bytes(new_commit)
            atomic_write(new_generation / "commit.json", commit_payload, immutable=True)
            publish_pointer(generation.parent / "current.json", "generation-000002/commit.json", commit_payload, 2)
            atomic_write(root / "series_result.json", json_bytes({
                "schema": "qcl-negf-series-result-v3", "contract_set": "qcl-negf.results.v1", "points": [{"id": "point-1", "status": "running",
                "data": {"result_commit": "point-1/artifacts/generation-000002/commit.json"}}]}))

    receipt = export_snapshot(root, output, progress=progress)
    manifest, members = unpack(receipt, output)
    assert len(manifest["coverage"]) == 1
    assert advanced and manifest["records"][0]["generation"] == 1
    source = manifest["records"][0]["included"][0]["object"]
    assert members[source] == (generation / "analysis.h5").read_bytes()
    phases = list(dict.fromkeys(value["phase"] for value in updates))
    assert phases == ["collecting", "capturing", "validating_closure", "compressing", "verifying", "publishing", "completed"]
    for phase in ("capturing", "compressing", "verifying", "completed"):
        last = [value for value in updates if value["phase"] == phase][-1]
        assert 0 < last["completed_bytes"] == last["total_bytes"]
    assert receipt["export_seconds"] >= receipt["compression_seconds"] >= 0
    assert receipt["captured_unix"] > 0 and receipt["committed_records"] == 1


def test_truncated_json_rejected_even_with_matching_hash(tmp_path: Path) -> None:
    root, output = tmp_path / "run", tmp_path / "exports"
    generation = fixture(root)
    (generation / "metadata.json").write_bytes(b'{"status":"running"')
    value = json.loads((generation / "commit.json").read_bytes())
    value["artifacts"].append(artifact_row(generation / "metadata.json", relative="metadata.json",
                                          role="model", schema="qcl-negf-resolved-configuration-v3",
                                          media_type="application/json"))
    (generation / "commit.json").write_bytes(json_bytes(value))
    with pytest.raises(ContractError, match="invalid complete UTF-8 JSON"):
        export_snapshot(root, output)
    assert not list(output.glob("*.tar.xz"))


def test_hdf5_parser_rejects_fake_extension(tmp_path: Path) -> None:
    root, output = tmp_path / "run", tmp_path / "exports"
    fixture(root, data=b"Not HDF5 despite matching byte count and SHA256")
    with pytest.raises(ContractError, match="invalid HDF5"):
        export_snapshot(root, output)
    assert not list(output.glob("*.tar.xz"))


def test_atomic_replacement_cannot_create_old_length_truncation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root, output = tmp_path / "run", tmp_path / "exports"
    generation = fixture(root)
    original = exporter._capture
    def replace_before_open(source: Path, destination: Path, artifact: object, on_bytes=None) -> None:
        replacement = source.with_suffix(".replacement")
        replacement.write_bytes(source.read_bytes() + b"expanded new generation")
        replacement.replace(source)
        original(source, destination, artifact, on_bytes)
    monkeypatch.setattr(exporter, "_capture", replace_before_open)
    with pytest.raises(ContractError, match="grew beyond committed"):
        export_snapshot(root, output)
    assert (generation / "analysis.h5").exists()
    assert not list(output.glob("*.tar.xz"))


def test_running_series_advancement_does_not_mix_generations(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root, output = tmp_path / "run", tmp_path / "exports"
    fixture(root)
    original = exporter._capture
    def advance_series(source: Path, destination: Path, artifact: object, on_bytes=None) -> None:
        atomic_write(root / "series_result.json", json_bytes({"schema": "qcl-negf-series-result-v3", "contract_set": "qcl-negf.results.v1", "points": [{"id": "point-2", "data": {}}]}))
        original(source, destination, artifact, on_bytes)
    monkeypatch.setattr(exporter, "_capture", advance_series)
    receipt = export_snapshot(root, output)
    manifest, _ = unpack(receipt, output)
    assert [point["id"] for point in manifest["coverage"]] == ["point-1"]
    assert manifest["snapshot_consistent"] is True


def test_mandatory_payload_is_not_trimmed_to_fit(tmp_path: Path) -> None:
    root, output = tmp_path / "run", tmp_path / "exports"
    generation = fixture(root)
    before = (generation / "analysis.h5").read_bytes()
    with pytest.raises(ContractError, match="metadata alone exceeds"):
        export_snapshot(root, output, maximum_bytes=16)
    assert (generation / "analysis.h5").read_bytes() == before
    assert not list(output.glob("*.tar.xz"))


def test_actual_compressed_cap_is_enforced(tmp_path: Path) -> None:
    root, output = tmp_path / "run", tmp_path / "exports"
    fixture(root)
    with pytest.raises(ContractError, match="metadata alone exceeds"):
        export_snapshot(root, output, maximum_bytes=100)
    assert not list(output.glob("*.tar.xz"))


def test_required_dependency_omission_is_rejected(tmp_path: Path) -> None:
    root, output = tmp_path / "run", tmp_path / "exports"
    generation = fixture(root)
    value = json.loads((generation / "commit.json").read_bytes())
    value["artifacts"][0]["dependencies"] = ["recovery.h5"]
    (generation / "commit.json").write_bytes(json_bytes(value))
    with pytest.raises(ContractError, match="omits a dependency"):
        export_snapshot(root, output)


def test_symlink_and_external_hdf5_link_are_rejected(tmp_path: Path) -> None:
    root, output = tmp_path / "run", tmp_path / "exports"
    generation = fixture(root)
    outside = tmp_path / "outside.h5"
    (generation / "analysis.h5").rename(outside)
    (generation / "analysis.h5").symlink_to(outside)
    with pytest.raises(ContractError, match="symbolic link"):
        export_snapshot(root, output)
    (generation / "analysis.h5").unlink()
    with h5py.File(generation / "analysis.h5", "w") as handle:
        declare_native(handle, "qcl-negf-physics-analysis-v4", "physics.analysis")
        handle["external"] = h5py.ExternalLink(str(outside), "/density")
    value = json.loads((generation / "commit.json").read_bytes())
    value["artifacts"][0] = artifact_row(generation / "analysis.h5", relative="analysis.h5",
                                         role="physics.analysis", schema="qcl-negf-physics-analysis-v4", media_type="application/x-hdf5")
    (generation / "commit.json").write_bytes(json_bytes(value))
    with pytest.raises(ContractError, match="external HDF5"):
        export_snapshot(root, output)


def test_changed_committed_bytes_fail_hash_verification(tmp_path: Path) -> None:
    root, output = tmp_path / "run", tmp_path / "exports"
    generation = fixture(root)
    payload = bytearray((generation / "analysis.h5").read_bytes())
    payload[-1] ^= 1
    (generation / "analysis.h5").write_bytes(payload)
    with pytest.raises(ContractError, match="do not match immutable commit"):
        export_snapshot(root, output)


def test_legacy_profile_is_not_silently_accepted(tmp_path: Path) -> None:
    with pytest.raises(ContractError, match="science or full-state"):
        export_snapshot(tmp_path, tmp_path / "exports", profile="compact")


def test_recovery_generation_keeps_its_previous_native_analysis(tmp_path: Path) -> None:
    root, output = tmp_path / "run", tmp_path / "exports"
    original = fixture(root)
    recovery = original.parent / "generation-000002"
    recovery.mkdir()
    value = json.loads((original / "commit.json").read_bytes())
    parent_sha = hashlib.sha256((original / "commit.json").read_bytes()).hexdigest()
    (recovery / "recovery.h5").write_bytes((original / "recovery.h5").read_bytes())
    value["generation"] = 2
    value["artifacts"] = [item for item in value["artifacts"] if item["role"] == "recovery"]
    value["science_parent_commit"] = {"path": "generation-000001/commit.json", "sha256": parent_sha}
    atomic_write(recovery / "commit.json", json_bytes(value), immutable=True)
    atomic_write(root / "series_result.json", json_bytes({"schema": "qcl-negf-series-result-v3", "contract_set": "qcl-negf.results.v1", "points": [{"id": "point-1", "status": "paused",
        "data": {"result_commit": "point-1/artifacts/generation-000002/commit.json"}}]}))
    manifest, _ = unpack(export_snapshot(root, output), output)
    assert len(manifest["records"]) == 2
    assert manifest["records"][0]["science_parent_sha256"] == parent_sha
    assert manifest["records"][1]["included"][0]["role"] == "physics.analysis"


def test_complete_queue_without_physical_record_is_not_complete_export(tmp_path: Path) -> None:
    root, output = tmp_path / "run", tmp_path / "exports"
    root.mkdir()
    receipt = export_snapshot(root, output, job_status="completed")
    manifest, _ = unpack(receipt, output)
    assert receipt["job_complete"] is True
    assert receipt["complete"] is False
    assert manifest["missing_records"] == [{"availability": "no_committed_scientific_records"}]


def test_compressed_cap_is_checked_before_writing_and_includes_footer(tmp_path: Path) -> None:
    from qcl_negf_results.multipart import _write_page, _PartTooLarge
    whole = tmp_path / "whole.tar.xz"
    size = _write_page(whole, {"hello.txt": b"hello"}, [], 10_000, 1)
    target = tmp_path / "bounded.tar.xz"
    with pytest.raises(_PartTooLarge):
        _write_page(target, {"hello.txt": b"hello"}, [], size - 1, 1)
    assert target.stat().st_size <= size - 1
    assert _write_page(target, {"hello.txt": b"hello"}, [], size, 1) == size
    assert target.read_bytes() == whole.read_bytes()


def test_empty_chunked_diagnostic_dataset_is_a_valid_native_state(tmp_path: Path) -> None:
    root, output = tmp_path / "run", tmp_path / "exports"
    generation = fixture(root)
    with h5py.File(generation / "analysis.h5", "a") as handle:
        handle.create_dataset("history", shape=(0, 12), maxshape=(None, 12), chunks=(1, 12), dtype="f8")
    value = json.loads((generation / "commit.json").read_bytes())
    value["artifacts"][0] = artifact_row(generation / "analysis.h5", relative="analysis.h5",
        role="physics.analysis", schema="qcl-negf-physics-analysis-v4", media_type="application/x-hdf5")
    (generation / "commit.json").write_bytes(json_bytes(value))
    assert export_snapshot(root, output)["snapshot_consistent"] is True


def test_campaign_commit_pins_series_instead_of_live_projection(tmp_path: Path) -> None:
    root, output = tmp_path / "run", tmp_path / "exports"
    fixture(root)
    generation = root / "control" / "generations" / "000001"
    generation.mkdir(parents=True)
    (generation / "series_result.json").write_bytes((root / "series_result.json").read_bytes())
    atomic_write(generation / "control-state.json", json_bytes({"workers": [], "status": "running"}), immutable=True)
    series = artifact_row(generation / "series_result.json", relative="series_result.json", role="control.series",
                          media_type="application/json", profile="local")
    control = artifact_row(generation / "control-state.json", relative="control-state.json", role="control.state",
                           media_type="application/json", profile="full-state")
    payload = json_bytes({"schema": COMMIT_SCHEMA, "contract_set": "qcl-negf.results.v1", "identity": {"campaign_id": "test"},
                          "generation": 1, "artifacts": [series, control]})
    atomic_write(generation / "commit.json", payload, immutable=True)
    publish_pointer(root / "current-commit.json", "control/generations/000001/commit.json", payload, 1)
    atomic_write(root / "series_result.json", json_bytes({"schema": "qcl-negf-series-result-v3", "contract_set": "qcl-negf.results.v1",
                                                        "points": [{"id": "new-uncommitted-point", "data": {}}]}))
    receipt = export_snapshot(root, output)
    manifest, _ = unpack(receipt, output)
    assert [item["id"] for item in manifest["coverage"]] == ["point-1"]
    assert "control.series" in {item["role"] for record in manifest["records"] for item in record["omitted_by_policy"]}
    assert "control.state" not in {item["role"] for record in manifest["records"] for item in record["included"]}


def test_previous_series_format_is_explicitly_rejected(tmp_path: Path) -> None:
    root, output = tmp_path / "run", tmp_path / "exports"
    root.mkdir()
    atomic_write(root / "series_result.json", json_bytes({"schema": "qcl-negf-series-result-v1", "points": []}))
    with pytest.raises(ContractError, match="unsupported result contract set"):
        export_snapshot(root, output)


def cumulative_fixture(root: Path, *, corrupt_prefix: bool = False, dependent: bool = False,
                       foreign_plan: bool = False) -> list[Path]:
    """Three generations, with full history spanning two numerical attempts."""
    first = fixture(root)
    original = json.loads((first / "commit.json").read_bytes())
    identity = {**original["identity"], "attempt": 2, "plan_fingerprint": "a" * 64}
    sources = []
    generations = []
    previous_payload = None
    for number in range(1, 4):
        generation = first.parent / f"generation-{number:06d}"
        generation.mkdir(exist_ok=True)
        source = {"sha256": hashlib.sha256(f"closed-segment-{number}".encode()).hexdigest(),
            "identity": {**identity, "attempt": 1 if number == 1 else 2},
            "domain_identity": "49/5/48", "scba_rows": 2, "outer_rows": 1}
        sources.append(source)
        with h5py.File(generation / "history.h5", "w") as handle:
            metadata = handle.create_group("metadata")
            metadata.attrs.update(schema="qcl-negf-scientific-history-v4", artifact_role="science.history",
                representation="lossless consolidation of every local closed history segment",
                scba_rows=number * 2, outer_rows=number)
            metadata.create_dataset("source_segments_json", data=json.dumps(sources))
            for table, count in (("scba", number * 2), ("outer", number)):
                group = handle.create_group(table)
                group.create_dataset("sequence", data=np.arange(1, count + 1, dtype=np.int64))
                group.create_dataset("iteration", data=np.arange(count, dtype=np.int64))
                values = np.arange(count, dtype=np.float64)
                values[0] = -0.0
                if count > 1:
                    values[1] = np.nan
                if corrupt_prefix and number == 3:
                    values[0] = 0.0  # Signed-zero loss must not pass numerical equality.
                dataset = group.create_dataset("J", data=values)
                dataset.attrs.update(units="A/m^2", logical_shape=str(count))
            declare_native(handle, "qcl-negf-scientific-history-v4", "science.history", scba_rows=number * 2)
        artifacts = [artifact_row(generation / "history.h5", relative="history.h5",
            role="science.history", schema="qcl-negf-scientific-history-v4", media_type="application/x-hdf5")]
        if number < 3:
            with h5py.File(generation / "analysis.h5", "w") as handle:
                declare_native(handle, "qcl-negf-physics-analysis-v4", "physics.analysis")
                handle.create_dataset("density", data=np.array([number], dtype=np.float64))
            physical = artifact_row(generation / "analysis.h5", relative="analysis.h5",
                role="physics.analysis", schema="qcl-negf-physics-analysis-v4", media_type="application/x-hdf5")
            if dependent:
                physical["dependencies"] = ["history.h5"]
            artifacts.append(physical)
        value = {"schema": COMMIT_SCHEMA, "contract_set": "qcl-negf.results.v1", "identity": identity, "generation": number,
            "terminal_status": "running", "scientific_accepted": False, "artifacts": artifacts}
        if foreign_plan and number == 1:
            value["identity"] = {**identity, "plan_fingerprint": "b" * 64}
        if previous_payload is not None:
            value["science_parent_commit"] = {"path": f"generation-{number - 1:06d}/commit.json",
                "sha256": hashlib.sha256(previous_payload).hexdigest()}
        previous_payload = json_bytes(value)
        (generation / "commit.json").write_bytes(previous_payload)
        generations.append(generation)
    atomic_write(root / "series_result.json", json_bytes({"schema": "qcl-negf-series-result-v3", "contract_set": "qcl-negf.results.v1", "points": [{
        "id": "point-1", "status": "running", "data": {"result_commit": "point-1/artifacts/generation-000003/commit.json"}}]}))
    return generations


def test_cumulative_science_history_keeps_all_attempts_once_and_parent_physics(tmp_path: Path) -> None:
    root, output = tmp_path / "run", tmp_path / "exports"
    generations = cumulative_fixture(root)
    manifest, members = unpack(export_snapshot(root, output), output)
    records = manifest["records"]
    histories = [item for record in records for item in record["included"] if item["role"] == "science.history"]
    assert len(histories) == 1
    assert members[histories[0]["object"]] == (generations[-1] / "history.h5").read_bytes()
    physics = [item for record in records for item in record["included"] if item["role"] == "physics.analysis"]
    assert len(physics) == 2
    assert {members[item["object"]] for item in physics} == {(path / "analysis.h5").read_bytes() for path in generations[:2]}
    omissions = [item for record in records for item in record["omitted_by_policy"] if item["role"] == "science.history"]
    assert len(omissions) == 2
    assert all(item["reason"] == "superseded_by_verified_cumulative_history" for item in omissions)
    assert all(item["replacement_sha256"] == histories[0]["sha256"] for item in omissions)
    assert sorted(item["replacement_verification"]["previous_rows"]["scba"] for item in omissions) == [2, 4]
    assert all(item["replacement_verification"]["replacement_rows"] == {"scba": 6, "outer": 3} for item in omissions)
    import io
    with h5py.File(io.BytesIO(members[histories[0]["object"]]), "r") as handle:
        assert handle["scba/sequence"][:].tolist() == list(range(1, 7))
        assert handle["outer/sequence"][:].tolist() == [1, 2, 3]
        assert {row["identity"]["attempt"] for row in json.loads(handle["metadata/source_segments_json"][()])} == {1, 2}
    full_manifest, _ = unpack(export_snapshot(root, output, profile="full-state"), output)
    assert sum(item["role"] == "science.history" for record in full_manifest["records"] for item in record["included"]) == 3


def test_cumulative_history_must_preserve_exact_previous_numerical_rows(tmp_path: Path) -> None:
    root, output = tmp_path / "run", tmp_path / "exports"
    cumulative_fixture(root, corrupt_prefix=True)
    with pytest.raises(ContractError, match="changes or omits earlier numerical rows"):
        export_snapshot(root, output)
    assert not list(output.glob("*.tar.xz"))


def test_cumulative_history_cannot_cross_frozen_plan_identity(tmp_path: Path) -> None:
    root, output = tmp_path / "run", tmp_path / "exports"
    cumulative_fixture(root, foreign_plan=True)
    with pytest.raises(ContractError, match="crosses scientific point provenance"):
        export_snapshot(root, output)
    assert not list(output.glob("*.tar.xz"))


def test_exact_parent_history_dependency_is_retained(tmp_path: Path) -> None:
    root, output = tmp_path / "run", tmp_path / "exports"
    cumulative_fixture(root, dependent=True)
    manifest, _ = unpack(export_snapshot(root, output), output)
    assert sum(item["role"] == "science.history" for record in manifest["records"] for item in record["included"]) == 3
    for record in manifest["records"]:
        for item in record["included"]:
            assert all(dependency in {row["path"] for row in manifest["files"]} for dependency in item["dependencies"])


def test_cumulative_history_keeps_compressed_limit_after_lossless_dedup(tmp_path: Path) -> None:
    root, output = tmp_path / "run", tmp_path / "exports"
    cumulative_fixture(root)
    with pytest.raises(ContractError, match="metadata alone exceeds"):
        export_snapshot(root, output, maximum_bytes=16)
    assert not list(output.iterdir())


def test_cumulative_history_cannot_hide_an_earlier_attempt_provenance(tmp_path: Path) -> None:
    root, output = tmp_path / "run", tmp_path / "exports"
    generations = cumulative_fixture(root)
    newest = generations[-1]
    with h5py.File(newest / "history.h5", "r+") as handle:
        sources = json.loads(handle["metadata/source_segments_json"][()])
        sources.pop(0)  # Keep numerical arrays, but incorrectly erase attempt-1 provenance.
        sources[0]["scba_rows"] += 2
        sources[0]["outer_rows"] += 1
        del handle["metadata/source_segments_json"]
        handle["metadata"].create_dataset("source_segments_json", data=json.dumps(sources))
    commit = json.loads((newest / "commit.json").read_bytes())
    commit["artifacts"][0] = artifact_row(newest / "history.h5", relative="history.h5",
        role="science.history", schema="qcl-negf-scientific-history-v4", media_type="application/x-hdf5")
    (newest / "commit.json").write_bytes(json_bytes(commit))
    with pytest.raises(ContractError, match="omits or changes an earlier source segment"):
        export_snapshot(root, output)
    assert not list(output.glob("*.tar.xz"))


def test_omitted_history_still_requires_matching_committed_bytes(tmp_path: Path) -> None:
    root, output = tmp_path / "run", tmp_path / "exports"
    generations = cumulative_fixture(root)
    with (generations[0] / "history.h5").open("ab") as stream:
        stream.write(b"not the committed history")
    with pytest.raises(ContractError, match="grew beyond committed"):
        export_snapshot(root, output)
    assert not list(output.glob("*.tar.xz"))


def test_verified_history_reports_raw_size_without_using_it_for_admission(tmp_path: Path) -> None:
    root, output = tmp_path / "run", tmp_path / "exports"
    generations = cumulative_fixture(root)
    receipt = export_snapshot(root, output)
    all_history_bytes = sum((path / "history.h5").stat().st_size for path in generations)
    manifest, members = unpack(receipt, output)
    exported_history_bytes = sum(len(members[item["object"]]) for record in manifest["records"]
                                 for item in record["included"] if item["role"] == "science.history")
    assert all_history_bytes > exported_history_bytes
    assert receipt["payload_bytes"] == sum(map(len, members.values()))
    assert receipt["bytes"] < receipt["payload_bytes"]
    assert receipt["size_policy"]["maximum_bytes"] == 200_000_000
    assert receipt["size_policy"]["on_overflow"] == "paginate whole objects; explicit checksummed chunks for oversized objects"


