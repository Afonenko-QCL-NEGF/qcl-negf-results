"""Bounded reads use stored axes, units and integrity, independent of solver."""
import hashlib
import json

import h5py
import numpy as np
import pytest

from qcl_negf_contracts.artifacts import CONTRACT_SET
from qcl_negf_contracts.messages import ContractError
from qcl_negf_results.commits import artifact_row, json_bytes
from qcl_negf_results.state import StateReader, verify_recovery_bundle
from qcl_negf_results.model import canonical_bytes
from native_result_fixtures import declare_native


IDENTITY = {"execution_id": "execution-1", "point_id": "point-1", "attempt": 1,
            "plan_fingerprint": "a" * 64, "state_id": "state-1", "state_sequence": 1}


def bundle(root, identity=IDENTITY, storage_class="recovery", omit_roles=()):
    root.mkdir()
    physics = root / "physics.h5"
    with h5py.File(physics, "w") as handle:
        declare_native(handle, "qcl-negf-physics-v4", "physics.full", scba_rows=0)
        handle["metadata/point_identity_json"] = json.dumps(identity)
        value = handle.create_dataset("state_dimensionless/GR/real",
                                      shape=(100000, 2, 2), dtype="f8", chunks=(10, 2, 2))
        value[5:7, 0, 1] = [3.0, 7.0]
        value.attrs.update(units="dimensionless", logical_axis_order="E,k,a",
                           axis_coordinate_paths_json=json.dumps(
                               ["/grids_dimensionless/energy", "/grids_dimensionless/k",
                                "/grids_dimensionless/state_index"]))
        handle.create_dataset("grids_dimensionless/energy", shape=(100000,), dtype="f8")
        handle["grids_dimensionless/energy"][5:7] = [0.05, 0.06]
        handle["grids_dimensionless/k"] = [0.0, 1.0]
        handle["grids_dimensionless/state_index"] = [1, 2]
        handle.create_dataset("grids_dimensionless/wE", shape=(100000,), dtype="f8")
        handle["grids_dimensionless/wE"][5:7] = [0.1, 0.1]
    commit = {"schema": "qcl-negf.artifact-commit.v2", "contract_set": CONTRACT_SET,
              "generation": 1, "state_id": "state-1", "state_sequence": 1,
              "identity": identity, "storage_class": storage_class,
              "artifacts": [artifact_row(physics, relative="physics.h5", role="physics.full",
                            media_type="application/x-hdf5", profile="full-state",
                            schema="qcl-negf-physics-v4")]}
    if storage_class == "archive":
        history = root / "history.h5"
        with h5py.File(history, "w") as handle:
            declare_native(handle, "qcl-negf-scientific-history-v4", "science.history", scba_rows=0)
            handle["metadata/state_identity_json"] = json.dumps(identity)
        model = root / "resolved_configuration.json"
        configuration = {"fixture_only": True}
        model.write_bytes(json_bytes({"schema": "qcl-negf-resolved-configuration-v3",
            "contract_set": CONTRACT_SET, "configuration": configuration,
            "hash_encoding": "qcl-negf-canonical-bytes-v1",
            "configuration_hash": hashlib.sha256(canonical_bytes(configuration)).hexdigest()}))
        commit["artifacts"].extend([
            artifact_row(history, relative=history.name, role="science.history", media_type="application/x-hdf5",
                         schema="qcl-negf-scientific-history-v4"),
            artifact_row(model, relative=model.name, role="model", media_type="application/json",
                         schema="qcl-negf-resolved-configuration-v3")])
    commit["artifacts"] = [row for row in commit["artifacts"] if row["role"] not in omit_roles]
    payload = json_bytes(commit)
    (root / "commit.json").write_bytes(payload)
    receipt = {"schema": "qcl-negf-recovery-receipt-v1", "status": "verified",
               "commit_sha256": hashlib.sha256(payload).hexdigest(),
               "state_id": "state-1", "state_sequence": 1, "identity": identity,
               "publication_scope": "local_filesystem"}
    (root / "receipt.json").write_bytes(json_bytes(receipt))
    return commit


def attach_progress(root, commit, archive, storage_class="archive", omit_roles=()):
    final = archive / "execution-1/point-0/final"
    final.parent.mkdir(parents=True)
    bundle(final, {**IDENTITY, "point_id": "point-0"}, storage_class=storage_class, omit_roles=omit_roles)
    receipt = json.loads((final / "receipt.json").read_text())
    progress = {"schema": "qcl-negf-execution-progress-v1", "contract_set": CONTRACT_SET,
                "identity": IDENTITY, "execution_id": "execution-1", "active_point_id": "point-1",
                "plan_fingerprint": "a"*64, "completed_points": [{
                    "point": {"id": "point-0", "execution_id": "execution-1", "attempt": 1,
                              "status": "completed", "coordinates": {"temperature_K": 70.0,
                              "voltage_per_period_V": 0.0, "branch": "base", "order": 1},
                              "data": {"result_commit": "archive/execution-1/point-0/final/commit.json"}},
                    "final_commit": "execution-1/point-0/final/commit.json", "receipt": receipt,
                    "files": [{"path": path.name, "bytes": path.stat().st_size,
                               "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                              for path in final.iterdir()]}]}
    progress_path = root / "execution_progress.json"
    progress_path.write_bytes(json_bytes(progress))
    commit["artifacts"].append(artifact_row(progress_path, relative=progress_path.name,
        role="execution.progress", media_type="application/json", profile="full-state",
        schema="qcl-negf-execution-progress-v1"))
    payload = json_bytes(commit)
    (root / "commit.json").write_bytes(payload)
    current = json.loads((root / "receipt.json").read_text())
    current["commit_sha256"] = hashlib.sha256(payload).hexdigest()
    (root / "receipt.json").write_bytes(json_bytes(current))
    return final


def test_state_block_reads_only_requested_hyperslab_and_matching_coordinates(tmp_path, monkeypatch):
    root = tmp_path / "bundle"
    bundle(root)
    original = h5py.Dataset.__getitem__

    def bounded(dataset, key):
        if dataset.name == "/state_dimensionless/GR/real":
            assert key == (slice(5, 7, 1), 0, 1), "full solver state was read"
        return original(dataset, key)

    monkeypatch.setattr(h5py.Dataset, "__getitem__", bounded)
    with StateReader(root / "commit.json", expected_identity=IDENTITY) as reader:
        block = reader.read("state_dimensionless/GR/real", (slice(5, 7), 0, 1), maximum_bytes=128)
    np.testing.assert_array_equal(block.values, [3.0, 7.0])
    np.testing.assert_array_equal(block.coordinates["E"], [0.05, 0.06])
    np.testing.assert_array_equal(block.weights["E"], [0.1, 0.1])
    assert block.units == "dimensionless"
    assert block.axes == ("E", "k", "a")


def test_state_block_rejects_memory_budget_before_reading_payload(tmp_path):
    root = tmp_path / "bundle"
    bundle(root)
    with StateReader(root / "commit.json") as reader, pytest.raises(ContractError, match="budget"):
        reader.read("state_dimensionless/GR/real", (slice(None), slice(None), slice(None)),
                    maximum_bytes=128)


@pytest.mark.parametrize("change", ["identity", "receipt", "payload"])
def test_verified_flag_never_substitutes_for_bundle_integrity(tmp_path, change):
    root = tmp_path / "bundle"
    bundle(root)
    if change == "payload":
        with (root / "physics.h5").open("ab") as stream:
            stream.write(b"corrupt")
    else:
        receipt = json.loads((root / "receipt.json").read_text())
        if change == "identity":
            receipt["identity"] = {**IDENTITY, "attempt": 2}
        else:
            receipt["commit_sha256"] = "b" * 64
        (root / "receipt.json").write_bytes(json_bytes(receipt))
    with pytest.raises(ContractError):
        verify_recovery_bundle(root)


def test_state_reader_rejects_another_attempt_even_with_valid_bytes(tmp_path):
    root = tmp_path / "bundle"
    bundle(root)
    with pytest.raises(ContractError, match="identity"):
        StateReader(root / "commit.json", expected_identity={**IDENTITY, "attempt": 2})


@pytest.mark.parametrize("change", ["mismatch", "missing"])
def test_state_reader_rejects_wrong_native_state_with_valid_checksums(tmp_path, change):
    root = tmp_path / "bundle"
    commit = bundle(root)
    with h5py.File(root / "physics.h5", "r+") as handle:
        if change == "missing":
            del handle["metadata/point_identity_json"]
        else:
            handle["metadata/point_identity_json"][()] = json.dumps({**IDENTITY, "state_id": "other"})
    commit["artifacts"] = [artifact_row(root / "physics.h5", relative="physics.h5",
        role="physics.full", media_type="application/x-hdf5", profile="full-state",
        schema="qcl-negf-physics-v4")]
    payload = json_bytes(commit)
    (root / "commit.json").write_bytes(payload)
    receipt = json.loads((root / "receipt.json").read_text())
    receipt["commit_sha256"] = hashlib.sha256(payload).hexdigest()
    (root / "receipt.json").write_bytes(json_bytes(receipt))
    with pytest.raises(ContractError, match="state identity"):
        StateReader(root / "commit.json")


def test_portable_recovery_requires_verified_prior_final_dependencies(tmp_path):
    root, archive = tmp_path / "bundle", tmp_path / "archive"
    commit = bundle(root)
    final = attach_progress(root, commit, archive)
    with pytest.raises(ContractError, match="prior archive"):
        verify_recovery_bundle(root)
    report = verify_recovery_bundle(root, archive_directory=archive)
    assert report["verified_prior_finals"] == 1
    with (final / "physics.h5").open("ab") as stream:
        stream.write(b"corrupt")
    with pytest.raises(ContractError):
        verify_recovery_bundle(root, archive_directory=archive)


def test_prior_final_cannot_be_a_hash_correct_recovery_checkpoint(tmp_path):
    root, archive = tmp_path / "bundle", tmp_path / "archive"
    commit = bundle(root)
    attach_progress(root, commit, archive, storage_class="recovery")
    with pytest.raises(ContractError, match="archive"):
        verify_recovery_bundle(root, archive_directory=archive)


@pytest.mark.parametrize("role", ["model", "science.history"])
def test_prior_final_requires_resolved_model_and_cumulative_history(tmp_path, role):
    root, archive = tmp_path / "bundle", tmp_path / "archive"
    commit = bundle(root)
    attach_progress(root, commit, archive, omit_roles=(role,))
    with pytest.raises(ContractError, match="archive"):
        verify_recovery_bundle(root, archive_directory=archive)


@pytest.mark.parametrize("storage", ["raw", "link", "virtual"])
def test_state_reader_rejects_unowned_external_or_virtual_values(tmp_path, storage):
    root = tmp_path / "bundle"
    commit = bundle(root)
    raw = tmp_path / "unowned.raw"
    with h5py.File(root / "physics.h5", "r+") as handle:
        target = "state_dimensionless/GR/real"
        attributes = dict(handle[target].attrs)
        del handle[target]
        if storage == "raw":
            dataset = handle.create_dataset(target, shape=(100000, 2, 2), dtype="f8",
                external=[(str(raw), 0, h5py.h5f.UNLIMITED)])
        elif storage == "link":
            source = tmp_path / "external.h5"
            with h5py.File(source, "w") as other:
                other.create_dataset("values", shape=(100000, 2, 2), dtype="f8")
            handle[target] = h5py.ExternalLink(str(source), "/values")
            dataset = handle[target]
        else:
            layout = h5py.VirtualLayout(shape=(100000, 2, 2), dtype="f8")
            layout[:] = h5py.VirtualSource(str(tmp_path / "missing.h5"), "values", shape=(100000, 2, 2))
            dataset = handle.create_virtual_dataset(target, layout)
        dataset.attrs.update(attributes)
    commit["artifacts"] = [artifact_row(root / "physics.h5", relative="physics.h5", role="physics.full",
        media_type="application/x-hdf5", profile="full-state", schema="qcl-negf-physics-v4")]
    payload = json_bytes(commit)
    (root / "commit.json").write_bytes(payload)
    receipt = json.loads((root / "receipt.json").read_text())
    receipt["commit_sha256"] = hashlib.sha256(payload).hexdigest()
    (root / "receipt.json").write_bytes(json_bytes(receipt))
    with pytest.raises(ContractError, match="external|virtual|unowned"):
        with StateReader(root / "commit.json") as reader:
            reader.read("state_dimensionless/GR/real", (slice(5, 7), 0, 1))


# R09: declaration normalization must not eliminate later byte obligations.
def _rewrite_progress(root, edit):
    path = root / "execution_progress.json"
    progress = json.loads(path.read_text())
    edit(progress["completed_points"][0]["files"])
    path.write_bytes(json_bytes(progress))
    commit = json.loads((root / "commit.json").read_text())
    commit["artifacts"][-1] = artifact_row(path, relative=path.name,
        role="execution.progress", media_type="application/json", profile="full-state",
        schema="qcl-negf-execution-progress-v1")
    payload = json_bytes(commit)
    (root / "commit.json").write_bytes(payload)
    receipt = json.loads((root / "receipt.json").read_text())
    receipt["commit_sha256"] = hashlib.sha256(payload).hexdigest()
    (root / "receipt.json").write_bytes(json_bytes(receipt))


def test_prior_declaration_union_retains_both_origins_and_owner_only_paths():
    from qcl_negf_results import state
    from qcl_negf_contracts.artifacts import Artifact
    digest = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    owner = Artifact("physics.h5", "physics.full", 3, digest,
                     "application/x-hdf5", "full-state", (), "qcl-negf-physics-v4")
    only = Artifact("history.h5", "science.history", 3, digest,
                    "application/x-hdf5", "full-state", (), "qcl-negf-scientific-history-v4")
    rows = [{"path": "extra.txt", "bytes": 3, "sha256": digest},
            {"path": "physics.h5", "bytes": 3, "sha256": digest}]
    result = state._prior_declarations(rows, (owner, only), IDENTITY, "point/final/commit.json")
    assert [row["path"] for row in result] == ["extra.txt", "physics.h5", "history.h5"]
    assert [row["origins"] for row in result] == [("index",), ("index", "owner"), ("owner",)]
    assert result[1]["owner"] == owner


@pytest.mark.parametrize("field,value", [("bytes", 4), ("sha256", "f" * 64),
    ("role", "science.history"), ("schema", "wrong"), ("media_type", "text/plain"),
    ("identity", {**IDENTITY, "point_id": "other"})])
def test_prior_conflicts_fail_before_dependency_hash_or_native(tmp_path, monkeypatch, field, value):
    from qcl_negf_results import state
    root, archive = tmp_path / "bundle", tmp_path / "archive"
    final = attach_progress(root, bundle(root), archive)
    def conflict(rows):
        next(row for row in rows if row["path"] == "physics.h5")[field] = value
    _rewrite_progress(root, conflict)
    original = state._verify_stream
    seen = []
    def verify(stream, size, digest):
        if str(final) in str(stream.name):
            seen.append(stream.name)
        return original(stream, size, digest)
    monkeypatch.setattr(state, "_verify_stream", verify)
    with pytest.raises(ContractError, match="prior.*physics.h5.*conflict") as error:
        verify_recovery_bundle(root, archive_directory=archive)
    assert error.value.code == "corrupt_result"
    assert seen == []


@pytest.mark.parametrize("omit_history", [False, True])
def test_prior_hash_order_and_native_closure_survive_normalization(tmp_path, monkeypatch, omit_history):
    from qcl_negf_results import state
    root, archive = tmp_path / "bundle", tmp_path / "archive"
    final = attach_progress(root, bundle(root), archive)
    (final / "extra.txt").write_bytes(b"abc")
    order = ["commit.json", "receipt.json", "physics.h5", "history.h5",
             "resolved_configuration.json", "extra.txt"]
    if omit_history:
        order.remove("history.h5")
    def inventory(rows):
        rows[:] = [{"path": name, "bytes": (final / name).stat().st_size,
                    "sha256": hashlib.sha256((final / name).read_bytes()).hexdigest()} for name in order]
    _rewrite_progress(root, inventory)
    original_hash, original_native = state._verify_stream, state.validate_native_handle
    events = []
    def verify(stream, size, digest):
        if str(final) in str(stream.name):
            events.append(("sha", str(stream.name).split("/")[-1]))
        return original_hash(stream, size, digest)
    def native(handle, role, schema):
        if str(final) in str(handle.filename):
            events.append(("native", role))
        return original_native(handle, role, schema)
    monkeypatch.setattr(state, "_verify_stream", verify)
    monkeypatch.setattr(state, "validate_native_handle", native)
    expected = [("sha", name) for name in order +
                ["physics.h5", "history.h5", "resolved_configuration.json"]]
    expected += [("native", "physics.full"), ("native", "science.history")]
    for _ in range(2):
        events.clear()
        assert verify_recovery_bundle(root, archive_directory=archive)["verified_prior_finals"] == 1
        assert events == expected


def test_prior_later_actual_sha_rejects_same_size_mutation_with_stale_stat(tmp_path, monkeypatch):
    from pathlib import Path
    from qcl_negf_results import state
    root, archive = tmp_path / "bundle", tmp_path / "archive"
    final = attach_progress(root, bundle(root), archive)
    target = final / "physics.h5"
    stale, original_stat, original_hash = target.stat(), Path.stat, state._verify_stream
    calls = []
    def stat(path, *args, **kwargs):
        return stale if path == target else original_stat(path, *args, **kwargs)
    def verify(stream, size, digest):
        if Path(stream.name) == target:
            calls.append("physics-sha")
        result = original_hash(stream, size, digest)
        if Path(stream.name) == target and len(calls) == 1:
            with h5py.File(target, "r+") as handle:
                handle["state_dimensionless/GR/real"][5, 0, 1] = 9.0
            assert original_stat(target).st_size == stale.st_size
            assert target.stat() == stale
            assert hashlib.sha256(target.read_bytes()).hexdigest() != digest
        return result
    monkeypatch.setattr(Path, "stat", stat)
    monkeypatch.setattr(state, "_verify_stream", verify)
    with pytest.raises(ContractError, match="checksum") as error:
        verify_recovery_bundle(root, archive_directory=archive)
    assert error.value.code == "corrupt_result"
    assert calls == ["physics-sha", "physics-sha"]


@pytest.mark.parametrize("change", ["duplicate", "escape", "absolute", "boolean-size"])
def test_prior_invalid_index_rows_keep_contract_rejection(tmp_path, change):
    root, archive = tmp_path / "bundle", tmp_path / "archive"
    attach_progress(root, bundle(root), archive)
    def invalid(rows):
        if change == "duplicate":
            rows.append(dict(rows[0]))
        elif change == "boolean-size":
            rows[0]["bytes"] = True
        else:
            rows[0]["path"] = "../escape" if change == "escape" else "/absolute"
    _rewrite_progress(root, invalid)
    with pytest.raises(ContractError) as error:
        verify_recovery_bundle(root, archive_directory=archive)
    assert error.value.code == "corrupt_result"


@pytest.mark.parametrize("name", ["commit.json", "receipt.json"])
def test_prior_metadata_read_does_not_cache_external_bytes(tmp_path, monkeypatch, name):
    from qcl_negf_results import state
    root, archive = tmp_path / "bundle", tmp_path / "archive"
    final = attach_progress(root, bundle(root), archive)
    original = state._receipt
    def receipt(directory, commit, payload):
        result = original(directory, commit, payload)
        if directory == final:
            with (directory / name).open("ab") as stream:
                stream.write(b" ")
        return result
    monkeypatch.setattr(state, "_receipt", receipt)
    with pytest.raises(ContractError, match="byte length"):
        verify_recovery_bundle(root, archive_directory=archive)


def test_prior_same_names_in_separate_finals_do_not_share_hash_proof(tmp_path, monkeypatch):
    from qcl_negf_results import state
    root, archive = tmp_path / "bundle", tmp_path / "archive"
    first = attach_progress(root, bundle(root), archive)
    second = first.parent.parent / "point-2/final"
    second.parent.mkdir()
    bundle(second, {**IDENTITY, "point_id": "point-2"}, storage_class="archive")
    path = root / "execution_progress.json"
    progress = json.loads(path.read_text())
    entry = json.loads(json.dumps(progress["completed_points"][0]))
    entry["point"]["id"] = "point-2"
    entry["point"]["coordinates"]["order"] = 2
    entry["final_commit"] = "execution-1/point-2/final/commit.json"
    entry["point"]["data"]["result_commit"] = "archive/" + entry["final_commit"]
    entry["receipt"] = json.loads((second / "receipt.json").read_text())
    entry["files"] = [{"path": name, "bytes": (second / name).stat().st_size,
        "sha256": hashlib.sha256((second / name).read_bytes()).hexdigest()}
        for name in ["commit.json", "receipt.json", "physics.h5", "history.h5",
                     "resolved_configuration.json"]]
    progress["completed_points"].append(entry)
    path.write_bytes(json_bytes(progress))
    _rewrite_progress(root, lambda rows: None)
    original = state._verify_stream
    seen = []
    def verify(stream, size, digest):
        if str(first) in str(stream.name) or str(second) in str(stream.name):
            seen.append(str(stream.name))
        return original(stream, size, digest)
    monkeypatch.setattr(state, "_verify_stream", verify)
    assert verify_recovery_bundle(root, archive_directory=archive)["verified_prior_finals"] == 2
    for directory in (first, second):
        for name in ["physics.h5", "history.h5", "resolved_configuration.json"]:
            assert seen.count(str(directory / name)) == 2
        for name in ["commit.json", "receipt.json"]:
            assert seen.count(str(directory / name)) == 1


def _tiny_identity_fixture(tmp_path):
    """Reuse contract fixture metadata but truncate large sparse state datasets."""
    root, archive = tmp_path / "bundle", tmp_path / "archive"
    final = attach_progress(root, bundle(root), archive)
    for directory in (root, final):
        commit = json.loads((directory / "commit.json").read_text())
        with h5py.File(directory / "physics.h5", "w") as handle:
            declare_native(handle, "qcl-negf-physics-v4", "physics.full", scba_rows=0)
            handle["metadata/point_identity_json"] = json.dumps(commit["identity"])
            handle["state_dimensionless/value"] = [3.0]
        commit["artifacts"][0] = artifact_row(directory / "physics.h5", relative="physics.h5",
            role="physics.full", media_type="application/x-hdf5", profile="full-state",
            schema="qcl-negf-physics-v4")
        payload = json_bytes(commit)
        (directory / "commit.json").write_bytes(payload)
        receipt = json.loads((directory / "receipt.json").read_text())
        receipt["commit_sha256"] = hashlib.sha256(payload).hexdigest()
        (directory / "receipt.json").write_bytes(json_bytes(receipt))
    path = root / "execution_progress.json"
    progress = json.loads(path.read_text())
    entry = progress["completed_points"][0]
    entry["receipt"] = json.loads((final / "receipt.json").read_text())
    entry["files"] = [{"path": name, "bytes": (final / name).stat().st_size,
        "sha256": hashlib.sha256((final / name).read_bytes()).hexdigest()}
        for name in ["commit.json", "receipt.json", "physics.h5", "history.h5",
                     "resolved_configuration.json"]]
    path.write_bytes(json_bytes(progress))
    _rewrite_progress(root, lambda rows: None)
    return root, archive, final


@pytest.mark.parametrize("field,value", [("attempt", True), ("state_sequence", True),
                                         ("attempt", 1.0), ("state_sequence", 1.0),
                                         (None, None)])
def test_prior_optional_identity_uses_exact_json_types(tmp_path, monkeypatch, field, value):
    from qcl_negf_results import state
    root, archive, final = _tiny_identity_fixture(tmp_path)
    identity = json.loads((final / "commit.json").read_text())["identity"]
    if field is not None:
        identity[field] = value
    def claim(rows):
        next(row for row in rows if row["path"] == "physics.h5")["identity"] = identity
    _rewrite_progress(root, claim)
    original_hash, original_native = state._verify_stream, state.validate_native_handle
    events = []
    def verify(stream, size, digest):
        if str(final) in str(stream.name):
            events.append(("sha", str(stream.name).split("/")[-1]))
        return original_hash(stream, size, digest)
    def native(handle, role, schema):
        if str(final) in str(handle.filename):
            events.append(("native", role))
        return original_native(handle, role, schema)
    monkeypatch.setattr(state, "_verify_stream", verify)
    monkeypatch.setattr(state, "validate_native_handle", native)
    if field is not None:
        with pytest.raises(ContractError, match="prior.*physics.h5.*identity.*conflict") as error:
            verify_recovery_bundle(root, archive_directory=archive)
        assert error.value.code == "corrupt_result"
        assert events == []
    else:
        assert verify_recovery_bundle(root, archive_directory=archive)["verified_prior_finals"] == 1
        assert events == [("sha", name) for name in ["commit.json", "receipt.json", "physics.h5",
            "history.h5", "resolved_configuration.json", "physics.h5", "history.h5",
            "resolved_configuration.json"]] + [("native", "physics.full"), ("native", "science.history")]
