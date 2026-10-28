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

# Native4 helper constructors above remain for the pending owning consumer
# migrations; supported export tests below use only an explicit native5 source.
from result_v2_fixture import (IDENTITY, source, record, reference, put,
    producer_root, producer_state, copy_constructor_closure, export_source)
from qcl_negf_results.state import StateReader


def test_science_preserves_exact_source_bytes_without_claiming_acceptance(tmp_path):
    root, output = tmp_path / 'source', tmp_path / 'export'
    ref = source(root, [record()])
    receipt = export_source(root, ref, output)
    manifest, members = unpack(receipt, output)
    assert manifest['source_state'] == ref
    assert manifest['scientific_accepted'] is None and receipt['scientific_accepted'] is None
    assert manifest['snapshot_consistent'] is True
    assert manifest['canonical_source_complete'] is False and manifest['complete'] is False
    assert 'raw_scientific_history' in manifest['capabilities']
    assert 'exact_restart' in manifest['requires_full_state']
    for row in manifest['source_locator']:
        assert members[row['object']] == (root / row['reference']['path']).read_bytes()


def test_science_omits_optional_checkpoint_owner_without_reading_it(producer_root, tmp_path):
    ref = producer_state(producer_root, 'exact', recovery=True)
    root = copy_constructor_closure(producer_root, tmp_path / 'source', ref)
    original = json.loads((root / ref['manifest']['path']).read_text())
    # These owners really are optional for the selected raw history. Their
    # absence must neither pull full state into science nor validate restore.
    full = [row for row in original['artifacts'] if row['role'] == 'recovery' and row['media_type'] == 'application/x-hdf5']
    assert full
    for row in full:
        (root / row['path']).unlink()
    receipt = export_source(root, ref, tmp_path / 'export')
    manifest, members = unpack(receipt, tmp_path / 'export')
    assert all(row['role'] != 'recovery' for row in manifest['records'][0]['included'])
    assert any(row['role'] == 'recovery' for row in manifest['omitted_by_policy'])
    assert members[next(row['object'] for row in manifest['source_locator']
        if row['reference'] == ref['manifest'])] == (root / ref['manifest']['path']).read_bytes()
    with pytest.raises(ContractError):
        export_source(root, ref, tmp_path / 'full', profile='full-state')


def test_preview_does_not_read_hdf_payload_and_reports_actual_declared_bytes(tmp_path, monkeypatch):
    root = tmp_path / 'source'
    ref = source(root, [record()])
    (root / 'records.h5').unlink()
    with StateReader(root, state_ref=ref, expected_identity=IDENTITY) as reader:
        monkeypatch.setattr(h5py, 'File', lambda *a, **kw: pytest.fail('preview opened HDF'))
        value = exporter.preview(reader, profile='science')
        assert value['bytes'] > 0 and value['source'] == ref
        assert value['integrity_scope'] == 'declared_metadata_availability'
        with pytest.raises(ContractError):
            exporter.export_snapshot(reader, tmp_path / 'export', profile='science',
                byte_budget=1024*1024, reserve_bytes=0)


@pytest.mark.parametrize('raw', [b'{"schema":', b'{"a":1,"a":2}', b'\xff'])
def test_manifest_json_strictness_even_with_matching_hash(tmp_path, raw):
    root = tmp_path / 'source'
    ref = source(root, [record()])
    (root / 'manifest.json').write_bytes(raw)
    ref['manifest'] = reference(root, 'manifest.json', ref['manifest']['schema'])
    with pytest.raises(ContractError):
        StateReader(root, state_ref=ref, expected_identity=IDENTITY)


def test_changed_closed_bytes_fail_hash_verification_and_do_not_publish(tmp_path):
    root, output = tmp_path / 'source', tmp_path / 'export'
    ref = source(root, [record()])
    path = root / 'records.h5'
    path.write_bytes(path.read_bytes() + b'not-declared')
    with pytest.raises(ContractError):
        export_source(root, ref, output)
    assert not output.exists() or not list(output.iterdir())


def test_implicit_legacy_root_profile_and_status_overrides_are_retired(tmp_path):
    root = tmp_path / 'source'
    ref = source(root, [record()])
    with pytest.raises(TypeError, match='explicit selected StateReader'):
        export_snapshot(root, tmp_path / 'bad', profile='science', byte_budget=1000, reserve_bytes=0)
    with StateReader(root, state_ref=ref, expected_identity=IDENTITY) as reader:
        with pytest.raises(ValueError, match='profile'):
            exporter.preview(reader, profile='legacy')
        with pytest.raises(TypeError):
            export_snapshot(reader, tmp_path / 'bad', profile='science', byte_budget=1000,
                reserve_bytes=0, job_status='completed')
    assert not (tmp_path / 'bad').exists()


def test_history_only_source_has_no_automatic_recovery_substitution(tmp_path):
    root = tmp_path / 'source'
    ref = source(root, [record()])
    with StateReader(root, state_ref=ref, expected_identity=IDENTITY) as reader:
        with pytest.raises(ContractError) as error:
            exporter.preview(reader, profile='full-state')
    assert error.value.code == 'capability_unavailable'


def test_unselected_orphans_do_not_change_snapshot_or_get_read(tmp_path):
    root, output = tmp_path / 'source', tmp_path / 'export'
    ref = source(root, [record()])
    first = export_source(root, ref, output)
    (root / 'series_result.json').write_bytes(b'legacy malformed untrusted orphan')
    (root / 'later.h5').write_bytes(b'not HDF')
    assert export_source(root, ref, output) == first


def test_explicit_parent_history_is_preserved_once(producer_root, tmp_path):
    ref = producer_state(producer_root, 'grandchild')
    receipt = export_source(producer_root, ref, tmp_path / 'export')
    manifest, members = unpack(receipt, tmp_path / 'export')
    paths = [row['reference']['path'] for row in manifest['source_locator']]
    assert len(paths) == len(set(paths))
    assert any(row['role'] == 'parent.checkpoint.proof' for row in manifest['records'][0]['included'])
    assert not set(paths) & {row['source_path'] for row in manifest['omitted_by_policy']}
    assert manifest['coverage']['scope'] == 'explicit_selected_state'
    assert not manifest['canonical_source_complete']
