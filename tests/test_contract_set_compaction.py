from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from qcl_negf_contracts.artifacts import CONTRACT_SET, MODEL_SCHEMA, validate_commit
from qcl_negf_contracts.messages import ContractError
from qcl_negf_contracts.telemetry import TABLE_FIELDS, TELEMETRY_SCHEMA
from qcl_negf_results.compaction import compact_performance, validate_performance_schema
from qcl_negf_results.model import canonical_bytes, validate_model
from test_export import fixture, unpack
from qcl_negf_results.export import export_snapshot
from qcl_negf_results.telemetry import TelemetryWriter


def test_registry_matches_pure_runtime_contracts():
    from qcl_negf_contracts.artifacts import ARTIFACT_SCHEMA_CONTRACTS, NATIVE_SCHEMA_VERSION
    from qcl_negf_contracts import load_schema
    registry = load_schema("results-contract-set.json")
    assert registry["contract_set"] == CONTRACT_SET
    assert registry["native_schema_version"] == NATIVE_SCHEMA_VERSION
    assert registry["previous_contracts_supported"] is False
    assert {(row["role"], row["media_type"]): frozenset(row["schemas"]) for row in registry["artifacts"]} == ARTIFACT_SCHEMA_CONTRACTS


def test_model_envelope_preserves_numeric_types_and_rejects_stale_or_changed_input():
    configuration = {"integer": 1, "float": 1.0, "negative_zero": -0.0, "name": "Φонон"}
    envelope = {"schema": MODEL_SCHEMA, "contract_set": CONTRACT_SET,
                "configuration": configuration, "hash_encoding": "qcl-negf-canonical-bytes-v1",
                "configuration_hash": hashlib.sha256(canonical_bytes(configuration)).hexdigest()}
    assert validate_model(envelope) == configuration
    assert canonical_bytes(1) != canonical_bytes(1.0)
    assert canonical_bytes(-0.0) != canonical_bytes(0.0)
    with pytest.raises(ContractError, match="contract set"):
        validate_model(configuration)
    envelope["configuration"] = {**configuration, "float": 1}
    with pytest.raises(ContractError, match="identity differs"):
        validate_model(envelope)


def test_missing_contract_rejected_before_payload_access(tmp_path):
    generation = fixture(tmp_path / "run")
    commit = json.loads((generation / "commit.json").read_bytes())
    commit.pop("contract_set")
    with pytest.raises(ContractError, match="contract set"):
        validate_commit(commit)


def test_compaction_preserves_order_null_nan_payload_and_signed_zero(tmp_path):
    types = {"string": pa.string(), "int64": pa.int64(), "float64": pa.float64(), "bool": pa.bool_()}
    schema = pa.schema([pa.field(name, types[kind]) for name, kind in TABLE_FIELDS["resources"]],
        metadata={b"schema": TELEMETRY_SCHEMA.encode(), b"contract_set": CONTRACT_SET.encode(), b"table": b"resources"})
    nan = np.array([0x7ff8000000000003], dtype=np.uint64).view(np.float64)[0]
    paths = []
    rows = [{"monotonic_ns": index, "cpu_time_seconds": value, "missing_reason": "пропуск" if value is None else None}
            for index, value in enumerate([0.0, -0.0, nan, None] * 1300)]
    for number, (start, stop) in enumerate(((0, 3), (3, 4098), (4098, 5200))):
        path = tmp_path / f"part-{number}.parquet"
        pq.write_table(pa.Table.from_pylist(rows[start:stop], schema=schema), path)
        paths.append(path)
    proof = compact_performance(paths, tmp_path / "compacted.parquet")
    assert proof["rows"] == 5200 and proof["source_files"] == 3
    assert proof["logical_payload_sha256"] == proof["verified_payload_sha256"]
    values = pq.read_table(tmp_path / "compacted.parquet")["cpu_time_seconds"].to_numpy()
    assert values.view(np.uint64)[1] == 0x8000000000000000
    assert values.view(np.uint64)[2] == 0x7ff8000000000003
    assert pq.read_table(tmp_path / "compacted.parquet")["monotonic_ns"].to_pylist() == list(range(5200))
    with pytest.raises(ContractError, match="contract set"):
        validate_performance_schema(schema.with_metadata({b"schema": b"qcl-negf.performance.v1"}))


def test_export_compaction_preserves_all_registered_resources(tmp_path):
    root = tmp_path / "run"
    fixture(root)
    recorder = TelemetryWriter(root / "performance", {"execution_id": "execution-1"}, {})
    for index in range(4):
        recorder.record_resource({"monotonic_ns": index + 1, "wall_seconds": float(index), "rss_bytes": 100 + index},
                                 science_window=index == 2)
        recorder.flush()
    recorder.close()
    receipt = export_snapshot(root, tmp_path / "export")
    manifest, members = unpack(receipt, tmp_path / "export")
    proofs = [proof for proof in manifest["telemetry_compaction"] if proof["table"] == "resources"]
    assert len(proofs) == 1 and proofs[0]["rows"] == 4 and proofs[0]["source_files"] == 4
    assert proofs[0]["object"] in members
    compacted = pq.read_table(io.BytesIO(members[proofs[0]["object"]]))
    assert compacted["rss_bytes"].to_pylist() == [100, 101, 102, 103]
    assert manifest["contract_set"] == CONTRACT_SET


def test_export_witness_selection_keeps_scalar_rows_and_native_bits(tmp_path):
    import h5py
    from native_result_fixtures import declare_native
    from test_psd_history_proof import _selected
    from qcl_negf_results.witness_selection import select_history_witnesses
    from qcl_negf_results.witnesses import selected_witness_records
    source, derived = tmp_path / "history.h5", tmp_path / "selected.h5"
    with h5py.File(source, "w") as handle:
        declare_native(handle, "qcl-negf-scientific-history-v4", "science.history", scba_rows=20)
        handle["metadata/source_segments_json"] = json.dumps([{
            "identity": {"attempt": 1}, "scba_rows": 20, "scba_sequence_first": 1, "scba_sequence_last": 20}])
        psd = handle["psd_history"]
        del psd["selected_blocks"]
        _selected(psd, 20)
        psd["ratio"] = np.asarray([-1.0] * 4 + [1.0] * 8 + [3.0] + [1.0] * 7)
        psd["available"][:] = 1
    before = source.read_bytes()
    proof = select_history_witnesses(source, derived)
    assert proof["source_witnesses"] == 10 and proof["selected_witnesses"] == 4
    assert proof["selected_sequences"] == [1, 5, 13, 19]
    assert source.read_bytes() == before
    with h5py.File(source, "r") as old, h5py.File(derived, "r") as new:
        assert old["psd_history/ratio"][:].tobytes() == new["psd_history/ratio"][:].tobytes()
        old_records = selected_witness_records(old["psd_history/selected_blocks"])
        new_records = selected_witness_records(new["psd_history/selected_blocks"])
        for sequence, record in new_records.items():
            for name, value in record.arrays.items():
                previous = old_records[sequence].arrays[name]
                assert value.dataset[value.offset:value.offset+value.size].tobytes() == previous.dataset[previous.offset:previous.offset+previous.size].tobytes()


def test_witness_selection_retains_independent_attempt_extrema(tmp_path):
    import h5py
    from native_result_fixtures import declare_native
    from test_psd_history_proof import _selected
    from qcl_negf_results.witness_selection import select_history_witnesses
    source, derived = tmp_path / "attempts.h5", tmp_path / "selected.h5"
    with h5py.File(source, "w") as handle:
        declare_native(handle, "qcl-negf-scientific-history-v4", "science.history", scba_rows=40)
        # Source inventory order does not establish row ownership: immutable
        # source coordinates do, even after cumulative rows are sorted.
        handle["metadata/source_segments_json"] = json.dumps([
            {"identity": {"attempt": 2}, "scba_rows": 20, "scba_sequence_first": 21, "scba_sequence_last": 40},
            {"identity": {"attempt": 1}, "scba_rows": 20, "scba_sequence_first": 1, "scba_sequence_last": 20}])
        psd = handle["psd_history"]
        del psd["selected_blocks"]
        _selected(psd, 40)
        psd["ratio"] = np.asarray([-1.0] * 4 + [1.0] * 8 + [30.0] + [1.0] * 7
                                  + [-1.0] * 4 + [1.0] * 8 + [3.0] + [1.0] * 7)
        psd["available"][:] = 1
    proof = select_history_witnesses(source, derived)
    assert proof["selected_sequences"] == [1, 5, 13, 19, 21, 25, 33, 39]
    assert proof["selected_witnesses"] == 8 and proof["source_witnesses"] == 20
    assert [row["selected_sequences"] for row in proof["attempts"]] == [[1, 5, 13, 19], [21, 25, 33, 39]]
    assert [row["omitted_witnesses"] for row in proof["attempts"]] == [6, 6]
    with h5py.File(source, "r+") as handle:
        del handle["metadata/source_segments_json"]
    with pytest.raises(ValueError, match="explicit source attempt"):
        select_history_witnesses(source, tmp_path / "rejected.h5")
