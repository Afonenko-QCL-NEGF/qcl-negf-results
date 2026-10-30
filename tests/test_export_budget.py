"""Native5 source capture and independent finite spool guards."""
import json
import shutil
import sys
from pathlib import Path
import pytest
from qcl_negf_contracts.messages import ContractError
from qcl_negf_results.export import export_snapshot
from qcl_negf_results.state import StateReader
from result_v2_fixture import IDENTITY, source, record, encoded, export_source


def test_tiny_budget_refuses_before_any_payload_capture(tmp_path):
    root, output = tmp_path / 'source', tmp_path / 'export'
    ref = source(root, [record()])
    original = (root / 'records.h5').read_bytes()
    events = []
    with pytest.raises(ContractError, match='export.*budget'):
        export_source(root, ref, output, byte_budget=1, progress=events.append)
    assert not any(row['phase'] == 'capturing' for row in events)
    assert not list(output.iterdir())
    assert (root / 'records.h5').read_bytes() == original


def test_free_space_reserve_refuses_before_payload_capture(tmp_path, monkeypatch):
    root, output = tmp_path / 'source', tmp_path / 'export'
    ref = source(root, [record()])
    usage = shutil.disk_usage(tmp_path)
    monkeypatch.setattr(shutil, 'disk_usage', lambda path: usage._replace(free=4096))
    events = []
    with pytest.raises(ContractError, match='space|reserve'):
        export_source(root, ref, output, reserve_bytes=4096, progress=events.append)
    assert not any(row['phase'] == 'capturing' for row in events)
    assert not list(output.iterdir())


def test_actual_archive_bytes_exhaust_spool_budget_without_publication(tmp_path):
    root, output = tmp_path / 'source', tmp_path / 'export'
    ref = source(root, [record()])
    # Exact captures fit this allowance, but capture+compressed archive+metadata
    # do not. The format must not trim any chosen scientific source object.
    catalog = json.loads((root / 'catalog.json').read_text())
    size = int(ref['manifest']['bytes']) + (root / 'catalog.json').stat().st_size + sum(int(row['bytes']) for row in catalog['artifacts'])
    events = []
    with pytest.raises(ContractError, match='export.*budget'):
        export_source(root, ref, output, byte_budget=size+32, progress=events.append)
    assert any(row['phase'] == 'compressing' for row in events)
    assert not list(output.iterdir())


def test_failed_new_export_preserves_previous_archive_and_receipt(tmp_path):
    root, output = tmp_path / 'source', tmp_path / 'export'
    ref = source(root, [record()])
    export_source(root, ref, output)
    previous = {p.name:p.read_bytes() for p in output.iterdir()}
    with pytest.raises(ContractError, match='export.*budget'):
        export_source(root, ref, output, byte_budget=1)
    assert {p.name:p.read_bytes() for p in output.iterdir()} == previous


@pytest.mark.parametrize('kwargs', [{'byte_budget':0}, {'byte_budget':True}, {'byte_budget':1.5},
    {'reserve_bytes':-1}, {'reserve_bytes':False}])
def test_invalid_budget_options_refused_before_destination_write(tmp_path, kwargs):
    root, output = tmp_path / 'source', tmp_path / 'export'
    ref = source(root, [record()])
    with pytest.raises(ValueError, match='budget|reserve'):
        export_source(root, ref, output, **kwargs)
    assert not output.exists()


def test_cli_passes_finite_budget_and_reports_refusal(tmp_path, monkeypatch, capsys):
    from qcl_negf_results.cli import main
    root, output = tmp_path / 'source', tmp_path / 'export'
    ref = source(root, [record()])
    identity, state = tmp_path / 'identity.json', tmp_path / 'state.json'
    identity.write_bytes(encoded(IDENTITY)); state.write_bytes(encoded(ref))
    monkeypatch.setattr(sys, 'argv', ['qcl-negf-results', 'export', str(root), str(output),
        '--identity', str(identity), '--state-ref', str(state), '--byte-budget','1','--reserve-bytes','0'])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
    assert 'intermediate export budget' in capsys.readouterr().err
    assert not output.exists() or not list(output.iterdir())


def test_seeked_write_and_truncate_cannot_bypass_total_budget(tmp_path: Path) -> None:
    from qcl_negf_results.export_budget import ExportBudget
    budget = ExportBudget(tmp_path, byte_budget=10, reserve_bytes=0)
    first, second = tmp_path / "first", tmp_path / "second"
    with budget.open(first, "w+b") as stream:
        stream.write(b"123456")
        stream.seek(0)
        stream.write(b"ab")
        stream.seek(8)
        with pytest.raises(ContractError, match="export.*budget"):
            stream.write(b"xyz")
        assert first.read_bytes() == b"ab3456"
        stream.truncate(3)
    with budget.open(second, "wb") as stream:
        stream.write(b"1234567")
        with pytest.raises(ContractError, match="export.*budget"):
            stream.write(b"8")
    assert first.stat().st_size + second.stat().st_size == 10


def test_typed_buffer_counts_bytes_instead_of_elements(tmp_path: Path) -> None:
    import numpy as np
    from qcl_negf_results.export_budget import ExportBudget
    budget = ExportBudget(tmp_path, byte_budget=16, reserve_bytes=0)
    path = tmp_path / "typed"
    with budget.open(path, "wb") as stream:
        with pytest.raises(ContractError, match="export.*budget"):
            stream.write(memoryview(np.arange(3, dtype=np.int64)))
    assert path.stat().st_size == 0


def test_parquet_derivation_writes_are_actually_limited(tmp_path: Path, monkeypatch) -> None:
    import os
    import struct
    import pyarrow as pa
    import pyarrow.parquet as pq
    from qcl_negf_contracts.artifacts import CONTRACT_SET
    from qcl_negf_contracts.telemetry import TABLE_FIELDS, TELEMETRY_SCHEMA
    from qcl_negf_results.compaction import compact_performance
    from qcl_negf_results.export_budget import ExportBudget
    types = {"string": pa.string(), "int64": pa.int64(), "float64": pa.float64(), "bool": pa.bool_()}
    schema = pa.schema([pa.field(name, types[kind]) for name, kind in TABLE_FIELDS["resources"]],
        metadata={b"schema": TELEMETRY_SCHEMA.encode(), b"contract_set": CONTRACT_SET.encode(), b"table": b"resources"})
    source, spool = tmp_path / "source.parquet", tmp_path / "spool"
    nan_bits = bytes.fromhex("341200000000f87f")
    pq.write_table(pa.Table.from_pylist([{"monotonic_ns": 1, "wall_seconds": -0.0,
        "cpu_time_seconds": struct.unpack("<d", nan_bits)[0]}], schema=schema), source)
    original = source.read_bytes()
    spool.mkdir()
    budget = ExportBudget(spool, byte_budget=64, reserve_bytes=0)
    target = spool / "derived.parquet"
    def measure_writes(allowance):
        writes = []
        original_open = allowance.open
        def measured_open(path, *args, **kwargs):
            sink = original_open(path, *args, **kwargs)
            stream = sink.stream
            class PhysicalWrites:
                def __getattr__(self, name):
                    return getattr(stream, name)
                def write(self, value):
                    count = stream.write(value)
                    # Observe actual unbuffered delivery before compaction cleanup.
                    writes.append((count, os.fstat(stream.fileno()).st_size))
                    return count
            sink.stream = PhysicalWrites()
            return sink
        monkeypatch.setattr(allowance, "open", measured_open)
        return writes
    writes = measure_writes(budget)
    with pytest.raises(ContractError, match="export.*budget"):
        compact_performance([source], target, budget=budget)
    assert writes and all(count > 0 and size <= 64 for count, size in writes)
    high_water = max(size for _, size in writes)
    assert budget.used == budget.sizes[target.resolve()] == high_water <= 64
    assert not target.exists()
    assert source.read_bytes() == original
    healthy = ExportBudget(spool, byte_budget=1024 * 1024, reserve_bytes=0)
    healthy_writes = measure_writes(healthy)
    completed = spool / "complete.parquet"
    proof = compact_performance([source], completed, budget=healthy)
    actual = pq.read_table(completed)
    assert proof["rows"] == 1 and actual["monotonic_ns"].to_pylist() == [1]
    assert actual.schema.equals(schema, check_metadata=True)
    assert struct.pack("<d", actual["wall_seconds"][0].as_py()) == bytes.fromhex("0000000000000080")
    assert struct.pack("<d", actual["cpu_time_seconds"][0].as_py()) == nan_bits
    assert healthy_writes and all(size <= healthy.byte_budget for _, size in healthy_writes)
    assert healthy.used == max(size for _, size in healthy_writes) == completed.stat().st_size
    assert source.read_bytes() == original


def test_hdf5_derivation_growth_is_limited_without_touching_source(tmp_path):
    import h5py
    import numpy as np
    from qcl_negf_results.export_budget import ExportBudget
    from result_v2_fixture import metadata
    root = tmp_path / 'source'
    source(root, [record()])
    original = (root / 'records.h5').read_bytes()
    spool = tmp_path / 'spool'
    spool.mkdir()
    budget = ExportBudget(spool, byte_budget=4096, reserve_bytes=0)
    with pytest.raises(ContractError, match='export.*budget'):
        with budget.open(spool / 'derived.h5', 'w+b', defer_errors=True) as stream:
            with h5py.File(stream, 'w') as handle:
                metadata(handle)
                dataset = handle.create_dataset('manufactured', shape=(32768,), dtype='<f8')
                for first in range(0, 32768, 1024):
                    dataset[first:first+1024] = np.ones(1024)
    assert sum(path.stat().st_size for path in spool.iterdir()) <= 4096
    assert (root / 'records.h5').read_bytes() == original
