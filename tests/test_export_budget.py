"""Finite intermediate storage refuses publication without harming existing data."""
from __future__ import annotations

import base64
import json
import os
from pathlib import Path
import shutil
import sys

import pytest

from qcl_negf_contracts.messages import ContractError
from qcl_negf_results.commits import artifact_row, json_bytes
from qcl_negf_results.export import export_snapshot
from test_export import fixture


def _add_noise(generation: Path) -> int:
    path = generation / "noise.txt"
    path.write_bytes(base64.b64encode(os.urandom(256 * 1024)))
    commit_path = generation / "commit.json"
    commit = json.loads(commit_path.read_bytes())
    commit["artifacts"].append(artifact_row(path, relative=path.name, role="plan", media_type="text/plain"))
    commit_path.write_bytes(json_bytes(commit))
    return path.stat().st_size


def test_tiny_budget_refuses_before_any_payload_capture(tmp_path: Path) -> None:
    root, output = tmp_path / "run", tmp_path / "exports"
    generation = fixture(root)
    source = (generation / "analysis.h5").read_bytes()
    progress = []
    with pytest.raises(ContractError, match="export.*budget"):
        export_snapshot(root, output, byte_budget=1, reserve_bytes=0, progress=progress.append)
    assert not any(row["phase"] == "capturing" for row in progress)
    assert not list(output.iterdir())
    assert (generation / "analysis.h5").read_bytes() == source


def test_free_space_reserve_refuses_before_payload_capture(tmp_path: Path, monkeypatch) -> None:
    root, output = tmp_path / "run", tmp_path / "exports"
    fixture(root)
    usage = shutil.disk_usage(tmp_path)
    monkeypatch.setattr(shutil, "disk_usage", lambda path: usage._replace(free=4096))
    progress = []
    with pytest.raises(ContractError, match="space|reserve"):
        export_snapshot(root, output, byte_budget=1024 * 1024, reserve_bytes=4096, progress=progress.append)
    assert not any(row["phase"] == "capturing" for row in progress)
    assert not list(output.iterdir())


def test_actual_compressed_bytes_exhaust_budget_and_never_acknowledge(tmp_path: Path) -> None:
    root, output = tmp_path / "run", tmp_path / "exports"
    generation = fixture(root)
    noise_bytes = _add_noise(generation)
    # Captures fit. The real XZ stream (incompressible ASCII) cannot fit in
    # the remaining 16 KiB. No scientific object may be removed to make it fit.
    budget = noise_bytes + (generation / "analysis.h5").stat().st_size + 16 * 1024
    progress = []
    with pytest.raises(ContractError, match="export.*budget"):
        export_snapshot(root, output, byte_budget=budget, reserve_bytes=0, progress=progress.append)
    assert any(row["phase"] == "compressing" for row in progress)
    assert not any(row["phase"] in {"publishing", "completed"} for row in progress)
    assert not list(output.iterdir())
    assert (generation / "noise.txt").stat().st_size == noise_bytes


def test_budget_failure_preserves_previous_archive_and_receipt(tmp_path: Path) -> None:
    root, output = tmp_path / "run", tmp_path / "exports"
    generation = fixture(root)
    old = export_snapshot(root, output)
    previous = {path.name: path.read_bytes() for path in output.iterdir()}
    _add_noise(generation)
    with pytest.raises(ContractError, match="export.*budget"):
        export_snapshot(root, output, byte_budget=1, reserve_bytes=0)
    assert {path.name: path.read_bytes() for path in output.iterdir()} == previous
    assert output.joinpath(old["archive"]).exists()


def test_matching_verified_cache_needs_no_new_spool_budget(tmp_path: Path, monkeypatch) -> None:
    root, output = tmp_path / "run", tmp_path / "exports"
    fixture(root)
    old = export_snapshot(root, output)
    usage = shutil.disk_usage(tmp_path)
    monkeypatch.setattr(shutil, "disk_usage", lambda path: usage._replace(free=0))
    progress = []
    actual = export_snapshot(root, output, byte_budget=1, reserve_bytes=4096, progress=progress.append)
    assert actual == old
    assert progress[-1]["phase"] == "cached"
    assert len(list(output.iterdir())) == 2


@pytest.mark.parametrize("kwargs", [{"byte_budget": 0}, {"byte_budget": True},
    {"byte_budget": 1.5}, {"reserve_bytes": -1}, {"reserve_bytes": False}])
def test_invalid_budget_options_refused_before_destination_write(tmp_path: Path, kwargs: dict) -> None:
    root, output = tmp_path / "run", tmp_path / "exports"
    fixture(root)
    with pytest.raises((ContractError, ValueError), match="budget|reserve"):
        export_snapshot(root, output, **kwargs)
    assert not output.exists()


def test_cli_passes_finite_budget_and_reports_refusal(tmp_path: Path, monkeypatch, capsys) -> None:
    from qcl_negf_results.cli import main
    root, output = tmp_path / "run", tmp_path / "exports"
    fixture(root)
    monkeypatch.setattr(sys, "argv", ["qcl-negf-results", "export", str(root), str(output),
                                     "--byte-budget", "1", "--reserve-bytes", "0"])
    with pytest.raises(SystemExit) as stopped:
        main()
    assert stopped.value.code == 2
    assert "intermediate export budget" in capsys.readouterr().err
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


def test_parquet_derivation_writes_are_actually_limited(tmp_path: Path) -> None:
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
    pq.write_table(pa.Table.from_pylist([{"monotonic_ns": 1}], schema=schema), source)
    original = source.read_bytes()
    spool.mkdir()
    budget = ExportBudget(spool, byte_budget=64, reserve_bytes=0)
    target = spool / "derived.parquet"
    with pytest.raises(ContractError, match="export.*budget"):
        compact_performance([source], target, budget=budget)
    assert target.stat().st_size <= 64
    assert source.read_bytes() == original


def test_hdf5_derivation_growth_is_limited_without_touching_source(tmp_path: Path) -> None:
    import h5py
    import numpy as np
    from native_result_fixtures import declare_native
    from test_psd_history_proof import _selected
    from qcl_negf_results.witness_selection import select_history_witnesses
    from qcl_negf_results.export_budget import ExportBudget
    source, spool = tmp_path / "history.h5", tmp_path / "spool"
    with h5py.File(source, "w") as handle:
        declare_native(handle, "qcl-negf-scientific-history-v4", "science.history", scba_rows=20)
        handle["metadata/source_segments_json"] = json.dumps([{
            "identity": {"attempt": 1}, "scba_rows": 20, "scba_sequence_first": 1, "scba_sequence_last": 20}])
        psd = handle["psd_history"]
        del psd["selected_blocks"]
        _selected(psd, 20)
        psd["ratio"] = np.asarray([-1.0] * 4 + [1.0] * 8 + [3.0] + [1.0] * 7)
        psd["available"][:] = 1
    original = source.read_bytes()
    spool.mkdir()
    byte_budget = source.stat().st_size
    budget = ExportBudget(spool, byte_budget=byte_budget, reserve_bytes=0)
    with pytest.raises(ContractError, match="export.*budget"):
        select_history_witnesses(source, spool / "selected.h5", budget=budget)
    assert sum(path.stat().st_size for path in spool.iterdir()) <= byte_budget
    assert source.read_bytes() == original
