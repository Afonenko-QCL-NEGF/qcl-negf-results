from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tarfile

import pyarrow.parquet as pq
import pytest

from qcl_negf_contracts.artifacts import Artifact
from qcl_negf_contracts.messages import ContractError
from qcl_negf_results.catalog import catalog_artifacts
from qcl_negf_results.export import export_snapshot
from qcl_negf_results.state import StateReader, ResultReadLimits
from result_v2_fixture import source, record, append, IDENTITY
from qcl_negf_results.telemetry import TelemetryWriter


def test_catalog_storage_is_linear_and_pinned_prefix_survives_appends(tmp_path: Path) -> None:
    writer = TelemetryWriter(tmp_path / "run/performance", {}, {}, batch_rows=8)
    sizes = []
    for index in range(100):
        writer.ingest_spans([{"phase": "dyson", "duration_seconds": 0.125}],
                            source_cursors={"source": {"offset": index + 1}})
        if index == 9:
            old = dict(writer.catalog_watermark)
            before = tuple(catalog_artifacts(writer.directory, old))
            sizes.append(sum(p.stat().st_size for p in writer.directory.glob("catalog-*.json")))
    sizes.append(sum(p.stat().st_size for p in writer.directory.glob("catalog-*.json")))
    # Ten times as many rows cannot write 100 times as much inventory.
    assert 7 < sizes[1] / sizes[0] < 12
    assert max(p.stat().st_size for p in writer.directory.glob("commit-*.json")) < 8192
    assert tuple(catalog_artifacts(writer.directory, old)) == before
    assert len(catalog_artifacts(writer.directory, writer.catalog_watermark)) > len(before)
    writer.close()


def test_corrupt_or_truncated_catalog_cannot_publish_snapshot(tmp_path: Path) -> None:
    writer = TelemetryWriter(tmp_path / "run/performance", {}, {}, batch_rows=1)
    writer.record_resource({"wall_seconds": 1.0})
    writer.close()
    watermark = writer.catalog_watermark
    assert watermark is not None
    chunk = writer.directory / watermark["path"]
    chunk.write_bytes(chunk.read_bytes() + b" ")
    with pytest.raises(ContractError, match="checksum"):
        catalog_artifacts(writer.directory, watermark)
    with pytest.raises(TypeError, match="StateReader"):
        export_snapshot(tmp_path / "run", tmp_path / "exports", profile="science", byte_budget=1024**2, reserve_bytes=0)
    assert not list((tmp_path / "exports").glob("*.tar.xz"))


def test_technical_tables_remain_complete_without_implicit_scientific_export(tmp_path):
    writer = TelemetryWriter(tmp_path / 'performance', {}, {}, batch_rows=2)
    for index in range(10):
        writer.record_resource({'wall_seconds': float(index), 'cpu_time_seconds': index * 0.5})
        writer.record_span({'phase': 'dyson', 'duration_seconds': (index + 1) / 1000})
    writer.close()
    seen = {}
    for artifact in catalog_artifacts(writer.directory, writer.catalog_watermark):
        if artifact.media_type == 'application/vnd.apache.parquet':
            table = pq.read_table(writer.directory / artifact.path)
            seen.setdefault(table.schema.metadata[b'table'].decode(), []).extend(table.to_pylist())
    assert len(seen['resources']) == len(seen['spans']) == 10
    assert [row['wall_seconds'] for row in seen['resources']] == list(map(float, range(10)))
    with pytest.raises(TypeError, match='StateReader'):
        export_snapshot(tmp_path, tmp_path / 'exports', profile='science', byte_budget=1024**2, reserve_bytes=0)


def test_import_reference_capacity_fails_before_advancing_trusted_prefix(tmp_path):
    first = source(tmp_path, [record()])
    second = append(tmp_path, first, record(2, 9))
    with StateReader(tmp_path, state_ref=second, expected_identity=IDENTITY,
                     limits=ResultReadLimits(maximum_catalog_entries=3)) as reader:
        with pytest.raises(ContractError) as error:
            reader.import_history()
        assert error.value.code == 'budget_exceeded'
        assert not reader.source.integrity['catalog_verified']
