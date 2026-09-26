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
from qcl_negf_results.export import PinnedCommit, _pin, _selection, export_snapshot
from qcl_negf_results.telemetry import TelemetryWriter


def test_catalog_storage_is_linear_and_pinned_prefix_survives_appends(tmp_path: Path) -> None:
    writer = TelemetryWriter(tmp_path / "run/performance", {}, {}, batch_rows=8)
    sizes = []
    for index in range(100):
        writer.ingest_spans([{"phase": "dyson", "duration_seconds": 0.125}],
                            source_cursors={"source": {"offset": index + 1}})
        if index == 9:
            old = _pin(writer.directory / f"commit-{writer.generation:06d}.json")
            before = tuple(old.artifacts)
            sizes.append(sum(p.stat().st_size for p in writer.directory.glob("catalog-*.json")))
    sizes.append(sum(p.stat().st_size for p in writer.directory.glob("catalog-*.json")))
    # Ten times as many rows cannot write 100 times as much inventory.
    assert 7 < sizes[1] / sizes[0] < 12
    assert max(p.stat().st_size for p in writer.directory.glob("commit-*.json")) < 8192
    assert _pin(old.path).artifacts == before
    assert len(_pin(writer.directory / f"commit-{writer.generation:06d}.json").artifacts) > len(before)
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
    with pytest.raises(ContractError, match="checksum"):
        export_snapshot(tmp_path / "run", tmp_path / "exports")
    assert not list((tmp_path / "exports").glob("*.tar.xz"))


def test_complete_committed_resource_and_span_histories_are_exported(tmp_path: Path) -> None:
    writer = TelemetryWriter(tmp_path / "run/performance", {}, {}, batch_rows=2)
    for index in range(10):
        writer.record_resource({"wall_seconds": float(index), "cpu_time_seconds": index * 0.5})
        writer.record_span({"phase": "dyson", "duration_seconds": (index + 1) / 1000})
    writer.close()
    receipt = export_snapshot(tmp_path / "run", tmp_path / "exports")
    seen: dict[str, list[dict]] = {}
    with tarfile.open(tmp_path / "exports" / f'{receipt["sha256"]}.tar.xz', "r:xz") as archive:
        manifest = json.load(archive.extractfile("manifest.json"))
        for record in manifest["records"]:
            for item in record["included"]:
                if item["role"] != "performance.window":
                    continue
                payload = archive.extractfile(item["object"]).read()
                assert hashlib.sha256(payload).hexdigest() == item["sha256"]
                path = tmp_path / (item["sha256"] + ".parquet")
                path.write_bytes(payload)
                table = pq.read_table(path)
                seen.setdefault(table.schema.metadata[b"table"].decode(), []).extend(table.to_pylist())
    assert len(seen["resources"]) == len(seen["spans"]) == 10
    assert min(row["wall_seconds"] for row in seen["resources"]) == 0
    assert max(row["wall_seconds"] for row in seen["resources"]) == 9
    assert max(row["duration_seconds"] for row in seen["spans"]) == 0.01
    assert receipt["compressor"]["preset"] == 1
    assert receipt["compressor"]["extreme"] is False
    assert receipt["manifest_bytes"] < 1024 * 1024
    assert set(receipt["phase_seconds"]) >= {"collecting", "capturing", "validating_closure", "compressing", "verifying", "publishing"}


def test_153022_excluded_objects_have_bounded_inventory(tmp_path: Path) -> None:
    artifacts = tuple(Artifact(f"spans-{index:09d}.parquet", "performance.full", 100,
        hashlib.sha256(str(index).encode()).hexdigest(), "application/vnd.apache.parquet", "full-state")
        for index in range(153_022))
    commit = PinnedCommit(tmp_path / "commit.json", {}, "a" * 64, artifacts)
    selected, omitted = _selection(commit, "science")
    assert selected == []
    assert len(omitted) == 1 and omitted[0]["count"] == 153_022
    assert omitted[0]["bytes"] == 15_302_200
    assert len(json.dumps(omitted)) < 1024
    assert "sha256" not in omitted[0]  # Catalog digest is not an omitted object hash.
    assert len(omitted[0]["catalog_digest"]) == 64
