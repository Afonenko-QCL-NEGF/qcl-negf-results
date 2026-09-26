from __future__ import annotations

import json
import io
from pathlib import Path
import tarfile

import pyarrow.parquet as pq
import pytest

from qcl_negf_results.export import export_snapshot
from qcl_negf_results.telemetry import CpuCoverage, TelemetryWriter
from qcl_negf_results.catalog import catalog_artifacts


def test_closed_segments_preserve_types_nulls_and_counts(tmp_path: Path) -> None:
    writer = TelemetryWriter(tmp_path / "performance", {"execution_id": "e-1"}, {"affinity": [0, 1]}, batch_rows=2)
    writer.record_resource({"wall_seconds": 1.0, "cpu_time_seconds": 1.7, "rss_bytes": 2048, "cpu_count": 2})
    writer.record_resource({"wall_seconds": 2.0, "missing_reason": "process_finished"})
    segments = list(writer.directory.glob("resources-*.parquet"))
    assert len(segments) == 1
    table = pq.read_table(segments[0])
    assert table.num_rows == 2
    assert table["rss_bytes"].to_pylist() == [2048, None]
    assert table.schema.metadata[b"schema"] == b"qcl-negf.performance.v2"
    path = writer.close()
    commit = json.loads(path.read_bytes())
    assert commit["closed"] is True
    assert all(not row["path"].startswith(".writing") for row in commit["artifacts"])
    assert writer.close() == path
    with pytest.raises(RuntimeError, match="closed"):
        writer.record_control({"event": "late"})


def test_thread_and_cache_metrics_survive_science_export_and_old_rows(tmp_path: Path) -> None:
    writer = TelemetryWriter(tmp_path / "run/performance", {}, {}, batch_rows=16)
    metrics = {"wall_seconds": 2., "voluntary_context_switches": 102,
        "involuntary_context_switches": 51, "voluntary_context_switches_delta": 33,
        "involuntary_context_switches_delta": 11, "thread_counter_interval_seconds": 1.,
        "thread_counter_scope": "root_process_live_threads; deltas_identity_matched_endpoints",
        "thread_listed_count": 2, "thread_sample_count": 2, "matched_thread_count": 2,
        "new_thread_count": 0, "vanished_thread_count": 0, "thread_sampling_complete": True,
        "thread_delta_complete": True, "thread_schedstat_enabled": False,
        "thread_runqueue_wait_seconds": .23, "thread_runqueue_wait_delta_seconds": .03,
        "thread_runtime_seconds": .2, "thread_runtime_delta_seconds": .1,
        "thread_counter_details": '{"42:10":{"cpu_seconds":0.1}}',
        "memory_anon_bytes": 1024, "memory_file_bytes": 4096, "memory_kernel_bytes": 512,
        "memory_inactive_file_bytes": 2048, "memory_file_dirty_bytes": 32,
        "memory_file_writeback_bytes": 64, "memory_shmem_bytes": 128, "memory_file_mapped_bytes": 256}
    writer.record_resource({"wall_seconds": 1.}, science_window=True)  # Old producer remains valid.
    writer.record_resource(metrics, science_window=True)
    writer.close()
    receipt = export_snapshot(tmp_path / "run", tmp_path / "exports")
    found = []
    with tarfile.open(tmp_path / "exports" / f'{receipt["sha256"]}.tar.xz', "r:xz") as archive:
        for member in archive.getmembers():
            if member.name.endswith(".parquet"):
                table = pq.read_table(io.BytesIO(archive.extractfile(member).read()))
                if "thread_counter_scope" in table.column_names:
                    found.extend(table.to_pylist())
    assert len(found) == 2
    assert found[0]["thread_counter_scope"] is None
    assert found[0]["memory_file_dirty_bytes"] is None
    for name, value in metrics.items():
        assert found[1][name] == value


def test_control_events_and_approved_windows_in_science(tmp_path: Path) -> None:
    writer = TelemetryWriter(tmp_path / "run/performance", {}, {}, batch_rows=16)
    writer.record_control({"event": "admission", "reason": "allocation_available", "attempt": 1})
    writer.record_resource({"wall_seconds": 1.0, "cpu_time_seconds": 0.5})
    writer.record_span({"phase": "startup", "duration_seconds": 0.5}, science_window=True)
    writer.close()
    receipt = export_snapshot(tmp_path / "run", tmp_path / "exports")
    with tarfile.open(tmp_path / "exports" / f'{receipt["sha256"]}.tar.xz', "r:xz") as archive:
        manifest = json.load(archive.extractfile("manifest.json"))
    roles = [item["role"] for record in manifest["records"] for item in record["included"]]
    assert "performance.window" in roles
    assert "performance.full" not in roles
    assert "performance.summary" in roles


def test_multiple_sessions_do_not_replace_each_other(tmp_path: Path) -> None:
    first = TelemetryWriter(tmp_path / "performance", {"worker_id": "a"}, {})
    second = TelemetryWriter(tmp_path / "performance", {"worker_id": "b"}, {})
    first.close()
    second.close()
    index = json.loads((tmp_path / "performance/index.json").read_bytes())
    assert len(index["sessions"]) == 2
    assert len({session["id"] for session in index["sessions"]}) == 2


def test_unknown_or_nonfinite_fields_are_not_lost(tmp_path: Path) -> None:
    writer = TelemetryWriter(tmp_path / "performance", {}, {})
    with pytest.raises(ValueError, match="unknown"):
        writer.record_resource({"wall_seconds": 1.0, "CPU": 2.0})
    with pytest.raises(ValueError, match="finite nonnegative"):
        writer.record_span({"phase": "solve", "duration_seconds": float("nan")})
    writer.close()


def test_ingestion_cursor_and_rows_share_one_commit_even_at_batch_boundary(tmp_path: Path) -> None:
    writer = TelemetryWriter(tmp_path / "performance", {}, {}, batch_rows=2)
    writer.record_resource({"wall_seconds": 1.0})
    generation = writer.generation
    commit_path = writer.ingest_spans([
        {"phase": "dyson", "duration_seconds": 0.125},
        {"phase": "total", "duration_seconds": 0.25}],
        source_cursors={"source": {"offset": 128, "rows": 2}})
    assert writer.generation == generation + 1
    commit = json.loads(commit_path.read_bytes())
    assert commit["source_cursors"]["source"]["offset"] == 128
    segments = [item for item in catalog_artifacts(writer.directory, commit["telemetry_catalog"])
                if item.media_type == "application/vnd.apache.parquet"]
    assert sum(pq.read_table(writer.directory / item.path).num_rows for item in segments) == 3
    writer.close()


def test_invalid_ingestion_batch_never_advances_source_or_partially_accepts(tmp_path: Path) -> None:
    writer = TelemetryWriter(tmp_path / "performance", {}, {})
    with pytest.raises(ValueError, match="finite"):
        writer.ingest_spans([{"phase": "total", "duration_seconds": 1.0},
            {"phase": "total", "duration_seconds": float("nan")}],
            source_cursors={"source": {"offset": 100}})
    assert writer.source_cursors == {}
    assert writer.counts["spans"] == 0
    writer.close()


def resource_sample(allocation: str, wall: float, cpu: float | None, *, width: int = 32) -> dict:
    return {"allocation_id": allocation, "execution_id": f"execution-{allocation}",
            "wall_seconds": wall, "monotonic_ns": int(wall * 1e9),
            "cpu_time_seconds": cpu, "cpu_count": width}


def test_cpu_coverage_separates_allocations_and_weights_measured_intervals() -> None:
    coverage = CpuCoverage()
    coverage.record(resource_sample("a", 10, 100))
    coverage.record(resource_sample("b", 11, 20, width=16))
    coverage.record(resource_sample("a", 12, 132))
    coverage.record(resource_sample("b", 12, 32, width=16))
    coverage.record(resource_sample("a", 15, 204))
    a, b = coverage.summary()["allocations"]
    assert a["covered_wall_seconds"] == 5
    assert a["cpu_seconds"] == 104  # First 100 CPU seconds are not observed intervals.
    assert a["fraction_of_allocated_cpu"] == pytest.approx(0.65)
    assert a["mean_busy_logical_cpus"] == pytest.approx(20.8)
    assert b["fraction_of_allocated_cpu"] == pytest.approx(0.75)
    assert b["covered_wall_seconds"] == 1


def test_missing_resets_and_capacity_changes_are_unknown_but_long_gaps_are_measured() -> None:
    coverage = CpuCoverage()
    for sample in [resource_sample("a", 1, 10), resource_sample("a", 2, None),
                   resource_sample("a", 3, 20), resource_sample("a", 4, 2),
                   resource_sample("a", 5, 10, width=16),
                   resource_sample("a", 20, 40, width=16)]:
        coverage.record(sample)
    row = coverage.summary()["allocations"][0]
    assert row["fraction_of_allocated_cpu"] == pytest.approx(0.125)
    assert row["covered_wall_seconds"] == 15
    assert row["coarse_wall_seconds"] == 15
    assert row["missing_samples"] == 1
    assert row["discontinuous_intervals"] == 2
    coverage.record(resource_sample("a", 21, 48, width=16))
    row = coverage.summary()["allocations"][0]
    assert row["fraction_of_allocated_cpu"] == pytest.approx(38 / (16 * 16))
    assert row["covered_wall_seconds"] == 16


def test_cpu_summary_survives_science_export_without_raw_resource_tables(tmp_path: Path) -> None:
    writer = TelemetryWriter(tmp_path / "run/performance", {}, {}, batch_rows=16)
    writer.record_resource(resource_sample("a", 1, 10))
    writer.record_resource(resource_sample("a", 2, 26))
    writer.close()
    receipt = export_snapshot(tmp_path / "run", tmp_path / "exports")
    summaries = []
    with tarfile.open(tmp_path / "exports" / f'{receipt["sha256"]}.tar.xz', "r:xz") as archive:
        for member in archive.getmembers():
            if member.name.startswith("objects/") and member.name.endswith(".json"):
                value = json.load(archive.extractfile(member))
                if "cpu_coverage" in value:
                    summaries.append(value)
        manifest = json.load(archive.extractfile("manifest.json"))
    assert len(summaries) == 1
    assert summaries[0]["cpu_coverage"]["allocations"][0]["fraction_of_allocated_cpu"] == 0.5
    assert not any(item["role"] == "performance.full"
                   for record in manifest["records"] for item in record["included"])


def test_unmeasured_monotonic_timestamp_is_not_a_cpu_interval(tmp_path: Path) -> None:
    writer = TelemetryWriter(tmp_path / "performance", {}, {})
    writer.record_resource({**resource_sample("a", 1, 10), "monotonic_ns": None})
    writer.record_resource(resource_sample("a", 2, 20))
    writer.close()
    coverage = writer.cpu_coverage.summary()["allocations"][0]
    assert coverage["missing_samples"] == 1
    assert coverage["fraction_of_allocated_cpu"] is None


def test_staged_batches_keep_durable_offsets_and_raw_values_atomic(tmp_path: Path) -> None:
    writer = TelemetryWriter(tmp_path / "performance", {}, {}, batch_rows=4)
    first = {"journal": {"offset": 10}}
    writer.stage_spans([{"phase": "total", "duration_seconds": 0.125}], source_cursors=first)
    first["journal"]["offset"] = 999
    assert writer.source_cursors == {}
    assert writer.staged_source_cursors["journal"]["offset"] == 10
    assert writer.pending_rows == 1
    assert not list(writer.directory.glob("spans-*.parquet"))
    for offset in (20, 30, 40):
        writer.stage_spans([{"phase": "total", "duration_seconds": offset / 1000}],
                           source_cursors={"journal": {"offset": offset}})
    assert writer.pending_rows == 0
    assert writer.source_cursors["journal"]["offset"] == 40
    segments = list(writer.directory.glob("spans-*.parquet"))
    assert len(segments) == 1
    assert pq.read_table(segments[0])["duration_seconds"].to_pylist() == [0.125, 0.02, 0.03, 0.04]
    assert pq.ParquetFile(segments[0]).metadata.row_group(0).column(0).compression == "ZSTD"
    writer.close()


def test_quiet_partial_batch_flushes_at_deadline_and_close_has_one_commit(tmp_path: Path) -> None:
    writer = TelemetryWriter(tmp_path / "performance", {}, {}, max_buffer_seconds=15)
    writer.record_resource({"wall_seconds": 1.0})
    start = writer.pending_since_ns
    assert start is not None
    generation = writer.generation
    writer.flush_due(now_ns=start + 14_999_999_999)
    assert writer.generation == generation
    writer.flush_due(now_ns=start + 15_000_000_000)
    assert writer.generation == generation + 1
    assert writer.pending_rows == 0
    writer.flush()
    assert writer.generation == generation + 1  # Empty poll does not fsync metadata.
    writer.record_resource({"wall_seconds": 2.0})
    writer.close()
    assert writer.generation == generation + 2
    commit = json.loads((writer.directory / f"commit-{writer.generation:06d}.json").read_bytes())
    summary = json.loads((writer.directory / commit["artifacts"][-1]["path"]).read_bytes())
    assert summary["rows"]["resources"] == 2
    assert summary["closed"] is True
    assert summary["durability"]["maximum_buffer_seconds"] == 15


def test_staging_byte_bound_rejects_before_partial_acceptance(tmp_path: Path) -> None:
    writer = TelemetryWriter(tmp_path / "performance", {}, {}, batch_bytes=1024)
    writer.stage_spans([{"phase": "total", "duration_seconds": 1.0}], source_cursors={"s": {"offset": 1}})
    with pytest.raises(ValueError, match="byte bound"):
        writer.stage_spans([{"phase": "x" * 1025, "duration_seconds": 2.0}],
                           source_cursors={"s": {"offset": 2}})
    assert writer.counts["spans"] == 1
    assert writer.staged_source_cursors["s"]["offset"] == 1
    writer.close()


def test_publication_failure_preserves_durable_cursor_and_retry_has_no_duplicate(tmp_path: Path, monkeypatch) -> None:
    import qcl_negf_results.telemetry as module
    writer = TelemetryWriter(tmp_path / "performance", {}, {})
    writer.ingest_spans([{"phase": "total", "duration_seconds": 1.0}], source_cursors={"s": {"offset": 1}})
    writer.stage_spans([{"phase": "total", "duration_seconds": 2.0}], source_cursors={"s": {"offset": 2}})
    publish = module.publish_pointer
    def fail(*args, **kwargs):
        raise OSError("pointer interrupted")
    monkeypatch.setattr(module, "publish_pointer", fail)
    with pytest.raises(OSError, match="interrupted"):
        writer.flush()
    assert writer.source_cursors["s"]["offset"] == 1
    monkeypatch.setattr(module, "publish_pointer", publish)
    commit = json.loads(writer.flush().read_bytes())
    assert commit["source_cursors"]["s"]["offset"] == 2
    artifacts = catalog_artifacts(writer.directory, commit["telemetry_catalog"])
    rows = [row for item in artifacts if item.media_type == "application/vnd.apache.parquet"
            for row in pq.read_table(writer.directory / item.path).to_pylist()]
    assert [row["duration_seconds"] for row in rows] == [1.0, 2.0]
    writer.close()


def test_live_export_is_exact_committed_prefix_and_barrier_includes_tail(tmp_path: Path) -> None:
    writer = TelemetryWriter(tmp_path / "run/performance", {}, {})
    writer.ingest_spans([{"phase": "total", "duration_seconds": 1.0}], source_cursors={"s": {"offset": 1}})
    writer.stage_spans([{"phase": "total", "duration_seconds": 2.0}], source_cursors={"s": {"offset": 2}})
    def values(receipt):
        result = []
        with tarfile.open(tmp_path / "exports" / f'{receipt["sha256"]}.tar.xz', "r:xz") as archive:
            manifest = json.load(archive.extractfile("manifest.json"))
            assert "uncommitted" in manifest["cutoff"]
            assert manifest["job_complete"] is False
            freshness = manifest["freshness"]["telemetry"][0]
            assert freshness["maximum_buffer_seconds"] == 15
            assert freshness["pending_rows"] is None
            assert freshness["committed_age_seconds"] >= 0
            for record in manifest["records"]:
                for item in record["included"]:
                    if item["object"].endswith(".parquet"):
                        import pyarrow as pa
                        table = pq.read_table(pa.BufferReader(archive.extractfile(item["object"]).read()))
                        result.extend(table["duration_seconds"].to_pylist())
        return sorted(result)
    assert values(export_snapshot(tmp_path / "run", tmp_path / "exports")) == [1.0]
    writer.flush()  # Scheduled export/terminal/checkpoint barrier.
    assert values(export_snapshot(tmp_path / "run", tmp_path / "exports")) == [1.0, 2.0]
    writer.close()


def test_long_cpu_intervals_preserve_work_and_report_coarse_resolution() -> None:
    coverage = CpuCoverage()
    for wall, cpu in [(1, 10), (16, 115), (31, 220)]:
        coverage.record({**resource_sample("a", wall, cpu, width=8), "pid": 123,
                         "process_start_ticks": 456, "clock_domain_id": "boot", "cpu_quota_cores": 8.0})
    summary = coverage.summary()
    row = summary["allocations"][0]
    assert row["cpu_seconds"] == 210
    assert row["covered_wall_seconds"] == 30
    assert row["coarse_intervals"] == 2
    assert row["fine_resolution_wall_seconds"] == 0
    assert row["mean_busy_logical_cpus"] == 7
    assert row["fraction_of_allocated_cpu"] == .875
    assert row["fraction_of_observed_window_covered"] == 1
    assert summary["maximum_interval_seconds"] is None
    assert summary["missing_is_zero"] is False


@pytest.mark.parametrize("changed", [{"pid": 124}, {"process_start_ticks": 457},
    {"clock_domain_id": "reboot"}, {"cpu_quota_cores": 4.0}, {"cpu_count": 16},
    {"cgroup_path": "/another"}])
def test_counter_identity_and_quota_changes_break_intervals(changed: dict) -> None:
    coverage = CpuCoverage()
    identity = {"pid": 123, "process_start_ticks": 456, "clock_domain_id": "boot",
                "cpu_quota_cores": 8.0, "cgroup_path": "/worker-a"}
    coverage.record({**resource_sample("a", 1, 10, width=8), **identity})
    coverage.record({**resource_sample("a", 16, 100, width=8), **identity, **changed})
    row = coverage.summary()["allocations"][0]
    assert row["fraction_of_allocated_cpu"] is None
    assert row["covered_wall_seconds"] == 0
    assert row["discontinuous_intervals"] == 1


def test_profile_and_out_of_order_samples_do_not_duplicate_cpu() -> None:
    coverage = CpuCoverage()
    coverage.record(resource_sample("a", 1, 10, width=8))
    coverage.record({**resource_sample("a", 2, 17, width=8), "sampling_mode": "profile_window"})
    coverage.record(resource_sample("a", 16, 115, width=8))
    coverage.record(resource_sample("a", 3, 24, width=8))  # Late publication.
    coverage.record(resource_sample("a", 31, 220, width=8))
    summary = coverage.summary()
    row = summary["allocations"][0]
    assert row["cpu_seconds"] == 210 and row["covered_wall_seconds"] == 30
    assert row["discontinuous_intervals"] == 1
    assert summary["profile_samples_excluded"] == 1


def test_phase_counters_and_missing_values_have_independent_coverage(tmp_path: Path) -> None:
    writer = TelemetryWriter(tmp_path / "performance", {}, {})
    writer.record_span({"phase": "dyson", "measurement_kind": "phase_boundary", "event_kind": "end",
                        "duration_seconds": 2.0, "process_cpu_seconds": 8.0,
                        "gc_seconds": 0.25, "allocated_bytes": 1024})
    writer.record_span({"phase": "dyson", "measurement_kind": "phase_boundary", "event_kind": "end",
                        "duration_seconds": 3.0})
    writer.record_control({"event": "resource_samples_coalesced", "samples_dropped": 4})
    commit = json.loads(writer.close().read_bytes())
    summary = json.loads((writer.directory / commit["artifacts"][-1]["path"]).read_bytes())
    measured = summary["producer_phase_measurements"]["dyson:inclusive"]
    assert measured["calls"] == 2 and measured["cpu_measured_calls"] == 1
    assert measured["cpu_seconds"] == 8 and measured["duration_seconds"] == 5
    assert measured["cpu_measured_wall_seconds"] == 2
    assert measured["gc_measured_wall_seconds"] == 2
    assert summary["phase_attribution"]["admission_wait_seconds"] is None
    assert summary["samples_dropped"] == 4


def test_retained_rss_peak_reaches_summary_after_current_counter_disappears(tmp_path: Path) -> None:
    writer = TelemetryWriter(tmp_path / "performance", {}, {})
    writer.record_resource({"wall_seconds": 1.0, "rss_bytes": None,
                            "peak_rss_bytes": 12345, "missing_reason": "process_counter_unavailable"})
    commit = json.loads(writer.close().read_bytes())
    summary = json.loads((writer.directory / commit["artifacts"][-1]["path"]).read_bytes())
    assert summary["max_observed_rss_bytes"] == 12345
    assert summary["cpu_coverage"]["allocations"] == []
