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
    assert all(not row["path"].startswith(".writing") for row in (dict(path=a.path) for a in catalog_artifacts(writer.directory,writer.catalog_watermark)))
    assert writer.close() == path
    with pytest.raises(RuntimeError, match="closed"):
        writer.record_control({"event": "late"})


def test_thread_and_cache_metrics_survive_science_export_and_old_rows(tmp_path: Path) -> None:
    from native_consumer_fixture import consumer_writer,consumer_export
    context,writer=consumer_writer(tmp_path/"input",batch_rows=16)
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
    receipt = consumer_export(context,writer,tmp_path/"exports")
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
    from native_consumer_fixture import consumer_writer,consumer_export
    context,writer=consumer_writer(tmp_path/"input",batch_rows=16)
    writer.record_control({"event": "admission", "reason": "allocation_available", "attempt": 1})
    writer.record_resource({"wall_seconds": 1.0, "cpu_time_seconds": 0.5})
    writer.record_span({"phase": "startup", "duration_seconds": 0.5}, science_window=True)
    writer.close()
    receipt = consumer_export(context,writer,tmp_path/"exports")
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
    from native_consumer_fixture import diagnostic_parts
    writer=TelemetryWriter(tmp_path/'performance',{}, {},batch_rows=2)
    writer.record_resource({'wall_seconds':1.})
    generation=writer.generation
    parts=diagnostic_parts(tmp_path/'producer',(.125,.25))
    from dataclasses import replace
    writer.stage_phase_batch(replace(parts[-1],first_row=0))
    assert writer.generation==generation+1
    commit=json.loads((writer.root/writer._previous_ref['path']).read_bytes())
    assert commit['source_cursors'][0]['cursor']['next_row']=='2'
    assert commit['source_cursors'][0]['cursor']['segment']==parts[-1].segment_ref
    segments=[a for a in catalog_artifacts(writer.directory,writer.catalog_watermark) if a.media_type=='application/vnd.apache.parquet']
    assert sum(pq.read_table(writer.directory/a.path).num_rows for a in segments)==3
    spans=[row for a in segments if a.schema=='qcl-negf.performance.v2' for row in pq.read_table(writer.directory/a.path).to_pylist() if 'duration_seconds' in row]
    assert [row['duration_seconds'] for row in spans]==[.125,.25]
    writer.close()


def test_invalid_ingestion_batch_never_advances_source_or_partially_accepts(tmp_path: Path) -> None:
    writer = TelemetryWriter(tmp_path / "performance", {}, {})
    with pytest.raises(ValueError, match="finite"):
        writer.ingest_spans([{"phase": "total", "duration_seconds": 1.0},
            {"phase": "total", "duration_seconds": float("nan")}],
            source_cursors={})
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
    from native_consumer_fixture import consumer_writer,consumer_export
    context,writer=consumer_writer(tmp_path/"input",batch_rows=16)
    writer.record_resource(resource_sample("a", 1, 10))
    writer.record_resource(resource_sample("a", 2, 26))
    writer.close()
    receipt = consumer_export(context,writer,tmp_path/"exports")
    summaries = []
    with tarfile.open(tmp_path / "exports" / f'{receipt["sha256"]}.tar.xz', "r:xz") as archive:
        manifest=json.load(archive.extractfile('manifest.json'))
        for record in manifest['records']:
            for item in record['included']:
                if item['role']=='performance.summary':summaries.append(json.load(archive.extractfile(item['object'])))
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
    from native_consumer_fixture import diagnostic_parts
    parts=diagnostic_parts(tmp_path/'producer',(.125,.02,.03,.04))
    writer=TelemetryWriter(tmp_path/'performance',{}, {},batch_rows=4)
    writer.stage_phase_batch(parts[0]);parts[0].cursor['next_row']='999'
    assert writer.source_cursors=={}
    assert next(iter(writer.staged_source_cursors.values()))['cursor']['next_row']=='1'
    assert writer.pending_rows==1
    assert not list(writer.directory.glob('spans-*.parquet'))
    for part in parts[1:]:writer.stage_phase_batch(part)
    assert writer.pending_rows==0
    assert next(iter(writer.source_cursors.values()))['cursor']['next_row']=='4'
    segments=list(writer.directory.glob('spans-*.parquet'));assert len(segments)==1
    assert pq.read_table(segments[0])['duration_seconds'].to_pylist()==[.125,.02,.03,.04]
    assert pq.ParquetFile(segments[0]).metadata.row_group(0).column(0).compression=='ZSTD'
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
    summary = json.loads((writer.root / commit["summary"]["path"]).read_bytes())
    assert summary["rows"]["resources"] == "2"
    assert summary["closed"] is True
    assert summary["durability"]["maximum_buffer_seconds"] == 15


def test_staging_byte_bound_rejects_before_partial_acceptance(tmp_path: Path) -> None:
    writer = TelemetryWriter(tmp_path / "performance", {}, {}, batch_bytes=1024)
    writer.stage_spans([{"phase": "total", "duration_seconds": 1.0}], source_cursors={})
    with pytest.raises(ValueError, match="byte bound"):
        writer.stage_spans([{"phase": "x" * 1025, "duration_seconds": 2.0}],
                           source_cursors={})
    assert writer.counts["spans"] == 1
    assert writer.staged_source_cursors == {}
    with pytest.raises(ValueError,match="actual P29"):
        writer.stage_spans([],source_cursors={"s":{"offset":1}})
    writer.close()


def test_publication_failure_preserves_durable_cursor_and_retry_has_no_duplicate(tmp_path: Path, monkeypatch) -> None:
    import qcl_negf_results.telemetry as module
    from native_consumer_fixture import diagnostic_parts
    first,second=diagnostic_parts(tmp_path/'producer',(1.,2.))
    writer=TelemetryWriter(tmp_path/'performance',{}, {})
    writer.stage_phase_batch(first);writer.flush();writer.stage_phase_batch(second)
    publish=module.publish_pointer
    def fail(*args,**kwargs):raise OSError('pointer interrupted')
    monkeypatch.setattr(module,'publish_pointer',fail)
    with pytest.raises(OSError,match='interrupted'):writer.flush()
    assert next(iter(writer.source_cursors.values()))['cursor']['next_row']=='1'
    monkeypatch.setattr(module,'publish_pointer',publish)
    commit=json.loads(writer.flush().read_bytes())
    assert commit['source_cursors'][0]['cursor']['next_row']=='2'
    artifacts=catalog_artifacts(writer.directory,writer.catalog_watermark)
    rows=[r for a in artifacts if a.media_type=='application/vnd.apache.parquet' for r in pq.read_table(writer.directory/a.path).to_pylist()]
    assert [r['duration_seconds'] for r in rows]==[1.,2.]
    writer.close()


def test_live_export_is_exact_committed_prefix_and_barrier_includes_tail(tmp_path: Path) -> None:
    from native_consumer_fixture import consumer_writer,consumer_export,diagnostic_parts
    context,writer=consumer_writer(tmp_path/'input')
    first,second=diagnostic_parts(tmp_path/'producer',(1.,2.))
    writer.stage_phase_batch(first);writer.flush();writer.stage_phase_batch(second)
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
    assert values(consumer_export(context,writer,tmp_path/"exports",observe=True)) == [1.0]
    writer.flush()  # Scheduled export/terminal/checkpoint barrier.
    assert values(consumer_export(context,writer,tmp_path/"exports",observe=True)) == [1.0, 2.0]
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
    from native_consumer_fixture import diagnostic_phase,sealed_phase
    from qcl_negf_results.capture import PhaseCaptureSource
    from qcl_negf_results.state import _IOBudget
    from result_v2_fixture import IDENTITY
    literal=diagnostic_phase(tmp_path/'producer',IDENTITY)
    descriptor=dict(status='available',session_id='native',current_path='phase-current.json',reason=None)
    with PhaseCaptureSource(tmp_path/'producer',descriptor=descriptor,expected_identity=IDENTITY,route_epoch=0) as source:
        batch=source.harvest(None,budget=_IOBudget(1024*1024))
        assert batch.segment==literal['segment']
        assert batch.segment_bytes==(tmp_path/'producer'/literal['segment_ref']['path']).read_bytes()
        events=batch.segment['rows'][batch.first_row:batch.stop_row]
        def metric(row,name):return next(m for m in row['metrics'] if m['name']==name)
        assert len(events)==2 and all(r['action']=='phase_end' for r in events)
        assert sum(metric(r,'duration_seconds')['value'] for r in events)==5
        measured={}
        for name,expected in (('cpu_seconds',8),('gc_seconds',.25),('allocated_bytes',1024)):
            observed=[r for r in events if metric(r,name)['measurement_status']=='measured']
            assert len(observed)==1 and sum(metric(r,name)['value'] for r in observed)==expected
            assert sum(metric(r,'duration_seconds')['value'] for r in observed)==2
            unavailable=[r for r in events if metric(r,name)['measurement_status']=='not_measured']
            assert len(unavailable)==1 and metric(unavailable[0],name)['value'] is None
            assert metric(unavailable[0],name)['reason']=='counter_unavailable'
        writer=TelemetryWriter(tmp_path/'performance',{},{});writer.stage_phase_batch(batch)
        writer.record_control({'event':'resource_samples_coalesced','samples_dropped':4})
        commit=json.loads(writer.close().read_bytes());summary=json.loads((writer.root/commit['summary']['path']).read_bytes())
        phase=summary['producer_phase_aggregates'][0]['aggregate']
        assert phase['call_count']=='2'
        counters={m['name']:m for m in phase['metrics']}
        assert counters['process_cpu_seconds']['sum']==8 and counters['duration_seconds']['sum']==5
        assert counters['process_cpu_seconds']['measured_calls']=='1' and counters['process_cpu_seconds']['unavailable_calls']=='1'
        # No admission-wait measurement exists: unavailable, never measured zero.
        assert next((m['sum'] for m in phase['metrics'] if m['name']=='admission_wait_seconds'),None) is None
        assert summary['totals']['collector_dropped_rows']=='4'
        assert summary['control_events']==[dict(event='resource_samples_coalesced',count='1')]
        assert all('measured_wall_seconds' not in m for m in phase['metrics'])
    normal=sealed_phase(tmp_path/'normal',IDENTITY)
    with PhaseCaptureSource(tmp_path/'normal',descriptor=descriptor,expected_identity=IDENTITY,route_epoch=0) as source:
        aggregate=source.harvest(None,budget=_IOBudget(1024*1024))
        assert aggregate.segment['mode']=='normal'
        assert all('cpu_measured_wall_seconds' not in r for r in aggregate.segment['rows'])
        assert source.correlation_capabilities()['per_metric_measured_wall']==dict(available=False,reason='correlation_not_recorded')


def test_retained_rss_peak_reaches_summary_after_current_counter_disappears(tmp_path: Path) -> None:
    writer = TelemetryWriter(tmp_path / "performance", {}, {})
    writer.record_resource({"wall_seconds": 1.0, "rss_bytes": None,
                            "peak_rss_bytes": 12345, "missing_reason": "process_counter_unavailable"})
    commit = json.loads(writer.close().read_bytes())
    summary = json.loads((writer.root / commit["summary"]["path"]).read_bytes())
    assert summary["max_observed_rss_bytes"] == "12345"
    assert summary["cpu_coverage"]["allocations"] == []


def test_T10_writer_commits_only_published_technical_envelope_with_integer_transport(tmp_path):
    from qcl_negf_contracts.telemetry import validate_telemetry_commit
    from qcl_negf_contracts.artifacts import validate_commit
    from qcl_negf_contracts.messages import ContractError
    transport=dict(selection=dict(calcjob_uuid='job3',execution_id='ex',attempt=3,
        plan_fingerprint='a'*64),frozen_plan=None)
    writer=TelemetryWriter(tmp_path,{}, {},transport=transport)
    writer.record_resource(dict(allocation_id='alloc',execution_id='ex',pid=42,
        process_start_ticks=7,monotonic_ns=0,wall_seconds=0.,cpu_time_seconds=1.,cpu_count=2,
        cgroup_scope='root_julia_process_including_threads'))
    writer.record_resource(dict(allocation_id='alloc',execution_id='ex',pid=42,
        process_start_ticks=7,monotonic_ns=2_000_000_000,wall_seconds=2.,cpu_time_seconds=4.,cpu_count=2,
        cgroup_scope='root_julia_process_including_threads'))
    path=writer.close();commit=json.loads(path.read_bytes())
    summary=json.loads((writer.root/commit['summary']['path']).read_bytes())
    assert commit['schema']=='qcl-negf.telemetry-commit.v1'
    assert commit['transport']['selection']['attempt']==3
    assert type(commit['transport']['selection']['attempt']) is int
    assert isinstance(commit['generation'],str)
    prior_path=writer.root/commit['previous']['path']
    prior=json.loads(prior_path.read_bytes())
    prior_summary=json.loads((writer.root/prior['summary']['path']).read_bytes())
    validate_telemetry_commit(commit,summary=summary,summary_ref=commit['summary'],
        previous=prior,previous_ref=commit['previous'],previous_summary=prior_summary,
        previous_summary_ref=prior['summary'])
    with pytest.raises(ContractError):validate_commit(commit)
    cpu=summary['cpu_coverage']['allocations'][0]
    assert (cpu['cpu_seconds'],cpu['allocated_cpu_seconds'],cpu['fraction_of_observed_capacity'])==(3.,4.,.75)
    assert summary['coverage']['scientific_acceptance']=='not_assessed'
    assert summary['observer_self_cost']['physical_read_bytes']['value'] is None
    assert writer.close()==path


def test_T10_phase_rows_and_exact_accepted_prefix_share_one_technical_commit(tmp_path):
    from native_consumer_fixture import sealed_phase
    from result_v2_fixture import IDENTITY
    from qcl_negf_results.capture import PhaseCaptureSource,PhaseCaptureLimits
    from qcl_negf_results._capture_io import ObserverIO
    literal=sealed_phase(tmp_path/'producer',IDENTITY)
    source=PhaseCaptureSource(tmp_path/'producer',descriptor=dict(status='available',session_id='native',current_path='phase-current.json',reason=None),
        expected_identity=IDENTITY,route_epoch=0,limits=PhaseCaptureLimits(maximum_rows=1))
    writer=TelemetryWriter(tmp_path/'performance',{},{});budget=ObserverIO(1024*1024)
    first=source.harvest(None,budget=budget);writer.stage_phase_batch(first)
    assert writer.source_cursors=={}
    old=json.loads(writer.flush().read_bytes());summary=json.loads((writer.root/old['summary']['path']).read_bytes())
    assert old['accepted_ranges'][0]['first_row']=='0' and old['accepted_ranges'][0]['stop_row']=='1'
    assert old['source_cursors'][0]['cursor']['last_sequence']=='8'
    assert summary['totals']['accepted_source_rows']=='1' and summary['totals']['accepted_aggregate_calls']=='2'
    assert summary['totals']['coalesced_source_events']=='0'
    assert summary['producer_phase_aggregates'][0]['aggregate']['metrics'][0]['sum']==4.5
    second=source.harvest(first.cursor,budget=budget);writer.stage_phase_batch(second)
    commit=json.loads(writer.close().read_bytes());summary=json.loads((writer.root/commit['summary']['path']).read_bytes())
    assert commit['accepted_ranges'][0]['first_row']=='1' and commit['accepted_ranges'][0]['stop_row']=='2'
    assert commit['source_cursors'][0]['cursor']['last_sequence']=='8'
    assert summary['totals']['accepted_source_rows']=='2' and summary['totals']['accepted_aggregate_calls']=='3'
    assert summary['totals']['coalesced_source_events']=='6'
    assert summary['rows']['spans']=='0'  # normal aggregate never becomes a fake per-call span
    raw=writer.root/commit['accepted_ranges'][0]['retained_segment']['path']
    assert raw.read_bytes()==first.segment_bytes==second.segment_bytes
    assert len(list((writer.directory/'raw').glob('*segment*')))==1
    assert literal['segment']['rows'][1]['last_sequence']=='6'
    source.close()


def test_T10_resume_preserves_closed_totals_and_never_bridges_old_cpu_endpoint(tmp_path):
    from result_v2_fixture import reference
    writer=TelemetryWriter(tmp_path,{}, {})
    for stamp,cpu in ((0,1.),(2_000_000_000,4.)):
        writer.record_resource(dict(monotonic_ns=stamp,wall_seconds=stamp/1e9,cpu_time_seconds=cpu,cpu_count=2,
            allocation_id='alloc',execution_id='ex',pid=42,process_start_ticks=7,cgroup_scope='root_julia_process_including_threads'))
    closed=writer.close();raw=closed.read_bytes();commit=json.loads(raw)
    cref=reference(tmp_path,str(closed.relative_to(tmp_path)),'qcl-negf.telemetry-commit.v1')
    resumed=TelemetryWriter.resume(tmp_path,commit_ref=cref,expected_session=commit['session'],hardware={})
    resumed.record_resource(dict(monotonic_ns=10_000_000_000,wall_seconds=10.,cpu_time_seconds=100.,cpu_count=2,
        allocation_id='alloc',execution_id='ex',pid=42,process_start_ticks=7,cgroup_scope='root_julia_process_including_threads'))
    middle=json.loads(resumed.flush().read_bytes());summary=json.loads((tmp_path/middle['summary']['path']).read_bytes())
    assert summary['cpu_totals']['cpu_seconds']['value']==3.
    assert summary['cpu_totals']['valid_intervals']=='1'
    assert summary['rows']['resources']=='1' and summary['totals']['rows']['resources']=='3'
    resumed.record_resource(dict(monotonic_ns=12_000_000_000,wall_seconds=12.,cpu_time_seconds=103.,cpu_count=2,
        allocation_id='alloc',execution_id='ex',pid=42,process_start_ticks=7,cgroup_scope='root_julia_process_including_threads'))
    final=json.loads(resumed.close().read_bytes());summary=json.loads((tmp_path/final['summary']['path']).read_bytes())
    assert summary['cpu_totals']['cpu_seconds']['value']==6.
    assert summary['cpu_totals']['allocated_cpu_seconds']['value']==8.
    assert summary['coverage']['incomplete'] is True
    assert summary['coverage']['gaps'][0]['kind']=='collector_restart'
    assert summary['session']['source']['session_id']!=commit['session']['source']['session_id']
    assert summary['baseline']['commit']==cref
    assert closed.read_bytes()==raw


def test_T10_explicit_collector_page_rotation_admits_seventeenth_source_and_exact_older_return(tmp_path):
    import copy
    from native_consumer_fixture import sealed_phase
    from result_v2_fixture import IDENTITY
    from qcl_negf_results.capture import PhaseCaptureSource
    from qcl_negf_results.state import _IOBudget
    producer=tmp_path/'producer';writer=TelemetryWriter(tmp_path/'technical',{}, {})
    accepted=None;first_batch=None;first_literal=None
    def batch(number,*,previous=None,generation=1,cursor=None):
        identity=dict(IDENTITY,point_id='point-'+str(number));root=producer/str(number)
        literal=sealed_phase(root,identity,previous=previous,generation=generation)
        with PhaseCaptureSource(root,descriptor=dict(status='available',session_id='native',current_path='phase-current.json',reason=None),
                expected_identity=identity,route_epoch=0) as source:
            return source.harvest(cursor,budget=_IOBudget(1024*1024)),literal
    for number in range(16):
        source,literal=batch(number);writer.stage_phase_batch(source)
        result=json.loads(writer.flush().read_bytes())
        if number==0:accepted=copy.deepcopy(result['accepted_ranges'][0]);first_batch=source;first_literal=literal
    seventeenth,_=batch(16);before=(writer.generation,copy.deepcopy(writer.source_cursors),dict(writer.accepted_totals))
    with pytest.raises(ValueError,match='page full'):writer.stage_phase_batch(seventeenth)
    assert (writer.generation,writer.source_cursors,writer.accepted_totals)==before
    closed=writer.rotate();old=json.loads(closed.read_bytes())
    assert old['closed'] is True and len(old['source_cursors'])==16
    assert writer.source_cursors=={} and not writer.closed
    writer.stage_phase_batch(seventeenth);writer.flush()
    assert writer.accepted_totals['accepted_source_rows']==34
    second_closed=writer.rotate();assert second_closed!=closed
    writer.record_resource(dict(wall_seconds=1.));writer.flush()  # no old source in predecessor
    returning,_=batch(0,previous=first_literal['publication_ref'],generation=2,cursor=first_batch.cursor)
    old_completion=dict(publication_ref=accepted['publication'],segment_ref=accepted['segment'],
        retained_publication_ref=accepted['retained_publication'],retained_segment_ref=accepted['retained_segment'])
    from result_v2_fixture import reference
    resume=dict(commit_ref=reference(writer.root,str(closed.relative_to(writer.root)),'qcl-negf.telemetry-commit.v1'),summary_ref=old['summary'])
    counts=dict(writer.accepted_totals)
    with pytest.raises((ValueError,__import__('qcl_negf_contracts.messages',fromlist=['ContractError']).ContractError)):
        writer.stage_phase_batch(returning)
    assert writer.accepted_totals==counts
    writer.stage_phase_batch(returning,phase_context_refs=(old_completion,),resume_context_refs=(resume,))
    final=json.loads(writer.close().read_bytes());summary=json.loads((writer.root/final['summary']['path']).read_bytes())
    assert summary['totals']['accepted_source_rows']=='36' and len(final['source_cursors'])==1
    assert final['source_cursors'][0]['cursor']['next_row']=='2'


def test_T10_new_publication_requires_completed_old_physical_rows_and_forgeries_change_no_cursor(tmp_path):
    import copy
    from native_consumer_fixture import sealed_phase
    from result_v2_fixture import IDENTITY
    from qcl_negf_results.capture import PhaseCaptureSource,PhaseCaptureLimits
    from qcl_negf_results.state import _IOBudget
    from qcl_negf_contracts.messages import ContractError
    producer=tmp_path/'producer';literal=sealed_phase(producer,IDENTITY)
    descriptor=dict(status='available',session_id='native',current_path='phase-current.json',reason=None)
    with PhaseCaptureSource(producer,descriptor=descriptor,expected_identity=IDENTITY,route_epoch=0,
            limits=PhaseCaptureLimits(maximum_rows=1)) as source:
        first=source.harvest(None,budget=_IOBudget(1024*1024))
        tail=source.harvest(first.cursor,budget=_IOBudget(1024*1024))
    writer=TelemetryWriter(tmp_path/'technical',{},{});writer.stage_phase_batch(first);writer.flush()
    second=sealed_phase(producer,IDENTITY,previous=literal['publication_ref'],generation=2)
    with PhaseCaptureSource(producer,descriptor=descriptor,expected_identity=IDENTITY,route_epoch=0) as source:
        next_batch=source.harvest(tail.cursor,budget=_IOBudget(1024*1024))
    before=copy.deepcopy(writer.source_cursors);totals=copy.deepcopy(writer.accepted_totals)
    with pytest.raises((ValueError,ContractError)):
        writer.stage_phase_batch(next_batch)
    assert writer.source_cursors==before and writer.accepted_totals==totals
    writer.stage_phase_batch(tail);writer.flush();phase=copy.deepcopy(next(iter(writer._last_phase_raw.values())));closed=writer.rotate()
    old=json.loads(closed.read_bytes())
    from result_v2_fixture import reference
    resume=dict(commit_ref=reference(writer.root,str(closed.relative_to(writer.root)),'qcl-negf.telemetry-commit.v1'),summary_ref=old['summary'])
    counts=copy.deepcopy(writer.accepted_totals);generation=writer.generation
    bad=copy.deepcopy(resume);bad['summary_ref']['sha256']='f'*64
    with pytest.raises((ValueError,ContractError)):
        writer.stage_phase_batch(next_batch,phase_context_refs=(phase,),resume_context_refs=(bad,))
    forged=copy.deepcopy(phase);forged['retained_segment_ref']['sha256']='f'*64
    with pytest.raises((ValueError,ContractError)):
        writer.stage_phase_batch(next_batch,phase_context_refs=(forged,),resume_context_refs=(resume,))
    assert writer.generation==generation and writer.accepted_totals==counts and writer.source_cursors=={}
    writer.stage_phase_batch(next_batch,phase_context_refs=(phase,),resume_context_refs=(resume,));final=writer.close()
    summary=json.loads((writer.root/json.loads(final.read_bytes())['summary']['path']).read_bytes())
    assert summary['totals']['accepted_source_rows']=='4'


def test_T0_close_context_detaches_exact_durable_close_and_reconciles_lost_ack(tmp_path, monkeypatch):
    """A lost pointer ACK must not select a newer collector or advance twice."""
    import copy
    import hashlib
    import qcl_negf_results.telemetry as module
    writer = TelemetryWriter(tmp_path, {}, {})
    writer.record_resource({'wall_seconds': 1.})
    publish = module.publish_pointer
    def lost_ack(*args, **kwargs):
        publish(*args, **kwargs)
        raise OSError('T0 original lost close ACK')
    with monkeypatch.context() as patch:
        patch.setattr(module, 'publish_pointer', lost_ack)
        with pytest.raises(OSError, match='original lost close ACK'):
            writer.close_context()
    snapshot = writer.close_context()
    ref, phases, resumes = snapshot
    raw = (tmp_path / ref['path']).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == ref['sha256']
    assert str(len(raw)) == ref['bytes']
    assert json.loads(raw)['closed'] is True
    assert json.loads(raw)['generation'] == '2'
    assert phases == () and resumes == ()
    frozen = copy.deepcopy(snapshot)
    ref['sha256'] = 'f' * 64
    other = TelemetryWriter(tmp_path, {}, {})
    other.close()
    assert writer.close_context() == frozen
    assert (tmp_path / frozen[0]['path']).read_bytes() == raw


def test_T0_close_context_preserves_explicit_retained_completion_path(tmp_path):
    """Equivalent raw bytes at another selected path must not rewrite that ref."""
    from native_consumer_fixture import sealed_phase
    from result_v2_fixture import IDENTITY
    from qcl_negf_results.capture import PhaseCaptureSource
    from qcl_negf_results.state import _IOBudget
    producer = tmp_path / 'producer'
    first = sealed_phase(producer, IDENTITY)
    writer = TelemetryWriter(tmp_path / 'technical', {}, {})
    descriptor = dict(status='available', session_id='native', current_path='phase-current.json', reason=None)
    with PhaseCaptureSource(producer, descriptor=descriptor, expected_identity=IDENTITY, route_epoch=0) as source:
        batch = source.harvest(None, budget=_IOBudget(1 << 20))
        writer.stage_phase_batch(batch)
        accepted = json.loads(writer.flush().read_bytes())['accepted_ranges'][0]
        completion = dict(publication_ref=accepted['publication'], segment_ref=accepted['segment'])
        for role in ('publication', 'segment'):
            ref = dict(accepted['retained_' + role])
            raw = (writer.root / ref['path']).read_bytes()
            ref['path'] = 'explicit-' + role + '.json'
            (writer.root / ref['path']).write_bytes(raw)
            completion['retained_' + role + '_ref'] = ref
        writer.record_resource(dict(wall_seconds=1.))
        writer.flush()
        sealed_phase(producer, IDENTITY, previous=first['publication_ref'], generation=2)
        returning = source.harvest(batch.cursor, budget=_IOBudget(1 << 20))
        writer.stage_phase_batch(returning, phase_context_refs=(completion,))
        assert writer.close_context()[1] == (completion,)


def s3_writer(root, **kwargs):
    transport = dict(selection=dict(calcjob_uuid='job', execution_id='ex', attempt=1,
                                   plan_fingerprint='a'*64), frozen_plan=None)
    return TelemetryWriter(root, {}, {}, transport=transport, technical_version=2, **kwargs)


def test_S3_opt_in_genesis_and_refuse_old_root_before_writes(tmp_path):
    from qcl_negf_contracts.telemetry import validate_telemetry_commit, validate_telemetry_catalog
    writer = s3_writer(tmp_path/'new', prior_capture='unimported')
    commit = json.loads(writer.flush().read_bytes())
    summary = json.loads((writer.root/commit['summary']['path']).read_bytes())
    catalog = json.loads((writer.root/commit['catalog']['path']).read_bytes())
    assert commit['schema'] == 'qcl-negf.telemetry-commit.v2'
    assert commit['summary']['schema'] == 'qcl-negf.telemetry-summary.v2'
    assert summary['cumulative']['prior_capture'] == 'unimported'
    assert 'unimported_capture' in summary['cumulative']['coverage']['reasons']
    validate_telemetry_catalog(catalog, expected_transport=writer.transport)
    validate_telemetry_commit(commit, summary=summary, summary_ref=commit['summary'], catalog=catalog)
    before = {str(p.relative_to(writer.root)):p.read_bytes() for p in writer.root.rglob('*') if p.is_file()}
    with pytest.raises(ValueError, match='empty|fresh|owned'):
        s3_writer(writer.root)
    assert before == {str(p.relative_to(writer.root)):p.read_bytes() for p in writer.root.rglob('*') if p.is_file()}
    with pytest.raises(ValueError, match='transport'):
        TelemetryWriter(tmp_path/'missing', {}, {}, technical_version=2)
    assert not (tmp_path/'missing').exists()


def test_S3_resource_window_keeps_previous_cut_and_cumulative_totals(tmp_path):
    writer = s3_writer(tmp_path/'new', batch_rows=8)
    for i in range(24):
        writer.record_resource(dict(wall_seconds=float(i), rss_bytes=2048+i))
        writer.flush()
        writer.retain_window(maximum_retained_bytes=65536, minimum_free_bytes=0, maximum_io_bytes=4194304)
    commit = json.loads((writer.root/writer._previous_ref['path']).read_bytes())
    summary = json.loads((writer.root/commit['summary']['path']).read_bytes())
    assert int(summary['totals']['rows']['resources']) == 24
    assert int(summary['retention']['omitted_resource_rows']) > 0
    assert int(summary['retention']['omitted_resource_rows']) + sum(int(a['rows']) for a in writer._catalog['artifacts'] if a['table']=='resources') == 24
    assert int(summary['cumulative']['max_observed_rss_bytes']['value']) == 2071
    assert (writer.root/commit['previous']['commit']['path']).is_file()
    previous = json.loads((writer.root/commit['previous']['commit']['path']).read_bytes())
    assert (writer.root/previous['catalog']['path']).is_file()
    assert len(list(writer.root.rglob('catalog-*.json'))) <= 3
    assert len(list(writer.root.rglob('commit-*.json'))) <= 3


def test_S3_lost_pointer_ack_reuses_exact_prepared_transaction(tmp_path, monkeypatch):
    import qcl_negf_results.telemetry as module
    writer=s3_writer(tmp_path/'new')
    writer.record_resource(dict(wall_seconds=1.))
    original=module.publish_pointer
    def lost_ack(path, ref):
        original(path, ref)
        raise OSError('lost ACK')
    monkeypatch.setattr(module, 'publish_pointer', lost_ack)
    with pytest.raises(OSError, match='lost ACK'):writer.flush()
    staged={p.name:p.read_bytes() for p in writer.directory.glob('commit-*.json')}
    monkeypatch.setattr(module, 'publish_pointer', original)
    writer.flush()
    assert staged == {p.name:p.read_bytes() for p in writer.directory.glob('commit-*.json')}
    assert writer._previous_summary['totals']['rows']['resources']=='1'
    assert writer._previous['cost_delta']['stop_operation']==writer._previous_summary['cumulative']['cost']['position']['stop_operation']


def test_S3_phase_delta_not_local_replay_across_flush_rotation_and_restart(tmp_path):
    from native_consumer_fixture import diagnostic_parts
    writer=s3_writer(tmp_path/'new',batch_rows=8)
    parts=diagnostic_parts(tmp_path/'source',(2.,3.))
    writer.stage_phase_batch(parts[0]);writer.flush()
    first=writer._previous_summary['cumulative']['phases']
    assert sum(int(r['call_count']) for r in first['entries'])==1
    writer.stage_phase_batch(parts[1]);writer.flush()
    cumulative=writer._previous_summary['cumulative']
    assert sum(int(r['call_count']) for r in cumulative['phases']['entries'])==2
    assert cumulative['phases']['entries'][0]['metrics']['duration_seconds']['known']==5.
    assert int(cumulative['phases']['entries'][0]['metrics']['process_cpu_seconds']['unavailable'])==1
    observer=cumulative['cost']['position']['observer_id']
    writer.rotate()
    assert writer._previous_summary['cumulative']['cost']['position']['observer_id']==observer
    assert writer._previous_summary['producer_phase_aggregates']==[]
    assert sum(int(r['call_count']) for r in writer._previous_summary['cumulative']['phases']['entries'])==2
    writer.record_resource(dict(wall_seconds=1.,rss_bytes=12345));writer.flush()
    old=writer._previous_summary['cumulative']
    resumed=TelemetryWriter.resume(writer.root,commit_ref=writer._previous_ref,expected_session=writer._previous['session'],hardware={},technical_version=2)
    new=resumed._previous_summary['cumulative']
    assert new['cost']['position']['observer_id']!=observer
    assert int(new['cost']['uncovered_restarts'])==1
    assert new['cost']['metrics']==old['cost']['metrics']
    assert new['max_observed_rss_bytes']['value']=='12345'
    assert new['coverage']['incomplete'] and 'collector_restart' in new['coverage']['reasons']
    assert new['phases']==old['phases']


def test_S3_failed_unlink_debt_is_authenticated_and_retried_after_barrier(tmp_path,monkeypatch):
    writer=s3_writer(tmp_path/'new')
    for i in range(3):writer.record_resource(dict(wall_seconds=float(i)));writer.flush()
    original=Path.unlink
    failed=[]
    def deny(path,*args,**kwargs):
        if path.name.startswith('resources-'):
            failed.append(path);raise PermissionError('fixture unavailable unlink')
        return original(path,*args,**kwargs)
    monkeypatch.setattr(Path,'unlink',deny)
    writer.retain_window(maximum_retained_bytes=0,minimum_free_bytes=0,maximum_io_bytes=4194304)
    debts=writer._previous['window_delta']['pending_unlink']
    assert failed and any(ref['path']==str(failed[0].relative_to(writer.root)) for ref in debts)
    assert failed[0].exists()
    monkeypatch.setattr(Path,'unlink',original)
    writer.retain_window(maximum_retained_bytes=0,minimum_free_bytes=0,maximum_io_bytes=4194304)
    assert not failed[0].exists()
    writer.retain_window(maximum_retained_bytes=0,minimum_free_bytes=0,maximum_io_bytes=4194304)
    assert all(ref['path']!=str(failed[0].relative_to(writer.root)) for ref in writer._previous['window_delta']['pending_unlink'])
    assert len(writer._unlinked)<=128


def test_S3_index_failure_keeps_old_files_and_exact_retry(tmp_path,monkeypatch):
    writer=s3_writer(tmp_path/'new')
    writer.record_resource(dict(wall_seconds=1.));writer.flush()
    before={str(p.relative_to(writer.root)):p.read_bytes() for p in writer.root.rglob('*') if p.is_file()}
    original=writer._metadata_write
    def fail_index(path,*args,**kwargs):
        if path==writer.root/'index.json':raise OSError('fixture index barrier')
        return original(path,*args,**kwargs)
    monkeypatch.setattr(writer,'_metadata_write',fail_index)
    with pytest.raises(OSError,match='index barrier'):
        writer.retain_window(maximum_retained_bytes=0,minimum_free_bytes=0,maximum_io_bytes=4194304)
    assert all((writer.root/name).exists() for name in before)
    staged=writer._publication_pending['ref'].copy()
    monkeypatch.setattr(writer,'_metadata_write',original)
    writer.flush()
    assert writer._previous_ref==staged


def test_S3_exact_pins_and_low_space_block_only_dependent_retirement(tmp_path,monkeypatch):
    writer=s3_writer(tmp_path/'new')
    writer.record_resource(dict(wall_seconds=1.));writer.flush()
    protected=tuple(row['ref'] for row in writer._catalog['artifacts'] if row['table']=='resources')
    writer.retain_window(maximum_retained_bytes=0,minimum_free_bytes=0,maximum_io_bytes=4194304,protected_refs=protected)
    assert writer._previous_summary['retention']['pinned_retention_refused']
    assert all((writer.root/ref['path']).exists() for ref in protected)
    generation=writer.generation
    with pytest.raises(ValueError,match='reserve'):
        writer.retain_window(maximum_retained_bytes=0,minimum_free_bytes=2**63,maximum_io_bytes=4194304)
    assert writer.generation==generation
    writer.record_resource(dict(wall_seconds=2.));writer.flush()
    assert writer._previous_summary['totals']['rows']['resources']=='2'


def s3_selected(context,writer):
    from native_consumer_fixture import telemetry_input
    from qcl_negf_results.capture import make_binding
    from qcl_negf_results.state import StateReader
    from qcl_negf_results.performance_views import TelemetrySelection
    from result_v2_fixture import reference
    selected=telemetry_input(context,writer,bind_closed=False)
    with StateReader(context['science'],state_ref=context['state'],expected_identity=context['state']['identity']) as reader:
        value=make_binding(reader,telemetry=selected,destination=writer.root/'close-binding.json',capture_status=dict(
            terminal_tail='unknown',scalar_live_route='unavailable',scalar_live_reason='fixture',performance_status='incomplete',reason='fixture'))
    data=selected.document();data['binding_ref']=reference(writer.root,'close-binding.json',value['schema'])
    return TelemetrySelection(writer.root,canonical_plan_root=selected.canonical_plan_root,**data)


def s3_phase_batch(root,index,*,padding=0,phase=None,extra_metrics=()):
    from native_consumer_fixture import sealed_phase
    from result_v2_fixture import IDENTITY,put,reference
    from qcl_negf_results.capture import PhaseCaptureSource,PhaseCaptureLimits
    from qcl_negf_results.state import _IOBudget
    raw=sealed_phase(root,IDENTITY)
    source=dict(raw['segment']['source'],session_id='source-'+str(index),clock_domain_id='clock-'+str(index))
    segment=dict(raw['segment'],source=source)
    segment['rows'][0]['phase']=phase or 'phase-'+str(index)
    segment['rows'][0]['metrics'].extend(extra_metrics)
    sr=put(root,'segment.json',segment,segment['schema'])
    if padding:
        path=root/sr['path'];path.write_bytes(path.read_bytes()+b' '*padding)
        sr=reference(root,'segment.json',segment['schema'])
    publication=dict(raw['publication'],source=source,segment=sr)
    pr=put(root,'publication.json',publication,publication['schema'])
    current=dict(raw['current'],source=source,publication=pr)
    put(root,'current.json',current,current['schema'])
    with PhaseCaptureSource(root,descriptor=dict(status='available',session_id=source['session_id'],current_path='current.json',reason=None),
            expected_identity=IDENTITY,route_epoch=0,limits=PhaseCaptureLimits(maximum_metadata_bytes=4<<20)) as source:
        return source.harvest(None,budget=_IOBudget(4<<20))


def test_S3_phase_key_overflow_preserves_all_calls_and_missing_metrics(tmp_path):
    writer=s3_writer(tmp_path/'new')
    for i in range(70):
        if i and i%8==0:writer.rotate()
        writer.stage_phase_batch(s3_phase_batch(tmp_path/str(i),i));writer.flush()
        writer.retain_window(maximum_retained_bytes=65536,minimum_free_bytes=0,maximum_io_bytes=4194304)
    phases=writer._previous_summary['cumulative']['phases']
    assert len(phases['entries'])<=64
    assert int(phases['overflow']['call_count'])>0
    assert sum(int(r['call_count']) for r in phases['entries'])+int(phases['overflow']['call_count'])==210
    assert phases['incomplete'] and 'phase_key_limit' in phases['reasons']
    assert all(int(row['metrics']['process_cpu_seconds']['unavailable'])==int(row['call_count']) for row in [*phases['entries'],phases['overflow']])


def test_S3_counter_overflow_and_io_refusal_precede_dependent_mutation(tmp_path):
    writer=s3_writer(tmp_path/'new')
    writer._operation_stop=2**64-1
    with pytest.raises(ValueError,match='overflow'):writer.record_resource(dict(wall_seconds=1.))
    assert writer.pending_rows==0 and writer.counts['resources']==0
    writer._operation_stop=0
    metric=dict(name='allocated_bytes',unit='B',sum=dict(kind='integer',type='Int64',decimal=str(2**63-1)),
                measured_calls='2',unavailable_calls='0',reasons=[])
    for i in range(2):
        writer.stage_phase_batch(s3_phase_batch(tmp_path/str(i),i,phase='fixed',extra_metrics=(metric,)));writer.flush()
    before=(writer.generation,dict(writer.accepted_totals),{str(p.relative_to(writer.root)):p.read_bytes() for p in writer.root.rglob('*') if p.is_file()})
    with pytest.raises(ValueError,match='overflow'):
        writer.stage_phase_batch(s3_phase_batch(tmp_path/'overflow',2,phase='fixed',extra_metrics=(metric,)))
    assert writer.generation==before[0] and writer.accepted_totals==before[1]
    assert before[2]=={str(p.relative_to(writer.root)):p.read_bytes() for p in writer.root.rglob('*') if p.is_file()}
    with pytest.raises(ValueError,match='I/O allowance'):
        writer.retain_window(maximum_retained_bytes=0,minimum_free_bytes=0,maximum_io_bytes=0)
    assert writer.generation==before[0]


def test_S3_resume_ignores_unpublished_crash_orphans(tmp_path,monkeypatch):
    import qcl_negf_results.telemetry as module
    writer=s3_writer(tmp_path/'new');selected=writer._previous_ref.copy();session=writer._previous['session'].copy()
    writer.record_resource(dict(wall_seconds=1.))
    original=module.publish_pointer
    def before_pointer(*args,**kwargs):raise OSError('fixture crash before publication')
    monkeypatch.setattr(module,'publish_pointer',before_pointer)
    with pytest.raises(OSError,match='before publication'):writer.flush()
    orphan=json.loads((writer.directory/'commit-000002.json').read_bytes())
    assert orphan['previous']['commit']==selected
    monkeypatch.setattr(module,'publish_pointer',original)
    # The fixture has quiesced the old writer; no shared-owner claim is made.
    resumed=TelemetryWriter.resume(writer.root,commit_ref=selected,expected_session=session,hardware={},technical_version=2)
    assert resumed._previous_summary['totals']['rows']['resources']=='0'
    assert resumed._previous['previous']['commit']==selected
    assert resumed._previous_ref['path']!=str((writer.directory/'commit-000002.json').relative_to(writer.root))
    assert resumed._previous_summary['cumulative']['cost']['uncovered_restarts']=='1'


@pytest.mark.parametrize('target',['raw','previous'])
def test_S3_late_retained_change_prevents_pointer_publication(tmp_path,target):
    from qcl_negf_contracts.messages import ContractError
    writer=s3_writer(tmp_path/'new')
    pointer=(writer.root/'current.json').read_bytes()
    if target=='raw':
        writer.stage_phase_batch(s3_phase_batch(tmp_path/'source',0))
        path=writer.root/writer.accepted_ranges[0]['retained_segment']['path']
    else:
        writer.record_resource(dict(wall_seconds=1.))
        path=writer.root/writer._previous_ref['path']
    path.write_bytes(path.read_bytes()+b'\n')
    with pytest.raises((ValueError,ContractError)):
        writer.flush()
    assert (writer.root/'current.json').read_bytes()==pointer


def test_S3_resume_partial_physical_source(tmp_path):
    from native_consumer_fixture import diagnostic_parts
    writer=s3_writer(tmp_path/'new')
    parts=diagnostic_parts(tmp_path/'source',(2.,3.))
    writer.stage_phase_batch(parts[0]);writer.flush()
    resumed=TelemetryWriter.resume(writer.root,commit_ref=writer._previous_ref,expected_session=writer._previous['session'],hardware={},technical_version=2)
    resumed.stage_phase_batch(parts[1]);resumed.flush()
    assert resumed._previous_summary['totals']['accepted_source_rows']=='2'
    assert resumed._previous_summary['cumulative']['phases']['entries'][0]['metrics']['duration_seconds']['known']==5.
