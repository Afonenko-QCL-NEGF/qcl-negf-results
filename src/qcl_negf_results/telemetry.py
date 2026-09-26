"""Bounded typed batches; only closed Parquet segments enter a durable commit."""
from __future__ import annotations

from copy import deepcopy
import fcntl
import hashlib
import math
import os
from pathlib import Path
import platform
import threading
import time
from typing import Any, Mapping
import uuid

import pyarrow as pa
import pyarrow.parquet as pq

from qcl_negf_contracts.artifacts import COMMIT_SCHEMA, CONTRACT_SET
from qcl_negf_contracts.telemetry import TABLE_FIELDS, TELEMETRY_SCHEMA
from .commits import (artifact_row, atomic_write, fsync_directory, json_bytes,
                      publish_pointer, read_json)
from .catalog import CATALOG_SCHEMA

_ARROW_TYPES = {"string": pa.string(), "int64": pa.int64(),
                "float64": pa.float64(), "bool": pa.bool_()}
_REQUIRED = {"spans": {"phase", "duration_seconds"},
             "resources": {"wall_seconds"}, "control_events": {"event"}}


class CpuCoverage:
    """Integrate stable cumulative counters independently of sample resolution.

    A slow observer loses temporal detail, not the CPU time accumulated between
    its endpoints. Missing counters, changed identities and capacity changes are
    gaps; their utilization remains unknown. Profile samples are retained in the
    raw table but excluded here to avoid double-counting the baseline stream.
    """
    RESOLUTION_TARGET_SECONDS = 10.0
    IDENTITY_FIELDS = ("execution_id", "pid", "process_start_ticks", "clock_domain_id",
                       "cpu_count", "cpu_quota_cores", "cgroup_path", "cgroup_scope")

    def __init__(self) -> None:
        self.rows: dict[str, dict[str, Any]] = {}
        self.previous: dict[str, dict[str, Any]] = {}
        self.unidentified_samples = 0
        self.profile_samples_excluded = 0

    def record(self, value: Mapping[str, Any]) -> None:
        if value.get("sampling_mode") == "profile_window":
            self.profile_samples_excluded += 1
            return
        allocation = value.get("allocation_id")
        if not allocation:
            self.unidentified_samples += 1
            return
        stamp = value.get("monotonic_ns")
        row = self.rows.setdefault(allocation, {
            "allocation_id": allocation, "execution_id": value.get("execution_id"),
            "samples": 0, "valid_intervals": 0, "missing_samples": 0,
            "discontinuous_intervals": 0, "covered_wall_seconds": 0.0,
            "cpu_seconds": 0.0, "allocated_cpu_seconds": 0.0,
            "observed_capacity_cpu_seconds": 0.0, "coarse_intervals": 0,
            "coarse_wall_seconds": 0.0, "fine_resolution_wall_seconds": 0.0,
            "maximum_observed_interval_seconds": None,
            "first_sample_monotonic_ns": stamp, "last_sample_monotonic_ns": stamp,
        })
        row["samples"] += 1
        if stamp is not None:
            if row["first_sample_monotonic_ns"] is None:
                row["first_sample_monotonic_ns"] = stamp
            row["last_sample_monotonic_ns"] = stamp
        previous = self.previous.pop(allocation, None)
        if (value.get("cpu_time_seconds") is None or stamp is None
                or not value.get("cpu_count") or value.get("missing_reason")):
            row["missing_samples"] += 1
            return
        self.previous[allocation] = dict(value)
        if previous is None:
            return
        wall = (stamp - previous["monotonic_ns"]) / 1e9
        cpu = value["cpu_time_seconds"] - previous["cpu_time_seconds"]
        if wall <= 0:
            self.previous[allocation] = previous
            row["last_sample_monotonic_ns"] = previous["monotonic_ns"]
            row["discontinuous_intervals"] += 1
            return
        if (cpu < 0 or any(value.get(name) != previous.get(name)
                           for name in self.IDENTITY_FIELDS)):
            row["discontinuous_intervals"] += 1
            return
        row["valid_intervals"] += 1
        row["covered_wall_seconds"] += wall
        row["cpu_seconds"] += cpu
        row["allocated_cpu_seconds"] += wall * value["cpu_count"]
        quota = value.get("cpu_quota_cores")
        effective_width = min(value["cpu_count"], quota) if quota is not None else value["cpu_count"]
        row["observed_capacity_cpu_seconds"] += wall * effective_width
        row["maximum_observed_interval_seconds"] = max(
            row["maximum_observed_interval_seconds"] or 0.0, wall)
        if wall > self.RESOLUTION_TARGET_SECONDS:
            row["coarse_intervals"] += 1
            row["coarse_wall_seconds"] += wall
        else:
            row["fine_resolution_wall_seconds"] += wall

    def summary(self) -> dict[str, Any]:
        allocations = []
        for value in self.rows.values():
            row = dict(value)
            wall = row["covered_wall_seconds"]
            capacity = row["allocated_cpu_seconds"]
            effective = row["observed_capacity_cpu_seconds"]
            first, last = row["first_sample_monotonic_ns"], row["last_sample_monotonic_ns"]
            observed = (last - first) / 1e9 if first is not None and last is not None else None
            row["observed_window_seconds"] = observed
            row["fraction_of_observed_window_covered"] = wall / observed if observed and observed > 0 else None
            row["mean_busy_logical_cpus"] = row["cpu_seconds"] / wall if wall else None
            row["fraction_of_allocated_cpu"] = row["cpu_seconds"] / capacity if capacity else None
            row["fraction_of_observed_capacity"] = row["cpu_seconds"] / effective if effective else None
            allocations.append(row)
        return {
            "schema": "qcl-negf.cpu-coverage.v2", "allocations": allocations,
            "unidentified_samples": self.unidentified_samples,
            "profile_samples_excluded": self.profile_samples_excluded,
            "scope": "baseline process-group CPU deltas / allocated logical CPU time; not host utilization",
            "maximum_interval_seconds": None,
            "temporal_resolution_target_seconds": self.RESOLUTION_TARGET_SECONDS,
            "long_interval_policy": "retain stable counter deltas as coarse interval averages",
            "phase_attribution": "not inferred from resource endpoints; use producer phase counters",
            "observed_capacity_scope": "minimum of allocation width and observed node quota; ancestor quotas may be unavailable",
            "missing_is_zero": False,
            "intervals_overlap_across_allocations": True,
        }


class TelemetryWriter:
    """Synchronous bounded flush outside numerical hot loops; no sample dropping.

    Multiple writers use isolated sessions. The short index update is serialized;
    numerical work and Parquet encoding never hold the index lock.
    """
    def __init__(self, root: Path, identity: Mapping[str, Any], hardware: Mapping[str, Any],
                 *, batch_rows: int = 256, batch_bytes: int = 4 * 1024 * 1024,
                 max_buffer_seconds: float = 15.0) -> None:
        if batch_rows < 1 or batch_rows > 65_536:
            raise ValueError("batch_rows must be in 1..65536")
        if not isinstance(batch_bytes, int) or batch_bytes < 1024:
            raise ValueError("batch_bytes must be an integer >=1024")
        if not math.isfinite(max_buffer_seconds) or not 0 < max_buffer_seconds <= 60:
            raise ValueError("max_buffer_seconds must be finite and in (0,60]")
        self.root = Path(root)
        self.session_id = uuid.uuid4().hex
        self.directory = self.root / "sessions" / self.session_id
        self.directory.mkdir(parents=True)
        self.identity = dict(identity)
        self.batch_rows = batch_rows
        self.batch_bytes = batch_bytes
        self.max_buffer_seconds = float(max_buffer_seconds)
        self.pending_bytes = 0
        self.pending_since_ns: int | None = None
        self.dirty = False
        self.lock = threading.RLock()
        self.buffers: dict[tuple[str, bool], list[dict[str, Any]]] = {}
        self.artifacts: list[dict[str, Any]] = []
        self.catalog_watermark: dict[str, Any] | None = None
        self.artifact_count = 0
        self.generation = 0
        self.closed = False
        self.counts = {name: 0 for name in TABLE_FIELDS}
        self.next_sequence = {name: 0 for name in TABLE_FIELDS}
        self.max_rss_bytes: int | None = None
        self.cpu_coverage = CpuCoverage()
        self.phase_seconds: dict[str, float] = {}
        self.control_counts: dict[str, int] = {}
        self.source_cursors: dict[str, Any] = {}
        self.staged_source_cursors: dict[str, Any] = {}
        self.phase_calls: dict[str, int] = {}
        self.producer_phase_seconds: dict[str, float] = {}
        self.producer_phase_calls: dict[str, int] = {}
        self.producer_phase_measurements: dict[str, dict[str, Any]] = {}
        self.samples_dropped = 0
        self.warmup_seconds: dict[str, float] = {}
        self.warmup_calls: dict[str, int] = {}
        self.measurements: dict[str, dict[str, float | int]] = {}
        self.utc_anchor_ns = time.time_ns()
        self.monotonic_anchor_ns = time.monotonic_ns()
        self.window_buckets: dict[str, int] = {}
        self.transition_windows: set[str] = {"resources", "spans"}
        passport = {"schema": TELEMETRY_SCHEMA, "contract_set": CONTRACT_SET, "identity": self.identity,
            "session_id": self.session_id, "utc_anchor_ns": self.utc_anchor_ns,
            "monotonic_anchor_ns": self.monotonic_anchor_ns,
            "python": platform.python_version(), "hardware": dict(hardware),
            "missing_hardware_fields": sorted({"cpu_model", "topology", "affinity",
                "cpu_quota", "julia_threads", "blas_threads", "gc_threads",
                "compile_target", "libraries"} - set(hardware)),
            "time_semantics": "monotonic nanoseconds; durations and CPU time in seconds",
            "sampling": "all received rows retained; absent measurements are null"}
        atomic_write(self.directory / "hardware.json", json_bytes(passport), immutable=True)
        self.artifacts.append(artifact_row(self.directory / "hardware.json",
            relative="hardware.json", role="performance.summary", media_type="application/json"))
        self._publish()
        self._register()

    def _register(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        with (self.root / ".index.lock").open("a+b") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            index_path = self.root / "index.json"
            index: dict[str, Any] = read_json(index_path)[0] if index_path.exists() else {
                "schema": "qcl-negf.performance-index.v2", "contract_set": CONTRACT_SET, "sessions": []}
            if (index.get("schema") != "qcl-negf.performance-index.v2"
                    or index.get("contract_set") != CONTRACT_SET):
                raise ValueError("incompatible_contract: performance index")
            index["sessions"].append({"id": self.session_id,
                "pointer": f"sessions/{self.session_id}/current.json"})
            atomic_write(index_path, json_bytes(index))

    def _normalise(self, table: str, row: Mapping[str, Any]) -> dict[str, Any]:
        fields = dict(TABLE_FIELDS[table])
        unknown = set(row) - fields.keys()
        if unknown:
            raise ValueError(f"unknown {table} fields: {sorted(unknown)}")
        if _REQUIRED[table] - row.keys():
            raise ValueError(f"missing {table} fields: {sorted(_REQUIRED[table] - row.keys())}")
        if any(row[name] is None for name in _REQUIRED[table]):
            raise ValueError(f"required {table} fields cannot be null")
        value: dict[str, Any] = {name: row.get(name) for name in fields}
        value["session_id"] = self.session_id
        value["sample_sequence"] = self.next_sequence[table]
        self.next_sequence[table] += 1
        value["clock_domain_id"] = row.get("clock_domain_id", f"collector:{self.session_id}")
        value["utc_anchor_ns"] = row.get("utc_anchor_ns", self.utc_anchor_ns)
        value["monotonic_anchor_ns"] = row.get("monotonic_anchor_ns", self.monotonic_anchor_ns)
        value["monotonic_ns"] = row.get("monotonic_ns", time.monotonic_ns())
        for name in ("worker_id", "execution_id", "point_id", "allocation_id"):
            value[name] = row.get(name, self.identity.get(name))
        if table == "spans":
            value["timing_kind"] = row.get("timing_kind", "inclusive")
            value["call_count"] = row.get("call_count", 1)
            value["warmup"] = row.get("warmup", False)
            value["measurement_kind"] = row.get("measurement_kind", "phase_duration")
            value["timestamp_kind"] = row.get("timestamp_kind", "measurement")
            if value["timing_kind"] not in {"inclusive", "exclusive"}:
                raise ValueError("timing_kind must be inclusive or exclusive")
            if value["measurement_kind"] not in {"phase_duration", "interval", "delay", "phase_boundary"}:
                raise ValueError("invalid measurement_kind")
        for name, kind in fields.items():
            item = value[name]
            if item is None:
                continue
            if kind == "string" and not isinstance(item, str):
                raise ValueError(f"{name} must be a string")
            if kind == "int64" and (not isinstance(item, int) or isinstance(item, bool) or item < 0 or item >= 2**63):
                raise ValueError(f"{name} must be a nonnegative int64")
            if kind == "float64" and (not isinstance(item, (float, int)) or isinstance(item, bool)
                                      or not math.isfinite(item) or item < 0):
                raise ValueError(f"{name} must be a finite nonnegative number")
            if kind == "bool" and not isinstance(item, bool):
                raise ValueError(f"{name} must be boolean")
        return value

    def _record(self, table: str, row: Mapping[str, Any], science_window: bool) -> None:
        with self.lock:
            if self.closed:
                raise RuntimeError("telemetry session is closed")
            value = self._normalise(table, row)
            self._reserve([value])
            self._append(table, value, science_window)

    @property
    def pending_rows(self) -> int:
        return sum(map(len, self.buffers.values()))

    @staticmethod
    def _row_bytes(value: Mapping[str, Any]) -> int:
        # Bound payload bytes independently of row count, including long strings.
        # Python/container overhead is additionally bounded by the fixed columns
        # and batch_rows. Numerical buffers are never downcast or sampled.
        return sum(len(item.encode("utf-8")) if isinstance(item, str) else 8
                   for item in value.values())

    def _reserve(self, values: list[dict[str, Any]]) -> None:
        size = sum(self._row_bytes(value) for value in values)
        if size > self.batch_bytes:
            raise ValueError("ingestion payload exceeds configured byte bound")
        if self.pending_rows + len(values) > self.batch_rows or self.pending_bytes + size > self.batch_bytes:
            self.flush()

    def _mark_pending(self) -> None:
        if self.pending_since_ns is None:
            self.pending_since_ns = time.monotonic_ns()
        self.dirty = True

    def _append(self, table: str, value: dict[str, Any], science_window: bool,
                *, auto_flush: bool = True) -> None:
        self.buffers.setdefault((table, science_window), []).append(value)
        self.pending_bytes += self._row_bytes(value)
        self._mark_pending()
        self.counts[table] += 1
        if table == "resources":
            observed_rss = [value[name] for name in ("rss_bytes", "peak_rss_bytes")
                            if value[name] is not None]
            if observed_rss:
                self.max_rss_bytes = max(self.max_rss_bytes or 0, *observed_rss)
            self.cpu_coverage.record(value)
        if table == "spans":
            self._record_span_summary(value)
        if table == "control_events":
            event = str(value["event"])
            self.control_counts[event] = self.control_counts.get(event, 0) + 1
            self.samples_dropped += value.get("samples_dropped") or 0
        if auto_flush and self.pending_rows >= self.batch_rows:
            self.flush()

    def _record_producer_measurements(self, key: str, value: dict[str, Any], duration: float) -> None:
        self.producer_phase_seconds[key] = self.producer_phase_seconds.get(key, 0.0) + duration
        self.producer_phase_calls[key] = self.producer_phase_calls.get(key, 0) + 1
        measured = self.producer_phase_measurements.setdefault(key, {
            "calls": 0, "duration_seconds": 0.0,
            "cpu_seconds": None, "gc_seconds": None, "allocated_bytes": None,
            "cpu_measured_calls": 0, "gc_measured_calls": 0, "allocations_measured_calls": 0,
            "cpu_measured_wall_seconds": 0.0, "gc_measured_wall_seconds": 0.0,
            "allocations_measured_wall_seconds": 0.0,
        })
        measured["calls"] += 1
        measured["duration_seconds"] += duration
        for source, target, count in (
            ("process_cpu_seconds", "cpu_seconds", "cpu_measured_calls"),
            ("gc_seconds", "gc_seconds", "gc_measured_calls"),
            ("allocated_bytes", "allocated_bytes", "allocations_measured_calls"),
        ):
            if value[source] is not None:
                measured[target] = (measured[target] or 0) + value[source]
                measured[count] += 1
                measured[count.replace("calls", "wall_seconds")] += duration

    def _record_span_summary(self, value: dict[str, Any]) -> None:
        key = f'{value["phase"]}:{value["timing_kind"]}'
        duration = float(value["duration_seconds"])
        if value["measurement_kind"] == "phase_duration":
            self.phase_seconds[key] = self.phase_seconds.get(key, 0.0) + duration
            self.phase_calls[key] = self.phase_calls.get(key, 0) + value["call_count"]
            if value["warmup"]:
                self.warmup_seconds[key] = self.warmup_seconds.get(key, 0.0) + duration
                self.warmup_calls[key] = self.warmup_calls.get(key, 0) + value["call_count"]
        elif value["measurement_kind"] == "phase_boundary":
            if value["event_kind"] == "end":
                self._record_producer_measurements(key, value, duration)
        else:
            aggregate = self.measurements.setdefault(str(value["phase"]), {
                "count": 0, "sum_seconds": 0.0, "max_seconds": 0.0})
            aggregate["count"] += 1
            aggregate["sum_seconds"] += duration
            aggregate["max_seconds"] = max(aggregate["max_seconds"], duration)

    def ingest_spans(self, rows: list[Mapping[str, Any]], *, source_cursors: Mapping[str, Any]) -> Path:
        """Publish source offsets and their exact rows in the same immutable commit.

        Parsing/validation precedes mutation. No intermediate automatic flush can
        acknowledge a source offset before every corresponding row is durable.
        """
        if len(rows) > self.batch_rows:
            raise ValueError("ingestion batch exceeds configured row bound")
        with self.lock:
            if self.closed:
                raise RuntimeError("telemetry session is closed")
            values = [self._normalise("spans", row) for row in rows]
            cursors = deepcopy(dict(source_cursors))
            json_bytes(cursors)
            # Synchronous ingestion retains its one-commit transaction,
            # including an already buffered resource tail (at most 2*batch_rows).
            if sum(self._row_bytes(value) for value in values) + self.pending_bytes > self.batch_bytes:
                raise ValueError("ingestion payload exceeds configured byte bound")
            for value in values:
                self._append("spans", value, False, auto_flush=False)
            if cursors:
                self.staged_source_cursors.update(cursors)
                self._mark_pending()
            return self.flush()

    def stage_spans(self, rows: list[Mapping[str, Any]], *, source_cursors: Mapping[str, Any]) -> None:
        """Accept a replayable source batch without promising immediate durability.

        The caller may advance its in-memory read offset after return. Recovery
        MUST use source_cursors from a published commit, never staged offsets.
        Rows and their offsets become durable together at a flush barrier. The
        caller must invoke flush_due during its polling loop, even with no rows.
        """
        if len(rows) > self.batch_rows:
            raise ValueError("ingestion batch exceeds configured row bound")
        with self.lock:
            if self.closed:
                raise RuntimeError("telemetry session is closed")
            values = [self._normalise("spans", row) for row in rows]
            cursors = deepcopy(dict(source_cursors))
            json_bytes(cursors)
            self._reserve(values)
            for value in values:
                self._append("spans", value, False, auto_flush=False)
            if cursors:
                self.staged_source_cursors.update(cursors)
                self._mark_pending()
            if self.pending_rows >= self.batch_rows:
                self.flush()

    def flush_due(self, *, now_ns: int | None = None) -> Path:
        """Publish aged tails; latency is max_buffer_seconds + collector poll delay."""
        with self.lock:
            now = time.monotonic_ns() if now_ns is None else now_ns
            if self.pending_since_ns is not None and now - self.pending_since_ns >= self.max_buffer_seconds * 1e9:
                return self.flush()
            return self.directory / f"commit-{self.generation:06d}.json"

    def record_span(self, row: Mapping[str, Any], *, science_window: bool = False) -> None:
        self._record("spans", row, science_window)

    def record_resource(self, row: Mapping[str, Any], *, science_window: bool = False) -> None:
        self._record("resources", row, science_window)

    def record_control(self, row: Mapping[str, Any]) -> None:
        # Capture real rows following a transition, with no synthetic zero samples.
        with self.lock:
            if self.closed:
                raise RuntimeError("telemetry session is closed")
            value = self._normalise("control_events", row)
            self._reserve([value])
            self.transition_windows.update({"resources", "spans"})
            self._append("control_events", value, True, auto_flush=False)
            self.flush()  # Critical transitions are durable before returning to admission.

    def flush(self) -> Path:
        return self._flush()

    def _flush(self, *, closing: bool = False) -> Path:
        with self.lock:
            if self.closed:
                raise RuntimeError("telemetry session is closed")
            if not self.dirty and not self.artifacts and not closing:
                return self.directory / f"commit-{self.generation:06d}.json"
            for (table, window), rows in list(self.buffers.items()):
                if not rows:
                    continue
                bucket = (time.monotonic_ns() - self.monotonic_anchor_ns) // 900_000_000_000
                selected_window = window or (table in {"resources", "spans"} and (
                    self.window_buckets.get(table) != bucket or table in self.transition_windows))
                name = f"{table}-{self.artifact_count + len(self.artifacts):09d}.parquet"
                temporary = self.directory / (".writing-" + name)
                schema = pa.schema([pa.field(name, _ARROW_TYPES[kind])
                    for name, kind in TABLE_FIELDS[table]], metadata={
                        b"schema": TELEMETRY_SCHEMA.encode(), b"table": table.encode(),
                        b"contract_set": CONTRACT_SET.encode(),
                        b"units": b"monotonic_ns=ns; *_seconds=s; *_bytes=B"})
                try:
                    pq.write_table(pa.Table.from_pylist(rows, schema=schema), temporary,
                                   compression="zstd", compression_level=3,
                                   use_dictionary=True, write_statistics=True)
                    with temporary.open("rb") as stream:
                        os.fsync(stream.fileno())
                    # A readable footer is necessary before publication.
                    if pq.ParquetFile(temporary).metadata.num_rows != len(rows):
                        raise RuntimeError("Parquet row count mismatch before commit")
                    destination = self.directory / name
                    os.link(temporary, destination)
                    fsync_directory(self.directory)
                    self.artifacts.append(artifact_row(destination, relative=name,
                        role="performance.window" if selected_window else "performance.full",
                        profile="science" if selected_window else "full-state",
                        media_type="application/vnd.apache.parquet"))
                    if selected_window:
                        self.window_buckets[table] = bucket
                        self.transition_windows.discard(table)
                    self.buffers[(table, window)] = []
                    self.pending_bytes -= sum(self._row_bytes(row) for row in rows)
                finally:
                    temporary.unlink(missing_ok=True)
            self.closed = closing
            try:
                result = self._publish()
            except BaseException:
                self.closed = False
                raise
            self.dirty = False
            self.pending_since_ns = None
            return result

    def _publish(self) -> Path:
        committed_cursors = {**self.source_cursors, **self.staged_source_cursors}
        self.generation += 1
        if self.artifacts:
            next_count = self.artifact_count + len(self.artifacts)
            catalog = {"schema": CATALOG_SCHEMA, "contract_set": CONTRACT_SET, "entries": next_count,
                       "previous": self.catalog_watermark, "artifacts": self.artifacts}
            catalog_name = f"catalog-{self.generation:09d}.json"
            catalog_payload = json_bytes(catalog)
            atomic_write(self.directory / catalog_name, catalog_payload, immutable=True)
            self.artifact_count = next_count
            self.catalog_watermark = {"path": catalog_name, "entries": self.artifact_count,
                "sha256": hashlib.sha256(catalog_payload).hexdigest()}
            self.artifacts = []
        summary_name = f"summary-{self.generation:06d}.json"
        summary = {"schema": TELEMETRY_SCHEMA, "contract_set": CONTRACT_SET, "session_id": self.session_id,
            "rows": self.counts, "max_observed_rss_bytes": self.max_rss_bytes,
            "cpu_coverage": self.cpu_coverage.summary(),
            "phase_seconds": self.phase_seconds, "phase_calls": self.phase_calls,
            "producer_phase_seconds": self.producer_phase_seconds,
            "producer_phase_calls": self.producer_phase_calls,
            "producer_phase_measurements": self.producer_phase_measurements,
            "phase_attribution": {
                "source": "producer begin/end boundaries; process-inclusive counters",
                "resource_endpoint_labels_are_not_interval_attribution": True,
                "admission_wait_seconds": None,
                "admission_wait_reason": "not inferred; requires explicit producer wait measurement",
                "missing_counters_are_zero": False,
            },
            "warmup_seconds": self.warmup_seconds, "warmup_calls": self.warmup_calls,
            "intervals_and_delays": self.measurements,
            "control_events": self.control_counts,
            "measurement_coverage": {
                "scope": "received scalar source rows, not all solver wall time",
                "sources": committed_cursors,
                "capture_error_events": self.control_counts.get("performance_capture_incomplete", 0),
                "all_observed_sources_consumed": all(
                    cursor.get("complete_at_observation") is True
                    for cursor in committed_cursors.values()),
                "jit_separately_measured": False,
                "initialization_includes_possible_compilation": True,
                "warmup": "first observed call per phase/source session; not measured JIT",
                "absent_phase": "not measured, never zero",
                "phase_scopes": "inclusive unless explicitly exclusive; do not sum overlapping scopes",
            },
            "phase_sum_is_not_wall_time": True, "samples_dropped": self.samples_dropped,
            "durability": {
                "published_rows": "all rows in this summary and source cursors share this commit",
                "live_rows": "staged rows after this commit are excluded; not durable",
                "source_recovery": "replay source journal from published cursor",
                "resource_tail_recovery": "unpublished resource samples can be lost on process crash",
                "maximum_staged_batch_rows": self.batch_rows,
                "maximum_synchronous_flush_rows": 2 * self.batch_rows,
                "maximum_batch_payload_bytes": self.batch_bytes,
                "maximum_buffer_seconds": self.max_buffer_seconds,
                "latency_bound": "buffer age plus coordinator polling/scheduling delay",
                "parquet_compression": "zstd-3 lossless",
                "barriers": ["flush", "control_transition", "checkpoint", "export", "close"],
            },
            "science_window_policy": {"period_seconds": 900,
                "selection": "first closed batch per table/period and after control transition",
                "maximum_batch_rows": self.batch_rows, "cutoff_tail_segments_per_table": 3,
                "scope": "selected actual rows; not a complete profile"},
            "closed": self.closed, "uncommitted_resource_batch_at_risk": not self.closed,
            "recorded_until_monotonic_ns": time.monotonic_ns()}
        atomic_write(self.directory / summary_name, json_bytes(summary), immutable=True)
        summary_row = artifact_row(self.directory / summary_name, relative=summary_name,
            role="performance.summary", media_type="application/json")
        commit = {"schema": COMMIT_SCHEMA, "contract_set": CONTRACT_SET, "identity": {**self.identity, "session_id": self.session_id},
            "generation": self.generation, "closed": self.closed,
            "published_unix": time.time(),
            "source_cursors": committed_cursors,
            "telemetry_catalog": self.catalog_watermark,
            "artifacts": [summary_row]}
        payload = json_bytes(commit)
        name = f"commit-{self.generation:06d}.json"
        atomic_write(self.directory / name, payload, immutable=True)
        publish_pointer(self.directory / "current.json", name, payload, self.generation)
        self.source_cursors = deepcopy(committed_cursors)
        self.staged_source_cursors.clear()
        return self.directory / name

    def close(self) -> Path:
        with self.lock:
            if self.closed:
                return self.directory / f"commit-{self.generation:06d}.json"
            return self._flush(closing=True)

    def __enter__(self) -> "TelemetryWriter":
        return self

    def __exit__(self, *error: object) -> None:
        self.close()
