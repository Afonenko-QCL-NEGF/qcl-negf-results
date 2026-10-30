"""Bounded typed batches; only closed Parquet segments enter a durable commit."""
from __future__ import annotations

from copy import deepcopy
from copy import copy
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
from functools import wraps

import pyarrow as pa
import pyarrow.parquet as pq

from qcl_negf_contracts.artifacts import CONTRACT_SET
from qcl_negf_contracts import load_schema, schema_validator
from qcl_negf_contracts.messages import encode
from qcl_negf_contracts.telemetry import validate_telemetry_commit, validate_telemetry_summary
from qcl_negf_contracts.telemetry import TABLE_FIELDS, TELEMETRY_SCHEMA
from .commits import (artifact_row, atomic_write, fsync_directory, json_bytes,
                      read_json)
from .catalog import CATALOG_SCHEMA

_ARROW_TYPES = {"string": pa.string(), "int64": pa.int64(),
                "float64": pa.float64(), "bool": pa.bool_()}
_REQUIRED = {"spans": {"phase", "duration_seconds"},
             "resources": {"wall_seconds"}, "control_events": {"event"}}


def publish_pointer(path: Path, reference: Mapping[str, Any]) -> None:
    """Technical pointer only; never use the scientific pointer wire adapter."""
    atomic_write(path,json_bytes(dict(schema=('qcl-negf.telemetry-pointer.v2' if reference['schema'].endswith('.v2') else 'qcl-negf.telemetry-pointer.v1'),
        contract_set=CONTRACT_SET,commit=dict(reference))))


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
                or value.get("missing_reason")):
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
        width = value.get("cpu_count")
        if width is None:
            row["allocated_cpu_seconds"] = None
            row["observed_capacity_cpu_seconds"] = None
            row["capacity_unavailable_reason"] = "actual allocation width unavailable"
        elif row["allocated_cpu_seconds"] is not None:
            row["allocated_cpu_seconds"] += wall * width
            quota = value.get("cpu_quota_cores")
            effective_width = min(width, quota) if quota is not None else width
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
            if not row["valid_intervals"]:
                row["cpu_seconds"] = None
                row["allocated_cpu_seconds"] = None
                row["observed_capacity_cpu_seconds"] = None
                row["cpu_unavailable_reason"] = "no stable measured CPU interval"
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


_UINT64_MAX = 2**64-1
_COST_FLOATS = ('probe_seconds','read_seconds','hash_seconds','parse_validate_seconds',
                'serialize_seconds','parquet_seconds','flush_sync_seconds','process_cpu_seconds')
_COST_METRICS = (*_COST_FLOATS,'bytes_read','bytes_written','physical_read_bytes','physical_write_bytes',
                 'polls','flushes','source_reopens','reverified_bytes')
_PHASE_METRICS = ('duration_seconds','process_cpu_seconds','allocated_bytes','gc_seconds','gc_count')


def _count(value):
    if not 0 <= value <= _UINT64_MAX:raise ValueError('UInt64 counter overflow before publication')
    return str(value)


def _empty_accumulation():
    return dict(known=None,covered='0',unavailable='0',overflow='0',reasons=[])


def _accumulate(left,right,integer=False):
    counts={k:_count(int(left[k])+int(right[k])) for k in ('covered','unavailable','overflow')}
    parts=[r['known'] for r in (left,right) if r['known'] is not None]
    known=None
    if parts:
        known=_count(sum(int(v) for v in parts)) if integer else math.fsum(parts)
        if not integer and not math.isfinite(known):raise ValueError('Float64 subtotal overflow before publication')
    reasons=set(left['reasons'])|set(right['reasons'])
    if any(int(v) for v in counts.values()):reasons.discard('no_observations')
    return dict(known=known,**counts,reasons=sorted(reasons))


def _nodes(value):
    pending=[value];total=0
    while pending:
        node=pending.pop();total+=1
        if isinstance(node,dict):pending.extend(node.values())
        elif isinstance(node,(list,tuple)):pending.extend(node)
    return total


def _observed(method):
    @wraps(method)
    def operation(self,*args,**kwargs):
        if self.technical_version != 2 or self._operation_depth:return method(self,*args,**kwargs)
        _count(self._operation_stop+1)
        self._io_remaining=min(4194304,kwargs.get('maximum_io_bytes',4194304))
        forecast=self._forecast_total
        self._operation_depth+=1
        cpu=self._cost_cpu_position;written=self._cost_written_position;failed=False
        try:return method(self,*args,**kwargs)
        except BaseException:
            failed=True
            raise
        finally:
            self._operation_depth-=1
            self._cost_cpu_position=time.process_time()
            values=dict(process_cpu_seconds=self._cost_cpu_position-cpu,
                        bytes_written=None if failed else self._metadata_written-written)
            self._cost_written_position=self._metadata_written
            if method.__name__=='flush':values['flushes']=1
            for key in _COST_METRICS:
                value=values.get(key)
                item=dict(known=(_count(value) if key not in _COST_FLOATS else value) if value is not None else None,
                          covered='1' if value is not None else '0',unavailable='0' if value is not None else '1',
                          overflow='0',reasons=[] if value is not None else ['missing_observation'])
                self._cost_metrics[key]=_accumulate(self._cost_metrics[key],item,key not in _COST_FLOATS)
            self._operation_stop+=1
            self._io_forecast_pending+=self._forecast_total-forecast
    return operation


class TelemetryWriter:
    """Synchronous bounded flush outside numerical hot loops; no sample dropping.

    Multiple writers use isolated sessions. The short index update is serialized;
    numerical work and Parquet encoding never hold the index lock.
    """
    def __init__(self, root: Path, identity: Mapping[str, Any], hardware: Mapping[str, Any],
                 *, batch_rows: int = 256, batch_bytes: int = 4 * 1024 * 1024,
                 max_buffer_seconds: float = 15.0, metadata_budget=None, transport=None,
                 technical_version: int = 1, prior_capture: str = 'none') -> None:
        if type(technical_version) is not int or technical_version not in (1, 2):
            raise ValueError('technical_version must be 1 or 2')
        if prior_capture not in ('none', 'unimported') or technical_version == 1 and prior_capture != 'none':
            raise ValueError('prior_capture requires explicit v2')
        if technical_version == 2:
            if transport is None:raise ValueError('v2 requires explicit transport')
            if Path(root).is_symlink() or Path(root).exists() and any(Path(root).iterdir()):
                raise ValueError('v2 requires a fresh empty technical root')
        self._initialize_writer(root,identity,hardware,batch_rows=batch_rows,batch_bytes=batch_bytes,
            max_buffer_seconds=max_buffer_seconds,metadata_budget=metadata_budget,transport=transport,
            technical_version=technical_version,prior_capture=prior_capture)

    def _initialize_writer(self,root,identity,hardware,*,batch_rows=256,batch_bytes=4*1024*1024,
            max_buffer_seconds=15.,metadata_budget=None,transport=None,create_outputs=True,publish=True,
            technical_version=1,prior_capture="none"):
        self.technical_version=technical_version
        self.prior_capture=prior_capture
        self._catalog=None
        self._catalog_ref=None
        self._window_removals=[]
        self._pin_refused=False
        self._unlinked=set()
        self._observer_id=uuid.uuid4().hex
        self._operation_stop=0
        self._cost_metrics={k:_empty_accumulation() for k in _COST_METRICS}
        self._operation_depth=0
        self._io_remaining=4194304
        self._minimum_free_bytes=0
        self._forecast_total=0
        self._io_forecast_pending=0
        self._v2_phase_rows=[]
        self._v2_restart=False
        if technical_version==2 and (type(batch_bytes) is not int or batch_bytes>4194304):
            raise ValueError('v2 batch payload exceeds4MiB operation bound')
        if type(batch_rows) is not int or batch_rows < 1 or batch_rows > 65_536:
            raise ValueError("batch_rows must be in 1..65536")
        if type(batch_bytes) is not int or batch_bytes < 1024:
            raise ValueError("batch_bytes must be an integer >=1024")
        if not math.isfinite(max_buffer_seconds) or not 0 < max_buffer_seconds <= 60:
            raise ValueError("max_buffer_seconds must be finite and in (0,60]")
        if transport is not None:
            # Public closed schema, no scientific manifest coercion or inferred transport.
            encode(transport)
            validator=schema_validator('telemetry-envelope.schema.json')
            validator.evolve(schema={'$ref':'#/$defs/Transport','$defs':validator.schema['$defs']}).validate(transport)
        self.transport = deepcopy(transport)
        self.baseline = None
        self._baseline_summary = None
        self._recovery_summary_override = None
        self._coverage_gap = None
        self._inherited_dropped = 0
        self._catalog_wire_override = None
        self._resume_guards = []
        self._previous = None
        self._previous_summary = None
        self._previous_ref = None
        self._publication_pending = None
        self._published_context_raw = None
        self._metadata_written = 0
        self._retained_bytes = 0
        self._began_cpu = time.process_time()
        self._cost_cpu_position=self._began_cpu
        self._cost_written_position=0
        self.root = Path(root)
        self.metadata_budget = metadata_budget
        self.session_id = uuid.uuid4().hex
        self.directory = self.root / "sessions" / self.session_id
        if create_outputs:self.directory.mkdir(parents=True)
        self.identity = dict(identity)
        self._hardware = deepcopy(hardware)
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
        self.phase_aggregates = []
        self.accepted_ranges = []
        self.phase_contexts = []
        self._phase_context_refs = []
        self._resume_contexts = []
        self._last_phase_contexts = {}
        self._last_phase_raw = {}
        self._context_guards = {}
        self._phase_raw = {}
        self.accepted_totals = dict(accepted_source_rows=0, accepted_aggregate_calls=0, accepted_diagnostic_events=0, coalesced_source_events=0)
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
        if not create_outputs:return
        passport = {"schema": TELEMETRY_SCHEMA, "contract_set": CONTRACT_SET, "identity": self.identity,
            "session_id": self.session_id, "utc_anchor_ns": self.utc_anchor_ns,
            "monotonic_anchor_ns": self.monotonic_anchor_ns,
            "python": platform.python_version(), "hardware": dict(hardware),
            "missing_hardware_fields": sorted({"cpu_model", "topology", "affinity",
                "cpu_quota", "julia_threads", "blas_threads", "gc_threads",
                "compile_target", "libraries"} - set(hardware)),
            "time_semantics": "monotonic nanoseconds; durations and CPU time in seconds",
            "sampling": "all received rows retained; absent measurements are null"}
        if self.technical_version == 1:
            self._metadata_write(self.directory / "hardware.json", json_bytes(passport), immutable=True)
            self.artifacts.append(self._metadata_artifact(self.directory / "hardware.json",
                relative="hardware.json", role="performance.summary", media_type="application/json",schema=TELEMETRY_SCHEMA))
        elif not (self.root/'hardware.json').exists():
            # Fixed root passport is not a retained-catalog member: S1 permits
            # performance.v2 only as Parquet, never as arbitrary JSON metadata.
            self._metadata_write(self.root/'hardware.json',json_bytes(passport),immutable=True)
        if publish:
            self._publish()
            self._register()

    @classmethod
    def resume(cls,root: Path,*,commit_ref,expected_session,hardware,
            batch_rows=256,batch_bytes=4*1024*1024,max_buffer_seconds=15.,metadata_budget=None,
            technical_version=1,prior_capture='none'):
        if technical_version == 2:
            return cls._resume2(root,commit_ref=commit_ref,expected_session=expected_session,hardware=hardware,
                batch_rows=batch_rows,batch_bytes=batch_bytes,max_buffer_seconds=max_buffer_seconds,
                metadata_budget=metadata_budget,prior_capture=prior_capture)
        if technical_version != 1 or prior_capture != 'none':raise ValueError('invalid technical version/prior capture')
        """Resume one exact immutable metadata baseline; never infer old CPU endpoints."""
        from ._capture_io import regular,stamp
        from .commits import decode_json
        from .state import _IOBudget
        root=Path(root);io=_IOBudget(4*1024*1024);guards=[]
        def read(ref,schema):
            common=schema_validator('result-common.schema.json')
            common.evolve(schema={'$ref':'#/$defs/FileRef','$defs':common.schema['$defs']}).validate(ref)
            if ref['schema']!=schema or ref['media_type']!='application/json':raise ValueError('resume reference schema differs')
            size=int(ref['bytes'])
            if size>1024*1024:raise ValueError('resume metadata bound exceeded')
            from ._capture_io import beneath
            path=beneath(root,ref['path'])
            with regular(path,size=size) as stream:
                before=stamp(stream,include_change=True);io.require(size)
                if metadata_budget is not None:metadata_budget.require(size)
                raw=stream.read(size);io.account(len(raw))
                if metadata_budget is not None:metadata_budget.account(len(raw))
                if len(raw)!=size or hashlib.sha256(raw).hexdigest()!=ref['sha256'] or stamp(stream,include_change=True)!=before:
                    raise ValueError('resume metadata bytes changed')
                current=path.stat(follow_symlinks=False)
                if (current.st_dev,current.st_ino,current.st_size,current.st_mtime_ns,current.st_ctime_ns)!=before:
                    raise ValueError('resume metadata path replaced')
                guards.append((path,before))
            return decode_json(raw,'resume metadata')
        commit=read(commit_ref,'qcl-negf.telemetry-commit.v1')
        validator=schema_validator('telemetry-envelope.schema.json')
        validator.evolve(schema={'$ref':'#/$defs/TelemetryCommit','$defs':validator.schema['$defs']}).validate(commit)
        if encode(commit['session'])!=encode(expected_session):raise ValueError('selected resume collector session differs')
        summary=read(commit['summary'],'qcl-negf.telemetry-summary.v1');validate_telemetry_summary(summary)
        for key in ('session','transport','generation','closed','baseline','source_cursors'):
            if encode(dict(v=commit[key]))!=encode(dict(v=summary[key])):raise ValueError('resume commit/summary transaction differs')
        def verify():
            for path,before in guards:
                with regular(path,size=before[2]) as stream:
                    if stamp(stream,include_change=True)!=before:raise ValueError('resume source changed before publication')
        verify()
        if not commit['closed']:
            recovery=cls.__new__(cls)
            recovery._initialize_writer(root,{},hardware,batch_rows=batch_rows,batch_bytes=batch_bytes,
                max_buffer_seconds=max_buffer_seconds,metadata_budget=metadata_budget,transport=commit['transport'],create_outputs=False)
            recovery.directory=root/Path(commit_ref['path']).parent
            recovery.session_id=commit['session']['source']['session_id']
            recovery._previous=deepcopy(commit);recovery._previous_summary=deepcopy(summary);recovery._previous_ref=deepcopy(commit_ref)
            recovery._recovery_summary_override=deepcopy(summary);recovery.generation=int(commit['generation'])
            recovery.source_cursors={r['source_key']:deepcopy(r) for r in commit['source_cursors']}
            recovery.baseline=deepcopy(commit['baseline']);recovery._catalog_wire_override=deepcopy(commit['catalog'])
            recovery._retained_bytes=int(summary['retention']['retained_bytes']);recovery._resume_guards=list(guards);recovery.closed=True
            recovery._publish()
            commit=deepcopy(recovery._previous);summary=deepcopy(recovery._previous_summary);commit_ref=deepcopy(recovery._previous_ref)
        obj=cls.__new__(cls)
        obj._initialize_writer(root,{},hardware,batch_rows=batch_rows,batch_bytes=batch_bytes,
            max_buffer_seconds=max_buffer_seconds,metadata_budget=metadata_budget,transport=commit['transport'],publish=False)
        obj._previous=deepcopy(commit);obj._previous_summary=deepcopy(summary);obj._previous_ref=deepcopy(commit_ref)
        obj.generation=int(commit['generation']);obj.baseline=dict(commit=deepcopy(commit_ref),summary=deepcopy(commit['summary']))
        obj._baseline_summary=deepcopy(summary);obj._resume_guards=list(guards)
        obj._retained_bytes=int(summary['retention']['retained_bytes'])
        obj._inherited_dropped=int(summary['totals']['collector_dropped_rows'])
        obj.accepted_totals={k:int(summary['totals'][k]) for k in obj.accepted_totals}
        obj.source_cursors={r['source_key']:deepcopy(r) for r in commit['source_cursors']}
        obj._coverage_gap=dict(source_key=None,clock_domain_id=None,kind='collector_restart',
            first_monotonic_ns=None,last_monotonic_ns=None,reason='endpoint_reset_uncovered')
        verify();obj._publish();obj._register()
        return obj

    def _charge_io2(self,count):
        if self.technical_version == 2:
            if count>self._io_remaining:raise ValueError('operation I/O allowance exhausted before dependent I/O')
            self._io_remaining-=count

    def _space2(self,burst):
        if self.technical_version==2:
            space=os.statvfs(self.root)
            debt=sum(int(r['bytes']) for r in self._previous['window_delta']['pending_unlink']) if self._previous else 0
            retained=sum(int(r['ref']['bytes']) for r in self._catalog['artifacts']) if self._catalog else 0
            if space.f_bavail*space.f_frsize < self._minimum_free_bytes+debt+retained+burst+1048576:
                raise ValueError('scientific free-space reserve unavailable before dependent write')

    def _metadata_write(self, path, payload, *, immutable=False):
        self._space2(len(payload))
        self._charge_io2(len(payload))
        if len(payload)>1024*1024:
            raise ValueError('technical metadata exceeds bounded page')
        if self.metadata_budget is not None:
            self.metadata_budget.write(len(payload))
        atomic_write(path, payload, immutable=immutable)
        self._metadata_written += len(payload)

    def _check_retained(self,path,payload):
        self._charge_io2(len(payload))
        from ._capture_io import regular,stamp
        with regular(path,size=len(payload)) as stream:
            before=stamp(stream,include_change=True)
            if self.metadata_budget is not None:self.metadata_budget.require(len(payload))
            raw=stream.read(len(payload))
            if self.metadata_budget is not None:self.metadata_budget.account(len(raw))
            if raw!=payload or stamp(stream,include_change=True)!=before:
                raise ValueError('retained immutable telemetry bytes changed')
            now=path.stat(follow_symlinks=False)
            if (now.st_dev,now.st_ino,now.st_size,now.st_mtime_ns,now.st_ctime_ns)!=before:
                raise ValueError('retained immutable telemetry path replaced')
            return before

    def _metadata_artifact(self, path, **kwargs):
        if self.metadata_budget is not None:
            size = path.stat().st_size
            self.metadata_budget.require(size)
            self.metadata_budget.account(size)
        return artifact_row(path, **kwargs)

    def _root_ref(self,reference):
        """Wire refs share the explicit technical root across collector pages."""
        return dict(reference,path=str(self.directory.relative_to(self.root)/reference['path']))

    def _register(self) -> None:
        if self.technical_version == 2:return
        self.root.mkdir(parents=True, exist_ok=True)
        with (self.root / ".index.lock").open("a+b") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            index_path = self.root / "index.json"
            if self.metadata_budget is not None and index_path.exists():
                from ._capture_io import small_json
                index = small_json(index_path, 1024 * 1024, budget=self.metadata_budget)[0]
            else:
                index = read_json(index_path)[0] if index_path.exists() else {
                    "schema": "qcl-negf.performance-index.v2", "contract_set": CONTRACT_SET, "sessions": []}
            if (index.get("schema") != "qcl-negf.performance-index.v2"
                    or index.get("contract_set") != CONTRACT_SET):
                raise ValueError("incompatible_contract: performance index")
            index["sessions"].append({"id": self.session_id,
                "pointer": f"sessions/{self.session_id}/current.json"})
            self._metadata_write(index_path, json_bytes(index))

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

    @_observed
    def _record(self, table: str, row: Mapping[str, Any], science_window: bool) -> None:
        with self.lock:
            if self.closed:
                raise RuntimeError("telemetry session is closed")
            if self.technical_version==2 and any(isinstance(v,str) and len(v)>self.batch_bytes for v in row.values()):
                raise ValueError('row string exceeds bounded payload before UTF8 allocation')
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
        value["sample_sequence"] = self.next_sequence[table]
        self.next_sequence[table] += 1
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

    @_observed
    def ingest_spans(self, rows: list[Mapping[str, Any]], *, source_cursors: Mapping[str, Any]) -> Path:
        """Publish source offsets and their exact rows in the same immutable commit.

        Parsing/validation precedes mutation. No intermediate automatic flush can
        acknowledge a source offset before every corresponding row is durable.
        """
        if source_cursors:
            raise ValueError('nonempty source cursors require stage_phase_batch with actual P29 contexts')
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

    @_observed
    def stage_spans(self, rows: list[Mapping[str, Any]], *, source_cursors: Mapping[str, Any]) -> None:
        """Accept a replayable source batch without promising immediate durability.

        The caller may advance its in-memory read offset after return. Recovery
        MUST use source_cursors from a published commit, never staged offsets.
        Rows and their offsets become durable together at a flush barrier. The
        caller must invoke flush_due during its polling loop, even with no rows.
        """
        if source_cursors:
            raise ValueError('nonempty source cursors require stage_phase_batch with actual P29 contexts')
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

    @staticmethod
    def _merge_phase(left,right):
        """Producer aggregation of accepted metadata; no physical recomputation."""
        result=deepcopy(left);result['call_count']=str(int(left['call_count'])+int(right['call_count']))
        for start,stop in (('first_sequence','last_sequence'),('first_core_sequence','last_core_sequence')):
            lo=[int(r[start]) for r in (left,right) if r[start] is not None];hi=[int(r[stop]) for r in (left,right) if r[stop] is not None]
            result[start]=str(min(lo)) if lo else None;result[stop]=str(max(hi)) if hi else None
        clocks=result['time_bounds'];other=right['time_bounds']
        for k in ('measured_calls','unavailable_calls'):clocks[k]=str(int(clocks[k])+int(other[k]))
        for k,op in (('min_start_monotonic_ns',min),('max_end_monotonic_ns',max)):
            values=[int(r['time_bounds'][k]) for r in (left,right) if r['time_bounds'][k] is not None]
            clocks[k]=str(op(values)) if values else None
        clocks['reasons']=sorted(set(clocks['reasons']+other['reasons']))
        a={(m['name'],m['unit']):m for m in left['metrics']};b={(m['name'],m['unit']):m for m in right['metrics']}
        if a.keys()!=b.keys():raise ValueError('metric profile changed; a separate detail page is required')
        result['metrics']=[]
        for key,x in a.items():
            y=b[key];vals=[r['sum'] for r in (x,y) if int(r['measured_calls'])]
            reasons=sorted(set(x['reasons']+y['reasons']));total=None
            if x['sum'] is not None and y['sum'] is not None and isinstance(x['sum'],dict)!=isinstance(y['sum'],dict):
                raise ValueError('native integer versus Float64 metric profile changed')
            if vals and all(v is not None for v in vals):
                if len(vals)==1:total=deepcopy(vals[0])
                elif isinstance(vals[0],dict):
                    value=sum(int(v['decimal']) for v in vals)
                    if -(2**63)<=value<2**63:total=dict(kind='integer',type='Int64',decimal=str(value))
                    else:reasons=sorted(set(reasons+['counter_overflow']))
                else:
                    try:total=vals[0]+vals[1];finite=math.isfinite(total)
                    except OverflowError:finite=False
                    if not finite:total=None;reasons=sorted(set(reasons+['counter_overflow']))
            result['metrics'].append(dict(name=key[0],unit=key[1],sum=total,
                measured_calls=str(int(x['measured_calls'])+int(y['measured_calls'])),
                unavailable_calls=str(int(x['unavailable_calls'])+int(y['unavailable_calls'])),reasons=reasons))
        return result

    @staticmethod
    def _diagnostic_phase(row):
        """Only explicitly measured end metrics/clocks, never infer normal spans."""
        metrics=[]
        allowed={'duration_seconds':('s',),'process_cpu_seconds':('s',),'allocated_bytes':('B','bytes'),'gc_seconds':('s',),'gc_count':('1','')}
        for item in row['metrics']:
            name='process_cpu_seconds' if item['name']=='cpu_seconds' else item['name']
            if name not in allowed:continue
            value=item['value'];numeric=type(value) in (int,float) or isinstance(value,dict) and value.get('kind')=='integer'
            measured=item['measurement_status']=='measured' and numeric and item['unit'] in allowed[name]
            metrics.append(dict(name=name,unit=item['unit'],sum=deepcopy(value) if measured else None,
                measured_calls='1' if measured else '0',unavailable_calls='0' if measured else '1',
                reasons=[] if measured else [item['reason'] or ('metric_type_or_unit_invalid' if item['measurement_status']=='measured' else 'producer_metric_unavailable')]))
        def clock(name):
            m=next((m for m in row['metrics'] if m['name']==name),None)
            if m and m['measurement_status']=='measured' and m['unit']=='ns' and isinstance(m['value'],dict) and m['value'].get('type')=='Int64':
                value=int(m['value']['decimal'])
                if value>=0:return value
            return None
        start,end=clock('start_monotonic_ns'),clock('end_monotonic_ns')
        producer=next((m for m in row['metrics'] if m['name']=='producer_monotonic_ns'),None)
        mismatch=producer is not None and producer['measurement_status']=='measured' and clock('producer_monotonic_ns')!=end
        valid=start is not None and end is not None and start<=end and not mismatch
        reason='producer_clock_mismatch' if mismatch else 'producer_clock_missing' if start is None or end is None else 'producer_clock_reversed'
        return dict(kind='aggregate',stage=row['stage'],phase=row['label'],status=row['status'],task_width=None,
            counter_scope='process_inclusive_delta_on_end',first_sequence=row['sequence'],last_sequence=row['sequence'],
            first_core_sequence=row['core_sequence'],last_core_sequence=row['core_sequence'],call_count='1',metrics=metrics,
            time_bounds=dict(min_start_monotonic_ns=str(start) if valid else None,max_end_monotonic_ns=str(end) if valid else None,
                measured_calls='1' if valid else '0',unavailable_calls='0' if valid else '1',reasons=[] if valid else [reason]))

    def _event_span(self,batch,row,offset):
        """Explicit optional technical projection of actual measured end Events.

        Raw events remain exact and authoritative for unsupported/missing metrics.
        No normal aggregate, duration, clock or per-call correlation is invented.
        """
        def metric(name,unit,integer=False):
            item=next((m for m in row['metrics'] if m['name']==name),None)
            if item is None or item['measurement_status']!='measured' or item['unit'] not in unit:return None
            value=item['value']
            if isinstance(value,dict) and value.get('type')=='Int64':value=int(value['decimal'])
            if integer:return value if type(value) is int and 0<=value<2**63 else None
            return value if type(value) in (float,int) and math.isfinite(value) and value>=0 else None
        duration=metric('duration_seconds',('s',))
        if row['action']!='phase_end' or duration is None:return None
        number=lambda x:int(x) if x is not None and 0<=int(x)<2**63 else None
        end=metric('end_monotonic_ns',('ns',),True)
        value=dict(phase=row['label'],duration_seconds=duration,event_kind='end',measurement_kind='phase_boundary',
            timing_kind='inclusive',timestamp_kind='measurement' if end is not None else 'unavailable',
            monotonic_ns=end,clock_domain_id=batch.source['clock_domain_id'],utc_anchor_ns=number(batch.source['utc_anchor_ns']),
            monotonic_anchor_ns=number(batch.source['monotonic_anchor_ns']),begin_monotonic_ns=metric('start_monotonic_ns',('ns',),True),end_monotonic_ns=end,
            source_session_id=batch.source['session_id'],source_sequence=number(row['sequence']),source_path=batch.segment_ref['path'],source_offset=offset,
            stage_path=row['stage'],iteration=number(row['iteration']),attempt=number(batch.identity['attempt']),
            process_cpu_seconds=metric('cpu_seconds',('s',)) if any(m['name']=='cpu_seconds' for m in row['metrics']) else metric('process_cpu_seconds',('s',)),
            gc_seconds=metric('gc_seconds',('s',)),allocated_bytes=metric('allocated_bytes',('B','bytes'),True),gc_count=metric('gc_count',('1',''),True),
            counter_scope='process_inclusive_delta_on_end',capability_reasons='exact_raw_events_retained; projection_only_measured_representable_metrics')
        return self._normalise('spans',value)

    def _input_contexts(self,phase_context_refs,resume_context_refs):
        """Authenticate explicit retained context refs, never derive raw paths."""
        from .performance_views import TelemetrySelection,_check_bindings
        from .state import _IOBudget,_schema_check,_metadata_weight
        from ._capture_io import regular,beneath,stamp
        from .commits import decode_json
        # Reuse the accepted closed input grammar without another reader/owner.
        for values,maximum,keys in ((phase_context_refs,128,{'publication_ref','segment_ref','retained_publication_ref','retained_segment_ref'}),
                (resume_context_refs,16,{'commit_ref','summary_ref'})):
            if not isinstance(values,(tuple,list)) or len(values)>maximum or any(not isinstance(v,dict) or set(v)!=keys for v in values):
                raise ValueError('bounded exact context reference inputs required')
        io=_IOBudget(4*1024*1024);guards={};cache={}
        def read(ref):
            common=schema_validator('result-common.schema.json')
            _schema_check(common.evolve(schema={'$ref':'#/$defs/FileRef','$defs':common.schema['$defs']}),ref)
            key=json_bytes(ref)
            if key in cache:return cache[key]
            count=int(ref['bytes'])
            if count>self.batch_bytes or ref['media_type']!='application/json':raise ValueError('context metadata allowance/schema exceeded')
            path=beneath(self.root,ref['path'])
            with regular(path,size=count) as raw:
                self._charge_io2(count)
                before=stamp(raw,include_change=True);io.require(count)
                if self.metadata_budget is not None:self.metadata_budget.require(count)
                data=raw.read(count);io.account(len(data))
                if self.metadata_budget is not None:self.metadata_budget.account(len(data))
                if len(data)!=count or hashlib.sha256(data).hexdigest()!=ref['sha256'] or stamp(raw,include_change=True)!=before:
                    raise ValueError('context original retained bytes changed')
                guards[(str(self.root.absolute()),ref['path'])]=before
                _check_bindings(guards)
            value=decode_json(data,'retained explicit context');cache[key]=value
            if _metadata_weight((cache,guards,phase_context_refs,resume_context_refs))>self.batch_bytes:
                raise ValueError('combined input context allowance exceeded')
            return value
        phases=[];seen=set()
        for row in phase_context_refs:
            key=json_bytes(row['publication_ref'])
            if key in seen:raise ValueError('duplicate phase context input')
            seen.add(key)
            for original,retained in (('publication_ref','retained_publication_ref'),('segment_ref','retained_segment_ref')):
                if any(row[original][k]!=row[retained][k] for k in ('bytes','sha256','media_type','schema')):
                    raise ValueError('retained context ref differs from original')
            phases.append(dict(publication=read(row['retained_publication_ref']),publication_ref=deepcopy(row['publication_ref']),
                segment=read(row['retained_segment_ref']),segment_ref=deepcopy(row['segment_ref'])))
        resumes=[dict(commit=read(r['commit_ref']),commit_ref=deepcopy(r['commit_ref']),summary=read(r['summary_ref']),summary_ref=deepcopy(r['summary_ref'])) for r in resume_context_refs]
        _check_bindings(guards)
        return phases,resumes,guards

    @_observed
    def rotate(self):
        if self.technical_version == 2:return self._rotate2()
        """Explicit barrier: close old page, then begin an empty active page."""
        from .performance_views import _check_bindings
        with self.lock:
            old_path=self.close();old=deepcopy(self._previous);old_summary=deepcopy(self._previous_summary);old_ref=deepcopy(self._previous_ref)
            # Exact actual baseline bytes are consumed here, not a post-read stat.
            _,_,guards=self._input_contexts((),(dict(commit_ref=old_ref,summary_ref=old['summary']),))
            candidate=type(self).__new__(type(self))
            try:
                candidate._initialize_writer(self.root,self.identity,self._hardware,batch_rows=self.batch_rows,batch_bytes=self.batch_bytes,
                    max_buffer_seconds=self.max_buffer_seconds,metadata_budget=self.metadata_budget,transport=self.transport,publish=False)
                candidate.lock=self.lock;candidate._previous=old;candidate._previous_summary=old_summary;candidate._previous_ref=old_ref
                candidate.generation=int(old['generation']);candidate.baseline=dict(commit=old_ref,summary=deepcopy(old['summary']))
                candidate._baseline_summary=old_summary;candidate._retained_bytes=self._retained_bytes
                candidate._inherited_dropped=int(old_summary['totals']['collector_dropped_rows'])
                candidate.accepted_totals=dict(self.accepted_totals)
                candidate._coverage_gap=dict(source_key=None,clock_domain_id=None,kind='collector_restart',
                    first_monotonic_ns=None,last_monotonic_ns=None,reason='explicit_page_rotation_endpoint_reset')
                _check_bindings(guards);candidate._publish();candidate._register();_check_bindings(guards)
            except BaseException as error:
                error.add_note('old collector page remains closed at '+str(old_path)+'; next page publication did not complete')
                raise
            self.__dict__.update(candidate.__dict__)
            return old_path

    @_observed
    def stage_phase_batch(self,batch,*,phase_context_refs=(),resume_context_refs=()) -> None:
        from .capture import PhaseBatch
        from .state import _metadata_weight
        from qcl_negf_contracts.telemetry import validate_phase_publication,validate_phase_cursor
        if not isinstance(batch,PhaseBatch):raise TypeError('actual PhaseBatch required')
        with self.lock:
            if self.closed:raise RuntimeError('telemetry session is closed')
            if self._publication_pending is not None:raise RuntimeError('pending publication requires explicit flush recovery')
            # Both decoded contexts and immutable raw buffers count before copies.
            if _metadata_weight((self.phase_contexts,self.phase_aggregates,self._last_phase_contexts,self._phase_raw,vars(batch)))>self.batch_bytes:
                raise ValueError('phase staging exceeds configured byte bound')
            originals=[(batch.segment_ref,batch.segment_bytes,batch.segment),(batch.publication_ref,batch.publication_bytes,batch.publication)]
            if batch.previous_ref is not None:originals.append((batch.previous_ref,batch.previous_bytes,batch.previous))
            for ref,raw,decoded in originals:
                if type(raw) is not bytes or len(raw)!=int(ref['bytes']) or hashlib.sha256(raw).hexdigest()!=ref['sha256']:
                    raise ValueError('phase raw bytes differ from exact original reference')
                from .commits import decode_json
                if encode(decode_json(raw))!=encode(decoded):raise ValueError('phase raw and decoded contexts differ')
            validate_phase_publication(batch.publication,segment=batch.segment,segment_ref=batch.segment_ref,
                previous=batch.previous,previous_ref=batch.previous_ref)
            context=dict(publication=deepcopy(batch.publication),publication_ref=deepcopy(batch.publication_ref),
                segment=deepcopy(batch.segment),segment_ref=deepcopy(batch.segment_ref))
            validate_phase_cursor(batch.cursor,**context)
            if type(batch.first_row) is not int or type(batch.stop_row) is not int or not 0<=batch.first_row<batch.stop_row<=len(batch.segment['rows']) or int(batch.cursor['next_row'])!=batch.stop_row:
                raise ValueError('invalid accepted physical phase prefix')
            sourcekey=hashlib.sha256(encode(dict(identity=batch.identity,source=batch.source))).hexdigest()
            cursors={**self.source_cursors,**deepcopy(self.staged_source_cursors)}
            supplied_phases,supplied_resumes,input_guards=self._input_contexts(phase_context_refs,resume_context_refs)
            resumes={json_bytes(c['commit_ref']):c for c in self._resume_contexts}
            for c in supplied_resumes:
                if json_bytes(c['commit_ref']) in resumes:raise ValueError('duplicate resume context input')
                resumes[json_bytes(c['commit_ref'])]=c
            old=cursors.get(sourcekey)
            if old is None:
                matching=[r for c in resumes.values() for r in c['commit']['source_cursors'] if r['source_key']==sourcekey]
                if len(matching)>1:raise ValueError('ambiguous resume source context')
                if matching:old=matching[0]
            if old is None and batch.publication['previous'] is not None:
                raise ValueError('non-genesis source requires exact prior collector resume context')
            expected=int(old['cursor']['next_row']) if old is not None and old['cursor']['publication']==batch.publication_ref else 0
            if batch.first_row!=expected:raise ValueError('phase prefix replay/skip')
            if len(cursors)+(sourcekey not in cursors)>16:raise ValueError('source context page full; rotate detail page')
            cursorrow=dict(source_key=sourcekey,identity=deepcopy(batch.identity),source=deepcopy(batch.source),cursor=deepcopy(batch.cursor),
                route_epoch=str(batch.route_epoch),binding=dict(status='unbound',state=None,invocation=None,evidence=None,reason='producer_mapping_missing'),
                producer_health=deepcopy(batch.current['health']),source_closed=batch.current['closed'],health_reason=None)
            cursors[sourcekey]=cursorrow
            contexts={json_bytes(c['publication_ref']):c for c in self.phase_contexts}
            for c in supplied_phases:
                if json_bytes(c['publication_ref']) in contexts:raise ValueError('duplicate phase completion context')
                contexts[json_bytes(c['publication_ref'])]=c
            contexts[json_bytes(batch.publication_ref)]=context
            if old is not None and old['cursor']['publication']!=batch.publication_ref:
                previous_context=contexts.get(json_bytes(old['cursor']['publication']))
                if previous_context is None and sourcekey in self._last_phase_raw:
                    actual,_,guards=self._input_contexts((self._last_phase_raw[sourcekey],),())
                    previous_context=actual[0];input_guards.update(guards)
                if previous_context is None:raise ValueError('old physical completion context missing')
                contexts[json_bytes(previous_context['publication_ref'])]=deepcopy(previous_context)
            rawrefs=[]
            for label,ref,raw in (('segment',batch.segment_ref,batch.segment_bytes),('publication',batch.publication_ref,batch.publication_bytes)):
                path='raw/'+sourcekey+'-'+label+'-'+ref['sha256']+'.json'
                rawrefs.append(self._root_ref(dict(ref,path=path)))
            accepted=dict(source_key=sourcekey,publication=deepcopy(batch.publication_ref),segment=deepcopy(batch.segment_ref),
                retained_publication=rawrefs[1],retained_segment=rawrefs[0],mode=batch.segment['mode'],
                first_row=str(batch.first_row),stop_row=str(batch.stop_row),cursor=deepcopy(batch.cursor))
            # Keep only the explicit original-to-retained refs used by this delta.
            # Decoded validator contexts alone cannot reproduce an offline read.
            reference_inputs = (*self._last_phase_raw.values(), *self._phase_context_refs,
                *phase_context_refs, dict(publication_ref=batch.publication_ref,
                    segment_ref=batch.segment_ref, retained_publication_ref=rawrefs[1], retained_segment_ref=rawrefs[0]))
            by_publication = {json_bytes(row['publication_ref']): row for row in reference_inputs}
            context_refs = [by_publication[json_bytes(row['publication_ref'])] for row in contexts.values()]
            rows=batch.segment['rows'][batch.first_row:batch.stop_row]
            totals=dict(self.accepted_totals);totals['accepted_source_rows']+=len(rows)
            if batch.stop_row==len(batch.segment['rows']):totals['coalesced_source_events']+=int(batch.segment['coalesced_events'])
            if batch.segment['mode']=='normal':totals['accepted_aggregate_calls']+=sum(int(r['call_count']) for r in rows);projected=rows;representation='normal_aggregate'
            else:
                totals['accepted_diagnostic_events']+=len(rows)
                projected=[self._diagnostic_phase(r) for r in rows if r['action']=='phase_end'];representation='diagnostic_end_projection'
            phases=deepcopy(self.phase_aggregates)
            def key(p):
                a=p['aggregate'];return (p['source_key'],p['representation'],a['stage'],a['phase'],a['status'],a['task_width'],a['counter_scope'])
            for row in projected:
                entry=dict(source_key=sourcekey,identity=deepcopy(batch.identity),source=deepcopy(batch.source),representation=representation,aggregate=deepcopy(row))
                found=next((i for i,p in enumerate(phases) if key(p)==key(entry)),None)
                if found is None:phases.append(entry)
                else:phases[found]['aggregate']=self._merge_phase(phases[found]['aggregate'],row)
            span_rows=[]
            if batch.segment['mode']=='diagnostic':
                span_rows=[value for index,row in enumerate(rows,batch.first_row) if (value:=self._event_span(batch,row,index)) is not None]
            size=sum(self._row_bytes(row) for row in span_rows)
            if size+self.pending_bytes>self.batch_bytes or self.pending_rows+len(span_rows)>2*self.batch_rows or len(span_rows)>self.batch_rows:
                raise ValueError('diagnostic projection exceeds row/byte bound; flush previous staged rows first')
            candidate=copy(self);candidate.phase_aggregates=phases;candidate.accepted_totals=totals
            candidate.counts=dict(self.counts,spans=self.counts['spans']+len(span_rows))
            summary=candidate._technical_summary(self.generation+1,list(cursors.values()))
            summary_raw=json_bytes(summary)
            commit=dict(schema='qcl-negf.telemetry-commit.v1',contract_set=CONTRACT_SET,session=summary['session'],transport=deepcopy(self.transport),
                generation=str(self.generation+1),closed=False,published_unix=time.time(),previous=deepcopy(self._previous_ref),baseline=deepcopy(self.baseline),
                summary=self._root_ref(dict(path=f'summary-{self.generation+1:06d}.json',bytes=str(len(summary_raw)),sha256=hashlib.sha256(summary_raw).hexdigest(),media_type='application/json',schema=summary['schema'])),
                catalog=self._root_ref(self.catalog_watermark) if self.catalog_watermark is not None else None,source_cursors=list(cursors.values()),accepted_ranges=deepcopy(self.accepted_ranges)+[accepted])
            kwargs={} if self._previous is None else dict(previous=self._previous,previous_ref=self._previous_ref,previous_summary=self._previous_summary,previous_summary_ref=self._previous['summary'])
            if self.technical_version == 1:
                validate_telemetry_commit(commit,summary=summary,summary_ref=commit['summary'],phase_contexts=list(contexts.values()),resume_contexts=list(resumes.values()),**kwargs)
            else:
                candidate.accepted_ranges=deepcopy(self.accepted_ranges)+[accepted]
                candidate.phase_contexts=list(contexts.values());candidate._phase_context_refs=context_refs
                candidate._resume_contexts=list(resumes.values());candidate.staged_source_cursors=cursors
                candidate._v2_phase_rows=self._v2_phase_rows+[(representation,deepcopy(row)) for row in projected]
                candidate.artifacts=deepcopy(self.artifacts)
                for ref in rawrefs:
                    if ref['path'] not in self._phase_raw:
                        candidate.artifacts.append(dict(ref,path=str((self.root/ref['path']).relative_to(self.directory)),
                            bytes=int(ref['bytes']),role='performance.window'))
                pending={table:sum(len(rows) for (name,_),rows in self.buffers.items() if name==table) for table in TABLE_FIELDS}
                pending['spans']+=len(span_rows)
                for table,count in pending.items():
                    if count:
                        candidate.artifacts.append(dict(path='.staging-check-'+table+'.parquet',bytes=1,sha256='0'*64,
                            media_type='application/vnd.apache.parquet',schema=TELEMETRY_SCHEMA,role='performance.full',table=table,_staging_rows=count))
                candidate._prepare2()

            if _metadata_weight((self.phase_aggregates,phases,self.phase_contexts,list(contexts.values()),self._last_phase_contexts,self._last_phase_raw,cursors,list(resumes.values()),input_guards,self._context_guards)) + 2*_metadata_weight(context_refs)>self.batch_bytes:
                raise ValueError('phase old/new working metadata exceeds byte bound')
            from .performance_views import _check_bindings
            _check_bindings(input_guards)
            # Invalid input above has not advanced counters/cursors or published files.
            additions=[];raw_guards={}
            for ref,raw in zip(rawrefs,(batch.segment_bytes,batch.publication_bytes)):
                target=self.root/ref['path']
                if target.exists():
                    self._check_retained(target,raw)
                else:
                    if self.metadata_budget is not None:self.metadata_budget.write(len(raw))
                    self._space2(len(raw));self._charge_io2(len(raw))
                    atomic_write(target,raw,immutable=True);self._metadata_written+=len(raw)
                if self.technical_version==2:
                    raw_guards[(str(self.root.absolute()),ref['path'])]=self._check_retained(target,raw)
                if ref['path'] not in self._phase_raw:
                    additions.append(dict(ref,path=str(target.relative_to(self.directory)),bytes=int(ref['bytes']),role='performance.window',profile='science',dependencies=[]))
            if self.technical_version == 2:
                self._v2_phase_rows.extend((representation,deepcopy(row)) for row in projected)
            self.artifacts.extend(additions)
            for ref in rawrefs:self._phase_raw[ref['path']]=deepcopy(ref)
            self._phase_context_refs = deepcopy(context_refs)
            self.phase_contexts=list(contexts.values());self._resume_contexts=list(resumes.values());self.phase_aggregates=phases;self.accepted_totals=totals
            self.accepted_ranges.append(accepted);self.staged_source_cursors[sourcekey]=cursorrow;self._last_phase_contexts[sourcekey]=context
            self._last_phase_raw[sourcekey]=dict(publication_ref=deepcopy(batch.publication_ref),segment_ref=deepcopy(batch.segment_ref),
                retained_publication_ref=deepcopy(rawrefs[1]),retained_segment_ref=deepcopy(rawrefs[0]))
            self._context_guards.update(input_guards)
            self._context_guards.update(raw_guards)
            for value in span_rows:self._append('spans',value,False,auto_flush=False)
            self._mark_pending()
            if self.pending_rows>=self.batch_rows:self.flush()

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

    @_observed
    def flush(self) -> Path:
        return self._flush()

    def _flush(self, *, closing: bool = False) -> Path:
        with self.lock:
            if self._publication_pending is not None:
                pending_closed=self._publication_pending['commit']['closed']
                self._publish()
                self.closed=pending_closed
                if pending_closed:return self.root/self._previous_ref['path']
                if not closing and not self.pending_rows and not self.artifacts and not self.staged_source_cursors:
                    self.dirty=False;self.pending_since_ns=None;return self.root/self._previous_ref['path']
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
                forecast=0
                if self.technical_version==2:
                    forecast=4*sum(self._row_bytes(row) for row in rows)+262144
                    self._space2(forecast);self._charge_io2(forecast);self._forecast_total+=forecast
                try:
                    pq.write_table(pa.Table.from_pylist(rows, schema=schema), temporary,
                                   compression="zstd", compression_level=3,
                                   use_dictionary=True, write_statistics=True)
                    with temporary.open("rb") as stream:
                        os.fsync(stream.fileno())
                    # A readable footer is necessary before publication.
                    if pq.ParquetFile(temporary).metadata.num_rows != len(rows):
                        raise RuntimeError("Parquet row count mismatch before commit")
                    if self.technical_version==2 and 3*temporary.stat().st_size>forecast:
                        raise ValueError('Arrow serialization exceeded admitted I/O forecast')
                    destination = self.directory / name
                    os.link(temporary, destination)
                    fsync_directory(self.directory)
                    self.artifacts.append(artifact_row(destination, relative=name,
                        role="performance.window" if selected_window else "performance.full",
                        profile="science" if selected_window else "full-state",
                        media_type="application/vnd.apache.parquet", schema=TELEMETRY_SCHEMA))
                    self.artifacts[-1]["table"] = table
                    if self.technical_version==2:self._metadata_written+=destination.stat().st_size
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

    @staticmethod
    def _constants(name):
        properties=load_schema('telemetry-envelope.schema.json')['$defs'][name]['properties']
        return {key:deepcopy(value['const']) for key,value in properties.items() if 'const' in value}

    @staticmethod
    def _measurement(value, unit, *, reason='not_instrumented', forecast=False):
        return dict(value=value,status=('forecast' if forecast else 'measured') if value is not None else ('unavailable' if forecast else 'not_measured'),
                    reason=None if value is not None else reason)

    def _technical_summary(self, generation, committed_cursors):
        count=lambda value:str(int(value))
        if self._recovery_summary_override is not None:
            value=deepcopy(self._recovery_summary_override)
            value.update(generation=count(generation),closed=self.closed)
            value['durability']['uncommitted_resource_batch_at_risk']=not self.closed
            return value
        cpu=self.cpu_coverage.summary()
        cpu['unidentified_samples']=count(cpu['unidentified_samples'])
        cpu['profile_samples_excluded']=count(cpu['profile_samples_excluded'])
        for row in cpu['allocations']:
            for key in ('samples','valid_intervals','missing_samples','discontinuous_intervals','coarse_intervals',
                        'first_sample_monotonic_ns','last_sample_monotonic_ns'):
                if row[key] is not None:row[key]=count(row[key])
            row.setdefault('cpu_unavailable_reason',None);row.setdefault('capacity_unavailable_reason',None)
        local=cpu['allocations'];contributing=[row for row in local if int(row['valid_intervals'])>0]
        totals=self._constants('CpuTotals')
        for key in ('valid_intervals','missing_samples','discontinuous_intervals'):
            totals[key]=count(sum(int(row[key]) for row in local))
        totals['covered_wall_seconds']=math.fsum(row['covered_wall_seconds'] for row in local)
        totals.update(incomplete=False,reasons=[])
        for key in ('cpu_seconds','allocated_cpu_seconds','observed_capacity_cpu_seconds'):
            value=None;reason='no_stable_interval'
            if contributing and all(row[key] is not None for row in contributing):
                try:value=math.fsum(row[key] for row in contributing)
                except OverflowError:value=None
                reason='counter_overflow' if value is None else None
                if value is not None and not math.isfinite(value):value=None;reason='counter_overflow'
            elif contributing:reason='allocation_width_unavailable'
            totals[key]=self._measurement(value,'s',reason=reason)
            if reason=='counter_overflow':totals[key]['status']='error'
        baseline=self._baseline_summary
        if baseline is not None:
            prior=baseline['cpu_totals'];local_valid=int(totals['valid_intervals'])
            for k in ('valid_intervals','missing_samples','discontinuous_intervals'):
                totals[k]=count(int(totals[k])+int(prior[k]))
            totals['covered_wall_seconds']=math.fsum([totals['covered_wall_seconds'],prior['covered_wall_seconds']])
            for k in ('cpu_seconds','allocated_cpu_seconds','observed_capacity_cpu_seconds'):
                values=[];unknown=None;overflow=False
                for valid,item in ((int(prior['valid_intervals']),prior[k]),(local_valid,totals[k])):
                    if valid:
                        if item['value'] is None:
                            unknown=item['reason'];overflow|=item['status']=='error' and 'overflow' in item['reason']
                        else:values.append(item['value'])
                value=None;reason=unknown or 'no_stable_interval'
                if unknown is None and values:
                    try:value=math.fsum(values)
                    except OverflowError:overflow=True
                    if value is not None and not math.isfinite(value):value=None;overflow=True
                if overflow:reason='counter_overflow'
                totals[k]=self._measurement(value,'s',reason=reason)
                if overflow:totals[k]['status']='error'
        coverage=self._constants('Coverage')
        coverage.update(all_observed_sources_consumed=all(r['cursor']['complete_at_observation'] for r in committed_cursors) if committed_cursors else None,
            terminal_tail='unknown',source_closed=None,capture_error_events=count(self.control_counts.get('performance_capture_incomplete',0)),
            gaps=[] if self._coverage_gap is None else [deepcopy(self._coverage_gap)],
            incomplete=bool(self.samples_dropped or self._inherited_dropped or self._coverage_gap),
            reasons=(['endpoint_reset_uncovered'] if self._coverage_gap is not None else [])+(['collector_rows_dropped'] if self.samples_dropped or self._inherited_dropped else []))
        cost=self._constants('ObserverSelfCost')
        for key in ('probe_seconds','read_seconds','hash_seconds','parse_validate_seconds','serialize_seconds','parquet_seconds','flush_sync_seconds'):
            cost[key]=self._measurement(None,'s')
        cost['process_cpu_seconds']=self._measurement(time.process_time()-self._began_cpu,'s')
        for key in ('bytes_read','physical_read_bytes','physical_write_bytes','polls','source_reopens','reverified_bytes'):
            cost[key]=self._measurement(None,'B' if 'bytes' in key else '1')
        cost.update(bytes_written=self._measurement(count(self._metadata_written),'B'),
            flushes=self._measurement(count(generation),'1'),pending_rows=self._measurement(count(self.pending_rows),'1'),
            pending_bytes=self._measurement(count(self.pending_bytes),'B'),buffer_high_water_bytes=self._measurement(None,'B'),
            arrow_io_forecast_bytes=self._measurement(None,'B',forecast=True))
        durability=self._constants('Durability')
        durability.update(maximum_staged_batch_rows=count(self.batch_rows),maximum_synchronous_flush_rows=count(2*self.batch_rows),
            maximum_batch_payload_bytes=count(self.batch_bytes),maximum_buffer_seconds=self.max_buffer_seconds,
            uncommitted_resource_batch_at_risk=not self.closed)
        retention=dict(mode='windows_and_cumulative',retained_bytes=count(self._retained_bytes+sum(a['bytes'] for a in self.artifacts)),retained_source_rows=count(self.accepted_totals['accepted_source_rows']),
            omitted_source_rows='0',omitted_resource_rows='0',omitted_span_rows='0',omissions=[],incomplete=False,reasons=[],pinned_retention_refused=False)
        windows=self._constants('WindowPolicy');windows.update(period_seconds=900.,maximum_batch_rows=count(self.batch_rows),cutoff_tail_segments_per_table='3')
        spans=[]
        for key,duration in self.phase_seconds.items():
            phase,timing=key.rsplit(':',1)
            spans.append(dict(phase=phase,timing_kind=timing,measurement_kind='phase_duration',call_count=count(self.phase_calls[key]),
                duration_seconds=self._measurement(duration,'s'),warmup_calls=count(self.warmup_calls.get(key,0)),
                warmup_seconds=self._measurement(self.warmup_seconds.get(key,0.),'s'),maximum_seconds=None))
        return dict(schema='qcl-negf.telemetry-summary.v1',contract_set=CONTRACT_SET,
            session=dict(source=dict(session_id=self.session_id,clock_domain_id='collector:'+self.session_id,
                utc_anchor_ns=count(self.utc_anchor_ns),monotonic_anchor_ns=count(self.monotonic_anchor_ns)),allocation_id=self.identity.get('allocation_id')),
            transport=deepcopy(self.transport),generation=count(generation),closed=self.closed,baseline=deepcopy(self.baseline),
            source_cursors=deepcopy(committed_cursors),totals=dict(rows={k:count(v+(int(baseline['totals']['rows'][k]) if baseline else 0)) for k,v in self.counts.items()},
                collector_dropped_rows=count(self.samples_dropped+self._inherited_dropped),**{k:count(v) for k,v in self.accepted_totals.items()}),
            rows={k:count(v) for k,v in self.counts.items()},max_observed_rss_bytes=None if self.max_rss_bytes is None else count(self.max_rss_bytes),
            cpu_coverage=cpu,cpu_totals=totals,span_aggregates=spans,producer_phase_aggregates=deepcopy(self.phase_aggregates),
            control_events=[dict(event=k,count=count(v)) for k,v in self.control_counts.items()],coverage=coverage,retention=retention,
            observer_self_cost=cost,recorded_until_monotonic_ns=count(time.monotonic_ns()),durability=durability,science_window_policy=windows)

    def _publish(self) -> Path:
        if self.technical_version == 2:return self._publish2()
        if self._publication_pending is None:
            generation=self.generation+1
            watermark=self.catalog_watermark
            artifact_count=self.artifact_count
            publications=[]
            if self.artifacts:
                artifact_count+=len(self.artifacts)
                catalog=dict(schema=CATALOG_SCHEMA,contract_set=CONTRACT_SET,entries=artifact_count,
                    previous=watermark,artifacts=deepcopy(self.artifacts))
                name=f'catalog-{generation:09d}.json';payload=json_bytes(catalog)
                publications.append((name,payload))
                watermark=dict(path=name,entries=artifact_count,sha256=hashlib.sha256(payload).hexdigest())
            committed={**self.source_cursors,**self.staged_source_cursors}
            summary=self._technical_summary(generation,list(committed.values()))
            validate_telemetry_summary(summary)
            name=f'summary-{generation:06d}.json';payload=json_bytes(summary)
            summary_ref=dict(path=name,bytes=str(len(payload)),sha256=hashlib.sha256(payload).hexdigest(),
                media_type='application/json',schema=summary['schema'])
            summary_ref=self._root_ref(summary_ref)
            publications.append((name,payload))
            commit=dict(schema='qcl-negf.telemetry-commit.v1',contract_set=CONTRACT_SET,session=deepcopy(summary['session']),
                transport=deepcopy(self.transport),generation=str(generation),closed=self.closed,published_unix=time.time(),
                previous=deepcopy(self._previous_ref),baseline=deepcopy(self.baseline),summary=summary_ref,
                catalog=deepcopy(self._catalog_wire_override) if self._catalog_wire_override is not None else self._root_ref(watermark) if watermark is not None else None,
                source_cursors=list(deepcopy(committed).values()),accepted_ranges=deepcopy(self.accepted_ranges))
            kwargs={} if self._previous is None else dict(previous=self._previous,previous_ref=self._previous_ref,
                previous_summary=self._previous_summary,previous_summary_ref=self._previous['summary'])
            validate_telemetry_commit(commit,summary=summary,summary_ref=summary_ref,phase_contexts=self.phase_contexts,resume_contexts=self._resume_contexts,**kwargs)
            name=f'commit-{generation:06d}.json';payload=json_bytes(commit)
            cref=dict(path=name,bytes=str(len(payload)),sha256=hashlib.sha256(payload).hexdigest(),media_type='application/json',schema=commit['schema'])
            cref=self._root_ref(cref)
            publications.append((name,payload))
            accepted_publications = {json_bytes(row['publication']) for row in self.accepted_ranges}
            context_value = dict(commit_ref=cref,
                phase_context_refs=[row for row in self._phase_context_refs
                    if json_bytes(row['publication_ref']) not in accepted_publications],
                resume_context_refs=[dict(commit_ref=row['commit_ref'], summary_ref=row['summary_ref'])
                    for row in self._resume_contexts])
            from .state import _metadata_weight
            # Ordinary publication keeps its existing writer allowance. A public
            # context snapshot is optional if its own prospective encoding and
            # detached copies cannot fit the tighter 1 MiB metadata allowance.
            context_raw = (json_bytes(context_value)
                if 4*_metadata_weight(context_value) <= 1024*1024 else None)
            self._publication_pending=dict(publications=publications,commit=commit,summary=summary,ref=cref,
                watermark=watermark,artifact_count=artifact_count,committed=committed,
                retained_bytes=self._retained_bytes+sum(a['bytes'] for a in self.artifacts),context_raw=context_raw)
        pending=self._publication_pending
        for name,payload in pending['publications']:
            target=self.directory/name
            if target.exists():
                self._check_retained(target,payload)
            else:self._metadata_write(target,payload,immutable=True)
        from .performance_views import _check_bindings
        _check_bindings(self._context_guards)
        pointer=json_bytes(dict(schema='qcl-negf.telemetry-pointer.v1',contract_set=CONTRACT_SET,commit=pending['ref']))
        if self._resume_guards:
            from ._capture_io import regular,stamp
            for path,before in self._resume_guards:
                with regular(path,size=before[2]) as source:
                    if stamp(source,include_change=True)!=before:raise ValueError('resume source changed before pointer publication')
        if self.metadata_budget is not None:self.metadata_budget.write(len(pointer))
        publish_pointer(self.directory/'current.json',pending['ref'])
        self._metadata_written+=len(pointer)
        self._previous=deepcopy(pending['commit']);self._previous_summary=deepcopy(pending['summary']);self._previous_ref=deepcopy(pending['ref'])
        self.generation=int(self._previous['generation']);self.catalog_watermark=pending['watermark'];self.artifact_count=pending['artifact_count']
        self._retained_bytes=pending['retained_bytes']
        self._published_context_raw=pending['context_raw']
        self.source_cursors=deepcopy(pending['committed']);self.staged_source_cursors.clear();self.artifacts=[]
        self.accepted_ranges=[];self.phase_contexts=[];self._phase_context_refs=[];self._resume_contexts=[];self._context_guards={};self._publication_pending=None
        return self.root/self._previous_ref['path']

    def _phase_rollup2(self,summary,catalog):
        prior=self._previous_summary['cumulative']['phases'] if self._previous_summary else None
        roll=deepcopy(prior) if prior else dict(scope='accepted_source_rows_across_sources_no_cross_clock_bounds',entries=[],
            overflow=dict(call_count='0',metrics={k:_empty_accumulation() for k in _PHASE_METRICS}),incomplete=False,reasons=[])
        entries={encode(r['key']):r for r in roll['entries']}
        for representation,row in self._v2_phase_rows:
            key={k:row[k] for k in ('stage','phase','status','task_width','counter_scope')};key['representation']=representation
            token=encode(key)
            empty=dict(key=key,call_count='0',metrics={k:_empty_accumulation() for k in _PHASE_METRICS})
            prospective=dict(roll,entries=[*entries.values(),empty])
            # Reserve duplicated current/previous/cache trees plus external binding,
            # plan and detached return headroom under the explicit v2 object ceiling.
            base=(summary,catalog,self._previous,self._previous_summary,self._catalog,self.phase_contexts,self._resume_contexts)
            if token not in entries and len(entries)<64 and 2*_nodes((base,prospective))+4096<=16384:
                entries[token]=empty
            target=entries.get(token,roll['overflow']);calls=int(row['call_count'])
            target['call_count']=_count(int(target['call_count'])+calls)
            for name in _PHASE_METRICS:
                allowed=('B','bytes') if name=='allocated_bytes' else ('1','') if name=='gc_count' else ('s',)
                found=[m for m in row['metrics'] if m['name']==name]
                m=found[0] if len(found)==1 and found[0]['unit'] in allowed else None
                measured=int(m['measured_calls']) if m else 0;missing=int(m['unavailable_calls']) if m else calls
                known=m['sum'] if m else None
                if isinstance(known,dict):known=int(known['decimal'])
                overflow=measured if measured and known is None else 0;covered=measured-overflow
                if known is not None and name in ('allocated_bytes','gc_count') and (type(known) is not int or not 0<=known<=_UINT64_MAX):
                    missing+=covered;covered=0;known=None
                reasons=(['missing_observation'] if missing else [])+(['counter_overflow'] if overflow else [])
                contribution=dict(known=str(known) if known is not None and name in ('allocated_bytes','gc_count') else known,
                    covered=_count(covered),unavailable=_count(missing),overflow=_count(overflow),reasons=reasons)
                target['metrics'][name]=_accumulate(target['metrics'][name],contribution,name in ('allocated_bytes','gc_count'))
                roll['reasons']=sorted(set(roll['reasons'])|set(reasons))
            if target is roll['overflow']:roll['reasons']=sorted(set(roll['reasons'])|{'phase_key_limit'})
        roll['entries']=[entries[k] for k in sorted(entries)]
        roll['incomplete']=bool(roll['reasons'])
        return roll

    def _descriptor2(self,artifact):
        ref=self._root_ref({k:artifact[k] for k in ('path','sha256','media_type','schema')}|{'bytes':str(artifact['bytes'])})
        table=artifact.get('table');rows=None
        if table is not None:
            if '_staging_rows' in artifact:rows=_count(artifact['_staging_rows'])
            else:
                self._charge_io2(int(ref['bytes']))
                rows=_count(pq.ParquetFile(self.root/ref['path']).metadata.num_rows)
        accepted=[]
        for ar in self.accepted_ranges:
            if ar['retained_segment']==ref:
                accepted.append({k:deepcopy(ar[k]) for k in ('source_key','segment','first_row','stop_row')})
        return dict(ref=ref,role=artifact['role'],table=table,rows=rows,accepted=accepted)

    def _protected2(self,extra=()):
        refs=list(extra)
        if self._previous:
            refs.extend((self._previous_ref,self._previous['summary'],self._previous['catalog']))
        for ar in self.accepted_ranges:
            refs.extend(ar[k] for k in ('retained_publication','retained_segment') if ar[k] is not None)
        for row in self._phase_context_refs:
            refs.extend(row[k] for k in ('retained_publication_ref','retained_segment_ref'))
        for row in self._last_phase_raw.values():
            refs.extend(row[k] for k in ('retained_publication_ref','retained_segment_ref'))
        for pair in self._resume_contexts:refs.extend((pair['commit_ref'],pair['summary_ref']))
        return {r['path']:r for r in refs}

    def _prepare2(self):
        from .state import _metadata_weight
        generation=self.generation+1
        old=self._catalog['artifacts'] if self._catalog else []
        before={r['ref']['path']:deepcopy(r) for r in old};after=deepcopy(before)
        protected=self._protected2()
        removed={r['ref']['path']:r for r in self._window_removals}
        # Historical catalog leaves leave membership through the same authenticated delta.
        for row in old:
            if row['ref']['schema']=='qcl-negf.telemetry-catalog.v3' and row['ref']['path'] not in protected:
                removed[row['ref']['path']]=row
        for path in removed:
            if path in protected:raise ValueError('protected technical member cannot retire')
            after.pop(path)
        for artifact in self.artifacts:
            row=self._descriptor2(artifact);path=row['ref']['path']
            if path in after:
                if after[path]['ref']!=row['ref']:raise ValueError('immutable retained path rebound')
                row['accepted']=after[path]['accepted']+[e for e in row['accepted'] if e not in after[path]['accepted']]
            after[path]=row
        # New ranges can extend an existing retained original without rewriting it.
        for ar in self.accepted_ranges:
            if ar['retained_segment'] is not None:
                row=after[ar['retained_segment']['path']];extent={k:deepcopy(ar[k]) for k in ('source_key','segment','first_row','stop_row')}
                if extent not in row['accepted']:row['accepted'].append(extent)
        if self._previous:
            ref=deepcopy(self._previous['catalog']);after[ref['path']]=dict(ref=ref,role='performance.provenance',table=None,rows=None,accepted=[])
        ordered=lambda rows:sorted(rows,key=lambda a:(a['ref']['path'],a['ref']['sha256']))
        catalog=dict(schema='qcl-negf.telemetry-catalog.v3',contract_set=CONTRACT_SET,transport=deepcopy(self.transport),
                     generation=_count(generation),artifacts=ordered(after.values()))
        introduced=ordered(row for k,row in after.items() if before.get(k)!=row)
        retired=ordered(removed.values())
        def reference(name,payload,schema):
            return self._root_ref(dict(path=name,bytes=str(len(payload)),sha256=hashlib.sha256(payload).hexdigest(),media_type='application/json',schema=schema))
        name=f'catalog-{generation:09d}.json';raw=json_bytes(catalog)
        if len(raw)>65536 or len(after)>128:raise ValueError('bounded retained catalog exhausted; retain_window before adding detail')
        catalog_ref=reference(name,raw,catalog['schema']);publications=[(name,raw)]
        committed={**self.source_cursors,**self.staged_source_cursors}
        summary=self._technical_summary(generation,list(committed.values()))
        summary['schema']='qcl-negf.telemetry-summary.v2';summary.pop('baseline')
        summary['coverage']['detail_scope']='collector_session_local'
        retention=summary['retention'];prior=self._previous_summary
        for k in ('omitted_source_rows','omitted_resource_rows','omitted_span_rows'):
            retention[k]=prior['retention'][k] if prior else '0'
        retention['omitted_source_rows']=_count(int(retention['omitted_source_rows'])+sum(int(e['stop_row'])-int(e['first_row']) for row in retired for e in row['accepted']))
        for table,k in (('resources','omitted_resource_rows'),('spans','omitted_span_rows')):
            retention[k]=_count(int(retention[k])+sum(int(row['rows']) for row in retired if row['table']==table))
        retention['retained_source_rows']=_count(sum(int(e['stop_row'])-int(e['first_row']) for row in after.values() for e in row['accepted']))
        retention['retained_bytes']=_count(sum(int(row['ref']['bytes']) for row in after.values()))
        retention['pinned_retention_refused']=self._pin_refused
        retention['incomplete']=self._pin_refused or any(int(retention[k]) for k in ('omitted_source_rows','omitted_resource_rows','omitted_span_rows'))
        retention['reasons']=(['pin_refused'] if self._pin_refused else [])+(['retention_omission'] if any(int(retention[k]) for k in ('omitted_source_rows','omitted_resource_rows','omitted_span_rows')) else [])
        previous_cost=prior['cumulative']['cost'] if prior else None
        same_observer=previous_cost is None or previous_cost['position']['observer_id']==self._observer_id
        first=int(previous_cost['position']['stop_operation']) if previous_cost and same_observer else 0
        delta=dict(observer_id=self._observer_id,first_operation=_count(first),stop_operation=_count(self._operation_stop),
            as_of_monotonic_ns=_count(time.monotonic_ns()),metrics=deepcopy(self._cost_metrics),arrow_io_forecast_bytes=_count(self._io_forecast_pending),
            publication_tail='not_included',restart_gap=not same_observer)
        cost=dict(scope='observer_process_completed_operations',position=dict(observer_id=self._observer_id,stop_operation=delta['stop_operation']),
            metrics={k:_accumulate(previous_cost['metrics'][k] if previous_cost else _empty_accumulation(),delta['metrics'][k],k not in _COST_FLOATS) for k in _COST_METRICS},
            arrow_io_forecast_bytes=_count((int(previous_cost['arrow_io_forecast_bytes']) if previous_cost else 0)+self._io_forecast_pending),
            uncovered_restarts=_count((int(previous_cost['uncovered_restarts']) if previous_cost else 0)+(not same_observer)),publication_tail='not_included')
        phases=self._phase_rollup2(summary,catalog);reasons=set(prior['cumulative']['coverage']['reasons'] if prior else [])-{'no_observations'}
        if self.prior_capture=='unimported':reasons.add('unimported_capture')
        if not same_observer:reasons.add('collector_restart')
        if summary['coverage']['incomplete']:reasons.add('missing_observation')
        if int(summary['coverage']['capture_error_events']):reasons.add('capture_error')
        if int(summary['totals']['collector_dropped_rows']):reasons.add('missing_observation')
        reasons.update(retention['reasons']);reasons.update(phases['reasons'])
        for metric in cost['metrics'].values():
            if int(metric['unavailable']) or int(metric['overflow']):reasons.update(metric['reasons'])
        prior_gaps=prior['coverage']['gaps'] if prior and prior['session']==summary['session'] else []
        new_gaps=len({encode(g) for g in summary['coverage']['gaps']}-{encode(g) for g in prior_gaps})
        gap_count=(int(prior['cumulative']['coverage']['gap_events']) if prior else 0)+max(new_gaps,int(not same_observer))
        rss=[int(v) for v in (summary['max_observed_rss_bytes'],prior['cumulative']['max_observed_rss_bytes']['value'] if prior else None) if v is not None]
        summary['cumulative']=dict(scope='enclosing_transport_since_v2_genesis',prior_capture=self.prior_capture,cost=cost,phases=phases,
            coverage=dict(gap_events=_count(gap_count),incomplete=bool(reasons),reasons=sorted(reasons)),
            max_observed_rss_bytes=self._measurement(_count(max(rss)) if rss else None,'B'))
        validate_telemetry_summary(summary)
        name=f'summary-{generation:06d}.json';raw=json_bytes(summary);summary_ref=reference(name,raw,summary['schema']);publications.append((name,raw))
        debts={r['path']:r for r in self._previous['window_delta']['pending_unlink']} if self._previous else {}
        debts.update((row['ref']['path'],row['ref']) for row in retired)
        if self._previous and self._previous['previous']:
            debts.update((ref['path'],ref) for ref in self._previous['previous'].values())
        if self._previous:
            debts.update((ref['path'],ref) for row in self._previous['contexts']['phase_context_refs'] for k,ref in row.items() if k.startswith('retained_'))
            debts.update((ref['path'],ref) for pair in self._previous['contexts']['resume_context_refs'] for ref in pair.values())
        debt=[deepcopy(ref) for path,ref in sorted(debts.items()) if path not in protected and path not in after and json_bytes(ref) not in self._unlinked]
        if len(debt)>128:raise ValueError('pending physical unlink debt exhausted')
        commit=dict(schema='qcl-negf.telemetry-commit.v2',contract_set=CONTRACT_SET,session=deepcopy(summary['session']),transport=deepcopy(self.transport),
            generation=_count(generation),closed=self.closed,published_unix=time.time(),
            previous=dict(commit=deepcopy(self._previous_ref),summary=deepcopy(self._previous['summary'])) if self._previous else None,
            summary=summary_ref,catalog=catalog_ref,source_cursors=list(deepcopy(committed).values()),accepted_ranges=deepcopy(self.accepted_ranges),
            contexts=dict(phase_context_refs=deepcopy(self._phase_context_refs),resume_context_refs=[dict(commit=r['commit_ref'],summary=r['summary_ref']) for r in self._resume_contexts]),
            window_delta=dict(introduced=introduced,removed=retired,pending_unlink=debt),cost_delta=delta)
        kwargs={} if self._previous is None else dict(previous=self._previous,previous_ref=self._previous_ref,previous_summary=prior,
            previous_summary_ref=self._previous['summary'],previous_catalog=self._catalog)
        validate_telemetry_commit(commit,summary=summary,summary_ref=summary_ref,catalog=catalog,
            phase_contexts=self.phase_contexts,resume_contexts=self._resume_contexts,**kwargs)
        name=f'commit-{generation:06d}.json';raw=json_bytes(commit);cref=reference(name,raw,commit['schema']);publications.append((name,raw))
        context_raw=json_bytes(dict(commit_ref=cref,phase_context_refs=commit['contexts']['phase_context_refs'],
            resume_context_refs=[dict(commit_ref=r['commit'],summary_ref=r['summary']) for r in commit['contexts']['resume_context_refs']]))
        if _metadata_weight((publications,commit,summary,catalog,self._previous,self._previous_summary,self._catalog,context_raw))>1024*1024:
            raise ValueError('combined writer metadata exceeds 1MiB allowance')
        return dict(publications=publications,commit=commit,summary=summary,ref=cref,catalog=catalog,
            catalog_ref=catalog_ref,committed=committed,context_raw=context_raw)

    def _check_reference2(self,ref):
        from ._capture_io import regular,beneath,stamp
        count=int(ref['bytes']);self._charge_io2(count)
        path=beneath(self.root,ref['path'])
        with regular(path,size=count) as stream:
            before=stamp(stream,include_change=True);raw=stream.read(count)
            if len(raw)!=count or hashlib.sha256(raw).hexdigest()!=ref['sha256'] or stamp(stream,include_change=True)!=before:
                raise ValueError('previous immutable transaction bytes changed before publication')
            now=path.stat(follow_symlinks=False)
            if (now.st_dev,now.st_ino,now.st_size,now.st_mtime_ns,now.st_ctime_ns)!=before:
                raise ValueError('previous immutable transaction path replaced before publication')

    def _publish2(self):
        if self._publication_pending is None:
            self._publication_pending=self._prepare2()
            self._cost_metrics={k:_empty_accumulation() for k in _COST_METRICS}
            self._io_forecast_pending=0
        pending=self._publication_pending
        for name,payload in pending['publications']:
            target=self.directory/name
            if not target.exists():self._metadata_write(target,payload,immutable=True)
            self._check_retained(target,payload)
        from .performance_views import _check_bindings
        _check_bindings(self._context_guards)
        if self._previous:
            for ref in (self._previous_ref,self._previous['summary'],self._previous['catalog']):self._check_reference2(ref)
        pointer=json_bytes(dict(schema='qcl-negf.telemetry-pointer.v2',contract_set=CONTRACT_SET,commit=pending['ref']))
        self._charge_io2(len(pointer))
        publish_pointer(self.root/'current.json',pending['ref'])
        self._metadata_written+=len(pointer)
        index=dict(schema='qcl-negf.performance-index.v3',contract_set=CONTRACT_SET,transport=self.transport,
                   current=dict(commit=pending['ref'],summary=pending['commit']['summary']),retained=[])
        self._metadata_write(self.root/'index.json',json_bytes(index))
        fsync_directory(self.root)
        self._previous=deepcopy(pending['commit']);self._previous_summary=deepcopy(pending['summary']);self._previous_ref=deepcopy(pending['ref'])
        self._catalog=pending['catalog'];self._catalog_ref=pending['catalog_ref'];self.generation=int(self._previous['generation'])
        self._retained_bytes=int(self._previous_summary['retention']['retained_bytes']);self._published_context_raw=pending['context_raw']
        self.source_cursors=deepcopy(pending['committed']);self.staged_source_cursors.clear();self.artifact_count+=len(self.artifacts);self.artifacts=[]
        self.accepted_ranges=[];self.phase_contexts=[];self._phase_context_refs=[];self._resume_contexts=[];self._context_guards={}
        self._v2_phase_rows=[];self._window_removals=[];self._publication_pending=None
        self._unlinked.clear()
        live={row['ref']['path'] for row in self._catalog['artifacts']}
        self._phase_raw={k:v for k,v in self._phase_raw.items() if k in live}
        return self.root/self._previous_ref['path']

    @_observed
    def retain_window(self, *, maximum_retained_bytes, minimum_free_bytes, maximum_io_bytes, protected_refs=()):
        """Retire exact authenticated detail in this exclusively owned local root.

        Caller-held refs constrain the fixture policy; they do not certify native
        RO/export leases. Real capture activation requires the separate TL-O gate.
        """
        if self.technical_version!=2:raise ValueError('retain_window requires explicit technical_version2')
        for value in (maximum_retained_bytes,minimum_free_bytes,maximum_io_bytes):
            if type(value) is not int or value<0:raise ValueError('nonnegative operation bounds required')
        if maximum_io_bytes>4194304:raise ValueError('operation I/O bound exceeds4MiB')
        if not isinstance(protected_refs,tuple) or len(protected_refs)>128:raise ValueError('bounded exact protected refs required')
        common=schema_validator('result-common.schema.json')
        for ref in protected_refs:common.evolve(schema={'$ref':'#/$defs/FileRef','$defs':common.schema['$defs']}).validate(ref)
        with self.lock:
            if self.closed:raise RuntimeError('telemetry session is closed')
            protected=self._protected2(protected_refs)
            free=os.statvfs(self.root);available=free.f_bavail*free.f_frsize
            burst=min(4194304,self.batch_bytes)+1048576
            if available<minimum_free_bytes+burst:
                raise ValueError('scientific free-space reserve unavailable before retention publication')
            self._minimum_free_bytes=minimum_free_bytes
            self.flush()
            used=sum(int(a['ref']['bytes']) for a in self._catalog['artifacts'])
            remove=[]
            for row in self._catalog['artifacts']:
                if used<=maximum_retained_bytes:break
                if row['ref']['path'] in protected:continue
                remove.append(row);used-=int(row['ref']['bytes'])
            self._pin_refused=used>maximum_retained_bytes
            self._window_removals=remove
            # Always advance the publication barrier before touching retired files.
            self._publish()
            spent=0;failed=False
            from ._capture_io import regular,beneath,stamp
            for ref in self._previous['window_delta']['pending_unlink']:
                count=int(ref['bytes'])
                if count>min(maximum_io_bytes-spent,self._io_remaining):continue
                self._charge_io2(count)
                try:
                    target=beneath(self.root,ref['path'])
                    with regular(target,size=count) as stream:
                        before=stamp(stream,include_change=True);raw=stream.read(count);spent+=len(raw)
                        if len(raw)!=count or hashlib.sha256(raw).hexdigest()!=ref['sha256'] or stamp(stream,include_change=True)!=before:
                            raise ValueError('retired physical debt bytes changed')
                        now=target.stat(follow_symlinks=False)
                        if (now.st_dev,now.st_ino,now.st_size,now.st_mtime_ns,now.st_ctime_ns)!=before:raise ValueError('retired debt path replaced')
                    target.unlink();fsync_directory(target.parent)
                    self._unlinked.add(json_bytes(ref))
                    parent=target.parent
                    while parent!=self.root and parent!=self.directory:
                        try:parent.rmdir()
                        except OSError:break
                        parent=parent.parent
                except FileNotFoundError:
                    # An unacknowledged unlink is not authenticated success.
                    failed=True
                except OSError:failed=True
            return deepcopy(self._previous_summary['retention'])

    def _rotate2(self):
        old_path=self.close();old=deepcopy(self._previous);summary=deepcopy(self._previous_summary);ref=deepcopy(self._previous_ref)
        candidate=type(self).__new__(type(self))
        candidate._initialize_writer(self.root,self.identity,self._hardware,batch_rows=self.batch_rows,batch_bytes=self.batch_bytes,
            max_buffer_seconds=self.max_buffer_seconds,metadata_budget=self.metadata_budget,transport=self.transport,publish=False,
            technical_version=2,prior_capture=self.prior_capture)
        for name in ('_observer_id','_operation_stop','_cost_metrics','_catalog','_catalog_ref','_unlinked','_io_forecast_pending','_metadata_written','_forecast_total','_operation_depth','_io_remaining','_cost_cpu_position','_cost_written_position','_minimum_free_bytes'):
            setattr(candidate,name,deepcopy(getattr(self,name)))
        candidate._previous=old;candidate._previous_summary=summary;candidate._previous_ref=ref;candidate.generation=int(old['generation'])
        candidate._baseline_summary=summary;candidate.accepted_totals=deepcopy(self.accepted_totals)
        candidate._inherited_dropped=int(summary['totals']['collector_dropped_rows'])
        candidate._coverage_gap=dict(source_key=None,clock_domain_id=None,kind='detail_page_rotation',first_monotonic_ns=None,last_monotonic_ns=None,reason='endpoint_reset_uncovered')
        candidate._publish();candidate.lock=self.lock
        self.__dict__.update(candidate.__dict__)
        return old_path

    @classmethod
    def _resume2(cls,root,*,commit_ref,expected_session,hardware,batch_rows,batch_bytes,max_buffer_seconds,metadata_budget,prior_capture):
        from ._capture_io import regular,beneath,stamp
        from .commits import decode_json
        root=Path(root);used=0;guards=[]
        def read(ref):
            nonlocal used
            common=schema_validator('result-common.schema.json')
            common.evolve(schema={'$ref':'#/$defs/FileRef','$defs':common.schema['$defs']}).validate(ref)
            count=int(ref['bytes'])
            if count>1048576-used:raise ValueError('bounded resume metadata allowance exceeded')
            with regular(beneath(root,ref['path']),size=count) as stream:
                before=stamp(stream,include_change=True);raw=stream.read(count);used+=len(raw)
                if len(raw)!=count or hashlib.sha256(raw).hexdigest()!=ref['sha256'] or stamp(stream,include_change=True)!=before:raise ValueError('resume bytes changed')
                guards.append((root/ref['path'],before))
            value=decode_json(raw,'retained resume')
            if value.get('schema')!=ref['schema']:raise ValueError('resume reference/content schema differs')
            return value
        commit=read(commit_ref)
        if commit['schema']!='qcl-negf.telemetry-commit.v2' or commit['session']!=expected_session:raise ValueError('resume version/session differs')
        summary=read(commit['summary']);catalog=read(commit['catalog'])
        if summary['cumulative']['prior_capture']!=prior_capture:raise ValueError('prior_capture changed on resume')
        previous=previous_summary=previous_catalog=None;kwargs={}
        if commit['previous']:
            previous=read(commit['previous']['commit']);previous_summary=read(commit['previous']['summary']);previous_catalog=read(previous['catalog'])
            kwargs=dict(previous=previous,previous_ref=commit['previous']['commit'],previous_summary=previous_summary,
                previous_summary_ref=commit['previous']['summary'],previous_catalog=previous_catalog)
        phases=[dict(publication=read(row['retained_publication_ref']),publication_ref=row['publication_ref'],
                     segment=read(row['retained_segment_ref']),segment_ref=row['segment_ref']) for row in commit['contexts']['phase_context_refs']]
        resumes=[dict(commit=read(pair['commit']),commit_ref=pair['commit'],summary=read(pair['summary']),summary_ref=pair['summary']) for pair in commit['contexts']['resume_context_refs']]
        validate_telemetry_commit(commit,summary=summary,summary_ref=commit['summary'],catalog=catalog,phase_contexts=phases,resume_contexts=resumes,**kwargs)
        obj=cls.__new__(cls)
        obj._initialize_writer(root,{},hardware,batch_rows=batch_rows,batch_bytes=batch_bytes,max_buffer_seconds=max_buffer_seconds,
            metadata_budget=metadata_budget,transport=commit['transport'],publish=False,technical_version=2,prior_capture=prior_capture)
        obj._previous=commit;obj._previous_summary=summary;obj._previous_ref=deepcopy(commit_ref);obj._catalog=catalog;obj._catalog_ref=commit['catalog']
        obj.generation=int(commit['generation']);obj._baseline_summary=summary;obj.accepted_totals={k:int(summary['totals'][k]) for k in obj.accepted_totals}
        obj._inherited_dropped=int(summary['totals']['collector_dropped_rows'])
        obj.source_cursors={r['source_key']:deepcopy(r) for r in commit['source_cursors']}
        for row in commit['source_cursors']:
            refs={}
            for name in ('publication','segment'):
                original=row['cursor'][name]
                matches=[r['ref'] for r in catalog['artifacts'] if all(r['ref'][k]==original[k]
                    for k in ('bytes','sha256','media_type','schema'))]
                if matches:refs[name]=deepcopy(min(matches,key=lambda r:r['path']))
            if len(refs)==2:
                obj._last_phase_raw[row['source_key']]=dict(publication_ref=deepcopy(row['cursor']['publication']),
                    segment_ref=deepcopy(row['cursor']['segment']),retained_publication_ref=refs['publication'],retained_segment_ref=refs['segment'])
        obj._coverage_gap=dict(source_key=None,clock_domain_id=None,kind='collector_restart',first_monotonic_ns=None,last_monotonic_ns=None,reason='endpoint_reset_uncovered')
        for path,before in guards:
            with regular(path,size=before[2]) as stream:
                if stamp(stream,include_change=True)!=before:raise ValueError('resume source changed before publication')
        obj._publish()
        return obj

    @_observed
    def close(self) -> Path:
        with self.lock:
            if self.closed:
                return self.directory / f"commit-{self.generation:06d}.json"
            return self._flush(closing=True)

    def close_context(self) -> tuple[dict, tuple[dict, ...], tuple[dict, ...]]:
        """Close durably and detach refs for exactly that transaction's delta.

        Additional phase refs exclude accepted ranges already carried by the
        commit. This snapshot selects bytes, and does not certify their owner.
        """
        from .commits import decode_json
        from .state import _metadata_weight
        with self.lock:
            self.close()  # Pending publication/primary errors retain their identity.
            raw = self._published_context_raw
            if raw is None or len(raw) > 1024*1024:
                raise ValueError('closed context metadata allowance exceeded')
            value = decode_json(raw, 'exact published close context')
            # Debit decoded snapshot, validation/return copies and commit encoding
            # before allocating any detached return or readback payload.
            if (_metadata_weight((raw, value, self._previous)) +
                    2*_metadata_weight(value) + 4*_metadata_weight(self._previous)) > 1024*1024:
                raise ValueError('combined closed context metadata allowance exceeded')
            if len(value['phase_context_refs']) > 128 or len(value['resume_context_refs']) > 16:
                raise ValueError('closed context reference page exceeded')
            self._check_retained(self.root/value['commit_ref']['path'], json_bytes(self._previous))
            # Decoding allocated this detached tree; immutable stored raw is not
            # exposed and repeated calls cannot observe caller mutations.
            return (value['commit_ref'], tuple(value['phase_context_refs']), tuple(value['resume_context_refs']))

    def __enter__(self) -> "TelemetryWriter":
        return self

    def __exit__(self, *error: object) -> None:
        self.close()
