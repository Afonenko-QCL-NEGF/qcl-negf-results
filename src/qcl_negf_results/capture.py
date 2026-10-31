"""One solver reaper and one disposable I/O observer; no solver retry."""

from __future__ import annotations

import hashlib
import os
import re
import signal
import stat
import subprocess
import sys
import time
from copy import deepcopy
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path

from ._atomic_io import atomic_write, fsync_directory
from ._capture_io import (
    beneath,
    copy_closed,
    digest,
    fail,
    integer,
    ObserverIO,
    regular,
    stamp,
    small_json,
)
from .commits import decode_json, json_bytes

MIB = 1024 * 1024
# F4 private retained/owned-buffer operational allowance. Public JSON/FileRef
# bounds stay1MiB. This is neither a transitive callee peak nor an RSS limit.
CAPTURE_METADATA_BYTES = 4 * MIB


@dataclass(frozen=True)
class PhaseCaptureLimits:
    maximum_metadata_bytes: int = MIB
    maximum_segment_bytes: int = 4*MIB
    maximum_rows: int = 128
    maximum_publications: int = 256
    maximum_pending_reference_bytes: int = 4*MIB

    def __post_init__(self):
        if any(type(v) is not int or v<1 for v in vars(self).values()):
            raise ValueError('positive phase capture bounds required')


@dataclass(frozen=True)
class PhaseBatch:
    identity: dict
    source: dict
    segment: dict
    segment_ref: dict
    segment_bytes: bytes
    publication: dict
    publication_ref: dict
    previous: dict | None
    previous_ref: dict | None
    current: dict
    first_row: int
    stop_row: int
    cursor: dict
    route_epoch: int
    io_counters: dict
    publication_bytes: bytes
    previous_bytes: bytes | None


class PhaseCaptureSource:
    """One bounded sealed-source epoch. No journal prefix, global index or solve."""
    def __init__(self,artifact_root,*,descriptor,expected_identity,route_epoch,limits=None):
        if (not isinstance(descriptor,dict) or set(descriptor)!={'status','session_id','current_path','reason'}
                or descriptor['status']!='available' or descriptor['reason'] is not None
                or not isinstance(descriptor['session_id'],str) or not descriptor['session_id']):
            raise ValueError('explicit available producer phase descriptor required')
        from qcl_negf_contracts.artifacts import relative_path
        relative_path(descriptor['current_path'])
        if not integer(route_epoch):raise ValueError('nonnegative route epoch required')
        self.root=Path(artifact_root);self.descriptor=deepcopy(descriptor);self.identity=deepcopy(expected_identity)
        self.route_epoch=route_epoch;self.limits=limits or PhaseCaptureLimits()
        if not isinstance(self.limits,PhaseCaptureLimits):raise TypeError('PhaseCaptureLimits required')
        self._cache={};self._pending=[];self._walk=None;self._stop=None;self._head=None;self._seen=set();self._discovering=False
        stat=self.root.stat();self._root_stamp=(stat.st_dev,stat.st_ino);self.closed=False;self.backlogged=False;self._mode=None;self.observed_closed=None

    def __enter__(self):return self
    def __exit__(self,*args):self.close()
    def close(self):self._cache.clear();self._pending.clear();self.closed=True

    def correlation_capabilities(self):
        """Raw representation capability, never actual metric coverage/acceptance."""
        return dict(per_metric_measured_wall=dict(available=self._mode=='diagnostic',
            reason=None if self._mode=='diagnostic' else 'correlation_not_recorded' if self._mode=='normal' else 'explicit_segment_required'))

    def _guard(self,path,before):
        from qcl_negf_contracts.messages import ContractError
        try:
            st=self.root.stat()
            if (st.st_dev,st.st_ino)!=self._root_stamp:fail('source_changed','phase artifact root replaced')
            with regular(path) as stream:
                if stamp(stream,include_change=True)!=before:fail('source_changed','immutable phase source replaced')
        except (OSError,ContractError) as error:
            raise ContractError('consumed immutable phase source changed','source_changed') from error

    def _read(self,ref,budget):
        from qcl_negf_contracts import schema_validator
        from .state import _metadata_weight,_schema_check
        common=schema_validator('result-common.schema.json')
        _schema_check(common.evolve(schema={'$ref':'#/$defs/FileRef','$defs':common.schema['$defs']}),ref)
        path=beneath(self.root,ref['path']);key=json_bytes(ref)
        if key in self._cache:
            value,raw,before=self._cache[key];self._guard(path,before);return deepcopy(value),raw
        is_segment=ref['schema']=='qcl-negf.producer-phase-segment.v1'
        maximum=self.limits.maximum_segment_bytes if is_segment else self.limits.maximum_metadata_bytes
        size=int(ref['bytes'])
        if size>maximum:fail('budget_exceeded','phase metadata/segment byte cap exceeded')
        with regular(path,size=size) as stream:
            before=stamp(stream,include_change=True);budget.require(size);raw=stream.read(size);budget.account(len(raw))
            if len(raw)!=size or stamp(stream,include_change=True)!=before or digest(raw)!=ref['sha256']:
                fail('source_changed','phase reference bytes changed')
            self._guard(path,before)
        value=decode_json(raw,'sealed phase source')
        if value.get('identity')!=self.identity or value.get('source',{}).get('session_id')!=self.descriptor['session_id']:
            fail('identity_mismatch','phase source identity/session differs')
        old=dict(self._cache)
        if is_segment:
            old={k:v for k,v in old.items() if v[0].get('schema')!='qcl-negf.producer-phase-segment.v1'}
        prospective=(value,raw,before)
        if (_metadata_weight((old,key,prospective,self._pending))+2*_metadata_weight(value))>self.limits.maximum_pending_reference_bytes+self.limits.maximum_segment_bytes:
            fail('budget_exceeded','phase retained decoded context exceeds allowance')
        old[key]=prospective;self._cache=old
        return deepcopy(value),raw

    def harvest(self,cursor,*,budget):
        from qcl_negf_contracts import schema_validator
        from qcl_negf_contracts.telemetry import validate_phase_current,validate_phase_publication,validate_phase_cursor
        from .state import _metadata_weight,_schema_check
        if self.closed:raise RuntimeError('phase source closed')
        before=budget.read_bytes if hasattr(budget,'read_bytes') else budget.bytes_read_total
        current,_=small_json(beneath(self.root,self.descriptor['current_path']),self.limits.maximum_metadata_bytes,budget=budget)
        current_validator=schema_validator('producer-phase-source.schema.json')
        _schema_check(current_validator.evolve(schema={'$ref':'#/$defs/Current','$defs':current_validator.schema['$defs']}),current)
        if current['identity']!=self.identity or current['source']['session_id']!=self.descriptor['session_id']:
            fail('identity_mismatch','phase current owner differs')
        head=current['publication']
        if head is None:
            validate_phase_current(current)
            if cursor is not None or self._head is not None or self._discovering or self._pending:
                # A locally valid empty Current cannot certify an earlier prefix.
                self.observed_closed=None
                fail('source_changed','observed phase head no longer reaches the prior prefix')
            self.observed_closed=current["closed"]
            return None
        head_value,_=self._read(head,budget)
        validate_phase_current(current,publication=head_value,publication_ref=head)
        self.observed_closed=current["closed"]
        if not self._discovering and not self._pending:
            self._head=deepcopy(head);self._walk=deepcopy(head);self._stop=None if cursor is None else deepcopy(cursor['publication'])
            self._seen=set();self._discovering=True
        if self._discovering:
            count=0
            while self._walk!=self._stop:
                if self._walk is None:fail('source_changed','committed phase publication is not an ancestor')
                key=json_bytes(self._walk)
                if key in self._seen:fail('corrupt_result','phase publication cycle')
                if count>=self.limits.maximum_publications:self.backlogged=True;return None
                pub,_=self._read(self._walk,budget)
                validator=schema_validator('producer-phase-source.schema.json')
                _schema_check(validator.evolve(schema={'$ref':'#/$defs/Publication','$defs':validator.schema['$defs']}),pub)
                if _metadata_weight((self._pending,self._seen,key,self._walk))>self.limits.maximum_pending_reference_bytes:
                    fail('budget_exceeded','phase pending reference cap exceeded')
                self._pending.append(deepcopy(self._walk));self._seen.add(key);self._walk=deepcopy(pub['previous']);count+=1
            self._discovering=False;self.backlogged=False
        # A partial old publication must finish before any newer page. A false
        # completion bit with a full physical prefix may advance legitimately.
        chosen=None
        if cursor is not None:
            oldpub,_=self._read(cursor['publication'],budget);oldseg,_=self._read(cursor['segment'],budget)
            validate_phase_cursor(cursor,publication=oldpub,publication_ref=cursor['publication'],segment=oldseg,segment_ref=cursor['segment'])
            if int(cursor['next_row'])<len(oldseg['rows']):chosen=cursor['publication']
        if chosen is None:
            if not self._pending:return None
            chosen=self._pending[-1]
        publication,publication_raw=self._read(chosen,budget);segment,raw=self._read(publication['segment'],budget)
        previous_ref=publication['previous'];previous=None;previous_raw=None
        if previous_ref is not None:previous,previous_raw=self._read(previous_ref,budget)
        validate_phase_publication(publication,segment=segment,segment_ref=publication['segment'],previous=previous,previous_ref=previous_ref)
        first=int(cursor['next_row']) if cursor is not None and cursor['publication']==chosen else 0
        stop=min(len(segment['rows']),first+self.limits.maximum_rows)
        maxima=[int(r['sequence'] if r['kind']=='event' else r['last_sequence']) for r in segment['rows'][:stop]]
        next_cursor=dict(schema='qcl-negf.producer-phase-cursor.v1',contract_set='qcl-negf.results.v2',identity=deepcopy(self.identity),
            source=deepcopy(segment['source']),publication=deepcopy(chosen),segment=deepcopy(publication['segment']),
            next_row=str(stop),last_sequence=str(max(maxima)),complete_at_observation=stop==len(segment['rows']))
        validate_phase_cursor(next_cursor,publication=publication,publication_ref=chosen,segment=segment,segment_ref=publication['segment'])
        if stop==len(segment['rows']) and self._pending and self._pending[-1]==chosen:self._pending.pop()
        after=budget.read_bytes if hasattr(budget,'read_bytes') else budget.bytes_read_total
        keep={json_bytes(r) for r in self._pending+[chosen,publication['segment']]+([previous_ref] if previous_ref else [])}
        keep.add(json_bytes(head))
        self._cache={k:v for k,v in self._cache.items() if k in keep}
        self._mode=segment['mode']
        return PhaseBatch(deepcopy(self.identity),deepcopy(segment['source']),segment,deepcopy(publication['segment']),raw,
            publication,deepcopy(chosen),previous,deepcopy(previous_ref),current,first,stop,next_cursor,self.route_epoch,
            dict(bytes_read_total=after-before),publication_raw,previous_raw)


def _start_ticks(pid):
    with open(f"/proc/{pid}/stat", encoding="ascii") as stream:
        return int(stream.read(16384).rpartition(") ")[2].split()[19])


def proc_sample(
    pid,
    start_ticks,
    *,
    ticks=None,
    proc_root=Path("/proc"),
    now_ns=None,
    wall_seconds=0.0,
    budget=None,
):
    row = {
        "pid": pid,
        "process_start_ticks": start_ticks,
        "monotonic_ns": time.monotonic_ns() if now_ns is None else now_ns,
        "wall_seconds": wall_seconds,
        "cgroup_scope": "root_julia_process_including_threads",
        "cpu_time_seconds": None,
        "rss_bytes": None,
        "peak_rss_bytes": None,
    }
    try:
        if budget is not None:
            budget.require(16384)
        with (Path(proc_root) / str(pid) / "stat").open("rb") as stream:
            payload = stream.read(16384)
        if budget is not None:
            budget.account(len(payload))
        raw = payload.decode("utf-8", errors="replace")
        fields = raw.rpartition(") ")[2].split()
        if int(fields[19]) != start_ticks:
            raise ValueError("PID identity changed")
        clock = os.sysconf("SC_CLK_TCK") if ticks is None else ticks
        row["cpu_time_seconds"] = (int(fields[11]) + int(fields[12])) / clock
    except OSError, ValueError, IndexError:
        row["missing_reason"] = "root_process_missing_or_identity_changed"
        return row
    try:
        if budget is not None:
            budget.require(65536)
        with (Path(proc_root) / str(pid) / "status").open("rb") as stream:
            payload = stream.read(65536)
            if budget is not None:
                budget.account(len(payload))
            for line in payload.decode("ascii", errors="replace").splitlines():
                key, _, value = line.partition(":")
                if key in ("VmRSS", "VmHWM"):
                    number, unit = value.split()
                    if unit != "kB" or not integer(int(number)):
                        raise ValueError("unsupported RSS unit")
                    row["rss_bytes" if key == "VmRSS" else "peak_rss_bytes"] = (
                        int(number) * 1024
                    )
        if row["rss_bytes"] is None:
            row["capability_reasons"] = "root_rss_unavailable"
    except OSError, ValueError:
        row["capability_reasons"] = "root_rss_unavailable"
    return row


def _kill_group(child, signum):
    if child is not None:
        try:
            os.killpg(child.pid, signum)
        except ProcessLookupError:
            pass


def supervise(solver_argv, *, observer_factory=None, drain_seconds=5.0):
    """No filesystem I/O after spawn except local proc. D-state observer is not awaited."""
    if not 0 <= drain_seconds <= 5:
        raise ValueError("observer drain must be in 0..5 seconds")
    child = subprocess.Popen(solver_argv, start_new_session=True)
    observer = None
    read_fd = write_fd = None
    termination = None
    previous = {}

    def forward(signum, frame):
        nonlocal termination
        if termination is None:
            termination = time.monotonic()
        _kill_group(child, signum)

    try:
        for signum in (signal.SIGTERM, signal.SIGINT):
            previous[signum] = signal.signal(signum, forward)
        try:
            start = _start_ticks(child.pid)
            if observer_factory is not None:
                read_fd, write_fd = os.pipe2(os.O_NONBLOCK | os.O_CLOEXEC)
                argv = observer_factory(child.pid, start, read_fd)
                observer = subprocess.Popen(
                    argv, pass_fds=(read_fd,), start_new_session=True
                )
        except OSError, ValueError, IndexError:
            # Julia was already spawned: observational failure can never retry it.
            # Inherited stderr can itself be an uninterruptible NFS write.
            pass
        finally:
            if read_fd is not None:
                os.close(read_fd)
                read_fd = None
        while child.poll() is None:
            if termination is not None and time.monotonic() - termination >= 5:
                _kill_group(child, signal.SIGKILL)
            time.sleep(0.025)
        code = child.wait()
        if write_fd is not None:
            try:
                os.write(write_fd, f"exit {code}\n".encode())
            except BlockingIOError, BrokenPipeError:
                pass
        deadline = time.monotonic() + drain_seconds
        while (
            observer is not None
            and observer.poll() is None
            and time.monotonic() < deadline
        ):
            time.sleep(0.025)
        if observer is not None and observer.poll() is None:
            _kill_group(observer, signal.SIGKILL)
            # Ordinary killed observer gets a short nonblocking reap opportunity;
            # uninterruptible remote I/O cannot hold the solver supervisor forever.
            deadline = time.monotonic() + 0.1
            while observer.poll() is None and time.monotonic() < deadline:
                time.sleep(0.005)
            # No log write/ack: an unreaped observer remains scheduler authority.
        return code if code >= 0 else 128 - code
    finally:
        if write_fd is not None:
            os.close(write_fd)
        for signum, handler in previous.items():
            signal.signal(signum, handler)


def journal_batch(path, cursor, *, source_key, maximum_bytes=65536, maximum_rows=128, budget=None):
    """Validate complete native lines and replay prefix on every source reopen."""
    from qcl_negf_contracts.telemetry import SPAN_FIELDS

    allowed = {key for key, _ in SPAN_FIELDS}
    offset = cursor.get("offset", 0)
    if (
        not integer(offset, 0, 64 * MIB)
        or not 1 <= maximum_bytes <= 65536
        or not 1 <= maximum_rows <= 128
    ):
        raise ValueError("journal cursor/batch exceeds bound")
    with regular(path) as stream:
        size = os.fstat(stream.fileno()).st_size
        if size < offset:
            raise ValueError("native journal truncated")
        suffix = min(maximum_bytes + 1, size - offset)
        if budget is not None:
            budget.require(offset + suffix)
        prefix = hashlib.sha256()
        consumed = 0
        remaining = offset
        while remaining:
            block = stream.read(min(MIB, remaining))
            if budget is not None:
                budget.account(len(block))
            consumed += len(block)
            if not block:
                raise ValueError("native journal prefix truncated")
            prefix.update(block)
            remaining -= len(block)
        if offset and prefix.hexdigest() != cursor.get("prefix_sha256"):
            raise ValueError("native journal prefix changed across source epochs")
        raw = stream.read(suffix)
        if budget is not None:
            budget.account(len(raw))
        consumed += len(raw)
        rows = []
        used = 0
        session = cursor.get("source_session_id")
        sequence = cursor.get("source_sequence", 0)
        clock = cursor.get("clock_domain_id")
        for line in raw.splitlines(keepends=True):
            if (
                not line.endswith(b"\n")
                or used + len(line) > maximum_bytes
                or len(rows) >= maximum_rows
            ):
                break
            if len(line) > 65536:
                raise ValueError("native journal line exceeds cap")
            value = decode_json(line, "native phase line")
            if (
                not isinstance(value, dict)
                or value.get("schema") != "qcl-negf.producer-phase.v2"
                or value.get("contract_set") != "qcl-negf.results.v1"
                or not isinstance(value.get("source_session_id"), str)
                or not value["source_session_id"]
                or not isinstance(value.get("clock_domain_id"), str)
                or not integer(value.get("source_sequence"), 1)
                or value["source_sequence"] != sequence + 1
                or (session is not None and value["source_session_id"] != session)
                or (clock is not None and value["clock_domain_id"] != clock)
            ):
                raise ValueError("native phase session/sequence/clock differs")
            session = value["source_session_id"]
            sequence = value["source_sequence"]
            clock = value["clock_domain_id"]
            row = {k: v for k, v in value.items() if k in allowed}
            row.update(
                measurement_kind="phase_boundary",
                timestamp_kind="measurement",
                source_path=source_key,
                source_offset=offset + used,
            )
            rows.append(row)
            prefix.update(line)
            used += len(line)
        if not rows and len(raw) > maximum_bytes and b"\n" not in raw[:maximum_bytes]:
            raise ValueError("native journal line exceeds cap")
        return (
            rows,
            {
                "offset": offset + used,
                "prefix_sha256": prefix.hexdigest(),
                "source_session_id": session,
                "source_sequence": sequence,
                "clock_domain_id": clock,
                "complete_at_observation": len(raw) == used,
            },
            consumed,
        )



def _capture_reader_settle(budget, reserved, delivered):
    """Consume an admitted reservation; settlement is never a new admission.

    Delivered reads remain spent even when validation or cleanup fails. The
    protected reader cannot deliver beyond its reserved temporary maximum.
    """
    budget.reserved -= reserved
    budget.read_bytes += delivered
    if not 0 <= delivered <= reserved:
        error = ValueError('reader delivery exceeded its admitted reservation')
        primary = sys.exception()
        if primary is None:raise error
        primary.add_note(str(error))


@contextmanager
def _capture_reader_credit(budget, io):
    """Reserve one whole reader operation, including its nested guard checks."""
    before = io.bytes_read_total; maximum = io.maximum
    io.require(1); budget.require(1)
    allowance = min(budget.remaining, maximum-before)
    budget.reserve(allowance)
    io.maximum = before+allowance
    try:
        yield
    finally:
        io.maximum = maximum
        _capture_reader_settle(budget, allowance, io.bytes_read_total-before)

def make_binding(reader,*,telemetry,destination,capture_status,budget=None,_retained=()):
    """Retain the exact returned close with explicit selected source/input contexts."""
    from .state import StateReader
    from .performance_views import TelemetrySelection,PerformanceLimits,_load_telemetry,_check_bindings
    from qcl_negf_contracts.telemetry import validate_telemetry_binding
    if not isinstance(reader,StateReader) or not isinstance(telemetry,TelemetrySelection):
        raise TypeError('explicit StateReader and TelemetrySelection required')
    destination=Path(destination)
    try:relative=str(destination.absolute().relative_to(telemetry.artifact_root.absolute()))
    except ValueError:raise ValueError('binding destination must belong to explicit technical artifact root') from None
    from qcl_negf_contracts.artifacts import relative_path
    relative_path(relative)
    # Explicit retained-v2 closure uses S00's ceiling; public default remains4096.
    limits=_capture_limits(_retained) if _retained else (PerformanceLimits(maximum_objects=16384) if telemetry.commit_ref['schema']=='qcl-negf.telemetry-commit.v2' else PerformanceLimits())
    reservation=None;io=None;before=None;bindings={}
    allowance=limits.maximum_io_bytes
    if budget is not None:
        allowance=min(allowance,budget.remaining)
        if allowance<1:raise ValueError('binding metadata/read allowance unavailable')
        budget.reserve(allowance);reservation=allowance
    try:
        with reader._consumer_operation(allowance,verify_source_stability=True,maximum_stability_bytes=limits.maximum_metadata_bytes) as io:
            before=io.bytes_read_total
            loaded=_load_telemetry(reader,telemetry,limits,bindings,creating_binding=True);data=loaded['input'];commit=loaded['commit'];summary=loaded['summary']
            if _retained:
                copies=(data['expected_transport'],capture_status)
                _capture_credit(*_retained,loaded,bindings,
                    extra_bytes=_capture_decoded_bound(copies)+128*_capture_objects((copies,))+
                        4*sum(len(loaded[key]) for key in ('commit_raw','summary_raw','source_binding_raw'))+
                        3*sys.getsizeof('\U00010000')+8192,
                    extra_objects=_capture_objects((copies,))+64)
            value=dict(schema=('qcl-negf.telemetry-binding.v2' if commit['schema']=='qcl-negf.telemetry-commit.v2' else 'qcl-negf.telemetry-binding.v1'),contract_set='qcl-negf.results.v2',
                scope='selected_state_with_execution_attempt_telemetry',selected_state=reader.source.state,
                transport=deepcopy(data['expected_transport']),capture_status=deepcopy(capture_status),
                commit=dict(ref=data['commit_ref'],raw_utf8=loaded['commit_raw'].decode('utf-8')),
                summary=dict(ref=commit['summary'],raw_utf8=loaded['summary_raw'].decode('utf-8')),
                source_binding=dict(ref=data['source_binding_ref'],raw_utf8=loaded['source_binding_raw'].decode('utf-8')))
            if _retained:_capture_encoding_credit(value,*_retained,loaded,bindings)
            validate_telemetry_binding(value,expected_state=reader.source.state,expected_transport=data['expected_transport'],
                commit=commit,commit_ref=data['commit_ref'],summary=summary,summary_ref=commit['summary'],
                source_binding=loaded['source_binding'],source_binding_ref=data['source_binding_ref'],
                manifest=reader._manifest,manifest_ref=reader.source.state['manifest'],source_transport=data['source_transport'],
                canonical_plan_ref=data['canonical_plan_ref'],expected_producer_provenance=data['expected_producer_provenance'])
            payload=_capture_encode(value,*_retained,loaded,bindings) if _retained else json_bytes(value)
            if len(payload)>MIB:fail('budget_exceeded','retained technical binding metadata cap exceeded')
            _check_bindings(bindings)
    finally:
        if reservation is not None:
            delivered=0 if io is None or before is None else io.bytes_read_total-before
            _capture_reader_settle(budget,reservation,delivered)
    # Both actual source guards have succeeded before the single atomic publication.
    if budget is not None:budget.write(len(payload))
    atomic_write(destination,payload)
    return value


class ScalarPublisher:
    """Actual scratch route only. Direct route points at the existing closed index."""

    def __init__(self, job_root, selection, *, scalar_credit, reserve_bytes, budget=None):
        if not integer(scalar_credit, 0, 64 * MIB) or not integer(reserve_bytes):
            raise ValueError("invalid pre-admitted scalar partition")
        self.job_root = Path(job_root)
        self.selection = selection
        self.maximum = scalar_credit
        self.reserve = reserve_bytes
        self.copied = {}
        self.bytes = 0
        self.disabled = False
        self.metadata = {}
        self.prefixes = {}
        self.io_bytes = 0
        self.last = None
        self.last_publish = -float("inf")
        self.budget = budget

    def publish(self, actual_root, point, *, mode, terminal=False, io_remaining=128*MIB):
        if self.disabled or (not terminal and time.monotonic()-self.last_publish<30):
            return self.last
        from .state import StateReader,ResultReadLimits,_IOBudget,_metadata_weight
        from .export import _selection
        from qcl_negf_contracts.messages import ContractError
        from qcl_negf_contracts.artifacts import CONTRACT_SET
        budget=self.budget if self.budget is not None else ObserverIO(io_remaining)
        initial=budget.used
        class Reads(_IOBudget):
            def require(inner,count):
                super().require(count);budget.require(count)
            def account(inner,count):
                budget.account(count);super().account(count)
        reader=None
        try:
            if mode not in ('direct','scratch','staged'):raise ValueError('unknown actual source route')
            ref=point['selected_state'];identity=ref['identity']
            if identity['point_id']!=point['point_id'] or identity['execution_id']!=self.selection['execution_id'] or int(identity['attempt'])!=point['source_attempt']:
                raise ValueError('selected canonical source differs from explicit point')
            source=Path(actual_root) if point['artifact_root']=='.' else beneath(actual_root,point['artifact_root'])
            reader=StateReader.__new__(StateReader)
            reader._initialize(source,state_ref=ref,expected_identity=identity,limits=ResultReadLimits(),budget=Reads(budget.remaining))
            with reader._consumer_operation(verify_source_stability=True):
                selected,omitted,_=_selection(reader,'science')
                history=reader._manifest['history']
                if history is None:return self.last
                key=digest(json_bytes(identity));previous=self.prefixes.get(key)
                if previous is not None and int(ref['state_sequence'])<int(previous['source_state']['state_sequence']):
                    raise ValueError('selected live source sequence reversed')
                if mode=='scratch':
                    target=self.job_root/'live'/'sources'/key
                    new=[row for path,row in selected.items() if (str(target/path),row['sha256']) not in self.copied]
                    size=sum(int(row['bytes']) for row in new)
                    if len(self.copied)+len(new)>4096 or self.bytes+size>self.maximum:
                        raise ValueError('admitted scalar partition exhausted')
                    if _metadata_weight((self.copied,selected,omitted,self.prefixes))>MIB:
                        raise ValueError('whole-job live metadata allowance exhausted')
                    free=os.statvfs(self.job_root)
                    if free.f_bavail*free.f_frsize<self.reserve+size+MIB:
                        raise ValueError('protected output reserve unavailable')
                    budget.require(2*size+16384)
                    target.mkdir(parents=True,exist_ok=True)
                    for path,row in sorted(selected.items()):
                        destination=beneath(target,path);count=int(row['bytes'])
                        if (str(destination),row['sha256']) in self.copied:
                            # Existing mirror bytes are checked, never trusted by name.
                            with regular(destination,size=count) as stream:
                                before=stamp(stream,include_change=True);budget.require(count)
                                actual=hashlib.sha256()
                                while block:=stream.read(min(MIB,count+1)):
                                    budget.account(len(block));actual.update(block)
                                if stamp(stream,include_change=True)!=before or actual.hexdigest()!=row['sha256']:
                                    fail('source_changed','retained scalar mirror changed')
                            continue
                        destination.parent.mkdir(parents=True,exist_ok=True)
                        temporary=destination.with_name('.copy-'+destination.name)
                        try:
                            budget.write(count)
                            copy_closed(beneath(source,path),temporary,size=count,sha256=row['sha256'],budget=reader._budget)
                            os.link(temporary,destination);fsync_directory(destination.parent)
                        finally:temporary.unlink(missing_ok=True)
                        self.copied[(str(destination),row['sha256'])]=True;self.bytes+=count
                    published=target
                else:
                    # Path.relative_to refuses an out-of-job direct route.
                    source.relative_to(self.job_root);published=source
                descriptor=dict(schema='qcl-negf.live-prefix.v2',contract_set=CONTRACT_SET,
                    source_state=ref,artifact_root=str(published.relative_to(self.job_root)),history=history,
                    route=mode,access_mode='selected_history_copy' if mode=='scratch' else 'direct_selected_source',
                    omitted_by_policy=omitted,canonical_source_complete=False,observed_unix=time.time(),live_publication_incomplete=False)
                payload=json_bytes(descriptor)
                if len(payload)>MIB or _metadata_weight((self.prefixes,descriptor))>MIB:
                    raise ValueError('live descriptor metadata allowance exhausted')
                # End of the same owner operation checks all consumed witnesses
                # before publishing this non-authoritative locator below.
            reader.close();reader=None
            budget.write(len(payload));atomic_write(self.job_root/'live'/'current.json',payload)
            self.last=descriptor;self.last_publish=time.monotonic();self.prefixes[key]=descriptor
            return descriptor
        except (OSError,ValueError,ContractError):
            self.disabled=True
            if self.last is not None:
                self.last={**self.last,'live_publication_incomplete':True};raw=json_bytes(self.last)
                budget.write(len(raw));atomic_write(self.job_root/'live'/'current.json',raw)
            raise
        finally:
            primary=sys.exception()
            if reader is not None:
                try:reader.close()
                except BaseException as cleanup:
                    if primary is None:raise
                    primary.add_note('scalar reader cleanup: '+str(cleanup))
            self.io_bytes+=budget.used-initial


def _spec(path, *, budget=None):
    """Private operational port, written by AiiDA before launch, never HTTP input."""
    budget = budget if budget is not None else ObserverIO(16384)
    value, _ = _capture_small_json(path,budget,(),maximum=16384)
    keys = {
        "schema",
        "selection",
        "plan",
        "capture_token",
        "allocation_id",
        "allocated_logical_cpus",
        "scratch_root",
        "credits",
    }
    from .performance_views import _selection

    native3 = isinstance(value,dict) and value.get('schema') == 'qcl-negf.capture-spec.v3'
    if native3:keys.add('native_performance')
    if (
        not isinstance(value, dict)
        or set(value) != keys
        or value["schema"] not in ("qcl-negf.capture-spec.v1", "qcl-negf.capture-spec.v3")
    ):
        raise ValueError("unsupported capture spec")
    # A valid grant prepays the one-time bounded spec read before revealing itself.
    credits = value["credits"]
    if (not isinstance(credits, dict) or set(credits) != {"performance_bytes", "scalar_bytes", "io_bytes", "reserve_bytes"}
        or not integer(credits["performance_bytes"], 8 * MIB, 32 * MIB)
        or not integer(credits["scalar_bytes"], 0, 64 * MIB)
        or not integer(credits["io_bytes"], 16384) or not integer(credits["reserve_bytes"])):
        raise ValueError("existing admitted observer partitions/startup credit required")
    budget.maximum = credits["io_bytes"]
    _selection(value["selection"])
    if native3:
        performance=value['native_performance']; selection=value['selection']
        if (type(performance) is not dict or set(performance)!={'run_uuid'}
                or not _native_uuid(performance['run_uuid']) or not _native_uuid(selection['calcjob_uuid'])
                or not integer(selection['attempt'],1,9007199254740991)
                or value['allocated_logical_cpus'] is not None or value['scratch_root'] is not None
                or value['allocation_id']!=f"calcjob-process:{selection['calcjob_uuid']}:{selection['attempt']}"
                or credits['performance_bytes']!=32*MIB or credits['scalar_bytes']!=64*MIB
                or credits['io_bytes']>128*MIB):
            raise ValueError('closed native3 ingress differs from original launch')
    for identifier in (value["selection"]["execution_id"],):
        if "/" in identifier or "\\" in identifier or identifier in (".", ".."):
            raise ValueError("execution identity is not one path component")
    token = value["capture_token"]
    if (
        not isinstance(token, str)
        or len(token) != 32
        or any(c not in "0123456789abcdef" for c in token)
    ):
        raise ValueError("invalid launch token")
    width = value["allocated_logical_cpus"]
    if width is not None and not integer(width, 1, 65536):
        raise ValueError("actual allocation width invalid")
    if not isinstance(value["allocation_id"], str) or not value["allocation_id"]:
        raise ValueError("actual allocation identity required")
    plan = value["plan"]
    if (
        not isinstance(plan, dict)
        or set(plan) != {"path", "bytes", "sha256"}
        or not integer(plan["bytes"], 1, MIB)
    ):
        raise ValueError("bounded frozen plan pin required")
    root = Path(path).absolute().parent
    parsed, raw = (_capture_small_json(beneath(root,plan['path']),budget,(value,)) if native3
        else small_json(beneath(root, plan["path"]), MIB, budget=budget))
    if len(raw) != plan["bytes"] or digest(raw) != plan["sha256"]:
        raise ValueError("frozen plan bytes differ")
    rows = parsed.get("points")
    if (
        parsed.get("fingerprint") != value["selection"]["plan_fingerprint"]
        or not isinstance(rows, list)
        or len(rows) > 256
        or any(
            not isinstance(r, dict) or not isinstance(r.get("id"), str) for r in rows
        )
    ):
        raise ValueError("frozen point inventory unavailable")
    known = {
        r["id"]
        for r in rows
        if r.get("execution_id") == value["selection"]["execution_id"]
    }
    if (
        not known
        or any(not p or "/" in p or "\\" in p or p in (".", "..") for p in known)
        or len(known)
        != sum(
            r.get("execution_id") == value["selection"]["execution_id"] for r in rows
        )
    ):
        raise ValueError("frozen execution points invalid")
    scratch = value["scratch_root"]
    if scratch is not None and (
        not isinstance(scratch, str)
        or not Path(scratch).is_absolute()
        or os.path.abspath(scratch) != scratch
    ):
        raise ValueError("trusted scratch parent invalid")
    return root, value, known


def _native_uuid(value):
    return (type(value) is str and re.fullmatch(r'[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}',value) is not None
        and value!='00000000-0000-0000-0000-000000000000')


def _native_capture_inputs(root, spec, native_execution_path, arguments, *, budget):
    """Fixed worker input assertions, never an original creator/access proof.

    The existing native execution document has no selected source transport.
    This ingress can associate resource capture with the enclosing launch, but
    cannot manufacture a selected TelemetrySelection from a native counter.
    """
    from qcl_negf_contracts import schema_validator
    from qcl_negf_contracts.artifacts import relative_path
    from qcl_negf_contracts.messages import scientific_plan
    from .state import _metadata_weight
    from .performance_views import _bounded_objects

    if (not isinstance(arguments, (tuple, list)) or len(arguments) not in (7, 11)
            or any(type(word) is not str for word in arguments)
            or sum(len(word.encode('utf-8')) for word in arguments) > 16384
            or tuple(arguments[:2]) != ('run-plan', 'plan.json')
            or tuple(arguments[3:7]) != ('--execution-id', spec['selection']['execution_id'],
                                      '--source-graph', 'source-graph.json')
            or (len(arguments) == 11 and tuple(arguments[7:]) !=
            ('--native-execution', 'native-execution.json', '--native-calcjob-uuid',
             spec['selection']['calcjob_uuid']))
            or spec['scratch_root'] is not None or spec['plan']['path'] != 'plan.json'):
        raise ValueError('unsupported original native run-plan argv')
    path = Path(os.path.abspath(native_execution_path))
    if path != root / 'native-execution.json':
        raise ValueError('native execution must be the explicit original workdir input')
    native3=spec['schema']=='qcl-negf.capture-spec.v3'
    if native3 and len(arguments)!=11:raise ValueError('native3 requires paired original native CLI operands')
    native, native_raw = (_capture_small_json(path,budget,(spec,arguments),maximum=16384) if native3
        else small_json(path,16384,budget=budget))
    if (set(native) != {'schema', 'request_id', 'execution_id', 'storage', 'source_graph',
                       'code', 'input_plan', 'access_receipt_ref'}
            or native['schema'] != 'qcl-negf.aiida-native-execution.v1'
            or native['execution_id'] != spec['selection']['execution_id']):
        raise ValueError('original native execution input differs')
    plan, plan_raw = (_capture_small_json(root/'plan.json',budget,(native,native_raw,arguments,spec)) if native3
        else small_json(root / 'plan.json', MIB, budget=budget))
    graph, graph_raw = (_capture_small_json(root/'source-graph.json',budget,
        (native,native_raw,plan,plan_raw,arguments,spec),maximum=65536) if native3
        else small_json(root/'source-graph.json',MIB,budget=budget))
    if native3:_capture_credit(native,native_raw,plan,plan_raw,graph,graph_raw,arguments,spec)
    elif _metadata_weight((native, native_raw, plan, plan_raw, graph, graph_raw, arguments, spec)) > MIB:
        raise ValueError('combined native input metadata allowance exhausted')
    _bounded_objects((native, plan, graph), 16384 if native3 else 4096)
    scientific_plan(plan)
    schema_validator('scientific-plan.schema.json').validate(plan)
    graph_validator = schema_validator('model-help.schema.json')
    graph_validator.evolve(schema={'$ref': '#/$defs/SourceGraph', '$defs': graph_validator.schema['$defs']}).validate(graph)
    if (plan['fingerprint'] != spec['selection']['plan_fingerprint']
            or len(plan_raw) != spec['plan']['bytes'] or digest(plan_raw) != spec['plan']['sha256']
            or len({row['path'] for row in graph['components']}) != len(graph['components'])):
        raise ValueError('original native plan/source graph differs')
    uuid = re.compile(r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}')
    def require_uuid(value):
        if type(value) is not str or uuid.fullmatch(value) is None or native3 and not _native_uuid(value):
            raise ValueError('canonical original native UUID required')
    def opaque(value):
        if (type(value) is not str or not 1 <= len(value) <= 256 or value.startswith('/')
                or any(ord(c) < 32 or ord(c) == 127 for c in value)):
            raise ValueError('bounded original native opaque reference required')
    require_uuid(native['request_id']); require_uuid(spec['selection']['calcjob_uuid'])
    opaque(native['access_receipt_ref'])
    for key, raw in (('input_plan', plan_raw), ('source_graph', graph_raw)):
        claim = native[key]
        if (type(claim) is not dict or set(claim) != {'node_uuid', 'bytes', 'sha256'}
                or claim['bytes'] != str(len(raw)) or claim['sha256'] != digest(raw)):
            raise ValueError('original native input whole byte claim differs')
        require_uuid(claim['node_uuid'])
    storage = native['storage']; code = native['code']
    if (type(storage) is not dict or set(storage) != {'namespace_id', 'storage_identity', 'relative_execution_root',
            'computer_uuid', 'execution_binding_uuid', 'retention_policy_id', 'retention_owner_id', 'hold_ref', 'access_binding_ref'}
            or type(code) is not dict or set(code) != {'code_uuid', 'computer_uuid', 'solver_executable', 'release_id', 'build_receipt_ref'}):
        raise ValueError('original native storage/code keys differ')
    for value in (storage['computer_uuid'], storage['execution_binding_uuid'], code['computer_uuid'], code['code_uuid']):
        require_uuid(value)
    atom = schema_validator('result-common.schema.json')
    for value in (storage['namespace_id'], code['release_id']):
        atom.evolve(schema={'$ref': '#/$defs/Id', '$defs': atom.schema['$defs']}).validate(value)
    for value in (storage['storage_identity'], storage['retention_policy_id'], storage['retention_owner_id'],
                  storage['hold_ref'], storage['access_binding_ref'], code['build_receipt_ref']):
        opaque(value)
    if (code['computer_uuid'] != storage['computer_uuid'] or type(code['solver_executable']) is not str
            or re.fullmatch(r'/nix/store/[A-Za-z0-9][A-Za-z0-9+._?-]*/bin/qcl-negf', code['solver_executable']) is None):
        raise ValueError('original native code/storage declaration differs')
    relative_path(storage['relative_execution_root'])
    if native3 and arguments[2] != os.path.abspath(arguments[2]):
        raise ValueError('native3 original output argv must be literal canonical path')
    output = Path(arguments[2]); suffix = Path(storage['relative_execution_root']).parts
    if (not output.is_absolute() or str(output) != os.path.abspath(output)
            or output.parts[-len(suffix):] != suffix or output.is_relative_to(root) or root.is_relative_to(output)):
        raise ValueError('original native output argv/storage suffix differs')
    # The absolute trusted jobs_root and actual InstalledCode/wrapper origins
    # remain controller/Platform gates; suffix/JSON/token are not authority.
    if native3:
        _capture_credit(native,native_raw,plan,plan_raw,graph,graph_raw,arguments,spec,
            extra_bytes=2*_metadata_weight(spec['selection'])+4096,
            extra_objects=2*_capture_objects((spec['selection'],))+32)
    transport = dict(selection=deepcopy(spec['selection']), frozen_plan=dict(
        path='plan.json', bytes=str(len(plan_raw)), sha256=digest(plan_raw),
        schema=plan['schema'], media_type='application/json'))
    if not native3:return output, transport
    if not _native_uuid(storage['execution_binding_uuid']):raise ValueError('original native UUID required')
    root_claim=lambda area,relative:dict(storage=storage,area=area,relative_path=relative)
    retained=(native,native_raw,plan,plan_raw,graph,graph_raw,arguments,spec,transport)
    _capture_credit(*retained,extra_bytes=4096,extra_objects=32)
    inputs=NativePerformanceInputs(provider_selection=dict(run_uuid=spec['native_performance']['run_uuid'],
        **{k:spec['selection'][k] for k in ('execution_id','attempt','calcjob_uuid')}),
        native_execution_uuid=storage['execution_binding_uuid'],runner_anchor_root=root_claim('native_execution','.'),
        technical_root=root_claim('native_execution','performance'),sources=[dict(selected_state=None,artifact_root='.',
            scientific_root=root_claim('native_execution','.'),scientific_directory=str(output),
            canonical_plan_root=root_claim('native_workdir','.'),canonical_plan_directory=str(root),source_transport=transport)],
        _retained=retained)
    return output,transport,inputs


def _locator(root, spec, *, budget=None, native_output=None):
    native3=spec['schema']=='qcl-negf.capture-spec.v3'
    if native3 and budget is None:budget=ObserverIO(16384)
    value, _ = (_capture_small_json(root/'capture-source.json',budget,(spec,),maximum=16384) if native3
        else small_json(root/'capture-source.json',16384,budget=budget))
    keys={'schema','capture_token','execution_id','plan_fingerprint','mode','actual_root','epoch'}
    if native3:keys.add('root_birth')
    if (
        not isinstance(value, dict)
        or set(value) != keys
        or value["schema"] != ('qcl-negf.capture-source.v3' if native3 else 'qcl-negf.capture-source.v1')
        or value["capture_token"] != spec["capture_token"]
        or value["execution_id"] != spec["selection"]["execution_id"]
        or value["plan_fingerprint"] != spec["selection"]["plan_fingerprint"]
        or not integer(value["epoch"])
        or value["mode"] not in ("direct", "scratch", "staged")
        or not isinstance(value["actual_root"], str)
        or (not native3 and not Path(value["actual_root"]).is_absolute())
    ):
        raise ValueError("actual-root locator differs from launch")
    if native3:
        if native_output is None or value['actual_root'] != str(native_output):
            raise ValueError('native3 actual root differs from literal original output')
        birth=value['root_birth']
        fields={'boot_id','mount_namespace_device','mount_namespace_inode','producer_pid','producer_start_ticks','device','inode'}
        if (value['mode']!='direct' or value['epoch']!=0 or native_output is None
                or type(birth) is not dict or set(birth)!=fields or not _native_uuid(birth['boot_id'])):
            raise ValueError('closed native3 birth locator required')
        for name in fields-{'boot_id'}:
            text=birth[name]
            if (type(text) is not str or re.fullmatch(r'0|[1-9][0-9]{0,19}',text) is None
                    or int(text)>2**64-1 or name in ('producer_pid','producer_start_ticks','inode','mount_namespace_inode') and int(text)==0):
                raise ValueError('canonical UInt64 native3 birth field required')
    actual = Path(value["actual_root"])
    if str(actual) != os.path.abspath(actual):
        raise ValueError("actual source root is not canonical")
    if value["mode"] in ("direct", "staged"):
        if actual != (root / "result" if native_output is None else native_output):
            raise ValueError("direct/staged locator outside selected job result")
    elif native_output is not None:
        raise ValueError('native scratch capture is not qualified')
    elif (
        spec["scratch_root"] is None
        or not actual.is_relative_to(Path(spec["scratch_root"]))
        or actual == Path(spec["scratch_root"])
    ):
        raise ValueError("scratch locator outside trusted worker parent")
    return value, actual


class _NativeResultGuard:
    """Finite native3 named/held directory identity; not a physical writer lease."""
    def __init__(self, workdir, output, birth, pid, start_ticks, budget):
        self.entries=[];self.closed=False;self.budget=budget;self.birth=birth
        self.credentials=(os.geteuid(),os.getegid());self.cleanup_reserved=0
        try:
            if (self.credentials[0]==0 or birth['producer_pid']!=str(pid)
                    or birth['producer_start_ticks']!=str(start_ticks)):
                raise ValueError('native3 original worker/producer generation differs')
            self._scope()
            self._chain(workdir);self.root_fd=self._chain(output)
            root=os.fstat(self.root_fd)
            if (str(root.st_dev)!=birth['device'] or str(root.st_ino)!=birth['inode']
                    or (root.st_uid,root.st_gid)!=self.credentials or stat.S_IMODE(root.st_mode)!=0o700):
                raise ValueError('native3 original born root differs')
            self.validate()
        except BaseException:
            self.close(validate=False)
            raise

    def _proc(self, path, maximum):
        _capture_credit(self.entries,self.birth,extra_bytes=maximum+8192,extra_objects=32)
        self.budget.require(maximum+1)
        stream=open(path,'rb',buffering=0)
        try:
            raw=stream.read(maximum+1);self.budget.account(len(raw))
        finally:
            primary=sys.exception()
            try:stream.close()
            except BaseException as cleanup:
                if primary is None:raise
                primary.add_note('native3 process metadata close: '+str(cleanup))
        if len(raw)>maximum:raise ValueError('native3 process metadata exceeds bound')
        return raw

    def _scope(self):
        boot=self._proc('/proc/sys/kernel/random/boot_id',128).decode('ascii').strip()
        ns=os.stat('/proc/self/ns/mnt')
        if (boot!=self.birth['boot_id'] or str(ns.st_dev)!=self.birth['mount_namespace_device']
                or str(ns.st_ino)!=self.birth['mount_namespace_inode']
                or (os.geteuid(),os.getegid())!=self.credentials):
            raise ValueError('native3 worker boot/namespace/credentials changed')

    def _mount(self, fd):
        raw=self._proc(f'/proc/self/fdinfo/{fd}',4096)
        rows=re.findall(rb'^mnt_id:\s*([0-9]+)$',raw,re.M)
        if len(rows)!=1:raise ValueError('native3 held mount identity unavailable')
        return int(rows[0])

    @staticmethod
    def _stamp(value):
        return value.st_dev,value.st_ino,value.st_uid,value.st_gid,stat.S_IMODE(value.st_mode)

    def _hold(self, parent, name):
        _capture_credit(self.entries,self.birth,extra_bytes=16384,extra_objects=32)
        fd=os.open(name,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW|os.O_CLOEXEC,dir_fd=parent)
        entry=[fd,parent,name,None,None];self.entries.append(entry)
        current=os.fstat(fd);entry[3]=self._stamp(current);entry[4]=self._mount(fd)
        if (not stat.S_ISDIR(current.st_mode) or current.st_uid not in (0,self.credentials[0])
                or stat.S_IMODE(current.st_mode)&0o022):
            raise ValueError('native3 original parent owner/mode differs')
        return fd

    def _chain(self, path):
        path=Path(path)
        if not path.is_absolute() or str(path)!=os.path.abspath(path) or len(str(path).encode())>4096:
            raise ValueError('native3 original parent path exceeds bound')
        fd=self._hold(None,'/')
        for part in path.parts[1:]:fd=self._hold(fd,part)
        return fd

    @property
    def validation_bytes(self):
        return 129+2*len(self.entries)*4097

    def validate(self):
        if self.closed:raise ValueError('native3 directory handle closed')
        self._scope()
        for fd,parent,name,expected,mount in self.entries:
            if self._stamp(os.fstat(fd))!=expected or self._mount(fd)!=mount:
                raise ValueError('native3 held directory identity/credentials/mount changed')
            named=os.open(name,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW|os.O_CLOEXEC,dir_fd=parent)
            try:
                if self._stamp(os.fstat(named))!=expected or self._mount(named)!=mount:
                    raise ValueError('native3 selected directory name/mount changed')
            finally:
                primary=sys.exception()
                try:os.close(named)
                except BaseException as cleanup:
                    if primary is None:raise
                    primary.add_note('native3 named validation close: '+str(cleanup))
        ns=os.stat('/proc/self/ns/mnt')
        if (str(ns.st_dev),str(ns.st_ino))!=(self.birth['mount_namespace_device'],self.birth['mount_namespace_inode']):
            raise ValueError('native3 namespace changed during validation')

    @contextmanager
    def effect(self, *, new_directories=0):
        self.validate()
        allowance=self.validation_bytes+2*new_directories*4097
        self.budget.reserve(allowance)
        try:yield
        finally:
            primary=sys.exception();self.budget.reserved-=allowance
            try:self.validate()
            except BaseException as cleanup:
                if primary is None:raise
                primary.add_note('native3 effect validation: '+str(cleanup))

    def create_technical(self):
        with self.effect(new_directories=1):
            os.mkdir('performance',0o700,dir_fd=self.root_fd)  # EEXIST never adopted.
            fd=self._hold(self.root_fd,'performance');current=os.fstat(fd)
            self.technical_fd=fd
            if (current.st_uid,current.st_gid)!=self.credentials or stat.S_IMODE(current.st_mode)!=0o700:
                raise ValueError('native3 fresh technical directory credentials differ')
            os.fsync(fd);os.fsync(self.root_fd)
            self.budget.reserve(self.validation_bytes);self.cleanup_reserved=self.validation_bytes

    def close(self, *, validate=True):
        if self.closed:return
        primary=sys.exception();error=None
        self.budget.reserved-=self.cleanup_reserved;self.cleanup_reserved=0
        try:
            if validate:self.validate()
        except BaseException as cleanup:error=cleanup
        self.closed=True
        for fd,*_ in reversed(self.entries):
            try:os.close(fd)
            except BaseException as cleanup:
                if error is None:error=cleanup
                else:error.add_note('native3 held close: '+str(cleanup))
        if error is not None:
            if primary is None:raise error
            primary.add_note('native3 final validation/close: '+str(error))


def _outward_point(row, selection, known, *, current=False):
    point=row['point_id'] if current else row['id']
    if row['execution_id']!=selection['execution_id'] or point not in known:
        raise ValueError('outward point differs from frozen execution')
    data=row if current else row['data'];source=data['native_source']
    state=source['selected_state']
    return dict(point_id=point,source_attempt=None if state is None else int(state['identity']['attempt']),
        artifact_root=source['artifact_root'],selected_state=state,phase_source=data['phase_source'])


def _current(actual, selection, known, *, budget=None):
    from qcl_negf_contracts.outward import validate_outward_result
    value,_=small_json(actual/'current_point.json',16384,budget=budget)
    checked=validate_outward_result(value)
    if checked['schema']!='qcl-negf-current-point-v2':raise ValueError('explicit current-point-v2 required')
    return _outward_point(checked,selection,known,current=True)


def _final_points(actual, selection, known, *, budget=None):
    from qcl_negf_contracts.outward import validate_outward_result
    value,_=small_json(actual/'series_result.json',MIB,budget=budget)
    checked=validate_outward_result(value)
    if checked['schema']!='qcl-negf-series-result-v4' or checked['plan_fingerprint']!=selection['plan_fingerprint']:
        raise ValueError('explicit selected native series-result-v4 required')
    if len(checked['points'])>256:raise ValueError('bounded final point inventory required')
    points=[_outward_point(row,selection,known) for row in checked['points'] if row['execution_id']==selection['execution_id']]
    return points,[p['selected_state'] for p in points if p['selected_state'] is not None]



def _performance_anchor(actual, *, budget, retained=()):
    """Read the fixed T2 current sideband and its exact immutable original.

    This validates local bytes, not original AiiDA origin or current RO access.
    No scientific publication, directory discovery or synthesized StateRef occurs.
    """
    from qcl_negf_contracts.telemetry import validate_performance_anchor
    from .state import _metadata_weight
    from .performance_views import _bounded_objects
    with regular(actual / 'performance-anchor-current.json') as stream:
        before = stamp(stream, include_change=True); count = before[2]
        if count > MIB:fail('budget_exceeded', 'Runner anchor raw byte allowance exhausted')
        _capture_credit(*retained,extra_bytes=count+sys.getsizeof(b''),extra_objects=1)
        budget.require(count); raw = stream.read(count); budget.account(len(raw))
        if len(raw) != count or stamp(stream, include_change=True) != before:
            fail('source_changed', 'Runner current anchor changed during read')
    value = _capture_decode(raw, *retained, source='original Runner anchor')
    _bounded_objects((value, raw), 16384)
    _capture_credit(*retained,value,raw,extra_bytes=2*_metadata_weight(value))
    validate_performance_anchor(value)
    _capture_credit(*retained,value,raw,extra_bytes=4096,extra_objects=16)
    reference = dict(path='performance-anchors/' + digest(raw) + '.json', bytes=str(len(raw)),
        sha256=digest(raw), schema='qcl-negf.performance-anchor.v1', media_type='application/json')
    _capture_credit(*retained,value,raw,reference,extra_bytes=len(raw)+sys.getsizeof(b''),extra_objects=1)
    with regular(beneath(actual, reference['path']), size=len(raw)) as stream:
        before = stamp(stream, include_change=True)
        budget.require(len(raw))
        original = stream.read(len(raw)); budget.account(len(original))
        if original != raw or stamp(stream, include_change=True) != before:
            fail('source_changed', 'immutable Runner anchor differs from current bytes')
    return value, reference, raw


def _capture_objects(values):
    from .performance_views import _bounded_objects
    return _bounded_objects(values, 16384)


def _capture_credit(*values, extra_bytes=0, extra_objects=0, maximum_bytes=CAPTURE_METADATA_BYTES):
    from .state import _metadata_weight
    if not 0 < maximum_bytes <= CAPTURE_METADATA_BYTES:raise ValueError('capture metadata bound may only tighten')
    if _metadata_weight(values) + extra_bytes > maximum_bytes:
        fail('budget_exceeded', 'combined performance capture metadata allowance exhausted')
    if _capture_objects(values) + extra_objects > 16384:
        fail('budget_exceeded', 'aggregate performance capture metadata object cap exceeded')


def _capture_path_bound(value):
    """Simultaneously live validate_json diagnostic paths, without making them.

    Each recursive frame has one path string. A child adds one dict separator
    and key, or an array index. Four-byte characters and the largest compact
    header bound every string, even mixed Unicode keys. A failed leaf formats
    its path once more. No caller/callee stack introspection is used.
    """
    header = sys.getsizeof('\U00010000'); peak = 0
    def visit(item, length=1, active=0, depth=0):
        nonlocal peak
        if depth > 64:fail('budget_exceeded','capture metadata depth allowance exhausted')
        active += header + 4*length
        peak = max(peak,active + header + 4*(length+64))
        if type(item) is dict:
            for key,child in item.items():
                if type(key) is not str:fail('invalid_input','JSON string keys required')
                visit(child,length+1+len(key),active,depth+1)
        elif type(item) in (list,tuple):
            for index,child in enumerate(item):visit(child,length+2+len(str(index)),active,depth+1)
    visit(value)
    return peak + 8192


def _capture_raw_workspace(raw):
    """Owned complete-JSON decoder workspace, including legal whitespace.

    Parsed trees/pair hooks use the existing bounded token recipe. In addition,
    validate_json can keep D+1 diagnostic paths live. A path contains at most
    all raw string characters plus D array subscripts (at most8 chars under the
    raw1MiB bound), so whitespace contributes to Unicode but not diagnostic keys.
    Malformed raw is bounded before the decoder too; no validity is inferred.
    """
    from .state import _context_raw_tree_bound
    depth=maximum=characters=punctuation=0; inside=escaped=False
    for byte in raw:
        if inside:
            if byte==34 and not escaped:inside=False
            else:
                characters+=1
                if escaped:escaped=False
                elif byte==92:escaped=True
        elif byte==34:inside=True
        elif byte in (91,123):
            depth+=1;maximum=max(maximum,depth);punctuation+=1
            if maximum>64:fail('budget_exceeded','capture metadata depth allowance exhausted')
        elif byte in (93,125):depth=max(0,depth-1);punctuation+=1
        elif byte in (44,58):punctuation+=1
    tree=_context_raw_tree_bound(raw)
    parser=max(512*(1+punctuation)+4*len(raw),3*tree+4*len(raw)+4096)
    paths=(maximum+1)*(sys.getsizeof('\U00010000')+4*(characters+8*maximum+1))
    return parser+paths+8*characters+8192


def _capture_decode(raw, *retained, source='performance metadata', maximum_bytes=CAPTURE_METADATA_BYTES):
    from .state import _context_raw_tree_bound
    if len(raw)>MIB:fail('budget_exceeded','performance JSON original exceeds byte allowance')
    # Each token/tree unit is at least64B in this conservative scanner. Two
    # forests cover parsed/detached operands; at most65 active diagnostic paths
    # plus Unicode/error/output headers are reserved before entering the codec.
    objects=2*(_context_raw_tree_bound(raw)//64+1)+67
    _capture_credit(raw,*retained,extra_bytes=_capture_raw_workspace(raw),extra_objects=objects,maximum_bytes=maximum_bytes)
    return decode_json(raw,source)


def _capture_decoded_bound(value):
    """Prospective JSON tree with the decoder's key memo, no value aliases.

    Measure only shallow container capacities, using incremental insertion as
    the prepared decoder does. Values are never copied into a second tree.
    """
    keys = set()
    def visit(item):
        if type(item) is dict:
            size = sys.getsizeof(dict.fromkeys(iter(item)))
            for key, child in item.items():
                if key not in keys:keys.add(key);size += sys.getsizeof(key)
                size += visit(child)
            return size
        if type(item) in (list,tuple):
            return sys.getsizeof([None for _ in item])+sum(visit(child) for child in item)
        return sys.getsizeof(item)
    size = visit(value)
    return size + sys.getsizeof(dict.fromkeys(iter(keys))) + max((sys.getsizeof(k) for k in keys),default=0)


def _capture_copy(value, *retained):
    # Detached JSON container capacities plus deepcopy's finite per-node memo.
    # Scalar occurrences deliberately overcount aliases in this scratch bound.
    objects=_capture_objects((value,))
    _capture_credit(value,*retained,extra_bytes=_capture_decoded_bound(value)+128*objects,
        extra_objects=objects)
    return deepcopy(value)


def _capture_text_bound(value, raw_bytes):
    # json.loads(bytes) also owns a temporary Unicode string. Its kind is set by
    # the widest input character; UTF8 bytes conservatively bound character count.
    def widest(item):
        if type(item) is str:return max(map(ord,item),default=0)
        if type(item) is dict:return max((max(widest(k),widest(v)) for k,v in item.items()),default=0)
        if type(item) in (tuple,list):return max(map(widest,item),default=0)
        return 0
    code = widest(value)
    return (1 if code<=255 else 2 if code<=65535 else 4)*raw_bytes+sys.getsizeof('\U00010000')


def _capture_document(inputs, *retained, maximum_bytes=CAPTURE_METADATA_BYTES):
    _capture_credit(inputs._raw, *retained, extra_bytes=inputs._decoded_bound,
        extra_objects=inputs._decoded_objects, maximum_bytes=maximum_bytes)
    value = inputs.document()
    _capture_credit(inputs._raw, value, *retained, maximum_bytes=maximum_bytes)
    return value


def _capture_limits(retained):
    from .state import _metadata_weight
    from .performance_views import PerformanceLimits
    _capture_credit(*retained)
    memory=min(MIB,CAPTURE_METADATA_BYTES-_metadata_weight(retained));objects=16384-_capture_objects(retained)
    if memory<1 or objects<1:fail('budget_exceeded','remaining performance metadata allowance exhausted')
    return PerformanceLimits(maximum_metadata_bytes=memory,maximum_objects=objects)


def _capture_encoding_credit(value, *retained, maximum_bytes=CAPTURE_METADATA_BYTES):
    from .state import _metadata_weight, _context_json_size
    size = _context_json_size(value)
    if size>MIB:fail('budget_exceeded','performance JSON original exceeds byte allowance before encoding')
    # Detachment, Unicode encoder output and UTF8 bytes share the caller forest.
    _capture_credit(value, *retained, extra_bytes=_capture_decoded_bound(value)+_capture_path_bound(value)+_capture_text_bound(value,size)+size+sys.getsizeof(b''),
        extra_objects=_capture_objects((value,))+67,maximum_bytes=maximum_bytes)


def _capture_encode(value, *retained, maximum_bytes=CAPTURE_METADATA_BYTES):
    _capture_encoding_credit(value,*retained,maximum_bytes=maximum_bytes)
    return json_bytes(value)


def _capture_reader_metadata(reader):
    # Protected StateReader validation/library W is separate. Its selected JSON
    # documents and source guards are live metadata shared by this operation.
    names = ('_state_ref','_manifest','_pages','_index','_known_states','_catalog_by_hash',
        '_published_paths','_catalog_by_leaf','_page_proofs','_parent_anchor_cache',
        '_declarations','_verified_refs','_metadata_bindings','_checkpoint')
    return tuple(getattr(reader,name,None) for name in names)


def _capture_small_json(path, budget, retained, *, maximum=MIB):
    with regular(path) as stream:
        before=stamp(stream,include_change=True);count=before[2]
        if count>maximum:fail('budget_exceeded','performance JSON original exceeds allowance')
        _capture_credit(*retained,extra_bytes=count+sys.getsizeof(b''),extra_objects=1)
        budget.require(count);raw=stream.read(count);budget.account(len(raw))
        if len(raw)!=count or stamp(stream,include_change=True)!=before:
            fail('source_changed','performance JSON original changed')
    value=_capture_decode(raw,*retained)
    _capture_credit(*retained,raw,value)
    return value,raw


def _capture_ref(path, raw, schema):
    return dict(path=path, bytes=str(len(raw)), sha256=digest(raw), schema=schema, media_type='application/json')


def _capture_read_admitted(root, reference, budget, *, retained=()):
    if int(reference['bytes'])>MIB:fail('budget_exceeded','performance metadata original exceeds allowance')
    _capture_credit(*retained,extra_bytes=int(reference['bytes'])+sys.getsizeof(b''),extra_objects=1)
    return _capture_read(root,reference,budget)


def _capture_read(root, reference, budget):
    count = int(reference['bytes'])
    if count > MIB:
        fail('budget_exceeded', 'performance metadata original exceeds allowance')
    _capture_credit(extra_bytes=count+sys.getsizeof(b''))
    with regular(beneath(root, reference['path']), size=count) as stream:
        before = stamp(stream, include_change=True); budget.require(count)
        raw = stream.read(count); budget.account(len(raw))
        if len(raw) != count or digest(raw) != reference['sha256'] or stamp(stream, include_change=True) != before:
            fail('source_changed', 'performance metadata original changed')
    return raw


def _capture_write(root, path, raw, schema, budget, *, immutable=False, retained=()):
    # Reference construction and the mandatory readback are admitted before a
    # write can escape; memory refusal must not advance a pointer first.
    if len(raw)>MIB:fail('budget_exceeded','performance publication original exceeds byte allowance')
    _capture_credit(*retained,raw,path,schema,extra_bytes=2048+len(path)+len(schema)+len(raw)+sys.getsizeof(b''))
    reference = _capture_ref(path, raw, schema)
    _capture_credit(*retained,raw,reference,extra_bytes=len(raw)+sys.getsizeof(b''),extra_objects=1)
    target = beneath(root, path)
    if immutable and target.exists():
        if _capture_read_admitted(root, reference, budget,retained=(*retained,raw,reference)) != raw:
            fail('source_changed', 'immutable performance metadata differs')
    else:
        budget.write(len(raw)); atomic_write(target, raw, immutable=immutable)
        if _capture_read_admitted(root, reference, budget,retained=(*retained,raw,reference)) != raw:
            fail('source_changed', 'performance publication readback differs')
    return reference


@dataclass(frozen=True, init=False)
class NativePerformanceInputs:
    """Explicit original operands for the owning factory, never an origin proof.

    Native3 derives these claims from the compact spec and original worker
    inputs. Four genuine original node pins remain the separate T4 binder's job.
    Local DTO consistency cannot certify native origin, RO access or acceptance.
    """
    _raw: bytes
    _decoded_bound: int
    _decoded_objects: int

    def __init__(self, *, provider_selection, native_execution_uuid, runner_anchor_root, technical_root, sources, _retained=()):
        from qcl_negf_contracts import schema_validator
        from qcl_negf_contracts.artifacts import relative_path
        from .state import _schema_check
        value = dict(provider_selection=provider_selection, native_execution_uuid=native_execution_uuid,
            runner_anchor_root=runner_anchor_root, technical_root=technical_root, sources=sources)
        _capture_credit(value,*_retained)
        if type(sources) is not list or len(sources) > 256:
            raise ValueError('bounded explicit original source operands required')
        validator = schema_validator('telemetry-envelope.schema.json')
        def shape(name, item):
            _schema_check(validator.evolve(schema={'$ref': '#/$defs/' + name, '$defs': validator.schema['$defs']}), item)
        shape('PerformanceProviderSelection', provider_selection)
        if not _native_uuid(native_execution_uuid):raise ValueError('canonical original native execution UUID required')
        def root_shape(root):
            shape('PerformanceRoot', root)
            relative_path(root['storage']['relative_execution_root'])
            if root['relative_path'] != '.':relative_path(root['relative_path'])
        for root in (runner_anchor_root, technical_root):root_shape(root)
        if runner_anchor_root['area'] != 'native_execution' or technical_root['area'] not in ('native_workdir','native_execution'):
            raise ValueError('explicit enclosing execution and technical roots required')
        if technical_root['relative_path'] != 'performance':
            raise ValueError('fixed workdir performance root required')
        for source in sources:
            if type(source) is not dict or set(source) != {'selected_state', 'artifact_root', 'scientific_root',
                    'canonical_plan_root', 'scientific_directory', 'canonical_plan_directory', 'source_transport'}:
                raise ValueError('exact original scientific source operands required')
            if source['artifact_root'] != '.':relative_path(source['artifact_root'])
            root_shape(source['scientific_root']);root_shape(source['canonical_plan_root'])
            common = schema_validator('result-common.schema.json')
            if source['selected_state'] is not None:
                _schema_check(common.evolve(schema={'$ref': '#/$defs/StateRef', '$defs': common.schema['$defs']}), source['selected_state'])
            shape('Transport', source['source_transport'])
            for field in ('scientific_directory', 'canonical_plan_directory'):
                if (type(source[field]) is not str or not Path(source[field]).is_absolute()
                        or os.path.abspath(source[field]) != source[field]):
                    raise ValueError('explicit normalized local scientific and plan mappings required')
        raw = _capture_encode(value,*_retained); _capture_credit(value,raw,*_retained)
        object.__setattr__(self, '_raw', raw)
        object.__setattr__(self, '_decoded_bound', _capture_decoded_bound(value)+_capture_text_bound(value,len(raw)))
        # Occurrence count is a conservative bound even when JSON memoizes keys.
        def objects(item):
            if type(item) is dict:return 1+sum(objects(k)+objects(v) for k,v in item.items())
            if type(item) in (list,tuple):return 1+sum(objects(v) for v in item)
            return 1
        object.__setattr__(self, '_decoded_objects', objects(value))

    def document(self):
        import json
        _capture_credit(self._raw, extra_bytes=self._decoded_bound, extra_objects=self._decoded_objects)
        # _raw is immutable, canonical JSON produced after complete validation.
        # A second validate_json copy adds no validation and doubles the live tree.
        return json.loads(self._raw)


class NativePerformanceCapture:
    """Small source-only factory/index owner on explicit original operands.

    Owns no RO lease or AiiDA graph. No newest lookup selects scientific state;
    only remembered exact T2 publications and explicit source operands qualify
    local byte relationships. An index is an assertion input to the T4 gate.
    """
    def __init__(self, workdir, artifact_root, *, writer, transport, inputs, budget, _native_guard=None):
        if not isinstance(inputs, NativePerformanceInputs):
            raise TypeError('NativePerformanceInputs required; callback is not native origin')
        from .telemetry import TelemetryWriter
        if not isinstance(writer, TelemetryWriter):raise TypeError('owning TelemetryWriter required')
        from .state import _metadata_weight
        _capture_credit(inputs._raw,transport,extra_bytes=_metadata_weight(transport),extra_objects=_capture_objects((transport,)))
        self.workdir = Path(workdir); self.actual = Path(artifact_root); self.writer = writer
        self.inputs = inputs; self.transport = _capture_copy(transport,inputs._raw); self.budget = budget
        self.anchors = {}; self.generation = 0
        self._native_guard=_native_guard
        data = _capture_document(inputs, self.transport); selection = transport['selection']
        if any(data['provider_selection'][k] != selection[k] for k in ('calcjob_uuid', 'execution_id', 'attempt')):
            raise ValueError('original provider/enclosing transport differs')
        if writer.metadata_budget is not budget:
            raise ValueError('capture and writer must share one aggregate I/O ledger')
        self._terminal_root=self.actual if data['technical_root']['area']=='native_execution' else self.workdir
        if writer.root != self._terminal_root / 'performance' or writer.transport != transport:
            raise ValueError('writer differs from explicit native workdir/transport')
        self._roots = tuple((path, (path.stat().st_dev, path.stat().st_ino))
            for path in (self.workdir, self.actual, writer.root))
        _capture_credit(data, self.transport)

    def _guard(self):
        if self._native_guard is not None:self._native_guard.validate()
        for path, expected in self._roots:
            current = path.stat()
            if (current.st_dev, current.st_ino) != expected:
                fail('source_changed', 'performance capture root replaced')
        for row in self.anchors.values():
            for path, expected in row['root_identities']:
                current = Path(path).stat()
                if (current.st_dev, current.st_ino) != expected:
                    fail('source_changed', 'performance source root replaced')

    def _write(self, *args, **kwargs):
        with self._native_guard.effect() if self._native_guard is not None else nullcontext():
            return _capture_write(*args, **kwargs)

    def _publication_guard(self, context, *, retained):
        """Recheck held roots and exact source bytes within this publication."""
        self._guard()
        for row in self.anchors.values():
            science = row['anchor']['scientific']; source = row['source']
            _capture_credit(*retained,extra_bytes=2*max(int(row['reference']['bytes']),
                int(science['selected_state']['manifest']['bytes']))+2*sys.getsizeof(b''))
            if _capture_read_admitted(self.actual,row['reference'],self.budget,retained=retained) != row['raw'].encode():
                fail('source_changed','remembered Runner anchor changed')
            if _capture_read_admitted(Path(source['scientific_directory']),science['selected_state']['manifest'],
                    self.budget,retained=retained) != science['manifest_utf8'].encode():
                fail('source_changed','selected manifest changed during source index publication')
            _capture_read_admitted(Path(source['canonical_plan_directory']),source['source_transport']['frozen_plan'],
                self.budget,retained=retained)
        _capture_read_admitted(self.writer.root,context[0],self.budget,retained=retained)
        self._guard()

    def remember(self, anchor, reference, raw):
        from qcl_negf_contracts.telemetry import validate_performance_anchor
        from .state import _metadata_weight
        _capture_credit(self.inputs._raw,self.anchors,self.transport,anchor,reference,raw,
            extra_bytes=2*_metadata_weight(anchor),extra_objects=2*_capture_objects((anchor,)))
        validate_performance_anchor(anchor)
        if _capture_ref('performance-anchors/' + digest(raw) + '.json', raw, anchor['schema']) != reference:
            raise ValueError('exact immutable Runner anchor reference required')
        if _capture_decode(raw,self.inputs._raw,self.anchors,self.transport,anchor,reference) != anchor or _capture_read_admitted(self.actual, reference, self.budget,retained=(self.inputs._raw,self.anchors,self.transport,anchor,reference,raw)) != raw:
            fail('source_changed', 'Runner anchor original differs')
        science = anchor['scientific']
        if science is None:return
        inputs = _capture_document(self.inputs, self.anchors, self.transport, anchor, reference, raw)
        candidates = [s for s in inputs['sources'] if s['artifact_root'] == science['artifact_root']
            and (s['selected_state'] == science['selected_state'] or s['selected_state'] is None
                and science['relation'] == 'produced' and s['source_transport'] == self.transport)]
        if len(candidates) != 1:
            raise ValueError('one explicit original scientific root/transport required')
        source = candidates[0]
        if anchor['origin'] is not None and anchor['origin'] != dict(calcjob_uuid=self.transport['selection']['calcjob_uuid'],
                native_execution_uuid=inputs['native_execution_uuid']):
            raise ValueError('Runner current origin differs from explicit original inputs')
        _capture_credit(inputs,self.inputs._raw,self.anchors,self.transport,anchor,reference,raw,source,
            extra_bytes=4096,extra_objects=32)
        operands = dict(scientific_root=source['scientific_root'], selected_state=science['selected_state'],
            invocation=science['invocation'], source_transport=source['source_transport'], source_origin=science['source_origin'])
        key = digest(_capture_encode(operands,inputs,self.inputs._raw,self.anchors,self.transport,anchor,reference,raw))
        _capture_credit(inputs,self.inputs._raw,self.anchors,self.transport,anchor,reference,raw,source,
            extra_bytes=4*len(raw)+sys.getsizeof('\U00010000')+1024,extra_objects=16)
        roots = []
        for field in ('scientific_directory','canonical_plan_directory'):
            current = Path(source[field]).stat()
            roots.append((source[field],(current.st_dev,current.st_ino)))
        row = dict(anchor=anchor, reference=reference, raw=raw.decode('utf-8'), source=source,
            root_identities=tuple(roots))
        old = self.anchors.get(key)
        if old is not None:
            if old['anchor'] != anchor or old['source'] != source:
                raise ValueError('selector collides with a different Runner/source association')
            return
        if len(self.anchors) >= 256:
            fail('budget_exceeded', 'whole performance anchor inventory exceeds 256')
        _capture_credit(inputs, self.inputs._raw, self.anchors, row, raw, raw)
        _capture_credit(inputs, self.inputs._raw, self.anchors, self.transport, row, raw,
            extra_bytes=_metadata_weight(row),
            extra_objects=_capture_objects((row,)))
        self.anchors[key] = _capture_copy(row,inputs,self.inputs._raw,self.anchors,self.transport,raw)

    def _entry(self, key, row, context, closed, status, *, inputs=None, retained=()):
        from .state import StateReader, ResultReadLimits, _IOBudget, _context_raw_tree_bound
        from .performance_views import TelemetrySelection, PerformanceLimits, read_performance_summary
        anchor = row['anchor']; science = anchor['scientific']; source = row['source']
        state = science['selected_state']
        if int(state['manifest']['bytes'])>MIB:
            fail('budget_exceeded','selected scientific manifest exceeds capture byte allowance before reader initialization')
        if inputs is None:inputs = _capture_document(self.inputs,self.anchors,self.transport,context,status,retained)
        live = (self.inputs._raw,inputs,self.anchors,self.transport,context,status,retained,
            None if self._native_guard is None else self._native_guard.entries)
        _capture_credit(*live)
        root = Path(source['scientific_directory'])
        plan_root = Path(source['canonical_plan_directory'])
        io = _IOBudget(4*MIB); reader = StateReader.__new__(StateReader)
        try:
            with _capture_reader_credit(self.budget, io):
                reader._initialize(root, state_ref=state, expected_identity=state['identity'], limits=ResultReadLimits(maximum_metadata_bytes=MIB), budget=io)
            live += _capture_reader_metadata(reader)
            _capture_credit(*live)
            manifest_raw = _capture_read_admitted(root, state['manifest'], self.budget,retained=live)
            _capture_credit(*live,manifest_raw,extra_bytes=int(state['manifest']['bytes'])+sys.getsizeof(b''))
            if manifest_raw != science['manifest_utf8'].encode('utf-8'):
                fail('source_changed', 'selected scientific manifest differs from original Runner anchor')
            provenance = reader._manifest['provenance']
            _capture_credit(*live,manifest_raw,provenance,extra_bytes=4096,extra_objects=32)
            binding = dict(schema='qcl-negf.telemetry-source-binding.v1', contract_set='qcl-negf.results.v2',
                enclosing_transport=self.transport, source_transport=source['source_transport'], selected_state=state,
                invocation=science['invocation'], relation=science['relation'], canonical_plan=source['source_transport']['frozen_plan'],
                producer_provenance=provenance, plan_relation='identical_frozen_bytes')
            raw = _capture_encode(binding, *live, manifest_raw, provenance)
            _capture_credit(*live, manifest_raw, provenance, binding, raw)
            source_ref = self._write(self.writer.root, 'source-bindings/' + digest(raw) + '.json', raw,
                binding['schema'], self.budget, immutable=True,retained=(*live,manifest_raw,provenance,binding))
            operands = dict(commit_ref=context[0],source_binding_ref=source_ref,
                expected_transport=self.transport,source_transport=source['source_transport'],
                canonical_plan_ref=source['source_transport']['frozen_plan'],expected_producer_provenance=provenance,
                phase_context_refs=list(context[1]),resume_context_refs=list(context[2]))
            _capture_encoding_credit(operands,*live,manifest_raw,provenance,binding,raw,source_ref)
            selected = TelemetrySelection(self.writer.root,canonical_plan_root=plan_root,**operands)
            del operands
            if closed:
                operands = dict(state=state,source=source_ref,context=[context[0],list(context[1]),list(context[2])],status=status)
                name_raw = _capture_encode(operands,*live,manifest_raw,provenance,binding,raw,source_ref,selected._raw)
                name = 'bindings/'+digest(name_raw)+'.json'
                del operands,name_raw
                if not (self.writer.root / name).exists():
                    with self._native_guard.effect() if self._native_guard is not None else nullcontext():
                        make_binding(reader,telemetry=selected,destination=self.writer.root/name,capture_status=status,budget=self.budget,
                            _retained=(*live,manifest_raw,provenance,binding,raw,source_ref,selected._raw))
                value,raw_binding = _capture_small_json(self.writer.root/name,self.budget,
                    (*live,manifest_raw,provenance,binding,raw,source_ref,selected._raw))
                binding_ref = _capture_ref(name,raw_binding,value['schema'])
                _capture_credit(*live,value,raw_binding,binding_ref,selected._raw,
                    extra_bytes=2*_context_raw_tree_bound(selected._raw))
                operands = selected.document();operands['binding_ref'] = binding_ref
                _capture_encoding_credit(operands,*live,manifest_raw,provenance,binding,raw,source_ref,value,raw_binding,selected._raw)
                selected = TelemetrySelection(self.writer.root,canonical_plan_root=plan_root,**operands)
                del value,raw_binding,operands,binding_ref
            with _capture_reader_credit(self.budget, io):
                read_performance_summary(reader, telemetry=selected, limits=_capture_limits(
                    (*live,manifest_raw,provenance,binding,raw,source_ref,selected._raw)))
        finally:
            primary = sys.exception()
            try:reader.close()
            except BaseException as cleanup:
                if primary is None:raise
                primary.add_note('native performance source cleanup: ' + str(cleanup))
        _capture_credit(*live,manifest_raw,provenance,binding,raw,source_ref,selected._raw,
            extra_bytes=2*_context_raw_tree_bound(selected._raw))
        return dict(anchor_key=key, runner_anchor_ref=row['reference'], runner_anchor_utf8=row['raw'],
            runner_anchor_root=inputs['runner_anchor_root'], scientific_root=source['scientific_root'],
            technical_root=inputs['technical_root'], canonical_plan_root=source['canonical_plan_root'], telemetry=selected.document())

    def publish(self, *, close_context=None, capture_status=None):
        from qcl_negf_contracts.telemetry import validate_performance_source_index, validate_performance_source_pointer
        self._guard()
        from .state import _metadata_weight, _context_raw_tree_bound
        before = (self.inputs._raw,self.anchors,self.transport,close_context,capture_status,
            None if self._native_guard is None else self._native_guard.entries)
        # Snapshot before the large operands decode. T0 owns the snapshot, but
        # its metadata scratch still shares this caller's remaining allowance.
        if close_context is None:
            stored = self.writer._published_context_raw
            _capture_credit(*before,stored,self.writer._previous,
                extra_bytes=2*_context_raw_tree_bound(stored)+4*_metadata_weight(self.writer._previous))
            context = self.writer.published_context()
        else:
            _capture_credit(*before,extra_bytes=_metadata_weight(close_context),extra_objects=_capture_objects((close_context,)))
            context = _capture_copy(close_context,*before)
        inputs = _capture_document(self.inputs,*before,context)
        live = (self.inputs._raw,inputs,self.anchors,self.transport,context,close_context,capture_status,
            None if self._native_guard is None else self._native_guard.entries)
        commit_raw = _capture_read_admitted(self.writer.root, context[0], self.budget,retained=live)
        commit = _capture_decode(commit_raw,*live)
        del commit_raw
        live += (commit,)
        closed = commit['closed']
        if (close_context is not None) != closed:
            raise ValueError('closed publication requires the exact owner close context')
        if commit['transport'] != self.transport:
            raise ValueError('acknowledged technical transport differs')
        entries = []
        for key, row in self.anchors.items():
            _capture_credit(self.inputs._raw, inputs, self.anchors,self.transport, entries, row, context,commit,capture_status)
            entries.append(self._entry(key, row, context, closed, capture_status,inputs=inputs,
                retained=(commit,entries,capture_status,close_context)))
        value = dict(schema='qcl-negf.performance-source-index.v1', contract_set='qcl-negf.results.v2',
            provider_selection=inputs['provider_selection'], generation=str(self.generation+1), closed=closed,
            technical=dict(root=inputs['technical_root'], commit_ref=context[0], phase_context_refs=list(context[1]), resume_context_refs=list(context[2])),
            entries=entries)
        _capture_credit(self.inputs._raw, inputs, self.anchors,self.transport,context,commit,capture_status, value)
        # Retained operands and conservative call buffers share capture W.
        # F4 does not qualify all transitive Contracts temporaries under W.
        # The public encoded index cap is checked before the validator's encoder.
        _capture_encoding_credit(value,*live)
        copied = 2*_metadata_weight(value)
        copy_objects = 2*_capture_objects((value,))
        for row in self.anchors.values():
            _capture_credit(*live,value,extra_bytes=int(row['anchor']['scientific']['selected_state']['manifest']['bytes'])+sys.getsizeof(b''))
            copied += 2*_capture_decoded_bound(row['anchor'])
            manifest_bound = _context_raw_tree_bound(row['anchor']['scientific']['manifest_utf8'].encode())
            copied += 2*manifest_bound
            copy_objects += 2*_capture_objects((row['anchor'],))+manifest_bound//64+1
        _capture_credit(*live,value,extra_bytes=copied,extra_objects=copy_objects)
        validate_performance_source_index(value)
        raw = _capture_encode(value, self.inputs._raw, inputs, self.anchors,self.transport,context,commit,capture_status)
        _capture_credit(self.inputs._raw, inputs, self.anchors,self.transport,context,commit,capture_status, value, raw)
        if len(raw) > MIB:fail('budget_exceeded', 'whole performance source index exceeds byte bound')
        self._publication_guard(context,retained=(*live,value,raw))
        # F4 prepare every own suffix object/encoding before the first immutable
        # index write. The stable retained roots plus the conservative suffix
        # reserve dominate each existing _capture_write/readback preflight.
        # This does not qualify transitive validators or a process RSS peak.
        _capture_credit(*live,value,raw,extra_bytes=8192,extra_objects=64)
        index_path='sources/' + digest(raw) + '.json'
        reference = _capture_ref(index_path,raw,value['schema'])
        pointer = dict(schema='qcl-negf.performance-source-pointer.v1', index=reference)
        pointer_raw = _capture_encode(pointer,self.inputs._raw,inputs,self.anchors,self.transport,context,commit,capture_status,value,raw,reference)
        validate_performance_source_pointer(pointer)
        _capture_credit(*live,value,raw,reference,pointer,pointer_raw,extra_bytes=_metadata_weight(reference),extra_objects=_capture_objects((reference,)))
        returned = _capture_copy(reference,*live,value,raw,pointer,pointer_raw)
        final = (*live,value,raw,reference,pointer,pointer_raw,returned)
        # Known fixed suffix paths/schema/ref constructions, tuple/header/file
        # buffering and one active same-open readback. Two payload capacities
        # also cover the current helper's prospective raw/readback reservations.
        guard_bytes = int(context[0]['bytes']); guard_capacity = guard_bytes
        for row in self.anchors.values():
            counts = (int(row['reference']['bytes']),int(row['anchor']['scientific']['selected_state']['manifest']['bytes']),
                int(row['source']['source_transport']['frozen_plan']['bytes']))
            guard_bytes += sum(counts); guard_capacity = max(guard_capacity,*counts)
        _capture_credit(*final,extra_bytes=131072+2*max(len(raw),len(pointer_raw),guard_capacity)
            +4*_metadata_weight((reference,pointer)),extra_objects=128)
        # One synchronous owner uses this ledger. Admit every remaining write,
        # delivered readback and source check before the first index effect;
        # existing helpers charge actual bytes without refunds or a second debit.
        # The finally check also covers a failed write/check, preserving its
        # primary exception. Successful open/closed suffixes use two/three checks.
        self.budget.require(2*(len(raw)+len(pointer_raw)+(len(raw) if closed else 0))
            +(3 if closed else 2)*guard_bytes
            +(0 if self._native_guard is None else (12 if closed else 8)*self._native_guard.validation_bytes))
        try:
            self._write(self.writer.root,index_path,raw,value['schema'],self.budget,immutable=True,retained=final)
            self._publication_guard(context,retained=final)
            self._write(self.writer.root, 'current-source.json', pointer_raw, pointer['schema'], self.budget,retained=final)
            if closed:
                self._publication_guard(context,retained=final)
                self._write(self._terminal_root, 'performance-source.json', raw, value['schema'], self.budget,retained=final)
        finally:
            primary = sys.exception()
            try:self._publication_guard(context,retained=final)
            except BaseException as cleanup:
                if primary is None:raise
                primary.add_note('native performance publication source cleanup: '+str(cleanup))
        self.generation += 1
        return returned

def observe(spec_path, pid, start_ticks, control_fd, *, transport=None, telemetry_selection_factory=None,
            native_execution_path=None, solver_arguments=(), native_performance_inputs=None):
    """Only this disposable process imports Arrow or touches job/scratch/NFS files."""
    budget = ObserverIO(16384)
    try:
        root, spec, known = _spec(spec_path, budget=budget)
    except BaseException as error:
        try:os.close(control_fd)
        except BaseException as cleanup:error.add_note('observer startup control close: '+str(cleanup))
        if not isinstance(error,Exception):raise
        return 2 # No traceback/inherited-log write before capture credit.
    selection = spec["selection"]
    native_output = None
    performance_capture = None
    native_guard = None
    try:
        if native_performance_inputs is not None and (native_execution_path is None or not isinstance(native_performance_inputs, NativePerformanceInputs)):
            raise ValueError('explicit typed native operands require the native ingress')
        if native_execution_path is not None:
            if transport is not None or telemetry_selection_factory is not None:
                raise ValueError('native ingress cannot accept caller transport/factory assertions')
            if spec['schema']=='qcl-negf.capture-spec.v3' and native_performance_inputs is not None:
                raise ValueError('native3 factory operands are constructed from original ingress only')
            operands = _native_capture_inputs(root, spec, native_execution_path,solver_arguments,budget=budget)
            native_output, transport = operands[:2]
            if len(operands)==3:native_performance_inputs=operands[2]
        elif spec['schema']=='qcl-negf.capture-spec.v3':
            raise ValueError('native3 requires the paired original native ingress')
        if telemetry_selection_factory is not None and not callable(telemetry_selection_factory):
            raise ValueError('trusted programmatic telemetry input factory required')
        if transport is not None:
            from qcl_negf_contracts import schema_validator
            from qcl_negf_contracts.messages import encode
            from .state import _schema_check,_metadata_weight
            validator=schema_validator('telemetry-envelope.schema.json')
            _schema_check(validator.evolve(schema={'$ref':'#/$defs/Transport','$defs':validator.schema['$defs']}),transport)
            encode(transport)
            if transport['selection']!=selection or _metadata_weight(transport)>MIB:
                raise ValueError('initial telemetry transport differs from explicit launch selection')
            transport=deepcopy(transport)
    except BaseException as error:
        try:os.close(control_fd)
        except BaseException as cleanup:error.add_note('observer native input control close: '+str(cleanup))
        if not isinstance(error,Exception):raise
        return 2
    credits = spec["credits"]
    began = time.monotonic()
    cursors = {}
    journal_bytes = 0
    previous_epoch = -1
    missing_sources = set()
    closed_sources={}
    old_source_contexts={}
    closed_page_bytes=0
    publisher = ScalarPublisher(
        root,
        selection,
        scalar_credit=credits["scalar_bytes"],
        reserve_bytes=credits["reserve_bytes"],
        budget=budget,
    )
    if native_output is not None:
        # Existing scalar descriptors are workdir-relative. Do not invent an
        # external-root alias or copy route before its owning access contract.
        publisher.disabled = True
    writer = None
    failure = None
    closed = None
    phase_source=None;phase_key=None
    last_actual=None;last_points=[]

    def session_bytes(directory):
        count=0;used=0;directories=[directory]
        while directories:
            current=directories.pop()
            with os.scandir(current) as entries:
                for entry in entries:
                    count+=1
                    if count>4096 or entry.is_symlink():raise ValueError('bounded session inventory exceeded')
                    if entry.is_dir(follow_symlinks=False) and current==directory and entry.name=='raw':
                        directories.append(Path(entry.path));continue
                    if not entry.is_file(follow_symlinks=False):raise ValueError('unexpected session entry')
                    used+=entry.stat(follow_symlinks=False).st_size
        return used

    def admit(estimate=8 * MIB):
        # Software forecast, not atomic filesystem quota/IO enforcement. Conservative
        # 8 MiB burst is inside the previously admitted 32 MiB partition.
        count = 0
        used = 0
        if writer is not None:used=closed_page_bytes+session_bytes(writer.directory)
        free = os.statvfs(native_output if native_guard is not None else root)
        budget.require(estimate)
        if (
            used + estimate > credits["performance_bytes"]
            or free.f_bavail * free.f_frsize < credits["reserve_bytes"] + estimate
        ):
            raise ValueError("observer admitted storage/IO partition exhausted")

    def _writer_call(method, *args, **kwargs):
        # Local finite forecast for unchanged Arrow serialization/footer/hash I/O.
        # Exact JSON metadata has its own before-read/write hook on the same grant.
        admit()
        reservation = 4 * (writer.batch_bytes if writer is not None else 4 * MIB) + MIB
        budget.reserve(reservation)
        def inventory_directory(directory):
            if native_guard is None:return directory
            # Read the original owned Arrow inventory even if its pathname was
            # moved during serialization. No writes or repairs use this FD path.
            selected=Path(f'/proc/self/fd/{native_guard.technical_fd}')/directory.relative_to(writer.root)
            if not selected.is_dir():raise ValueError('native3 original Arrow inventory unavailable')
            return selected
        before_directory=None;before=set()
        try:
            before_directory=inventory_directory(writer.directory) if writer is not None else None
            before = set(before_directory.glob("*.parquet")) if before_directory is not None else set()
            return method(*args, **kwargs)
        finally:
            primary=sys.exception()
            try:
                paths = set(before_directory.glob("*.parquet")) if before_directory is not None else set()
                if writer is not None:paths.update(inventory_directory(writer.directory).glob("*.parquet"))
                forecast = 4 * sum(path.stat().st_size for path in paths - before)
            except BaseException as cleanup:
                # Unknown serialization tail consumes its entire prepaid
                # forecast; delivered metadata already stays on the same ledger.
                budget.settle(reservation,reservation)
                if primary is None:raise
                primary.add_note('observer writer inventory cleanup: '+str(cleanup))
            else:
                try:budget.settle(reservation, forecast)
                except BaseException as cleanup:
                    if primary is None:raise
                    primary.add_note('observer writer accounting cleanup: '+str(cleanup))
                if forecast > reservation:
                    if primary is None:raise ValueError('writer exceeded finite I/O forecast; capture unknown')
                    primary.add_note('writer exceeded finite I/O forecast; capture unknown')

    def writer_call(method, *args, **kwargs):
        with native_guard.effect() if native_guard is not None else nullcontext():
            return _writer_call(method,*args,**kwargs)

    def capture_point(actual, point, mode, terminal, epoch):
        nonlocal journal_bytes,phase_source,phase_key,closed_page_bytes
        state=point['selected_state'];descriptor=point['phase_source']
        identity=point.get('phase_identity') or (None if state is None else state['identity'])
        if identity is None:
            missing_sources.add(point['point_id']);return
        key=digest(json_bytes(dict(identity=identity,session=descriptor['session_id'])))
        if descriptor['status']=='available':
            qualified=(str(actual),key,epoch)
            if phase_key!=qualified:
                if phase_source is not None:phase_source.close()
                phase_source=PhaseCaptureSource(actual,descriptor=descriptor,expected_identity=identity,route_epoch=epoch)
                phase_key=qualified
            while True:
                before=budget.read_bytes
                batch=phase_source.harvest(cursors.get(key),budget=budget)
                journal_bytes+=budget.read_bytes-before
                if journal_bytes>64*MIB:raise ValueError('v1 observer processed-byte credit exhausted')
                closed_sources[key]=phase_source.observed_closed
                if batch is None:
                    if terminal and phase_source.backlogged:continue
                    break
                from qcl_negf_contracts.messages import encode
                source_key=hashlib.sha256(encode(dict(identity=batch.identity,source=batch.source))).hexdigest()
                active={**writer.source_cursors,**writer.staged_source_cursors}
                if source_key not in active and len(active)>=16:
                    completed=deepcopy(writer._last_phase_raw)
                    closed_path=writer_call(writer.rotate)
                    closed_page_bytes+=session_bytes(closed_path.parent)
                    resume=dict(commit_ref=deepcopy(writer.baseline['commit']),summary_ref=deepcopy(writer.baseline['summary']))
                    for source in writer._baseline_summary['source_cursors']:
                        if source['source_key'] in completed:
                            old_source_contexts[source['source_key']]=dict(resume=deepcopy(resume),phase=completed[source['source_key']])
                    from .state import _metadata_weight
                    if _metadata_weight((old_source_contexts,cursors,closed_sources))>4*MIB:
                        raise ValueError('bounded capture page-context allowance exhausted')
                proof=old_source_contexts.get(source_key) if source_key not in writer.source_cursors else None
                writer_call(writer.stage_phase_batch,batch,phase_context_refs=() if proof is None else (proof['phase'],),
                    resume_context_refs=() if proof is None else (proof['resume'],))
                writer_call(writer.flush)
                cursors[key]=deepcopy(batch.cursor)
                from .state import _metadata_weight
                if _metadata_weight(cursors)>MIB:raise ValueError('whole-observer cursor metadata bound exceeded')
                if not terminal:break
        else:missing_sources.add(point['point_id'])
        if state is not None:
            publisher.publish(actual,point,mode=mode,terminal=terminal,io_remaining=budget.remaining)

    def bind_selected(actual,point,close_context,capture_status):
        from .state import StateReader,ResultReadLimits,_IOBudget
        from .performance_views import TelemetrySelection
        close_ref, phase_refs, resume_refs = close_context
        if point['selected_state'] is None:raise ValueError('selected native state required for binding')
        source=actual if point['artifact_root']=='.' else beneath(actual,point['artifact_root'])
        io=_IOBudget(min(budget.remaining,4*MIB));reader=StateReader.__new__(StateReader);charged=0
        try:
            reader._initialize(source,state_ref=point['selected_state'],expected_identity=point['selected_state']['identity'],
                limits=ResultReadLimits(),budget=io)
            budget.account(io.bytes_read_total);charged=io.bytes_read_total
            with reader._consumer_operation(verify_source_stability=True):
                selected=telemetry_selection_factory(reader,deepcopy(close_ref))
            budget.account(io.bytes_read_total-charged);charged=io.bytes_read_total
            if not isinstance(selected,TelemetrySelection) or selected.expected_transport!=transport or selected.commit_ref!=close_ref or selected.artifact_root!=writer.root:
                raise ValueError('factory selection differs from exact close/initial transport/root')
            inputs = selected.document()
            from .state import _metadata_weight
            if (_metadata_weight((inputs, phase_refs, resume_refs)) +
                    2*_metadata_weight((phase_refs, resume_refs))) > MIB:
                raise ValueError('combined terminal close context metadata allowance exhausted')
            for field, extra, key in (('phase_context_refs', phase_refs, 'publication_ref'),
                                      ('resume_context_refs', resume_refs, 'commit_ref')):
                rows = inputs[field]
                for row in extra:
                    matches = [existing for existing in rows if existing[key]['path'] == row[key]['path']]
                    if matches and any(existing != row for existing in matches):
                        raise ValueError('factory context conflicts with exact public close snapshot')
                    if not matches:
                        rows.append(deepcopy(row))
            selected = TelemetrySelection(selected.artifact_root,
                canonical_plan_root=selected.canonical_plan_root, **inputs)
            name='performance-binding.json' if len(last_points)==1 else 'performance-bindings/'+digest(json_bytes(point['selected_state']))+'.json'
            try:
                make_binding(reader,telemetry=selected,destination=writer.root/name,capture_status=capture_status,budget=budget)
            finally:
                # The binder settles its reservation on success and failure.
                charged=io.bytes_read_total
        finally:
            primary=sys.exception()
            try:
                if io.bytes_read_total>charged:budget.account(io.bytes_read_total-charged)
            finally:
                try:reader.close()
                except BaseException as cleanup:
                    if primary is None:raise
                    primary.add_note('observer source cleanup: '+str(cleanup))

    try:
        # Runner publishes this existing token/fullFP locator only after charging
        # its fixed operational capture reserve. A valid spec alone is not credit.
        # Waiting/remote I/O is confined to this disposable observer, never reaper.
        while not (root / "capture-source.json").exists():
            try:
                os.read(control_fd, 64)
                raise ValueError("capture unavailable: no credit locator before terminal")
            except BlockingIOError:
                time.sleep(2)
        locator, _ = _locator(root, spec, budget=budget, native_output=native_output)
        if spec['schema']=='qcl-negf.capture-spec.v3':
            native_guard=_NativeResultGuard(root,native_output,locator['root_birth'],pid,start_ticks,budget)
        from .telemetry import TelemetryWriter
        admit()
        if native_guard is not None:native_guard.create_technical()
        writer = writer_call(TelemetryWriter,
            (native_output if native_guard is not None else root) / "performance",
            {**selection, "allocation_id": spec["allocation_id"]},
            {
                "scope": "root_julia_process_including_threads",
                "allocated_logical_cpus": spec["allocated_logical_cpus"],
            },
            metadata_budget=budget,transport=transport,
            **({'technical_version': 2} if native_performance_inputs is not None else {}),
        )
        if native_performance_inputs is not None:
            performance_capture = NativePerformanceCapture(root, native_output, writer=writer,
                transport=transport, inputs=native_performance_inputs, budget=budget,_native_guard=native_guard)
        while True:
            try:
                os.read(control_fd, 64)
                terminal = True
            except BlockingIOError:
                terminal = False
            sample = proc_sample(
                pid, start_ticks, wall_seconds=time.monotonic() - began, budget=budget
            )
            sample.update(
                allocation_id=spec["allocation_id"],
                execution_id=selection["execution_id"],
                cpu_count=spec["allocated_logical_cpus"],
            )
            admit()
            writer_call(writer.record_resource, sample)
            try:
                if not (root / "capture-source.json").exists():
                    if terminal:
                        raise ValueError("actual source locator never published")
                    admit()
                    writer_call(writer.flush_due)
                    time.sleep(2)
                    continue
                locator, actual = _locator(root, spec, budget=budget, native_output=native_output)
                if native_guard is not None and locator['root_birth']!=native_guard.birth:
                    raise ValueError('native3 original birth locator changed')
                if locator["epoch"] < previous_epoch:
                    raise ValueError("actual source epoch reversed")
                previous_epoch = locator["epoch"]
                has_anchor = (actual / 'performance-anchor-current.json').exists()
                if not terminal and not has_anchor and not (actual / "current_point.json").exists():
                    admit()
                    writer_call(writer.flush_due)
                    time.sleep(2)
                    continue
                if has_anchor:
                    anchor, anchor_ref, anchor_raw = _performance_anchor(actual,budget=budget,
                        retained=() if performance_capture is None else (performance_capture.inputs._raw,performance_capture.anchors,performance_capture.transport))
                    if native_guard is not None:
                        original_origin=dict(calcjob_uuid=selection['calcjob_uuid'],
                            native_execution_uuid=native_performance_inputs.document()['native_execution_uuid'])
                        if (anchor['origin']!=original_origin or anchor['scientific'] is not None
                                and anchor['scientific']['relation']!='produced'):
                            raise ValueError('native3 exact original produced/phase0 origin required')
                    if performance_capture is not None:performance_capture.remember(anchor, anchor_ref, anchor_raw)
                    identity = anchor['phase_identity']; science = anchor['scientific']
                    if identity['execution_id'] != selection['execution_id'] or identity['point_id'] not in known:
                        raise ValueError('Runner phase anchor differs from frozen execution')
                    points = [dict(point_id=identity['point_id'], phase_identity=identity,
                        source_attempt=None if science is None else int(science['selected_state']['identity']['attempt']),
                        artifact_root='.' if science is None else science['artifact_root'],
                        selected_state=None if science is None else science['selected_state'], phase_source=anchor['phase_source'])]
                else:
                    points, _ = (_final_points(actual, selection, known, budget=budget) if terminal
                        else ([_current(actual, selection, known, budget=budget)], None))
                last_actual=actual;last_points=points
                for point in points:
                    capture_point(actual,point,locator['mode'],terminal,locator['epoch'])
            except FileNotFoundError:
                # Most errors are wrapped safely below; no current point at launch
                # is an explicit capture gap, never fabricated rows.
                pass
            admit()
            writer_call(writer.flush_due)
            if terminal:
                admit()
                if performance_capture is None and (telemetry_selection_factory is None or transport is None):
                    writer_call(writer.record_control,dict(event='source_binding_missing',reason='explicit_transport_or_factory_missing'))
                if performance_capture is not None and not performance_capture.anchors:
                    writer_call(writer.record_control, dict(event='scientific_anchor_pending', reason='no_published_scientific_anchor'))
                close_context = writer_call(writer.close_context)
                close_ref, _, _ = close_context
                closed = writer.root / close_ref['path']
                if performance_capture is None and (telemetry_selection_factory is None or transport is None):
                    raise ValueError('source_binding_missing: explicit transport/factory and production ingress policy required')
                if last_actual is None or not last_points:raise ValueError('selected source unavailable for exact close binding')
                complete=not missing_sources and bool(cursors) and all(c['complete_at_observation'] for c in cursors.values())
                status=dict(terminal_tail='closed_observed' if complete and all(closed_sources.values()) else 'known_incomplete' if missing_sources else 'unknown',
                    scalar_live_route=locator['mode'] if not publisher.disabled else 'unavailable',
                    scalar_live_reason='live_publication_incomplete' if publisher.disabled else None,
                    performance_status='available' if complete else 'incomplete',reason=None if complete else 'phase_sources_or_tail_incomplete')
                if performance_capture is not None:
                    performance_capture.publish(close_context=close_context, capture_status=status)
                    if not performance_capture.anchors:
                        raise ValueError('scientific_anchor_pending: technical evidence is not available R5')
                else:
                    for point in last_points:bind_selected(last_actual,point,close_context,status)
                break
            if performance_capture is not None:performance_capture.publish()
            time.sleep(2)
    except Exception as error:
        failure = error
        # Missing/incomplete capture is evidence; no unbudgeted inherited-log write.
    finally:
        if writer is not None and closed is None:
            try:
                admit()
                if performance_capture is not None and failure is not None:
                    # Retain the last good accepted prefix and the failure as
                    # acknowledged technical evidence, without replaying phase
                    # discovery or publishing any new scientific state.
                    writer_call(writer.record_control, dict(event='capture_failed', reason='observer_source_or_tail_failed'))
                    close_context = writer_call(writer.close_context)
                    closed = writer.root / close_context[0]['path']
                    performance_capture.publish(close_context=close_context, capture_status=dict(
                        terminal_tail='known_incomplete', scalar_live_route='unavailable',
                        scalar_live_reason='capture_failed', performance_status='incomplete', reason='capture_failed'))
                else:
                    writer_call(writer.close)
            except Exception as cleanup:
                if failure is not None:failure.add_note('observer incomplete close: ' + str(cleanup))
        if phase_source is not None:phase_source.close()
        if native_guard is not None:
            try:native_guard.close()
            except BaseException as cleanup:
                if not isinstance(cleanup,Exception):raise
                if failure is None:failure=cleanup
                else:failure.add_note('observer native3 final close: '+str(cleanup))
        os.close(control_fd)
    return 0 if failure is None else 2


def capture_run(spec_path, solver, arguments, *, native_execution_path=None):
    spec_path = str(Path(spec_path).absolute())
    arguments = tuple(arguments)
    native_path = None if native_execution_path is None else str(Path(native_execution_path).absolute())

    def factory(pid, start, fd):
        command = [
            sys.executable,
            "-m",
            "qcl_negf_results.capture",
            "observe",
            spec_path,
            str(pid),
            str(start),
            str(fd),
        ]
        if native_path is not None:
            command.extend(['--native-execution', native_path, '--', *arguments])
        return command

    return supervise([solver, *arguments], observer_factory=factory)


if __name__ == "__main__":
    if len(sys.argv) < 6 or sys.argv[1] != "observe":
        raise SystemExit(2)
    native_path = None; arguments = ()
    if len(sys.argv) != 6:
        if len(sys.argv) < 10 or sys.argv[6] != '--native-execution' or sys.argv[8] != '--':
            raise SystemExit(2)
        native_path, arguments = sys.argv[7], tuple(sys.argv[9:])
    # -m defines this file as __main__. Dispatch through the canonical module
    # so producer batches and the strict writer share the same PhaseBatch class.
    from .capture import observe as canonical_observe
    raise SystemExit(
        canonical_observe(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]),
                native_execution_path=native_path, solver_arguments=arguments)
    )
