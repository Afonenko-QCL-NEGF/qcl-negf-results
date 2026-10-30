"""One solver reaper and one disposable I/O observer; no solver retry."""

from __future__ import annotations

import hashlib
import os
import re
import signal
import subprocess
import sys
import time
from copy import deepcopy
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


def make_binding(reader,*,telemetry,destination,capture_status,budget=None):
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
    limits=PerformanceLimits();reservation=None;consumed=0;bindings={}
    allowance=limits.maximum_io_bytes
    if budget is not None:
        allowance=min(allowance,budget.remaining-MIB)
        if allowance<1:raise ValueError('binding metadata/output allowance unavailable')
        budget.reserve(allowance);reservation=allowance
    try:
        with reader._consumer_operation(allowance,verify_source_stability=True,maximum_stability_bytes=limits.maximum_metadata_bytes) as io:
            before=io.bytes_read_total
            try:
                loaded=_load_telemetry(reader,telemetry,limits,bindings,creating_binding=True);data=loaded['input'];commit=loaded['commit'];summary=loaded['summary']
                value=dict(schema='qcl-negf.telemetry-binding.v1',contract_set='qcl-negf.results.v2',
                    scope='selected_state_with_execution_attempt_telemetry',selected_state=reader.source.state,
                    transport=deepcopy(data['expected_transport']),capture_status=deepcopy(capture_status),
                    commit=dict(ref=data['commit_ref'],raw_utf8=loaded['commit_raw'].decode('utf-8')),
                    summary=dict(ref=commit['summary'],raw_utf8=loaded['summary_raw'].decode('utf-8')),
                    source_binding=dict(ref=data['source_binding_ref'],raw_utf8=loaded['source_binding_raw'].decode('utf-8')))
                validate_telemetry_binding(value,expected_state=reader.source.state,expected_transport=data['expected_transport'],
                    commit=commit,commit_ref=data['commit_ref'],summary=summary,summary_ref=commit['summary'],
                    source_binding=loaded['source_binding'],source_binding_ref=data['source_binding_ref'],
                    manifest=reader._manifest,manifest_ref=reader.source.state['manifest'],source_transport=data['source_transport'],
                    canonical_plan_ref=data['canonical_plan_ref'],expected_producer_provenance=data['expected_producer_provenance'])
                payload=json_bytes(value)
                if len(payload)>MIB:fail('budget_exceeded','retained technical binding metadata cap exceeded')
                _check_bindings(bindings)
            finally:consumed=io.bytes_read_total-before
    finally:
        if reservation is not None:
            budget.reserved-=reservation;budget.account(consumed)
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
    value, _ = small_json(path, 16384, budget=budget)
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

    if (
        not isinstance(value, dict)
        or set(value) != keys
        or value["schema"] != "qcl-negf.capture-spec.v1"
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
    parsed, raw = small_json(beneath(root, plan["path"]), MIB, budget=budget)
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
    native, native_raw = small_json(path, 16384, budget=budget)
    if (set(native) != {'schema', 'request_id', 'execution_id', 'storage', 'source_graph',
                       'code', 'input_plan', 'access_receipt_ref'}
            or native['schema'] != 'qcl-negf.aiida-native-execution.v1'
            or native['execution_id'] != spec['selection']['execution_id']):
        raise ValueError('original native execution input differs')
    plan, plan_raw = small_json(root / 'plan.json', MIB, budget=budget)
    graph, graph_raw = small_json(root / 'source-graph.json', MIB, budget=budget)
    if (_metadata_weight((native, native_raw, plan, plan_raw, graph, graph_raw, arguments, spec)) > MIB):
        raise ValueError('combined native input metadata allowance exhausted')
    _bounded_objects((native, plan, graph), 4096)
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
        if type(value) is not str or uuid.fullmatch(value) is None:
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
    output = Path(arguments[2]); suffix = Path(storage['relative_execution_root']).parts
    if (not output.is_absolute() or str(output) != os.path.abspath(output)
            or output.parts[-len(suffix):] != suffix or output.is_relative_to(root) or root.is_relative_to(output)):
        raise ValueError('original native output argv/storage suffix differs')
    # The absolute trusted jobs_root and actual InstalledCode/wrapper origins
    # remain controller/Platform gates; suffix/JSON/token are not authority.
    transport = dict(selection=deepcopy(spec['selection']), frozen_plan=dict(
        path='plan.json', bytes=str(len(plan_raw)), sha256=digest(plan_raw),
        schema=plan['schema'], media_type='application/json'))
    return output, transport


def _locator(root, spec, *, budget=None, native_output=None):
    value, _ = small_json(root / "capture-source.json", 16384, budget=budget)
    if (
        not isinstance(value, dict)
        or set(value)
        != {
            "schema",
            "capture_token",
            "execution_id",
            "plan_fingerprint",
            "mode",
            "actual_root",
            "epoch",
        }
        or value["schema"] != "qcl-negf.capture-source.v1"
        or value["capture_token"] != spec["capture_token"]
        or value["execution_id"] != spec["selection"]["execution_id"]
        or value["plan_fingerprint"] != spec["selection"]["plan_fingerprint"]
        or not integer(value["epoch"])
        or value["mode"] not in ("direct", "scratch", "staged")
        or not isinstance(value["actual_root"], str)
        or not Path(value["actual_root"]).is_absolute()
    ):
        raise ValueError("actual-root locator differs from launch")
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


def observe(spec_path, pid, start_ticks, control_fd, *, transport=None, telemetry_selection_factory=None,
            native_execution_path=None, solver_arguments=()):
    """Only this disposable process imports Arrow or touches job/scratch/NFS files."""
    budget = ObserverIO(16384)
    try:
        root, spec, known = _spec(spec_path, budget=budget)
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        os.close(control_fd)
        return 2 # No traceback/inherited-log write before capture credit.
    selection = spec["selection"]
    native_output = None
    try:
        if native_execution_path is not None:
            if transport is not None or telemetry_selection_factory is not None:
                raise ValueError('native ingress cannot accept caller transport/factory assertions')
            native_output, transport = _native_capture_inputs(root, spec, native_execution_path,
                solver_arguments, budget=budget)
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
    except Exception:
        os.close(control_fd);return 2
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
        free = os.statvfs(root)
        budget.require(estimate)
        if (
            used + estimate > credits["performance_bytes"]
            or free.f_bavail * free.f_frsize < credits["reserve_bytes"] + estimate
        ):
            raise ValueError("observer admitted storage/IO partition exhausted")

    def writer_call(method, *args, **kwargs):
        # Local finite forecast for unchanged Arrow serialization/footer/hash I/O.
        # Exact JSON metadata has its own before-read/write hook on the same grant.
        admit()
        reservation = 4 * (writer.batch_bytes if writer is not None else 4 * MIB) + MIB
        budget.reserve(reservation)
        before_directory=writer.directory if writer is not None else None
        before = set(before_directory.glob("*.parquet")) if before_directory is not None else set()
        try:
            return method(*args, **kwargs)
        finally:
            paths = set(before_directory.glob("*.parquet")) if before_directory is not None else set()
            if writer is not None:paths.update(writer.directory.glob("*.parquet"))
            forecast = 4 * sum(path.stat().st_size for path in paths - before)
            budget.settle(reservation, forecast)
            if forecast > reservation:
                raise ValueError("writer exceeded its finite I/O forecast; capture unknown")

    def capture_point(actual, point, mode, terminal, epoch):
        nonlocal journal_bytes,phase_source,phase_key,closed_page_bytes
        state=point['selected_state'];descriptor=point['phase_source']
        if state is None:
            missing_sources.add(point['point_id']);return
        key=digest(json_bytes(dict(identity=state['identity'],session=descriptor['session_id'])))
        if descriptor['status']=='available':
            qualified=(str(actual),key,epoch)
            if phase_key!=qualified:
                if phase_source is not None:phase_source.close()
                phase_source=PhaseCaptureSource(actual,descriptor=descriptor,expected_identity=state['identity'],route_epoch=epoch)
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
            make_binding(reader,telemetry=selected,destination=writer.root/name,capture_status=capture_status,budget=budget)
            charged=io.bytes_read_total  # make_binding already charged its exact delta
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
        _locator(root, spec, budget=budget, native_output=native_output)  # Refuse foreign before ANY output.
        from .telemetry import TelemetryWriter
        admit()
        writer = writer_call(TelemetryWriter,
            root / "performance",
            {**selection, "allocation_id": spec["allocation_id"]},
            {
                "scope": "root_julia_process_including_threads",
                "allocated_logical_cpus": spec["allocated_logical_cpus"],
            },
            metadata_budget=budget,transport=transport,
        )
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
                if locator["epoch"] < previous_epoch:
                    raise ValueError("actual source epoch reversed")
                previous_epoch = locator["epoch"]
                if not terminal and not (actual / "current_point.json").exists():
                    admit()
                    writer_call(writer.flush_due)
                    time.sleep(2)
                    continue
                points, pins = (
                    _final_points(actual, selection, known, budget=budget)
                    if terminal
                    else ([_current(actual, selection, known, budget=budget)], None)
                )
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
                if telemetry_selection_factory is None or transport is None:
                    writer_call(writer.record_control,dict(event='source_binding_missing',reason='explicit_transport_or_factory_missing'))
                close_context = writer_call(writer.close_context)
                close_ref, _, _ = close_context
                closed = writer.root / close_ref['path']
                if telemetry_selection_factory is None or transport is None:
                    raise ValueError('source_binding_missing: explicit transport/factory and production ingress policy required')
                if last_actual is None or not last_points:raise ValueError('selected source unavailable for exact close binding')
                complete=not missing_sources and bool(cursors) and all(c['complete_at_observation'] for c in cursors.values())
                status=dict(terminal_tail='closed_observed' if complete and all(closed_sources.values()) else 'known_incomplete' if missing_sources else 'unknown',
                    scalar_live_route=locator['mode'] if not publisher.disabled else 'unavailable',
                    scalar_live_reason='live_publication_incomplete' if publisher.disabled else None,
                    performance_status='available' if complete else 'incomplete',reason=None if complete else 'phase_sources_or_tail_incomplete')
                for point in last_points:bind_selected(last_actual,point,close_context,status)
                break
            time.sleep(2)
    except Exception as error:
        failure = error
        # Missing/incomplete capture is evidence; no unbudgeted inherited-log write.
    finally:
        if writer is not None and closed is None:
            try:
                admit()
                writer_call(writer.close)
            except Exception:
                pass
        if phase_source is not None:phase_source.close()
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
