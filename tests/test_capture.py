"""No-solve wrapper/observer controls, executed only by the separate finite gate."""

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest

from qcl_negf_results.capture import journal_batch, proc_sample, supervise


def test_T0_observer_public_close_readback_failure_preserves_science_and_no_binding(tmp_path, monkeypatch):
    """Observer must use the durable public close result before binding science."""
    import hashlib
    from native_consumer_fixture import technical_source, sealed_phase, outward_series
    from qcl_negf_results.commits import json_bytes
    from qcl_negf_results.telemetry import TelemetryWriter
    from qcl_negf_results.performance_views import TelemetrySelection
    import qcl_negf_results.capture as module
    job = tmp_path / 'job'
    job.mkdir()
    result = job / 'result'
    c = technical_source(result, checkpoint=True, execution='exec')
    state = c['state']
    manifest = c['science'] / state['manifest']['path']
    scientific_raw = manifest.read_bytes()
    sealed_phase(result, state['identity'])
    phase = dict(status='available', session_id='native', current_path='phase-current.json', reason=None)
    selection = c['transport']['selection']
    raw = json_bytes(dict(fingerprint=selection['plan_fingerprint'], points=[dict(id=state['identity']['point_id'], execution_id='exec')]))
    (job / 'plan.json').write_bytes(raw)
    (job / 'capture-spec.json').write_bytes(json_bytes(dict(schema='qcl-negf.capture-spec.v1', selection=selection,
        plan=dict(path='plan.json', bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest()), capture_token='c' * 32,
        allocation_id='allocation', allocated_logical_cpus=2, scratch_root=None,
        credits=dict(performance_bytes=32 << 20, scalar_bytes=0, io_bytes=32 << 20, reserve_bytes=0))))
    (job / 'capture-source.json').write_bytes(json_bytes(dict(schema='qcl-negf.capture-source.v1', capture_token='c' * 32,
        execution_id='exec', plan_fingerprint=selection['plan_fingerprint'], mode='direct', actual_root=str(result), epoch=0)))
    (result / 'series_result.json').write_bytes(json_bytes(outward_series(state, phase)))
    technical = job / 'performance'
    technical.mkdir()
    (technical / c['source_binding']['path']).write_bytes((c['technical'] / c['source_binding']['path']).read_bytes())
    def factory(reader, close_ref):
        return TelemetrySelection(technical, commit_ref=close_ref, source_binding_ref=c['source_binding'], expected_transport=c['transport'],
            source_transport=c['source_transport'], canonical_plan_root=c['root'], canonical_plan_ref=c['plan'],
            expected_producer_provenance=c['producer_provenance'])
    close = TelemetryWriter.close_context
    closed_refs = []
    def fail_after_durable_close(writer):
        ref, _, _ = close(writer)
        closed_refs.append(ref)
        raise OSError('T0 close context readback unavailable')
    monkeypatch.setattr(TelemetryWriter, 'close_context', fail_after_durable_close)
    monkeypatch.setattr(module, 'proc_sample', lambda *a, **k: dict(wall_seconds=2., monotonic_ns=2_000_000_000,
        cpu_time_seconds=None, missing_reason='root_process_missing_or_identity_changed'))
    read_fd, write_fd = os.pipe2(os.O_NONBLOCK | os.O_CLOEXEC)
    os.write(write_fd, b'exit 0\n')
    os.close(write_fd)
    assert module.observe(job / 'capture-spec.json', 42, 7, read_fd,
        transport=c['transport'], telemetry_selection_factory=factory) == 2
    assert manifest.read_bytes() == scientific_raw
    assert len(closed_refs) == 1 and json.loads((technical / closed_refs[0]['path']).read_bytes())['closed'] is True
    assert not (technical / 'performance-binding.json').exists()


def test_proc_root_process_counter_identity_and_missing_are_not_zero(tmp_path):
    root = tmp_path / "proc"
    point = root / "42"
    point.mkdir(parents=True)
    fields = ["S"] + ["0"] * 49
    fields[11], fields[12], fields[19] = "100", "200", "500"
    (point / "stat").write_text("42 (name with ) spaces) " + " ".join(fields))
    (point / "status").write_text("VmRSS:\t12 kB\nVmHWM:\t18 kB\n")
    row = proc_sample(42, 500, ticks=100, proc_root=root, now_ns=10, wall_seconds=2)
    assert row["cpu_time_seconds"] == 3.0
    assert row["rss_bytes"] == 12288 and row["peak_rss_bytes"] == 18432
    assert row["cgroup_scope"] == "root_julia_process_including_threads"
    foreign = proc_sample(42, 501, ticks=100, proc_root=root, now_ns=11, wall_seconds=3)
    assert foreign["cpu_time_seconds"] is None and foreign["missing_reason"]
    (point / "stat").unlink()
    gone = proc_sample(42, 500, ticks=100, proc_root=root, now_ns=12, wall_seconds=4)
    assert gone["cpu_time_seconds"] is None and gone["rss_bytes"] is None


@pytest.mark.parametrize("mode, expected", [("normal", 7), ("term", 23)])
def test_supervisor_ready_child_exit_and_term_reap_stalled_observer(tmp_path, mode, expected):
    helper = Path(__file__).parent / "helpers/capture_lifecycle_probe.py"
    child = subprocess.Popen(
        [sys.executable, str(helper), str(tmp_path), mode],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    try:
        deadline = time.monotonic() + 3
        while not all((tmp_path / name).exists() for name in ("solver-ready", "observer-ready")):
            assert time.monotonic() < deadline, "stand-ins failed to become ready"
            assert child.poll() is None, "helper exited before readiness"
            time.sleep(.01)
        if mode == "term":
            child.send_signal(signal.SIGTERM)
        output, errors = child.communicate(timeout=5)
        assert child.returncode == 0, errors
        row = json.loads(output)
        assert row["code"] == expected and row["observer_reaped"] is True
        assert row["elapsed"] < 5
    finally:
        if child.poll() is None:
            for name in ("solver-ready", "observer-ready"):
                marker = tmp_path / name
                if marker.exists():
                    try:
                        os.killpg(int(marker.read_text()), signal.SIGKILL)
                    except ProcessLookupError:
                        pass
            try:
                child.communicate(timeout=2)
            except subprocess.TimeoutExpired:
                child.kill()
                child.communicate(timeout=1)


def test_supervisor_never_relaunches_after_observer_spawn_error(tmp_path, monkeypatch):
    marker = tmp_path / "once"
    code = 'from pathlib import Path; p=Path(__import__("sys").argv[1]); p.write_text("once")'

    def fail(*args):
        raise OSError("observer unavailable")

    class ForbiddenStderr:
        def write(self, *args):
            raise AssertionError("supervisor must not write an inherited log")

    with monkeypatch.context() as patch:
        patch.setattr("qcl_negf_results.capture.sys.stderr", ForbiddenStderr())
        code = supervise(
            [sys.executable, "-c", code, str(marker)], observer_factory=fail,
            drain_seconds=0.1,
        )
    assert code == 0
    assert marker.read_text() == "once"


def test_observer_prefix_and_repeated_metadata_refuse_before_read(tmp_path, monkeypatch):
    from contextlib import contextmanager
    import qcl_negf_results._capture_io as io
    import qcl_negf_results.capture as capture

    reads = []
    original = io.regular

    class Spy:
        def __init__(self, stream):
            self.stream = stream
        def fileno(self):
            return self.stream.fileno()
        def read(self, count):
            reads.append(count)
            return self.stream.read(count)

    @contextmanager
    def tracked(*args, **kwargs):
        with original(*args, **kwargs) as stream:
            yield Spy(stream)

    monkeypatch.setattr(io, "regular", tracked)
    monkeypatch.setattr(capture, "regular", tracked)
    journal = tmp_path / "journal"
    journal.write_bytes(b"x" * 1024 + b"\n")
    budget = io.ObserverIO(100)
    with pytest.raises(ValueError):
        journal_batch(journal, {"offset": 1024}, source_key="p", budget=budget)
    assert reads == [] and budget.read_bytes == 0
    metadata = tmp_path / "metadata.json"
    metadata.write_bytes(b'{"a":1}')
    budget = io.ObserverIO(7)
    assert io.small_json(metadata, 100, budget=budget)[0] == {"a": 1}
    first = list(reads)
    with pytest.raises(ValueError):
        io.small_json(metadata, 100, budget=budget)
    assert reads == first and budget.read_bytes == 7


def test_writer_startup_metadata_and_binding_share_observer_credit(tmp_path):
    from qcl_negf_results._capture_io import ObserverIO
    from qcl_negf_results.capture import make_binding
    from qcl_negf_results.telemetry import TelemetryWriter

    from native_consumer_fixture import technical_source,telemetry_input
    from qcl_negf_results.state import StateReader
    c=technical_source(tmp_path/'input')
    selection=c['transport']['selection']
    budget=ObserverIO(0)
    with pytest.raises(ValueError):TelemetryWriter(tmp_path/'denied',selection,{},metadata_budget=budget)
    assert not list((tmp_path/'denied').rglob('*.json'))
    budget=ObserverIO(64*1024)
    writer=TelemetryWriter(c['technical'],selection,{},metadata_budget=budget,transport=c['transport']);writer.close()
    assert budget.read_bytes>0 and budget.write_bytes>0
    before=budget.used;budget.maximum=before
    with StateReader(c['science'],state_ref=c['state'],expected_identity=c['state']['identity']) as reader:
        with pytest.raises(ValueError):make_binding(reader,telemetry=telemetry_input(c,writer,bind_closed=False),
            destination=c['technical']/'performance-binding.json',budget=budget,
            capture_status=dict(terminal_tail='unknown',scalar_live_route='direct',scalar_live_reason=None,performance_status='available',reason=None))
    assert budget.used==before and not(c['technical']/'performance-binding.json').exists()


def test_complete_native_lines_are_atomic_cursor_batches(tmp_path):
    source = tmp_path / "phases.jsonl"
    row = {
        "schema": "qcl-negf.producer-phase.v2",
        "contract_set": "qcl-negf.results.v1",
        "source_session_id": "native",
        "source_sequence": 1,
        "clock_domain_id": "native-clock",
        "phase": "basis",
        "event_kind": "end",
        "duration_seconds": 2.0,
        "monotonic_ns": 20,
        "begin_monotonic_ns": 10,
        "end_monotonic_ns": 20,
    }
    raw = json.dumps(row).encode() + b"\n"
    source.write_bytes(raw + b'{"partial":')
    rows, cursor, consumed = journal_batch(source, {}, source_key="p/attempt-2")
    assert len(rows) == 1 and rows[0]["measurement_kind"] == "phase_boundary"
    assert rows[0]["source_sequence"] == 1
    assert cursor["offset"] == len(raw) and cursor["complete_at_observation"] is False
    assert consumed == len(raw) + len(b'{"partial":')
    rows2, cursor2, _ = journal_batch(source, cursor, source_key="p/attempt-2")
    assert rows2 == [] and cursor2["offset"] == len(raw)
    source.write_bytes(raw.replace(b'"native"', b'"foreign"') + b"\n")
    with pytest.raises(ValueError):
        journal_batch(source, cursor, source_key="p/attempt-2")


def test_actual_scratch_closed_publication_then_direct_route_no_whole_copy(tmp_path):
    from native_consumer_fixture import witness_history
    from qcl_negf_results.capture import ScalarPublisher
    from qcl_negf_results._capture_io import ObserverIO
    job=tmp_path/'job';job.mkdir();scratch=tmp_path/'scratch';scratch.mkdir()
    point_root=scratch/'p';point_root.mkdir();ref,_=witness_history(point_root)
    (point_root/'.open.h5').write_bytes(b'not closed')
    selection=dict(calcjob_uuid='job',execution_id=ref['identity']['execution_id'],attempt=10,plan_fingerprint='a'*64)
    budget=ObserverIO(128*1024*1024)
    publisher=ScalarPublisher(job,selection,scalar_credit=64*1024*1024,reserve_bytes=0,budget=budget)
    point=dict(point_id=ref['identity']['point_id'],source_attempt=int(ref['identity']['attempt']),artifact_root='p',selected_state=ref)
    before={str(p.relative_to(point_root)):p.read_bytes() for p in point_root.rglob('*') if p.is_file()}
    value=publisher.publish(scratch,point,mode='scratch',terminal=True)
    mirror=job/value['artifact_root']
    assert (mirror/ref['manifest']['path']).read_bytes()==before[ref['manifest']['path']]
    assert not(mirror/'.open.h5').exists()
    assert value['source_state']==ref and value['history']['records']=='20'
    for p in mirror.rglob('*'):
        if p.is_file():assert p.read_bytes()==before[str(p.relative_to(mirror))]
    descriptor=json.loads((job/'live/current.json').read_bytes())
    assert descriptor==value and descriptor['schema']=='qcl-negf.live-prefix.v2'
    assert value['canonical_source_complete'] is False
    assert budget.write_bytes==sum(p.stat().st_size for p in mirror.rglob('*') if p.is_file())+len((job/'live/current.json').read_bytes())
    assert {str(p.relative_to(point_root)):p.read_bytes() for p in point_root.rglob('*') if p.is_file()}==before
    denied=ScalarPublisher(job,selection,scalar_credit=0,reserve_bytes=0)
    with pytest.raises((ValueError,__import__('qcl_negf_contracts.messages',fromlist=['ContractError']).ContractError)):
        denied.publish(scratch,point,mode='scratch',terminal=True)
    assert denied.disabled and (point_root/ref['manifest']['path']).read_bytes()==before[ref['manifest']['path']]
    result=job/'result';result.mkdir();direct_point=result/'p';direct_point.mkdir();direct_ref,_=witness_history(direct_point)
    direct=ScalarPublisher(job,selection,scalar_credit=0,reserve_bytes=0)
    shown=direct.publish(result,dict(point,selected_state=direct_ref),mode='direct',terminal=True)
    assert shown['artifact_root']=='result/p' and direct.bytes==0


def test_real_observer_direct_close_sidecar_and_scientific_sentinel(
    tmp_path, monkeypatch
):
    import hashlib
    from native_consumer_fixture import technical_source,sealed_phase,outward_series
    from qcl_negf_results.capture import observe
    from qcl_negf_results.commits import json_bytes
    from qcl_negf_results.performance_views import TelemetrySelection
    from qcl_negf_contracts.outward import validate_outward_result
    job=tmp_path/'job';job.mkdir();result=job/'result'
    c=technical_source(result,checkpoint=True,execution='exec');selected=c['state']
    commit=c['science']/selected['manifest']['path'];sentinel=commit.read_bytes()
    literal=sealed_phase(result,selected['identity'])
    phase=dict(status='available',session_id='native',current_path='phase-current.json',reason=None)
    selection=c['transport']['selection']
    raw=json_bytes(dict(fingerprint=selection['plan_fingerprint'],points=[dict(id=selected['identity']['point_id'],execution_id='exec')]))
    (job/'plan.json').write_bytes(raw)
    spec=dict(schema='qcl-negf.capture-spec.v1',selection=selection,
        plan=dict(path='plan.json',bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()),capture_token='c'*32,
        allocation_id='allocation',allocated_logical_cpus=2,scratch_root=None,
        credits=dict(performance_bytes=32*1024*1024,scalar_bytes=0,io_bytes=32*1024*1024,reserve_bytes=0))
    (job/'capture-spec.json').write_bytes(json_bytes(spec))
    (job/'capture-source.json').write_bytes(json_bytes(dict(schema='qcl-negf.capture-source.v1',capture_token='c'*32,
        execution_id='exec',plan_fingerprint=selection['plan_fingerprint'],mode='direct',actual_root=str(result),epoch=0)))
    series=outward_series(selected,phase);validate_outward_result(series)
    (result/'series_result.json').write_bytes(json_bytes(series))
    technical=job/'performance';technical.mkdir()
    (technical/c['source_binding']['path']).write_bytes((c['technical']/c['source_binding']['path']).read_bytes())
    def factory(reader,close_ref):
        assert reader.source.state==selected
        return TelemetrySelection(technical,commit_ref=close_ref,source_binding_ref=c['source_binding'],expected_transport=c['transport'],
            source_transport=c['source_transport'],canonical_plan_root=c['root'],canonical_plan_ref=c['plan'],
            expected_producer_provenance=c['producer_provenance'])
    read_fd,write_fd=os.pipe2(os.O_NONBLOCK|os.O_CLOEXEC);os.write(write_fd,b'exit 0\n');os.close(write_fd)
    monkeypatch.setattr('qcl_negf_results.capture.proc_sample',lambda *a,**k:dict(wall_seconds=2.,monotonic_ns=2_000_000_000,
        cpu_time_seconds=None,missing_reason='root_process_missing_or_identity_changed'))
    assert observe(job/'capture-spec.json',42,7,read_fd,transport=c['transport'],telemetry_selection_factory=factory)==0
    assert commit.read_bytes()==sentinel
    descriptor=json.loads((job/'live/current.json').read_bytes())
    assert descriptor['artifact_root']=='result/science' and descriptor['source_state']==selected
    sidecar=json.loads((technical/'performance-binding.json').read_bytes())
    native=json.loads(sidecar['commit']['raw_utf8'])
    assert native['closed'] is True
    assert native['source_cursors'][0]['cursor']['next_row']=='2'
    assert sidecar['selected_state']['identity']['attempt']=='1' and sidecar['transport']['selection']['attempt']==3
    assert sidecar['capture_status']['performance_status']=='available'
    assert sidecar['capture_status']['terminal_tail']=='closed_observed'
    assert native['source_cursors'][0]['binding']['status']=='unbound'


def test_root_switch_same_native_prefix_does_not_replay_rows(tmp_path):
    first = tmp_path / "scratch.jsonl"
    second = tmp_path / "staged.jsonl"
    row = {
        "schema": "qcl-negf.producer-phase.v2",
        "contract_set": "qcl-negf.results.v1",
        "source_session_id": "native",
        "source_sequence": 1,
        "clock_domain_id": "clock",
        "phase": "basis",
        "event_kind": "end",
        "duration_seconds": 1.0,
    }
    raw = json.dumps(row).encode() + b"\n"
    first.write_bytes(raw)
    second.write_bytes(raw)
    rows, cursor, _ = journal_batch(first, {}, source_key="p/attempt-2")
    replay, advanced, _ = journal_batch(second, cursor, source_key="p/attempt-2")
    assert len(rows) == 1 and replay == [] and advanced["offset"] == len(raw)
    row["source_sequence"] = 2
    second.write_bytes(raw + json.dumps(row).encode() + b"\n")
    newer, cutoff, _ = journal_batch(second, advanced, source_key="p/attempt-2")
    assert [item["source_sequence"] for item in newer] == [2]
    assert cutoff["source_sequence"] == 2


def test_T10_sealed_phase_file_prefix_uses_rows_not_max_sequence_and_quiet_payload_cache(tmp_path):
    from native_consumer_fixture import sealed_phase
    from result_v2_fixture import IDENTITY
    from qcl_negf_results.capture import PhaseCaptureSource, PhaseCaptureLimits
    from qcl_negf_results._capture_io import ObserverIO
    from qcl_negf_contracts.telemetry import validate_phase_cursor
    literal=sealed_phase(tmp_path,IDENTITY)
    budget=ObserverIO(1024*1024)
    descriptor=dict(status='available',session_id='native',current_path='phase-current.json',reason=None)
    with PhaseCaptureSource(tmp_path,descriptor=descriptor,expected_identity=IDENTITY,route_epoch=0,
            limits=PhaseCaptureLimits(maximum_rows=1)) as source:
        first=source.harvest(None,budget=budget)
        assert (first.first_row,first.stop_row)==(0,1)
        assert first.cursor['last_sequence']=='8' and first.cursor['next_row']=='1'
        assert first.cursor['complete_at_observation'] is False
        delivered=budget.read_bytes
        second=source.harvest(first.cursor,budget=budget)
        assert (second.first_row,second.stop_row)==(1,2)
        assert second.cursor['last_sequence']=='8' and second.cursor['next_row']=='2'
        assert second.segment_bytes==first.segment_bytes
        validate_phase_cursor(second.cursor,publication=literal['publication'],publication_ref=literal['publication_ref'],
            segment=literal['segment'],segment_ref=literal['segment_ref'])
        assert source.harvest(second.cursor,budget=budget) is None
        # Two bounded current observations, not repeated segment/publication delivery.
        assert budget.read_bytes-delivered==2*len((tmp_path/'phase-current.json').read_bytes())
    assert literal['segment']['rows'][0]['metrics'][0]['sum']==4.5


@pytest.mark.parametrize('fault',['replace','grow','cancel'])
def test_T10_raw_phase_actual_read_faults_close_fds_and_keep_exact_primary(tmp_path,monkeypatch,fault):
    from contextlib import contextmanager
    from native_consumer_fixture import diagnostic_phase
    from result_v2_fixture import IDENTITY
    from qcl_negf_results.capture import PhaseCaptureSource
    from qcl_negf_results.state import _IOBudget
    from qcl_negf_contracts.messages import ContractError
    import qcl_negf_results.capture as module
    literal=diagnostic_phase(tmp_path,IDENTITY);target=tmp_path/literal['segment_ref']['path']
    original=module.regular;active=[];seen=[]
    @contextmanager
    def injected(path,**kwargs):
        with original(path,**kwargs) as stream:
            active.append(stream.fileno())
            class Proxy:
                def fileno(self):return stream.fileno()
                def read(self,size):
                    data=stream.read(size)
                    if Path(path)==target and not seen:
                        seen.append(True)
                        if fault=='cancel':raise KeyboardInterrupt('exact raw cancellation')
                        if fault=='replace':
                            other=target.with_suffix('.replacement');other.write_bytes(data);os.replace(other,target)
                        else:
                            with target.open('ab') as out:out.write(b' ')
                    return data
            try:yield Proxy()
            finally:active.pop()
    with PhaseCaptureSource(tmp_path,descriptor=dict(status='available',session_id='native',current_path='phase-current.json',reason=None),
            expected_identity=IDENTITY,route_epoch=0) as source:
        with monkeypatch.context() as patch:
            patch.setattr(module,'regular',injected)
            with pytest.raises(KeyboardInterrupt if fault=='cancel' else ContractError) as error:
                source.harvest(None,budget=_IOBudget(1024*1024))
        assert seen and active==[]
        if fault=='cancel':
            assert str(error.value)=='exact raw cancellation'
            assert source.harvest(None,budget=_IOBudget(1024*1024)).segment==literal['segment']
        else:assert error.value.code=='source_changed'
    assert source.closed and source._cache=={}


def test_T10_raw_phase_shared_read_credit_is_not_reset_by_failure_or_retry(tmp_path):
    from native_consumer_fixture import diagnostic_phase
    from result_v2_fixture import IDENTITY
    from qcl_negf_results.capture import PhaseCaptureSource
    from qcl_negf_results.state import _IOBudget
    from qcl_negf_contracts.messages import ContractError
    literal=diagnostic_phase(tmp_path,IDENTITY)
    budget=_IOBudget(len((tmp_path/'phase-current.json').read_bytes())+int(literal['publication_ref']['bytes'])+1)
    with PhaseCaptureSource(tmp_path,descriptor=dict(status='available',session_id='native',current_path='phase-current.json',reason=None),
            expected_identity=IDENTITY,route_epoch=0) as source:
        with pytest.raises(ContractError) as error:source.harvest(None,budget=budget)
        assert error.value.code=='budget_exceeded' and 0<budget.bytes_read_total<=budget.maximum
        delivered=budget.bytes_read_total
        with pytest.raises(ContractError):source.harvest(None,budget=budget)
        assert budget.bytes_read_total==delivered


@pytest.mark.parametrize('fault',['countstr_attempt','unknown_field','nan'])
def test_T10_observe_initial_public_transport_refuses_before_output(tmp_path,monkeypatch,fault):
    import copy
    from native_consumer_fixture import technical_source
    import qcl_negf_results.capture as module
    c=technical_source(tmp_path/'input');expected=c['transport'];bad=copy.deepcopy(expected)
    if fault=='countstr_attempt':bad['selection']['attempt']='3'
    elif fault=='unknown_field':bad['new_authority']=True
    else:bad['selection']['plan_fingerprint']=float('nan')
    monkeypatch.setattr(module,'_spec',lambda *a,**k:(tmp_path,dict(selection=expected['selection']),{'point'}))
    read_fd,write_fd=os.pipe();os.close(write_fd)
    assert module.observe(tmp_path/'unused',42,7,read_fd,transport=bad)==2
    assert not(tmp_path/'performance').exists()
    with pytest.raises(OSError):os.fstat(read_fd)


def test_T10_observe_rotates_seventeenth_source_and_reappearing_older_source(tmp_path,monkeypatch):
    """Real outward/P29 files through the normal caller, no parser/writer stand-in."""
    import copy
    import hashlib
    import qcl_negf_results.capture as module
    from native_consumer_fixture import technical_source,sealed_phase,outward_series
    from result_v2_fixture import put
    from qcl_negf_results.commits import json_bytes
    from qcl_negf_results.performance_views import TelemetrySelection
    from qcl_negf_contracts.outward import validate_outward_result
    job=tmp_path/'job';result=job/'result';result.mkdir(parents=True)
    seed=technical_source(tmp_path/'seed',checkpoint=True,execution='exec')
    plan=json.loads((seed['root']/seed['plan']['path']).read_bytes())
    template=plan['points'][0];names=['point-'+str(i) for i in range(17)]
    plan['points']=[dict(copy.deepcopy(template),id=name,order=i+1) for i,name in enumerate(names)]
    plan['executions'][0]['point_ids']=names;plan['maximum_solver_runs']=plan['computation_count']=17
    planref=put(result,'canonical-plan.json',plan,plan['schema'])
    contexts={};points=[];phases=[];sentinels={}
    technical=job/'performance';technical.mkdir()
    for i,name in enumerate(names):
        c=technical_source(result/'points'/name,checkpoint=True,execution='exec',
            identity_override=dict(point_id=name,plan_sha256=planref['sha256']),
            canonical_plan=(result,planref),source_binding_path='source-bindings/'+name+'.json')
        contexts[name]=c
        literal=sealed_phase(result,c['state']['identity'],prefix='phase/'+name)
        phases.append(literal)
        phase=dict(status='available',session_id='native',current_path='phase/'+name+'/phase-current.json',reason=None)
        row=outward_series(c['state'],phase,artifact_root='points/'+name+'/science')['points'][0]
        row['coordinates']['order']=i+1;points.append(row)
        target=technical/c['source_binding']['path'];target.parent.mkdir(exist_ok=True)
        target.write_bytes((c['technical']/c['source_binding']['path']).read_bytes())
        source=c['science']/c['state']['manifest']['path'];sentinels[source]=source.read_bytes()
    series=outward_series(contexts[names[0]]['state'],points[0]['data']['phase_source'])
    series['points']=points;series['executions'][0]['point_ids']=names
    series['expected_point_count']=series['available_point_count']=17
    validate_outward_result(series);(result/'series_result.json').write_bytes(json_bytes(series))
    transport=contexts[names[0]]['transport'];selection=transport['selection']
    raw=json_bytes(dict(fingerprint=selection['plan_fingerprint'],points=[dict(id=n,execution_id='exec') for n in names]))
    (job/'plan.json').write_bytes(raw)
    spec=dict(schema='qcl-negf.capture-spec.v1',selection=selection,
        plan=dict(path='plan.json',bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()),capture_token='c'*32,
        allocation_id='allocation',allocated_logical_cpus=2,scratch_root=None,
        credits=dict(performance_bytes=32*1024*1024,scalar_bytes=0,io_bytes=256*1024*1024,reserve_bytes=0))
    (job/'capture-spec.json').write_bytes(json_bytes(spec))
    (job/'capture-source.json').write_bytes(json_bytes(dict(schema='qcl-negf.capture-source.v1',capture_token='c'*32,
        execution_id='exec',plan_fingerprint=selection['plan_fingerprint'],mode='direct',actual_root=str(result),epoch=0)))
    def current(index):
        row=points[index]
        value=dict(schema='qcl-negf-current-point-v2',point_id=row['id'],execution_id='exec',attempt=1,status='running',
            updated_unix=100,artifact_root='.',native_source=row['data']['native_source'],phase_source=row['data']['phase_source'],
            stage='scba',iteration=1,total=2,metrics=[],message='literal')
        validate_outward_result(value);(result/'current_point.json').write_bytes(json_bytes(value))
    current(0);read_fd,write_fd=os.pipe2(os.O_NONBLOCK|os.O_CLOEXEC);polls=[];rotations=[]
    from qcl_negf_results.telemetry import TelemetryWriter
    rotate=TelemetryWriter.rotate
    def tracked(writer):
        old=rotate(writer);rotations.append(old);return old
    monkeypatch.setattr(TelemetryWriter,'rotate',tracked)
    def sleep(_):
        polls.append(True)
        if len(polls)<17:current(len(polls))
        else:
            sealed_phase(result,contexts[names[0]]['state']['identity'],previous=phases[0]['publication_ref'],generation=2,prefix='phase/'+names[0])
            os.write(write_fd,b'exit 0\n')
    monkeypatch.setattr(module.time,'sleep',sleep)
    monkeypatch.setattr(module,'proc_sample',lambda *a,**k:dict(wall_seconds=float(len(polls)+1),monotonic_ns=(len(polls)+1)*1000000000,
        cpu_time_seconds=None,missing_reason='root_process_missing_or_identity_changed'))
    selected=[]
    def factory(reader,close_ref):
        c=contexts[reader.source.state['identity']['point_id']];selected.append(c['state'])
        return TelemetrySelection(technical,commit_ref=close_ref,source_binding_ref=c['source_binding'],expected_transport=transport,
            source_transport=c['source_transport'],canonical_plan_root=result,canonical_plan_ref=planref,
            expected_producer_provenance=c['producer_provenance'])
    try:
        assert module.observe(job/'capture-spec.json',42,7,read_fd,transport=transport,telemetry_selection_factory=factory)==0
    finally:os.close(write_fd)
    assert len(polls)==17 and len(rotations)==1 and len(selected)==17
    assert all(path.read_bytes()==raw for path,raw in sentinels.items())
    bindings=list((technical/'performance-bindings').glob('*.json'));assert len(bindings)==17
    for path in bindings:
        binding=json.loads(path.read_bytes());commit=json.loads(binding['commit']['raw_utf8']);summary=json.loads(binding['summary']['raw_utf8'])
        assert binding['transport']==transport and binding['selected_state'] in selected
        assert int(summary['totals']['accepted_source_rows'])==36
        assert commit['closed'] is True and len(commit['source_cursors'])<=16
        assert binding['capture_status']['terminal_tail']=='closed_observed'


@pytest.mark.parametrize('closed', [False, True])
@pytest.mark.parametrize('history', ['partial', 'complete', 'fresh_partial', 'fresh_complete', 'observed', 'pending'])
def test_T10_F1_lost_head_refuses_cursor_or_observed_prefix(tmp_path, closed, history):
    """An unrelated null head cannot certify a previously observed physical prefix."""
    from copy import deepcopy
    from native_consumer_fixture import sealed_phase
    from result_v2_fixture import IDENTITY
    from qcl_negf_results.capture import PhaseCaptureSource, PhaseCaptureLimits
    from qcl_negf_results.state import _IOBudget
    from qcl_negf_results.commits import json_bytes
    from qcl_negf_contracts.messages import ContractError
    literal = sealed_phase(tmp_path, IDENTITY)
    if history == 'pending':
        literal = sealed_phase(tmp_path, IDENTITY, previous=literal['publication_ref'], generation=2)
    descriptor = dict(status='available', session_id='native', current_path='phase-current.json', reason=None)
    limits = PhaseCaptureLimits(maximum_rows=1 if 'partial' in history else 128, maximum_publications=1)
    source = PhaseCaptureSource(tmp_path, descriptor=descriptor, expected_identity=IDENTITY, route_epoch=0, limits=limits)
    budget = _IOBudget(1 << 20)
    try:
        batch = source.harvest(None, budget=budget)
        if history == 'pending':
            assert batch is None and source.backlogged
            cursor = None
        else:
            cursor = deepcopy(batch.cursor)
            assert cursor['complete_at_observation'] is ('partial' not in history)
            if history == 'observed':
                cursor = None
        if history.startswith('fresh_'):
            source.close()
            source = PhaseCaptureSource(tmp_path, descriptor=descriptor, expected_identity=IDENTITY, route_epoch=0, limits=limits)
        before_cursor = deepcopy(cursor)
        immutable = {ref['path']: (tmp_path / ref['path']).read_bytes()
                     for ref in (literal['publication_ref'], literal['segment_ref'])}
        (tmp_path / 'phase-current.json').write_bytes(json_bytes(dict(literal['current'], publication=None, closed=closed)))
        with pytest.raises(ContractError) as error:
            source.harvest(cursor, budget=budget)
        assert error.value.code == 'source_changed'
        assert source.observed_closed is None
        assert cursor == before_cursor
        assert {path: (tmp_path / path).read_bytes() for path in immutable} == immutable
    finally:
        source.close()


@pytest.mark.parametrize('closed', [False, True])
def test_T10_F1_initial_empty_head_remains_valid_without_observed_prefix(tmp_path, closed):
    from native_consumer_fixture import sealed_phase
    from result_v2_fixture import IDENTITY
    from qcl_negf_results.capture import PhaseCaptureSource
    from qcl_negf_results.state import _IOBudget
    from qcl_negf_results.commits import json_bytes
    literal = sealed_phase(tmp_path, IDENTITY)
    (tmp_path / 'phase-current.json').write_bytes(json_bytes(dict(literal['current'], publication=None, closed=closed)))
    with PhaseCaptureSource(tmp_path, descriptor=dict(status='available', session_id='native', current_path='phase-current.json', reason=None),
                            expected_identity=IDENTITY, route_epoch=0) as source:
        assert source.harvest(None, budget=_IOBudget(1 << 20)) is None
        assert source.harvest(None, budget=_IOBudget(1 << 20)) is None
        assert source.observed_closed is closed


def test_T10_F1_observe_lost_head_preserves_rows_without_success_binding(tmp_path, monkeypatch):
    from native_consumer_fixture import technical_source, sealed_phase, outward_series
    from qcl_negf_results.commits import json_bytes
    from qcl_negf_results.performance_views import TelemetrySelection
    from qcl_negf_contracts.outward import validate_outward_result
    import hashlib
    import qcl_negf_results.capture as module
    job = tmp_path / 'job'
    context = technical_source(job / 'result', checkpoint=True, execution='exec')
    result, selected = context['root'], context['state']
    sentinel = (context['science'] / selected['manifest']['path']).read_bytes()
    literal = sealed_phase(result, selected['identity'])
    descriptor = dict(status='available', session_id='native', current_path='phase-current.json', reason=None)
    series = outward_series(selected, descriptor)
    validate_outward_result(series)
    (result / 'series_result.json').write_bytes(json_bytes(series))
    point = series['points'][0]
    current = dict(schema='qcl-negf-current-point-v2', point_id=point['id'], execution_id='exec', attempt=int(point['attempt']),
        status='running', updated_unix=100, artifact_root='.', native_source=point['data']['native_source'], phase_source=descriptor,
        stage='scba', iteration=1, total=2, metrics=[], message='synthetic continuity fixture')
    validate_outward_result(current)
    (result / 'current_point.json').write_bytes(json_bytes(current))
    selection = context['transport']['selection']
    plan = json_bytes(dict(fingerprint=selection['plan_fingerprint'], points=[dict(id=point['id'], execution_id='exec')]))
    (job / 'plan.json').write_bytes(plan)
    spec = dict(schema='qcl-negf.capture-spec.v1', selection=selection,
        plan=dict(path='plan.json', bytes=len(plan), sha256=hashlib.sha256(plan).hexdigest()), capture_token='c' * 32,
        allocation_id='allocation', allocated_logical_cpus=2, scratch_root=None,
        credits=dict(performance_bytes=32 << 20, scalar_bytes=0, io_bytes=256 << 20, reserve_bytes=0))
    (job / 'spec.json').write_bytes(json_bytes(spec))
    (job / 'capture-source.json').write_bytes(json_bytes(dict(schema='qcl-negf.capture-source.v1', capture_token='c' * 32,
        execution_id='exec', plan_fingerprint=selection['plan_fingerprint'], mode='direct', actual_root=str(result), epoch=0)))
    technical = job / 'performance'
    technical.mkdir()
    (technical / context['source_binding']['path']).write_bytes((context['technical'] / context['source_binding']['path']).read_bytes())
    read_fd, write_fd = os.pipe2(os.O_NONBLOCK | os.O_CLOEXEC)
    polls = []
    def finish(_):
        polls.append(True)
        (result / 'phase-current.json').write_bytes(json_bytes(dict(literal['current'], publication=None)))
        os.write(write_fd, b'exit 0\n')
    monkeypatch.setattr(module.time, 'sleep', finish)
    monkeypatch.setattr(module, 'proc_sample', lambda *a, **k: dict(wall_seconds=1., monotonic_ns=1_000_000_000,
        cpu_time_seconds=None, missing_reason='synthetic process sample unavailable'))
    def factory(reader, close_ref):
        return TelemetrySelection(technical, commit_ref=close_ref, source_binding_ref=context['source_binding'], expected_transport=context['transport'],
            source_transport=context['source_transport'], canonical_plan_root=context['root'], canonical_plan_ref=context['plan'],
            expected_producer_provenance=context['producer_provenance'])
    try:
        outcome = module.observe(job / 'spec.json', 42, 7, read_fd, transport=context['transport'], telemetry_selection_factory=factory)
    finally:
        os.close(write_fd)
    assert polls == [True]
    assert outcome == 2 and not (technical / 'performance-binding.json').exists()
    pointer = next(technical.glob('sessions/*/current.json'))
    commit_ref = json.loads(pointer.read_bytes())['commit']
    commit = json.loads((technical / commit_ref['path']).read_bytes())
    summary = json.loads((technical / commit['summary']['path']).read_bytes())
    assert commit['closed'] is True
    assert summary['totals']['accepted_source_rows'] == '2'
    assert summary['totals']['accepted_aggregate_calls'] == '3'
    assert commit['source_cursors'][0]['cursor']['next_row'] == '2'
    assert (context['science'] / selected['manifest']['path']).read_bytes() == sentinel


def _T1_native_job(tmp_path):
    """Literal operational inputs and tiny engineering source, no actor authority."""
    import hashlib
    from native_consumer_fixture import technical_source, sealed_phase, outward_series
    from qcl_negf_results.commits import json_bytes
    job = tmp_path / 'upload with spaces'
    job.mkdir()
    result = tmp_path / "jobs with 'quote and space" / 'native' / 'domain'
    c = technical_source(result, checkpoint=True, execution='exec')
    plan_raw = (c['root'] / c['plan']['path']).read_bytes()
    plan = json.loads(plan_raw)
    manifest = c['science'] / c['state']['manifest']['path']
    graph_raw = b' \n' + json_bytes(json.loads(manifest.read_bytes())['provenance']) + b'\n'
    uid = lambda n: f'00000000-0000-4000-8000-{n:012d}'
    claim = lambda n, raw: dict(node_uuid=uid(n), bytes=str(len(raw)), sha256=hashlib.sha256(raw).hexdigest())
    native = dict(schema='qcl-negf.aiida-native-execution.v1', request_id=uid(1), execution_id='exec',
        input_plan=claim(2, plan_raw), source_graph=claim(3, graph_raw), access_receipt_ref='access-receipt',
        storage=dict(namespace_id='native', storage_identity='storage', relative_execution_root='native/domain',
            computer_uuid=uid(4), execution_binding_uuid=uid(5), retention_policy_id='retention',
            retention_owner_id='owner', hold_ref='hold', access_binding_ref='access'),
        code=dict(code_uuid=uid(6), computer_uuid=uid(4), solver_executable='/nix/store/synthetic/bin/qcl-negf',
            release_id='synthetic', build_receipt_ref='build'))
    (job / 'plan.json').write_bytes(plan_raw)
    (job / 'source-graph.json').write_bytes(graph_raw)
    (job / 'native-execution.json').write_bytes(b' \n' + json_bytes(native) + b'\n')
    selection = dict(calcjob_uuid=uid(7), execution_id='exec', attempt=3, plan_fingerprint=plan['fingerprint'])
    spec = dict(schema='qcl-negf.capture-spec.v1', selection=selection,
        plan=dict(path='plan.json', bytes=len(plan_raw), sha256=hashlib.sha256(plan_raw).hexdigest()),
        capture_token='c' * 32, allocation_id=f'calcjob-process:{uid(7)}:3', allocated_logical_cpus=None,
        scratch_root=None, credits=dict(performance_bytes=32 << 20, scalar_bytes=0, io_bytes=128 << 20, reserve_bytes=0))
    (job / 'capture-spec.json').write_bytes(json_bytes(spec))
    (job / 'capture-source.json').write_bytes(json_bytes(dict(schema='qcl-negf.capture-source.v1',
        capture_token='c' * 32, execution_id='exec', plan_fingerprint=selection['plan_fingerprint'],
        mode='direct', actual_root=str(result), epoch=0)))
    sealed_phase(result, c['state']['identity'])
    phase = dict(status='available', session_id='native', current_path='phase-current.json', reason=None)
    (result / 'series_result.json').write_bytes(json_bytes(outward_series(c['state'], phase, fingerprint=plan['fingerprint'])))
    args = ('run-plan', 'plan.json', str(result), '--execution-id', 'exec', '--source-graph', 'source-graph.json')
    return dict(job=job, result=result, context=c, manifest=manifest, native=native, spec=spec, arguments=args)



def test_T1_native_eleven_arguments_accept_original_calcjob(tmp_path):
    from qcl_negf_results.capture import _native_capture_inputs
    from qcl_negf_results.state import _IOBudget
    fixture = _T1_native_job(tmp_path)
    args = (*fixture['arguments'], '--native-execution', 'native-execution.json',
            '--native-calcjob-uuid', fixture['spec']['selection']['calcjob_uuid'])
    output, transport = _native_capture_inputs(fixture['job'], fixture['spec'],
        fixture['job'] / 'native-execution.json', args, budget=_IOBudget(128 << 20))
    assert output == fixture['result']
    assert transport['selection'] == fixture['spec']['selection']


@pytest.mark.parametrize('tail', [
    ('--native-execution', 'other.json', '--native-calcjob-uuid', '00000000-0000-4000-8000-000000000007'),
    ('--native-execution', 'native-execution.json', '--native-calcjob-uuid', '00000000-0000-4000-8000-000000000008'),
    ('--native-execution', 'native-execution.json', '--native-calcjob-uuid', '00000000-0000-4000-8000-00000000000A'),
    ('--native-calcjob-uuid', '00000000-0000-4000-8000-000000000007', '--native-execution', 'native-execution.json'),
    ('--native-execution', 'native-execution.json'),
    ('--native-execution', 'native-execution.json', '--native-calcjob-uuid', '00000000-0000-4000-8000-000000000007', '--native-execution', 'native-execution.json'),
])
def test_T1_native_eleven_invalid_tail_fails_before_output(tmp_path, tail):
    import qcl_negf_results.capture as module
    fixture = _T1_native_job(tmp_path)
    read_fd, write_fd = os.pipe2(os.O_NONBLOCK | os.O_CLOEXEC)
    os.write(write_fd, b'exit 0\n'); os.close(write_fd)
    assert module.observe(fixture['job'] / 'capture-spec.json', 42, 7, read_fd,
        native_execution_path=fixture['job'] / 'native-execution.json',
        solver_arguments=(*fixture['arguments'], *tail)) == 2
    assert not (fixture['job'] / 'performance').exists()

def test_T1_native_fixed_ingress_keeps_raw_without_fabricating_original_transport(tmp_path, monkeypatch):
    """External canonical root must capture phases, never guess source transport."""
    import qcl_negf_results.capture as module
    fixture = _T1_native_job(tmp_path)
    before = {str(p.relative_to(fixture['result'])): p.read_bytes()
              for p in fixture['result'].rglob('*') if p.is_file()}
    monkeypatch.setattr(module, 'proc_sample', lambda *a, **k: dict(wall_seconds=2., monotonic_ns=2_000_000_000,
        cpu_time_seconds=None, missing_reason='synthetic process sample unavailable'))
    read_fd, write_fd = os.pipe2(os.O_NONBLOCK | os.O_CLOEXEC)
    os.write(write_fd, b'exit 7\n'); os.close(write_fd)
    assert module.observe(fixture['job'] / 'capture-spec.json', 42, 7, read_fd,
        native_execution_path=fixture['job'] / 'native-execution.json', solver_arguments=fixture['arguments']) == 2
    technical = fixture['job'] / 'performance'
    pointer = next(technical.glob('sessions/*/current.json'))
    commit_ref = json.loads(pointer.read_bytes())['commit']
    commit = json.loads((technical / commit_ref['path']).read_bytes())
    summary = json.loads((technical / commit['summary']['path']).read_bytes())
    assert commit['closed'] is True and commit['transport']['selection']['attempt'] == 3
    assert commit['source_cursors'][0]['identity']['attempt'] == '1'
    assert commit['source_cursors'][0]['binding']['status'] == 'unbound'
    assert summary['totals']['accepted_source_rows'] == '2'
    assert summary['control_events'] == [dict(event='source_binding_missing', count='1')]
    assert not (technical / 'performance-binding.json').exists()
    assert not (technical / 'performance-bindings').exists()
    assert not (fixture['job'] / 'live').exists()
    assert {str(p.relative_to(fixture['result'])): p.read_bytes()
            for p in fixture['result'].rglob('*') if p.is_file()} == before


def test_T1_cli_native_path_literal_argv_one_solver_and_primary_exit(tmp_path):
    """The real child must receive native context without shell re-parsing argv."""
    fixture = _T1_native_job(tmp_path)
    solver = tmp_path / 'sentinel solver'
    marker = tmp_path / 'literal-argv.json'
    solver.write_text('#!' + sys.executable + '\nimport json, pathlib, sys\n'
        + 'with pathlib.Path(' + repr(str(marker)) + ').open("x") as out: json.dump(sys.argv[1:], out)\nraise SystemExit(23)\n')
    solver.chmod(0o700)
    process = subprocess.run([sys.executable, '-c', 'from qcl_negf_results.cli import main; main()', 'capture-run',
        '--spec', str(fixture['job'] / 'capture-spec.json'), '--solver', str(solver),
        '--native-execution', str(fixture['job'] / 'native-execution.json'), '--', *fixture['arguments']],
        cwd=fixture['job'], capture_output=True, timeout=15)
    assert process.returncode == 23, process.stderr.decode()
    assert json.loads(marker.read_bytes()) == list(fixture['arguments'])
    assert len(list((fixture['job'] / 'performance').glob('sessions/*/current.json'))) == 1
    assert process.stdout == process.stderr == b''


@pytest.mark.parametrize('fault', ['callback', 'transport', 'plan', 'graph', 'output', 'locator', 'attempt', 'duplicate', 'scratch', 'credit'])
def test_T1_native_invalid_inputs_fail_before_observer_output(tmp_path, fault):
    import qcl_negf_results.capture as module
    fixture = _T1_native_job(tmp_path)
    args = list(fixture['arguments'])
    kwargs = {}
    if fault == 'callback':
        kwargs['telemetry_selection_factory'] = lambda *_: pytest.fail('native caller callback executed')
    elif fault == 'transport':
        kwargs['transport'] = dict(selection=fixture['spec']['selection'], frozen_plan=None)
    elif fault in ('plan', 'graph'):
        target = fixture['job'] / ('plan.json' if fault == 'plan' else 'source-graph.json')
        target.write_bytes(target.read_bytes() + b' ')
    elif fault == 'output':
        args[2] = str(tmp_path / 'foreign')
    elif fault == 'locator':
        target = fixture['job'] / 'capture-source.json'
        value = json.loads(target.read_bytes()); value['actual_root'] = str(tmp_path / 'foreign')
        target.write_text(json.dumps(value))
    elif fault == 'attempt':
        args += ['--attempt', '3']
    elif fault == 'duplicate':
        args += ['--execution-id', 'exec']
    elif fault == 'scratch':
        args += ['--scratch-root', str(tmp_path / 'scratch')]
    elif fault == 'credit':
        target = fixture['job'] / 'capture-spec.json'
        value = json.loads(target.read_bytes()); value['credits']['io_bytes'] = 16384
        target.write_text(json.dumps(value))
    read_fd, write_fd = os.pipe2(os.O_NONBLOCK | os.O_CLOEXEC)
    os.write(write_fd, b'exit 0\n'); os.close(write_fd)
    assert module.observe(fixture['job'] / 'capture-spec.json', 42, 7, read_fd,
        native_execution_path=fixture['job'] / 'native-execution.json', solver_arguments=tuple(args), **kwargs) == 2
    assert not (fixture['job'] / 'performance').exists()


def test_T1_terminal_attaches_public_close_phase_resume_without_changing_callback(tmp_path, monkeypatch):
    """Dropping exact old completion/resume refs must break actual close binding."""
    import hashlib
    from native_consumer_fixture import technical_source, sealed_phase, outward_series
    from result_v2_fixture import reference
    from qcl_negf_results.commits import json_bytes
    from qcl_negf_results.telemetry import TelemetryWriter
    from qcl_negf_results.state import StateReader, _IOBudget
    from qcl_negf_results.performance_views import TelemetrySelection, read_performance_context
    import qcl_negf_results.capture as module
    job = tmp_path / 'job'; job.mkdir()
    result = job / 'result'
    c = technical_source(result, checkpoint=True, execution='exec')
    technical = job / 'performance'; technical.mkdir()
    (technical / c['source_binding']['path']).write_bytes((c['technical'] / c['source_binding']['path']).read_bytes())
    sentinel = (c['science'] / c['state']['manifest']['path']).read_bytes()
    producer = tmp_path / 'producer'
    first = sealed_phase(producer, c['state']['identity'])
    descriptor = dict(status='available', session_id='native', current_path='phase-current.json', reason=None)
    with module.PhaseCaptureSource(producer, descriptor=descriptor, expected_identity=c['state']['identity'], route_epoch=0) as source:
        batch = source.harvest(None, budget=_IOBudget(1 << 20))
        writer = TelemetryWriter(technical, {}, {}, transport=c['transport'])
        writer.stage_phase_batch(batch)
        accepted = json.loads(writer.flush().read_bytes())['accepted_ranges'][0]
        old_path = writer.rotate(); old = json.loads(old_path.read_bytes())
        resume = dict(commit_ref=reference(technical, str(old_path.relative_to(technical)), 'qcl-negf.telemetry-commit.v1'), summary_ref=old['summary'])
        completion = dict(publication_ref=accepted['publication'], segment_ref=accepted['segment'],
            retained_publication_ref=accepted['retained_publication'], retained_segment_ref=accepted['retained_segment'])
        writer.record_resource(dict(wall_seconds=1.)); writer.flush()
        sealed_phase(producer, c['state']['identity'], previous=first['publication_ref'], generation=2)
        writer.stage_phase_batch(source.harvest(batch.cursor, budget=_IOBudget(1 << 20)),
            phase_context_refs=(completion,), resume_context_refs=(resume,))
    unavailable = dict(status='unavailable', session_id=None, current_path=None, reason='not_recorded')
    (result / 'series_result.json').write_bytes(json_bytes(outward_series(c['state'], unavailable)))
    raw = json_bytes(dict(fingerprint='a' * 64, points=[dict(id=c['state']['identity']['point_id'], execution_id='exec')]))
    (job / 'plan.json').write_bytes(raw)
    (job / 'capture-spec.json').write_bytes(json_bytes(dict(schema='qcl-negf.capture-spec.v1', selection=c['transport']['selection'],
        plan=dict(path='plan.json', bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest()), capture_token='c' * 32,
        allocation_id='allocation', allocated_logical_cpus=None, scratch_root=None,
        credits=dict(performance_bytes=32 << 20, scalar_bytes=0, io_bytes=128 << 20, reserve_bytes=0))))
    (job / 'capture-source.json').write_bytes(json_bytes(dict(schema='qcl-negf.capture-source.v1', capture_token='c' * 32,
        execution_id='exec', plan_fingerprint='a' * 64, mode='direct', actual_root=str(result), epoch=0)))
    calls = []
    def factory(reader, close_ref):
        calls.append(reader.source.state)
        return TelemetrySelection(technical, commit_ref=close_ref, source_binding_ref=c['source_binding'], expected_transport=c['transport'],
            source_transport=c['source_transport'], canonical_plan_root=c['root'], canonical_plan_ref=c['plan'], expected_producer_provenance=c['producer_provenance'])
    observed_files = []
    make_binding = module.make_binding
    def capture_binding(reader, *, telemetry, destination, capture_status, budget=None):
        value = make_binding(reader, telemetry=telemetry, destination=destination, capture_status=capture_status, budget=budget)
        inputs = telemetry.document()
        inputs['binding_ref'] = reference(technical, str(destination.relative_to(technical)), 'qcl-negf.telemetry-binding.v1')
        selected = TelemetrySelection(technical, canonical_plan_root=telemetry.canonical_plan_root, **inputs)
        observed_files.extend(read_performance_context(reader, telemetry=selected)[1])
        return value
    monkeypatch.setattr(module, 'make_binding', capture_binding)
    monkeypatch.setattr('qcl_negf_results.telemetry.TelemetryWriter', lambda *a, **k: writer)
    monkeypatch.setattr(module, 'proc_sample', lambda *a, **k: dict(wall_seconds=2., monotonic_ns=2_000_000_000,
        cpu_time_seconds=None, missing_reason='synthetic sample'))
    read_fd, write_fd = os.pipe2(os.O_NONBLOCK | os.O_CLOEXEC)
    os.write(write_fd, b'exit 0\n'); os.close(write_fd)
    assert module.observe(job / 'capture-spec.json', 42, 7, read_fd,
        transport=c['transport'], telemetry_selection_factory=factory) == 0
    assert calls == [c['state']]
    assert any(ref == completion['retained_segment_ref'] for _, ref, _ in observed_files)
    assert any(ref == resume['commit_ref'] for _, ref, _ in observed_files)
    binding = reference(technical, 'performance-binding.json', 'qcl-negf.telemetry-binding.v1')
    selected = TelemetrySelection(technical, commit_ref=writer.close_context()[0], source_binding_ref=c['source_binding'],
        expected_transport=c['transport'], source_transport=c['source_transport'], canonical_plan_root=c['root'],
        canonical_plan_ref=c['plan'], expected_producer_provenance=c['producer_provenance'], binding_ref=binding,
        phase_context_refs=(completion,), resume_context_refs=(resume,))
    with StateReader(c['science'], state_ref=c['state'], expected_identity=c['state']['identity']) as reader:
        _, files = read_performance_context(reader, telemetry=selected)
    assert any(ref == completion['retained_segment_ref'] for _, ref, _ in files)
    assert any(ref == resume['commit_ref'] for _, ref, _ in files)
    assert (c['science'] / c['state']['manifest']['path']).read_bytes() == sentinel


@pytest.mark.parametrize('route', ['capture_run', 'runner', 'runner-term'])
def test_T1_F1_actual_observer_retains_two_phase_rows_and_original_raw(tmp_path, monkeypatch, route):
    """A __main__/canonical PhaseBatch split must not pass as a closed capture."""
    from qcl_negf_results.capture import capture_run
    fixture = _T1_native_job(tmp_path)
    source = fixture['result']; job = fixture['job']
    before = {str(p.relative_to(source)): p.read_bytes() for p in source.rglob('*') if p.is_file()}
    assert len(before) == 22
    inputs = {name: (job / name).read_bytes() for name in ('plan.json', 'source-graph.json', 'native-execution.json')}
    marker = tmp_path / 'single-launch.json'
    solver = tmp_path / 'one SENTINEL solver'
    solver.write_text('#!' + sys.executable + '\nimport json, pathlib, signal, sys, time\n'
        + 'def stop(*_): raise SystemExit(143)\n'
        + 'signal.signal(signal.SIGTERM, stop)\n'
        + 'with pathlib.Path(' + repr(str(marker)) + ').open("x") as out: json.dump(sys.argv[1:], out)\n'
        + ('while True: time.sleep(.01)\n' if route == 'runner-term' else 'raise SystemExit(29)\n'))
    solver.chmod(0o700)
    if route == 'capture_run':
        code = capture_run(job / 'capture-spec.json', str(solver), fixture['arguments'],
                           native_execution_path=job / 'native-execution.json')
    else:
        template = os.environ.get('QCL_NEGF_TEST_RUNNER_LAUNCHER')
        if not template:
            pytest.skip('paired SOURCE/L requires the explicitly selected frozen Runner launcher')
        capture = tmp_path / 'real Results CLI'
        capture.write_text('#!' + sys.executable + '\nfrom qcl_negf_results.cli import main\nmain()\n')
        capture.chmod(0o700)
        launcher = tmp_path / 'frozen Runner launcher'
        launcher.write_text(Path(template).read_text().replace('@bash@', '/bin/bash')
            .replace('@solver@', str(solver)).replace('@capture@', str(capture)))
        launcher.chmod(0o700)
        env = dict(os.environ, QCL_NEGF_CAPTURE_SPEC=str(job / 'capture-spec.json'),
                   QCL_NEGF_NATIVE_CAPTURE_INPUT=str(job / 'native-execution.json'))
        process = subprocess.Popen([str(launcher), *fixture['arguments']], cwd=job, env=env,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            if route == 'runner-term':
                deadline = time.monotonic() + 5
                while not marker.exists():
                    assert time.monotonic() < deadline and process.poll() is None
                    time.sleep(.01)
                process.send_signal(signal.SIGTERM)
            stdout, stderr = process.communicate(timeout=15)
            code = process.returncode
            assert stdout == stderr == b''
        finally:
            if process.poll() is None:
                process.kill(); process.communicate(timeout=3)
    assert code == (143 if route == 'runner-term' else 29)
    assert json.loads(marker.read_bytes()) == list(fixture['arguments'])
    technical = job / 'performance'
    pointers = list(technical.glob('sessions/*/current.json'))
    assert len(pointers) == 1
    reference = json.loads(pointers[0].read_bytes())['commit']
    commit = json.loads((technical / reference['path']).read_bytes())
    summary = json.loads((technical / commit['summary']['path']).read_bytes())
    assert commit['closed'] is True
    assert summary['totals']['accepted_source_rows'] == '2'
    assert len(commit['source_cursors']) == 1
    assert commit['source_cursors'][0]['binding']['status'] == 'unbound'
    assert commit['source_cursors'][0]['identity']['attempt'] == '1'
    assert commit['transport']['selection']['attempt'] == 3
    assert summary['control_events'] == [dict(event='source_binding_missing', count='1')]
    retained = [p.read_bytes() for p in technical.glob('sessions/*/raw/*')]
    assert (source / 'phase-segment-1.json').read_bytes() in retained
    assert (source / 'phase-publication-1.json').read_bytes() in retained
    assert not (technical / 'performance-binding.json').exists()
    assert not (technical / 'performance-bindings').exists()
    assert not (job / 'live').exists()
    assert {str(p.relative_to(source)): p.read_bytes() for p in source.rglob('*') if p.is_file()} == before
    assert {name: (job / name).read_bytes() for name in inputs} == inputs


@pytest.mark.parametrize('route', ['runner', 'runner-term'])
def test_T1_eleven_actual_observer_retains_two_phase_rows_and_original_raw(tmp_path, monkeypatch, route):
    """A __main__/canonical PhaseBatch split must not pass as a closed capture."""
    from qcl_negf_results.capture import capture_run
    fixture = _T1_native_job(tmp_path)
    fixture['arguments'] = (*fixture['arguments'], '--native-execution', 'native-execution.json',
                            '--native-calcjob-uuid', fixture['spec']['selection']['calcjob_uuid'])
    source = fixture['result']; job = fixture['job']
    before = {str(p.relative_to(source)): p.read_bytes() for p in source.rglob('*') if p.is_file()}
    assert len(before) == 22
    inputs = {name: (job / name).read_bytes() for name in ('plan.json', 'source-graph.json', 'native-execution.json')}
    marker = tmp_path / 'single-launch.json'
    solver = tmp_path / 'one SENTINEL solver'
    solver.write_text('#!' + sys.executable + '\nimport json, pathlib, signal, sys, time\n'
        + 'def stop(*_): raise SystemExit(143)\n'
        + 'signal.signal(signal.SIGTERM, stop)\n'
        + 'with pathlib.Path(' + repr(str(marker)) + ').open("x") as out: json.dump(sys.argv[1:], out)\n'
        + ('while True: time.sleep(.01)\n' if route == 'runner-term' else 'raise SystemExit(29)\n'))
    solver.chmod(0o700)
    if route == 'capture_run':
        code = capture_run(job / 'capture-spec.json', str(solver), fixture['arguments'],
                           native_execution_path=job / 'native-execution.json')
    else:
        template = os.environ.get('QCL_NEGF_TEST_RUNNER_LAUNCHER')
        if not template:
            pytest.skip('paired SOURCE/L requires the explicitly selected frozen Runner launcher')
        capture = tmp_path / 'real Results CLI'
        capture.write_text('#!' + sys.executable + '\nfrom qcl_negf_results.cli import main\nmain()\n')
        capture.chmod(0o700)
        launcher = tmp_path / 'frozen Runner launcher'
        launcher.write_text(Path(template).read_text().replace('@bash@', '/bin/bash')
            .replace('@solver@', str(solver)).replace('@capture@', str(capture)))
        launcher.chmod(0o700)
        env = dict(os.environ, QCL_NEGF_CAPTURE_SPEC=str(job / 'capture-spec.json'),
                   QCL_NEGF_NATIVE_CAPTURE_INPUT=str(job / 'native-execution.json'))
        process = subprocess.Popen([str(launcher), *fixture['arguments']], cwd=job, env=env,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            if route == 'runner-term':
                deadline = time.monotonic() + 5
                while not marker.exists():
                    assert time.monotonic() < deadline and process.poll() is None
                    time.sleep(.01)
                process.send_signal(signal.SIGTERM)
            stdout, stderr = process.communicate(timeout=15)
            code = process.returncode
            assert stdout == stderr == b''
        finally:
            if process.poll() is None:
                process.kill(); process.communicate(timeout=3)
    assert code == (143 if route == 'runner-term' else 29)
    assert json.loads(marker.read_bytes()) == list(fixture['arguments'])
    technical = job / 'performance'
    pointers = list(technical.glob('sessions/*/current.json'))
    assert len(pointers) == 1
    reference = json.loads(pointers[0].read_bytes())['commit']
    commit = json.loads((technical / reference['path']).read_bytes())
    summary = json.loads((technical / commit['summary']['path']).read_bytes())
    assert commit['closed'] is True
    assert summary['totals']['accepted_source_rows'] == '2'
    assert len(commit['source_cursors']) == 1
    assert commit['source_cursors'][0]['binding']['status'] == 'unbound'
    assert commit['source_cursors'][0]['identity']['attempt'] == '1'
    assert commit['transport']['selection']['attempt'] == 3
    assert summary['control_events'] == [dict(event='source_binding_missing', count='1')]
    retained = [p.read_bytes() for p in technical.glob('sessions/*/raw/*')]
    assert (source / 'phase-segment-1.json').read_bytes() in retained
    assert (source / 'phase-publication-1.json').read_bytes() in retained
    assert not (technical / 'performance-binding.json').exists()
    assert not (technical / 'performance-bindings').exists()
    assert not (job / 'live').exists()
    assert {str(p.relative_to(source)): p.read_bytes() for p in source.rglob('*') if p.is_file()} == before
    assert {name: (job / name).read_bytes() for name in inputs} == inputs


def test_S3_make_binding_dispatches_v2_without_changing_source_binding(tmp_path):
    from native_consumer_fixture import technical_source,telemetry_input
    from qcl_negf_results.telemetry import TelemetryWriter
    c=technical_source(tmp_path/'input');old=c['technical'];c['technical']=old/'retained'
    writer=TelemetryWriter(c['technical'],{}, {},transport=c['transport'],technical_version=2)
    original=(old/c['source_binding']['path']).read_bytes()
    (c['technical']/c['source_binding']['path']).write_bytes(original)
    writer.record_resource(dict(wall_seconds=1.));writer.close()
    from test_telemetry import s3_selected
    selected=s3_selected(c,writer)
    binding=json.loads((c['technical']/selected.binding_ref['path']).read_bytes())
    assert binding['schema']=='qcl-negf.telemetry-binding.v2'
    assert binding['source_binding']['raw_utf8'].encode()==original


def _T3_anchor(fixture, *, pending=False):
    """Publish literal T2 wire, retaining its original whitespace and immutable bytes."""
    from copy import deepcopy
    from qcl_negf_results.commits import json_bytes
    c = fixture['context']
    manifest = json.loads(fixture['manifest'].read_bytes())
    value = dict(schema='qcl-negf.performance-anchor.v1', contract_set='qcl-negf.results.v2',
        phase_identity=deepcopy(c['state']['identity']),
        phase_source=dict(status='available', session_id='native', current_path='phase-current.json', reason=None),
        origin=None, invocation=deepcopy(manifest['invocation']), scientific=None)
    if not pending:
        value['scientific'] = dict(artifact_root='science', selected_state=deepcopy(c['state']),
            manifest_utf8=fixture['manifest'].read_bytes().decode(), invocation=deepcopy(manifest['invocation']),
            source_origin=None, relation='produced')
    raw = b' \n' + json_bytes(value) + b'\n'
    target = fixture['result'] / 'performance-anchors' / (__import__('hashlib').sha256(raw).hexdigest() + '.json')
    target.parent.mkdir(exist_ok=True)
    target.write_bytes(raw)
    (fixture['result'] / 'performance-anchor-current.json').write_bytes(raw)
    return value, raw, target


def _N3_fixture(tmp_path, *, produced=False):
    """Literal NEW engineering wire and real local identities, no origin authority."""
    import hashlib
    import shutil
    import qcl_negf_results.capture as cap
    from native_consumer_fixture import technical_source, sealed_phase
    f = _T1_native_job(tmp_path)
    f['spec']['credits']['scalar_bytes'] = 64 << 20
    f['spec'].update(schema='qcl-negf.capture-spec.v3',
        native_performance=dict(run_uuid='00000000-0000-4000-8000-000000000009'))
    f['result'].chmod(0o700)
    f['arguments'] = (*f['arguments'], '--native-execution', 'native-execution.json', '--native-calcjob-uuid', f['spec']['selection']['calcjob_uuid'])
    if produced:
        c = technical_source(tmp_path / 'history-only-engineering-source', checkpoint=False, execution='exec')
        for child in c['science'].iterdir():
            destination = f['result'] / child.name
            if child.is_dir():shutil.copytree(child, destination)
            else:shutil.copyfile(child, destination)
        f['context'] = c
        f['manifest'] = f['result'] / c['state']['manifest']['path']
        plan_raw = (c['root'] / c['plan']['path']).read_bytes()
        (f['job'] / 'plan.json').write_bytes(plan_raw)
        f['spec']['plan'].update(bytes=len(plan_raw),sha256=hashlib.sha256(plan_raw).hexdigest())
        f['spec']['selection']['plan_fingerprint'] = json.loads(plan_raw)['fingerprint']
        f['native']['input_plan'].update(bytes=str(len(plan_raw)),sha256=hashlib.sha256(plan_raw).hexdigest())
        (f['job'] / 'native-execution.json').write_bytes(cap.json_bytes(f['native']))
        sealed_phase(f['result'], c['state']['identity'])
    anchor, raw, original = _T3_anchor(f, pending=not produced)
    anchor['origin'] = dict(calcjob_uuid=f['spec']['selection']['calcjob_uuid'],
        native_execution_uuid=f['native']['storage']['execution_binding_uuid'])
    if produced:
        anchor['scientific']['artifact_root'] = '.'
        anchor['scientific']['source_origin'] = dict(anchor['origin'])
    raw = b' \n' + cap.json_bytes(anchor) + b'\n'
    original = f['result'] / 'performance-anchors' / (hashlib.sha256(raw).hexdigest() + '.json')
    original.write_bytes(raw);(f['result'] / 'performance-anchor-current.json').write_bytes(raw)
    ns = os.stat('/proc/self/ns/mnt'); born = f['result'].stat()
    f['birth'] = dict(boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        mount_namespace_device=str(ns.st_dev), mount_namespace_inode=str(ns.st_ino),
        producer_pid=str(os.getpid()), producer_start_ticks=str(cap._start_ticks(os.getpid())),
        device=str(born.st_dev), inode=str(born.st_ino))
    f['locator'] = dict(schema='qcl-negf.capture-source.v3', capture_token=f['spec']['capture_token'],
        execution_id='exec', plan_fingerprint=f['spec']['selection']['plan_fingerprint'], mode='direct',
        actual_root=str(f['result']), epoch=0, root_birth=f['birth'])
    (f['job'] / 'capture-spec.json').write_bytes(cap.json_bytes(f['spec']))
    (f['job'] / 'capture-source.json').write_bytes(cap.json_bytes(f['locator']))
    return f


def _N3_observe(f):
    import qcl_negf_results.capture as cap
    read_fd, write_fd = os.pipe2(os.O_NONBLOCK | os.O_CLOEXEC)
    os.write(write_fd, b'exit 0\n');os.close(write_fd)
    return cap.observe(f['job'] / 'capture-spec.json', int(f['birth']['producer_pid']),
        int(f['birth']['producer_start_ticks']), read_fd,
        native_execution_path=f['job'] / 'native-execution.json', solver_arguments=f['arguments'])


@pytest.mark.parametrize('produced', [False, True])
def test_N3_observe_result_domain_phase0_and_produced_history_only(tmp_path, produced):
    f = _N3_fixture(tmp_path, produced=produced)
    assert _N3_observe(f) == (0 if produced else 2)
    terminal = f['result'] / 'performance-source.json'
    assert terminal.exists()
    index = json.loads(terminal.read_bytes())
    assert index['closed'] and len(index['entries']) == int(produced)
    assert index['provider_selection']['run_uuid'] == f['spec']['native_performance']['run_uuid']
    assert index['provider_selection']['run_uuid'] != f['native']['request_id']
    assert index['technical']['root']['area'] == 'native_execution'
    assert not (f['job'] / 'performance').exists() and not (f['job'] / 'performance-source.json').exists()
    if produced:
        row = index['entries'][0]
        assert row['canonical_plan_root']['area'] == 'native_workdir'
        assert row['scientific_root']['area'] == 'native_execution'
        assert row['telemetry']['canonical_plan_ref']['sha256'] == f['spec']['plan']['sha256']
        assert json.loads(f['manifest'].read_bytes())['checkpoint'] is None


def test_N3_spec3_and_locator3_are_the_exact_new_ingress(tmp_path):
    import qcl_negf_results.capture as cap
    from qcl_negf_results._capture_io import ObserverIO
    f = _N3_fixture(tmp_path)
    root, spec, known = cap._spec(f['job'] / 'capture-spec.json', budget=ObserverIO(128 << 20))
    assert root == f['job'] and known == {f['context']['state']['identity']['point_id']} and spec == f['spec']
    locator, actual = cap._locator(root, spec, budget=ObserverIO(1 << 20), native_output=f['result'])
    assert locator == f['locator'] and actual == f['result']


@pytest.mark.parametrize('alias', ['/', '/.', 'double-separator'])
@pytest.mark.parametrize('role', ['locator', 'original_argv'])
def test_N3_F1_literal_output_alias_refuses_before_factory_and_effects(tmp_path, monkeypatch, alias, role):
    import qcl_negf_results.capture as cap
    from qcl_negf_results._capture_io import ObserverIO
    f = _N3_fixture(tmp_path, produced=True)
    literal = str(f['result'])
    candidate = literal.replace('/native/', '/native//') if alias == 'double-separator' else literal + alias
    assert candidate != literal and Path(candidate) == f['result']
    if role == 'locator':
        f['locator']['actual_root'] = candidate
        (f['job'] / 'capture-source.json').write_bytes(cap.json_bytes(f['locator']))
        with pytest.raises(ValueError):
            cap._locator(f['job'], f['spec'], budget=ObserverIO(16384), native_output=f['result'])
    else:
        f['arguments'] = (*f['arguments'][:2], candidate, *f['arguments'][3:])
        with pytest.raises(ValueError):
            cap._native_capture_inputs(f['job'], f['spec'], f['job'] / 'native-execution.json',
                f['arguments'], budget=ObserverIO(128 << 20))
    monkeypatch.setattr('qcl_negf_results.telemetry.TelemetryWriter',
        lambda *a, **k: pytest.fail('native3 path alias reached writer constructor'))
    monkeypatch.setattr(cap, 'NativePerformanceCapture',
        lambda *a, **k: pytest.fail('native3 path alias reached capture factory'))
    before = len(list(Path('/proc/self/fd').iterdir()))
    assert _N3_observe(f) == 2
    assert len(list(Path('/proc/self/fd').iterdir())) == before
    for root in (f['job'], f['result']):
        assert not (root / 'performance').exists() and not (root / 'performance-source.json').exists()


@pytest.mark.parametrize('alias', ['/', '/.', 'double-separator'])
def test_N3_F1_literal_strictness_preserves_v1_path_alias_semantics(tmp_path, alias):
    import qcl_negf_results.capture as cap
    from qcl_negf_results._capture_io import ObserverIO
    f = _T1_native_job(tmp_path)
    literal = str(f['result'])
    candidate = literal.replace('/native/', '/native//') if alias == 'double-separator' else literal + alias
    arguments = (*f['arguments'][:2], candidate, *f['arguments'][3:])
    output, _ = cap._native_capture_inputs(f['job'], f['spec'], f['job'] / 'native-execution.json',
        arguments, budget=ObserverIO(128 << 20))
    locator = json.loads((f['job'] / 'capture-source.json').read_bytes())
    locator['actual_root'] = candidate
    (f['job'] / 'capture-source.json').write_bytes(cap.json_bytes(locator))
    _, actual = cap._locator(f['job'], f['spec'], budget=ObserverIO(16384), native_output=output)
    assert actual == output == f['result']


@pytest.mark.parametrize('fault', ['pid', 'start', 'boot', 'namespace', 'inode', 'mode', 'existing', 'old_locator'])
def test_N3_invalid_birth_has_zero_writer_effects(tmp_path, monkeypatch, fault):
    import qcl_negf_results.capture as cap
    f = _N3_fixture(tmp_path)
    field = dict(pid='producer_pid', start='producer_start_ticks', boot='boot_id',
        namespace='mount_namespace_inode', inode='inode').get(fault)
    if field:f['birth'][field] = '00000000-0000-4000-8000-000000000099' if field == 'boot_id' else str(int(f['birth'][field])+1)
    elif fault == 'mode':f['result'].chmod(0o755)
    elif fault == 'existing':(f['result'] / 'performance').mkdir()
    else:f['locator'].pop('root_birth');f['locator']['schema'] = 'qcl-negf.capture-source.v1'
    (f['job'] / 'capture-source.json').write_bytes(cap.json_bytes(f['locator']))
    monkeypatch.setattr('qcl_negf_results.telemetry.TelemetryWriter', lambda *a, **k: pytest.fail('birth refused after writer effect'))
    # Supervisor generation is independent of the altered locator assertion.
    read_fd, write_fd = os.pipe2(os.O_NONBLOCK | os.O_CLOEXEC);os.write(write_fd,b'exit 0\n');os.close(write_fd)
    assert cap.observe(f['job'] / 'capture-spec.json', os.getpid(), cap._start_ticks(os.getpid()), read_fd,
        native_execution_path=f['job'] / 'native-execution.json', solver_arguments=f['arguments']) == 2
    assert not (f['job'] / 'performance').exists()
    assert not (f['result'] / 'performance-source.json').exists()


@pytest.mark.parametrize('field,value', [('inode', True), ('device', 1), ('producer_pid', '01'),
    ('mount_namespace_inode', '0'), ('device', str(2**64)), ('boot_id', '00000000-0000-0000-0000-000000000000')])
def test_N3_locator_birth_strict_types_before_effects(tmp_path, field, value):
    import qcl_negf_results.capture as cap
    from qcl_negf_results._capture_io import ObserverIO
    f = _N3_fixture(tmp_path);f['birth'][field] = value
    (f['job'] / 'capture-source.json').write_bytes(cap.json_bytes(f['locator']))
    with pytest.raises(ValueError):cap._locator(f['job'],f['spec'],budget=ObserverIO(16384),native_output=f['result'])
    assert not (f['result'] / 'performance').exists()


@pytest.mark.parametrize('fault', ['extra', 'bool_attempt', 'run_uuid', 'v2', 'scratch', 'duplicates'])
def test_N3_spec_is_closed_and_never_downgrades(tmp_path, fault):
    import qcl_negf_results.capture as cap
    from qcl_negf_results._capture_io import ObserverIO
    f = _N3_fixture(tmp_path);spec = f['spec']
    if fault == 'extra':spec['extra'] = None
    elif fault == 'bool_attempt':spec['selection']['attempt'] = True
    elif fault == 'run_uuid':spec['native_performance']['run_uuid'] = '00000000-0000-0000-0000-000000000000'
    elif fault == 'v2':spec['schema'] = 'qcl-negf.capture-spec.v2'
    elif fault == 'scratch':spec['scratch_root'] = str(tmp_path)
    raw = cap.json_bytes(spec)
    if fault == 'duplicates':raw = raw[:-1] + b',"native_performance":null}'
    (f['job'] / 'capture-spec.json').write_bytes(raw)
    with pytest.raises(ValueError):cap._spec(f['job'] / 'capture-spec.json',budget=ObserverIO(16384))


def test_N3_before_first_open_replacement_has_zero_constructor_effect(tmp_path, monkeypatch):
    import qcl_negf_results.capture as cap
    f = _N3_fixture(tmp_path);saved=f['result'].with_name('born-original-owned');changed=[]
    locate=cap._locator
    def replaced(*a, **k):
        value=locate(*a, **k)
        if not changed:f['result'].rename(saved);f['result'].mkdir(mode=0o700);changed.append(True)
        return value
    monkeypatch.setattr(cap,'_locator',replaced)
    monkeypatch.setattr('qcl_negf_results.telemetry.TelemetryWriter',lambda *a,**k:pytest.fail('writer before SAME'))
    assert _N3_observe(f)==2 and changed
    assert list(f['result'].iterdir())==[]


@pytest.mark.parametrize('schedule', ['observer_first', 'science_first'])
def test_N3_held_guard_accepts_legitimate_children_and_holds_final_identity(tmp_path, schedule):
    import qcl_negf_results.capture as cap
    from qcl_negf_results._capture_io import ObserverIO
    f = _N3_fixture(tmp_path);budget=ObserverIO(128 << 20)
    before=len(list(Path('/proc/self/fd').iterdir()))
    if schedule=='science_first':(f['result'] / 'science-child').mkdir()
    guard=cap._NativeResultGuard(f['job'],f['result'],f['birth'],os.getpid(),cap._start_ticks(os.getpid()),budget)
    try:
        guard.create_technical()
        if schedule=='observer_first':(f['result'] / 'science-child').mkdir()
        guard.validate()
        with guard.effect():(f['result'] / 'performance' / 'owned.json').write_bytes(b'{}')
        assert budget.read_bytes>0 and budget.reserved==guard.cleanup_reserved
    finally:guard.close()
    assert budget.reserved==0 and len(list(Path('/proc/self/fd').iterdir()))==before


@pytest.mark.parametrize('role', ['workdir', 'result', 'technical'])
@pytest.mark.parametrize('cancel', [False, True])
def test_N3_late_name_change_keeps_primary_and_closes_every_fd(tmp_path, role, cancel):
    import qcl_negf_results.capture as cap
    from qcl_negf_results._capture_io import ObserverIO
    f=_N3_fixture(tmp_path);budget=ObserverIO(128 << 20)
    before=len(list(Path('/proc/self/fd').iterdir()))
    guard=cap._NativeResultGuard(f['job'],f['result'],f['birth'],os.getpid(),cap._start_ticks(os.getpid()),budget)
    guard.create_technical()
    target=dict(workdir=f['job'],result=f['result'],technical=f['result']/'performance')[role]
    saved=target.with_name(target.name+'-moved-owned')
    primary=KeyboardInterrupt('native3 cancellation') if cancel else OSError('native3 primary')
    try:
        with pytest.raises(type(primary)) as caught:
            with guard.effect():
                target.rename(saved);target.mkdir(mode=0o700)
                raise primary
        assert caught.value is primary and any('effect validation' in x for x in primary.__notes__)
    finally:
        target.rmdir();saved.rename(target);guard.close()
    assert budget.reserved==0 and len(list(Path('/proc/self/fd').iterdir()))==before


@pytest.mark.parametrize('stage', ['spec', 'native_inputs'])
def test_N3_startup_cancellation_preserves_primary_and_closes_control_fd(tmp_path, monkeypatch, stage):
    import qcl_negf_results.capture as cap
    f=_N3_fixture(tmp_path);primary=KeyboardInterrupt('native3 startup cancelled')
    def cancelled(*a,**k):raise primary
    monkeypatch.setattr(cap,'_spec' if stage=='spec' else '_native_capture_inputs',cancelled)
    read_fd,write_fd=os.pipe2(os.O_NONBLOCK|os.O_CLOEXEC)
    try:
        with pytest.raises(KeyboardInterrupt) as caught:
            cap.observe(f['job']/'capture-spec.json',os.getpid(),cap._start_ticks(os.getpid()),read_fd,
                native_execution_path=f['job']/'native-execution.json',solver_arguments=f['arguments'])
        assert caught.value is primary
        with pytest.raises(OSError):os.fstat(read_fd)
        assert not (f['result']/'performance').exists()
    finally:
        os.close(write_fd)
        try:os.close(read_fd)
        except OSError:pass


def test_N3_actual_held_fd_close_failure_does_not_leak_other_descriptors(tmp_path):
    import qcl_negf_results.capture as cap
    from qcl_negf_results._capture_io import ObserverIO
    f=_N3_fixture(tmp_path);budget=ObserverIO(128 << 20)
    before=len(list(Path('/proc/self/fd').iterdir()))
    guard=cap._NativeResultGuard(f['job'],f['result'],f['birth'],os.getpid(),cap._start_ticks(os.getpid()),budget)
    guard.create_technical();os.close(guard.entries[-1][0])
    with pytest.raises(OSError):guard.close()
    assert guard.closed and budget.reserved==0
    guard.close()  # No retry of an uncertain descriptor number.
    assert len(list(Path('/proc/self/fd').iterdir()))==before


@pytest.mark.parametrize('fault', ['namespace', 'mount', 'mode'])
def test_N3_late_scope_or_credentials_change_refuses_before_next_effect(tmp_path, monkeypatch, fault):
    import qcl_negf_results.capture as cap
    from qcl_negf_results._capture_io import ObserverIO
    from types import SimpleNamespace
    f=_N3_fixture(tmp_path);budget=ObserverIO(128 << 20)
    guard=cap._NativeResultGuard(f['job'],f['result'],f['birth'],os.getpid(),cap._start_ticks(os.getpid()),budget)
    guard.create_technical();effects=[]
    try:
        with monkeypatch.context() as patch:
            if fault=='namespace':
                original=cap.os.stat
                def changed(path,*a,**k):
                    value=original(path,*a,**k)
                    return SimpleNamespace(st_dev=value.st_dev,st_ino=value.st_ino+1) if str(path)=='/proc/self/ns/mnt' else value
                patch.setattr(cap.os,'stat',changed)
            elif fault=='mount':
                original=guard._mount;patch.setattr(guard,'_mount',lambda fd:original(fd)+1)
            else:f['result'].chmod(0o750)
            with pytest.raises(ValueError):
                with guard.effect():effects.append(True)
        assert effects==[]
    finally:f['result'].chmod(0o700);guard.close()


def test_N3_original_parent_symlink_refuses_without_technical_adoption(tmp_path):
    import qcl_negf_results.capture as cap
    f=_N3_fixture(tmp_path);parent=f['result'].parent;saved=parent.with_name('original-native-parent-owned')
    parent.rename(saved);parent.symlink_to(saved,target_is_directory=True)
    try:
        assert _N3_observe(f)==2
        assert not (f['result']/'performance').exists()
    finally:parent.unlink();saved.rename(parent)


@pytest.mark.parametrize('window', ['constructor', 'close', 'terminal', 'flush'])
def test_N3_observer_late_root_change_never_acknowledges_or_repairs(tmp_path, monkeypatch, window):
    import qcl_negf_results.capture as cap
    from qcl_negf_results.telemetry import TelemetryWriter
    f=_N3_fixture(tmp_path);saved=f['result'].with_name('born-root-moved-owned');changed=[];ledgers=[]
    before=len(list(Path('/proc/self/fd').iterdir()))
    def swap():
        if not changed:f['result'].rename(saved);f['result'].mkdir(mode=0o700);changed.append(True)
    if window=='constructor':
        original=TelemetryWriter.__init__
        def construct(self,*a,**k):original(self,*a,**k);swap()
        monkeypatch.setattr(TelemetryWriter,'__init__',construct)
    elif window=='close':
        original=TelemetryWriter.close_context
        def close(self,*a,**k):result=original(self,*a,**k);swap();return result
        monkeypatch.setattr(TelemetryWriter,'close_context',close)
    elif window=='flush':
        original=TelemetryWriter.flush
        def flush(self,*a,**k):
            result=original(self,*a,**k);ledgers.append(self.metadata_budget);swap();return result
        monkeypatch.setattr(TelemetryWriter,'flush',flush)
    else:
        original=cap._capture_write
        def write(root,path,*a,**k):
            result=original(root,path,*a,**k)
            if path=='performance-source.json':swap()
            return result
        monkeypatch.setattr(cap,'_capture_write',write)
    assert _N3_observe(f)==2 and changed
    assert list(f['result'].iterdir())==[]
    assert (saved/'performance').exists()
    if window=='terminal':assert (saved/'performance-source.json').exists()
    if window=='flush':
        actual=sum(p.stat().st_size for p in (saved/'performance').rglob('*.parquet'))
        assert actual>0 and ledgers[0].forecast_bytes>=4*actual
        assert ledgers[0].reserved==0
    assert len(list(Path('/proc/self/fd').iterdir()))==before


def test_T3_typed_anchor_requires_original_immutable_copy(tmp_path):
    import qcl_negf_results.capture as module
    from qcl_negf_results._capture_io import ObserverIO
    from qcl_negf_contracts.messages import ContractError
    fixture = _T1_native_job(tmp_path)
    value, raw, immutable = _T3_anchor(fixture, pending=True)
    observed, ref, original = module._performance_anchor(fixture['result'], budget=ObserverIO(4 << 20))
    assert observed == value and original == raw
    assert ref['path'] == 'performance-anchors/' + __import__('hashlib').sha256(raw).hexdigest() + '.json'
    immutable.write_bytes(raw + b' ')
    with pytest.raises((ContractError, ValueError)):
        module._performance_anchor(fixture['result'], budget=ObserverIO(4 << 20))


def test_T3_native_phase0_keeps_original_rows_without_fabricating_science(tmp_path, monkeypatch):
    import qcl_negf_results.capture as module
    fixture = _T1_native_job(tmp_path)
    _T3_anchor(fixture, pending=True)
    (fixture['result'] / 'series_result.json').unlink()
    monkeypatch.setattr(module, 'proc_sample', lambda *a, **k: dict(wall_seconds=2., monotonic_ns=2_000_000_000,
        cpu_time_seconds=None, missing_reason='synthetic process sample unavailable'))
    read_fd, write_fd = os.pipe2(os.O_NONBLOCK | os.O_CLOEXEC)
    os.write(write_fd, b'exit 7\n'); os.close(write_fd)
    assert module.observe(fixture['job'] / 'capture-spec.json', 42, 7, read_fd,
        native_execution_path=fixture['job'] / 'native-execution.json', solver_arguments=fixture['arguments']) == 2
    technical = fixture['job'] / 'performance'
    pointer = next(technical.glob('sessions/*/current.json'))
    commit = json.loads((technical / json.loads(pointer.read_bytes())['commit']['path']).read_bytes())
    summary = json.loads((technical / commit['summary']['path']).read_bytes())
    assert summary['totals']['accepted_source_rows'] == '2'
    assert summary['totals']['accepted_aggregate_calls'] == '3'
    assert commit['source_cursors'][0]['identity'] == fixture['context']['state']['identity']
    assert not (technical / 'performance-binding.json').exists()
    assert not (fixture['job'] / 'performance-source.json').exists()


def _T3_capture(tmp_path, *, history_only=False):
    from copy import deepcopy
    import qcl_negf_results.capture as module
    from qcl_negf_results.telemetry import TelemetryWriter
    from qcl_negf_results._capture_io import ObserverIO
    fixture = _T1_native_job(tmp_path)
    if history_only:
        from native_consumer_fixture import technical_source, sealed_phase
        c = technical_source(tmp_path / 'history', checkpoint=False, execution='exec')
        fixture['context'] = c
        fixture['manifest'] = c['science'] / c['state']['manifest']['path']
        fixture['result'] = c['root']
        sealed_phase(c['root'], c['state']['identity'])
        plan_raw = (c['root'] / c['plan']['path']).read_bytes()
        (fixture['job'] / 'plan.json').write_bytes(plan_raw)
        fixture['spec']['selection']['plan_fingerprint'] = json.loads(plan_raw)['fingerprint']
    raw = (fixture['job'] / 'plan.json').read_bytes()
    plan_ref = dict(path='plan.json', bytes=str(len(raw)), sha256=__import__('hashlib').sha256(raw).hexdigest(),
        schema=json.loads(raw)['schema'], media_type='application/json')
    transport = dict(selection=deepcopy(fixture['spec']['selection']), frozen_plan=plan_ref)
    storage = fixture['native']['storage']
    root = lambda area, path: dict(storage=deepcopy(storage), area=area, relative_path=path)
    uid = lambda n: f'00000000-0000-4000-8000-{n:012d}'
    pins = {key: dict(node_uuid=uid(n), bytes='1', sha256='1' * 64)
        for n, key in enumerate(('plan', 'source_graph', 'native_execution', 'submit_script'), 10)}
    inputs = module.NativePerformanceInputs(provider_selection=dict(run_uuid=uid(9),
        **{k: transport['selection'][k] for k in ('execution_id', 'attempt', 'calcjob_uuid')}),
        native_execution_uuid=pins['native_execution']['node_uuid'], runner_anchor_root=root('native_execution', '.'),
        technical_root=root('native_workdir', 'performance'),
        sources=[dict(selected_state=fixture['context']['state'], artifact_root='science',
            scientific_root=root('native_execution', 'science'), scientific_directory=str(fixture['context']['science']),
            canonical_plan_root=root('native_workdir', '.'),
            canonical_plan_directory=str(fixture['job']), source_transport=transport)])
    budget = ObserverIO(128 << 20)
    writer = TelemetryWriter(fixture['job'] / 'performance', {}, {}, transport=transport, technical_version=2, metadata_budget=budget)
    capture = module.NativePerformanceCapture(fixture['job'], fixture['result'], writer=writer,
        transport=transport, inputs=inputs, budget=budget)
    fixture.update(writer=writer, capture=capture, inputs=inputs, transport=transport)
    return fixture


def test_T3_history_only_factory_publishes_open_then_exact_closed_index(tmp_path):
    from qcl_negf_results.performance_views import TelemetrySelection, PerformanceLimits, read_performance_summary
    from qcl_negf_results.state import StateReader
    from qcl_negf_contracts.telemetry import validate_performance_source_index
    fixture = _T3_capture(tmp_path, history_only=True)
    anchor, raw, immutable = _T3_anchor(fixture)
    capture = fixture['capture']; writer = fixture['writer']
    capture.remember(anchor, dict(path=str(immutable.relative_to(fixture['result'])), bytes=str(len(raw)),
        sha256=__import__('hashlib').sha256(raw).hexdigest(), schema=anchor['schema'], media_type='application/json'), raw)
    first = capture.publish()
    opened = json.loads((writer.root / first['path']).read_bytes())
    assert opened['closed'] is False and len(opened['entries']) == 1
    assert not writer.closed
    capture.remember(anchor, opened['entries'][0]['runner_anchor_ref'], raw)
    assert len(capture.anchors) == 1
    writer.record_resource(dict(wall_seconds=1.))
    writer.flush()
    close = writer.close_context()
    exact = capture.publish(close_context=close, capture_status=dict(terminal_tail='unknown', scalar_live_route='unavailable',
        scalar_live_reason='external_root', performance_status='incomplete', reason='tail_unknown'))
    closed_raw = (writer.root / exact['path']).read_bytes()
    index = validate_performance_source_index(json.loads(closed_raw))
    assert index['closed'] is True and index['generation'] == '2'
    assert (fixture['job'] / 'performance-source.json').read_bytes() == closed_raw
    assert (writer.root / first['path']).read_bytes() == __import__('qcl_negf_results.commits', fromlist=['json_bytes']).json_bytes(opened)
    row = index['entries'][0]
    assert row['telemetry']['commit_ref'] == close[0]
    selection = TelemetrySelection(writer.root, canonical_plan_root=fixture['job'], **row['telemetry'])
    with StateReader(fixture['context']['science'], state_ref=fixture['context']['state'],
            expected_identity=fixture['context']['state']['identity']) as reader:
        view = read_performance_summary(reader, telemetry=selection, limits=PerformanceLimits(maximum_objects=16384))
        assert view['status'] == 'available' and view['closed'] is True
        assert reader._manifest['checkpoint'] is None
        assert view['transport']['selection']['attempt'] == 3


def test_T3_native_explicit_operands_use_owned_factory_and_v2_index(tmp_path, monkeypatch):
    import qcl_negf_results.capture as module
    fixture = _T3_capture(tmp_path)
    # The factory fixture's private writer is quiesced; the real observe owns a
    # separate fresh workdir rather than reusing a live writer directory.
    fixture['writer'].close()
    import shutil
    shutil.rmtree(fixture['job'] / 'performance')
    _T3_anchor(fixture)
    monkeypatch.setattr(module, 'proc_sample', lambda *a, **k: dict(wall_seconds=2., monotonic_ns=2_000_000_000,
        cpu_time_seconds=None, missing_reason='synthetic process sample unavailable'))
    read_fd, write_fd = os.pipe2(os.O_NONBLOCK | os.O_CLOEXEC)
    os.write(write_fd, b'exit 0\n'); os.close(write_fd)
    assert module.observe(fixture['job'] / 'capture-spec.json', 42, 7, read_fd,
        native_execution_path=fixture['job'] / 'native-execution.json', solver_arguments=fixture['arguments'],
        native_performance_inputs=fixture['inputs']) == 0
    index = json.loads((fixture['job'] / 'performance-source.json').read_bytes())
    assert index['closed'] is True and len(index['entries']) == 1
    technical = fixture['job'] / 'performance'; row = index['entries'][0]
    commit = json.loads((technical / row['telemetry']['commit_ref']['path']).read_bytes())
    summary = json.loads((technical / commit['summary']['path']).read_bytes())
    assert commit['schema'] == 'qcl-negf.telemetry-commit.v2'
    assert summary['totals']['accepted_source_rows'] == '2'
    assert summary['totals']['accepted_aggregate_calls'] == '3'
    assert not any(r['event'] == 'source_binding_missing' for r in summary['control_events'])
    assert row['telemetry']['binding_ref'] is not None
    retained = [p.read_bytes() for p in technical.glob('sessions/*/raw/*')]
    assert (fixture['result'] / 'phase-segment-1.json').read_bytes() in retained


def test_T3_phase0_to_scientific_anchor_keeps_one_cursor_and_unknown_capacity(tmp_path, monkeypatch):
    import shutil
    import qcl_negf_results.capture as module
    fixture = _T3_capture(tmp_path)
    fixture['writer'].close(); shutil.rmtree(fixture['job'] / 'performance')
    _T3_anchor(fixture, pending=True)
    read_fd, write_fd = os.pipe2(os.O_NONBLOCK | os.O_CLOEXEC)
    samples = iter(((0, 1.), (2_000_000_000, 4.)))
    def sample(*args, **kwargs):
        stamp, cpu = next(samples)
        return dict(wall_seconds=stamp / 1e9, monotonic_ns=stamp, cpu_time_seconds=cpu,
            pid=42, process_start_ticks=7, clock_domain_id='root-clock', cgroup_scope='root_julia_process_including_threads')
    snapshots = []
    def ready(_):
        pointer = json.loads((fixture['job'] / 'performance' / 'current-source.json').read_bytes())
        snapshot = json.loads((fixture['job'] / 'performance' / pointer['index']['path']).read_bytes())
        snapshots.append(snapshot)
        assert snapshot['closed'] is False and snapshot['entries'] == []
        _T3_anchor(fixture)
        os.write(write_fd, b'exit 0\n')
    monkeypatch.setattr(module, 'proc_sample', sample)
    monkeypatch.setattr(module.time, 'sleep', ready)
    try:
        assert module.observe(fixture['job'] / 'capture-spec.json', 42, 7, read_fd,
            native_execution_path=fixture['job'] / 'native-execution.json', solver_arguments=fixture['arguments'],
            native_performance_inputs=fixture['inputs']) == 0
    finally:
        os.close(write_fd)
    index = json.loads((fixture['job'] / 'performance-source.json').read_bytes())
    technical = fixture['job'] / 'performance'
    commit = json.loads((technical / index['technical']['commit_ref']['path']).read_bytes())
    summary = json.loads((technical / commit['summary']['path']).read_bytes())
    assert len(snapshots) == 1 and len(index['entries']) == 1
    assert summary['totals']['accepted_source_rows'] == '2'
    assert summary['totals']['accepted_aggregate_calls'] == '3'
    assert len(commit['source_cursors']) == 1
    cpu = summary['cpu_coverage']['allocations'][0]
    assert cpu['cpu_seconds'] == 3. and cpu['allocated_cpu_seconds'] is None
    assert cpu['fraction_of_observed_capacity'] is None


@pytest.mark.parametrize('fault', ['root_traversal', 'state_type', 'plan_alias'])
def test_T3_input_roots_and_states_remain_explicit_typed_operands(tmp_path, fault):
    import qcl_negf_results.capture as module
    from qcl_negf_contracts.messages import ContractError
    fixture = _T3_capture(tmp_path)
    values = fixture['inputs'].document()
    if fault == 'root_traversal':values['sources'][0]['canonical_plan_root']['relative_path'] = '../other'
    elif fault == 'state_type':values['sources'][0]['selected_state'] = {'manifest': None}
    else:values['sources'][0]['canonical_plan_directory'] = str(fixture['job']) + '/../other'
    with pytest.raises((ValueError, ContractError)):
        module.NativePerformanceInputs(**values)
    fixture['writer'].close()


def test_T3_produced_root_operands_need_no_future_StateRef(tmp_path):
    import qcl_negf_results.capture as module
    fixture = _T3_capture(tmp_path, history_only=True)
    values = fixture['inputs'].document()
    values['sources'][0]['selected_state'] = None
    inputs = module.NativePerformanceInputs(**values)
    capture = module.NativePerformanceCapture(fixture['job'], fixture['result'], writer=fixture['writer'],
        transport=fixture['transport'], inputs=inputs, budget=fixture['capture'].budget)
    anchor, raw, immutable = _T3_anchor(fixture)
    capture.remember(anchor, module._capture_ref(str(immutable.relative_to(fixture['result'])), raw, anchor['schema']), raw)
    pin = capture.publish()
    assert len(json.loads((fixture['writer'].root / pin['path']).read_bytes())['entries']) == 1
    fixture['writer'].close()


def test_T3_distinct_qualified_roots_keep_distinct_selectors_and_borrowed_cost(tmp_path):
    from copy import deepcopy
    import qcl_negf_results.capture as module
    fixture = _T3_capture(tmp_path, history_only=True)
    values = fixture['inputs'].document()
    original = tmp_path / 'original-plan-domain'; original.mkdir()
    original_raw = (fixture['job'] / 'plan.json').read_bytes()
    (original / 'frozen.json').write_bytes(original_raw)
    source = values['sources'][0]
    source['source_transport'] = deepcopy(fixture['transport'])
    source['source_transport']['selection'].update(calcjob_uuid='00000000-0000-4000-8000-000000000008', attempt=2)
    source['source_transport']['frozen_plan']['path'] = 'frozen.json'
    source['canonical_plan_directory'] = str(original)
    source['canonical_plan_root']['storage']['execution_binding_uuid'] = '00000000-0000-4000-8000-000000000014'
    source['scientific_root']['storage']['execution_binding_uuid'] = '00000000-0000-4000-8000-000000000015'
    # One exact state in two explicitly supplied domains remains two anchors.
    import shutil
    shutil.copytree(fixture['context']['science'], fixture['result'] / 'other-science')
    other = deepcopy(source); other['artifact_root'] = 'other-science'; other['scientific_root']['relative_path'] = 'other-science'
    other['scientific_directory'] = str(fixture['result'] / 'other-science')
    values['sources'].append(other)
    inputs = module.NativePerformanceInputs(**values)
    capture = module.NativePerformanceCapture(fixture['job'], fixture['result'], writer=fixture['writer'],
        transport=fixture['transport'], inputs=inputs, budget=fixture['capture'].budget)
    anchor, _, _ = _T3_anchor(fixture)
    anchor['scientific']['relation'] = 'borrowed'
    anchor['phase_identity'] = dict(anchor['phase_identity'], attempt='9')
    anchor['invocation'] = dict(source_epoch='current-technical-epoch')
    for artifact_root in ('science', 'other-science'):
        anchor['scientific']['artifact_root'] = artifact_root
        raw = module.json_bytes(anchor)
        path = 'performance-anchors/' + __import__('hashlib').sha256(raw).hexdigest() + '.json'
        (fixture['result'] / path).write_bytes(raw)
        capture.remember(anchor, module._capture_ref(path, raw, anchor['schema']), raw)
    pin = capture.publish()
    index = json.loads((fixture['writer'].root / pin['path']).read_bytes())
    assert len(index['entries']) == 2
    assert len({entry['anchor_key'] for entry in index['entries']}) == 2
    assert all(entry['telemetry']['expected_transport']['selection']['attempt'] == 3 for entry in index['entries'])
    assert all(entry['telemetry']['source_transport']['selection']['attempt'] == 2 for entry in index['entries'])
    assert all(json.loads(entry['runner_anchor_utf8'])['phase_identity']['attempt'] == '9' for entry in index['entries'])
    assert index['entries'][0]['telemetry']['commit_ref'] == index['entries'][1]['telemetry']['commit_ref']
    fixture['writer'].close()


def test_T3_scientific_mapping_is_read_independently_of_runner_anchor_root(tmp_path):
    import shutil
    import qcl_negf_results.capture as module
    fixture = _T3_capture(tmp_path, history_only=True)
    anchor, raw, immutable = _T3_anchor(fixture)
    original = tmp_path / 'independently-qualified-science'
    shutil.copytree(fixture['context']['science'], original)
    values = fixture['inputs'].document()
    values['sources'][0]['scientific_directory'] = str(original)
    inputs = module.NativePerformanceInputs(**values)
    capture = module.NativePerformanceCapture(fixture['job'], fixture['result'], writer=fixture['writer'],
        transport=fixture['transport'], inputs=inputs, budget=fixture['capture'].budget)
    # The enclosing Runner copy is a decoy; the original scientific mapping is
    # supplied independently and must be the one consumed by StateReader.
    fixture['manifest'].write_bytes(b'{"decoy":true}')
    capture.remember(anchor, module._capture_ref(str(immutable.relative_to(fixture['result'])), raw, anchor['schema']), raw)
    pin = capture.publish()
    assert len(json.loads((fixture['writer'].root / pin['path']).read_bytes())['entries']) == 1
    fixture['writer'].close()


def test_T3_pending_terminal_retains_technical_index_without_scientific_state(tmp_path, monkeypatch):
    import shutil
    import qcl_negf_results.capture as module
    fixture = _T3_capture(tmp_path)
    fixture['writer'].close(); shutil.rmtree(fixture['job'] / 'performance')
    _T3_anchor(fixture, pending=True)
    monkeypatch.setattr(module, 'proc_sample', lambda *a, **k: dict(wall_seconds=2., monotonic_ns=2_000_000_000,
        cpu_time_seconds=None, missing_reason='synthetic process sample unavailable'))
    read_fd, write_fd = os.pipe2(os.O_NONBLOCK | os.O_CLOEXEC)
    os.write(write_fd, b'exit 7\n'); os.close(write_fd)
    assert module.observe(fixture['job'] / 'capture-spec.json', 42, 7, read_fd,
        native_execution_path=fixture['job'] / 'native-execution.json', solver_arguments=fixture['arguments'],
        native_performance_inputs=fixture['inputs']) == 2
    index = json.loads((fixture['job'] / 'performance-source.json').read_bytes())
    assert index['closed'] is True and index['entries'] == []
    root = fixture['job'] / 'performance'
    commit = json.loads((root / index['technical']['commit_ref']['path']).read_bytes())
    summary = json.loads((root / commit['summary']['path']).read_bytes())
    assert summary['totals']['accepted_source_rows'] == '2'
    assert summary['control_events'] == [dict(event='scientific_anchor_pending', count='1')]
    assert not (root / 'bindings').exists()


def test_T3_unacknowledged_writer_tail_cannot_publish_source_index(tmp_path, monkeypatch):
    import qcl_negf_results.telemetry as telemetry
    fixture = _T3_capture(tmp_path, history_only=True)
    writer = fixture['writer']; writer.record_resource(dict(wall_seconds=1.))
    original = telemetry.publish_pointer
    def lost_ack(*args, **kwargs):
        original(*args, **kwargs)
        raise OSError('original technical ACK unavailable')
    with monkeypatch.context() as patch:
        patch.setattr(telemetry, 'publish_pointer', lost_ack)
        with pytest.raises(OSError, match='ACK unavailable'):writer.flush()
    with pytest.raises(ValueError, match='pending'):fixture['capture'].publish()
    assert not (writer.root / 'current-source.json').exists()
    assert not (fixture['job'] / 'performance-source.json').exists()
    writer.flush(); writer.close()


def test_T3_source_change_before_pointer_keeps_previous_exact_index(tmp_path, monkeypatch):
    import qcl_negf_results.capture as module
    fixture = _T3_capture(tmp_path, history_only=True)
    capture = fixture['capture']; writer = fixture['writer']
    anchor, raw, immutable = _T3_anchor(fixture)
    ref = module._capture_ref(str(immutable.relative_to(fixture['result'])), raw, anchor['schema'])
    capture.remember(anchor, ref, raw); first = capture.publish()
    pointer = (writer.root / 'current-source.json').read_bytes()
    original = module._capture_read; changed = []
    def late_change(root, reference, budget):
        result = original(root, reference, budget)
        if root == fixture['result'] and reference == ref and not changed:
            changed.append(True); fixture['manifest'].write_bytes(b'{"changed":true}')
        return result
    with monkeypatch.context() as patch:
        patch.setattr(module, '_capture_read', late_change)
        with pytest.raises(__import__('qcl_negf_contracts.messages', fromlist=['ContractError']).ContractError):capture.publish()
    assert changed
    assert (writer.root / 'current-source.json').read_bytes() == pointer
    assert json.loads(pointer)['index'] == first
    assert capture.generation == 1
    writer.close()


@pytest.mark.parametrize('window', ['index', 'pointer', 'terminal'])
@pytest.mark.parametrize('role', ['manifest', 'technical_root'])
def test_F5_publication_suffix_change_is_not_acknowledged(tmp_path, monkeypatch, window, role):
    import qcl_negf_results.capture as cap
    fixture = _T3_capture(tmp_path, history_only=True)
    owner, writer = fixture['capture'], fixture['writer']
    anchor, raw, original = _T3_anchor(fixture)
    owner.remember(anchor, cap._capture_ref(str(original.relative_to(fixture['result'])), raw, anchor['schema']), raw)
    first = owner.publish()
    previous = (writer.root / 'current-source.json').read_bytes()
    options = dict(close_context=writer.close_context(), capture_status=dict(terminal_tail='unknown',
        scalar_live_route='unavailable', scalar_live_reason='source_fixture',
        performance_status='incomplete', reason='tail_unknown')) if window == 'terminal' else {}
    before = fixture['manifest'].read_bytes()
    saved = writer.root.with_name('performance-original-owned-root')
    replacement = writer.root.with_name('performance-replacement-owned-root')
    effects = []; changed = []
    write = cap._capture_write
    def changed_after_readback(root, path, raw, schema, budget, **kwargs):
        result = write(root, path, raw, schema, budget, **kwargs)
        if path.startswith('sources/'):effects.append('index')
        elif path == 'current-source.json':effects.append('pointer')
        elif path == 'performance-source.json':effects.append('terminal')
        if effects and effects[-1] == window and not changed:
            changed.append(True)
            if role == 'manifest':fixture['manifest'].write_bytes(b'{"changed":true}')
            else:writer.root.rename(saved);writer.root.mkdir()
        return result
    try:
        with monkeypatch.context() as patch:
            patch.setattr(cap, '_capture_write', changed_after_readback)
            with pytest.raises((OSError, ValueError)):
                owner.publish(**options)
        assert changed and owner.generation == 1
        # Effects already visible are retained; no rollback or success is invented.
        original_root = saved if saved.exists() else writer.root
        pointer = (original_root / 'current-source.json').read_bytes()
        if window == 'index':assert pointer == previous and effects == ['index']
        else:
            assert pointer != previous
            assert (original_root / json.loads(pointer)['index']['path']).exists()
        assert (original_root / first['path']).exists()
        if window == 'terminal':assert (fixture['job'] / 'performance-source.json').exists()
    finally:
        fixture['manifest'].write_bytes(before)
        if saved.exists():writer.root.rename(replacement);saved.rename(writer.root)
        writer.close()


@pytest.mark.parametrize('role', ['anchor', 'plan', 'commit', 'scientific_root', 'plan_root'])
def test_F5_original_source_roles_are_held_through_pointer_readback(tmp_path, monkeypatch, role):
    import shutil
    import qcl_negf_results.capture as cap
    fixture = _T3_capture(tmp_path, history_only=True)
    owner, writer = fixture['capture'], fixture['writer']
    anchor, raw, original = _T3_anchor(fixture)
    owner.remember(anchor, cap._capture_ref(str(original.relative_to(fixture['result'])), raw, anchor['schema']), raw)
    context = writer.published_context()
    target = dict(anchor=original, plan=fixture['job'] / fixture['transport']['frozen_plan']['path'],
        commit=writer.root / context[0]['path'], scientific_root=fixture['context']['science'],
        plan_root=fixture['job'])[role]
    directory = role.endswith('_root')
    saved = target.with_name(target.name + '-original-owned')
    replacement = target.with_name(target.name + '-replacement-owned')
    before = None if directory else target.read_bytes()
    changed = []
    write = cap._capture_write
    def changed_after_pointer(root, path, raw, schema, budget, **kwargs):
        result = write(root, path, raw, schema, budget, **kwargs)
        if path == 'current-source.json' and not changed:
            changed.append(True)
            if directory:target.rename(saved);shutil.copytree(saved, target)
            else:target.write_bytes(b'{"changed":true}')
        return result
    try:
        with monkeypatch.context() as patch:
            patch.setattr(cap, '_capture_write', changed_after_pointer)
            with pytest.raises((OSError, ValueError)):owner.publish()
        assert changed and owner.generation == 0
        assert (writer.root / 'current-source.json').exists()
    finally:
        if directory and saved.exists():target.rename(replacement);saved.rename(target)
        elif before is not None:target.write_bytes(before)
        writer.close()


@pytest.mark.parametrize('cancel', [False, True])
def test_F5_publication_primary_error_retains_secondary_guard_diagnostic(tmp_path, monkeypatch, cancel):
    import qcl_negf_results.capture as cap
    fixture = _T3_capture(tmp_path, history_only=True)
    owner, writer = fixture['capture'], fixture['writer']
    anchor, raw, original = _T3_anchor(fixture)
    owner.remember(anchor, cap._capture_ref(str(original.relative_to(fixture['result'])), raw, anchor['schema']), raw)
    before = fixture['manifest'].read_bytes()
    primary = KeyboardInterrupt('publication cancelled') if cancel else OSError('publication primary failure')
    write = cap._capture_write
    def fail_after_pointer(root, path, raw, schema, budget, **kwargs):
        result = write(root, path, raw, schema, budget, **kwargs)
        if path == 'current-source.json':
            fixture['manifest'].write_bytes(b'{"changed":true}')
            raise primary
        return result
    try:
        with monkeypatch.context() as patch:
            patch.setattr(cap, '_capture_write', fail_after_pointer)
            with pytest.raises(type(primary)) as caught:owner.publish()
        assert caught.value is primary and owner.generation == 0
        assert any('publication source cleanup' in note for note in primary.__notes__)
        assert (writer.root / 'current-source.json').exists()
    finally:fixture['manifest'].write_bytes(before);writer.close()


def test_T3_phase_failure_retains_last_ack_with_explicit_incomplete_close(tmp_path, monkeypatch):
    import shutil
    import qcl_negf_results.capture as module
    fixture = _T3_capture(tmp_path)
    fixture['writer'].close(); shutil.rmtree(fixture['job'] / 'performance')
    _T3_anchor(fixture)
    before = fixture['manifest'].read_bytes()
    read_fd, write_fd = os.pipe2(os.O_NONBLOCK | os.O_CLOEXEC)
    monkeypatch.setattr(module, 'proc_sample', lambda *a, **k: dict(wall_seconds=2., monotonic_ns=2_000_000_000,
        cpu_time_seconds=None, missing_reason='synthetic process sample unavailable'))
    def fail_tail(_):
        (fixture['result'] / 'phase-current.json').write_bytes(b'{')
        os.write(write_fd, b'exit 7\n')
    monkeypatch.setattr(module.time, 'sleep', fail_tail)
    try:
        assert module.observe(fixture['job'] / 'capture-spec.json', 42, 7, read_fd,
            native_execution_path=fixture['job'] / 'native-execution.json', solver_arguments=fixture['arguments'],
            native_performance_inputs=fixture['inputs']) == 2
    finally:
        os.close(write_fd)
    index = json.loads((fixture['job'] / 'performance-source.json').read_bytes())
    row = index['entries'][0]; technical = fixture['job'] / 'performance'
    binding = json.loads((technical / row['telemetry']['binding_ref']['path']).read_bytes())
    assert index['closed'] is True
    assert binding['capture_status']['performance_status'] == 'incomplete'
    assert binding['capture_status']['terminal_tail'] == 'known_incomplete'
    commit = json.loads((technical / index['technical']['commit_ref']['path']).read_bytes())
    summary = json.loads((technical / commit['summary']['path']).read_bytes())
    assert summary['totals']['accepted_source_rows'] == '2'
    assert any(event['event'] == 'capture_failed' for event in summary['control_events'])
    assert fixture['manifest'].read_bytes() == before


def test_F2_post_replace_readback_error_leaves_uncertain_valid_pointer(tmp_path, monkeypatch):
    import qcl_negf_results.capture as module
    from qcl_negf_contracts.telemetry import validate_performance_source_index
    fixture = _T3_capture(tmp_path, history_only=True)
    anchor, raw, immutable = _T3_anchor(fixture)
    capture = fixture['capture']; writer = fixture['writer']
    capture.remember(anchor, module._capture_ref(str(immutable.relative_to(fixture['result'])), raw, anchor['schema']), raw)
    first = capture.publish(); old = (writer.root / first['path']).read_bytes()
    pointer_before = (writer.root / 'current-source.json').read_bytes()
    writer.record_resource(dict(wall_seconds=1.)); writer.flush()
    read = module._capture_read
    primary = OSError('F2 readback unavailable after durable pointer replacement')
    def unavailable(root, ref, budget):
        answer = read(root, ref, budget)
        if ref['schema'] == 'qcl-negf.performance-source-pointer.v1':raise primary
        return answer
    with monkeypatch.context() as patch:
        patch.setattr(module, '_capture_read', unavailable)
        with pytest.raises(OSError) as failure:capture.publish()
        assert failure.value is primary
    pointer_after = (writer.root / 'current-source.json').read_bytes()
    assert pointer_after != pointer_before
    selected = json.loads(pointer_after)['index']
    original = (writer.root / selected['path']).read_bytes()
    assert len(original) == int(selected['bytes'])
    assert __import__('hashlib').sha256(original).hexdigest() == selected['sha256']
    assert validate_performance_source_index(json.loads(original))['generation'] == '2'
    assert capture.generation == 1  # Failed acknowledgement is not success.
    assert (writer.root / first['path']).read_bytes() == old
    writer.close()


@pytest.mark.parametrize('stage',['inputs_decode','index_encode'])
def test_F3_materialization_failure_keeps_primary_old_pin_and_actual_spent(tmp_path,monkeypatch,stage):
    import qcl_negf_results.capture as cap
    from test_capture_credit import _F1_remembered,_F1_actual_delivery
    f=_F1_remembered(tmp_path);owner=f['capture'];writer=f['writer']
    first=owner.publish();pointer=(writer.root/'current-source.json').read_bytes()
    original=(writer.root/first['path']).read_bytes();inputs=owner.inputs._raw
    primary=MemoryError('F3 synthetic materialization failure after admission')
    if stage=='inputs_decode':
        document=cap.NativePerformanceInputs.document
        def decode(value):
            if value is owner.inputs:raise primary
            return document(value)
        monkeypatch.setattr(cap.NativePerformanceInputs,'document',decode)
    else:
        serializer=cap.json_bytes
        def encode(value):
            if isinstance(value,dict) and value.get('schema')=='qcl-negf.performance-source-index.v1':raise primary
            return serializer(value)
        monkeypatch.setattr(cap,'json_bytes',encode)
    trace,error=_F1_actual_delivery(f,monkeypatch)
    assert error is primary and owner.generation==1
    assert trace['reserved']==0 and not trace['before_effect_overshoot']
    assert trace['ledger_delta']==trace['reads']+trace['writes']
    assert (writer.root/'current-source.json').read_bytes()==pointer
    assert (writer.root/first['path']).read_bytes()==original and owner.inputs._raw==inputs
    assert len(list((writer.root/'sources').glob('*.json')))==1
    writer.close()


@pytest.mark.parametrize('observer_mode,want', [('terminal', 0), ('failed', 125),
    ('early', 125), ('stalled', 125), ('spawn_error', 125), ('missing', 125)])
def test_fixture_supervisor_requires_terminal_observer_reaped_zero(tmp_path, observer_mode, want):
    """Losing the explicit fixture observer must never turn child0 into success."""
    marker = tmp_path / 'solver-spawns'
    solver = [sys.executable, '-c',
        'import pathlib,sys,time; p=pathlib.Path(sys.argv[1]); '
        'p.open("a").write("spawn\\n"); time.sleep(.15)', str(marker)]

    def observer(pid, start, fd):
        if observer_mode == 'spawn_error':raise OSError('observer spawn denied')
        code = {
            'terminal': 'import os,sys,time; fd=int(sys.argv[1]); data=b""; '
                '\nwhile not data.endswith(b"\\n"):\n try: data+=os.read(fd,64)'
                '\n except BlockingIOError: time.sleep(.005)'
                '\nraise SystemExit(0 if data==b"exit 0\\n" else 2)',
            'failed': 'raise SystemExit(2)',
            'early': 'raise SystemExit(0)',
            'stalled': 'import time; time.sleep(30)',
        }[observer_mode]
        return [sys.executable, '-c', code, str(fd)]

    assert supervise(solver, observer_factory=None if observer_mode == 'missing' else observer,
        drain_seconds=.1, require_observer_success=True) == want
    assert marker.read_text().splitlines() == ['spawn']


@pytest.mark.parametrize('child_code,want', [(7, 7), (-15, 143)])
def test_fixture_supervisor_preserves_child_failure_once(tmp_path, child_code, want):
    marker=tmp_path/'solver-spawns'
    script=('import os,pathlib,signal,sys; pathlib.Path(sys.argv[1]).open("a").write("spawn\\n"); '
        + ('os.kill(os.getpid(),signal.SIGTERM)' if child_code<0 else 'raise SystemExit(7)'))
    assert supervise([sys.executable,'-c',script,str(marker)],
        observer_factory=lambda *a: (_ for _ in ()).throw(OSError('observer unavailable')),
        drain_seconds=0,require_observer_success=True)==want
    assert marker.read_text().splitlines()==['spawn']


def test_capture_run_selects_strict_fixture_mode_before_remote_read(tmp_path, monkeypatch):
    import qcl_negf_results.capture as cap
    observed=[]
    def run(argv,**kwargs):
        observed.append((argv,kwargs.get('require_observer_success',False)))
        return 125 if kwargs.get('require_observer_success') else 0
    monkeypatch.setattr(cap,'supervise',run)
    # These paths deliberately do not exist: the reaper must not read them.
    spec=tmp_path/'remote'/'capture-spec.json'
    assert cap.capture_run(spec,'/actual/solver',('native-publication-fixture','plan.json'))==125
    assert cap.capture_run(spec,'/actual/solver',('run-plan','plan.json'))==0
    assert observed==[(['/actual/solver','native-publication-fixture','plan.json'],True),
        (['/actual/solver','run-plan','plan.json'],False)]



def _publication_birth_fixture(tmp_path):
    """Reader L fixture only: local native3 identities, no original creator authority."""
    f=_N3_fixture(tmp_path)
    (f['result']/'performance-anchor-current.json').unlink()
    anchors=f['result']/'performance-anchors'
    for path in anchors.iterdir():path.unlink()
    anchors.rmdir()
    return f


def test_publication_fixture_birth_observer_closes_without_technical_effects(tmp_path, monkeypatch):
    import qcl_negf_results.capture as cap
    from qcl_negf_results._capture_io import ObserverIO
    f=_publication_birth_fixture(tmp_path);budget=ObserverIO(128<<20)
    for owner,name in ((cap,'ScalarPublisher'),(cap,'NativePerformanceInputs'),
        (cap._NativeResultGuard,'create_technical')):
        monkeypatch.setattr(owner,name,lambda *a,**k:pytest.fail('fixture created technical capture'))
    monkeypatch.setattr('qcl_negf_results.telemetry.TelemetryWriter',
        lambda *a,**k:pytest.fail('fixture created telemetry writer'))
    before={str(p.relative_to(f['result'])):p.read_bytes() for p in f['result'].rglob('*') if p.is_file()}
    descriptors=len(os.listdir('/proc/self/fd'))
    read_fd,write_fd=os.pipe2(os.O_NONBLOCK|os.O_CLOEXEC)
    os.write(write_fd,b'exit 0\n');os.close(write_fd)
    cap._observe_native_publication_fixture(f['job'],f['spec'],f['result'],os.getpid(),
        cap._start_ticks(os.getpid()),read_fd,budget)
    assert {str(p.relative_to(f['result'])):p.read_bytes() for p in f['result'].rglob('*') if p.is_file()}==before
    assert budget.read_bytes>0 and budget.reserved==0 and budget.write_bytes==0
    assert len(os.listdir('/proc/self/fd'))==descriptors
    assert not(f['job']/'live').exists() and not(f['job']/'performance').exists()


@pytest.mark.parametrize('terminal',[b'',b'exit 7\n',b'exit 0',b'exit 0\nexit 0\n',b'unknown\n',b'x'*65])
def test_publication_birth_observer_requires_exact_successful_terminal(tmp_path, terminal):
    import qcl_negf_results.capture as cap
    from qcl_negf_results._capture_io import ObserverIO
    f=_publication_birth_fixture(tmp_path);budget=ObserverIO(128<<20)
    descriptors=len(os.listdir('/proc/self/fd'))
    read_fd,write_fd=os.pipe2(os.O_NONBLOCK|os.O_CLOEXEC)
    if terminal:os.write(write_fd,terminal)
    os.close(write_fd)
    with pytest.raises(ValueError):
        cap._observe_native_publication_fixture(f['job'],f['spec'],f['result'],os.getpid(),
            cap._start_ticks(os.getpid()),read_fd,budget)
    assert budget.reserved==0 and len(os.listdir('/proc/self/fd'))==descriptors
    assert not(f['result']/'performance').exists()


@pytest.mark.parametrize('technical',['performance','performance-source.json',
    'performance-anchor-current.json','performance-anchors'])
def test_publication_birth_observer_rejects_existing_technical_without_removal(tmp_path, technical):
    import qcl_negf_results.capture as cap
    from qcl_negf_results._capture_io import ObserverIO
    f=_publication_birth_fixture(tmp_path);path=f['result']/technical
    path.write_bytes(b'preexisting')
    read_fd,write_fd=os.pipe2(os.O_NONBLOCK|os.O_CLOEXEC);os.write(write_fd,b'exit 0\n');os.close(write_fd)
    with pytest.raises(ValueError):
        cap._observe_native_publication_fixture(f['job'],f['spec'],f['result'],os.getpid(),
            cap._start_ticks(os.getpid()),read_fd,ObserverIO(128<<20))
    assert path.read_bytes()==b'preexisting'


def test_publication_birth_observer_prepays_final_validation_before_holding(tmp_path, monkeypatch):
    import qcl_negf_results.capture as cap
    from qcl_negf_results._capture_io import ObserverIO
    f=_publication_birth_fixture(tmp_path)
    # Locator fits but the whole held-parent final validation allowance does not.
    budget=ObserverIO(32768)
    monkeypatch.setattr(cap,'_NativeResultGuard',lambda *a,**k:pytest.fail('guard held before close admission'))
    read_fd,write_fd=os.pipe2(os.O_NONBLOCK|os.O_CLOEXEC);os.write(write_fd,b'exit 0\n');os.close(write_fd)
    with pytest.raises(ValueError):
        cap._observe_native_publication_fixture(f['job'],f['spec'],f['result'],os.getpid(),
            cap._start_ticks(os.getpid()),read_fd,budget)
    assert budget.reserved==0 and budget.read_bytes>0
    with pytest.raises(OSError):os.fstat(read_fd)


@pytest.mark.parametrize('fault',['start_ticks','runtime_error','short_terminal','terminal_oserror','unreaped'])
def test_fixture_supervisor_uncertain_observation_is_finite_nonzero_once(tmp_path, monkeypatch, fault):
    import qcl_negf_results.capture as cap
    marker=tmp_path/'solver-spawns';killed=[]
    solver=[sys.executable,'-c',
        'import pathlib,sys,time; pathlib.Path(sys.argv[1]).open("a").write("spawn\\n"); time.sleep(.05)',str(marker)]
    if fault=='start_ticks':
        monkeypatch.setattr(cap,'_start_ticks',lambda pid:(_ for _ in ()).throw(ValueError('generation unavailable')))
    if fault in ('short_terminal','terminal_oserror'):
        write=os.write
        def uncertain(fd,data):
            if data.startswith(b'exit '):
                if fault=='terminal_oserror':raise OSError('terminal write uncertain')
                write(fd,data);return 0
            return write(fd,data)
        monkeypatch.setattr(os,'write',uncertain)
    def factory(pid,start,fd):
        if fault=='runtime_error':raise RuntimeError('observer setup error')
        return [sys.executable,'-c',
            'import os,sys,time; fd=int(sys.argv[1]); data=b""; '
            '\nwhile not data.endswith(b"\\n"):\n try: data+=os.read(fd,64)'
            '\n except BlockingIOError: time.sleep(.005)',str(fd)]
    if fault=='unreaped':
        popen=cap.subprocess.Popen
        class Stuck:
            def poll(self):return None
        stuck=Stuck()
        def spawn(argv,**kwargs):
            return stuck if 'pass_fds' in kwargs else popen(argv,**kwargs)
        kill=cap._kill_group
        monkeypatch.setattr(cap.subprocess,'Popen',spawn)
        monkeypatch.setattr(cap,'_kill_group',lambda child,sig:killed.append(sig) if child is stuck else kill(child,sig))
    began=time.monotonic()
    assert cap.supervise(solver,observer_factory=factory,drain_seconds=.05,require_observer_success=True)==125
    assert time.monotonic()-began<2 and marker.read_text().splitlines()==['spawn']
    if fault=='unreaped':assert killed==[signal.SIGKILL]


@pytest.mark.parametrize('fault',['pid','start','boot','namespace','inode','mode','token','no_locator'])
def test_publication_birth_observer_refuses_wrong_or_missing_actual_birth(tmp_path, fault):
    import qcl_negf_results.capture as cap
    from qcl_negf_results._capture_io import ObserverIO
    f=_publication_birth_fixture(tmp_path)
    field={'pid':'producer_pid','start':'producer_start_ticks','boot':'boot_id',
        'namespace':'mount_namespace_inode','inode':'inode'}.get(fault)
    if field:
        f['birth'][field]='00000000-0000-4000-8000-000000000099' if fault=='boot' else str(int(f['birth'][field])+1)
    elif fault=='mode':f['result'].chmod(0o755)
    elif fault=='token':f['locator']['capture_token']='f'*32
    locator=f['job']/'capture-source.json'
    locator.write_bytes(cap.json_bytes(f['locator']))
    if fault=='no_locator':locator.unlink()
    descriptors=len(os.listdir('/proc/self/fd'))
    read_fd,write_fd=os.pipe2(os.O_NONBLOCK|os.O_CLOEXEC);os.write(write_fd,b'exit 0\n');os.close(write_fd)
    budget=ObserverIO(128<<20)
    with pytest.raises(ValueError):
        cap._observe_native_publication_fixture(f['job'],f['spec'],f['result'],os.getpid(),
            cap._start_ticks(os.getpid()),read_fd,budget)
    assert budget.reserved==0 and len(os.listdir('/proc/self/fd'))==descriptors
    assert not(f['result']/'performance').exists()


@pytest.mark.parametrize('fault',['namespace','locator','technical'])
def test_publication_birth_observer_rechecks_scope_and_absence_at_terminal(tmp_path, monkeypatch, fault):
    import qcl_negf_results.capture as cap
    from qcl_negf_results._capture_io import ObserverIO
    from types import SimpleNamespace
    f=_publication_birth_fixture(tmp_path);budget=ObserverIO(128<<20)
    read_fd,write_fd=os.pipe2(os.O_NONBLOCK|os.O_CLOEXEC);os.write(write_fd,b'exit 0\n');os.close(write_fd)
    original_read=os.read;original_stat=os.stat
    def terminal(fd,count):
        value=original_read(fd,count)
        if fd==read_fd:
            if fault=='namespace':
                def changed(path,*a,**k):
                    st=original_stat(path,*a,**k)
                    return SimpleNamespace(st_dev=st.st_dev,st_ino=st.st_ino+1) if str(path)=='/proc/self/ns/mnt' else st
                monkeypatch.setattr(os,'stat',changed)
            elif fault=='locator':
                f['locator']['capture_token']='f'*32
                (f['job']/'capture-source.json').write_bytes(cap.json_bytes(f['locator']))
            else:(f['result']/'performance-source.json').write_bytes(b'foreign technical')
        return value
    monkeypatch.setattr(os,'read',terminal)
    with pytest.raises(ValueError):
        cap._observe_native_publication_fixture(f['job'],f['spec'],f['result'],os.getpid(),
            cap._start_ticks(os.getpid()),read_fd,budget)
    assert budget.reserved==0
    if fault=='technical':assert (f['result']/'performance-source.json').read_bytes()==b'foreign technical'



def _publication_fixture_plan():
    """Source-owned constructor-only plan, with portable provenance; no creator authority.

    Frozen Runner original 19cf3545fd108d38de38d83c46f2828d4557226cf524978b72cf5472bffcf06c.
    V3 profile canonical SHA256 376aedbbfa31311f40bdffdb988242613629d128c0a97e8cdd2d6b14b8c42e47.
    Compressed JSON is input metadata only, not numerical output or a solve fixture.
    """
    import base64
    import zlib
    return json.loads(zlib.decompress(base64.b85decode(
        'c%1EhU9aTEnciP%aButc@UuxC-$YQfS+Bi{vsMHwj4@GU6}#!fFBQo?b0i$V@&?%id$CwxuQtIVz+MEo(=If&63H?U<a*rm6XahK'
        'yj4ZA$Y%FBO?ID|kz0V_GfncTVpZ|2x89HE{Y(i^lA~yqD%WWh$8#mflDsNW8D*&#WK~)!*V<emFGx_qZ-sLGv(FSv!YGZ{JrP!}'
        'l@yiHhFs;9A4dUu_9~2Clocgc8%>!jBn(Law|G&SS7o7Gf2NR>(AAxnpcfwbSh-dg?!5UeO4)B$>cVJ#OfyUh<+{DASVv*0Tq{M6'
        'XhFO@MyWDa3PLcz2Qf-=_%lRtLF6v*)>Ku~E&1mnKr!5}E>ul7-Ohh=Wkn<Gg>jZq@w{ereypF?7QIKExkAY&6)8%ByG^%0A8wjf'
        'F;T7+@*<C;AS%7)Ic2WgX7naXR~{)*4Ss(=S4dGt2?FSxpc~J>D`Ds~*HGQ%Twznz%;*(NP>g&ME7!_T|LVh!{_KN~e)#^3Pw7mW'
        '(wUgj*+%=pEX_)<2qKb}Q5XfYXEPraB#u&|%$0<c>kRXi5oBo?tt!fTo8^QuD3~?A#t^-O3ssp{B}^Nl!8+Oy_%p2H*bAc+#maR_'
        'D>7H2v?MekI6@`y)=^Pr^iH`p0Jw|-xKB)WPP1RYw6gWee!GY94zdk_Cq7ePL_EL3D^e=gj^((nVOp*ZYYX{oE(8*5a)V;8AVHR5'
        '7)3`_7b?INgl<SZxJz489cO96g%U-nw}NHL-my@w_4~O}l9(i<6dkC-kWi0oND7~WZY}$5!#w6$l$Kru&Hjx4<4RMW!|$*ZpxMt9'
        '_)gbRjJ*&=aYbRN1X)@}X$5mp1U~Z0%*(T=DA>RwjB>W%#oAHk$|j3Tv|@uEr5hAS*vl!wBp^kRQ8rmw?)em@!5Zc_!_Z2UVv>^-'
        'leAP=XKA(}DM|wdRAdzmcn?v$%4k%slVX8VOlXm%3yhK+8o&OTQcDW{#5E-$p(G8+m9A>;m95?^?oblLYYzd5W#D0wm+R?pl&q>4'
        '(dbjLPDkRw3{VtI!XSxHFdiNxl%n{Kt@$e)p;ej{WfU9)@p3|=jIX58U>(8x=+~N$Zg3sNKA}(3)(xR4i9NrHVmx}z;w(V%)Ws8L'
        'w;o05%_2cBqNGaLKc=Hm5;K|Zm03*STPCA1ljy~ik2ec~D(5Ojn$hIYh&_7xK<ZLVuty1sy&N62YPwJbi9@em>m{uMfEbNdQj``O'
        '0xC+9B_u7Y<TxyxC9t0`xy^=Y!3FRR#(piV!D@E>1**zy;m28UQ-Ek-y8)rc^{j^`hAdcD>5YftXqCe2o&xe0nSZig3Yp>sSzIOQ'
        '$zWqdSH$yC5fxs!rleSB$I6i&VnPzw*lIl^9JQbHF{6}86nUyn0@&-UP>N0foK+>E@I9kACNUdYgr`BnY!^8SY7Ou-L))Ul5h^Ax'
        'JVYtF;|olTH~Zekw&KUAnEL*pBYIgXV7w$M5>y83v2eer;s}lC(|nTXDzbP(sOO{LhNPpvv7)4?B&)kKDHmg2QK%q6mhx?ft*w+P'
        'q#jet`JS1hh#t8DhLEEquV_@>d36&b^@2fcunwTui?i&e%ImZHR8)bMXSc@+>mjsSyE<(jr)pV4m_<pWpq!2)t)W`v9r?y-l@JQ+'
        'y;+yX4%@lEp|*#7FW!IbIi+yww+hw`@^p>1wzTsdm3CW;(kE=~?OkW0gC(ts(n}Cs9W^c;s5nbkQCVRydNrqoBd7HdfN$?up5mR~'
        'f!72hwdd0eVUiwqkRJUR6>#82@mn`2-1NH$)+D&$=FXgwFgkMf1qvp9wer?Dv_?L@i2~xSNXG26ivVpvKZ;4ZI@Y$h2l4pbXg?N>'
        '@QQf15gyyOh?0Pt&__JYP7)z1%27|p(P|CC@<n|zVpCsC#A?3$lb{tT%8PYYzAazNp9qfv!PmU6SIiA?H{5x45}h;-8{%!Eq8ek*'
        'JX#Q(VJvc*ttcsq1xCf%&k)647=b5k#9%a7jf-Xy2snMbI9r{dL+dI*DY%hFe#Q+@3<_)6Ecz34t%z1BI0sQ$dN^7|BNt+W2fjb<'
        '4LsXAe8jS%B+n?k&=lcGNl_)_&Av#1Hr!N&XjFg`vDl1ymp)D&A15;VNJn+!T;oDs<X|IC5WT_K?UDMt48)w+>@@fmyw$N&bB=!0'
        '@`Z^kZrvR>x?LdfbDvL*&ewMZw|WFw1y?*wLhxL1dt6pR9K=zQsRJ6v+1)6Vv9E!%6k=A^gp?j8X%Ur2%CQD1j7W?<KSISwfOWK5'
        'dm)QyNds^o2=iGE(!MMs(7;DRHwMlkpoAT?yj!#(X_ffLF1i*I;z2@g<)^%}Svu;3FaSxjYPC*DQFv<tn%q>tv2zk-snGdGx<(qL'
        '7?mX)T1t|z*q`GZsT1Un)Zi6*5f!~&-H~H0=<UOqm#?!lOU2P;Jj@A~Wt_cbL!6+Zm?&EEtl%Mr8I6v%skJCHN+Kpv0usmG2E4xH'
        '2}EpzH*J{|S(RdML*gtLKaxcrr5+}H>VnZI5PcAwR6>iuGv*@Yrja9oXoiJvLoy(ZRxJ*}9jRb+(Z;*mC6Hd=!f2evmp;mYDg&)i'
        'LA)|b&S%0LV^NfQD9(-}m&9pXpQ@rB*El6bgew$JTyz^kqwvlP!xFTU6lY;L8jDBS)KTSU9SB;biy%ft!3+>ntyhOdKD32N5zr_<'
        'UQ>dHQ}k4}09vJ25d`dl$bJsd!6+tmT<38e7mtq7e8r>f4#|`S7At);P+1ZhrK|Q}#&xqZFE6=ac6h^4kx@KNxSUuFN_`wSVLaUc'
        'vBLzxUVR#Sx|SPF{oGBmcqEAV-VnM1&_z3e-X~x|FTt5q73{Ag*ZAYWGnNB$i=xtt$~N!d^ev6DfIklUhQ#=;Sf2y>p)ljg4P-t9'
        'KElwPWz=~T<jA<v$2Gx^TgNN&k3$CoPeci30ZA+b4^9xMD9$rpQV921;8{SH2a+bq&;7-f5LpZknHP}H1xJ#}t<=G_A}i>WFHl3X'
        '#G_N<4#&K|VMnagL&?@X(EZ`{FNXg_-@#xTY#vV3KvAL<j7^rdR)8pBd0$6H`X8xcAMVR-l;Z66Nn0-|Nijr_dRY-fF*gHG9^V#~'
        '5_WPtA*lE)DG{f!6XE7GOQHg7n5WyEIP%f6+Je=Qy_3Y-1WmARf4mGl`{96hv>u=_`CI}*PL`Sg(?m-b6BT6{D2wB0^fx2}Rg^-='
        'hRCZuLDxoE#s^^TSWfRW5HWcKLk93PsPSf^U92^tB_)%)6eVaCL7Mr9oWNbQDMu6uCM8u#PqN0RA8aJBsfQQ|s>wG$35>Tw)%4Xl'
        '@r2VsIywVo7i8)gMTW{r01L`%jC)f7K~-S96Vq@?6vd-3q*N8I2sCmEAV<2T_hUDaqf^k!3WJ#^4k`$y^8_gc<V5D6kUx!y7j&_h'
        'T(Qnby^`XyXa=!;#hdL;tL>dEk?542I#triDf@J~yoOWu>STEZQ#-|3XWncVCfX>bwqHveQ02zvA9AF8os~_5%kjxLgCNI-a14Qr'
        '325gyt+z=Cz&ANjl;osj!7Xoey-E>0Z?qpG45Szr2XeCRFcPDR<{9kOicivD4FSLkP&(D~#n{j&VVVRpv@v!op?%9#N9<pMRS5oW'
        '2#+)%kLs}&EdjGjJfx8;4OpW(OhO8r8;J)@0Klr&5(FuXkBxZ28b!c@HbDDhK0ut-aP25zSSn2dvEai|c)&YQg>@)yl>F12{m4mX'
        ';7z-%HH;~1HzgW~%H~RKFi?Ow3{hYL+e;0jK-xPW^9^~yCS{lLMY*9UA++#rXqHC6ZMBEYg|Uk%b7hd7rI}#met>?3y(6c<O0$%}'
        'Jq@b>oAs*97{%8vxk@_+J0vMib7_a{L~kZ%khH`B`nEyx5km(uwmmE&>VmP*4Kso`?S)}x4_z~D(=nE&bmu;*4wLiL?V;RI7pf)S'
        '@F05z`z*R-7}{Lf%WNTt+<H_MT74R5qoAZv6<9Y7J%o~=v_dh=^FUHP|IYVuG#GQ`5Y2){uHfZ|*%Fo(d{;8#fexZ$n6|*ujJ+S4'
        's_N{07>597(%ZuXpe}T4t}ftzw)Om8${&?qQ#5JLm0pUyR{v6`zcWI;R)tX1g{#iU#~`32r9c6m$L{YtWRKnKYQ<4!3A_J5#<ZfT'
        '3w4$zKj#dGhcclJAhqi}t1g!9vjey9Wbi85LbpVxdOERg-OCh*1F5cz4H4-2<OEi+aC>Q=456Ess%Q&mmL$qtX@xy3t4@NbqFL(f'
        ')i((fX>IZnh`K5+i)X;!2wQ2J)+yTGo2qGPx_osf`4YGg2eT=S&hMQ>MO_7fGL(DRGCMNQ2;Wk>iG7^Fv<Xn)E4`m=*C0IO+=~lH'
        'IGQV+JUaOB2Gil%hxv8!@KC0k351Vz!dgo~*-st=T2Bz{{0iIEcxsq<-@z;`{$_g;ujGrpn|HFBq-%1blJv6xt$Q<=y#umzC>v*K'
        '-nX&7On^L)J)_tQAqI9*V)zHit_>!yDCWW<nmJ2;ex);wuuV(h%i?i)CYri%TuEjRhv+NnLT{zR;gk@?WkRTP-)^3VqAs-lZA>K>'
        'KN<^t$^X)roZ){O3yuBxe69?pY%o=GFa<-gK{oz~F~5pj2vA-t*vhM<vpa*50Mw^^yYqaMV&N-b!(iK1JuNcbZb#fUgAIXxY)f)%'
        '8*~o@ky>ne-;4hpdJmM^eG{VtKXAE@f*T&bCJl+C!wt7|Ki~Iq>i%~d@m5Q_^G?UUk-c!;N4(L}-v2&t>tt!L@qe!-%UPJ}#L0T>'
        '`*3l5-q+C|H$kU(ohO+dYoR?xIxOw`=g^gYN|$0R7j~m!K5852r(m@rwwnv3ScYcLZdT8dpBtdE^|PF6k-sN#2{t-y>5wbkopFTS'
        'Orv_h`^iI!rJL$3c?Ki7lM}?w2!}F!_#h9Z>?qc9mV6AFrF4YqYPubT0anUzevR5aAaf;hC-DW&wYym?N`1Hkr9;k4>8Xx8%FAG$'
        '1~>C`QYaWn_h3Q<|3bU@4I1cs6(5IQCIUBmF?C-`8&J0Yxcy@L+wHg8-)_Hp{Wsf}uRq`ZpY2!M|2^COZu`aS&$qwZe*5~DuYWar'
        '{hzj9ZGR)(^DW%-&Gy^vm)n=yueV>#{=@q}{=sbfo9!3d-@X28xP|}Y%k3Ah|Nix_UVr}j?`GTIZohc_*RTI(``g!lyZvUi{p$7S'
        'uRq^@y?wd;a{KM}Us+GS+P>U=vry&=&rmo_^MBPaz~9#){w>Wv*Kqi%Y>zEsr4J_p2x%h)bazz9I<i|#Za)lPrf=^@zSq}w0@EAG'
        ')J4CqZ3S#DTi*vegY3j>@&j5E!UA?YCK-nVv|)YsPApA~0jm@+S9*3ZcA(ehO2=6*ZruqJu5XrI-(YlL2FC*tx$J~1xu*sGfw)&^'
        't_%gL@(wUl-~NzOFjMUd8-;7!*fv01|EQ=+eb0TJ$VhnMhoT$R&OLQ}nhIMC6F|)L*$oH;69hBabRzn=9cMQp4=gXrU_@O#k9DJl'
        '4#w(3ac*GJ*cXOou*w3R$S+>dVknqPU1)Qqb?NV0-c6_#EH7uS48=@|mDg;A@Navrh$tq2f>I-LIFfjxD{V!aO@NSq=|}pA+nUcw'
        'bWD!D5x4ZdFmdC~2gW}n5!WJm+Y3+;fW6O+|HMpqHBRU25)BSSaljRKuC(D8;zQ(d6>N0wJJviA&>$69=|wbfbL+l%2X&#r!s7cA'
        'o9X@03_@BChZa1a_a$ZD&ktn0T`&9dT}vDM*7q=jU9xrdK5yg$cjmA%Bg1acW)63qo2@}{z-g&ygV$=Cm7z0TVaL75?G7DRAIx0O'
        '=O(no=Z$97_kzKM%|7PdsmAeC+gsQcL_$=>C0igKcY(3_e?Oz-x}d=`?!kE0_@>xgW@)kb1!&fWOF~BvbX7M(T~|p+w7_wff$uH@'
        '-Nx8eiLMhT(AB^<T#YPUf-RF6dZ=rvt1p%LPD><)g$yGl{@^XyR$vE2)9mHaa4f@L8oFu*q3zp9$HXW4a_Or^7}}<3JGS8<gdM}c'
        's^u(^X$ICnOXXco#CwFM?w}>Kpa(WKZFh;6KG9vrz$6I60Q+H}TcNIzz*JSVR8`e*oTU*iZG+gFFQW^yckv8HDgiCw88`ilwvemi'
        'B_^R6m<XA+;fDkx4HE~ep@l6HVq#)F2nh-u&BU(0bggBm6Wu}K(pm-&wAE_SOnk!EDH?W6vP8C~x`E?j+d{T)qQEkB4F$St1p3la'
        'ZO6tT)_pTH1IKj?RZ{~$G<9U+B_3?JS-a0*rx2qVnAkCGJJfM#8rVl}fGpc|RA1LDWEg?#yBZ-_*Y%}lnug)q*w-yn3w7NNY&a18'
        'WtFCcvZkUmq^R+}J>zcEXC3dUSR#ZhVZv?O3^l{@RaYmPZ!eLqSwwfSuWL96HH1Sm43O;;gCNy*1ERWtuIVnbZW9`T?+!#@TixkW'
        '7K#hX*z=>5JB?wB2r}V7{v?R6QnCuK0+ePc3p8i0K3>492(p}Xsy2D-T+Phcyr!4OZ(JkYT7t|p{iSO7fr<$UeHRhav`w4nhD!q7'
        'R}E~~j)`^b7#Ld)wsafTnS*Vx0cn=XPLptqliaZ$CP9o~y|k6Y<_bl(7f1ZX5r1*SUmWolNBqSRe{sZL9Pt-N{KXM}al~I7@fSz@'
        '#SwpT#9tio7f1ZX5r1*SUmWolNBqSRe{sZL9Pt-N{KXMJ$q`=!Ymy+@ZC3lJt~y>#;XMN<BntrOXNWQ<mIu*+D;8eL^Fkx;z?X{1'
        '*O{b?CXMu%4Co=CUIyw^xPJQTNW4zA>qrbKE%s-}5xot894H|Bc(6|1?YU^`qz0Q@-Y00Qn<qSS&bmp(qd|BQ@}ir8JQZzfMP~_f'
        '3_+Uo`2OVR@oRG^i+W`D_M}>mL>tV^9#`reGO`x~>DUv{_N8%8hulg3o&b3$ZF~ako^<mGh<mftr(h0dwx0lSI3xZ9!21&DCqN%a'
        'ygv!*gW3AyTIr_ukHgr@2rv$=M<8%k9?`bdK1sp2zPDrsZwUh_B*fSGPT4I}eVGEyy30eIA$z6?XjQI{4=4OLJQnSGYE0puz!zB%'
        'Ptx4yN>%U(!&(9My8JH9&$~n~&80!cmv@cu#oJ+fX|C*Hqj(!wUk+1$Jj4mJj}+$MVQ%M%eQaUh%Wr3QyxLD_MYG#--AO`V>Bf9r'
        'b#nwXm51;*S6}2PE&7>4jFyKdNyz#MPnYP^^&FhFiw=3wAul@QMTflTkQW{D6#HPi^GGUWRc6426I@8+zK5i7`&r`-DjoyREcVmP'
        '9djxiLO#S%HyL}7+-@prml$s<+AycyRMe@4;~~tN#&0Ul9s<BeqrQuW5Rt;!IlX^Bi5vUOjR!?dbLZ)(*iT%ortM*maPq81xPGfO'
        'spS(jI*NPyNH^YYYe$iEPi^h(U1x)=qsPeH7U}>)=_vu<-myHzJHG?3xx)y1{?F)SwHN;AZTszR*(wJg@CEysQl$|XnHdGh2E`TI'
        'oZb65+yQc7@DAWKzMm_9;eD_dLUr!v%7<`+3;*vk6Xu8B4bwh8^gj7i{P8JJkHH_m|Ih4P^b_uSW#<uQfR|bjWOSkpAh}@wF}C$j'
        'K0SCdv#m<EzxqwXnlRjbRnx8e`wqA9K}Cb#wiw_)K1L`A6qzY`)Z-mi<wcKD`2>i^D=GW2I!w-E@LIy*5l>5;&?BDS%O5=nYbVKd'
        '5}HI#{Z<H3sYC_ut5jTexAPqa?NQ^fOSgSKCHD+tQ4=^W`i$Ts2BX0mRp@*Nq{lu!1*=0*J_Tr)+k6V*5CQrWyp}orDXPHiTUNVl'
        '?IYf)=5v3OFEU6XKjK~XFwT!gI;tDz8kZeX{IRbiGw8qZw>oxe&e4y-VgX}Dcc{3)c}`dFqFGaD-|fNW<ON3B14B>$cht^*;A1&Z'
        'f@B|zs#z?i=2Dg9QkCRMs*?1|k{mp=);BI5DqeDoU~579bxzJ@G-_HLte!He?{bZmQ2>uC$ubK1fdVe$Fb~#wnFzUC5@s6U*f|-j'
        'BQpwVu<p!L=p#~tSLj7l^m--A-HdvrojMU?VNSR#`*kPY_z-u?U5t9CouU}yP=+gLoL$837UUS!{eJb1aUlINALCFC76uswIaEjF'
        '`yGDJs2sA3fs$~)f&G52{HfSrn_E7f^{n=etLc`$a4k!9blU_S{6i0C`Tzab@4oz^YwXwC$4slExwE!D`tHlGqz8V~-fg!xeD~!y'
        '2k!~Jw)EJ_<6+sIU1`rL0O|_uUVtKBK!HDeRf7_y@@p_6{eAZuTIXJ!-TS;#0qukDe*NF~P6sQW1?QCElT6#yUZJC;$>{p-<v%}A'
        'P%HWF*I#$<df~kn*Ok3qy=2#*5D&cM!QQI-dkS^N9@*33Dok=O;f}hG>Tz9sU_iBn`)-V<voPO=;XKeoy+4%vEcNzu5XyBu#TY)H'
        'E1!T++T#M%yZ%xx?jC*rrhDhC=-dmnx7cl;{^3`@+-nTEpDX|GfBycz?g-l6NE)m((yM>?*kdQiy*EO2kXQfk-Ip)F`}P0$?#tip'
        '`_Q^(91x8f^8o+12cC3$PqK0)@Zdk~>5<Jn%m1r41gwr(y)!lS&EU?LFs8in*sGV`AO6cfH8nba|GAe)7&j}}7$qDiv$jx~f4bgz'
        'Jw|~4Ao0e#tX$Nu*y<QKDsn=3Z5~!N%?Cp*qzv|}ba2Zbr4gaE{_YjgUKIlBn?N#EQ)5A{-X&`B)=hLyDEwNK262VK{vNDR3dM7T'
        'Q0nAxwFe~}CD12UKS|{1g^<oJou5RH;tUD}pgS*@zsJ1je`s+gYE`xAheBk;85_rD6hKK$jP6Jn18{kkL$w?}*5#T&DkGFn%hG1x'
        'YdG?8=M(9O3Xa#jI?MKaL2zd{dZp6jOVVI_KN?on)cVWl)a$#j(OlSQE^IUxHku0?&4rET!bWppqq(rrT-a#-7}#iJJy;G;9v!%H'
        '2=2)6l`h(NE-MLSr=IpmoZS-KV}Uy5y5Gh{F<d+RteIf@Tb~*LbzI~9D(EBUc(88zNUUz5^)n8{J821$6VjVR*ADCKg|zCAkF;to'
        'hXr<|7?9!*Z&KYWEKmoMK!k0tvlX#A_AJ=Fh~VG+o}Hb>0Fow7pC~vT1bd>Z>xN<Ls<Ct|)3z-KDtdE1#HMmN>(K6|qtyHiqF`vh'
        'qut#__!YRlv|wSs%=-tXPt0poLzw`cSRv|hbLvpY)n>Ki#R_2-!dn)t<ArgSLCl#joDXjoP<nbulC$#BT_LMm%egWrGP#koD*<`U'
        '?f|jU+YgxW_-#UwI8FkKN5@1wiAUSW2-$r3{dW6{d6As{UFS#FVWUA>;U9kWOYhIYFSAs2RxR6vkq{TaaI7x0x$@^-ocCs3^So{v'
        're>IqYU}pGH5>;#MnB%`dlIg5b_Y0CL{0jn7ol*olCldInri5dtvZgO8O-}*K={F{uf3l@V8TLmElqbc+qHGu(G9UmOYq(C`oovV'
        '%S+_tCGzqTd3lMvyhL7JA}`;G$V+((Z*wAeSoE2V5^}>VZvsa}M)5RD%!zGAQXdC``=4%rSR6mWUY!W>bS*bQ@)yg@?n$85ecRK('
        '5C8Npn?vc3|9$gk{bjX3@Hj8=uXbA54@|RW=7ERLf9y6Ww6h^}1=GK1*S+@%Z~&A*hfx*muOmgn<G?dE9JeSc+rxcvj+RE{@IMav'
        'hQ#=;Sf2y>q1rB!8_0a9TmeIKmQm+sn~{Y<9@hj<5IKGmd>lF$kZJ>7!Ms+m^P-|S&v;38zq7zI0znUCPm-Veiz}ftOex~(nB8!4'
        'D|L00Jc|lo9XR0&)X-puJ1Ookwf`HYl975SW$O=ge|Y_i;Xl!LFxUp0hZ8kWlxPKG!zyhE?Kw(vnr+CDrJ9b+13TQ8+bG4^?US}%'
        'Qj%iexb?Ckh~hdxb>jH8sB%Ts(}0T4l9DDk^(44C&622KA+jgioH+8)v)Y2yzNks!ZIsU5ZGXHBJo}l7ceEa$F-7zQgzRCP0Mqye'
        'i;0S|43x!jH2NEofhtPD4+nLGqO>|e*GA>?4#3<M)89@^9>I_>Fb!&aD8eq*8qt!H$z6&Pw2D$h??x12+(nymM3G=pQh^ur1S??r'
        '!A1g`dWeyrgk(7pjJHD7^wl}>gwsJfnj?P~Wa=43hN|7_5eC+i=`UD-ssO&<i8xR%Y7~Y<%FY#mMozI3RO5&=_G35Pm!r|&Nk2Nb'
        'Z%H&)tro`f-1>)69@HWB=UyQ&i1FBGhhniTwdIJ%TL@7qgD?Ut$nBD3TK_uS(8{TSr*}*x^;7NVzf-{sqD2naCyZ_&(Qcau>;VoG'
        'P3s@vce{5F;ZXYz<q+eu?_k{)X=AnEpyLqB!fm3DI(M=VoQKYiFbEHw|9IBi?#WDrU@8__^qI1AXwrBvSnt%s>pSe8T398!!Ba;)'
        '&vm1Q&I;2>QiEUFsd)oUH~W<#0+m=85&WYg83quvxzY+ectPsD$Cs5mSE%=X6Zb*Af6KoU2u@<bC}QbBMqzW$Po#T(@>B7I3jg|#'
        '{4N{*hHp^1#C5j@4|E>sYAQV_G!N-&<`?bQ(Ym}w`$MbGHIEb!2q%_4WdkEM{_3Blw|Vtz>9JQ|w=prV{srDtU1-1=@GtBkh|~e2'
        'ds*SThi&d~*CIf10NJ6$%W`%&0-2k`3Vs^W9$11gx#C@h4TkqhqVuE~P{9^&wma?7`efxuPT8qbm998tpH5dD;gr2PS#^S`ok~c#'
        'K5x00hr2M*AUw7GTH=809b1lyBjxL?YzWYfFX=G|axBBw5Xcz*bB@ynr3wM~CMSxLoRp0F?2WEhi9+Cw_Cv&~6yt(bPS%}w^1Q^l'
        'E*+N+s2?J#U|1J;KJOp3`WrK3_jR|$=aMce{np<SU|?J6{Jc3E@X?1Hi#>9yIVSB&qeI132IS8E;wC%y*CNKyS;YR1)Zpm`oi7;l'
        '?e3?vmpbkFTp3>NE%jTcgVz1?d&c1bD-jl*M~X56W+XO6O{HzGIm*_NBp4#d2=?kk44#3(dk<kI0V*@<Wg9}*P!+RtT~wZd(>HBe'
        'WQ4M?c$I6#bbX?u&~&lC3_~3)HL|oq)6rc&^no#YsS=C=+clP<O>D=v^}t2}Qe8JtT>`8slzloLu<TYbYN0L1!Nf5%4Fy_gYQE{}'
        'nrkfsvh)MTMMSrKL@YnVmWrIfahGbK5o}wQZZ3%#YAgu`zB-ZTp|pMqev+gceoU-VsCe%)Nx(R>17?}`ZX%v{6Qh*LfM&qiJuzZO'
        'Qk1hgu9P6bCokn&ttf)S$XxoRf@&2_A=B<feu%hZKeF1T(tU;rDIx(%F_?#dzMF|IejK%2NEc`?a<-g|CX5><qllkOG)T9B`<R_U'
        '0yo{I!we)zj)KxF)=^k`8C%8~Exd@y2R|l|C!kq6ynYHIhuxPTFffqrZR%WM_J1z|Ndi`vwl#fes%j%bI0%e!QKN&UUb<g%UFh}-'
        '_-<-)XxlK&Hmm10DjAn<ow#G$y3=G3dUaM({X#sis7a<0bt1h01#3Pb6y177^uhzG)250fyHKa?$h|Y!X_AyEN(uIE5lx_zV9wWd'
        '6KqS65rSxICy<}ERJftcU@4KrCm1s)m1bEm5aNQIM8VRH2)e01Q_VG0m)VF4W?yZ7H&nwh@1>PmJN~+&+np=*vc_#$EN`d@QT_6='
        '+%L=h?OpD5iPoJ}o)A=208c?s$a`G#GJR0^;;Xcb60&D^EP^#jkaDej62w<2S%p{i=6qFuR971y`U0s1Mll6#P!z-d3@Myh2?TC@'
        'R~AW(UHmG&Nkxs4en*s{y&Jg4_?EV!D1{mk@>_$NT_+Nv9C4OmP<u9rMo`Kpg5AzX1%aGb1vHc}N>_yDl*M%^z8;v%(DE(UQe6yN'
        'I|_8))g9fn152}ALIPq~I#!A1>VD|>hUV&aXy~T1)G;KhF%~%3T)A$9NysJ)HVSBbzrxPPu_2Z4B7VScez?-$^%zLg2IE+Zkxzg)'
        '@~40G;YWY=!AC!Qf5sb`NsY`zBeRY6gISuDUJ*p#s|p!$UXGZ99k4})DdPE%hkN@R2C#{?;K{tS5?Ugo?ei$vh!_B#&kOa3uE3EG'
        'CfutH3|NYQ!+JvRYu)=Zq3$!!L~mY^uIpytXtru#TMe8g@)5x<)*a2XEEmomu5S@tRm0FAmZteWHq0ROb>DF<Sa#A;LTF9wHL^G+'
        'nC+w8dm3%J9#ZLU7+rBZz$7OrE<EP#dM@u&U0t@83i9#*6<=B$b0y5EA7M<OrdF*I0%%Sn$d%`6SL15v-a`p!i|xq?+z5m>Qj{pg'
        'C<gjMnDE9f0XiW{p}4&Ak@hUi=q;i!QZ0WXvmx^Y{THP0P@cz8>w@Z>7riwC$G&11261d##eS&ie&A?E=p!939mjTU&CxV12z_EB'
        'kf*MWUF0rJ%^*bARm%!}SWtlEku=D-=xs$;fYK~wK|E|!`3J|>X`_<6W`J}DyEYD7%hFX#)d(^SUH3yDlDHjB3mwPML(|dxrG+(h'
        'sbk-8HR3p`hAohKUA96v3pFPp=^*5>v{x1Qf)CCr`-u%Y|0fs!ZEAi1?iV=D<0d7WU=om`$f)#5AKNRwCRnb0TUc#)JAiwxI<v`#'
        'w%#h(3TRjM&~8^t5pe9V5>{dage(z`C|($-e%4-sUMha8;p<AGN|nOb2o5cR8<8S1hGvLTbXWXf2HQ(K13H2kQ@dwTF^f|EQIfQr'
        'W#Oz`lUaMLnjspj0g5O4GXl#s!Bxzcn)nBy?HhO)t>BzSvkdB!9Jd>=hlIGP%^$mS&`QHAIAw6L%6<%pCpyduC$SQX?bcpD)f1Ms'
        'W!QG;JH8$gHPFKV8>&l8Csdcd;W{?KXi3~K3^g}2L&tK0r9*7hG?#{k17s-o{~y;Q?@<'
    )))


def _publication_fixture_job(tmp_path, mutate=None, command='native-publication-fixture'):
    import hashlib
    import qcl_negf_results.capture as cap
    from qcl_negf_results.model import canonical_bytes
    f=_publication_birth_fixture(tmp_path);plan=_publication_fixture_plan()
    if mutate is not None:mutate(plan)
    plan.pop('fingerprint')
    plan['fingerprint']=hashlib.sha256(canonical_bytes(plan)).hexdigest()
    raw=cap.json_bytes(plan)
    f['spec']['selection'].update(execution_id='execution-1',plan_fingerprint=plan['fingerprint'])
    f['spec']['plan'].update(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
    f['native']['execution_id']='execution-1'
    f['native']['input_plan'].update(bytes=str(len(raw)),sha256=hashlib.sha256(raw).hexdigest())
    f['locator'].update(execution_id='execution-1',plan_fingerprint=plan['fingerprint'])
    f['arguments']=(command,'plan.json',str(f['result']),'--execution-id','execution-1',
        '--source-graph','source-graph.json','--native-execution','native-execution.json',
        '--native-calcjob-uuid',f['spec']['selection']['calcjob_uuid'])
    for name,value in (('plan.json',plan),('capture-spec.json',f['spec']),
        ('native-execution.json',f['native']),('capture-source.json',f['locator'])):
        (f['job']/name).write_bytes(cap.json_bytes(value))
    return f


@pytest.mark.parametrize('relocate',[False,True])
def test_publication_fixture_original_ingress_is_readonly_and_profile_portable(tmp_path, monkeypatch, relocate):
    import qcl_negf_results.capture as cap
    def moved(plan):
        execution=plan['executions'][0]
        execution['resource_estimate']['worker_capacity']=64
        for history in [execution['provenance']['sources']]+[
            section['input_assigning_histories'] for section in execution['resolved_snapshot']['sections'].values()]:
            for key,values in history.items():
                history[key]=[value.replace('fixture-source:', '/nix/store/fixture-source/') for value in values]
    f=_publication_fixture_job(tmp_path,moved if relocate else None)
    for name in ('ScalarPublisher','NativePerformanceInputs','NativePerformanceCapture'):
        monkeypatch.setattr(cap,name,lambda *a,**k:pytest.fail('fixture entered technical capture'))
    monkeypatch.setattr('qcl_negf_results.telemetry.TelemetryWriter',
        lambda *a,**k:pytest.fail('fixture entered telemetry writer'))
    before={str(p.relative_to(f['job'])):p.read_bytes() for p in f['job'].rglob('*') if p.is_file()}
    assert _N3_observe(f)==0
    assert {str(p.relative_to(f['job'])):p.read_bytes() for p in f['job'].rglob('*') if p.is_file()}==before
    assert not(f['result']/'performance').exists() and not(f['result']/'performance-source.json').exists()


@pytest.mark.parametrize('fault',['root','method','point','scattering','snapshot_model','raw_configuration',
    'history_nonpath','history_fragment','history_order','history_unknown_source','history_trailing_separator','history_missing_key','worker_count'])
def test_publication_fixture_rejects_whole_profile_mutation_before_technical_effect(tmp_path, monkeypatch, fault):
    import qcl_negf_results.capture as cap
    def mutated(plan):
        execution=plan['executions'][0]
        if fault=='root':plan['root_definition_id']='ordinary-study'
        elif fault=='method':execution['method_id']='other'
        elif fault=='point':plan['points'][0]['voltage_per_period_V']=.056
        elif fault=='scattering':execution['resolved_configuration']['scattering']['lo_phonon']=True
        elif fault=='snapshot_model':execution['resolved_snapshot']['model_binding']['model_sha256']='f'*64
        elif fault=='raw_configuration':execution['resolved_snapshot']['raw_configuration']['scattering']['lo_phonon']=True
        elif fault=='worker_count':execution['resolved_configuration']['execution']['worker_count']=2
        else:
            history=execution['resolved_snapshot']['sections']['output']['input_assigning_histories']
            key='output.save_plots'
            if fault=='history_nonpath':history[key][0]='unexpected-rule'
            elif fault=='history_fragment':history[key][-1]+='changed'
            elif fault=='history_order':history[key].reverse()
            elif fault=='history_unknown_source':history[key][0]='/nix/store/other/unknown.yaml'
            elif fault=='history_trailing_separator':history[key][0]='/nix/store/source/reference-2019-70k.yaml/'
            else:del history[key]
    f=_publication_fixture_job(tmp_path,mutated)
    for name in ('ScalarPublisher','NativePerformanceInputs'):
        monkeypatch.setattr(cap,name,lambda *a,**k:pytest.fail('invalid fixture reached capture effects'))
    assert _N3_observe(f)==2
    assert not(f['result']/'performance').exists() and not(f['job']/'live').exists()


@pytest.mark.parametrize('fault',['ordinary_command','v1','missing_native','wrong_plan_claim','wrong_graph_claim'])
def test_publication_fixture_never_downgrades_original_ingress(tmp_path, monkeypatch, fault):
    import qcl_negf_results.capture as cap
    f=_publication_fixture_job(tmp_path,command='run-plan' if fault=='ordinary_command' else 'native-publication-fixture')
    if fault=='v1':
        f['spec']['schema']='qcl-negf.capture-spec.v1';f['spec'].pop('native_performance')
        (f['job']/'capture-spec.json').write_bytes(cap.json_bytes(f['spec']))
    elif fault.startswith('wrong_'):
        f['native']['input_plan' if fault=='wrong_plan_claim' else 'source_graph']['sha256']='f'*64
        (f['job']/'native-execution.json').write_bytes(cap.json_bytes(f['native']))
    monkeypatch.setattr(cap,'ScalarPublisher',lambda *a,**k:pytest.fail('fixture downgraded to ordinary capture'))
    read_fd,write_fd=os.pipe2(os.O_NONBLOCK|os.O_CLOEXEC);os.write(write_fd,b'exit 0\n');os.close(write_fd)
    assert cap.observe(f['job']/'capture-spec.json',os.getpid(),cap._start_ticks(os.getpid()),read_fd,
        native_execution_path=None if fault=='missing_native' else f['job']/'native-execution.json',
        solver_arguments=f['arguments'])==2
    assert not(f['result']/'performance').exists()



def test_capture_admits_full_source_owned_plan_workspace_without_relaxing_actual_objects():
    import qcl_negf_results.capture as cap
    from qcl_negf_contracts.messages import ContractError

    plan=_publication_fixture_plan();raw=cap.json_bytes(plan)
    assert cap._capture_decode(raw)==plan
    # The raw decoder has a separate conservative temporary-object forecast;
    # accepted retained native/performance metadata still has its actual16k cap.
    with pytest.raises(ContractError):cap._capture_credit([None]*16384)


@pytest.mark.parametrize('fault',['caller_bytes','workspace','prospective_objects'])
def test_capture_rejects_unfunded_decode_before_materialization(monkeypatch, fault):
    import qcl_negf_results.capture as cap
    from qcl_negf_contracts.messages import ContractError
    monkeypatch.setattr(cap,'decode_json',lambda *a,**k:pytest.fail('unfunded JSON materialized'))
    if fault=='workspace':raw=b'{"long":"'+b'a'*700000+b'"}'
    elif fault=='prospective_objects':raw=b'['+b'0,'*20000+b'0]'
    else:raw=b'{"bounded":true}'
    with pytest.raises((ValueError,ContractError)):
        cap._capture_decode(raw,maximum_bytes=1024 if fault=='caller_bytes' else cap.CAPTURE_METADATA_BYTES)


def test_publication_fixture_insufficient_io_refuses_before_original_plan_read(tmp_path):
    import qcl_negf_results.capture as cap
    from qcl_negf_results._capture_io import ObserverIO
    f=_publication_fixture_job(tmp_path)
    f['spec']['credits']['io_bytes']=16384
    raw=cap.json_bytes(f['spec']);(f['job']/'capture-spec.json').write_bytes(raw)
    budget=ObserverIO(16384)
    with pytest.raises(ValueError):cap._spec(f['job']/'capture-spec.json',budget=budget)
    assert budget.read_bytes==len(raw) and budget.reserved==0
    assert not(f['result']/'performance').exists()


@pytest.mark.parametrize('strict,want',[(False,0),(True,125)])
def test_fixture_supervisor_forwarded_cancel_cannot_accept_child_zero(tmp_path, strict, want):
    solver_ready=tmp_path/'solver-ready';observer_ready=tmp_path/'observer-ready'
    solver_script=('import os,pathlib,signal,sys,time; '
        'signal.signal(signal.SIGTERM,lambda *a:sys.exit(0)); '
        'pathlib.Path(sys.argv[1]).write_text(str(os.getpid())); '
        '\nwhile True: time.sleep(.01)')
    observer_script=('import os,pathlib,sys,time; fd=int(sys.argv[1]); '
        'pathlib.Path(sys.argv[2]).write_text(str(os.getpid())); data=b""; '
        '\nwhile not data.endswith(b"\\n"):\n try: data+=os.read(fd,64)'
        '\n except BlockingIOError: time.sleep(.005)')
    helper=('import sys; from qcl_negf_results.capture import supervise; '
        'print(supervise('+repr([sys.executable,'-c',solver_script,str(solver_ready)])+
        ',observer_factory=lambda pid,start,fd:[sys.executable,"-c",'+repr(observer_script)+
        ',str(fd),'+repr(str(observer_ready))+'],drain_seconds=.5,require_observer_success='+repr(strict)+'))')
    process=subprocess.Popen([sys.executable,'-c',helper],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    try:
        deadline=time.monotonic()+3
        while not solver_ready.exists() or not observer_ready.exists():
            assert process.poll() is None and time.monotonic()<deadline
            time.sleep(.01)
        process.send_signal(signal.SIGTERM)
        output,errors=process.communicate(timeout=3)
        assert process.returncode==0 and not errors
        assert output.strip()==str(want).encode()
    finally:
        if process.poll() is None:process.kill();process.communicate(timeout=2)
        for marker in (solver_ready,observer_ready):
            if marker.exists():
                try:os.killpg(int(marker.read_text()),signal.SIGKILL)
                except ProcessLookupError:pass
