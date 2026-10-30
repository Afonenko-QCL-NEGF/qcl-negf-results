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
