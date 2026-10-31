"""Startup credit is the existing validated locator, not a new admission service."""
import hashlib
import os

import pytest

from qcl_negf_results.capture import observe
from qcl_negf_results.commits import json_bytes


def setup(tmp_path):
    plan = json_bytes({"fingerprint": "a" * 64, "points": [{"id": "P", "execution_id": "E"}]})
    (tmp_path / "plan.json").write_bytes(plan)
    spec = {"schema": "qcl-negf.capture-spec.v1",
            "selection": {"calcjob_uuid": "job", "execution_id": "E", "attempt": 2, "plan_fingerprint": "a" * 64},
            "plan": {"path": "plan.json", "bytes": len(plan), "sha256": hashlib.sha256(plan).hexdigest()},
            "capture_token": "b" * 32, "allocation_id": "calcjob-process:job:2",
            "allocated_logical_cpus": None, "scratch_root": None,
            "credits": {"performance_bytes": 32 * 1024**2, "scalar_bytes": 64 * 1024**2,
                        "io_bytes": 128 * 1024**2, "reserve_bytes": 0}}
    path = tmp_path / "capture-spec.json"
    path.write_bytes(json_bytes(spec))
    return path, spec


def test_no_locator_before_terminal_means_zero_observer_output(tmp_path, monkeypatch):
    path, _ = setup(tmp_path)
    monkeypatch.setattr("qcl_negf_results.telemetry.TelemetryWriter", lambda *a, **k: pytest.fail("writer started without credit"))
    read_fd, write_fd = os.pipe2(os.O_NONBLOCK | os.O_CLOEXEC)
    os.write(write_fd, b"exit 0\n"); os.close(write_fd)
    assert observe(path, 42, 7, read_fd) == 2
    assert set(p.name for p in tmp_path.iterdir()) == {"plan.json", "capture-spec.json"}


def test_corrupt_startup_spec_has_no_writer_or_inherited_log_write(tmp_path, monkeypatch):
    path, _ = setup(tmp_path)
    path.write_bytes(b"{")
    class ForbiddenStderr:
        def write(self, *_): pytest.fail("pre-credit inherited log write")
    read_fd, write_fd = os.pipe2(os.O_NONBLOCK | os.O_CLOEXEC)
    try:
        with monkeypatch.context() as patch:
            patch.setattr("qcl_negf_results.capture.sys.stderr", ForbiddenStderr())
            assert observe(path, 42, 7, read_fd) == 2
    finally:
        os.close(write_fd)
    assert set(p.name for p in tmp_path.iterdir()) == {"plan.json", "capture-spec.json"}


@pytest.mark.parametrize("foreign", [False, True])
def test_delayed_locator_is_checked_before_first_writer_call(tmp_path, monkeypatch, foreign):
    path, spec = setup(tmp_path)
    read_fd, write_fd = os.pipe2(os.O_NONBLOCK | os.O_CLOEXEC)
    def publish(_):
        assert not (tmp_path / "performance").exists()
        locator = {"schema": "qcl-negf.capture-source.v1", "capture_token": "c" * 32 if foreign else spec["capture_token"],
                   "execution_id": "E", "plan_fingerprint": "a" * 64, "mode": "direct",
                   "actual_root": str(tmp_path / "result"), "epoch": 0}
        (tmp_path / "capture-source.json").write_bytes(json_bytes(locator))
    calls = []
    def writer(*a, **k):
        calls.append(True)
        raise ValueError("fixture stop after permitted writer entry")
    monkeypatch.setattr("qcl_negf_results.capture.time.sleep", publish)
    monkeypatch.setattr("qcl_negf_results.telemetry.TelemetryWriter", writer)
    try:
        assert observe(path, 42, 7, read_fd) == 2
    finally:
        os.close(write_fd)
    assert calls == ([] if foreign else [True])


def test_T3_index_io_refusal_keeps_last_immutable_publication(tmp_path):
    from test_capture import _T3_capture, _T3_anchor
    from qcl_negf_results._capture_io import ObserverIO
    fixture = _T3_capture(tmp_path, history_only=True)
    capture = fixture['capture']; writer = fixture['writer']
    anchor, raw, immutable = _T3_anchor(fixture)
    ref = dict(path=str(immutable.relative_to(fixture['result'])), bytes=str(len(raw)),
        sha256=hashlib.sha256(raw).hexdigest(), schema=anchor['schema'], media_type='application/json')
    capture.remember(anchor, ref, raw)
    first = capture.publish()
    pointer = (writer.root / 'current-source.json').read_bytes()
    original = (writer.root / first['path']).read_bytes()
    capture.budget.maximum = capture.budget.used + 1
    with pytest.raises(ValueError, match='credit'):
        capture.publish()
    assert (writer.root / 'current-source.json').read_bytes() == pointer
    assert (writer.root / first['path']).read_bytes() == original
    assert not writer.closed
    with pytest.raises(ValueError, match='credit'):
        writer.close()


@pytest.mark.parametrize('fault', ['metadata', 'objects', 'count', 'callback'])
def test_T3_explicit_inputs_refuse_combined_credit_or_untyped_factory(tmp_path, fault):
    import qcl_negf_results.capture as module
    from qcl_negf_contracts.messages import ContractError
    from test_capture import _T3_capture
    fixture = _T3_capture(tmp_path)
    values = fixture['inputs'].document()
    if fault == 'metadata':
        values['sources'][0]['canonical_plan_directory'] = '/' + 'a' * (1 << 20)
    elif fault == 'objects':
        # Many small objects breach object credit while encoded bytes fit.
        values['sources'] = [dict(x=None) for _ in range(8200)]
    elif fault == 'count':
        values['sources'] = [values['sources'][0]] * 257
    else:
        with pytest.raises(TypeError):
            module.NativePerformanceCapture(fixture['job'], fixture['result'], writer=fixture['writer'],
                transport=fixture['transport'], inputs=lambda *_: fixture['inputs'], budget=fixture['capture'].budget)
        fixture['writer'].close()
        return
    with pytest.raises((ValueError, ContractError)):
        module.NativePerformanceInputs(**values)
    assert not (fixture['writer'].root / 'current-source.json').exists()
    fixture['writer'].close()


def test_T3_factory_requires_the_same_aggregate_writer_io_ledger(tmp_path):
    from test_capture import _T3_capture
    from qcl_negf_results.capture import NativePerformanceCapture
    from qcl_negf_results._capture_io import ObserverIO
    fixture = _T3_capture(tmp_path)
    with pytest.raises(ValueError, match='ledger'):
        NativePerformanceCapture(fixture['job'], fixture['result'], writer=fixture['writer'],
            transport=fixture['transport'], inputs=fixture['inputs'], budget=ObserverIO(128 << 20))
    fixture['writer'].close()


def _F1_actual_delivery(fixture, monkeypatch, *, allowance=None, publish=None):
    """Trace actual read/readinto calls once, independently of ledger mirroring."""
    from contextlib import contextmanager
    import json
    import qcl_negf_results._capture_io as capture_io
    import qcl_negf_results.capture as capture
    import qcl_negf_results.catalog as catalog
    import qcl_negf_results.performance_views as views
    budget = fixture['capture'].budget
    baseline = budget.used
    if allowance is not None:
        budget.maximum = baseline + allowance
    trace = dict(reads=0, writes=0, before_effect_overshoot=[])
    original_regular = capture_io.regular
    original_write = capture_io.ObserverIO.write
    class Stream:
        def __init__(self, raw):self.raw = raw
        def __getattr__(self, name):return getattr(self.raw, name)
        def before_read(self, size):
            position = self.raw.tell(); end = self.raw.seek(0, 2); self.raw.seek(position)
            count = max(0, end-position) if size < 0 else min(size, max(0, end-position))
            granted = budget.maximum-baseline
            if trace['reads']+trace['writes']+count > granted:
                trace['before_effect_overshoot'].append(dict(delivered=trace['reads']+trace['writes'], next=count, grant=granted))
        def read(self, size=-1):
            self.before_read(size); raw = self.raw.read(size); trace['reads'] += len(raw); return raw
        def readinto(self, target):
            self.before_read(len(target)); count = self.raw.readinto(target); trace['reads'] += count; return count
    @contextmanager
    def regular(*args, **kwargs):
        with original_regular(*args, **kwargs) as raw:yield Stream(raw)
    def write(owner, count):
        result = original_write(owner, count)
        if owner is budget:trace['writes'] += count
        return result
    failure = None
    with monkeypatch.context() as patch:
        for module in (capture_io, capture, catalog, views):patch.setattr(module, 'regular', regular)
        patch.setattr(capture_io.ObserverIO, 'write', write)
        try:
            (fixture['capture'].publish if publish is None else publish)()
        except BaseException as error:
            failure = error
    trace['ledger_delta'] = budget.used-baseline
    trace['reserved'] = budget.reserved
    trace['outcome'] = 'success' if failure is None else type(failure).__name__+': '+str(failure)
    print(json.dumps(trace, sort_keys=True))
    return trace, failure


def _F1_remembered(tmp_path):
    from test_capture import _T3_capture, _T3_anchor
    from qcl_negf_results.capture import _capture_ref
    fixture = _T3_capture(tmp_path, history_only=True)
    anchor, raw, immutable = _T3_anchor(fixture)
    fixture['capture'].remember(anchor, _capture_ref(str(immutable.relative_to(fixture['result'])), raw, anchor['schema']), raw)
    return fixture


def test_F1_actual_stream_delivery_cannot_spend_a_stale_reader_allowance(tmp_path, monkeypatch):
    fixture = _F1_remembered(tmp_path)
    trace, error = _F1_actual_delivery(fixture, monkeypatch, allowance=16000)
    assert error is not None
    assert not trace['before_effect_overshoot'], trace
    assert trace['ledger_delta'] == trace['reads']+trace['writes']
    assert trace['reserved'] == 0
    assert not (fixture['writer'].root / 'current-source.json').exists()


def _F1_close_status():
    return dict(terminal_tail='unknown', scalar_live_route='unavailable', scalar_live_reason='source_fixture',
        performance_status='incomplete', reason='tail_unknown')


@pytest.mark.parametrize('closed', [False, True])
def test_F1_actual_stream_healthy_and_closed_binding_share_exact_spent_ledger(tmp_path, monkeypatch, closed):
    import json
    fixture = _F1_remembered(tmp_path)
    options = dict(close_context=fixture['writer'].close_context(), capture_status=_F1_close_status()) if closed else {}
    trace, error = _F1_actual_delivery(fixture, monkeypatch, publish=lambda: fixture['capture'].publish(**options))
    assert error is None
    assert trace['reads'] > 0 and trace['writes'] > 0
    assert trace['ledger_delta'] == trace['reads']+trace['writes']
    assert trace['reserved'] == 0 and not trace['before_effect_overshoot']
    pointer = json.loads((fixture['writer'].root / 'current-source.json').read_bytes())
    index = json.loads((fixture['writer'].root / pointer['index']['path']).read_bytes())
    assert index['closed'] is closed
    assert (index['entries'][0]['telemetry']['binding_ref'] is not None) is closed
    fixture['writer'].close()


@pytest.mark.parametrize('closed', [False, True])
@pytest.mark.parametrize('short', [0, 1])
def test_F1_selected_entry_exact_or_one_byte_short_admission(tmp_path, monkeypatch, closed, short):
    fixture = _F1_remembered(tmp_path); capture = fixture['capture']
    context = fixture['writer'].close_context() if closed else fixture['writer'].published_context()
    key, row = next(iter(capture.anchors.items()))
    operation = lambda: capture._entry(key, row, context, closed, _F1_close_status() if closed else None)
    # Warm only immutable binding publication; the measured and boundary calls
    # each construct and close a fresh StateReader and consume the same originals.
    operation()
    measured, error = _F1_actual_delivery(fixture, monkeypatch, publish=operation)
    assert error is None and measured['ledger_delta'] == measured['reads']+measured['writes']
    allowance = measured['ledger_delta']-short
    trace, error = _F1_actual_delivery(fixture, monkeypatch, allowance=allowance, publish=operation)
    assert (error is None) is (short == 0)
    assert trace['ledger_delta'] == trace['reads']+trace['writes'] <= allowance
    assert trace['reserved'] == 0 and not trace['before_effect_overshoot']
    assert not (fixture['writer'].root / 'current-source.json').exists()


@pytest.mark.parametrize('fault', ['initialize', 'summary', 'binding_read', 'binding_write', 'cleanup'])
def test_F1_failure_preserves_primary_and_every_delivered_byte(tmp_path, monkeypatch, fault):
    from qcl_negf_contracts.messages import ContractError
    import qcl_negf_results.capture as capture
    import qcl_negf_results.state as state
    import qcl_negf_results.performance_views as views
    fixture = _F1_remembered(tmp_path)
    closed = fault.startswith('binding')
    options = dict(close_context=fixture['writer'].close_context(), capture_status=_F1_close_status()) if closed else {}
    primary = ContractError('F1 injected primary after delivered source bytes', 'source_changed')
    cleanup = OSError('F1 injected source cleanup')
    close = state.StateReader.close
    def failing_close(reader):
        close(reader)
        raise cleanup
    monkeypatch.setattr(state.StateReader, 'close', failing_close)
    if fault == 'initialize':
        initialize = state.StateReader._initialize
        def fail_initialize(reader, *args, **kwargs):
            initialize(reader, *args, **kwargs)
            raise primary
        monkeypatch.setattr(state.StateReader, '_initialize', fail_initialize)
    elif fault == 'summary':
        summary = views.read_performance_summary
        def fail_summary(*args, **kwargs):
            summary(*args, **kwargs)
            raise primary
        monkeypatch.setattr(views, 'read_performance_summary', fail_summary)
    elif fault == 'binding_read':
        load = views._load_telemetry
        def fail_binding(*args, **kwargs):
            answer = load(*args, **kwargs)
            if kwargs.get('creating_binding'):raise primary
            return answer
        monkeypatch.setattr(views, '_load_telemetry', fail_binding)
    elif fault == 'binding_write':
        write = capture.atomic_write
        def fail_write(path, payload, **kwargs):
            write(path, payload, **kwargs)
            if path.parent.name == 'bindings':raise primary
        monkeypatch.setattr(capture, 'atomic_write', fail_write)
    trace, error = _F1_actual_delivery(fixture, monkeypatch, publish=lambda: fixture['capture'].publish(**options))
    assert error is (cleanup if fault == 'cleanup' else primary)
    if fault != 'cleanup':assert any('F1 injected source cleanup' in note for note in primary.__notes__)
    assert trace['reads'] > 0
    assert trace['ledger_delta'] == trace['reads']+trace['writes']
    assert trace['reserved'] == 0 and not trace['before_effect_overshoot']
    assert not (fixture['writer'].root / 'current-source.json').exists()


@pytest.mark.parametrize('short', [0, 1])
def test_F1_new_closed_binding_exact_or_one_short_keeps_actual_spent_delta(tmp_path, monkeypatch, short):
    fixture = _F1_remembered(tmp_path); capture = fixture['capture']
    context = fixture['writer'].close_context()
    key, row = next(iter(capture.anchors.items()))
    def operation(reason):
        status = _F1_close_status(); status['reason'] = reason
        return lambda: capture._entry(key, row, context, True, status)
    # Distinct equal-width status values create new immutable close bindings.
    # No cache or previously published file is cleared between operations.
    operation('tail_unknown_a')()
    measured, error = _F1_actual_delivery(fixture, monkeypatch, publish=operation('tail_unknown_b'))
    assert error is None and measured['writes'] > 0
    assert measured['ledger_delta'] == measured['reads']+measured['writes']
    allowance = measured['ledger_delta']-short
    trace, error = _F1_actual_delivery(fixture, monkeypatch, allowance=allowance, publish=operation('tail_unknown_c'))
    assert (error is None) is (short == 0)
    assert trace['ledger_delta'] == trace['reads']+trace['writes'] <= allowance
    assert trace['reserved'] == 0 and not trace['before_effect_overshoot']
    assert capture.generation == 0
    assert not (fixture['writer'].root / 'current-source.json').exists()


def _F3_weight(roots):
    """Independent alias-aware live object census, not product credit or RSS."""
    import sys
    seen = set()
    def visit(value):
        if id(value) in seen:return 0
        seen.add(id(value)); size = sys.getsizeof(value)
        if type(value) is dict:
            return size + sum(visit(key)+visit(item) for key,item in value.items())
        if type(value) in (tuple,list):return size+sum(visit(item) for item in value)
        return size
    return visit(roots)


def _F3_many_sources(tmp_path, count=80):
    from copy import deepcopy
    from test_capture import _T3_capture, _T3_anchor
    import qcl_negf_results.capture as cap
    # Canonical existing physical paths remain legal; length must distinguish
    # the retained reviewer failure from a short /tmp near-miss.
    path = tmp_path / ('explicit-existing-mapping-' + 'p'*72)
    path.mkdir()
    f = _T3_capture(path, history_only=True)
    data = f['inputs'].document(); source = data['sources'][0]
    data['sources'] += [dict(deepcopy(source), artifact_root='unused-'+str(n)) for n in range(count-1)]
    try:inputs = cap.NativePerformanceInputs(**data)
    except BaseException:
        f['writer'].close()
        raise
    owner = cap.NativePerformanceCapture(f['job'],f['result'],writer=f['writer'],transport=f['transport'],inputs=inputs,budget=f['capture'].budget)
    a,raw,p = _T3_anchor(f)
    owner.remember(a,cap._capture_ref(str(p.relative_to(f['result'])),raw,a['schema']),raw)
    f['capture'] = owner; f['inputs'] = inputs
    return f


@pytest.mark.parametrize('closed', [False, True])
def test_F4_owned_boundary_forest_many_mappings_keeps_one_document_and_complete_index(tmp_path, monkeypatch, closed):
    import sys,json
    import qcl_negf_results.capture as cap
    f = _F3_many_sources(tmp_path); owner = f['capture']
    peaks = {}; documents = []
    old_trace = sys.gettrace()
    input_document = cap.NativePerformanceInputs.document
    def document(value):
        answer=input_document(value)
        if value is owner.inputs:documents.append(id(answer))
        return answer
    monkeypatch.setattr(cap.NativePerformanceInputs,'document',document)
    # New F4 identity: this observes capture-owned boundaries. It deliberately
    # makes no assertion about excluded transitive validator/reader frames.
    watched = {cap.__file__: ('publish','_entry','_capture_encode','_capture_write','_capture_read','_capture_read_admitted','_capture_document','make_binding','document','_capture_encoding_credit','_capture_small_json','_capture_decode','_capture_copy')}
    def trace(frame,event,arg):
        if frame.f_code.co_name not in watched.get(frame.f_code.co_filename,()):return trace
        roots = [owner.inputs._raw,owner.anchors,owner.transport]
        # Include selected reader documents, not the protected fixed library W.
        for name in ('_state_ref','_manifest','_pages','_index','_known_states','_catalog_by_hash','_published_paths','_catalog_by_leaf','_page_proofs','_parent_anchor_cache','_declarations','_verified_refs','_metadata_bindings','_checkpoint'):
            reader = frame.f_locals.get('reader')
            if reader is not None:roots.append(getattr(reader,name,None))
        current = frame
        active = False
        while current is not None:
            if current.f_code.co_name in watched.get(current.f_code.co_filename,()):
                local = current.f_locals
                if local.get('self') is owner or current.f_code.co_name in ('_capture_encode','_capture_write'):active=True
                for key,value in local.items():
                    if key in ('self','reader') or key.startswith('__'):continue
                    if type(value) in (dict,list,tuple,str,bytes,int,float,bool,type(None)):roots.append(value)
                    elif hasattr(value,'_raw'):roots.append(value._raw)
            current=current.f_back
        if active:
            if event=='return' and type(arg) in (dict,list,tuple,str,bytes):roots.append(arg)
            weight=_F3_weight(tuple(roots)); label=frame.f_code.co_name
            peaks[label]=max(peaks.get(label,0),weight)
        return trace
    options = dict(close_context=f['writer'].close_context(),capture_status=_F1_close_status()) if closed else {}
    try:
        sys.settrace(trace)
        reference=owner.publish(**options)
    finally:sys.settrace(old_trace)
    print('F4_CAPTURE_OBSERVED_BOUNDARY_FOREST_NOT_TRANSITIVE',json.dumps(dict(peaks=peaks,cap=cap.CAPTURE_METADATA_BYTES,source_mappings=80,anchors=len(owner.anchors),closed=closed),sort_keys=True))
    assert peaks and max(peaks.values()) <= cap.CAPTURE_METADATA_BYTES
    assert len(documents)==1
    assert {'_capture_encode','_capture_write','_entry','_capture_decode'} <= peaks.keys()
    if closed:assert 'make_binding' in peaks
    # One whole inputs decode; TelemetrySelection documents are separate small
    # contexts and must not restore another full source mapping inventory.
    assert len(owner.inputs.document()['sources']) == 80
    assert len(owner.anchors) == 1 and owner.generation == 1
    index=json.loads((f['writer'].root/reference['path']).read_bytes())
    assert len(index['entries'])==1 and index['closed'] is closed
    assert json.loads((f['writer'].root/'current-source.json').read_bytes())['index']==reference
    f['writer'].close()


@pytest.mark.parametrize('short',[0,1])
def test_F4_prospective_inputs_exact_or_one_short_never_decodes_on_refusal(tmp_path, monkeypatch, short):
    import qcl_negf_results.capture as cap
    f=_F3_many_sources(tmp_path);owner=f['capture'];inputs=owner.inputs
    retained=(owner.anchors,owner.transport)
    required=_F3_weight((inputs._raw,*retained))+inputs._decoded_bound
    assert required <= cap.CAPTURE_METADATA_BYTES
    original=cap.NativePerformanceInputs.document;calls=[]
    def document(value):calls.append(value);return original(value)
    monkeypatch.setattr(cap.NativePerformanceInputs,'document',document)
    if short:
        with pytest.raises(ValueError,match='metadata allowance'):
            cap._capture_document(inputs,*retained,maximum_bytes=required-1)
        assert not calls
    else:
        value=cap._capture_document(inputs,*retained,maximum_bytes=required)
        assert len(value['sources'])==80 and calls==[inputs]
        assert _F3_weight(value) <= inputs._decoded_bound
        # Public callers still receive detached documents; no operation cache
        # can turn one caller's mutation into another caller's assertions.
        value['sources'].clear()
        assert len(inputs.document()['sources'])==80
    assert owner.generation==0 and not (f['writer'].root/'current-source.json').exists()
    f['writer'].close()


@pytest.mark.parametrize('short',[0,1])
def test_F4_pointer_encoding_keeps_inventory_or_refuses_before_serializer(tmp_path, monkeypatch, short):
    import json
    import qcl_negf_results.capture as cap
    f=_F3_many_sources(tmp_path);owner=f['capture'];document=owner.inputs.document()
    value=dict(schema='qcl-negf.performance-source-pointer.v1',index=dict(path='sources/'+'a'*64+'.json',bytes='123',sha256='a'*64,schema='qcl-negf.performance-source-index.v1',media_type='application/json'))
    retained=(owner.inputs._raw,document,owner.anchors,owner.transport)
    encoded=json.dumps(value,ensure_ascii=False,allow_nan=False,sort_keys=True,separators=(',',':')).encode()
    import sys
    # F4 additionally prepays container capacities and active diagnostic paths.
    # A deliberately tiny remaining workspace must refuse before serialization;
    # a separate small literal fixture below controls the exact one-byte edge.
    required=cap.CAPTURE_METADATA_BYTES if not short else _F3_weight((value,*retained))+64
    calls=[];serializer=cap.json_bytes
    def serialize(item):calls.append(item);return serializer(item)
    monkeypatch.setattr(cap,'json_bytes',serialize)
    if short:
        with pytest.raises(ValueError,match='metadata allowance'):
            cap._capture_encode(value,*retained,maximum_bytes=required)
        assert not calls
    else:assert cap._capture_encode(value,*retained,maximum_bytes=required)==encoded and calls==[value]
    assert not (f['writer'].root/'current-source.json').exists()
    f['writer'].close()


@pytest.mark.parametrize('short',[0,1])
def test_F4_raw_read_exact_or_one_short_admits_before_open(tmp_path, monkeypatch, short):
    import qcl_negf_results.capture as cap
    from qcl_negf_results._capture_io import ObserverIO
    payload=b'bounded original performance metadata';(tmp_path/'bounded.json').write_bytes(payload)
    reference=cap._capture_ref('bounded.json',payload,'fixture')
    # Independent bytes+tuple+incoming bytes sizes give an exact common limit.
    padding=b'p'*(cap.CAPTURE_METADATA_BYTES-_F3_weight((b'',))-len(payload)-__import__('sys').getsizeof(b'')+short)
    calls=[];opened=cap.regular
    def regular(*a,**kw):calls.append(a);return opened(*a,**kw)
    monkeypatch.setattr(cap,'regular',regular)
    budget=ObserverIO(1024)
    if short:
        with pytest.raises(ValueError,match='metadata allowance'):
            cap._capture_read_admitted(tmp_path,reference,budget,retained=(padding,))
        assert not calls and budget.used==0
    else:
        assert cap._capture_read_admitted(tmp_path,reference,budget,retained=(padding,))==payload
        assert len(calls)==1 and budget.read_bytes==len(payload)


def test_F4_readback_memory_is_admitted_before_atomic_publication(tmp_path,monkeypatch):
    import qcl_negf_results.capture as cap
    from qcl_negf_results._capture_io import ObserverIO
    payload=b'original exact immutable bytes'
    # An existing retained forest leaves no room for publication + readback.
    padding=b'p'*(cap.CAPTURE_METADATA_BYTES-1024)
    calls=[]
    monkeypatch.setattr(cap,'atomic_write',lambda *a,**k:calls.append((a,k)))
    budget=ObserverIO(1024)
    with pytest.raises(ValueError,match='metadata allowance'):
        cap._capture_write(tmp_path,'current-source.json',payload,'fixture',budget,retained=(padding,))
    assert not calls and budget.used==0
    assert not (tmp_path/'current-source.json').exists()


@pytest.mark.parametrize('exhausted',['bytes','objects'])
def test_F4_exhausted_outer_credit_cannot_grant_a_new_positive_public_allowance(exhausted):
    import sys
    import qcl_negf_results.capture as cap
    if exhausted=='bytes':
        retained=(b'p'*(cap.CAPTURE_METADATA_BYTES-sys.getsizeof((b'',))-sys.getsizeof(b'')),)
        assert _F3_weight(retained)==cap.CAPTURE_METADATA_BYTES
    else:
        retained=(tuple(None for _ in range(16383)),)
        assert cap._capture_objects(retained)==16384 and _F3_weight(retained)<cap.MIB
    with pytest.raises(ValueError,match='remaining performance'):
        cap._capture_limits(retained)


# F4 is the explicitly narrower retained/owned-buffer operational contract.
# Historical F3 whole-transitive failures remain in their immutable packets.
def test_F4_retained_room_above_one_mib_does_not_enlarge_public_reader_limits():
    import qcl_negf_results.capture as cap
    retained=(b'p'*(2<<20),)
    cap._capture_credit(*retained)
    limits=cap._capture_limits(retained)
    assert limits.maximum_metadata_bytes==1<<20
    assert limits.maximum_objects<=16384


@pytest.mark.parametrize('operation',['decode','encode'])
def test_F4_deep_owned_codec_refuses_before_diagnostic_paths_are_allocated(tmp_path,monkeypatch,operation):
    import json
    import qcl_negf_results.capture as cap
    from qcl_negf_results._capture_io import ObserverIO
    value=0
    for _ in range(64):value={'k'*1000:value}
    raw=json.dumps(value,separators=(',',':')).encode()
    calls=[]
    if operation=='decode':
        path=tmp_path/'deep.json';path.write_bytes(raw)
        original=cap.decode_json
        def decode(*a,**k):calls.append(True);return original(*a,**k)
        monkeypatch.setattr(cap,'decode_json',decode)
        budget=ObserverIO(2<<20)
        with pytest.raises(ValueError,match='allowance'):
            cap._capture_small_json(path,budget,())
        assert budget.read_bytes==len(raw)
    else:
        original=cap.json_bytes
        def encode(v):calls.append(True);return original(v)
        monkeypatch.setattr(cap,'json_bytes',encode)
        with pytest.raises(ValueError,match='allowance'):cap._capture_encode(value)
    assert calls==[]


def test_F4_whitespace_refusal_spends_original_read_but_never_enters_decoder(tmp_path,monkeypatch):
    import qcl_negf_results.capture as cap
    from qcl_negf_results._capture_io import ObserverIO
    raw=b' '*(600000-7)+b'{"v":1}';path=tmp_path/'space.json';path.write_bytes(raw)
    calls=[];original=cap.decode_json
    def decode(*a,**k):calls.append(True);return original(*a,**k)
    monkeypatch.setattr(cap,'decode_json',decode)
    budget=ObserverIO(2<<20)
    with pytest.raises(ValueError,match='allowance'):
        cap._capture_small_json(path,budget,(b'p'*(3<<20),))
    assert calls==[] and budget.read_bytes==600000


def test_F4_manifest_declared_limit_is_checked_before_actual_reader_init(tmp_path,monkeypatch):
    from copy import deepcopy
    from test_capture import _T3_capture,_T3_anchor
    import qcl_negf_results.capture as cap
    from qcl_negf_results.state import StateReader
    f=_T3_capture(tmp_path,history_only=True);owner=f['capture']
    a,raw,p=_T3_anchor(f);owner.remember(a,cap._capture_ref(str(p.relative_to(f['result'])),raw,a['schema']),raw)
    key,row=next(iter(owner.anchors.items()));row=deepcopy(row)
    row['anchor']['scientific']['selected_state']['manifest']['bytes']=str((1<<20)+1)
    calls=[]
    def initialize(*a,**k):calls.append(True);raise RuntimeError('reader entered before manifest admission')
    monkeypatch.setattr(StateReader,'_initialize',initialize)
    try:
        with pytest.raises(ValueError,match='manifest'):
            owner._entry(key,row,f['writer'].published_context(),False,None)
        assert calls==[]
    finally:f['writer'].close()


def test_F4_actual_reader_init_receives_one_mib_metadata_limit(tmp_path,monkeypatch):
    from test_capture import _T3_capture,_T3_anchor
    import qcl_negf_results.capture as cap
    from qcl_negf_results.state import StateReader
    f=_T3_capture(tmp_path,history_only=True);owner=f['capture']
    a,raw,p=_T3_anchor(f);owner.remember(a,cap._capture_ref(str(p.relative_to(f['result'])),raw,a['schema']),raw)
    calls=[];original=StateReader._initialize
    def initialize(self,*a,**k):
        calls.append(k['limits'].maximum_metadata_bytes)
        assert k['limits'].maximum_metadata_bytes==1<<20
        return original(self,*a,**k)
    monkeypatch.setattr(StateReader,'_initialize',initialize)
    try:
        ref=owner.publish()
        assert calls and (f['writer'].root/ref['path']).is_file()
    finally:f['writer'].close()


def test_F4_pointer_encoding_failure_precedes_first_immutable_index_write(tmp_path,monkeypatch):
    from test_capture import _T3_capture,_T3_anchor
    import qcl_negf_results.capture as cap
    f=_T3_capture(tmp_path,history_only=True);owner=f['capture']
    a,raw,p=_T3_anchor(f);owner.remember(a,cap._capture_ref(str(p.relative_to(f['result'])),raw,a['schema']),raw)
    writes=[];write=cap._capture_write;encode=cap.json_bytes
    def observed_write(root,path,*a,**k):writes.append(path);return write(root,path,*a,**k)
    def fail_pointer(value):
        if type(value) is dict and value.get('schema')=='qcl-negf.performance-source-pointer.v1':
            raise ValueError('pointer encoding deliberately unavailable')
        return encode(value)
    monkeypatch.setattr(cap,'_capture_write',observed_write);monkeypatch.setattr(cap,'json_bytes',fail_pointer)
    try:
        with pytest.raises(ValueError,match='pointer encoding'):owner.publish()
        assert not any(p.startswith('sources/') for p in writes)
        assert not (f['writer'].root/'current-source.json').exists()
        assert owner.generation==0
    finally:f['writer'].close()


@pytest.mark.parametrize('short',[0,1])
@pytest.mark.parametrize('operation',['decode','encode'])
def test_F4_small_literal_codec_exact_and_one_byte_short(tmp_path,monkeypatch,short,operation):
    import sys
    import qcl_negf_results.capture as cap
    raw=b'{"v":1}';value={'v':1};header=sys.getsizeof('\U00010000')
    # Hand-derived literal fixture: raw7; root object+key+integer token326;
    # four JSON punctuation/header parser max5102; depth1/string chars1.
    if operation=='decode':
        workspace=5102+2*(header+40)+8+8192
        required=_F3_weight((raw,))+workspace
        original=cap.decode_json;calls=[]
        def decode(*a,**k):calls.append(True);return original(*a,**k)
        monkeypatch.setattr(cap,'decode_json',decode)
        invoke=lambda:cap._capture_decode(raw,maximum_bytes=required-short)
    else:
        # Literal one-key dict clone + one decoded key/int + key memo/largest
        # key. Runtime primitive headers differ between supported CPython builds.
        clone=2*sys.getsizeof(dict(v=None))+2*sys.getsizeof('v')+sys.getsizeof(1)
        # Active '$' and '$.v', plus one leaf diagnostic formatting string.
        paths=(header+4)+(header+12)+(header+4*67)+8192
        required=_F3_weight((value,))+clone+paths+7+header+sys.getsizeof(raw)
        original=cap.json_bytes;calls=[]
        def encode(v):calls.append(True);return original(v)
        monkeypatch.setattr(cap,'json_bytes',encode)
        invoke=lambda:cap._capture_encode(value,maximum_bytes=required-short)
    if short:
        with pytest.raises(ValueError,match='allowance'):invoke()
        assert calls==[]
    else:
        assert invoke()==(value if operation=='decode' else raw)
        assert calls==[True]


@pytest.mark.parametrize('kind',['reference','file','write','encode','inputs'])
def test_F4_public_one_mib_raw_limit_precedes_read_write_or_encoder(tmp_path,monkeypatch,kind):
    import qcl_negf_results.capture as cap
    from qcl_negf_results._capture_io import ObserverIO
    from test_capture import _T3_capture
    f=_T3_capture(tmp_path,history_only=True) if kind=='inputs' else None
    calls=[];original=cap.json_bytes
    def encode(v):calls.append('encode');return original(v)
    monkeypatch.setattr(cap,'json_bytes',encode)
    atomic=cap.atomic_write
    def write(*a,**k):calls.append('write');return atomic(*a,**k)
    monkeypatch.setattr(cap,'atomic_write',write)
    budget=ObserverIO(4<<20);raw=b'p'*((1<<20)+1)
    try:
        with pytest.raises(ValueError,match='(allowance|byte cap)'):
            if kind=='reference':
                cap._capture_read_admitted(tmp_path,cap._capture_ref('absent',raw,'fixture'),budget)
            elif kind=='file':
                path=tmp_path/'oversize.json';path.write_bytes(raw)
                cap._capture_small_json(path,budget,())
            elif kind=='write':cap._capture_write(tmp_path,'must-not-exist.json',raw,'fixture',budget)
            elif kind=='encode':cap._capture_encode({'v':'p'*(1<<20)})
            else:
                values=f['inputs'].document();values['sources'][0]['canonical_plan_directory']='/'+'p'*(1<<20)
                cap.NativePerformanceInputs(**values)
        assert calls==[] and budget.used==0
        assert not (tmp_path/'must-not-exist.json').exists()
    finally:
        if f is not None:f['writer'].close()


@pytest.mark.parametrize('kind',['ascii','whitespace','bmp','wide','many-small','malformed'])
def test_F4_owned_decoder_adversarial_controls_keep_honest_spent_io(tmp_path,monkeypatch,kind):
    import json
    import qcl_negf_results.capture as cap
    from qcl_negf_results._capture_io import ObserverIO
    raw={'ascii':b'{"v":"small"}', 'whitespace':b' '*599993+b'{"v":1}',
         'bmp':json.dumps({'v':'я'*100000},ensure_ascii=False).encode(),
         'wide':json.dumps({'v':'😀'*100000},ensure_ascii=False).encode(),
         'many-small':json.dumps({'v':[{'k':str(n)} for n in range(1000)]},separators=(',',':')).encode(),
         'malformed':b'{"v":[}'}[kind]
    path=tmp_path/'owned.json';path.write_bytes(raw);budget=ObserverIO(2<<20)
    calls=[];original=cap.decode_json
    def decode(*a,**k):calls.append(True);return original(*a,**k)
    monkeypatch.setattr(cap,'decode_json',decode)
    if kind in ('bmp','wide'):
        # Legal raw can exceed the conservative owned workspace. Early refusal
        # remains an honest result, never a whitespace/Unicode format ban.
        with pytest.raises(ValueError,match='allowance'):cap._capture_small_json(path,budget,())
        assert calls==[]
    elif kind=='malformed':
        with pytest.raises(ValueError,match='invalid complete'):cap._capture_small_json(path,budget,())
        assert calls==[True]
    else:
        value,original_raw=cap._capture_small_json(path,budget,())
        assert original_raw==raw and value==json.loads(raw) and calls==[True]
    assert budget.read_bytes==len(raw) and budget.reserved==0


@pytest.mark.parametrize('one_over',[False,True])
def test_F4_retained_object_exact_limit_or_one_over(one_over):
    import qcl_negf_results.capture as cap
    # The supplied roots tuple is traversal bookkeeping; one retained tuple and
    #16383 scalar occurrences form the existing public16384-object definition.
    retained=tuple(None for _ in range(16383))
    if one_over:
        with pytest.raises(ValueError,match='object cap'):cap._capture_credit(retained,extra_objects=1)
    else:cap._capture_credit(retained)


@pytest.mark.parametrize('case,count',[('normal',80),('long',80),('unicode',80),('many',256)])
def test_F4_real_mapping_paths_preserve_all_sources_or_refuse_honestly(tmp_path,case,count):
    import json
    import qcl_negf_results.capture as cap
    component='я'*40+'😀'*30 if case=='unicode' else 'p'*180 if case=='long' else 'normal'
    assert len(component.encode())<=255
    path=tmp_path/component;path.mkdir()
    if count==256:
        with pytest.raises(ValueError,match='(object cap|allowance)'):_F3_many_sources(path,count)
        print('F4_MAPPING_CONTROL',json.dumps(dict(case=case,sources=count,result='refused_full_inventory')))
        return
    f=_F3_many_sources(path,count);owner=f['capture']
    try:
        assert len(owner.inputs.document()['sources'])==count
        reference=owner.publish(close_context=f['writer'].close_context(),capture_status=_F1_close_status())
        index=json.loads((f['writer'].root/reference['path']).read_bytes())
        assert index['closed'] and len(index['entries'])==1
        assert (f['job']/'performance-source.json').read_bytes()==(f['writer'].root/reference['path']).read_bytes()
        assert len(owner.inputs.document()['sources'])==count
        print('F4_MAPPING_CONTROL',json.dumps(dict(case=case,sources=count,result='accepted_complete_inventory',index_bytes=int(reference['bytes']))))
    finally:f['writer'].close()


def test_F4_original_phase_zero_anchor_accepts_legal_whitespace_under_new_owned_budget(tmp_path):
    import json
    import qcl_negf_results.capture as cap
    from qcl_negf_results._capture_io import ObserverIO
    from test_capture import _T3_capture,_T3_anchor
    f=_T3_capture(tmp_path,history_only=True)
    try:
        anchor,original,_=_T3_anchor(f,pending=True)
        raw=b' '*(600000-len(original))+original
        reference=cap._capture_ref('performance-anchors/'+hashlib.sha256(raw).hexdigest()+'.json',raw,anchor['schema'])
        (f['result']/reference['path']).write_bytes(raw)
        (f['result']/'performance-anchor-current.json').write_bytes(raw)
        budget=ObserverIO(4<<20)
        checked,actual_ref,actual_raw=cap._performance_anchor(f['result'],budget=budget)
        assert checked==anchor and checked['scientific'] is None
        assert actual_ref==reference and actual_raw==raw and budget.read_bytes==1200000
        print('F4_NEW_CONTRACT_ANCHOR',json.dumps(dict(raw_bytes=len(raw),read_bytes=budget.read_bytes,
            qualification='owned_retained_buffers_only_not_historical_R2_or_RSS')))
    finally:f['writer'].close()


@pytest.mark.parametrize('operation',['read','decode','encode','write'])
def test_F4_prospective_owned_objects_refuse_before_effect_at_retained_edge(tmp_path,monkeypatch,operation):
    import qcl_negf_results.capture as cap
    from qcl_negf_results._capture_io import ObserverIO
    calls=[];budget=ObserverIO(4096);raw=b'{"v":1}'
    if operation=='read':
        retained=(tuple(None for _ in range(16383)),)
        original=cap.regular
        def opened(*a,**k):calls.append('open');return original(*a,**k)
        monkeypatch.setattr(cap,'regular',opened)
        invoke=lambda:cap._capture_read_admitted(tmp_path,cap._capture_ref('absent.json',raw,'fixture'),budget,retained=retained)
    elif operation=='decode':
        retained=(tuple(None for _ in range(16380)),)
        original=cap.decode_json
        def decode(*a,**k):calls.append('decode');return original(*a,**k)
        monkeypatch.setattr(cap,'decode_json',decode)
        invoke=lambda:cap._capture_decode(raw,*retained)
    elif operation=='encode':
        retained=(tuple(None for _ in range(16375)),)
        original=cap.json_bytes
        def encode(*a,**k):calls.append('encode');return original(*a,**k)
        monkeypatch.setattr(cap,'json_bytes',encode)
        invoke=lambda:cap._capture_encode({'v':1},*retained)
    else:
        retained=(tuple(None for _ in range(16376)),)
        original=cap.atomic_write
        def write(*a,**k):calls.append('write');return original(*a,**k)
        monkeypatch.setattr(cap,'atomic_write',write)
        invoke=lambda:cap._capture_write(tmp_path,'edge.json',raw,'fixture',budget,retained=retained)
    with pytest.raises(ValueError,match='object cap'):invoke()
    assert calls==[] and budget.used==0
    assert not (tmp_path/'edge.json').exists()


@pytest.mark.parametrize('closed', [False, True])
@pytest.mark.parametrize('short', [0, 1])
def test_F5_complete_publication_exact_io_or_one_short_precedes_index(tmp_path, monkeypatch, closed, short):
    import qcl_negf_results.capture as cap
    fixture = _F1_remembered(tmp_path)
    owner, writer = fixture['capture'], fixture['writer']
    options = dict(close_context=writer.close_context(), capture_status=_F1_close_status()) if closed else {}
    operation = lambda:owner.publish(**options)
    operation()  # Warm immutable source/binding objects, keeping every read charged.
    measured, error = _F1_actual_delivery(fixture, monkeypatch, publish=operation)
    assert error is None and measured['ledger_delta'] == measured['reads'] + measured['writes']
    pointer = (writer.root / 'current-source.json').read_bytes()
    generation = owner.generation
    effects = []; write = cap._capture_write
    def index_effect(root, path, raw, schema, budget, **kwargs):
        if path.startswith('sources/'):effects.append(path)
        return write(root, path, raw, schema, budget, **kwargs)
    try:
        with monkeypatch.context() as patch:
            patch.setattr(cap, '_capture_write', index_effect)
            trace, error = _F1_actual_delivery(fixture, patch, allowance=measured['ledger_delta']-short, publish=operation)
        assert (error is None) is (short == 0), trace
        assert trace['ledger_delta'] == trace['reads'] + trace['writes']
        assert trace['reserved'] == 0 and not trace['before_effect_overshoot']
        if short:
            assert effects == [] and owner.generation == generation
            assert (writer.root / 'current-source.json').read_bytes() == pointer
        else:
            assert len(effects) == 1 and owner.generation == generation + 1
        print('F5_EXACT_IO', dict(closed=closed, short=short, admitted=measured['ledger_delta']-short,
            delivered=trace['ledger_delta'], index_effects=len(effects)))
    finally:
        owner.budget.maximum = 128 << 20
        writer.close()


@pytest.mark.parametrize('closed', [False, True])
def test_F5_before_index_admission_dominates_every_owned_suffix_check(tmp_path, monkeypatch, closed):
    import qcl_negf_results.capture as cap
    from qcl_negf_results.state import _metadata_weight
    fixture = _F3_many_sources(tmp_path, 80)
    owner, writer = fixture['capture'], fixture['writer']
    options = dict(close_context=writer.close_context(), capture_status=_F1_close_status()) if closed else {}
    phase = dict(started=False, preflight=None); observed = []
    credit, write = cap._capture_credit, cap._capture_write
    def admitted(*values, extra_bytes=0, extra_objects=0, maximum_bytes=cap.CAPTURE_METADATA_BYTES):
        prospective = (_metadata_weight(values)+extra_bytes, cap._capture_objects(values)+extra_objects)
        credit(*values, extra_bytes=extra_bytes, extra_objects=extra_objects, maximum_bytes=maximum_bytes)
        if phase['started']:observed.append(prospective)
        else:phase['preflight'] = prospective
    def suffix(root, path, raw, schema, budget, **kwargs):
        if path.startswith('sources/'):phase['started'] = True
        return write(root, path, raw, schema, budget, **kwargs)
    try:
        with monkeypatch.context() as patch:
            patch.setattr(cap, '_capture_credit', admitted)
            patch.setattr(cap, '_capture_write', suffix)
            owner.publish(**options)
        assert observed and phase['preflight'] is not None
        maximum = tuple(max(row[column] for row in observed) for column in (0, 1))
        assert all(actual <= admitted for actual, admitted in zip(maximum, phase['preflight']))
        print('F5_OWNED_SUFFIX_DOMINANCE', dict(closed=closed, source_mappings=80,
            preflight=phase['preflight'], actual_checks_maximum=maximum,
            qualification='capture_retained_and_explicit_owned_checks_only'))
    finally:writer.close()


@pytest.mark.parametrize('raw_bytes,admitted', [(200000, True), (600000, False)])
def test_N3_original_plan_role_keeps_public_cap_and_honest_private_admission(tmp_path, raw_bytes, admitted):
    import qcl_negf_results.capture as cap
    from qcl_negf_results._capture_io import ObserverIO
    from test_capture import _N3_fixture
    f=_N3_fixture(tmp_path);raw=(f['job']/'plan.json').read_bytes()
    raw=b' '*(raw_bytes-len(raw))+raw
    (f['job']/'plan.json').write_bytes(raw)
    f['spec']['plan'].update(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
    f['native']['input_plan'].update(bytes=str(len(raw)),sha256=hashlib.sha256(raw).hexdigest())
    (f['job']/'native-execution.json').write_bytes(cap.json_bytes(f['native']))
    (f['job']/'capture-spec.json').write_bytes(cap.json_bytes(f['spec']))
    budget=ObserverIO(16384)
    if not admitted:
        with pytest.raises(ValueError,match='metadata allowance'):
            cap._spec(f['job']/'capture-spec.json',budget=budget)
        assert budget.read_bytes>=raw_bytes and not (f['result']/'performance').exists()
        return
    root,spec,_=cap._spec(f['job']/'capture-spec.json',budget=budget)
    output,transport,inputs=cap._native_capture_inputs(root,spec,f['job']/'native-execution.json',f['arguments'],budget=budget)
    assert output==f['result'] and transport['frozen_plan']['bytes']==str(raw_bytes)
    assert inputs.document()['native_execution_uuid']==f['native']['storage']['execution_binding_uuid']
    assert budget.read_bytes>2*raw_bytes and budget.reserved==0
    assert not (f['result']/'performance').exists()


def test_N3_guard_credit_denial_and_final_close_use_one_spent_ledger(tmp_path):
    import qcl_negf_results.capture as cap
    from qcl_negf_results._capture_io import ObserverIO
    from test_capture import _N3_fixture
    from pathlib import Path
    f=_N3_fixture(tmp_path);budget=ObserverIO(128 << 20)
    before=len(list(Path('/proc/self/fd').iterdir()))
    guard=cap._NativeResultGuard(f['job'],f['result'],f['birth'],os.getpid(),cap._start_ticks(os.getpid()),budget)
    guard.create_technical();spent=budget.used
    # The reserved final validation is not offered to another effect.
    budget.maximum=budget.used+budget.reserved
    effects=[]
    try:
        with pytest.raises(ValueError,match='credit'):
            with guard.effect():effects.append('unadmitted effect')
        assert effects==[] and budget.used==spent
    finally:guard.close()
    assert budget.used>spent and budget.reserved==0
    assert len(list(Path('/proc/self/fd').iterdir()))==before
