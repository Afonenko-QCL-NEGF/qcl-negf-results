"""Exact raw close binding, independent native source attempts, literal CPU oracle."""

import copy
import hashlib

import pytest
from qcl_negf_contracts.messages import ContractError

from qcl_negf_results.capture import make_binding
from qcl_negf_results.commits import json_bytes
from qcl_negf_results.performance_views import (
    PerformanceLimits,
    _bounded_objects,
    read_performance_summary,
    validate_performance_binding,
)
from qcl_negf_results.telemetry import CpuCoverage, TelemetryWriter

SELECTION = {
    "calcjob_uuid": "job3",
    "execution_id": "ex",
    "attempt": 3,
    "plan_fingerprint": "a" * 64,
}


def test_T10_performance_input_is_detached_and_original_source_transport_domains_stay_distinct(tmp_path):
    from native_consumer_fixture import technical_source
    from result_v2_fixture import reference
    from qcl_negf_results.performance_views import TelemetrySelection
    from qcl_negf_results.state import StateReader
    c=technical_source(tmp_path)
    writer=TelemetryWriter(c['technical'],{}, {},transport=c['transport'])
    for stamp,cpu in ((0,1.),(2_000_000_000,4.)):
        writer.record_resource(dict(monotonic_ns=stamp,wall_seconds=stamp/1e9,cpu_time_seconds=cpu,cpu_count=2,
            allocation_id='alloc',execution_id='ex',pid=42,process_start_ticks=7,cgroup_scope='root_julia_process_including_threads'))
    closed=writer.close();commitref=reference(c['technical'],str(closed.relative_to(c['technical'])),'qcl-negf.telemetry-commit.v1')
    transport=copy.deepcopy(c['transport'])
    selection=TelemetrySelection(c['technical'],commit_ref=commitref,source_binding_ref=c['source_binding'],
        expected_transport=transport,source_transport=c['source_transport'],canonical_plan_root=c['root'],
        canonical_plan_ref=c['plan'],expected_producer_provenance=c['producer_provenance'])
    from native_consumer_fixture import retain_close_binding
    selection=retain_close_binding(c,selection)
    transport['selection']['attempt']=999
    exposed=selection.expected_transport;exposed['selection']['attempt']=888
    assert selection.expected_transport['selection']['attempt']==3
    with StateReader(c['science'],state_ref=c['state'],expected_identity=c['state']['identity']) as reader:
        view=read_performance_summary(reader,telemetry=selection)
        assert view['source_state']['identity']['attempt']=='2'
        assert view['transport']['selection']['attempt']==3
        assert type(view['transport']['selection']['attempt']) is int
        assert view['summary']['cpu_coverage']['allocations'][0]['cpu_seconds']==3.
        assert view['summary']['cpu_coverage']['allocations'][0]['allocated_cpu_seconds']==4.
        assert view['summary']['cpu_coverage']['allocations'][0]['fraction_of_observed_capacity']==.75
        assert view['coverage']['total_scheduler_grant']=='not_measured'
        assert view['integrity_scope']=='selected_bytes_and_supplied_context_relations'
        assert read_performance_summary(reader,telemetry=None)['reason']=='source_binding_missing'


def test_close_return_raw_bytes_not_newer_session_and_literal_cpu_coverage(tmp_path):
    from native_consumer_fixture import technical_source
    from result_v2_fixture import reference
    from qcl_negf_results.performance_views import TelemetrySelection
    from qcl_negf_results.state import StateReader
    c=technical_source(tmp_path)
    sentinel=(c['science']/c['state']['manifest']['path']).read_bytes()
    writer=TelemetryWriter(c['technical'],{}, {},transport=c['transport'])
    for stamp,cpu in ((0,1.),(2_000_000_000,4.)):
        writer.record_resource(dict(monotonic_ns=stamp,wall_seconds=stamp/1e9,cpu_time_seconds=cpu,cpu_count=2,
            allocation_id='alloc',execution_id='ex',pid=42,process_start_ticks=7,cgroup_scope='root_julia_process_including_threads'))
    closed=writer.close();raw=closed.read_bytes()
    newer=TelemetryWriter(c['technical'],{}, {},transport=dict(c['transport'],selection=dict(c['transport']['selection'],calcjob_uuid='unrelated')))
    newer.close()
    cref=reference(c['technical'],str(closed.relative_to(c['technical'])),'qcl-negf.telemetry-commit.v1')
    def selected(binding_ref=None):
        return TelemetrySelection(c['technical'],commit_ref=cref,source_binding_ref=c['source_binding'],
            expected_transport=c['transport'],source_transport=c['source_transport'],canonical_plan_root=c['root'],canonical_plan_ref=c['plan'],
            expected_producer_provenance=c['producer_provenance'],binding_ref=binding_ref)
    with StateReader(c['science'],state_ref=c['state'],expected_identity=c['state']['identity']) as reader:
        sidecar=c['technical']/'binding.json'
        value=make_binding(reader,telemetry=selected(),destination=sidecar,
            capture_status=dict(terminal_tail='unknown',scalar_live_route='direct',scalar_live_reason=None,performance_status='available',reason=None))
        assert value['commit']['raw_utf8'].encode()==raw
        assert value['commit']['ref']==cref
        assert (c['science']/c['state']['manifest']['path']).read_bytes()==sentinel
        binding_ref=reference(c['technical'],'binding.json','qcl-negf.telemetry-binding.v1')
        view=read_performance_summary(reader,telemetry=selected(binding_ref))
        coverage=view['summary']['cpu_coverage']['allocations'][0]
        assert coverage['cpu_seconds']==3.0
        assert coverage['allocated_cpu_seconds']==4.0
        assert coverage['fraction_of_observed_capacity']==0.75
        with pytest.raises(ContractError):
            read_performance_summary(reader,telemetry=selected(binding_ref),limits=PerformanceLimits(maximum_objects=1))
        assert view['scope']=='execution_attempt'
        assert view['source_state']['identity']['attempt']=='2'
        assert selected().source_transport['selection']['attempt']==2
        assert view['transport']['selection']['attempt']==3
        assert view['coverage']['total_scheduler_grant']=='not_measured'
        assert read_performance_summary(reader,telemetry=None)['reason']=='source_binding_missing'
        for mutate in ('foreign','summary','point','result'):
            bad=copy.deepcopy(value)
            if mutate=='foreign':bad['transport']['selection']['calcjob_uuid']='job2'
            if mutate=='summary':bad['summary']['raw_utf8']+=' '
            if mutate=='point':bad['selected_state']['identity']['attempt']='3'
            if mutate=='result':
                native=__import__('json').loads(bad['source_binding']['raw_utf8'])
                native['selected_state']['manifest']['sha256']='c'*64
                bad['source_binding']['raw_utf8']=__import__('json').dumps(native)
            sidecar.write_bytes(json_bytes(bad))
            changed=reference(c['technical'],'binding.json','qcl-negf.telemetry-binding.v1')
            with pytest.raises(ContractError):read_performance_summary(reader,telemetry=selected(changed))
        sidecar.write_bytes(b'{')
        with pytest.raises(ContractError):read_performance_summary(reader,telemetry=selected(binding_ref))


def test_three_metadata_roots_share_one_object_budget():
    # Each literal root has exactly three nodes: dict, list, scalar.
    roots = ({"a": [1]}, {"b": [2]}, {"c": [3]})
    _bounded_objects(roots, 9)
    with pytest.raises(ContractError):
        _bounded_objects(roots, 8)


def test_measured_cpu_survives_unknown_capacity_and_missing_is_not_zero():
    coverage = CpuCoverage()
    for stamp, cpu in ((0, 1.0), (2_000_000_000, 4.0)):
        coverage.record({
            "allocation_id": "a", "execution_id": "ex", "pid": 42,
            "process_start_ticks": 7, "clock_domain_id": "clock",
            "monotonic_ns": stamp, "cpu_time_seconds": cpu, "cpu_count": None,
        })
    row = coverage.summary()["allocations"][0]
    assert row["cpu_seconds"] == 3.0 and row["mean_busy_logical_cpus"] == 1.5
    assert row["allocated_cpu_seconds"] is None
    assert row["fraction_of_observed_capacity"] is None
    assert row["capacity_unavailable_reason"]
    missing = CpuCoverage()
    missing.record({"allocation_id": "a", "monotonic_ns": 0, "cpu_time_seconds": None})
    row = missing.summary()["allocations"][0]
    assert row["cpu_seconds"] is None and row["cpu_unavailable_reason"]


@pytest.mark.parametrize('fault',['replace','grow','cancel'])
def test_T10_actual_metadata_read_boundary_stability_and_baseexception_unwind(tmp_path,monkeypatch,fault):
    from contextlib import contextmanager
    from pathlib import Path
    import os
    import qcl_negf_results.performance_views as module
    from native_consumer_fixture import technical_source
    from result_v2_fixture import reference
    from qcl_negf_results.state import StateReader
    c=technical_source(tmp_path)
    writer=TelemetryWriter(c['technical'],{},{},transport=c['transport']);closed=writer.close()
    cref=reference(c['technical'],str(closed.relative_to(c['technical'])),'qcl-negf.telemetry-commit.v1')
    selection=module.TelemetrySelection(c['technical'],commit_ref=cref,source_binding_ref=c['source_binding'],
        expected_transport=c['transport'],source_transport=c['source_transport'],canonical_plan_root=c['root'],canonical_plan_ref=c['plan'],
        expected_producer_provenance=c['producer_provenance'])
    from native_consumer_fixture import retain_close_binding
    selection=retain_close_binding(c,selection)
    original=module.regular;target=c['technical']/c['source_binding']['path'];seen=[];active=[]
    @contextmanager
    def injected(path,**kwargs):
        with original(path,**kwargs) as stream:
            active.append(stream.fileno())
            class Proxy:
                def fileno(self):return stream.fileno()
                def read(self,count):
                    raw=stream.read(count)
                    if Path(path)==target and not seen:
                        seen.append(True)
                        if fault=='cancel':raise KeyboardInterrupt('literal cancellation')
                        if fault=='replace':
                            changed=target.with_suffix('.new');changed.write_bytes(raw);os.replace(changed,target)
                        else:
                            with target.open('ab') as out:out.write(b' ')
                    return raw
            try:yield Proxy()
            finally:active.pop()
    with StateReader(c['science'],state_ref=c['state'],expected_identity=c['state']['identity']) as reader:
        previous_limits=reader.limits
        with monkeypatch.context() as patch:
            patch.setattr(module,'regular',injected)
            with pytest.raises(KeyboardInterrupt if fault=='cancel' else ContractError) as error:
                read_performance_summary(reader,telemetry=selection)
        assert seen and active==[] and reader._shared_budget is None
        assert reader.limits==previous_limits and reader._budget._source_guard is None
        if fault!='cancel':assert error.value.code=='source_changed'
        if fault=='cancel':assert read_performance_summary(reader,telemetry=selection)['status']=='available'


def test_T10_publication_transition_after_resource_only_predecessor_requires_exact_old_completion_input(tmp_path):
    import json
    from native_consumer_fixture import technical_source,telemetry_input,sealed_phase
    from qcl_negf_results.capture import PhaseCaptureSource
    from qcl_negf_results.state import StateReader,_IOBudget
    c=technical_source(tmp_path/'source');producer=tmp_path/'producer'
    first=sealed_phase(producer,c['state']['identity'])
    descriptor=dict(status='available',session_id='native',current_path='phase-current.json',reason=None)
    with PhaseCaptureSource(producer,descriptor=descriptor,expected_identity=c['state']['identity'],route_epoch=0) as phase:
        batch=phase.harvest(None,budget=_IOBudget(1024*1024))
        writer=TelemetryWriter(c['technical'],{}, {},transport=c['transport'])
        writer.stage_phase_batch(batch);accepted=json.loads(writer.flush().read_bytes())['accepted_ranges'][0]
        writer.record_resource(dict(wall_seconds=1.));writer.flush()  # predecessor has no range context
        second=sealed_phase(producer,c['state']['identity'],previous=first['publication_ref'],generation=2)
        batch2=phase.harvest(batch.cursor,budget=_IOBudget(1024*1024))
        writer.stage_phase_batch(batch2);writer.flush()
        old=dict(publication_ref=accepted['publication'],segment_ref=accepted['segment'],
            retained_publication_ref=accepted['retained_publication'],retained_segment_ref=accepted['retained_segment'])
        with StateReader(c['science'],state_ref=c['state'],expected_identity=c['state']['identity']) as reader:
            with pytest.raises(ContractError,match='completion proof'):
                read_performance_summary(reader,telemetry=telemetry_input(c,writer))
            result=read_performance_summary(reader,telemetry=telemetry_input(c,writer,phase_context_refs=(old,)))
            assert result['summary']['totals']['accepted_source_rows']=='4'
            assert result['integrity']['commit_delta_contexts']=='public_validator_exact_current_delta'
            for field in ('publication_ref','retained_segment_ref'):
                bad=copy.deepcopy(old);bad[field]['sha256']='0'*64
                with pytest.raises(ContractError):
                    read_performance_summary(reader,telemetry=telemetry_input(c,writer,phase_context_refs=(bad,)))
            with pytest.raises(ContractError,match='duplicate'):
                read_performance_summary(reader,telemetry=telemetry_input(c,writer,phase_context_refs=(old,old)))
        writer.close()


def test_T10_closed_selected_summary_requires_exact_close_binding(tmp_path):
    from native_consumer_fixture import technical_source,telemetry_input
    from qcl_negf_results.state import StateReader
    c=technical_source(tmp_path);writer=TelemetryWriter(c['technical'],{}, {},transport=c['transport']);writer.close()
    with StateReader(c['science'],state_ref=c['state'],expected_identity=c['state']['identity']) as reader:
        with pytest.raises(ContractError,match='closed.*binding'):
            read_performance_summary(reader,telemetry=telemetry_input(c,writer,bind_closed=False))


def test_T0_raw_context_roundtrip_keeps_old_completion_resume_and_original_bytes(tmp_path):
    """Removing any consumed original context must prevent a portable delta proof."""
    import json
    from native_consumer_fixture import technical_source, sealed_phase, retain_close_binding
    from result_v2_fixture import reference
    from qcl_negf_results.capture import PhaseCaptureSource
    from qcl_negf_results.performance_views import TelemetrySelection, read_performance_context
    from qcl_negf_results.state import StateReader, _IOBudget
    c = technical_source(tmp_path / 'source')
    producer = tmp_path / 'producer'
    first = sealed_phase(producer, c['state']['identity'])
    descriptor = dict(status='available', session_id='native', current_path='phase-current.json', reason=None)
    with PhaseCaptureSource(producer, descriptor=descriptor, expected_identity=c['state']['identity'], route_epoch=0) as phase:
        batch = phase.harvest(None, budget=_IOBudget(1 << 20))
        writer = TelemetryWriter(c['technical'], {}, {}, transport=c['transport'])
        writer.stage_phase_batch(batch)
        accepted = json.loads(writer.flush().read_bytes())['accepted_ranges'][0]
        old_path = writer.rotate()
        old = json.loads(old_path.read_bytes())
        resume = dict(commit_ref=reference(c['technical'], str(old_path.relative_to(c['technical'])), 'qcl-negf.telemetry-commit.v1'),
                      summary_ref=old['summary'])
        completion = dict(publication_ref=accepted['publication'], segment_ref=accepted['segment'],
                          retained_publication_ref=accepted['retained_publication'], retained_segment_ref=accepted['retained_segment'])
        writer.record_resource(dict(wall_seconds=1.))
        writer.flush()  # Immediate predecessor does not carry the old source.
        sealed_phase(producer, c['state']['identity'], previous=first['publication_ref'], generation=2)
        returning = phase.harvest(batch.cursor, budget=_IOBudget(1 << 20))
        writer.stage_phase_batch(returning, phase_context_refs=(completion,), resume_context_refs=(resume,))
        close_ref, phases, resumes = writer.close_context()
    assert phases == (completion,) and resumes == (resume,)
    selection = TelemetrySelection(c['technical'], commit_ref=close_ref, source_binding_ref=c['source_binding'],
        expected_transport=c['transport'], source_transport=c['source_transport'], canonical_plan_root=c['root'],
        canonical_plan_ref=c['plan'], expected_producer_provenance=c['producer_provenance'],
        phase_context_refs=phases, resume_context_refs=resumes)
    selection = retain_close_binding(c, selection)
    roots = dict(scientific=c['science'], canonical_plan=c['root'], technical=c['technical'])
    sentinels = {scope: {str(p.relative_to(root)): p.read_bytes() for p in root.rglob('*') if p.is_file()}
                 for scope, root in roots.items()}
    with StateReader(c['science'], state_ref=c['state'], expected_identity=c['state']['identity']) as reader:
        assertion_raw, files = read_performance_context(reader, telemetry=selection)
        view = read_performance_summary(reader, telemetry=selection)
        assert view['summary']['totals']['accepted_source_rows'] == '4'
        with pytest.raises(ContractError):
            read_performance_context(reader, telemetry=selection, limits=PerformanceLimits(maximum_metadata_bytes=1))
        with pytest.raises(ContractError):
            read_performance_context(reader, telemetry=selection, limits=PerformanceLimits(maximum_objects=1))
        with pytest.raises(ContractError):
            read_performance_context(reader, telemetry=selection, limits=PerformanceLimits(maximum_sessions=1))
    assert json.loads(assertion_raw) == selection.document()
    assert len(files) == 14
    assert [scope for scope, _, _ in files].count('technical') == 12
    seen = set()
    for scope, ref, raw in files:
        assert (scope, ref['path']) not in seen
        seen.add((scope, ref['path']))
        assert raw == (roots[scope] / ref['path']).read_bytes()
        assert len(raw) == int(ref['bytes']) and hashlib.sha256(raw).hexdigest() == ref['sha256']
        target = tmp_path / 'offline' / scope / ref['path']
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
    offline = TelemetrySelection(tmp_path / 'offline' / 'technical',
        canonical_plan_root=tmp_path / 'offline' / 'canonical_plan', **json.loads(assertion_raw))
    with StateReader(tmp_path / 'offline' / 'scientific', state_ref=c['state'], expected_identity=c['state']['identity']) as reader:
        assert read_performance_summary(reader, telemetry=offline)['summary'] == view['summary']
        missing = tmp_path / 'offline' / 'technical' / completion['retained_segment_ref']['path']
        original = missing.read_bytes()
        missing.unlink()
        with pytest.raises(ContractError):
            read_performance_context(reader, telemetry=offline)
        missing.write_bytes(original + b' ')
        with pytest.raises(ContractError):
            read_performance_context(reader, telemetry=offline)
    # Mutating detached output refs does not change assertions or the next read.
    last_ref = copy.deepcopy(files[-1][1])
    files[-1][1]['sha256'] = 'f' * 64
    with StateReader(c['science'], state_ref=c['state'], expected_identity=c['state']['identity']) as reader:
        assert read_performance_context(reader, telemetry=selection)[1][-1][1] == last_ref
    for scope, root in roots.items():
        assert {str(p.relative_to(root)): p.read_bytes() for p in root.rglob('*') if p.is_file()} == sentinels[scope]


@pytest.mark.parametrize('fault', ['wrong_ref', 'late_change', 'cancel', 'io', 'response'])
def test_T0_raw_context_wrong_ref_late_copy_mutation_and_budgets_do_not_return_partial(tmp_path, monkeypatch, fault):
    from native_consumer_fixture import technical_source, telemetry_input
    from qcl_negf_results.state import StateReader
    import qcl_negf_results.performance_views as module
    c = technical_source(tmp_path)
    writer = TelemetryWriter(c['technical'], {}, {}, transport=c['transport'])
    writer.close()
    selection = telemetry_input(c, writer)
    target = c['technical'] / c['source_binding']['path']
    original = target.read_bytes()
    changed = []
    limits = PerformanceLimits()
    if fault == 'wrong_ref':
        values = selection.document()
        values['source_binding_ref']['sha256'] = 'f' * 64
        selection = module.TelemetrySelection(selection.artifact_root, canonical_plan_root=selection.canonical_plan_root, **values)
    if fault in ('late_change', 'cancel'):
        clone = module.deepcopy
        def change_at_return_copy(value):
            result = clone(value)
            if value == selection.binding_ref and not changed:
                changed.append(True)
                if fault == 'cancel':
                    raise KeyboardInterrupt('T0 original return cancellation')
                target.write_bytes(original + b' ')
            return result
        monkeypatch.setattr(module, 'deepcopy', change_at_return_copy)
    if fault == 'io':
        limits = PerformanceLimits(maximum_io_bytes=1)
    if fault == 'response':
        limits = PerformanceLimits(maximum_json_bytes=1)
    with StateReader(c['science'], state_ref=c['state'], expected_identity=c['state']['identity']) as reader:
        with pytest.raises(KeyboardInterrupt if fault == 'cancel' else ContractError) as failure:
            module.read_performance_context(reader, telemetry=selection, limits=limits)
        if fault == 'late_change':
            assert changed and failure.value.code == 'source_changed'
        if fault == 'cancel':
            assert str(failure.value) == 'T0 original return cancellation'
            assert changed and target.read_bytes() == original
        if fault in ('io', 'response'):
            assert failure.value.code == 'budget_exceeded'
        assert reader._shared_budget is None and reader._budget._source_guard is None


def test_T0_F1_object_credit_traverses_nested_raw_tuples():
    # Four nodes in each root: dict/list/two values, tuple/dict/value/raw.
    roots = ({'kept': [1, 2]}, ({'path': 'x'}, b'raw'))
    with pytest.raises(ContractError) as failure:
        _bounded_objects(roots, 7)
    assert failure.value.code == 'budget_exceeded'
    _bounded_objects(roots, 8)


def test_T0_F1_return_object_credit_is_reserved_before_detached_copy(tmp_path, monkeypatch):
    from native_consumer_fixture import technical_source, telemetry_input
    from qcl_negf_results.state import StateReader
    import qcl_negf_results.performance_views as module

    c = technical_source(tmp_path)
    writer = TelemetryWriter(c['technical'], {}, {}, transport=c['transport'])
    writer.close()
    selection = telemetry_input(c, writer)
    observed = []
    original_check = module._bounded_objects

    def observe(roots, maximum):
        # Independent traversal of the contract's object grammar, not the guard.
        pending = list(roots)
        count = 0
        while pending:
            value = pending.pop()
            count += 1
            if isinstance(value, dict):
                pending.extend(value.values())
            elif isinstance(value, (list, tuple)):
                pending.extend(value)
        observed.append(count)
        return original_check(roots, maximum)

    monkeypatch.setattr(module, '_bounded_objects', observe)
    with StateReader(c['science'], state_ref=c['state'], expected_identity=c['state']['identity']) as reader:
        healthy = module.read_performance_context(reader, telemetry=selection)
        loader_objects = observed[0]
        # Two output roots. Each row is tuple + scope + five-field ref + raw:
        # 1 + 1 + (1 + 5) + 1 = nine nodes, independent of file contents.
        return_objects = 2 + 9 * len(healthy[1])
        sufficient = loader_objects + return_objects
        assert sufficient <= 4096
        copied = []
        original_copy = module.deepcopy

        def unexpected_copy(value):
            copied.append(value)
            raise AssertionError('return copy occurred before object-credit admission')

        monkeypatch.setattr(module, 'deepcopy', unexpected_copy)
        with pytest.raises(ContractError) as failure:
            module.read_performance_context(reader, telemetry=selection,
                limits=PerformanceLimits(maximum_objects=sufficient - 1))
        assert failure.value.code == 'budget_exceeded'
        assert copied == []
        monkeypatch.setattr(module, 'deepcopy', original_copy)
        assert module.read_performance_context(reader, telemetry=selection,
            limits=PerformanceLimits(maximum_objects=sufficient)) == healthy


# S2 wire fixtures emit no v2 writer operations; the existing v1 writer supplies
# only unchanged local-field grammar. All v2 transitions/expected values below
# are literal manufactured records, not production writer output or authority.
_S2_FLOATS=('probe_seconds','read_seconds','hash_seconds','parse_validate_seconds',
    'serialize_seconds','parquet_seconds','flush_sync_seconds','process_cpu_seconds')
_S2_COUNTS=('bytes_read','bytes_written','physical_read_bytes','physical_write_bytes',
    'polls','flushes','source_reopens','reverified_bytes')


def _s2_acc(known=None,covered='0',unavailable='0',reasons=()):
    return dict(known=known,covered=covered,unavailable=unavailable,overflow='0',reasons=list(reasons))


def s2_wire_case(tmp_path,*,generations=3,parquet=False,closed=True,declared_rows=2,parquet_files=1,malformed_parquet=False,checkpoint=False):
    import json
    from native_consumer_fixture import technical_source
    from result_v2_fixture import put,reference
    from qcl_negf_results.performance_views import TelemetrySelection
    from qcl_negf_contracts.telemetry import validate_telemetry_commit
    c=technical_source(tmp_path/'source',checkpoint=checkpoint)
    local=TelemetryWriter(tmp_path/'legacy-local-shape',{}, {},transport=c['transport']);local.close()
    template=copy.deepcopy(local._previous_summary)
    del template['baseline'];template['schema']='qcl-negf.telemetry-summary.v2';template['coverage']['detail_scope']='collector_session_local'
    leaves=[]
    if parquet:
        import pyarrow as pa
        import pyarrow.parquet as pq
        from qcl_negf_contracts.telemetry import TABLE_FIELDS,TELEMETRY_SCHEMA
        types=dict(string=pa.string(),int64=pa.int64(),float64=pa.float64(),bool=pa.bool_())
        schema=pa.schema([pa.field(k,types[t]) for k,t in TABLE_FIELDS['resources']],metadata={b'schema':TELEMETRY_SCHEMA.encode(),b'contract_set':b'qcl-negf.results.v2',b'table':b'resources'})
        for number in range(parquet_files):
            name=f'retained/resources-{number}.parquet';target=c['technical']/name;target.parent.mkdir(parents=True,exist_ok=True)
            pq.write_table(pa.Table.from_pylist([dict(rss_bytes=100+number,cpu_time_seconds=-0.0),dict(rss_bytes=101+number,cpu_time_seconds=1.0)],schema=schema),target)
            if malformed_parquet:target.write_bytes(b'PAR1x'+(1).to_bytes(4,'little')+b'PAR1')
            leaf=dict(ref=reference(c['technical'],name,TELEMETRY_SCHEMA,'application/vnd.apache.parquet'),role='performance.full',table='resources',rows=str(declared_rows),accepted=[])
            leaves.append(leaf)
        template['rows']['resources']=template['totals']['rows']['resources']=str(declared_rows*parquet_files)
    previous=None;history=[]
    for n in range(1,generations+1):
        summary=copy.deepcopy(template);summary.update(generation=str(n),closed=closed and n==generations)
        summary['durability']['uncommitted_resource_batch_at_risk']=not summary['closed']
        costmetrics={k:_s2_acc(float(n) if k in _S2_FLOATS else str(n),str(n)) for k in (*_S2_FLOATS,*_S2_COUNTS)}
        summary['cumulative']=dict(scope='enclosing_transport_since_v2_genesis',prior_capture='none',
            cost=dict(scope='observer_process_completed_operations',position=dict(observer_id='literal-observer',stop_operation=str(n)),
                metrics=costmetrics,arrow_io_forecast_bytes='0',uncovered_restarts='0',publication_tail='not_included'),
            phases=dict(scope='accepted_source_rows_across_sources_no_cross_clock_bounds',entries=[],
                overflow=dict(call_count='0',metrics={k:_s2_acc() for k in ('duration_seconds','process_cpu_seconds','allocated_bytes','gc_seconds','gc_count')}),incomplete=False,reasons=[]),
            coverage=dict(gap_events='0',incomplete=True,reasons=['publication_tail']),
            max_observed_rss_bytes=dict(value=None,status='not_measured',reason='no_observations'))
        artifacts=copy.deepcopy(leaves)
        if previous:artifacts.append(dict(ref=previous['catalog_ref'],role='performance.provenance',table=None,rows=None,accepted=[]))
        artifacts.sort(key=lambda r:(r['ref']['path'],r['ref']['sha256']))
        catalog=dict(schema='qcl-negf.telemetry-catalog.v3',contract_set='qcl-negf.results.v2',transport=copy.deepcopy(c['transport']),generation=str(n),artifacts=artifacts)
        catalog_ref=put(c['technical'],f'retained/catalog-{n}.json',catalog,catalog['schema'])
        summary['retention']['retained_bytes']=str(sum(int(r['ref']['bytes']) for r in artifacts))
        summary_ref=put(c['technical'],f'retained/summary-{n}.json',summary,summary['schema'])
        commit=dict(schema='qcl-negf.telemetry-commit.v2',contract_set='qcl-negf.results.v2',session=copy.deepcopy(summary['session']),transport=copy.deepcopy(c['transport']),
            generation=str(n),closed=summary['closed'],published_unix=float(n),previous=None if previous is None else dict(commit=previous['commit_ref'],summary=previous['summary_ref']),
            summary=summary_ref,catalog=catalog_ref,source_cursors=[],accepted_ranges=[],contexts=dict(phase_context_refs=[],resume_context_refs=[]),
            cost_delta=dict(observer_id='literal-observer',first_operation=str(n-1),stop_operation=str(n),as_of_monotonic_ns=str(n),
                metrics={k:_s2_acc(1.0 if k in _S2_FLOATS else '1','1') for k in (*_S2_FLOATS,*_S2_COUNTS)},arrow_io_forecast_bytes='0',publication_tail='not_included',restart_gap=False),
            window_delta=dict(introduced=[copy.deepcopy(r) for r in artifacts if previous is None or r not in previous['catalog']['artifacts']],
                removed=[] if previous is None else [copy.deepcopy(r) for r in previous['catalog']['artifacts'] if r not in artifacts],pending_unlink=[]))
        context={} if previous is None else dict(previous=previous['commit'],previous_ref=previous['commit_ref'],previous_summary=previous['summary'],previous_summary_ref=previous['summary_ref'],previous_catalog=previous['catalog'])
        validate_telemetry_commit(commit,summary=summary,summary_ref=summary_ref,catalog=catalog,**context)
        commit_ref=put(c['technical'],f'retained/commit-{n}.json',commit,commit['schema'])
        previous=dict(commit=commit,commit_ref=commit_ref,summary=summary,summary_ref=summary_ref,catalog=catalog,catalog_ref=catalog_ref);history.append(previous)
    binding_ref=None
    if closed:
        embedded=lambda ref:dict(ref=copy.deepcopy(ref),raw_utf8=(c['technical']/ref['path']).read_text())
        binding=dict(schema='qcl-negf.telemetry-binding.v2',contract_set='qcl-negf.results.v2',scope='selected_state_with_execution_attempt_telemetry',
            selected_state=c['state'],transport=c['transport'],commit=embedded(previous['commit_ref']),summary=embedded(previous['summary_ref']),source_binding=embedded(c['source_binding']),
            capture_status=dict(terminal_tail='unknown',scalar_live_route='direct',scalar_live_reason=None,performance_status='available',reason=None))
        binding_ref=put(c['technical'],'retained/binding.json',binding,binding['schema'])
    selection=TelemetrySelection(c['technical'],commit_ref=previous['commit_ref'],source_binding_ref=c['source_binding'],expected_transport=c['transport'],source_transport=c['source_transport'],
        canonical_plan_root=c['root'],canonical_plan_ref=c['plan'],expected_producer_provenance=c['producer_provenance'],binding_ref=binding_ref)
    # Opaque historical catalogs and older transaction nodes have actually gone.
    for old in history[:-2]:
        for k in ('commit_ref','summary_ref','catalog_ref'):(c['technical']/old[k]['path']).unlink()
    c.update(wire=previous,wire_history=history,selection=selection,leaves=leaves)
    return c


def test_S2_v2_raw_closure_keeps_current_pair_catalogs_and_no_recursive_promise(tmp_path):
    import json
    from qcl_negf_results.state import StateReader
    from qcl_negf_results.performance_views import read_performance_context
    c=s2_wire_case(tmp_path)
    with StateReader(c['science'],state_ref=c['state'],expected_identity=c['state']['identity']) as reader:
        view=read_performance_summary(reader,telemetry=c['selection'],limits=PerformanceLimits(maximum_objects=16384))
        raw,files=read_performance_context(reader,telemetry=c['selection'],limits=PerformanceLimits(maximum_objects=16384))
        assert view['summary']['cumulative']['cost']['metrics']['read_seconds']['known']==3.0
        assert view['integrity']['provenance_origin'].startswith('caller_assertions')
        assert reader._shared_budget is None and reader._budget._source_guard is None
    names={(scope,ref['path']) for scope,ref,_ in files}
    assert {('technical',f'retained/{kind}-{n}.json') for n in (2,3) for kind in ('commit','summary','catalog')}<=names
    assert not any(path.endswith(('catalog-1.json','commit-1.json','summary-1.json')) for _,path in names)
    for scope,ref,payload in files:
        assert len(payload)==int(ref['bytes']) and hashlib.sha256(payload).hexdigest()==ref['sha256']
    assert json.loads(raw)['commit_ref']==c['wire']['commit_ref']
    c['selection'].commit_ref['sha256']='f'*64
    assert json.loads(raw)['commit_ref']==c['wire']['commit_ref']


@pytest.mark.parametrize('fault',['missing_catalog','missing_artifact','hash','catalog_schema','mixed_previous','external_context','wrong_transport','footer_rows','debt_not_member'])
def test_S2_v2_read_rejects_missing_or_forged_promises_before_return(tmp_path,fault):
    from qcl_negf_results.state import StateReader
    from qcl_negf_results.performance_views import TelemetrySelection,read_performance_context
    from result_v2_fixture import put
    c=s2_wire_case(tmp_path,parquet=fault in ('missing_artifact','footer_rows'),closed=False,declared_rows=3 if fault=='footer_rows' else 2);v=c['wire'];selection=c['selection']
    if fault=='missing_catalog':(c['technical']/v['catalog_ref']['path']).unlink()
    elif fault=='missing_artifact':(c['technical']/c['leaves'][0]['ref']['path']).unlink()
    elif fault=='hash':(c['technical']/v['summary_ref']['path']).write_bytes(b'{}')
    elif fault=='debt_not_member':
        # A pending unlink leaf is historical physical debt, not promised payload.
        old=c['wire_history'][0]['catalog_ref'];v['commit']['window_delta']['pending_unlink']=[old]
        ref=put(c['technical'],v['commit_ref']['path'],v['commit'],v['commit']['schema']);data=selection.document();data['commit_ref']=ref
        selection=TelemetrySelection(c['technical'],canonical_plan_root=c['root'],**data)
    elif fault=='external_context':
        data=selection.document();data['resume_context_refs']=[dict(commit_ref=c['wire_history'][1]['commit_ref'],summary_ref=c['wire_history'][1]['summary_ref'])]
        selection=TelemetrySelection(c['technical'],canonical_plan_root=c['root'],**data)
    elif fault=='wrong_transport':
        data=selection.document();data['expected_transport']['selection']['attempt']=88
        selection=TelemetrySelection(c['technical'],canonical_plan_root=c['root'],**data)
    elif fault=='catalog_schema':
        v['catalog']['schema']='qcl-negf.telemetry-catalog.v2'
        v['commit']['catalog']=put(c['technical'],v['catalog_ref']['path'],v['catalog'],'qcl-negf.telemetry-catalog.v3')
        data=selection.document();data['commit_ref']=put(c['technical'],v['commit_ref']['path'],v['commit'],v['commit']['schema'])
        selection=TelemetrySelection(c['technical'],canonical_plan_root=c['root'],**data)
    elif fault=='mixed_previous':
        v['commit']['previous']['summary']['schema']='qcl-negf.telemetry-summary.v1';data=selection.document();data['commit_ref']=put(c['technical'],v['commit_ref']['path'],v['commit'],v['commit']['schema'])
        selection=TelemetrySelection(c['technical'],canonical_plan_root=c['root'],**data)
    else:
        assert fault=='footer_rows'  # Structurally consistent declaration differs from actual footer2.
    with StateReader(c['science'],state_ref=c['state'],expected_identity=c['state']['identity']) as reader:
        if fault=='debt_not_member':
            _,files=read_performance_context(reader,telemetry=selection,limits=PerformanceLimits(maximum_objects=16384))
            assert not any(ref==old for _,ref,_ in files)
        else:
            with pytest.raises((ContractError,OSError)):read_performance_context(reader,telemetry=selection,limits=PerformanceLimits(maximum_objects=16384))
        assert reader._shared_budget is None and reader._budget._source_guard is None


@pytest.mark.parametrize('fault',['io','objects','late_change','cancel'])
def test_S2_raw_context_v2_prepaid_limits_and_failure_cleanup(tmp_path,monkeypatch,fault):
    from qcl_negf_results.state import StateReader
    import qcl_negf_results.performance_views as module
    c=s2_wire_case(tmp_path,parquet=True);original_copy=module.deepcopy;changed=[]
    target=c['technical']/c['leaves'][0]['ref']['path'];original=target.read_bytes()
    if fault in ('late_change','cancel'):
        def copy_at_return(value):
            answer=original_copy(value)
            if value==c['leaves'][0]['ref'] and not changed:
                changed.append(True)
                if fault=='cancel':raise KeyboardInterrupt('S2 original return cancellation')
                target.write_bytes(original+b'x')
            return answer
        monkeypatch.setattr(module,'deepcopy',copy_at_return)
    limits=PerformanceLimits(maximum_objects=1 if fault=='objects' else 16384,
        maximum_io_bytes=1 if fault=='io' else 4*1024*1024)
    with StateReader(c['science'],state_ref=c['state'],expected_identity=c['state']['identity']) as reader:
        with pytest.raises(KeyboardInterrupt if fault=='cancel' else ContractError) as failure:
            module.read_performance_context(reader,telemetry=c['selection'],limits=limits)
        if fault=='late_change':assert changed and failure.value.code=='source_changed'
        elif fault=='cancel':assert changed and str(failure.value)=='S2 original return cancellation' and target.read_bytes()==original
        else:assert failure.value.code=='budget_exceeded'
        assert reader._shared_budget is None and reader._budget._source_guard is None
        if fault=='io':assert reader.io_counters['bytes_read_total']<=1


def test_S2_catalog_raw_ceiling_refuses_before_catalog_open(tmp_path,monkeypatch):
    from qcl_negf_results.state import StateReader
    from result_v2_fixture import put
    import qcl_negf_results.performance_views as module
    c=s2_wire_case(tmp_path,closed=False);commit=copy.deepcopy(c['wire']['commit']);commit['catalog']['bytes']='65537'
    data=c['selection'].document();data['commit_ref']=put(c['technical'],c['wire']['commit_ref']['path'],commit,commit['schema'])
    selection=module.TelemetrySelection(c['technical'],canonical_plan_root=c['root'],**data)
    seen=[];original=module.regular
    def observed(path,*args,**kwargs):seen.append(Path(path));return original(path,*args,**kwargs)
    from pathlib import Path
    monkeypatch.setattr(module,'regular',observed)
    with StateReader(c['science'],state_ref=c['state'],expected_identity=c['state']['identity']) as reader:
        with pytest.raises(ContractError) as failure:module.read_performance_context(reader,telemetry=selection,limits=PerformanceLimits(maximum_objects=16384))
        assert failure.value.code=='budget_exceeded'
    assert c['technical']/c['wire']['catalog_ref']['path'] not in seen


def test_S2_object_ceiling_extension_is_explicit_v2_only(tmp_path,monkeypatch):
    from native_consumer_fixture import technical_source,telemetry_input
    from qcl_negf_results.state import StateReader
    import qcl_negf_results.performance_views as module
    c=technical_source(tmp_path);writer=TelemetryWriter(c['technical'],{}, {},transport=c['transport']);writer.close();selection=telemetry_input(c,writer)
    seen=[]
    def forbidden(*args,**kwargs):seen.append(True);raise AssertionError('v1 extra object allowance used before rejection')
    monkeypatch.setattr(module,'_read_reference',forbidden)
    with StateReader(c['science'],state_ref=c['state'],expected_identity=c['state']['identity']) as reader:
        with pytest.raises(ValueError):module.read_performance_context(reader,telemetry=selection,limits=PerformanceLimits(maximum_objects=4097))
    assert seen==[]
    assert PerformanceLimits().maximum_objects==4096


def test_S2_committed_phase_raw_uses_retained_root_and_counts_only_accepted_prefix(tmp_path):
    import json,shutil
    from native_consumer_fixture import sealed_phase
    from result_v2_fixture import put,reference
    from qcl_negf_results.state import StateReader
    from qcl_negf_results.performance_views import TelemetrySelection,read_performance_context
    c=s2_wire_case(tmp_path,generations=1,closed=False);v=c['wire'];producer=tmp_path/'producer'
    p=sealed_phase(producer,c['state']['identity'])
    retained={}
    for kind in ('publication','segment'):
        target=c['technical']/f'raw/{kind}.json';target.parent.mkdir(exist_ok=True)
        target.write_bytes((producer/p[kind+'_ref']['path']).read_bytes())
        retained[kind]=reference(c['technical'],f'raw/{kind}.json',p[kind+'_ref']['schema'])
    source=p['segment']['source'];identity=p['segment']['identity']
    source_key=hashlib.sha256(json_bytes(dict(identity=identity,source=source))).hexdigest()
    cursor=dict(schema='qcl-negf.producer-phase-cursor.v1',contract_set='qcl-negf.results.v2',identity=identity,source=source,
        publication=p['publication_ref'],segment=p['segment_ref'],next_row='1',last_sequence='8',complete_at_observation=False)
    entry=dict(source_key=source_key,identity=identity,source=source,cursor=cursor,route_epoch='0',
        binding=dict(status='unbound',state=None,invocation=None,evidence=None,reason='fixture_mapping_not_asserted'),
        producer_health=p['segment']['health'],source_closed=True,health_reason=None)
    summary=copy.deepcopy(v['summary']);commit=copy.deepcopy(v['commit'])
    commit['source_cursors']=[copy.deepcopy(entry)];summary['source_cursors']=[copy.deepcopy(entry)]
    commit['contexts']['phase_context_refs']=[dict(publication_ref=p['publication_ref'],segment_ref=p['segment_ref'],
        retained_publication_ref=retained['publication'],retained_segment_ref=retained['segment'])]
    ar=dict(source_key=source_key,publication=p['publication_ref'],segment=p['segment_ref'],retained_publication=retained['publication'],retained_segment=retained['segment'],
        mode='normal',first_row='0',stop_row='1',cursor=cursor);commit['accepted_ranges']=[ar]
    summary['totals'].update(accepted_source_rows='1',accepted_aggregate_calls='2')
    summary['producer_phase_aggregates']=[dict(source_key=source_key,identity=identity,source=source,representation='normal_aggregate',aggregate=p['segment']['rows'][0])]
    metrics={k:_s2_acc(unavailable='2',reasons=['missing_observation']) for k in ('duration_seconds','process_cpu_seconds','allocated_bytes','gc_seconds','gc_count')}
    metrics['duration_seconds']=_s2_acc(4.5,'2')
    phase=dict(key=dict(stage='scba',phase='dyson',status='ok',task_width='2',representation='normal_aggregate',counter_scope='process_inclusive_delta_on_end'),call_count='2',metrics=metrics)
    summary['cumulative']['phases'].update(entries=[phase],incomplete=True,reasons=['missing_observation'])
    summary['cumulative']['coverage']['reasons'].append('missing_observation')
    leaves=[dict(ref=ref,role='performance.provenance' if kind=='publication' else 'performance.full',table=None,rows=None,
        accepted=[] if kind=='publication' else [{k:ar[k] for k in ('source_key','segment','first_row','stop_row')}]) for kind,ref in retained.items()]
    leaves.sort(key=lambda r:(r['ref']['path'],r['ref']['sha256']))
    catalog=dict(v['catalog'],artifacts=copy.deepcopy(leaves));commit['window_delta']['introduced']=copy.deepcopy(leaves)
    summary['retention'].update(retained_source_rows='1',retained_bytes=str(sum(int(r['ref']['bytes']) for r in leaves)))
    commit['catalog']=put(c['technical'],v['catalog_ref']['path'],catalog,catalog['schema'])
    commit['summary']=put(c['technical'],v['summary_ref']['path'],summary,summary['schema'])
    cref=put(c['technical'],v['commit_ref']['path'],commit,commit['schema'])
    data=c['selection'].document();data['commit_ref']=cref;selection=TelemetrySelection(c['technical'],canonical_plan_root=c['root'],**data)
    shutil.rmtree(producer)
    with StateReader(c['science'],state_ref=c['state'],expected_identity=c['state']['identity']) as reader:
        view=read_performance_summary(reader,telemetry=selection,limits=PerformanceLimits(maximum_objects=16384))
        _,files=read_performance_context(reader,telemetry=selection,limits=PerformanceLimits(maximum_objects=16384))
    raw=next(payload for scope,ref,payload in files if scope=='technical' and ref==retained['segment'])
    assert len(json.loads(raw)['rows'])==2
    assert view['summary']['totals']['accepted_source_rows']=='1'
    assert view['summary']['cumulative']['phases']['entries'][0]['call_count']=='2'
    assert view['summary']['cumulative']['phases']['entries'][0]['metrics']['gc_count']['known'] is None


def test_S2_hash_valid_malformed_parquet_is_a_contract_rejection(tmp_path):
    from qcl_negf_results.state import StateReader
    from qcl_negf_results.performance_views import read_performance_context
    c=s2_wire_case(tmp_path,parquet=True,closed=False,malformed_parquet=True)
    with StateReader(c['science'],state_ref=c['state'],expected_identity=c['state']['identity']) as reader:
        with pytest.raises(ContractError):read_performance_context(reader,telemetry=c['selection'],limits=PerformanceLimits(maximum_objects=16384))
        assert reader._shared_budget is None and reader._budget._source_guard is None


@pytest.mark.parametrize('bad',[False,True])
def test_S2_explicit_catalog_index_leaf_uses_typed_membership_without_discovery(tmp_path,bad):
    from qcl_negf_results.state import StateReader
    from qcl_negf_results.performance_views import TelemetrySelection,read_performance_context
    from result_v2_fixture import put
    c=s2_wire_case(tmp_path,closed=False);v=c['wire'];prior=c['wire_history'][-2]
    pair=dict(commit=prior['commit_ref'],summary=prior['summary_ref'])
    index=dict(schema='qcl-negf.performance-index.v3',contract_set='qcl-negf.results.v2',transport=c['transport'],current=pair,retained=[pair,pair] if bad else [])
    ref=put(c['technical'],'retained/discovery-snapshot.json',index,index['schema'])
    leaf=dict(ref=ref,role='performance.provenance',table=None,rows=None,accepted=[])
    catalog=copy.deepcopy(v['catalog']);catalog['artifacts'].append(leaf);catalog['artifacts'].sort(key=lambda r:(r['ref']['path'],r['ref']['sha256']))
    summary=copy.deepcopy(v['summary']);summary['retention']['retained_bytes']=str(sum(int(r['ref']['bytes']) for r in catalog['artifacts']))
    commit=copy.deepcopy(v['commit']);commit['window_delta']['introduced'].append(leaf);commit['window_delta']['introduced'].sort(key=lambda r:(r['ref']['path'],r['ref']['sha256']))
    commit['catalog']=put(c['technical'],v['catalog_ref']['path'],catalog,catalog['schema']);commit['summary']=put(c['technical'],v['summary_ref']['path'],summary,summary['schema'])
    data=c['selection'].document();data['commit_ref']=put(c['technical'],v['commit_ref']['path'],commit,commit['schema'])
    selection=TelemetrySelection(c['technical'],canonical_plan_root=c['root'],**data)
    with StateReader(c['science'],state_ref=c['state'],expected_identity=c['state']['identity']) as reader:
        if bad:
            with pytest.raises(ContractError):read_performance_context(reader,telemetry=selection,limits=PerformanceLimits(maximum_objects=16384))
        else:
            _,files=read_performance_context(reader,telemetry=selection,limits=PerformanceLimits(maximum_objects=16384))
            assert any(reference==ref for _,reference,_ in files)


@pytest.mark.parametrize('fault',['missing_bytes','not_ref','unknown_schema'])
def test_S2_invalid_explicit_reference_is_rejected_before_technical_open(tmp_path,monkeypatch,fault):
    from qcl_negf_results.state import StateReader
    import qcl_negf_results.performance_views as module
    c=s2_wire_case(tmp_path,closed=False);data=c['selection'].document()
    if fault=='missing_bytes':data['commit_ref'].pop('bytes')
    elif fault=='not_ref':data['commit_ref']=[]
    else:data['commit_ref']['schema']='qcl-negf.telemetry-commit.v99'
    selected=module.TelemetrySelection(c['technical'],canonical_plan_root=c['root'],**data)
    seen=[]
    def forbidden(*args,**kwargs):seen.append(True);raise AssertionError('invalid ref opened a technical file')
    monkeypatch.setattr(module,'regular',forbidden)
    with StateReader(c['science'],state_ref=c['state'],expected_identity=c['state']['identity']) as reader:
        with pytest.raises(ContractError):module.read_performance_summary(reader,telemetry=selected,limits=PerformanceLimits(maximum_objects=16384))
        assert reader._shared_budget is None and reader._budget._source_guard is None
    assert seen==[]


def test_S2_growing_payload_cannot_fill_raw_buffer_beyond_selected_whole_bytes(tmp_path, monkeypatch):
    from contextlib import contextmanager
    from qcl_negf_results.state import _IOBudget
    import qcl_negf_results.catalog as module
    c=s2_wire_case(tmp_path,parquet=True,closed=False)
    descriptor=c['leaves'][0];selected_size=int(descriptor['ref']['bytes'])
    target=c['technical']/descriptor['ref']['path'];original=module.regular;changed=[]
    class Growing:
        def __init__(self,raw):self.raw=raw
        def __getattr__(self,name):return getattr(self.raw,name)
        def read(self,size=-1):
            result=self.raw.read(size)
            if result and not changed:
                changed.append(True)
                with target.open('ab') as extra:extra.write(b'x'*(2*1024*1024))
            return result
    @contextmanager
    def open_growing(*args,**kwargs):
        with original(*args,**kwargs) as raw:yield Growing(raw)
    monkeypatch.setattr(module,'regular',open_growing)
    budget=_IOBudget(4*1024*1024)
    with pytest.raises(ContractError) as error:
        module._retained_payload(c['technical'],descriptor,budget=budget,maximum_metadata_bytes=1024*1024,bindings={},retain_raw=True)
    assert changed and error.value.code=='source_changed'
    assert budget.bytes_read_total<=selected_size


def _s2_r1_index_leaves(tmp_path, *, groups, generations):
    from result_v2_fixture import put
    from qcl_negf_results.performance_views import TelemetrySelection
    c=s2_wire_case(tmp_path,generations=generations,closed=False);v=c['wire'];leaves=[]
    for number,entries in enumerate(groups):
        pairs=[]
        for n in range(entries):
            pair={}
            for kind in ('commit','summary'):
                ref=copy.deepcopy(v[kind+'_ref']);ref['path']=f'opaque/{number}/{kind}-{n:03}.json';pair[kind]=ref
            pairs.append(pair)
        pairs.sort(key=json_bytes)
        index=dict(schema='qcl-negf.performance-index.v3',contract_set='qcl-negf.results.v2',
            transport=c['transport'],current=pairs[0],retained=pairs)
        # Independent structural measure: neither the admission helper nor the
        # shape validators supply this expectation. Each pair contributes 13.
        pending=[index];nodes=0
        while pending:
            node=pending.pop();nodes+=1
            if isinstance(node,dict):pending.extend(node.values())
            elif isinstance(node,(tuple,list)):pending.extend(node)
        assert nodes==29+13*entries
        ref=put(c['technical'],f'retained/object-credit-index-{number}.json',index,index['schema'])
        leaves.append(dict(ref=ref,role='performance.provenance',table=None,rows=None,accepted=[]))
    catalog=copy.deepcopy(v['catalog']);catalog['artifacts'].extend(leaves)
    catalog['artifacts'].sort(key=lambda row:(row['ref']['path'],row['ref']['sha256']))
    summary=copy.deepcopy(v['summary']);summary['retention']['retained_bytes']=str(sum(int(row['ref']['bytes']) for row in catalog['artifacts']))
    commit=copy.deepcopy(v['commit']);commit['window_delta']['introduced'].extend(leaves)
    commit['window_delta']['introduced'].sort(key=lambda row:(row['ref']['path'],row['ref']['sha256']))
    commit['catalog']=put(c['technical'],v['catalog_ref']['path'],catalog,catalog['schema'])
    commit['summary']=put(c['technical'],v['summary_ref']['path'],summary,summary['schema'])
    data=c['selection'].document();data['commit_ref']=put(c['technical'],v['commit_ref']['path'],commit,commit['schema'])
    c.update(selection=TelemetrySelection(c['technical'],canonical_plan_root=c['root'],**data),object_leaves=leaves)
    return c


@pytest.mark.parametrize('route,groups,generations,limit',[
    ('summary',(128,),1,1600),
    ('context',(128,),1,2400),
    ('summary',(64,64),1,2400),
    ('context',(64,64),3,5000),
])
@pytest.mark.parametrize('generous',[False,True])
def test_S2_R1_every_decoded_after_leaf_shares_complete_operation_object_credit(tmp_path,route,groups,generations,limit,generous):
    import json
    from qcl_negf_results.state import StateReader
    from qcl_negf_results.performance_views import read_performance_context
    c=_s2_r1_index_leaves(tmp_path,groups=groups,generations=generations)
    limits=PerformanceLimits(maximum_objects=16384 if generous else limit)
    with StateReader(c['science'],state_ref=c['state'],expected_identity=c['state']['identity']) as reader:
        def selected_read():
            if route=='summary':return read_performance_summary(reader,telemetry=c['selection'],limits=limits)
            return read_performance_context(reader,telemetry=c['selection'],limits=limits)
        if not generous:
            with pytest.raises(ContractError,match='object') as error:selected_read()
            assert error.value.code=='budget_exceeded'
        elif route=='summary':
            view=selected_read()
            assert view['status']=='available' and view['summary']['generation']==str(generations)
            assert view['summary']['rows']['resources']=='0'
            assert view['summary']['cumulative']['cost']['metrics']['read_seconds']['known']==float(generations)
        else:
            _,files=selected_read()
            for leaf,entries in zip(c['object_leaves'],groups):
                matches=[raw for scope,ref,raw in files if scope=='technical' and ref==leaf['ref']]
                assert len(matches)==1
                assert len(json.loads(matches[0])['retained'])==entries
            assert not any('/opaque/' in ref['path'] or ref['path'].startswith('opaque/') for _,ref,_ in files)
        assert reader._shared_budget is None and reader._budget._source_guard is None


def test_T3_exact_index_pin_reads_closed_history_without_mutable_pointer_or_missing_binding(tmp_path):
    import json
    from test_capture import _T3_capture, _T3_anchor
    from qcl_negf_results.state import StateReader
    from qcl_negf_results.performance_views import TelemetrySelection, read_performance_context
    fixture = _T3_capture(tmp_path, history_only=True)
    anchor, raw, immutable = _T3_anchor(fixture)
    ref = dict(path=str(immutable.relative_to(fixture['result'])), bytes=str(len(raw)),
        sha256=hashlib.sha256(raw).hexdigest(), schema=anchor['schema'], media_type='application/json')
    capture = fixture['capture']; writer = fixture['writer']; capture.remember(anchor, ref, raw)
    close = writer.close_context()
    pin = capture.publish(close_context=close, capture_status=dict(terminal_tail='unknown', scalar_live_route='unavailable',
        scalar_live_reason='external_root', performance_status='incomplete', reason='tail_unknown'))
    index = json.loads((writer.root / pin['path']).read_bytes()); data = index['entries'][0]['telemetry']
    # A previously pinned selection cannot rebase through mutable lookup.
    (writer.root / 'current-source.json').write_bytes(b'{"foreign_pointer":true}')
    selected = TelemetrySelection(writer.root, canonical_plan_root=fixture['job'], **data)
    with StateReader(fixture['context']['science'], state_ref=fixture['context']['state'],
            expected_identity=fixture['context']['state']['identity']) as reader:
        view = read_performance_summary(reader, telemetry=selected, limits=PerformanceLimits(maximum_objects=16384))
        assert view['status'] == 'available' and view['closed'] is True
        _, originals = read_performance_context(reader, telemetry=selected, limits=PerformanceLimits(maximum_objects=16384))
        assert any(scope == 'scientific' and raw == anchor['scientific']['manifest_utf8'].encode() for scope, _, raw in originals)
        missing = dict(data, binding_ref=None)
        with pytest.raises(ContractError):
            read_performance_summary(reader, telemetry=TelemetrySelection(writer.root, canonical_plan_root=fixture['job'], **missing),
                limits=PerformanceLimits(maximum_objects=16384))
        for limits in (PerformanceLimits(maximum_objects=16384, maximum_io_bytes=1),
                       PerformanceLimits(maximum_objects=16384, maximum_json_bytes=1)):
            with pytest.raises(ContractError) as error:read_performance_context(reader, telemetry=selected, limits=limits)
            assert error.value.code == 'budget_exceeded'
        assert reader._shared_budget is None and reader._budget._source_guard is None


def test_T3_joined_retained_forest_keeps_default_object_credit_and_explicit_ceiling(tmp_path):
    from qcl_negf_results.state import StateReader
    from qcl_negf_results.performance_views import read_performance_context
    context = _s2_r1_index_leaves(tmp_path, groups=(64, 64), generations=3)
    with StateReader(context['science'], state_ref=context['state'], expected_identity=context['state']['identity']) as reader:
        with pytest.raises(ContractError, match='object') as error:
            read_performance_context(reader, telemetry=context['selection'])
        assert error.value.code == 'budget_exceeded'
        _, originals = read_performance_context(reader, telemetry=context['selection'],
            limits=PerformanceLimits(maximum_objects=16384))
        for leaf in context['object_leaves']:
            assert any(ref == leaf['ref'] for _, ref, _ in originals)
        assert reader._shared_budget is None and reader._budget._source_guard is None
