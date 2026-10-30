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
