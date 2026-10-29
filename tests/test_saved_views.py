"""Literal saved-view oracles; no solver or state/full owner is needed."""

import h5py
import numpy as np
import pytest

from qcl_negf_contracts.messages import ContractError
from qcl_negf_results.saved_views import (AnalysisReader, SavedViewLimits, brief_result,
                                        describe_saved_views, read_saved_view)

IDENTITY = {"point_id": "p0", "execution_id": "ex", "attempt": 2, "plan_fingerprint": "a" * 64}
SELECTION = {"run_uuid": "run", "execution_id": "ex", "attempt": 2, "calcjob_uuid": "job"}


def metadata(rows):
    return {"plan": {"schema": "qcl-negf-scientific-plan-v2", "root_definition_id": "study",
        "root_kind": "study", "name": "Study", "fingerprint": "a"*64, "computation_count": 3,
        "inclusions": [], "executions": [{"id": "ex", "definition_id": "study", "variant_id": "base"}],
        "points": [{"id": f"p{i}", "execution_id": "ex", "temperature_K": 70.,
                    "voltage_per_period_V": i*.01, "branch": "b", "order": i} for i in range(3)]},
        "series": {"schema": "qcl-negf-series-result-v3", "contract_set": "qcl-negf.results.v1",
                   "plan_fingerprint": "a"*64, "points": rows},
        "source": {"kind": "aiida_retrieved", "locator_id": "registered", "calcjob_uuid": "job"}}


def row(i, current=True, **extra):
    return {"id": f"p{i}", "execution_id": "ex", "attempt": 2, "status": "completed",
            "quality": "strict", "converged": True,
            "observables": {**({"current_density_A_per_m2": i+1} if current else {}),
                "scientific_assessment": {"scientific_accepted": True, "stationary_candidate_accepted": True}}, **extra}


@pytest.mark.parametrize("middle", [None, row(1, False), row(1, observables={"current_density_A_per_m2": None}), row(1, status="failed")])
def test_brief_preserves_missing_middle_and_never_connects_endpoints(middle, monkeypatch):
    monkeypatch.setattr(h5py, "File", lambda *a, **k: pytest.fail("brief opened HDF5"))
    rows = [row(2), row(0)] + ([] if middle is None else [middle])
    brief = brief_result(metadata(rows), selection=SELECTION, maximum_bytes=1024*1024)
    assert [p["point_id"] for p in brief["points"]] == ["p0", "p1", "p2"]
    assert [p["coordinates"]["voltage_per_period_V"] for p in brief["points"]] == [0., .01, .02]
    assert [p["point_id"] for p in brief["points"] if p["line_connectable"]] == ["p0", "p2"]
    assert brief["current_segments"] == [["p0"], ["p2"]]
    assert brief["points"][1]["missing"] == (["planned_row_absent"] if middle is None else
        ["current_unavailable"] if middle.get("observables", {}).get("current_density_A_per_m2") is None else [])


def test_brief_preserves_retained_and_borrowed_attempts_in_frozen_order(monkeypatch):
    monkeypatch.setattr(h5py, "File", lambda *a, **k: pytest.fail("brief opened HDF5"))
    accepted = {"scientific_accepted": True, "stationary_candidate_accepted": True}
    paused_assessment = {"scientific_accepted": False, "reason": "saved paused assessment"}
    retained = {"id": "p0", "execution_id": "ex", "attempt": 1, "status": "completed",
                "observables": {"current_density_A_per_m2": 7., "scientific_assessment": accepted},
                "data": {"source_commit_sha256": "1" * 64}}
    current = {"id": "p1", "execution_id": "ex", "attempt": 2, "status": "completed",
               "observables": {"current_density_A_per_m2": 8., "scientific_assessment": accepted},
               "data": {"source_commit_sha256": "2" * 64}}
    paused = {"id": "p2", "execution_id": "ex", "attempt": 2, "status": "paused",
              "observables": {"current_density_A_per_m2": 9., "scientific_assessment": paused_assessment},
              "data": {"checkpoint_source_attempt": 1, "source_commit_sha256": "3" * 64,
                       "pause_reason": "resource_pressure", "resume_kind": "checkpoint",
                       "recovery_origin": "last_committed_before_resource_pause"}}
    result = brief_result(metadata([paused, current, retained]), selection=SELECTION)
    assert result["selection"] == SELECTION
    assert [point["point_id"] for point in result["points"]] == ["p0", "p1", "p2"]
    assert [point["source_attempt"] for point in result["points"]] == [1, 2, 1]
    assert [point["source_commit_sha256"] for point in result["points"]] == ["1" * 64, "2" * 64, "3" * 64]
    assert [point["recorded_statuses"]["scientific_assessment"] for point in result["points"]] == [accepted, accepted, paused_assessment]
    assert result["current_segments"] == [["p0", "p1"]]
    gap = brief_result(metadata([paused, retained]), selection=SELECTION)
    assert [point["point_id"] for point in gap["points"]] == ["p0", "p1", "p2"]
    assert gap["points"][1]["missing"] == ["planned_row_absent"]
    assert gap["points"][1]["source_attempt"] is None
    assert gap["current_segments"] == [["p0"]]


@pytest.mark.parametrize("attempt", [3, 0, -1, True, False, 1.0, "1", None])
def test_brief_rejects_recorded_invalid_or_future_row_attempt(attempt):
    with pytest.raises(ContractError) as error:
        brief_result(metadata([row(0, attempt=attempt)]), selection=SELECTION)
    assert error.value.code == "identity_mismatch"


@pytest.mark.parametrize("row_attempt,source_attempt", [(2, 3), (1, 2), (2, 0), (2, -1),
                                                       (2, True), (2, False), (2, 1.0), (2, "1"), (2, None)])
def test_brief_rejects_checkpoint_source_outside_real_row_bounds(row_attempt, source_attempt):
    saved = row(0, attempt=row_attempt, status="paused", data={"checkpoint_source_attempt": source_attempt})
    with pytest.raises(ContractError) as error:
        brief_result(metadata([saved]), selection=SELECTION)
    assert error.value.code == "identity_mismatch"


def test_brief_legacy_absent_attempt_stays_unknown():
    legacy = {"id": "p0", "execution_id": "ex", "status": "completed",
              "observables": {"current_density_A_per_m2": 7.}}
    result = brief_result(metadata([legacy]), selection=SELECTION)
    assert result["points"][0]["source_attempt"] is None
    assert result["points"][0]["recorded_statuses"]["scientific_accepted"] is None
    assert result["current_segments"] == []


def fixture(root, *, denominator=True):
    from consumer_context_fixture import semantic_fixture
    root.mkdir(parents=True,exist_ok=True)
    selected,artifact,_,_,_=semantic_fixture(root,denominator=denominator)
    return root,selected,artifact


from contextlib import contextmanager
from qcl_negf_results.state import StateReader


@contextmanager
def selected_view(bundle, *, limits=None, expected_identity=None):
    root,selected,artifact=bundle
    with StateReader(root,state_ref=selected,expected_identity=expected_identity or selected['identity']) as owner:
        with AnalysisReader(owner,artifact_ref=artifact,limits=limits) as view:
            yield view,owner


def semantic_read(bundle,view_id,selection,*,limits=None):
    with selected_view(bundle,limits=limits) as (view,_): return view.read(view_id,selection)


def semantic_describe(bundle,*,limits=None,expected_identity=None):
    with selected_view(bundle,limits=limits,expected_identity=expected_identity) as (view,_): return view.describe()



def test_manifest_never_reads_numeric_arrays_or_full_owner(tmp_path, monkeypatch):
    bundle=fixture(tmp_path/'record');original=h5py.Dataset.__getitem__
    def guarded(dataset,key):
        if dataset.dtype.kind in 'biufc': pytest.fail('manifest read numeric payload')
        return original(dataset,key)
    monkeypatch.setattr(h5py.Dataset,'__getitem__',guarded)
    manifest=semantic_describe(bundle)
    assert manifest['integrity']['observations_file_verified'] is True
    assert manifest['integrity']['diagnostics_file_verified'] is True
    assert manifest['integrity']['full_core_payloads_scope']=='not_assessed_by_this_operation'
    heat=manifest['views']['spatial_energy_density']
    assert heat['shape']==[2,3] and heat['axes']==['E','z']
    assert manifest['artifact_ref']['schema']=='qcl-negf-physics-analysis-v5'



def test_native_cells_axes_invalid_reasons_and_explicit_components(tmp_path):
    bundle=fixture(tmp_path/'record')
    view=semantic_read(bundle,'spatial_energy_density',[dict(start=0,stop=2,step=1),dict(start=0,stop=3,step=2)])
    assert view['values']==[[-11.,None],[21.,None]]
    assert view['invalid_mask']==[[None,'nan'],[None,'negative_infinity']]
    assert view['axes'][1]['values']==[0.,7.] and view['axes'][1]['weights']==[.1,.7]
    assert view['integrity']['observations_file_verified'] is True
    assert semantic_read(bundle,'spectral_trace',[slice(0,2),1])['values']==[102.,202.]
    assert semantic_read(bundle,'effective_wavefunctions_imag',[1,2])['values']==-23.



def test_population_target_doping_not_selected_subset_sum(tmp_path):
    data=semantic_read(fixture(tmp_path/'with'),'effective_populations',[slice(0,2)])
    assert data['values']==[2.,4.]
    assert data['fractions']['values']==[.4,.8]
    assert data['fractions']['denominator']['value']==5.
    missing=semantic_read(fixture(tmp_path/'without',denominator=False),'effective_populations',[0])
    assert missing['fractions']['values'] is None
    assert missing['fractions']['reason']=='saved_target_doping_nonpositive_or_invalid'



def test_corrupt_sha_identity_and_budgets_refuse_before_payload(tmp_path,monkeypatch):
    from copy import deepcopy
    bundle=fixture(tmp_path/'record');root,selected,artifact=bundle
    with pytest.raises(ContractError): semantic_describe(bundle,expected_identity={**selected['identity'],'attempt':'3'})
    with pytest.raises(ContractError): semantic_read(bundle,'density',[slice(None)],limits=SavedViewLimits(maximum_selected_bytes=1))
    bad=deepcopy(artifact);bad['sha256']='0'*64
    with StateReader(root,state_ref=selected,expected_identity=selected['identity']) as owner:
        with pytest.raises(ContractError): AnalysisReader(owner,artifact_ref=bad).read('density',[0])
    monkeypatch.setattr(h5py,'File',lambda *a,**k:pytest.fail('over-cap owner opened'))
    with pytest.raises(ContractError): semantic_describe(bundle,limits=SavedViewLimits(maximum_owner_bytes=1))



def test_selection_and_total_allocations_are_checked_before_numeric_reads(tmp_path,monkeypatch):
    bundle=fixture(tmp_path/'record');original=h5py.Dataset.__getitem__
    def no_numeric(dataset,key):
        if dataset.dtype.kind in 'biufc': pytest.fail('refused selection read numeric array')
        return original(dataset,key)
    monkeypatch.setattr(h5py.Dataset,'__getitem__',no_numeric)
    for selection,limits in (([slice(None)],SavedViewLimits(maximum_selected_bytes=24)),
        ([dict(start=0,stop=4,step=1)],SavedViewLimits()),([dict(start=0,stop=3,step=0)],SavedViewLimits())):
        with pytest.raises(ContractError): semantic_read(bundle,'density',selection,limits=limits)
    with pytest.raises(ContractError): semantic_read(bundle,'../../state_dimensionless/GR',[0])



def test_missing_units_and_gauge_are_explicit_capabilities(tmp_path):
    from consumer_context_fixture import change
    for field in ('units','gauge'):
        bundle=fixture(tmp_path/field);root,selected,artifact=bundle
        def remove(value):
            if field=='units': del value['views']['density_per_m3']['units']
            else: del value['context']['effective_gauge']
        change(root,selected,artifact,remove)
        # Both are mandatory native5 contract fields, never optional capability.
        with pytest.raises(ContractError): semantic_describe(bundle)



def test_reader_closes_on_interrupted_view_and_reports_total_sha_delivery(tmp_path):
    bundle=fixture(tmp_path/'record')
    with pytest.raises(RuntimeError):
        with selected_view(bundle) as (view,owner):
            result=view.read('density',[0]);handles=list(owner._handles.values())
            raise RuntimeError('caller cancelled')
    assert all(v['stream']._raw.closed and v['raw'].closed for v in handles)
    assert not owner._handles
    with selected_view(bundle) as (view,owner):
        result=view.read('density',[0]);counters=result['io_counters']
        assert counters['sha_bytes']>=int(bundle[2]['bytes'])
        assert counters['bytes_read_total']>=int(bundle[2]['bytes'])
        assert counters['logical_selected_bytes']==24
        assert result['values']==0. and result['invalid_mask'] is None
        assert result['axes'][0]['values']==[0.]



def test_native_nested_assessment_positive_unknown_and_false_are_projected_verbatim(monkeypatch):
    monkeypatch.setattr(h5py, "File", lambda *a, **k: pytest.fail("brief opened HDF5"))
    certificate = {"scientific_accepted": True, "stationary_candidate_accepted": True,
                   "final_fixed_point_audit": {"status": "recorded_sentinel"}}
    native = {"id": "p1", "execution_id": "ex", "attempt": 2, "status": "completed", "quality": "strict",
              "converged": True, "observables": {"current_density_A_per_m2": 2., "scientific_assessment": certificate}}
    brief = brief_result(metadata([row(0), native, row(2)]), selection=SELECTION)
    assert brief["current_segments"] == [["p0", "p1", "p2"]]
    recorded = brief["points"][1]["recorded_statuses"]
    assert recorded["scientific_accepted"] is True
    assert recorded["stationary_assessment"] == certificate
    assert recorded["scientific_assessment"] == certificate
    assert recorded["assessment_source_path"] == "observables.scientific_assessment"
    for assessment, expected in ((None, None), ({"scientific_accepted": False}, False)):
        middle = {**native, "observables": {"current_density_A_per_m2": 2.}}
        if assessment is not None:
            middle["observables"]["scientific_assessment"] = assessment
        # An imagined top-level flag cannot promote missing native assessment.
        middle["scientific_accepted"] = True
        result = brief_result(metadata([row(0), middle, row(2)]), selection=SELECTION)
        assert result["current_segments"] == [["p0"], ["p2"]]
        assert result["points"][1]["recorded_statuses"]["scientific_accepted"] is expected


def test_attribute_datatype_and_scalar_fence_precedes_high_level_read(tmp_path,monkeypatch):
    from qcl_negf_results.native import _native_header
    with h5py.File(tmp_path/'attributes.h5','w') as h:
        for name in ('schema','schema_version','contract_set','artifact_role','representation','core_layout_sha256'):
            h.attrs[name]=np.bytes_('healthy')
        h.attrs.modify('schema',np.bytes_('healthy'))
        assert _native_header(h,1024)['schema']=='healthy'
        manager=type(h.attrs);original=manager.__getitem__
        def fenced(attributes,name):
            if name=='schema':pytest.fail('forbidden attribute reached high level read')
            return original(attributes,name)
        numeric=np.empty((),dtype=object);numeric[()]=np.array([1,2],dtype='i8')
        compound=np.empty((),dtype=np.dtype([('payload',h5py.vlen_dtype(np.dtype('i8')))]));compound[()]=(np.array([3,4],dtype='i8'),)
        for dtype,value in ((h5py.vlen_dtype(np.dtype('i8')),numeric),
                            (compound.dtype,compound),(np.dtype('f8'),np.array([1.,2.])),(np.dtype('f8'),np.float64(1.25))):
            del h.attrs['schema'];h.attrs.create('schema',value,dtype=dtype)
            monkeypatch.setattr(manager,'__getitem__',fenced)
            with pytest.raises(ContractError):_native_header(h,1024)
            monkeypatch.setattr(manager,'__getitem__',original)





@pytest.mark.parametrize("recorded,expected", [({"run_uuid": "A"}, {"run_uuid": "B"}),
    ({"calcjob_uuid": "job-A"}, {"calcjob_uuid": "job-B"}), ({"run_uuid": 1}, {"run_uuid": True})])
def test_native_only_foreign_identity_cannot_be_relabelled(tmp_path,recorded,expected):
    bundle=fixture(tmp_path/'record');root,selected,artifact=bundle
    # Native identity is closed: transport UUID extras are never aliases/proof.
    with pytest.raises(ContractError): semantic_describe(bundle,expected_identity={**selected['identity'],**expected})
    assert semantic_describe(bundle)['identity']==selected['identity']
    from consumer_context_fixture import change
    change(root,selected,artifact,lambda value:value['identity'].update(recorded))
    with pytest.raises(ContractError): semantic_describe(bundle)



def test_genuinely_absent_legacy_extra_identity_remains_unknown(tmp_path):
    bundle=fixture(tmp_path/'record')
    view=semantic_read(bundle,'density',[0])
    assert view['identity']==bundle[1]['identity']
    assert 'run_uuid' not in view['identity'] and 'calcjob_uuid' not in view['identity']
    with pytest.raises(ContractError): semantic_describe(bundle,expected_identity={**bundle[1]['identity'],'run_uuid':'provider-run','calcjob_uuid':'provider-job'})





@pytest.mark.parametrize("target", ["commit", "owner", "dependency"])
def test_final_symlink_swap_after_ancestry_check_is_rejected(tmp_path,monkeypatch,target):
    import os
    bundle=fixture(tmp_path/'record');root,state,artifact=bundle
    selected={'commit':root/state['manifest']['path'],'owner':root/artifact['path'],'dependency':root/'model.json'}[target]
    original_open=os.open;swapped=[]
    def swap_then_open(path,flags,*args,**kwargs):
        if os.fspath(path) in (str(selected),selected.name) and not swapped:
            held=selected.with_name(selected.name+'.held');selected.rename(held);selected.symlink_to(held.name);swapped.append(True)
        return original_open(path,flags,*args,**kwargs)
    monkeypatch.setattr(os,'open',swap_then_open)
    with pytest.raises(ContractError):semantic_describe(bundle)
    assert swapped==[True]



@pytest.mark.parametrize("kind", ["fifo", "socket", "directory"])
def test_regular_open_refuses_nonregular_without_blocking(tmp_path, monkeypatch, kind):
    import os
    import socket
    import qcl_negf_results.saved_views as saved
    path = tmp_path / "special"
    original_open = os.open
    descriptors = []
    def capture(*args, **kwargs):
        descriptor = original_open(*args, **kwargs)
        descriptors.append(descriptor)
        return descriptor
    monkeypatch.setattr(saved.os, "open", capture)
    sock = None
    try:
        if kind == "fifo": os.mkfifo(path)
        elif kind == "socket":
            sock = socket.socket(socket.AF_UNIX)
            sock.bind(str(path))
        else: path.mkdir()
        with pytest.raises(ContractError) as failure:
            saved._open_regular(path)
        assert failure.value.code == "corrupt_result"
    finally:
        if sock is not None: sock.close()
    for descriptor in descriptors:
        with pytest.raises(OSError): os.fstat(descriptor)


def test_opened_inode_size_check_closes_fd_and_never_stats_path(tmp_path, monkeypatch):
    import os
    from pathlib import Path
    import qcl_negf_results.saved_views as saved
    path = tmp_path / "owner"
    path.write_bytes(b"four")
    original_open = os.open
    descriptors = []
    def capture(*args, **kwargs):
        descriptor = original_open(*args, **kwargs)
        descriptors.append(descriptor)
        return descriptor
    monkeypatch.setattr(saved.os, "open", capture)
    monkeypatch.setattr(Path, "stat", lambda *a, **k: pytest.fail("size checked by pathname"))
    with pytest.raises(ContractError) as failure:
        saved._open_regular(path, expected_size=3)
    assert failure.value.code == "corrupt_result"
    assert len(descriptors) == 1
    with pytest.raises(OSError): os.fstat(descriptors[0])


@pytest.mark.parametrize("stage", ["fstat", "fdopen"])
def test_interruption_before_stream_transfer_closes_descriptor(tmp_path, monkeypatch, stage):
    import os
    import qcl_negf_results.saved_views as saved
    path = tmp_path / "small"
    path.write_bytes(b"tiny")
    original_open, original_fstat = os.open, os.fstat
    descriptors = []
    def capture(*args, **kwargs):
        descriptor = original_open(*args, **kwargs)
        descriptors.append(descriptor)
        return descriptor
    def interrupted(*args, **kwargs): raise KeyboardInterrupt("literal interrupt")
    monkeypatch.setattr(saved.os, "open", capture)
    monkeypatch.setattr(saved.os, stage, interrupted)
    with pytest.raises(KeyboardInterrupt, match="literal interrupt"):
        saved._open_regular(path)
    assert len(descriptors) == 1
    with pytest.raises(OSError): original_fstat(descriptors[0])


def test_missing_required_nofollow_primitive_has_no_follow_fallback(tmp_path, monkeypatch):
    import qcl_negf_results.saved_views as saved
    monkeypatch.delattr(saved.os, "O_NOFOLLOW")
    monkeypatch.setattr(saved.os, "open", lambda *a, **k: pytest.fail("unsupported open attempted"))
    with pytest.raises(ContractError) as failure:
        saved._open_regular(tmp_path / "absent")
    assert failure.value.code == "unsupported_contract"


@pytest.mark.parametrize("failure_type", [ContractError, KeyboardInterrupt])
def test_constructor_metadata_failure_closes_owned_inode(tmp_path,monkeypatch,failure_type):
    import os
    import qcl_negf_results.state as state
    bundle=fixture(tmp_path/'record');root,selected,artifact=bundle;owned=[]
    original=h5py.File
    def capture(stream,*args,**kwargs):
        if hasattr(stream,'_raw'): owned.append((stream._raw,stream._raw.fileno()))
        return original(stream,*args,**kwargs)
    def broken(*args,**kwargs):
        if failure_type is ContractError:raise ContractError('literal metadata error','corrupt_result')
        raise KeyboardInterrupt('literal metadata interruption')
    monkeypatch.setattr(h5py,'File',capture);monkeypatch.setattr(state,'_internal_storage',broken)
    with pytest.raises(failure_type):semantic_describe(bundle)
    assert len(owned)==1 and owned[0][0].closed
    with pytest.raises(OSError):os.fstat(owned[0][1])



@pytest.mark.parametrize("failure_type", [RuntimeError, KeyboardInterrupt])
def test_hdf_close_failure_still_closes_raw_and_repeated_close_is_safe(tmp_path,failure_type):
    import os
    bundle=fixture(tmp_path/'record');root,selected,artifact=bundle
    owner=StateReader(root,state_ref=selected,expected_identity=selected['identity'])
    view=AnalysisReader(owner,artifact_ref=artifact);view.describe();handles=list(owner._handles.values())
    descriptors=[v['raw'].fileno() for v in handles]
    def interrupted():raise failure_type('literal close failure')
    # Every stack must finish its existing HDF/stream/raw callbacks despite error.
    handles[0]['stack'].callback(interrupted)
    view.close();assert owner._handles
    with pytest.raises(failure_type,match='literal close failure'):owner.close()
    assert all(v['raw'].closed and v['stream']._raw.closed for v in handles)
    for fd in descriptors:
        with pytest.raises(OSError):os.fstat(fd)
    owner.close();view.close();assert not owner._handles



def test_owner_hash_and_hdf_share_open_inode_after_path_replacement(tmp_path,monkeypatch):
    import os
    bundle=fixture(tmp_path/'original');foreign_bundle=fixture(tmp_path/'foreign')
    ownerpath=bundle[0]/bundle[2]['path'];foreign=foreign_bundle[0]/foreign_bundle[2]['path']
    with h5py.File(foreign,'r+') as handle:handle['final_observations/arrays/density_per_m3'][0]=99.
    original=h5py.File;streams=[]
    def capture(stream,*args,**kwargs):
        handle=original(stream,*args,**kwargs)
        if hasattr(stream,'_raw') and not streams:
            streams.append(stream);os.replace(foreign,ownerpath)
        return handle
    monkeypatch.setattr(h5py,'File',capture)
    # Immutable source replacement must reject before publishing either inode's
    # values. Earlier v4 returned original0; native5 strengthens source truth.
    with pytest.raises(ContractError,match='changed'):semantic_read(bundle,'density',[0])
    assert len(streams)==1 and streams[0]._raw.closed



@pytest.mark.parametrize("stage", ["hdf_constructor", "owner_hash"])
def test_hdf_or_hash_error_closes_owner_descriptor(tmp_path,monkeypatch,stage):
    import os
    import qcl_negf_results.state as state
    import qcl_negf_results._capture_io as capture_io
    bundle=fixture(tmp_path/'record');original_regular=capture_io.regular;owned=[]
    @contextmanager
    def capture(path,**kwargs):
        with original_regular(path,**kwargs) as raw:
            if str(path).endswith('observations.h5'):owned.append((raw,raw.fileno()))
            yield raw
    error=ValueError('literal native read failure')
    def broken(*args,**kwargs):raise error
    monkeypatch.setattr(capture_io,'regular',capture)
    if stage=='hdf_constructor':monkeypatch.setattr(h5py,'File',broken)
    else:monkeypatch.setattr(state,'_verify_stream',broken)
    with pytest.raises(ContractError) as failure:semantic_describe(bundle)
    assert failure.value.code=='corrupt_result' and failure.value.__cause__ is error
    assert len(owned)==1 and owned[0][0].closed
    with pytest.raises(OSError):os.fstat(owned[0][1])



def test_raw_descriptor_closes_even_if_budget_stream_close_is_interrupted(tmp_path):
    import os
    bundle=fixture(tmp_path/'record');root,selected,artifact=bundle
    owner=StateReader(root,state_ref=selected,expected_identity=selected['identity'])
    view=AnalysisReader(owner,artifact_ref=artifact);view.describe()
    item=next(iter(owner._handles.values()));stream=item['stream'];raw=item['raw'];fd=raw.fileno();original_close=stream.close
    def interrupted():raise KeyboardInterrupt('literal stream close interruption')
    stream.close=interrupted
    # Borrowed wrapper owns no additional FD and is not an owner-stack callback.
    owner.close()
    assert raw.closed
    with pytest.raises(OSError):os.fstat(fd)
    owner.close();view.close();stream.close=original_close;stream.close()



def test_regular_open_uses_readonly_nofollow_cloexec_nonblock(tmp_path, monkeypatch):
    import os
    import qcl_negf_results.saved_views as saved
    path = tmp_path / "ordinary"
    path.write_bytes(b"literal")
    original_open = os.open
    flags_seen = []
    def capture(path, flags):
        flags_seen.append(flags)
        return original_open(path, flags)
    monkeypatch.setattr(saved.os, "open", capture)
    with saved._open_regular(path, expected_size=7) as raw:
        descriptor = raw.fileno()
        assert raw.read() == b"literal"
        assert os.get_inheritable(descriptor) is False
    assert len(flags_seen) == 1
    assert flags_seen[0] & os.O_ACCMODE == os.O_RDONLY
    for flag in (os.O_NOFOLLOW, os.O_CLOEXEC, os.O_NONBLOCK):
        assert flags_seen[0] & flag == flag
    with pytest.raises(OSError): os.fstat(descriptor)

# Native5 successor controls use only manufactured literal O1 cells.
def test_native5_borrowed_facade_metadata_brief_and_alias(tmp_path, monkeypatch):
    from consumer_context_fixture import fixture as native_fixture
    from qcl_negf_results.saved_views import SavedAnalysis
    from qcl_negf_results.state import StateReader
    selected, artifact, value, _, _ = native_fixture(tmp_path)
    original = h5py.Dataset.__getitem__
    def no_numeric(dataset, key):
        assert '/arrays/' not in dataset.name
        return original(dataset, key)
    with StateReader(tmp_path, state_ref=selected, expected_identity=selected['identity']) as owner:
        with SavedAnalysis(owner, artifact_ref=artifact) as view:
            assert SavedAnalysis is AnalysisReader
            monkeypatch.setattr(h5py.Dataset, '__getitem__', no_numeric)
            description = view.describe()
            assert description['source_state'] == selected
            assert description['recorded_statuses']['current'] == value['summary']
            assert description['views']['spatial_energy_density']['units'] == 'eV^-1 m^-3'
            brief = view.brief()
            assert brief['summary']['scientific_accepted'] is None
            assert brief['scalar_observables']['current_density_A_per_m2'] == value['views']['current_density_A_per_m2']
            assert brief['request'] == value['request']
        assert owner.source.state == selected


def test_native5_complex_selection_full_square_population_and_raw_locator(tmp_path):
    from consumer_context_fixture import fixture as native_fixture, change
    from qcl_negf_results.state import StateReader
    selected, artifact, _, _, _ = native_fixture(tmp_path)
    change(tmp_path, selected, artifact, lambda d: d['context']['inputs'].update(
        N_dop_2D_per_m2={'kind':'ieee','type':'f64','bits':'4024000000000000'},
        f_ion={'kind':'ieee','type':'f64','bits':'3fe0000000000000'}))
    with StateReader(tmp_path, state_ref=selected, expected_identity=selected['identity']) as owner:
        with AnalysisReader(owner, artifact_ref=artifact) as view:
            assert view.read('effective_wavefunctions_imag', [1,1])['values'] == 4.
            result = view.read('effective_populations', [slice(0,2)])
            assert result['values'] == [0.,3.]
            assert result['fractions']['values'] == [0.,.6]
            assert result['fractions']['denominator']['value'] == 5.
            assert result['raw_selection']['artifact'] == artifact
            assert result['raw_selection']['dataset'].endswith('/effective_density_matrix_per_m2')
            assert len(result['raw_selection']['selection']) == 2
            assert result['logical_axes'] == ['state']
            assert result['recorded_statuses']['current']['scientific_accepted'] is None


def test_native5_complex_preflight_counts_full_square_and_output_before_payload(tmp_path, monkeypatch):
    from consumer_context_fixture import semantic_fixture
    from qcl_negf_results.state import StateReader
    selected, artifact, _, _, _ = semantic_fixture(tmp_path)
    original = h5py.Dataset.__getitem__
    def no_numeric(dataset,key):
        assert '/arrays/' not in dataset.name
        return original(dataset,key)
    with StateReader(tmp_path,state_ref=selected,expected_identity=selected['identity']) as owner:
        monkeypatch.setattr(h5py.Dataset,'__getitem__',no_numeric)
        with AnalysisReader(owner,artifact_ref=artifact,limits=SavedViewLimits(maximum_selected_bytes=200)) as view:
            with pytest.raises(ContractError,match='complex data, output buffers'):
                view.read('effective_populations',[slice(0,2)])


def test_native5_narrow_matrix_output_containers_are_admitted_before_payload(tmp_path,monkeypatch):
    from consumer_context_fixture import fixture as native_fixture
    selected,artifact,_,_,_=native_fixture(tmp_path,nz=1,ne=32)
    original=h5py.Dataset.__getitem__
    def no_numeric(dataset,key):
        assert '/arrays/' not in dataset.name,'nested finite-output allocation must reject before payload'
        return original(dataset,key)
    with StateReader(tmp_path,state_ref=selected,expected_identity=selected['identity']) as owner:
        monkeypatch.setattr(h5py.Dataset,'__getitem__',no_numeric)
        with AnalysisReader(owner,artifact_ref=artifact,limits=SavedViewLimits(maximum_selected_bytes=15000)) as view:
            with pytest.raises(ContractError,match='output buffers'):
                view.read('spatial_energy_density',[slice(None),slice(None)])


def test_native5_public_wrappers_borrow_selected_owner_and_close_only_facade(tmp_path):
    from consumer_context_fixture import semantic_fixture
    selected,artifact,_,_,_=semantic_fixture(tmp_path)
    with StateReader(tmp_path,state_ref=selected,expected_identity=selected['identity']) as owner:
        description=describe_saved_views(owner,artifact_ref=artifact)
        result=read_saved_view(owner,'density',[slice(None)],artifact_ref=artifact)
        assert description['source_state']==result['source_state']==selected
        assert result['values']==[0.,-2.,8.]
        assert owner._handles and not owner._closed
        assert AnalysisReader(owner,artifact_ref=artifact).brief()['summary']['scientific_accepted'] is None


def test_native5_semantic_publication_rechecks_owner_after_finite_projection(tmp_path,monkeypatch):
    import qcl_negf_results.saved_views as saved
    bundle=fixture(tmp_path/'source');root,_,artifact=bundle;original=saved._finite;changed=[]
    def finite_then_tamper(value):
        result=original(value)
        if not changed and isinstance(value,np.ndarray) and value.shape==(3,) and np.array_equal(value,[0.,-2.,8.]):
            with (root/artifact['path']).open('r+b') as raw:
                raw.seek(-1,2);cell=raw.read(1);raw.seek(-1,2);raw.write(bytes([cell[0]^1]))
            changed.append(True)
        return result
    monkeypatch.setattr(saved,'_finite',finite_then_tamper)
    with pytest.raises(ContractError) as failure:semantic_read(bundle,'density',[slice(None)])
    assert failure.value.code=='source_changed' and changed==[True]


def test_native5_publication_guard_preserves_single_open_owner_cap(tmp_path):
    from qcl_negf_results.state import ResultReadLimits
    root,selected,artifact=fixture(tmp_path/'source')
    with StateReader(root,state_ref=selected,expected_identity=selected['identity'],limits=ResultReadLimits(maximum_open_files=1)) as owner:
        with AnalysisReader(owner,artifact_ref=artifact) as view:
            assert view.describe()['source_state']==selected
            assert view.read('density',[slice(None)])['values']==[0.,-2.,8.]
            assert view.brief()['summary']['scientific_accepted'] is None
            assert len(owner._handles)==1
