"""Canonical native5 envelopes; retained scalar/provenance/budget oracles."""
from copy import deepcopy
import json
import h5py
import pytest
from qcl_negf_contracts.messages import ContractError
from qcl_negf_results import diagnostic_views as views
from qcl_negf_results.state import StateReader
from result_v2_fixture import (IDENTITY, record, source, integer, symbol, put,
    producer_root, producer_state, pointer, append)


def request(stage='scba',fields=('r_K','J'),maximum_records=256):
    assert hasattr(views,'HistoryViewRequest'), 'canonical diagnostic request API missing'
    return views.HistoryViewRequest(stage,fields,maximum_records)


def rows(bits=('0000000000000000','bff0000000000000','7ff8000000001234','7ff0000000000000','fff0000000000000')):
    result=[]
    for index,b in enumerate(bits,1):
        row=record(index,index*3)
        core=row['core']['fields']; core['row']['fields']['r_K']['bits']=b
        core['units'].append(dict(type='Type_458abebe7784aa53',fields=dict(first=symbol('r_K'),second='1')))
        result.append(row)
    return result


def test_metadata_does_not_read_numeric_or_full_owner(tmp_path,monkeypatch):
    ref=source(tmp_path,rows())
    with StateReader(tmp_path,state_ref=ref,expected_identity=IDENTITY) as reader:
        monkeypatch.setattr(h5py,'File',lambda *a,**kw:pytest.fail('describe opened HDF'))
        result=views.describe_saved_diagnostics(reader)
        assert result['source']==ref and result['history']['records']=='5'
        assert result['units']['status']=='not_loaded'
        assert result['resources']['checkpoint']['status']=='not_available'
        assert result['resources']['performance_summary']['status']=='not_available'


def test_exact_selected_rows_native_cycles_units_and_full_owner_hash(tmp_path):
    data=rows(('3fb999999999999a',)*3)
    for row in data: row['core']['fields']['sequence']['decimal']=row['stream_ordinal']
    ref=source(tmp_path,data)
    with StateReader(tmp_path,state_ref=ref,expected_identity=IDENTITY) as reader:
        reader.import_history(); result=views.read_saved_history(reader,request(maximum_records=2))
        assert [r['stream_ordinal'] for r in result.rows]==['1','2']
        assert [r['break_before'] for r in result.rows]==[True,False]
        assert result.rows[0]['cells']['J']==dict(tagged=data[0]['core']['fields']['row']['fields']['J'],
            value=3.,invalid_reason=None,units='A/m^2',missing_reason=None)
        assert result.io_counters['sha_bytes']>=int(result.cursor['leaf']['bytes'])
        assert not result.at_cutoff
        last=views.read_saved_history(reader,request(),cursor=result.cursor)
        assert last.at_cutoff and last.cursor['last_record']=='3'


def test_nonfinite_masks_preserve_zero_negative_and_disconnect_omissions(tmp_path):
    ref=source(tmp_path,rows())
    with StateReader(tmp_path,state_ref=ref,expected_identity=IDENTITY) as reader:
        reader.import_history(); result=views.DiagnosticReader(reader).read(request())
        cells=[r['cells']['r_K'] for r in result.rows]
        assert [r['value'] for r in cells]==[0.,-1.,None,None,None]
        assert [r['invalid_reason'] for r in cells]==[None,None,'nan','+Inf','-Inf']
        assert all(r['break_before'] for r in result.rows)
        assert cells[2]['tagged']['bits']=='7ff8000000001234'


def test_exact_large_int_coordinates_and_unknown_units(tmp_path):
    row=record(sequence=9007199254740992); row['core']['fields']['units']=[]
    ref=source(tmp_path,[row])
    with StateReader(tmp_path,state_ref=ref,expected_identity=IDENTITY) as reader:
        reader.import_history(); result=views.DiagnosticReader(reader).read(request(fields=('J',)))
        assert result.rows[0]['sequence']==integer(9007199254740992)
        assert result.rows[0]['cells']['J']['units'] is None


def test_phase_filter_consumes_events_and_returns_unchanged_canonical_cursor(tmp_path):
    ref=source(tmp_path,rows(('0000000000000000',)*2))
    with StateReader(tmp_path,state_ref=ref,expected_identity=IDENTITY) as reader:
        reader.import_history(); result=views.DiagnosticReader(reader).read(request('outer',('J',),1))
        assert result.rows==() and len(result.records)==1 and not result.at_cutoff
        assert result.cursor['last_record']=='1'
        assert reader.history_batch(result.cursor).cursor['last_record']=='2'


def test_actual_outer_vectors_remain_lazy_and_complete(producer_root):
    ref=producer_state(producer_root,'history')
    with StateReader(producer_root,state_ref=ref,expected_identity=ref['identity']) as reader:
        reader.import_history(); result=views.DiagnosticReader(reader).read(request('outer',('J',)))
        assert len(result.rows)==1
        outer=result.records[1]['core']['fields']['row']['fields']
        assert reader.read_array(outer['populations'],(slice(None),)).values.tolist()==[.25,.75]
        assert reader.read_array(outer['density'],(slice(None),)).values.tolist()==[1.,2.,3.]


@pytest.mark.parametrize('kind',['domain','basis','outer','restart'])
def test_boundaries_do_not_connect_incomparable_rows(tmp_path,kind):
    data=rows(('0000000000000000',)*2)
    for row in data: row['core']['fields']['sequence']['decimal']=row['stream_ordinal']
    core=data[1]['core']['fields']
    if kind=='domain': core['domain_revision']=integer(1)
    elif kind=='basis': core['basis_id']='different'
    elif kind=='outer': core['outer_iteration']=integer(2)
    else: core['inner_iteration']=integer(1);core['row']['fields']['ν']=integer(1)
    data[1]['row_key']={key:deepcopy(core[key]) for key in data[1]['row_key']}
    ref=source(tmp_path,data)
    with StateReader(tmp_path,state_ref=ref,expected_identity=IDENTITY) as reader:
        reader.import_history(); result=views.DiagnosticReader(reader).read(request())
        assert [r['break_before'] for r in result.rows]==[True,True]


def test_missing_history_keeps_exact_source_and_checkpoint_absence(tmp_path):
    ref=source(tmp_path)
    with StateReader(tmp_path,state_ref=ref,expected_identity=IDENTITY) as reader:
        assert views.describe_saved_diagnostics(reader)['history']['records']=='0'
        reader.import_history(); result=views.DiagnosticReader(reader).read(request())
        assert result.at_cutoff and not result.records and result.cursor['last_record']=='0'


@pytest.mark.parametrize('field',['run_id','execution_id','point_id','attempt','branch_id','model_sha256','plan_sha256'])
def test_wrong_exact_binding_refuses_before_owner_open(tmp_path,monkeypatch,field):
    ref=source(tmp_path,[record()]); foreign=deepcopy(IDENTITY)
    foreign[field]='f'*64 if field.endswith('sha256') else 'foreign'
    monkeypatch.setattr(h5py,'File',lambda *a,**kw:pytest.fail('foreign source opened HDF'))
    with pytest.raises(ContractError): StateReader(tmp_path,state_ref=ref,expected_identity=foreign)


def test_unselected_analysis_bytes_never_gate_history(tmp_path):
    ref=source(tmp_path,[record()]); manifest=json.loads((tmp_path/ref['manifest']['path']).read_text())
    manifest['artifacts'].append(dict(path='unselected.h5',bytes='999999999',sha256='f'*64,
        media_type='application/x-hdf5',schema='qcl-negf-physics-analysis-v5',role='physics.analysis',profile='science',dependencies=[]))
    ref['manifest']=put(tmp_path,'selected.json',manifest,manifest['schema'])
    with StateReader(tmp_path,state_ref=ref,expected_identity=IDENTITY) as reader:
        reader.import_history(); assert views.DiagnosticReader(reader).read(request()).rows[0]['cells']['J']['value']==3.
    assert not (tmp_path/'unselected.h5').exists()


@pytest.mark.parametrize('bad',[('scba',('J','J'),1),('x',('J',),1),('scba',('row/J',),1),
                             ('scba',('populations',),1),('scba',('J',),True)])
def test_request_fields_and_counts_are_explicit(bad):
    with pytest.raises((ValueError,TypeError)): request(*bad)


@pytest.mark.parametrize('limit,value',[('maximum_rows',1),('maximum_columns',1),
    ('maximum_selected_bytes',1),('maximum_owner_bytes',1),('maximum_source_entries',1)])
def test_bad_selection_and_caps_refuse_before_numeric_read(tmp_path,monkeypatch,limit,value):
    ref=source(tmp_path,[record(),record(2,11)])
    with StateReader(tmp_path,state_ref=ref,expected_identity=IDENTITY) as reader:
        reader.import_history(); original=h5py.Dataset.__getitem__
        def guarded(dataset,key):
            assert '/records/' not in dataset.name, 'cap crossed before record read'
            return original(dataset,key)
        monkeypatch.setattr(h5py.Dataset,'__getitem__',guarded)
        with pytest.raises(ContractError):
            views.DiagnosticReader(reader,limits=views.DiagnosticViewLimits(**{limit:value})).read(request())


def test_json_response_and_common_io_caps_fail_closed(tmp_path):
    ref=source(tmp_path,[record()])
    with StateReader(tmp_path,state_ref=ref,expected_identity=IDENTITY) as reader:
        with pytest.raises(ContractError): views.describe_saved_diagnostics(reader,limits=views.DiagnosticViewLimits(maximum_json_bytes=1))
        reader.import_history()
        with pytest.raises(ContractError): views.DiagnosticReader(reader,limits=views.DiagnosticViewLimits(maximum_io_bytes=1)).read(request())


def test_explicit_missing_reasons_and_no_resource_zero_defaults(tmp_path):
    row=record()
    from importlib.resources import files
    graph=json.loads(files('qcl_negf_contracts').joinpath('core_layout.json').read_text())['graph']['nodes']
    pair=next(name for name,node in graph.items() if node.get('kind')=='struct' and
        node.get('fields')==[{'name':'first','type':'Symbol'},{'name':'second','type':'Symbol'}])
    row['core']['fields']['missing_reasons']=[dict(type=pair,fields=dict(first=symbol('J'),second=symbol('no_previous_iteration')))]
    ref=source(tmp_path,[row])
    with StateReader(tmp_path,state_ref=ref,expected_identity=IDENTITY) as reader:
        reader.import_history(); result=views.DiagnosticReader(reader).read(request())
        assert result.rows[0]['cells']['J']['missing_reason']=='no_previous_iteration'
        assert views.describe_saved_diagnostics(reader)['resources']['performance_summary']['status']=='not_available'


def test_pointer_rebase_and_foreign_cursor_are_explicit(tmp_path):
    ref=source(tmp_path,[record()]);pointer(tmp_path,ref)
    with StateReader.from_pointer(tmp_path,expected_identity=IDENTITY,pointer='live') as reader:
        reader.import_history(); first=views.DiagnosticReader(reader).read(request())
        future=append(tmp_path,ref,record(2,11));pointer(tmp_path,future);reader.poll()
        with pytest.raises(ContractError): views.DiagnosticReader(reader).read(request(),cursor=first.cursor)
        rebased=reader.rebase_cursor(first.cursor)
        assert views.DiagnosticReader(reader).read(request(),cursor=rebased).cursor['last_record']=='2'
        rebased['identity']['attempt']='foreign'
        with pytest.raises(ContractError): views.DiagnosticReader(reader).read(request(),cursor=rebased)


def test_description_never_runs_full_validator_progress_or_summary(tmp_path,monkeypatch):
    import qcl_negf_results.progress as progress
    import qcl_negf_results.presentation as presentation
    ref=source(tmp_path,[record()])
    def forbidden(*a,**kw): pytest.fail('unbounded/full/mutating helper called')
    for name in ('live_history','materialize_history','_inspect','_attempt_spans'): monkeypatch.setattr(progress,name,forbidden)
    monkeypatch.setattr(presentation,'read_performance_summary',forbidden)
    with StateReader(tmp_path,state_ref=ref,expected_identity=IDENTITY) as reader:
        views.describe_saved_diagnostics(reader);reader.import_history()
        assert views.DiagnosticReader(reader).read(request()).at_cutoff


def test_description_provenance_is_detached_from_verified_manifest(tmp_path):
    ref=source(tmp_path,[record()])
    with StateReader(tmp_path,state_ref=ref,expected_identity=IDENTITY) as reader:
        original=deepcopy(reader._manifest['provenance'])
        result=views.describe_saved_diagnostics(reader)
        result['provenance']['git_graph']['components'][0]['revision']='f'*40
        assert reader._manifest['provenance']==original


def test_warm_record_proof_does_not_bypass_tighter_chunk_budget(tmp_path,monkeypatch):
    import numpy as np
    ref=source(tmp_path,[record(),record(2,11)])
    with h5py.File(tmp_path/'records.h5','r+') as f:
        values=f['records/json_utf8'][:]
        del f['records/json_utf8']
        f.create_dataset('records/json_utf8',data=values,chunks=(8192,),maxshape=(None,),compression='gzip')
    page=json.loads((tmp_path/'catalog.json').read_text())
    from result_v2_fixture import reference
    page['artifacts'][-1].update(reference(tmp_path,'records.h5','qcl-negf-scientific-history-v5','application/x-hdf5'))
    manifest=json.loads((tmp_path/'manifest.json').read_text())
    manifest['history']['head']=put(tmp_path,'catalog.json',page,page['schema'])
    ref['manifest']=put(tmp_path,'manifest.json',manifest,manifest['schema'])
    with StateReader(tmp_path,state_ref=ref,expected_identity=IDENTITY) as reader:
        reader.import_history(); first=reader.history_batch(maximum_records=1)
        original=h5py.Dataset.__getitem__
        def forbidden(dataset,key):
            assert '/records/' not in dataset.name, 'warm chunk crossed tighter consumer cap'
            return original(dataset,key)
        monkeypatch.setattr(h5py.Dataset,'__getitem__',forbidden)
        with pytest.raises(ContractError,match='chunk'):
            views.DiagnosticReader(reader,limits=views.DiagnosticViewLimits(maximum_selected_bytes=4096)).read(
                request(fields=('J',),maximum_records=1),cursor=first.cursor)


@pytest.mark.parametrize('name',['updates','child','grandchild'])
def test_actual_updates_keep_all_records_and_original_source_proofs(producer_root,name):
    ref=producer_state(producer_root,name)
    with StateReader(producer_root,state_ref=ref,expected_identity=ref['identity']) as raw:
        raw.import_history(); expected=raw.history_batch()
    with StateReader(producer_root,state_ref=ref,expected_identity=ref['identity']) as reader:
        reader.import_history(); batch=views.DiagnosticReader(reader).read(request(fields=('J',)))
        assert batch.records==expected.records
        assert batch.witness_updates==tuple(r for r in expected.records if r['target'] is not None)
        assert all(r['target'] is None for r in expected.records if r['stream_ordinal'] in
            {row['stream_ordinal'] for row in batch.rows})
        assert len(batch.rows)+len(batch.witness_updates)==len(batch.records)


@pytest.mark.parametrize('tamper',[False,True])
def test_f1_sealed_leaf_substitute_stays_rejected_through_projection(tmp_path,tamper):
    from test_result_v2_cursor import _sealed_update_context
    ref,cursor=_sealed_update_context(tmp_path,tamper=tamper)
    with StateReader(tmp_path,state_ref=ref,expected_identity=IDENTITY) as reader:
        reader.import_history()
        if tamper:
            with pytest.raises(ContractError): views.DiagnosticReader(reader).read(request(),cursor=cursor)
        else:
            assert views.DiagnosticReader(reader).read(request(),cursor=cursor).at_cutoff


def test_required_parent_hdf_proofs_keep_consumer_owner_admission(producer_root,monkeypatch):
    ref=producer_state(producer_root,'child')
    with StateReader(producer_root,state_ref=ref,expected_identity=ref['identity']) as reader:
        reader.import_history()
        original=StateReader._open_hdf; observed=[]
        def checked(session,reference,**kwargs):
            if session is not reader:
                observed.append(reference['path'])
                assert getattr(session,'_consumer_owner_limit',None)==64*1024*1024, 'ancestor lost consumer owner allowance'
            return original(session,reference,**kwargs)
        monkeypatch.setattr(StateReader,'_open_hdf',checked)
        assert views.DiagnosticReader(reader).read(request(fields=('J',))).at_cutoff
        assert observed


def test_required_parent_json_bytes_are_metadata_in_shared_counters(producer_root,monkeypatch):
    ref=producer_state(producer_root,'child')
    with StateReader(producer_root,state_ref=ref,expected_identity=ref['identity']) as reader:
        reader.import_history(); original=StateReader._json; observed=[]
        def checked(session,reference):
            before=session.io_counters
            value=original(session,reference)
            after=session.io_counters
            observed.append(reference['path'])
            assert after['metadata_bytes']-before['metadata_bytes']==int(reference['bytes']), 'nested JSON misclassified as HDF bytes'
            assert after['bytes_read_total']-before['bytes_read_total']==int(reference['bytes'])
            return value
        monkeypatch.setattr(StateReader,'_json',checked)
        assert views.DiagnosticReader(reader).read(request(fields=('J',))).at_cutoff
        assert observed
