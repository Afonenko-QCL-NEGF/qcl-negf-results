"""Whole-call Results metadata allowance, never a science/native certificate."""
from copy import deepcopy
from dataclasses import fields,is_dataclass
import json
from pathlib import Path

import pytest

from qcl_negf_contracts.messages import ContractError
from qcl_negf_results import final_observations as api
from qcl_negf_results.state import ResultReadLimits,StateReader
from consumer_context_fixture import fixture

M=1024*1024


def context_fixture(root, *, final=False):
    state,observations,value,checkpoint,_=fixture(root,plan_kind='scientific_plan_v2')
    manifest=json.loads((root/state['manifest']['path']).read_bytes())
    refs={a['role']:{k:a[k] for k in ('path','bytes','sha256','schema','media_type')} for a in manifest['artifacts'] if a['role'] in ('plan','model')}
    return dict(artifact_root=root,state_ref=state,expected_identity=state['identity'],plan_ref=refs['plan'],model_ref=refs['model'],
        diagnostics_ref=value['diagnostics'] if final else None,observations_ref=observations if final else None,
        limits=ResultReadLimits(maximum_metadata_bytes=2*M,maximum_io_bytes=16*M,maximum_catalog_bytes=2*M,
            maximum_catalog_pages=16,maximum_catalog_entries=256,maximum_open_files=4,maximum_array_bytes=65536,
            maximum_record_bytes=65536,maximum_batch_bytes=65536,maximum_batch_records=16,maximum_hdf_objects=4096,maximum_hdf_depth=32),
        maximum_context_bytes=128*M),manifest,checkpoint,value


def bridge():
    function=getattr(api,'describe_native_metadata_context',None)
    assert callable(function),'missing one public bounded metadata-context operation'
    return function


def detached(value):
    if is_dataclass(value):
        for f in fields(value):detached(getattr(value,f.name))
    elif type(value) is dict:
        for k,v in value.items():assert type(k) is str;detached(v)
    elif type(value) in (tuple,list):
        for v in value:detached(v)
    else:assert value is None or type(value) in (str,bytes,int,float,bool),type(value)


def test_full_original_checkpoint_detached_tuple_and_whole_call_counters(tmp_path,monkeypatch):
    args,manifest,checkpoint,_=context_fixture(tmp_path);original=deepcopy(args);closed=[]
    close=StateReader.close
    def closing(reader):close(reader);closed.append(reader)
    monkeypatch.setattr(StateReader,'close',closing)
    result=bridge()(**args)
    assert type(result) is tuple and len(result)==5
    actual,cp,inputs,diagnostics,observations=result
    assert actual==manifest and cp==checkpoint and diagnostics is observations is None
    assert inputs.plan_raw==(tmp_path/'plan.json').read_bytes()
    assert inputs.model_raw==(tmp_path/'model.json').read_bytes()
    assert inputs.source==args['state_ref'] and inputs.io_counters['bytes_read_total']>int(args['state_ref']['manifest']['bytes'])
    assert inputs.io_counters['maximum_io_bytes']==16*M
    assert closed and all(r._closed and not r._handles for r in closed)
    detached(result)
    actual['identity']['point_id']='changed';cp['identity']['point_id']='changed';inputs.source['identity']['point_id']='changed'
    assert args==original
    again=bridge()(**args)
    assert again[0]==manifest and again[1]==checkpoint


@pytest.mark.parametrize('field,bad',[('maximum_context_bytes',True),('maximum_context_bytes',0),('maximum_context_bytes',128*M+1),('maximum_context_bytes','134217728'),('maximum_retained_bytes',True),('maximum_retained_bytes',0),('maximum_retained_bytes',128*M+1),('maximum_retained_bytes','4194304')])
def test_bad_budget_before_root_access(tmp_path,monkeypatch,field,bad):
    args,_,_,_=context_fixture(tmp_path);args[field]=bad;calls=[]
    monkeypatch.setattr(StateReader,'_root_identity',lambda self:calls.append(True))
    with pytest.raises((ValueError,TypeError,ContractError)):bridge()(**args)
    assert calls==[]


@pytest.mark.parametrize('final',[False,True])
def test_whole_context_originals(tmp_path,monkeypatch,final):
    args,manifest,checkpoint,value=context_fixture(tmp_path,final=final);arenas=[]
    original=api._MetadataArena;forecast=[]
    from qcl_negf_results.state import _context_weight,_context_json_size
    contract=api._context_contract
    def measured_contract(name,value,**contexts):
        if name=='validate_final_observations':
            forecast.append({k:dict(weight=_context_weight(v),json_bytes=_context_json_size(v))
                for k,v in dict(value=value,**contexts).items() if v is not None})
        return contract(name,value,**contexts)
    class MeasuredArena(original):
        def __init__(self,*a,**k):super().__init__(*a,**k);arenas.append(self)
    monkeypatch.setattr(api,'_MetadataArena',MeasuredArena)
    monkeypatch.setattr(api,'_context_contract',measured_contract)
    result=bridge()(**args)
    assert result[0]==manifest and result[1]==checkpoint
    if final:
        assert result[4].declaration==value
        assert result[2].io_counters==result[3].io_counters==result[4].io_counters
    assert 112*M<=arenas[0].high_water<=128*M
    print('CONTEXT_SUCCESS',final,result[2].io_counters,arenas[0].trace)


def test_one_io_object_initialization_nested_stability_and_close(tmp_path,monkeypatch):
    from dataclasses import replace
    from qcl_negf_results import state
    args,_,_,_=context_fixture(tmp_path)
    operations=[];readers=[];stability=[];deliveries=[]
    original_operation=StateReader._operation;original_close=StateReader.close
    original_stability=StateReader._check_source_bindings;original_account=state._IOBudget.account
    def operation(reader):
        original_operation(reader);operations.append(reader._budget)
    def closing(reader):
        readers.append((reader,reader._budget,reader._budget.bytes_read_total));original_close(reader)
    def stable(reader):
        original_stability(reader);stability.append((reader._budget,reader._budget.bytes_read_total))
    def account(budget,count):
        assert budget.bytes_read_total+count<=budget.maximum
        original_account(budget,count);deliveries.append((budget,count))
    monkeypatch.setattr(StateReader,'_operation',operation)
    monkeypatch.setattr(StateReader,'close',closing)
    monkeypatch.setattr(StateReader,'_check_source_bindings',stable)
    monkeypatch.setattr(state._IOBudget,'account',account)
    result=bridge()(**args);budget=operations[0]
    assert all(b is budget for b in operations)
    assert len(stability)==1 and all(b is budget for b,_ in stability)
    assert all(b is budget for _,b,_ in readers)
    assert sum(n for b,n in deliveries if b is budget)==result[2].io_counters['bytes_read_total']
    assert result[2].io_counters['bytes_read_total']==budget.bytes_read_total
    print('SHARED_IO',len(operations),len(stability),len(readers),result[2].io_counters)
    args['limits']=replace(args['limits'],maximum_io_bytes=budget.bytes_read_total-1)
    before=len(operations)
    with pytest.raises(ContractError,match='I/O budget'):bridge()(**args)
    failed=operations[before]
    assert failed.bytes_read_total<=failed.maximum and failed is not budget
    assert all(b is failed for b in operations[before:])
    assert state._context_arena.get() is None


@pytest.mark.parametrize('cancel',[False,True])
def test_primary_failure_or_cancellation_survives_cleanup(tmp_path,monkeypatch,cancel):
    from qcl_negf_results import state
    args,_,_,_=context_fixture(tmp_path);closed=[]
    primary=KeyboardInterrupt('cancelled') if cancel else ContractError('primary failure','corrupt_result')
    original=StateReader.close
    def fail(reader):raise primary
    def close(reader):
        original(reader);closed.append(reader)
        raise RuntimeError('secondary close')
    monkeypatch.setattr(StateReader,'_load_checkpoint',fail)
    monkeypatch.setattr(StateReader,'close',close)
    with pytest.raises(type(primary)) as caught:bridge()(**args)
    assert caught.value is primary and any('cleanup also raised RuntimeError' in n for n in primary.__notes__)
    assert len(closed)==1 and closed[0]._closed and not closed[0]._handles
    assert state._context_arena.get() is None


def test_successful_close_error_is_not_silenced(tmp_path,monkeypatch):
    from qcl_negf_results import state
    args,_,_,_=context_fixture(tmp_path);closed=[];original=StateReader.close
    def close(reader):original(reader);closed.append(reader);raise RuntimeError('close failed')
    monkeypatch.setattr(StateReader,'close',close)
    with pytest.raises(RuntimeError,match='close failed'):bridge()(**args)
    assert len(closed)==1 and closed[0]._closed
    assert state._context_arena.get() is None


def test_independent_weight_alias_copies_and_preallocation_oracle(monkeypatch):
    from qcl_negf_results import state
    def oracle(v):
        if v is None or type(v) in (int,float,bool):return 64
        if type(v) in (str,bytes):return 128+6*len(v)
        if type(v) is dict:return 128+sum(oracle(k)+oracle(x) for k,x in v.items())
        return 128+sum(64+oracle(x) for x in v)
    value={'alpha':[1,'β',None],'other':{'number':False}}
    assert state._context_weight(value)==oracle(value)
    arena=state._MetadataArena(128*M)
    arena.held.extend((value,value,deepcopy(value)))
    arena.before('independent resident oracle')
    assert arena.trace[-1][1]==2*oracle(value)
    clone_calls=[];monkeypatch.setattr(state.copy,'deepcopy',lambda v:clone_calls.append(v))
    arena.maximum=arena.trace[-1][3]+1024
    token=state._context_arena.set(arena)
    try:
        with pytest.raises(ContractError,match='before deepcopy'):state._context_copy(value)
        assert clone_calls==[]
        assert arena.failure['requested']==112*M+oracle(value)
        assert arena.failure['total']>arena.maximum
    finally:state._context_arena.reset(token)


def test_raw_json_bound_and_recursive_contract_forecast_precede_work():
    from qcl_negf_results import state
    value={'unicode':'☃\\n','rows':[{'value':1},None,True,[]]}
    raw=json.dumps(value,ensure_ascii=False).encode()
    assert state._context_raw_tree_bound(raw)>=state._context_weight(value)
    arena=state._MetadataArena(128*M)
    arena.maximum=arena.trace[-1][3]+1000
    token=state._context_arena.set(arena)
    try:
        with pytest.raises(ContractError,match='before JSON'):state._context_decoding(raw,'oracle')
        assert arena.failure['requested']>=3*state._context_weight(value)
        with pytest.raises(ContractError,match='Contracts validate_checkpoint_declaration'):
            state._context_contract('validate_checkpoint_declaration',value,manifest={'large':'x'*5000})
        assert arena.failure['requested']>2*state._context_weight(value)
    finally:state._context_arena.reset(token)


@pytest.mark.parametrize('mutation',['wrong_identity','wrong_plan_pin','wrong_model_role','one_final_ref','foreign_nested','cycle'])
def test_original_reference_and_argument_negatives(tmp_path,monkeypatch,mutation):
    from qcl_negf_results import state
    args,_,_,value=context_fixture(tmp_path);readers=[];original=StateReader.close
    def close(reader):original(reader);readers.append(reader)
    monkeypatch.setattr(StateReader,'close',close)
    if mutation=='wrong_identity':args['expected_identity']=dict(args['expected_identity'],point_id='different')
    elif mutation=='wrong_plan_pin':args['plan_ref']=dict(args['plan_ref'],sha256='0'*64)
    elif mutation=='wrong_model_role':args['model_ref']=deepcopy(args['plan_ref'])
    elif mutation=='one_final_ref':args['diagnostics_ref']=value['diagnostics']
    elif mutation=='foreign_nested':args['state_ref']=dict(args['state_ref'],identity=object())
    else:args['state_ref']['cycle']=args['state_ref']
    with pytest.raises((ContractError,ValueError,TypeError)):bridge()(**args)
    assert all(r._closed and not r._handles for r in readers)
    assert state._context_arena.get() is None


def test_two_states_nondefault_dimensions_and_yaml_original_bytes(tmp_path,monkeypatch):
    from qcl_negf_results import state
    first=tmp_path/'first';second=tmp_path/'second';first.mkdir();second.mkdir()
    args,_,_,_=context_fixture(first)
    original=bridge()(**args)
    selected,_,_,checkpoint,_=fixture(second,nz=5,ne=4,nk=3,nb=2)
    manifest=json.loads((second/selected['manifest']['path']).read_bytes())
    refs={row['role']:{k:row[k] for k in ('path','bytes','sha256','schema','media_type')} for row in manifest['artifacts'] if row['role'] in ('plan','model')}
    args.update(artifact_root=second,state_ref=selected,expected_identity=selected['identity'],plan_ref=refs['plan'],model_ref=refs['model'])
    monkeypatch.setattr(StateReader,'_read_selected_array',lambda *a,**k:pytest.fail('numerical payload selected'))
    result=bridge()(**args)
    assert result[0]==manifest and result[1]==checkpoint
    assert result[2].plan_raw==(second/'plan.yaml').read_bytes()
    assert original[2].source['identity']!=result[2].source['identity']
    assert original[2].io_counters['maximum_io_bytes']==result[2].io_counters['maximum_io_bytes']==16*M
    assert state._context_arena.get() is None


def test_final_retained_cap_includes_final_counter_headers(tmp_path):
    from qcl_negf_results.state import _context_weight
    args,_,_,_=context_fixture(tmp_path,final=True)
    original=bridge()(**args)
    retained=_context_weight(original)
    assert bridge()(**args,maximum_retained_bytes=retained)==original
    with pytest.raises(ContractError,match='retained metadata'):
        bridge()(**args,maximum_retained_bytes=retained-1)
    assert original[4] is not None


def test_validation_envelope_refuses_old_capacity_before_factory_or_root(tmp_path,monkeypatch):
    args,_,_,_=context_fixture(tmp_path);args['maximum_context_bytes']=4*M
    calls=[]
    monkeypatch.setattr(StateReader,'_root_identity',lambda self:calls.append('root'))
    monkeypatch.setattr(api,'schema_validator',lambda *a:calls.append('factory'))
    with pytest.raises(ContractError,match='arena exceeded'):bridge()(**args)
    assert calls==[]


@pytest.mark.parametrize('schema',['result-manifest.schema.json','telemetry-envelope.schema.json',
    'producer-phase-source.schema.json','producer-phase-segment.schema.json'])
@pytest.mark.parametrize('mutation',['schema','execution'])
def test_mutated_cached_schema_holds_before_factory_or_root(tmp_path,monkeypatch,mutation,schema):
    import qcl_negf_contracts.results as contracts
    args,_,_,_=context_fixture(tmp_path)
    cached=contracts._validator(schema)
    old=cached.schema.get('title');present='title' in cached.schema;execution=cached._validators
    if mutation=='schema':cached.schema['title']='unqualified mutated cache'
    else:cached._validators=[]
    roots=[];monkeypatch.setattr(StateReader,'_root_identity',lambda self:roots.append(True))
    try:
        with pytest.raises(ContractError,match='validation profile'):bridge()(**args)
        assert roots==[]
    finally:
        if mutation=='schema':
            if present:cached.schema['title']=old
            else:del cached.schema['title']
        cached._validators=execution


def test_reentry_and_foreign_factory_hold_before_root(tmp_path,monkeypatch):
    from qcl_negf_results import state
    args,_,_,_=context_fixture(tmp_path);roots=[]
    monkeypatch.setattr(StateReader,'_root_identity',lambda self:roots.append(True))
    token=state._context_arena.set(object())
    try:
        with pytest.raises(ContractError,match='reentr'):bridge()(**args)
    finally:state._context_arena.reset(token)
    monkeypatch.setattr(api,'schema_validator',lambda *a:pytest.fail('unknown factory called'))
    with pytest.raises(ContractError,match='validation profile'):bridge()(**args)
    assert roots==[]


@pytest.mark.parametrize('path',['anchors','parent_checkpoint'])
@pytest.mark.parametrize('kind',['healthy','error','cancel','close_error'])
def test_parent_primary_and_cancellation_on_original_owner_paths(tmp_path,monkeypatch,path,kind):
    from consumer_context_fixture import completed_fixture
    selected,*_=completed_fixture(tmp_path)
    reader=StateReader(tmp_path,state_ref=selected,expected_identity=selected['identity'])
    # Prime anchors so the second case reaches ORIGINAL _load_checkpoint's own
    # nested parent call. Never replace the method under test.
    if path=='parent_checkpoint':reader._parent_anchors()
    original_json=StateReader._json;original_close=StateReader.close;closed=[]
    primary=KeyboardInterrupt('parent cancel') if kind=='cancel' else ContractError('parent failure','corrupt_result')
    secondary=RuntimeError('parent close')
    def reading(owner,ref):
        if owner._lineage_depth>0 and ref['path']=='checkpoint.json':
            owner._fault_target=True
            if kind in ('error','cancel'):raise primary
        return original_json(owner,ref)
    def closing(owner):
        original_close(owner);closed.append(owner)
        if getattr(owner,'_fault_target',False) and kind!='healthy':raise secondary
    monkeypatch.setattr(StateReader,'_json',reading);monkeypatch.setattr(StateReader,'close',closing)
    try:
        operation=reader._parent_anchors if path=='anchors' else reader._load_checkpoint
        if kind=='healthy':operation()
        else:
            with pytest.raises(BaseException) as caught:operation()
            assert caught.value is (primary if kind in ('error','cancel') else secondary)
            if kind in ('error','cancel'):assert any('cleanup also raised RuntimeError' in n for n in primary.__notes__)
    finally:original_close(reader)
    assert closed and all(r._closed and not r._handles for r in closed)


def test_actual_full_shared_cache_and_retained_first_output(tmp_path,monkeypatch):
    import sys,weakref
    import qcl_negf_contracts as contracts
    import qcl_negf_contracts.results as contract_results
    from qcl_negf_results import state
    args,manifest,_,_=context_fixture(tmp_path,final=True)
    kinds={v[0] for v in contract_results._SCHEMAS.values()}|{
        'core-state.schema.json','result-common.schema.json','scientific-plan.schema.json','nested-sweep-plan.schema.json',
        'producer-phase-segment.schema.json','producer-phase-source.schema.json','telemetry-envelope.schema.json'}
    cached=[contract_results._validator(k) for k in kinds]
    assert len(cached)==contract_results._validator.cache_info().currsize==15
    refs=[weakref.ref(r) for v in cached for r in (v._registry,v._resolver._registry)]
    maxima=[];calls=[];arenas=[];original_arena=api._MetadataArena
    class MeasuredArena(original_arena):
        def __init__(self,*a,**k):super().__init__(*a,**k);arenas.append(self)
    monkeypatch.setattr(api,'_MetadataArena',MeasuredArena)
    def forest():
        seen=set()
        def weight(value):
            if value is None or type(value) in (bool,int,float):return 64
            if type(value) in (str,bytes):return 128+6*len(value)
            if id(value) in seen:return 0
            seen.add(id(value))
            if type(value) is dict:return 128+sum(weight(k)+weight(v) for k,v in value.items())
            assert type(value) in (list,tuple)
            return 128+sum(64+weight(v) for v in value)
        return sum(weight(r.contents) for ref in refs if (registry:=ref()) is not None for _,r in registry.items())
    def observe(frame,event,value):
        if event=='return' and frame.f_code is contracts.schema_validator.__code__ and value is not None:
            refs.extend((weakref.ref(value._registry),weakref.ref(value._resolver._registry)))
            calls.append(frame.f_locals['name']);maxima.append(forest())
    old=sys.getprofile();sys.setprofile(observe)
    try:
        first=bridge()(**args);retained=state._context_weight(first)
        second=bridge()(**args)
    finally:sys.setprofile(old)
    assert first[0]==second[0]==manifest and first[4] is not None and second[4] is not None
    assert state._context_weight(first)==retained<=4*M
    assert max(maxima)==94_695_342 and max(maxima)+2_676_240+6*M<112*M  # includes real shared standard metaschemas
    assert calls and arenas[0].validation_reservation==112*M
    assert 112*M<=second[2].io_counters['metadata_high_water_bytes']<=128*M
    assert arenas[0].profile_bytes_read>5*M and arenas[1].profile_bytes_read==arenas[0].profile_bytes_read
    assert 0<arenas[1].profile_io_remaining<1*M
    assert set(first[2].io_counters)=={'bytes_read_total','sha_bytes','hdf_bytes','metadata_bytes',
        'logical_selected_bytes','maximum_io_bytes','maximum_context_bytes','metadata_high_water_bytes'}
    print('FULL_SHARED_CACHE15',dict(max_schema_forest=max(maxima),retained=retained,calls=calls,counters=second[2].io_counters))


def test_unknown_loaded_library_and_after_close_cache_mutation_hold(tmp_path,monkeypatch):
    from jsonschema import Draft202012Validator
    import qcl_negf_contracts.results as contracts
    args,_,_,_=context_fixture(tmp_path);roots=[]
    with monkeypatch.context() as patch:
        patch.setattr(StateReader,'_root_identity',lambda self:roots.append(True))
        patch.setattr(Draft202012Validator,'validate',lambda *a:None)
        with pytest.raises(ContractError,match='validation profile'):bridge()(**args)
        assert not roots
    cached=contracts._validator('result-common.schema.json');original=StateReader.close
    def close(reader):original(reader);cached.schema['_changed_after_read']=True
    monkeypatch.setattr(StateReader,'close',close)
    try:
        with pytest.raises(ContractError,match='validation profile'):bridge()(**args)
    finally:cached.schema.pop('_changed_after_read',None)


@pytest.mark.parametrize('kind',['unknown_name','bounded_layout'])
def test_unknown_cache_qualification_holds_without_clearing_original(tmp_path,monkeypatch,kind):
    import functools
    import qcl_negf_contracts.results as contracts
    args,_,_,_=context_fixture(tmp_path);original=contracts._validator;info=original.cache_info()
    isolated=functools.lru_cache(maxsize=1 if kind=='bounded_layout' else None)(original.__wrapped__)
    if kind=='unknown_name':isolated('run.schema.json')
    roots=[];monkeypatch.setattr(StateReader,'_root_identity',lambda self:roots.append(True))
    monkeypatch.setattr(contracts,'_validator',isolated)
    with pytest.raises(ContractError,match='validation profile'):bridge()(**args)
    assert not roots and original.cache_info()==info


def test_profile_source_never_uses_unbounded_read_bytes(tmp_path,monkeypatch):
    args,_,_,_=context_fixture(tmp_path)
    original=Path.read_bytes
    def bounded(path):
        if 'qcl_negf_contracts' in path.parts and path.name in ('__init__.py','results.py'):
            pytest.fail('unbounded source reread before compile')
        return original(path)
    monkeypatch.setattr(Path,'read_bytes',bounded)
    assert bridge()(**args)[2].plan_raw==original(tmp_path/'plan.json')


@pytest.mark.parametrize('fault',['entry_growth','exit_growth','read_error','cancel_close'])
def test_profile_source_eof_request_charge_and_cleanup(tmp_path,monkeypatch,fault):
    from qcl_negf_results import state
    args,_,_,_=context_fixture(tmp_path);arenas=[];streams=[];requests=[]
    original_arena=api._MetadataArena;original_open=Path.open;opened=0
    primary=KeyboardInterrupt('profile read cancel')
    class MeasuredArena(original_arena):
        def __init__(self,*a,**k):super().__init__(*a,**k);arenas.append(self)
    class FaultStream:
        def __init__(self,stream):self.stream=stream
        def read(self,count):
            requests.append((count,arenas[-1].profile_io_remaining))
            if fault=='read_error':raise OSError('profile read failed')
            if fault=='cancel_close':raise primary
            raw=self.stream.read(count)
            # Simulate one byte appended after the admitted stat/read boundary.
            # Protected package source is never changed by the test.
            if count==1 and not raw:return b'x'
            return raw
        def close(self):
            self.stream.close()
            if fault=='cancel_close':raise RuntimeError('profile secondary close')
    def opening(path,*a,**k):
        nonlocal opened
        stream=original_open(path,*a,**k)
        if 'qcl_negf_contracts' in path.parts and path.name=='__init__.py':
            opened+=1
            if fault!='exit_growth' or opened>=3:
                stream=FaultStream(stream);streams.append(stream)
        return stream
    monkeypatch.setattr(api,'_MetadataArena',MeasuredArena)
    with monkeypatch.context() as patch:
        patch.setattr(Path,'open',opening)
        if fault=='cancel_close':
            with pytest.raises(KeyboardInterrupt) as caught:bridge()(**args)
            assert caught.value is primary and any('profile cleanup also raised RuntimeError' in note for note in primary.__notes__)
        else:
            with pytest.raises(ContractError,match='validation profile'):bridge()(**args)
    assert streams and all(s.stream.closed for s in streams)
    assert requests and all(0<count<=65536 and remaining<=6*M-count for count,remaining in requests)
    assert arenas[-1].profile_io_remaining==requests[-1][1]  # failed/EOF request never refunded
    assert all(r._closed and not r._handles for r in arenas[-1].readers)
    if fault=='exit_growth':assert arenas[-1].readers
    else:assert not arenas[-1].readers
    assert state._context_arena.get() is None
    assert bridge()(**args)[2].plan_raw==(tmp_path/'plan.json').read_bytes()


def test_profile_source_credit_refuses_before_read(tmp_path,monkeypatch):
    args,_,_,_=context_fixture(tmp_path);original=api._MetadataArena;arenas=[]
    class StrictArena(original):
        def __init__(self,*a,**k):super().__init__(*a,**k);self.profile_io_remaining=1;arenas.append(self)
    monkeypatch.setattr(api,'_MetadataArena',StrictArena)
    with pytest.raises(ContractError,match='validation profile') as caught:bridge()(**args)
    assert arenas[0].profile_io_remaining==1 and arenas[0].profile_bytes_read==0
    assert caught.value.code=='budget_exceeded'


@pytest.mark.parametrize('name',['_validator','_layout','_row_fields'])
def test_actual_c_cache_target_cannot_be_hidden_by_wrapped(tmp_path,monkeypatch,name):
    import functools
    import qcl_negf_contracts.results as contracts
    args,_,_,_=context_fixture(tmp_path);original=getattr(contracts,name);calls=[];roots=[]
    def foreign(*a,**k):calls.append(True);return original(*a,**k)
    wrapper=functools.cache(foreign)
    wrapper.__wrapped__=original.__wrapped__
    with monkeypatch.context() as patch:
        patch.setattr(contracts,name,wrapper)
        patch.setattr(StateReader,'_root_identity',lambda self:roots.append(True))
        with pytest.raises(ContractError,match='validation profile'):bridge()(**args)
    assert not calls and not roots


@pytest.mark.parametrize('binding',['contracts_files','results_files','both_stdlib_files',
    'files_closure','contracts_json','results_json','json_loads','json_loads_defaults','package'])
def test_direct_factory_bindings_hold_before_callback_or_root(tmp_path,monkeypatch,binding):
    import importlib.resources as resources
    import importlib.resources._common as common
    import types
    import qcl_negf_contracts as contracts
    import qcl_negf_contracts.results as results
    args,_,_,_=context_fixture(tmp_path);calls=[];roots=[]
    def foreign(*a,**k):calls.append(True);raise AssertionError('unqualified callback reached')
    with monkeypatch.context() as patch:
        if binding=='contracts_files':patch.setattr(contracts,'files',foreign)
        elif binding=='results_files':patch.setattr(results,'files',foreign)
        elif binding=='both_stdlib_files':
            for module in (contracts,results,resources,common):patch.setattr(module,'files',foreign)
        elif binding=='files_closure':
            original=resources.files
            altered=types.FunctionType(original.__code__,original.__globals__,closure=(
                types.CellType(foreign),original.__closure__[1]),argdefs=original.__defaults__)
            altered.__wrapped__=original.__wrapped__
            for module in (contracts,results,resources,common):patch.setattr(module,'files',altered)
        elif binding in ('contracts_json','results_json'):
            patch.setattr(contracts if binding=='contracts_json' else results,'json',types.SimpleNamespace(loads=foreign))
        elif binding=='json_loads':patch.setattr(json,'loads',foreign)
        elif binding=='json_loads_defaults':
            patch.setattr(json.loads,'__kwdefaults__',{**json.loads.__kwdefaults__,'object_hook':foreign})
        else:patch.setattr(contracts,'__package__','foreign_package')
        patch.setattr(StateReader,'_root_identity',lambda self:roots.append(True))
        with pytest.raises(ContractError,match='validation profile'):bridge()(**args)
    assert not calls and not roots


@pytest.mark.parametrize('binding',['c_target','files','json_loads'])
def test_after_close_factory_binding_mutation_holds_without_transfer(tmp_path,monkeypatch,binding):
    import functools
    import qcl_negf_contracts as contracts
    import qcl_negf_contracts.results as results
    args,_,_,_=context_fixture(tmp_path);calls=[];closed=[];close=StateReader.close
    original=results._validator
    def foreign(*a,**k):calls.append(True);raise AssertionError('unqualified exit callback reached')
    wrapper=functools.cache(foreign);wrapper.__wrapped__=original.__wrapped__
    def closing(reader):
        close(reader);closed.append(reader)
        if binding=='c_target':monkeypatch.setattr(results,'_validator',wrapper)
        elif binding=='files':monkeypatch.setattr(contracts,'files',foreign)
        else:monkeypatch.setattr(json,'loads',foreign)
    monkeypatch.setattr(StateReader,'close',closing)
    with pytest.raises(ContractError,match='validation profile'):bridge()(**args)
    assert closed and all(r._closed and not r._handles for r in closed) and not calls


def test_public_science_then_phase_and_performance_reads_preserve_closed_context(tmp_path):
    """Ordinary technical readers must not poison the shared science cache."""
    from qcl_negf_results.capture import PhaseCaptureSource
    from qcl_negf_results.final_diagnostics import FinalDiagnosticsReader
    from qcl_negf_results.performance_views import read_performance_context,read_performance_summary
    from qcl_negf_results.telemetry import TelemetryWriter
    from qcl_negf_results.state import _IOBudget
    from native_consumer_fixture import technical_source,sealed_phase,telemetry_input
    args,manifest,checkpoint,value=context_fixture(tmp_path/'closed',final=True)
    first=bridge()(**args);saved=deepcopy(first)
    with StateReader(args['artifact_root'],state_ref=args['state_ref'],expected_identity=args['expected_identity'],limits=args['limits']) as reader:
        inputs=api.describe_checkpoint_inputs(reader,plan_ref=args['plan_ref'],model_ref=args['model_ref'])
        with FinalDiagnosticsReader(reader,artifact_ref=args['diagnostics_ref']) as final:diagnostic=final.describe()
        with api.FinalObservationsReader(reader,artifact_ref=args['observations_ref']) as final:observation=final.describe()
    c=technical_source(tmp_path/'technical');producer=tmp_path/'producer'
    sealed_phase(producer,c['state']['identity'])
    descriptor=dict(status='available',session_id='native',current_path='phase-current.json',reason=None)
    with PhaseCaptureSource(producer,descriptor=descriptor,expected_identity=c['state']['identity'],route_epoch=0) as phase:
        batch=phase.harvest(None,budget=_IOBudget(M))
        assert batch is not None
        writer=TelemetryWriter(c['technical'],{},{},transport=c['transport'])
        writer.stage_phase_batch(batch);writer.close()
    selection=telemetry_input(c,writer)
    with StateReader(c['science'],state_ref=c['state'],expected_identity=c['state']['identity']) as reader:
        assertions,rows=read_performance_context(reader,telemetry=selection)
        assert assertions and rows
        assert read_performance_summary(reader,telemetry=selection)['status']=='available'
    second=bridge()(**args)
    assert first==saved and first==second
    assert second[0]==manifest and second[1]==checkpoint
    assert second[2].plan_raw==inputs.plan_raw and second[2].model_raw==inputs.model_raw
    assert second[3].declaration==diagnostic.declaration and second[4].declaration==observation.declaration==value
