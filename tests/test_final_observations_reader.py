"""Raw stored observations: independent literal cells and source/budget fences."""
import importlib
import importlib.util
import json
from copy import deepcopy

import h5py
import numpy as np
import pytest
from qcl_negf_contracts.results import validate_final_observations
from qcl_negf_contracts.messages import ContractError
from qcl_negf_results.state import StateReader, ResultReadLimits
from consumer_context_fixture import fixture, refresh, change, fixed_json, PLAN


def api():
    assert importlib.util.find_spec('qcl_negf_results.final_observations') is not None, 'raw stored final observations reader missing'
    return importlib.import_module('qcl_negf_results.final_observations')


def test_closed_literal_source_metadata_without_numeric_read_or_science_promotion(tmp_path,monkeypatch):
    selected,artifact,value,checkpoint,_=fixture(tmp_path)
    import yaml
    manifest=json.loads((tmp_path/selected['manifest']['path']).read_bytes())
    model=json.loads((tmp_path/'model.json').read_bytes())
    with h5py.File(tmp_path/'analysis.h5','r') as handle:
        diagnostic=json.loads(bytes(handle['final_diagnostics/declaration_json'][()]))
    assert validate_final_observations(value,checkpoint=checkpoint,manifest=manifest,artifact_ref=artifact,
        diagnostics=diagnostic,plan=yaml.safe_load(PLAN),model=model)==value
    original=h5py.Dataset.__getitem__
    def no_numeric(dataset,key):
        assert '/arrays/' not in dataset.name, 'describe read numeric dataset'
        return original(dataset,key)
    with StateReader(tmp_path,state_ref=selected,expected_identity=selected['identity']) as reader:
        monkeypatch.setattr(h5py.Dataset,'__getitem__',no_numeric)
        view=api().FinalObservationsReader(reader,artifact_ref=artifact)
        description=view.describe()
        assert description.parent_manifest is description.parent_checkpoint is None
        assert description.declaration==value and description.source==selected
        assert description.plan['name']=='configured_operating_points'
        assert description.model==model
        assert description.declaration['summary']['scientific_accepted'] is None
        assert reader.source.status['physics']['state']=='recorded'
        assert {v['path'] for v in reader.source.integrity['selected_payloads_verified']}=={'analysis.h5','observations.h5'}
        description.declaration['summary']['status']='mutated'
        assert view.describe().declaration==value
        view.close();view.close()
        assert reader.source.state==selected


def test_all26_logical_axes_ieee_integer_bool_and_full_complex_cells(tmp_path):
    selected,artifact,_,_,values=fixture(tmp_path)
    with StateReader(tmp_path,state_ref=selected,expected_identity=selected['identity']) as reader:
        with api().FinalObservationsReader(reader,artifact_ref=artifact) as view:
            for name,array in values.items():
                d=view.describe_array(name)
                assert d.logical_shape==d.stored_shape==array.shape
                block=view.read_array(name,(slice(None),)*array.ndim)
                assert block.values.tobytes()==array.tobytes() and not block.values.flags.writeable
            assert view.read_array('localized_wavefunctions_per_sqrt_nm',(3,1)).values==7+8j
            assert view.read_array('state_index',(1,)).values==1
            assert bool(view.read_array('trusted_energy',(2,)).values)
            assert view.read_array('spectral_trace_per_eV',(slice(0,0),slice(None))).values.size==0
            assert view.read_array('spectral_trace_per_eV',(slice(0,3,2),1)).values.tolist()==[1.,5.]
            with pytest.raises(ContractError,match='buffer|budget|limit'):
                view.read_array('localized_wavefunctions_per_sqrt_nm',(slice(None),)*2,maximum_bytes=16)


@pytest.mark.parametrize('field,value',[('sha256','f'*64),('bytes','12'),('schema','qcl-negf-physics-v5'),('media_type','text/plain')])
def test_exact_selected_ref_cannot_be_substituted(tmp_path,field,value):
    selected,artifact,_,_,_=fixture(tmp_path);artifact[field]=value
    with StateReader(tmp_path,state_ref=selected,expected_identity=selected['identity']) as reader:
        with pytest.raises(ContractError):api().FinalObservationsReader(reader,artifact_ref=artifact).describe()


@pytest.mark.parametrize('name',['spatial_energy_density_per_eV_m3','localized_levels_eV','effective_levels_eV'])
def test_optional_nulls_have_exact_issue_and_do_not_synthesize_arrays(tmp_path,name):
    effective=('effective_levels_eV','effective_wavefunctions_per_sqrt_nm','effective_transform','effective_density_matrix_per_m2')
    nulls=effective if name.startswith('effective') else (name,)
    selected,artifact,value,_,_=fixture(tmp_path,null_outputs=nulls)
    with StateReader(tmp_path,state_ref=selected,expected_identity=selected['identity']) as reader:
        view=api().FinalObservationsReader(reader,artifact_ref=artifact)
        assert view.describe().declaration==value
        with pytest.raises(ContractError,match='projection_failed: literal unavailable'):view.describe_array(name)
        with pytest.raises(ValueError,match='unknown'):view.describe_array('not_an_array')


@pytest.mark.parametrize('kind',['vlen','ascii','rank1','null','invalid_utf8','duplicate','bare_nan','extra_root','extra_attr','extra_array','hard_alias'])
def test_closed_fixed_utf8_metadata_and_tree_fences(tmp_path,monkeypatch,kind):
    selected,artifact,value,_,_=fixture(tmp_path)
    with h5py.File(tmp_path/artifact['path'],'r+') as handle:
        name='final_observations/declaration_json'
        if kind.startswith('extra'):
            if kind=='extra_root':handle.create_group('unowned')
            elif kind=='extra_attr':handle['final_observations'].attrs['unowned']=1
            else:handle['final_observations/arrays/unowned']=np.zeros(1)
        elif kind=='hard_alias':
            name='final_observations/arrays/potential_external_eV';del handle[name]
            handle[name]=handle['final_observations/arrays/potential_structure_eV']
        else:
            del handle[name]
            if kind=='vlen':handle.create_dataset(name,data=json.dumps(value),dtype=h5py.string_dtype())
            elif kind=='ascii':fixed_json(handle,name,value,cset='ascii')
            elif kind=='rank1':handle[name]=np.array([b'{}'])
            elif kind=='null':handle.create_dataset(name,dtype='S4')
            else:fixed_json(handle,name,{'invalid_utf8':b'\xff','duplicate':b'{"a":1,"a":2}','bare_nan':b'{"a":NaN}'}[kind])
    refresh(tmp_path,selected,artifact)
    original=h5py.Dataset.__getitem__
    def guarded(dataset,key):
        if kind in ('vlen','ascii','rank1','null'):assert dataset.name!='/final_observations/declaration_json'
        assert '/arrays/' not in dataset.name
        return original(dataset,key)
    monkeypatch.setattr(h5py.Dataset,'__getitem__',guarded)
    with StateReader(tmp_path,state_ref=selected,expected_identity=selected['identity']) as reader:
        with pytest.raises(ContractError):api().FinalObservationsReader(reader,artifact_ref=artifact).describe()


@pytest.mark.parametrize('kind',['shape','endian','float32','custom_float','compound','int_unsigned','int_endian','bool_uint8','bool_signed','bool_enum','filter','fletcher32','scaleoffset','external','soft','virtual','raw'])
def test_local_array_dtypes_shapes_links_and_filters_are_closed(tmp_path,kind):
    selected,artifact,_,_,_=fixture(tmp_path)
    name='state_index' if kind.startswith('int_') else 'trusted_energy' if kind.startswith('bool_') else 'spectral_trace_per_eV'
    path='/final_observations/arrays/'+name;shape=(2,) if name=='state_index' else (3,) if name=='trusted_energy' else (3,2)
    with h5py.File(tmp_path/artifact['path'],'r+') as handle:
        del handle[path]
        if kind=='soft':handle[path]=h5py.SoftLink('/final_observations/arrays/occupied_trace_per_eV')
        elif kind=='external':handle[path]=h5py.ExternalLink('foreign.h5','/data')
        elif kind=='virtual':handle.create_virtual_dataset(path,h5py.VirtualLayout(shape=shape,dtype='<f8'))
        elif kind=='raw':handle.create_dataset(path,shape=shape,dtype='<f8',external=[('foreign.raw',0,48)])
        elif kind in ('custom_float','filter'):
            dtype=h5py.h5t.IEEE_F64LE.copy();plist=h5py.h5p.create(h5py.h5p.DATASET_CREATE)
            if kind=='custom_float':dtype.set_ebias(1022)
            else:plist.set_chunk(shape);plist.set_filter(32015,h5py.h5z.FLAG_OPTIONAL)
            d=h5py.h5d.create(handle.id,path.encode(),dtype,h5py.h5s.create_simple(shape),dcpl=plist)
            d.close();plist.close();dtype.close()
        else:
            dtype={'shape':'<f8','endian':'>f8','float32':'<f4','compound':[('r','<f8'),('i','<f8')],
                'int_unsigned':'<u8','int_endian':'>i8','bool_uint8':'u1',
                'bool_signed':h5py.enum_dtype({'FALSE':0,'TRUE':1},basetype='i1'),
                'bool_enum':h5py.enum_dtype({'FALSE':0,'TRUE':2},basetype='u1')}.get(kind,'<f8')
            kwargs={'fletcher32':True} if kind=='fletcher32' else {'scaleoffset':1} if kind=='scaleoffset' else {}
            handle.create_dataset(path,shape=(2,3) if kind=='shape' else shape,dtype=dtype,**kwargs)
    refresh(tmp_path,selected,artifact)
    with StateReader(tmp_path,state_ref=selected,expected_identity=selected['identity']) as reader:
        with pytest.raises(ContractError):api().FinalObservationsReader(reader,artifact_ref=artifact).describe()


@pytest.mark.parametrize('kind',['axis','units','coordinate','weight','checkpoint','identity','current_status','dependencies'])
def test_all_metadata_contexts_and_current_evidence_are_bound(tmp_path,kind):
    selected,artifact,_,_,_=fixture(tmp_path)
    from result_v2_fixture import put
    if kind in ('dependencies','current_status'):
        manifest=json.loads((tmp_path/selected['manifest']['path']).read_bytes())
        if kind=='dependencies':next(r for r in manifest['artifacts'] if r['path']==artifact['path'])['dependencies'].reverse()
        else:manifest['status']['physics']['evidence']['sha256']='f'*64
        selected['manifest']=put(tmp_path,'selected.json',manifest,manifest['schema'])
    else:
        def altered(v):
            d=v['context']['energy_scaled']
            if kind=='axis':d['axes']=['k']
            elif kind=='units':d['units']='eV'
            elif kind=='coordinate':d['coordinate_fields']=['z_nm']
            elif kind=='weight':d['weight_fields']=['wk']
            elif kind=='checkpoint':v['checkpoint']['sha256']='f'*64
            else:v['identity']['attempt']='2'
        change(tmp_path,selected,artifact,altered)
    with StateReader(tmp_path,state_ref=selected,expected_identity=selected['identity']) as reader:
        with pytest.raises(ContractError):api().FinalObservationsReader(reader,artifact_ref=artifact).describe()


@pytest.mark.parametrize('filename',['plan.yaml','model.json','checkpoint.json','analysis.h5'])
def test_missing_selected_dependencies_are_contract_failures(tmp_path,filename):
    selected,artifact,_,_,_=fixture(tmp_path);(tmp_path/filename).unlink()
    with StateReader(tmp_path,state_ref=selected,expected_identity=selected['identity']) as reader:
        with pytest.raises(ContractError):api().FinalObservationsReader(reader,artifact_ref=artifact).describe()


def test_model_digest_is_checked_from_actual_canonical_preimage(tmp_path):
    selected,artifact,_,_,_=fixture(tmp_path,invalid_model=True)
    with StateReader(tmp_path,state_ref=selected,expected_identity=selected['identity']) as reader:
        with pytest.raises(ContractError,match='resolved configuration identity differs'):
            api().FinalObservationsReader(reader,artifact_ref=artifact).describe()


@pytest.mark.parametrize('raw,reason',[(PLAN+b'---\n{}\n','one document'),
    (PLAN.replace(b'200.0',b'&x 200.0'),'anchors'),(PLAN.replace(b'200.0',b'*x'),'aliases'),
    (PLAN.replace(b'200.0',b'!unowned 200.0'),'tags'),(PLAN+b'name: duplicate\n','duplicate'),
    (PLAN.replace(b'200.0',b'.nan'),'nonfinite'),(PLAN.replace(b'[200.0]',b'[200.0, 200.0]'),'canonical duplicate'),
    (PLAN.replace(b'  kind: level',b'  <<: {kind: level}\n  kind: level',1),'merge')])
def test_original_yaml_admission_before_tree_and_contract_join(tmp_path,raw,reason):
    selected,artifact,_,_,_=fixture(tmp_path,plan_bytes=raw)
    with StateReader(tmp_path,state_ref=selected,expected_identity=selected['identity']) as reader:
        with pytest.raises(ContractError,match=reason):api().FinalObservationsReader(reader,artifact_ref=artifact).describe()


@pytest.mark.parametrize('bits',['0000000000000000','8000000000000000','7ff0000000000000','fff0000000000000','7ff8000000001234','7ff0000000000001'])
def test_raw_observation_ieee_bits_are_not_sanitized(tmp_path,bits):
    selected,artifact,_,_,_=fixture(tmp_path)
    with h5py.File(tmp_path/artifact['path'],'r+') as handle:
        values=np.array([int(bits,16)]*4,dtype='<u8').view('<f8');handle['final_observations/arrays/potential_structure_eV'][:]=values
    refresh(tmp_path,selected,artifact)
    with StateReader(tmp_path,state_ref=selected,expected_identity=selected['identity']) as reader:
        view=api().FinalObservationsReader(reader,artifact_ref=artifact)
        assert view.read_array('potential_structure_eV',(slice(None),)).values.view('<u8').tolist()==[int(bits,16)]*4


@pytest.mark.parametrize('moment',['before','after'])
@pytest.mark.parametrize('change_kind',['rewrite','replace','unlink'])
def test_actual_selected_owner_stable_through_block_publication(tmp_path,monkeypatch,moment,change_kind):
    selected,artifact,_,_,_=fixture(tmp_path);path=tmp_path/artifact['path']
    with StateReader(tmp_path,state_ref=selected,expected_identity=selected['identity']) as reader:
        view=api().FinalObservationsReader(reader,artifact_ref=artifact);view.describe()
        original=h5py.Dataset.__getitem__
        def alter():
            if change_kind=='unlink':path.unlink()
            elif change_kind=='rewrite':
                with open(path,'r+b') as raw:raw.write(b'X')
            else:
                replacement=path.with_suffix('.new');replacement.write_bytes(path.read_bytes());replacement.replace(path)
        def intercepted(dataset,key):
            if dataset.name=='/final_observations/arrays/spectral_trace_per_eV':
                if moment=='before':alter()
                value=original(dataset,key)
                if moment=='after':alter()
                return value
            return original(dataset,key)
        monkeypatch.setattr(h5py.Dataset,'__getitem__',intercepted)
        with pytest.raises(ContractError,match='changed') as error:view.read_array('spectral_trace_per_eV',(0,0))
        assert error.value.code=='source_changed'


def test_chunk_limit_precedes_scalar_selection_and_warm_hash_is_reused(tmp_path,monkeypatch):
    selected,artifact,_,_,_=fixture(tmp_path,chunks=True)
    with StateReader(tmp_path,state_ref=selected,expected_identity=selected['identity'],limits=ResultReadLimits(maximum_array_bytes=32)) as reader:
        view=api().FinalObservationsReader(reader,artifact_ref=artifact);d=view.describe()
        assert d.io_counters['sha_bytes']>=int(artifact['bytes'])
        original=h5py.Dataset.__getitem__
        def guarded(dataset,key):
            assert '/arrays/' not in dataset.name,'numeric payload before chunk guard'
            return original(dataset,key)
        monkeypatch.setattr(h5py.Dataset,'__getitem__',guarded)
        with pytest.raises(ContractError,match='chunk'):view.read_array('localized_wavefunctions_per_sqrt_nm',(0,0),maximum_bytes=16)
        monkeypatch.setattr(h5py.Dataset,'__getitem__',original)
    with StateReader(tmp_path,state_ref=selected,expected_identity=selected['identity']) as reader:
        view=api().FinalObservationsReader(reader,artifact_ref=artifact);view.describe()
        assert view.read_array('spectral_trace_per_eV',(0,0)).io_counters['sha_bytes']==0


def test_original_scientific_plan_json_and_canonical_model_are_distinct(tmp_path):
    selected,artifact,value,_,_=fixture(tmp_path,plan_kind='scientific_plan_v2')
    with StateReader(tmp_path,state_ref=selected,expected_identity=selected['identity']) as reader:
        d=api().describe_final_observations(reader,artifact_ref=artifact)
        assert d.plan['points'][0]['temperature_K']==200.
        assert d.plan['scientific_fingerprint']!=''+selected['identity']['plan_sha256']
        assert value['request']['model']['sha256']!=d.model['configuration_hash']
        assert d.model['configuration_hash']==selected['identity']['model_sha256']


def test_completed_restore_preserves_failed_original_and_current_receipts(tmp_path):
    from consumer_context_fixture import completed_fixture
    selected,artifact,value,checkpoint,_,parent=completed_fixture(tmp_path)
    with StateReader(tmp_path,state_ref=selected,expected_identity=selected['identity']) as reader:
        d=api().FinalObservationsReader(reader,artifact_ref=artifact).describe()
        assert d.declaration==value and d.source==selected
        assert d.original_evidence==checkpoint['state']['fields']['final_evidence']
        assert d.original_evidence['fields']['original_status']['value']=='invalid'
        assert d.declaration['summary']['status']=='validation_failed'
        assert d.declaration['restoration']['parent_state']==parent
        assert d.declaration['restoration']['independent_oracle']['fields']['status']['value']=='not_measured'
        parent_manifest=json.loads((tmp_path/parent['manifest']['path']).read_bytes())
        parent_checkpoint=json.loads((tmp_path/parent_manifest['checkpoint']['path']).read_bytes())
        assert d.parent_manifest==parent_manifest and d.parent_checkpoint==parent_checkpoint
        assert d.parent_manifest['invocation']['source_epoch']=='epoch'
        assert d.invocation['source_epoch']=='restored-epoch'
        assert d.parent_manifest['identity']['attempt']=='1' and d.source['identity']['attempt']=='2'


def test_parent_fields_preserve_old_constructor_and_positional_pattern_contract():
    from dataclasses import asdict,fields
    descriptor=api().FinalObservationsDescription
    names=('source','invocation','artifact','checkpoint','diagnostics','declaration','plan','model',
        'original_evidence','integrity','io_counters')
    old=descriptor(*({} for _ in names))
    assert descriptor.__match_args__==names
    assert old.parent_manifest is old.parent_checkpoint is None
    added={f.name:f for f in fields(descriptor) if f.name.startswith('parent_')}
    assert set(added)=={'parent_manifest','parent_checkpoint'}
    assert all(f.kw_only and f.default is None for f in added.values())
    assert asdict(old)['parent_manifest'] is asdict(old)['parent_checkpoint'] is None
    with pytest.raises(TypeError):descriptor(*({} for _ in range(13)))


def test_completed_parent_descriptions_are_detached_and_array_paths_do_not_copy_them(tmp_path,monkeypatch):
    from consumer_context_fixture import completed_fixture
    selected,artifact,_,_,_,parent=completed_fixture(tmp_path)
    original_close=StateReader.close;closed=[]
    def closing(owner):
        original_close(owner)
        if owner._lineage_depth:closed.append(owner)
    monkeypatch.setattr(StateReader,'close',closing)
    with StateReader(tmp_path,state_ref=selected,expected_identity=selected['identity']) as reader:
        view=api().FinalObservationsReader(reader,artifact_ref=artifact)
        description=view.describe()
        expected_manifest=deepcopy(description.parent_manifest);expected_checkpoint=deepcopy(description.parent_checkpoint)
        assert closed and all(owner._closed and not owner._handles for owner in closed)
        assert all(description.parent_manifest is not owner._manifest and
            description.parent_checkpoint is not owner._checkpoint for owner in closed)
        description.parent_manifest['identity']['point_id']='mutated'
        description.parent_checkpoint['identity']['point_id']='mutated'
        again=view.describe()
        assert again.parent_manifest==expected_manifest and again.parent_checkpoint==expected_checkpoint
        assert reader.source.state==selected and parent==again.declaration['restoration']['parent_state']
        assert all(owner._manifest['identity']['point_id']=='point' for owner in closed)
        original_copy=api()._context_copy;copies=[]
        def copying(value):
            if value==expected_manifest or value==expected_checkpoint:copies.append(value)
            return original_copy(value)
        monkeypatch.setattr(api(),'_context_copy',copying)
        assert view.describe_array('state_index').logical_shape==(2,)
        assert view.read_array('state_index',(1,)).values==1
        assert copies==[]


@pytest.mark.parametrize('mutation',['missing','identity','sequence','checkpoint_ref','history','integrity','evidence','epoch'])
def test_public_completed_parent_operands_reject_foreign_context(tmp_path,mutation):
    from consumer_context_fixture import completed_fixture
    selected,artifact,_,checkpoint,_,_=completed_fixture(tmp_path)
    with StateReader(tmp_path,state_ref=selected,expected_identity=selected['identity']) as reader:
        d=api().describe_final_observations(reader,artifact_ref=artifact)
        manifest=json.loads((tmp_path/selected['manifest']['path']).read_bytes())
        with h5py.File(tmp_path/d.diagnostics['path'],'r') as handle:
            diagnostic=json.loads(bytes(handle['final_diagnostics/declaration_json'][()]))
        if mutation=='missing':d=__import__('dataclasses').replace(d,parent_checkpoint=None)
        elif mutation=='identity':d.parent_manifest['identity']['attempt']='99'
        elif mutation=='sequence':d.parent_manifest['state_sequence']='99'
        elif mutation=='checkpoint_ref':d.parent_manifest['checkpoint']['sha256']='f'*64
        elif mutation=='history':d.parent_checkpoint['history']['last_record']='1'
        elif mutation=='integrity':d.parent_checkpoint['state']['fields']['integrity'][0]['decimal']='0'
        elif mutation=='evidence':d.parent_checkpoint['state']['fields']['final_evidence']['fields']['original_status']['value']='converged'
        else:d.parent_checkpoint['invocation']['source_epoch']='foreign'
        with pytest.raises(ContractError):
            validate_final_observations(d.declaration,checkpoint=checkpoint,manifest=manifest,artifact_ref=artifact,
                diagnostics=diagnostic,plan=d.plan,model=d.model,parent_manifest=d.parent_manifest,parent_checkpoint=d.parent_checkpoint)


def test_standalone_parent_copy_forecast_refuses_before_deepcopy(tmp_path,monkeypatch):
    from dataclasses import replace
    from consumer_context_fixture import completed_fixture
    from qcl_negf_results import state
    selected,artifact,_,_,_,_=completed_fixture(tmp_path)
    forecasts=[];original_before=api()._context_before;original_copy=api()._context_copy
    with StateReader(tmp_path,state_ref=selected,expected_identity=selected['identity']) as reader:
        view=api().FinalObservationsReader(reader,artifact_ref=artifact)
        def before(operation,**kwargs):
            if operation=='observations description':
                # Independently total the existing per-operation metadata model.
                import sys
                def weight(value):
                    pending=[value];seen=set();total=0
                    while pending:
                        item=pending.pop()
                        if id(item) in seen:continue
                        seen.add(id(item));total+=sys.getsizeof(item)
                        if type(item) is dict:pending.extend(item.keys());pending.extend(item.values())
                        elif type(item) in (list,tuple):pending.extend(item)
                    return total
                forecasts.append(weight(kwargs['roots'])+sum(weight(v) for v in kwargs['copies'])+kwargs['extra']+reader._source_stability_bytes())
            return original_before(operation,**kwargs)
        monkeypatch.setattr(api(),'_context_before',before)
        first=view.describe()
        assert first.parent_manifest is not None and len(forecasts)==1 and state._context_arena.get() is None
        forecast=forecasts[0];original_load=view._load;copy_calls=[]
        def load():
            loaded=original_load()
            reader.limits=replace(reader.limits,maximum_metadata_bytes=forecast-1)
            return loaded
        def copying(value):copy_calls.append(value);return original_copy(value)
        monkeypatch.setattr(view,'_load',load);monkeypatch.setattr(api(),'_context_copy',copying)
        with pytest.raises(ContractError,match='observations description.*metadata') as caught:view.describe()
        assert caught.value.code=='budget_exceeded' and copy_calls==[]
        print('STANDALONE_PARENT_COPY_FORECAST',forecast)


@pytest.mark.parametrize('kind',['missing_manifest','tampered_manifest','missing_checkpoint','changed_parent','changed_cutoff','original_integrity','wrong_current_evidence'])
def test_completed_restore_requires_exact_immediate_parent_and_cutoff(tmp_path,kind):
    from consumer_context_fixture import completed_fixture
    selected,artifact,_,_,_,parent=completed_fixture(tmp_path)
    if kind=='missing_manifest':(tmp_path/parent['manifest']['path']).unlink()
    elif kind=='tampered_manifest':
        p=tmp_path/parent['manifest']['path'];p.write_bytes(p.read_bytes()+b' ')
    elif kind=='missing_checkpoint':(tmp_path/'checkpoint.json').unlink()
    else:
        def alter(v):
            r=v['restoration']
            if kind=='changed_parent':r['parent_state']['identity']['attempt']='99'
            elif kind=='changed_cutoff':r['history']['last_record']='1'
            elif kind=='original_integrity':r['original']['integrity'][0]['decimal']='0'
            else:r['original']['checkpoint']['sha256']='f'*64
        change(tmp_path,selected,artifact,alter)
    with pytest.raises(ContractError):
        with StateReader(tmp_path,state_ref=selected,expected_identity=selected['identity']) as reader:
            api().FinalObservationsReader(reader,artifact_ref=artifact).describe()


def test_one_owner_cache_reopens_with_new_proof_and_no_full_payload_scan(tmp_path):
    selected,artifact,_,_,_=fixture(tmp_path)
    with StateReader(tmp_path,state_ref=selected,expected_identity=selected['identity'],limits=ResultReadLimits(maximum_open_files=1)) as reader:
        view=api().FinalObservationsReader(reader,artifact_ref=artifact)
        assert view.read_array('state_index',(1,)).values==1
        assert len(reader._handles)==1
        assert not any(key[0]=='full.h5' for key in reader._handles)


@pytest.mark.parametrize('literal,expected',[('yes','yes'),('off','off'),('TrUe','TrUe'),('True',True),
    ('FALSE',False),('NULL',None),('~',None),('1',1),('1.0',1.0),('1e3',1000.0),('-0.0',-0.0),
    ('010',8),('+010',10),('0x10',16),('"010"','010'),('!!str 200.0','200.0'),('!!float 200',200.),
    ('"2001-01-01"','2001-01-01'),('"Сетка"','Сетка'),('[]',[]),('{}',{})])
def test_portable_yaml_scalar_profile_from_exact_0417_primary_source(literal,expected):
    # Resolver/constructor source is YAML0.4.17 in the qualified depot. These
    # fixed expected types/bits are not computed by the Results parser.
    raw=('name: literal\nroot: {kind: level, name: x, axes: [{name: value, values: ['+literal+']}], child: {kind: leaf, name: end}}\n').encode()
    value=api()._portable_plan(raw,1_000_000)['root']['axes'][0]['values'][0]
    assert type(value) is type(expected) and value==expected
    if type(expected) is float:
        import struct
        assert struct.pack('>d',value)==struct.pack('>d',expected)


def test_all_nested_operations_share_physical_allowance_and_report_real_deliveries(tmp_path,monkeypatch):
    from qcl_negf_results import state
    selected,artifact,_,_,_=fixture(tmp_path)
    with StateReader(tmp_path,state_ref=selected,expected_identity=selected['identity']) as reader:
        deliveries=[];original=state._IOBudget.account
        def counted(budget,n):deliveries.append((id(budget),n));return original(budget,n)
        monkeypatch.setattr(state._IOBudget,'account',counted)
        d=api().FinalObservationsReader(reader,artifact_ref=artifact).describe()
        assert len({key for key,_ in deliveries})==1
        assert d.io_counters['bytes_read_total']==sum(n for _,n in deliveries)
        assert d.io_counters['sha_bytes']>=int(artifact['bytes'])+int(d.diagnostics['bytes'])
    with StateReader(tmp_path,state_ref=selected,expected_identity=selected['identity'],limits=ResultReadLimits(maximum_io_bytes=50_000)) as reader:
        with pytest.raises(ContractError,match='budget'):
            api().FinalObservationsReader(reader,artifact_ref=artifact).describe()
        assert reader.io_counters['bytes_read_total']<=50_000


def test_original_input_read_keeps_actual_fd_and_path_binding(tmp_path,monkeypatch):
    from qcl_negf_results import state
    selected,artifact,_,_,_=fixture(tmp_path)
    original=state._BudgetStream.read
    def replaced(stream,*args):
        value=original(stream,*args)
        if __import__('os').readlink('/proc/self/fd/'+str(stream._raw.fileno())).endswith('/model.json'):
            path=tmp_path/'model.json';new=tmp_path/'substitute.json'
            new.write_bytes(path.read_bytes());new.replace(path)
        return value
    with StateReader(tmp_path,state_ref=selected,expected_identity=selected['identity']) as reader:
        monkeypatch.setattr(state._BudgetStream,'read',replaced)
        with pytest.raises(ContractError,match='binding changed'):
            api().FinalObservationsReader(reader,artifact_ref=artifact).describe()


def test_metadata_limit_precedes_fixed_declaration_high_level_read(tmp_path,monkeypatch):
    selected,artifact,_,_,_=fixture(tmp_path)
    original=h5py.Dataset.__getitem__
    def guarded(dataset,key):
        assert dataset.name!='/final_observations/declaration_json','oversized scalar read'
        return original(dataset,key)
    with h5py.File(tmp_path/artifact['path'],'r') as handle:
        monkeypatch.setattr(h5py.Dataset,'__getitem__',guarded)
        with pytest.raises(ContractError,match='metadata|limit|budget'):
            api().final_observations_metadata(handle,identity=selected['identity'],invocation={'source_epoch':'epoch'},maximum_bytes=8192)


def test_yaml_depth_event_budget_and_canonical_integer_float_types():
    parser=api()._portable_plan
    prefix=b'name: literal\nroot: {kind: level, name: x, axes: [{name: a, values: ['
    suffix=b']}], child: {kind: leaf, name: end}}\n'
    with pytest.raises(ContractError,match='depth'):parser(prefix+b'['*65+b'0'+b']'*65+suffix,1_000_000)
    with pytest.raises(ContractError,match='allowance') as caught:parser(prefix+b'200.0'+suffix,100)
    assert caught.value.code=='budget_exceeded'
    result=parser(prefix+b'1, 1.0, -0.0, 0.0'+suffix,1_000_000)['root']['axes'][0]['values']
    assert [type(v) for v in result]==[int,float,float,float]


@pytest.mark.parametrize('error_type',[RuntimeError,KeyboardInterrupt])
def test_original_input_temporary_fd_closes_for_any_base_exception(tmp_path,monkeypatch,error_type):
    from qcl_negf_results import state
    selected,artifact,_,_,_=fixture(tmp_path);opened=[]
    original=state._BudgetStream.read
    def interrupted(stream,*args):
        if __import__('os').readlink('/proc/self/fd/'+str(stream._raw.fileno())).endswith('/model.json'):
            opened.append(stream._raw);raise error_type('literal input cancellation')
        return original(stream,*args)
    with StateReader(tmp_path,state_ref=selected,expected_identity=selected['identity']) as reader:
        monkeypatch.setattr(state._BudgetStream,'read',interrupted)
        with pytest.raises(error_type,match='literal input cancellation'):
            api().FinalObservationsReader(reader,artifact_ref=artifact).describe()
        assert opened and all(raw.closed for raw in opened)


def test_native_header_invalid_utf8_is_a_contract_error(tmp_path):
    selected,artifact,_,_,_=fixture(tmp_path)
    with h5py.File(tmp_path/artifact['path'],'r+') as handle:
        del handle['metadata'].attrs['representation']
        handle['metadata'].attrs['representation']=np.bytes_(b'\xff')
    refresh(tmp_path,selected,artifact)
    with StateReader(tmp_path,state_ref=selected,expected_identity=selected['identity']) as reader:
        with pytest.raises(ContractError,match='UTF'):
            api().FinalObservationsReader(reader,artifact_ref=artifact).describe()


@pytest.mark.parametrize('literal',['0b10','-0b10'])
def test_binary_spelling_is_invalid_in_actual_owning_yaml_constructor(literal):
    # Root exact Julia1.13/YAML0.4.17 probe refused these with base8/base10
    # construction errors. They are invalid source input, not a missing codec.
    raw=('name: literal\nroot: {kind: level, name: x, axes: [{name: value, values: ['+literal+']}], child: {kind: leaf, name: end}}\n').encode()
    with pytest.raises(ContractError,match='binary') as caught:api()._portable_plan(raw,1_000_000)
    assert caught.value.code=='corrupt_result'


@pytest.mark.parametrize('raw',[
    b'name: literal\nroot: !!seq {kind: leaf, name: end}\n',
    b'name: literal\nroot: {kind: level, name: x, axes: [{name: a, values: [!!map [1]]}], child: {kind: leaf, name: end}}\n'])
def test_standard_container_tag_must_match_node_kind_before_composition(raw):
    with pytest.raises(ContractError,match='tag.*node kind'):api()._portable_plan(raw,1_000_000)
