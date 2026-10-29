"""Literal constructor inputs for native5 consumer tests, never a solver oracle.

Only public bundled schema/declaration JSON supplies grammar. Array cells and
input bytes below are manufactured independently from Core getters/archives.
"""
from copy import deepcopy
import hashlib
import json
import struct
from pathlib import Path

import h5py
import numpy as np
from final_diagnostics_fixture import fixture as diagnostic_fixture, fixed_json
from result_v2_fixture import encoded, ieee, integer, put, reference, symbol

PLAN = b'''# original manufactured input; preserve this comment\nname: configured_operating_points\nroot:\n  kind: level\n  name: temperature\n  axes: [{name: temperature_K, values: [200.0]}]\n  child:\n    kind: level\n    name: voltage\n    axes: [{name: voltage_per_period_V, values: [0.05]}]\n    child: {kind: leaf, name: poisson_scba}\n'''


def canonical(value):
    """Literal test implementation of the published canonical preimage grammar."""
    if value is None: return b'n;'
    if type(value) is bool: return b'b1;' if value else b'b0;'
    if type(value) in (str, int, float):
        tag, raw = ((b's', value.encode()) if type(value) is str else
                    (b'i', str(value).encode()) if type(value) is int else
                    (b'f', struct.pack('>d', value).hex().encode()))
        return tag+str(len(raw)).encode()+b':'+raw
    if type(value) is list:
        return b'a'+str(len(value)).encode()+b'['+b''.join(map(canonical,value))+b']'
    return b'm'+str(len(value)).encode()+b'{'+b''.join(canonical(k)+canonical(value[k]) for k in sorted(value))+b'}'


def fixture(root, *, null_outputs=(), chunks=False, plan_bytes=PLAN, invalid_model=False, plan_kind="nested_sweep"):
    root=Path(root)
    selected, diagnostics, diagnostic, checkpoint=diagnostic_fixture(root)
    manifest=json.loads((root/selected['manifest']['path']).read_bytes())
    state=checkpoint['state']['fields']; scba=state['scba']['fields']
    state['published_U']['shape']=state['published_n']['shape']=['4']
    for owner in [scba['green']['fields']]+[entry['value']['fields'] for entry in scba['scattering']['entries']]+[scba[n]['fields'] for n in ('embedding','embedding_plus','embedding_minus')]:
        for value in owner.values():
            if isinstance(value,dict) and value.get('kind')=='array' and len(value['shape'])==4: value['shape']=['3','2','2','2']
    # Every declared Core dataset exists, rather than the A1 unopened owner stub.
    def arrays(value):
        if isinstance(value,dict):
            if value.get('kind')=='array' and 'file' in value: yield value
            for child in value.values(): yield from arrays(child)
        elif isinstance(value,list):
            for child in value: yield from arrays(child)
    with h5py.File(root/'full.h5','r+') as handle:
        for d in arrays(checkpoint['state']):
            dtype={'Float64':'<f8','ComplexF64':'<c16','Int64':'<i8','Bool':'?'}[d['eltype']]
            if d['dataset'] not in handle: handle.create_dataset(d['dataset'],shape=tuple(map(int,d['shape'])),dtype=dtype)
    full=reference(root,'full.h5','qcl-negf-physics-v5','application/x-hdf5')
    for d in arrays(checkpoint['state']): d['file']=deepcopy(full)
    checkpoint['payloads']=[full]
    golden=json.loads((Path(__file__).parents[2]/'qcl-negf-contracts/tests/fixtures/results-v2/core-producer.json').read_bytes())
    typed={v['type']:v for v in golden['values']}
    scattering=dict(type='ScatteringOptions',fields={n:False for n in ('LO','acoustic','impurity','IFR','alloy')});flags=deepcopy(scattering['fields'])
    core_schema=json.loads((Path(__file__).parents[2]/'qcl-negf-contracts/src/qcl_negf_contracts/schemas/core-state.schema.json').read_bytes())
    def options(name):
        if name=='Float64':return ieee()
        if name=='Int64':return integer(0)
        if name=='Symbol':return symbol('none')
        if name=='Bool':return False
        if name.startswith('Union_'):return None
        fields=core_schema['$defs'][name]['properties']['fields']['properties']
        return dict(type=name,fields={k:options(v['$ref'].split('/')[-1]) for k,v in fields.items()})
    configuration=dict(scope='stationary_transport_inputs_v1',inputs={name:{} for name in
        ('physical','numerical','scales','solver','kernel_construction','algorithms','physical_models','domain_adaptation')},derived=dict(effective_seed_eV=.01))
    configuration['inputs'].update(scattering=flags,production=dict(physics_markers='none',verify_fft_roundoff=False))
    model=dict(schema='qcl-negf-resolved-configuration-v3',contract_set='qcl-negf.results.v2',configuration=configuration,
               configuration_hash=hashlib.sha256(canonical(configuration)).hexdigest(),hash_encoding='qcl-negf-canonical-bytes-v1')
    if invalid_model:model['configuration']['inputs']['physical']['changed']=True
    modelref=put(root,'model.json',model,model['schema'])
    if plan_kind=='scientific_plan_v2':
        planref=put(root,'plan.json',scientific_plan(model['configuration_hash']),'qcl-negf-scientific-plan-v2')
    else:
        (root/'plan.yaml').write_bytes(plan_bytes)
        planref=reference(root,'plan.yaml','qcl-negf.nested-sweep-plan.v1','text/plain')
    identity=deepcopy(selected['identity']);identity.update(model_sha256=model['configuration_hash'],plan_sha256=planref['sha256'])
    for value in (manifest,checkpoint,diagnostic,selected):value['identity']=deepcopy(identity)
    checkpoint['numerical_binding']=deepcopy(manifest['numerical_binding'])
    cpref=put(root,'checkpoint.json',checkpoint,checkpoint['schema'])
    diagnostic['checkpoint']=cpref
    # Fixed UTF8 metadata at both HDF owners follows the new invocation binding.
    for filename in ('analysis.h5','full.h5'):
        with h5py.File(root/filename,'r+') as handle:
            del handle['metadata/identity_json'];fixed_json(handle,'metadata/identity_json',identity)
            if filename=='analysis.h5':
                del handle['final_diagnostics/declaration_json'];fixed_json(handle,'final_diagnostics/declaration_json',diagnostic)
    full=reference(root,'full.h5',full['schema'],'application/x-hdf5')
    for d in arrays(checkpoint['state']):d['file']=deepcopy(full)
    checkpoint['payloads']=[full];cpref=put(root,'checkpoint.json',checkpoint,checkpoint['schema'])
    diagnostic['checkpoint']=cpref
    with h5py.File(root/'analysis.h5','r+') as handle:
        del handle['final_diagnostics/declaration_json'];fixed_json(handle,'final_diagnostics/declaration_json',diagnostic)
    diagnostics=reference(root,'analysis.h5',diagnostics['schema'],'application/x-hdf5')
    schema=json.loads((Path(__file__).parents[2]/'qcl-negf-contracts/src/qcl_negf_contracts/schemas/final-observations.schema.json').read_bytes())
    shape={'z':4,'E':3,'k':2,'state':2,'state_prime':2}; descriptors={};values={}
    for key,item in schema['$defs'].items():
        if not key.startswith('Array_'):continue
        name=key[6:];p=item['properties'];d={k:deepcopy(v['const']) for k,v in p.items() if 'const' in v}
        for k in ('axes','coordinate_fields','weight_fields'):d[k]=[q['const'] for q in p[k]['prefixItems']]
        d['shape']=[str(shape[a]) for a in d['axes']];descriptors[name]=d
        count=np.prod(list(map(int,d['shape'])));a=np.arange(count,dtype='<f8').reshape(tuple(map(int,d['shape'])))
        values[name]=((a+1j*(a+1)).astype('<c16') if d['eltype']=='ComplexF64' else
                      a.astype('<i8') if d['eltype']=='Int64' else (a%2==0) if d['eltype']=='Bool' else a)
    context=dict(dimensions={k:str(v) for k,v in dict(Nz=4,NE=3,Nk=2,Nb=2).items()},domain_revision=integer(0),
        discretization_id='c'*64,basis_id='d'*64,scales={k:ieee('3ff0000000000000') for k in ('E0_eV','L0_m','J0_A_per_m2')},
        inputs={k:ieee() for k in schema['$defs']['FinalObservationInputs']['properties']},problem_scattering=scattering,
        enabled_scattering=[],selfenergy_families=[],physical_models=options('PhysicalModelOptions'),
        algorithms=deepcopy(state['measurement_context']['fields']['algorithms']),energy_shift_discretization='nodal_linear',
        localization='pzp',boundary_condition='stationary_field_periodic_embedding',basis_contract='orthonormal_finite_projected',
        energy_reference='scaled_E0_plus_E_ref',effective_gauge='k0_ascending_spatial_first_max_real_positive')
    context['inputs'].update(T_L_K=ieee('4069000000000000'),V_period_V=ieee('3fa999999999999a'),g_s=integer(2))
    views=dict(current_density_A_per_m2=ieee('8000000000000000'),sheet_density_per_m2=ieee('7ff8000000001234'))
    for name,d in descriptors.items():(context if name in schema['$defs']['FinalObservationContext']['properties'] else views)[name]=d
    def components(v):return {a:deepcopy(v['fields'][b]) for a,b in [('retarded','Σᴿ'),('lesser','Σˡ'),('greater','Σᵍ')]}
    families={e['key']['value']:e['value'] for e in scba['scattering']['entries']}
    order=[v['value'] for v in state['family_order']][:-2]
    context.update(selfenergy_families=order,enabled_scattering=[n for n in order if n!='electron_electron'])
    bindings=dict(green={a:deepcopy(scba['green']['fields'][b]) for a,b in [('retarded','Gᴿ'),('lesser','Gˡ'),('greater','Gᵍ'),('spectral','A')]},
        scattering=[dict(family=n,**components(families[n])) for n in order],
        embedding={n:components(scba[k]) for n,k in [('total','embedding'),('plus','embedding_plus'),('minus','embedding_minus')]})
    coords=dict(temperature_K=context['inputs']['T_L_K'],voltage_per_period_V=context['inputs']['V_period_V'])
    request=dict(kind=plan_kind,plan=planref,model=modelref,**coords,scattering_requested=flags,
        coordinate_checks={k:dict(status='verified',reason=None,requested=coords[n],actual=coords[n]) for k,n in [('temperature','temperature_K'),('voltage','voltage_per_period_V')]})
    value=dict(schema='qcl-negf.final-observations.v1',contract_set='qcl-negf.results.v2',identity=identity,
        invocation=dict(source_epoch='epoch'),state_id='state-1',state_sequence='1',checkpoint=cpref,diagnostics=diagnostics,
        source_mode='produced',binding=dict(identity=deepcopy(state['identity']),terminal_integrity=deepcopy(state['integrity'])),
        context=context,request=request,state_bindings=bindings,views=views,summary=dict(status='invalid',quality='invalid',
        converged=False,registry_version='literal-v1',iterative_converged=False,fixed_hartree_converged=False,physical_gates_passed=False,
        stationary_candidate_accepted=False,scientific_accepted=None,discretization_verified='not_measured',scientific_acceptance_reason='not_measured',outer_confirmation_count=integer(0)),restoration=None,issues=[])
    for name in null_outputs:
        views[name]=None;value['issues'].append(dict(target=name,status='error',reason='projection_failed',detail='literal unavailable'))
    with h5py.File(root/'observations.h5','w') as handle:
        group=handle.create_group('metadata')
        group.attrs.update({k:np.bytes_(v) for k,v in dict(schema='qcl-negf-physics-analysis-v5',schema_version='5.0',contract_set='qcl-negf.results.v2',artifact_role='physics.analysis',representation='final_observations',core_layout_sha256=checkpoint['core_layout_sha256']).items()})
        fixed_json(group,'identity_json',identity);fixed_json(group,'invocation_json',value['invocation']);fixed_json(handle,'final_observations/declaration_json',value)
        for name,array in values.items():
            if name not in null_outputs:
                dtype=h5py.enum_dtype({'FALSE':0,'TRUE':1},basetype='<u1') if array.dtype.kind=='b' else array.dtype
                handle.create_dataset('/final_observations/arrays/'+name,data=array.astype('u1') if array.dtype.kind=='b' else array,dtype=dtype,chunks=True if chunks else None,compression='gzip' if chunks else None)
    artifact=reference(root,'observations.h5','qcl-negf-physics-analysis-v5','application/x-hdf5')
    manifest.update(identity=identity,checkpoint=cpref,artifacts=[dict(full,role='physics.full',profile='full-state',dependencies=[]),dict(cpref,role='recovery',profile='full-state',dependencies=[full['path']]),dict(diagnostics,role='physics.analysis',profile='science',dependencies=[cpref['path']]),dict(planref,role='plan',profile='science',dependencies=[]),dict(modelref,role='model',profile='science',dependencies=[]),dict(artifact,role='physics.analysis',profile='science',dependencies=[r['path'] for r in (cpref,diagnostics,planref,modelref)])])
    for key in ('convergence','physics'):manifest['status'][key]=dict(state='recorded',reason=None,evidence=artifact)
    selected.update(identity=identity,manifest=put(root,'selected.json',manifest,manifest['schema']))
    return selected,artifact,value,checkpoint,values


def refresh(root,selected,artifact):
    artifact.update(reference(root,artifact['path'],artifact['schema'],artifact['media_type']))
    manifest=json.loads((root/selected['manifest']['path']).read_bytes())
    next(v for v in manifest['artifacts'] if v['path']==artifact['path']).update(artifact)
    for key in ('convergence','physics'):manifest['status'][key]['evidence']=deepcopy(artifact)
    selected['manifest']=put(root,'selected.json',manifest,manifest['schema'])


def change(root,selected,artifact,fn):
    with h5py.File(root/artifact['path'],'r+') as handle:
        value=json.loads(bytes(handle['final_observations/declaration_json'][()]))
        fn(value);del handle['final_observations/declaration_json'];fixed_json(handle,'final_observations/declaration_json',value)
    refresh(root,selected,artifact)


def scientific_plan(model_sha256):
    """One literal point plus public resolved-passport grammar metadata."""
    public=json.loads((Path(__file__).parents[2]/'qcl-negf-contracts/tests/fixtures/resolved-plan/metadata.json').read_bytes())
    snapshot=public['snapshot']
    return dict(schema='qcl-negf-scientific-plan-v2',model_revision='transport-contract-v2',
        resolved_snapshot_schema='qcl-negf-resolved-configuration-v1',root_definition_id='literal',root_kind='study',
        name='Literal consumer input',fingerprint='1'*64,scientific_fingerprint='2'*64,maximum_solver_runs=1,
        computation_count=1,inclusions=[],executions=[dict(id='exec',definition_id='literal',variant_id='variant',
            method_id='method',label='',point_ids=['point'],resolved_configuration=snapshot['raw_configuration'],provenance={},
            policies={},estimated_peak_bytes=256,purpose='diagnostic',repetition=1,operation='stationary',effective_inputs=dict(
                seed={},energy_shift='none',energy_nodes=3,energy_min_eV=-1.,energy_max_eV=1.),output=public['output'],
            resolved_snapshot=snapshot,resource_estimate=public['receipt'])],points=[dict(id='point',execution_id='exec',
            temperature_K=200.,voltage_per_period_V=.05,branch='forward',order=1,predecessor_id=None,initialization='cold',
            effective_inputs=dict(temperature_K=200.,voltage_per_period_V=.05,configuration_hash='3'*64,
                hash_encoding='qcl-negf-canonical-bytes-v1',axes_applied=True,model_sha256=model_sha256))],
        resources=dict(maximum_active_executions=1,estimated_peak_bytes=256,minimum_output_bytes=0),nodes=[],campaign=None,science_archive_preflight=None)


def completed_fixture(root):
    """Fresh immediate parent and child, retaining original failed evidence."""
    import shutil
    root=Path(root)
    selected,artifact,value,checkpoint,values=fixture(root)
    manifest=json.loads((root/selected['manifest']['path']).read_bytes())
    parent=deepcopy(selected);parent['manifest']=put(root,'parent-state.json',manifest,manifest['schema'])
    identity=deepcopy(selected['identity']);identity.update(attempt='2',branch_id='restored')
    invocation=dict(source_epoch='restored-epoch')
    child_checkpoint=deepcopy(checkpoint);child_checkpoint.update(identity=identity,invocation=invocation,state_id='state-child',lineage=dict(parent_state=parent))
    shutil.copyfile(root/'full.h5',root/'child-full.h5')
    with h5py.File(root/'child-full.h5','r+') as handle:
        for n,v in [('identity_json',identity),('invocation_json',invocation)]:
            del handle['metadata/'+n];fixed_json(handle,'metadata/'+n,v)
    full=reference(root,'child-full.h5','qcl-negf-physics-v5','application/x-hdf5')
    def rebind(node):
        if isinstance(node,dict):
            if node.get('kind')=='array' and 'file' in node:node['file']=deepcopy(full)
            for v in node.values():rebind(v)
        elif isinstance(node,list):
            for v in node:rebind(v)
    rebind(child_checkpoint['state']);child_checkpoint['payloads']=[full]
    cpref=put(root,'child-checkpoint.json',child_checkpoint,checkpoint['schema'])
    shutil.copyfile(root/'analysis.h5',root/'child-diagnostics.h5')
    with h5py.File(root/'child-diagnostics.h5','r+') as handle:
        diagnostic=json.loads(bytes(handle['final_diagnostics/declaration_json'][()]))
        diagnostic.update(identity=identity,invocation=invocation,state_id='state-child',checkpoint=cpref)
        for n,v in [('metadata/identity_json',identity),('metadata/invocation_json',invocation),('final_diagnostics/declaration_json',diagnostic)]:
            del handle[n];fixed_json(handle,n,v)
    diagnosticref=reference(root,'child-diagnostics.h5','qcl-negf-physics-analysis-v5','application/x-hdf5')
    value=deepcopy(value);value.update(identity=identity,invocation=invocation,state_id='state-child',checkpoint=cpref,
        diagnostics=diagnosticref,source_mode='completed_restore')
    def state_arrays(node):
        if isinstance(node,dict):
            if node.get('kind')=='array' and 'file' in node:node['file']=deepcopy(full)
            for v in node.values():state_arrays(v)
        elif isinstance(node,list):
            for v in node:state_arrays(v)
    state_arrays(value['state_bindings'])
    golden=json.loads((Path(__file__).parents[2]/'qcl-negf-contracts/tests/fixtures/results-v2/core-producer.json').read_bytes())
    receipts=deepcopy(next(v['fields'] for v in golden['values'] if v['type']=='CompletedStateEvidence'))
    receipts['revalidation']['fields']['context']['fields']['final_state_alignment']=symbol('restored_aligned')
    value['restoration']=dict(parent_state=parent,parent_checkpoint=manifest['checkpoint'],history=manifest['history'],
        original=dict(checkpoint=cpref,field='final_evidence',integrity=deepcopy(checkpoint['state']['fields']['integrity'])),
        revalidation=receipts['revalidation'],independent_oracle=receipts['independent_oracle'])
    value['summary']['status']='validation_failed'
    shutil.copyfile(root/'observations.h5',root/'child-observations.h5')
    with h5py.File(root/'child-observations.h5','r+') as handle:
        for n,v in [('metadata/identity_json',identity),('metadata/invocation_json',invocation),('final_observations/declaration_json',value)]:
            del handle[n];fixed_json(handle,n,v)
    artifact=reference(root,'child-observations.h5',artifact['schema'],'application/x-hdf5')
    inputs=[deepcopy(row) for row in manifest['artifacts'] if row['role'] in ('plan','model')]
    manifest.update(identity=identity,invocation=invocation,state_id='state-child',lineage=dict(parent_state=parent),checkpoint=cpref,
        artifacts=[dict(full,role='physics.full',profile='full-state',dependencies=[]),dict(cpref,role='recovery',profile='full-state',dependencies=[full['path']]),
        dict(diagnosticref,role='physics.analysis',profile='science',dependencies=[cpref['path']]),*inputs,
        dict(artifact,role='physics.analysis',profile='science',dependencies=[cpref['path'],diagnosticref['path'],value['request']['plan']['path'],value['request']['model']['path']])])
    for key in ('convergence','physics'):manifest['status'][key]['evidence']=deepcopy(artifact)
    selected.update(identity=identity,state_id='state-child',manifest=put(root,'selected-child.json',manifest,manifest['schema']))
    return selected,artifact,value,child_checkpoint,values,parent
