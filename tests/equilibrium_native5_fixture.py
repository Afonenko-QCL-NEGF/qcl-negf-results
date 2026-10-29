"""Constructor-only native5 projection of independent equilibrium literals.

The staging names are test mutation locators, never a product format/fallback.
All emitted FileRefs, O1/D1 metadata and Core arrays use actual public native5.
No solver/getter or production numerical archive supplies a value.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import struct

import h5py
import numpy as np

from consumer_context_fixture import fixture as context_fixture, canonical
from final_diagnostics_fixture import fixed_json
from result_v2_fixture import put, reference, symbol, integer

FAMILIES=('LO','acoustic','impurity','IFR','alloy')
BOUNDARIES=('embedding_total','embedding_plus','embedding_minus')


def tag(value):
    return dict(kind='ieee',type='f64',bits=struct.pack('>d',float(value)).hex())


def _arrays(node):
    if isinstance(node,dict):
        if node.get('kind')=='array' and 'file' in node:yield node
        for value in node.values():yield from _arrays(value)
    elif isinstance(node,list):
        for value in node:yield from _arrays(value)


def publish(root, commit):
    """Emit fresh tiny typed inputs, preserving staging matrices/attrs verbatim."""
    root=Path(root);native=root/'native';context=commit.get('stationary_assessment',{}).get('model_context')
    with h5py.File(root/'physics.h5','r') as stage:
        ne,nk,nb=(int(stage['numerical_inputs/'+n][()]) for n in ('NE','Nk','Nb'))
        plan=('name: configured_operating_points\nroot:\n  kind: level\n  name: temperature\n'
              '  axes: [{name: temperature_K, values: [70.0]}]\n  child:\n    kind: level\n'
              '    name: voltage\n    axes: [{name: voltage_per_period_V, values: [0.0]}]\n'
              '    child: {kind: leaf, name: poisson_scba}\n').encode()
        selected,observations,value,checkpoint,_=context_fixture(native,ne=ne,nk=nk,nb=nb,plan_bytes=plan)
        with h5py.File(native/'analysis.h5','r') as handle:
            diagnostics=json.loads(bytes(handle['final_diagnostics/declaration_json'][()]))
        manifest=json.loads((native/selected['manifest']['path']).read_bytes())
        state=checkpoint['state']['fields'];scba=state['scba']['fields']
        # Unique exact native datasets for all Core declarations, including
        # untouched grammar scaffolding. No ambiguous shared shape/path stubs.
        for index,d in enumerate(_arrays(checkpoint['state'])):d['dataset']='/arrays/scaffold/'+str(index)
        dummy=deepcopy(checkpoint['payloads'][0]);matrices={};bad_axes=[]
        def matrix(path):
            d=dict(kind='array',file=deepcopy(dummy),dataset='/arrays/'+path,eltype='ComplexF64',
                   shape=list(map(str,(ne,nk,nb,nb))),order='F')
            if path in stage:
                real,imag=stage[path+'/real'],stage[path+'/imag']
                matrices[d['dataset']]=np.asarray(real[...])+1j*np.asarray(imag[...])
                if any(ds.attrs.get('logical_axis_order')!='E,k,a,b' for ds in (real,imag)):bad_axes.append(d)
            return d
        scba['green']['fields'].update({key:matrix('state_dimensionless/'+name) for key,name in
            (('A','A'),('Gᴿ','GR'),('Gˡ','GL'),('Gᵍ','GG'))})
        present=set(stage['selfenergy_dimensionless'])-set(BOUNDARIES)
        actual=commit.get('_family_order',[n for n in (*FAMILIES,'electron_electron') if n in present]+sorted(present-set((*FAMILIES,'electron_electron'))))
        def component(family):
            retarded=matrix('selfenergy_dimensionless/'+family+'/SR')
            matrices[retarded['dataset']]=np.zeros((ne,nk,nb,nb),dtype='<c16')
            return dict(type='SelfEnergyComponents',fields={'Σᴿ':retarded,
                'Σˡ':matrix('selfenergy_dimensionless/'+family+'/SL'),'Σᵍ':matrix('selfenergy_dimensionless/'+family+'/SG')})
        template_type=scba['embedding']['type']
        def typed_component(family):
            d=component(family);d['type']=template_type;return d
        scba['scattering']['entries']=[dict(key=symbol(n),value=typed_component(n)) for n in actual]
        for key,family in (('embedding','embedding_total'),('embedding_plus','embedding_plus'),('embedding_minus','embedding_minus')):
            scba[key]=typed_component(family)
        state['family_order']=[symbol(n) for n in actual+['embedding_plus','embedding_minus']]
        flags={n:bool(stage['numerical_inputs/scattering_'+n][()]) for n in FAMILIES}
        value['context']['problem_scattering']['fields']=flags
        value['context']['enabled_scattering']=deepcopy(context['enabled_scattering'] if context is not None else actual)
        value['context']['selfenergy_families']=actual
        value['context']['scales']={n:tag(stage['scales/'+n][()]) for n in ('E0_eV','L0_m','J0_A_per_m2')}
        for n in ('F_bias_V_per_m','V_period_V','T_L_K','T_LO_K','E_ref_eV','N_dop_2D_per_m2','f_ion'):
            value['context']['inputs'][n]=tag(stage['inputs/'+n][()])
        value['context']['inputs']['g_s']=integer(int(stage['inputs/spin_degeneracy'][()]))
        if context is None:del value['context']['physical_models']
        else:
            value['context']['boundary_condition']=context['boundary_condition']
            assert context['basis_contract']=='orthonormal finite projected basis'
            value['context']['basis_contract']='orthonormal_finite_projected'
            models=value['context']['physical_models']['fields']
            models['lo_population']=symbol(context['physical_models']['lo_population'])
            models['electron_electron']['fields']['mode']=symbol(context['physical_models']['electron_electron']['mode'])
        value['request']['scattering_requested']=flags
        value['request'].update(temperature_K=tag(70.),voltage_per_period_V=tag(0.))
        for key,name,actual_name in [('temperature','temperature_K','T_L_K'),('voltage','voltage_per_period_V','V_period_V')]:
            wanted=value['request'][name];observed=value['context']['inputs'][actual_name]
            value['request']['coordinate_checks'][key]=dict(requested=wanted,actual=observed,
                status='verified' if wanted==observed else 'not_verified',reason=None if wanted==observed else 'field_period_roundtrip')
        model=json.loads((native/'model.json').read_bytes());model['configuration']['inputs']['scattering']=flags
        model['configuration_hash']=hashlib.sha256(canonical(model['configuration'])).hexdigest()
        modelref=put(native,'model.json',model,model['schema']);value['request']['model']=modelref
        identity=deepcopy(selected['identity']);identity['model_sha256']=model['configuration_hash']
        for owner in (value,checkpoint,diagnostics,manifest,selected):owner['identity']=deepcopy(identity)
        def header(handle,role,schema,representation):
            g=handle.create_group('metadata');g.attrs.update({k:np.bytes_(v) for k,v in dict(
                schema=schema,schema_version='5.0',contract_set='qcl-negf.results.v2',artifact_role=role,
                representation=representation,core_layout_sha256=checkpoint['core_layout_sha256']).items()})
            fixed_json(g,'identity_json',identity);fixed_json(g,'invocation_json',value['invocation'])
        with h5py.File(native/'full.h5','w') as handle:
            header(handle,'physics.full','qcl-negf-physics-v5','array_payload')
            seen=set()
            for d in _arrays(checkpoint['state']):
                if d['dataset'] in seen:continue
                seen.add(d['dataset'])
                dtype={'Float64':'<f8','ComplexF64':'<c16','Int64':'<i8','Bool':'?'}[d['eltype']]
                if d['dataset'] in matrices:handle.create_dataset(d['dataset'],data=matrices[d['dataset']],dtype=dtype,chunks=commit.get('_chunks',(1,1,nb,nb)))
                elif d['dataset'].startswith('/arrays/scaffold/'):handle.create_dataset(d['dataset'],shape=tuple(map(int,d['shape'])),dtype=dtype)
        full=reference(native,'full.h5','qcl-negf-physics-v5','application/x-hdf5')
        for d in _arrays(checkpoint['state']):d['file']=deepcopy(full)
        checkpoint['payloads']=[full];cpref=put(native,'checkpoint.json',checkpoint,checkpoint['schema'])
        diagnostics['checkpoint']=cpref
        with h5py.File(native/'analysis.h5','r+') as handle:
            del handle['metadata']
            # Reuse the original D1 declaration/arrays, changing only closed bindings.
            g=handle.create_group('metadata');g.attrs.update({k:np.bytes_(v) for k,v in dict(schema='qcl-negf-physics-analysis-v5',schema_version='5.0',contract_set='qcl-negf.results.v2',artifact_role='physics.analysis',representation='final_diagnostics',core_layout_sha256=checkpoint['core_layout_sha256']).items()})
            fixed_json(g,'identity_json',identity);fixed_json(g,'invocation_json',value['invocation'])
            del handle['final_diagnostics/declaration_json'];fixed_json(handle,'final_diagnostics/declaration_json',diagnostics)
        diagnosticref=reference(native,'analysis.h5','qcl-negf-physics-analysis-v5','application/x-hdf5')
        value.update(checkpoint=cpref,diagnostics=diagnosticref)
        value['state_bindings']['green']={a:deepcopy(scba['green']['fields'][b]) for a,b in [('spectral','A'),('retarded','Gᴿ'),('lesser','Gˡ'),('greater','Gᵍ')]}
        def components(owner):return {a:deepcopy(owner['fields'][b]) for a,b in [('retarded','Σᴿ'),('lesser','Σˡ'),('greater','Σᵍ')]}
        value['state_bindings']['scattering']=[dict(family=n,**components(e['value'])) for n,e in zip(actual,scba['scattering']['entries'],strict=True)]
        value['state_bindings']['embedding']={n:components(scba[k]) for n,k in [('total','embedding'),('plus','embedding_plus'),('minus','embedding_minus')]}
        if bad_axes:value['state_bindings']['green']['retarded']['axes']=['E','k','b','a']
        with h5py.File(native/observations['path'],'r+') as handle:
            del handle['metadata/identity_json'];fixed_json(handle,'metadata/identity_json',identity)
            for name,old in [('energy_scaled','energy'),('wE','wE'),('momentum_scaled','k'),('wk','wk'),('state_index','state_index')]:
                handle[value['context'][name]['dataset']][...]=stage['grids_dimensionless/'+old][...]
            del handle['final_observations/declaration_json'];fixed_json(handle,'final_observations/declaration_json',value)
        observations=reference(native,observations['path'],observations['schema'],'application/x-hdf5')
        inputs=[dict(r) for r in manifest['artifacts'] if r['role'] in ('plan','model')]
        next(r for r in inputs if r['role']=='model').update(modelref)
        manifest.update(checkpoint=cpref,artifacts=[dict(full,role='physics.full',profile='full-state',dependencies=[]),
            dict(cpref,role='recovery',profile='full-state',dependencies=['full.h5']),dict(diagnosticref,role='physics.analysis',profile='science',dependencies=['checkpoint.json']),
            *inputs,dict(observations,role='physics.analysis',profile='science',dependencies=[cpref['path'],diagnosticref['path'],value['request']['plan']['path'],modelref['path']])])
        for k in ('convergence','physics'):manifest['status'][k]=dict(state='recorded',reason=None,evidence=observations)
        selected['manifest']=put(native,'selected.json',manifest,manifest['schema'])
        commit.update(_native_root=native,_selected=selected,_observations=observations,_value=value,
            artifacts=[dict(full,bytes=int(full['bytes']))])
        return selected,observations
