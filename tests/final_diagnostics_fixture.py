"""Tiny constructor-only native5 sources; no getter or scientific data import.

The public P01 constructor golden supplies the typed envelope, not numerical
expected values. Local four-array values below are independent wire literals.
"""
from copy import deepcopy
import json
from pathlib import Path

import h5py
import numpy as np

from result_v2_fixture import IDENTITY, LAYOUT, encoded, ieee, integer, metadata, put, reference, source, symbol

KINDS = ['spectral', 'occupied', 'unoccupied', 'broadening', 'raw_occupied', 'raw_unoccupied']
SCALARS = ['minimum_eigenvalue', 'block_norm', 'floor', 'ratio', 'absolute_defect',
           'relative_defect', 'backward_error', 'hermiticity_defect', 'construction_scale']


def fixed_json(parent, name, value, *, cset='utf-8'):
    raw = value if isinstance(value, bytes) else encoded(value)
    dataset = parent.create_dataset(name, shape=(), dtype=h5py.string_dtype(cset, len(raw)))
    # Explicit memory type avoids a high-level ASCII->UTF8 conversion. These
    # are literal UTF8 bytes, stored in an inspectable fixed-width HDF scalar.
    dataset.id.write(h5py.h5s.ALL, h5py.h5s.ALL, np.asarray(raw, dtype=f'S{len(raw)}'),
                     mtype=dataset.id.get_type())
    return dataset


def wire_values():
    return {
        'spectral_integral': np.array([[[1+2j, 3-4j], [5+6j, 7-8j]],
                                       [[9-10j, 11+12j], [13-14j, 15+16j]]], dtype='<c16'),
        'spectral_integral_eigenvalues': np.array([[17., -18.], [19., 20.]], dtype='<f8'),
        'spectral_trace_cumulative': np.array([[21.,22.], [23.,24.], [25.,26.]], dtype='<f8'),
        'negative_energy_integral_scaled': np.array([[31.,32.], [33.,34.], [35.,36.],
                                                    [37.,38.], [39.,40.], [41.,42.]], dtype='<f8'),
    }


def fixture(root, *, status='measured', values=None, chunks=None, compression=None):
    root = Path(root)
    initial = source(root)
    manifest = json.loads((root / initial['manifest']['path']).read_text())
    golden_path = Path(__file__).parents[2] / 'qcl-negf-contracts/tests/fixtures/results-v2/core-producer.json'
    golden = json.loads(golden_path.read_text())
    typed = {item['type']: deepcopy(item) for item in golden['values']}
    state = typed['ExactSolverState']; fields = state['fields']
    fields.update(boundary=symbol('final'), continuation=None, status=symbol('invalid'),
        final_evidence=dict(type='FinalEvidenceSnapshot', fields=dict(
            report=dict(type='ConvergenceReport', fields=dict(passed=False,
                metrics=dict(kind='dict', entries=[]), messages=dict(kind='array', eltype='String',
                    shape=['1'], order='F', values=['manufactured failure']))),
            original_converged=False, original_status=symbol('invalid'),
            observations=dict(type='CoreRecord', fields=dict(entries=[])))))
    for name in ('Gᴿ','Gˡ','Gᵍ','A'):
        fields['scba']['fields']['green']['fields'][name]['shape'] = ['3','2','2','2']
    with h5py.File(root / 'full.h5','w') as handle:
        metadata(handle, representation='array_payload')
        handle['metadata'].attrs['schema'] = np.bytes_('qcl-negf-physics-v5')
        handle['metadata'].attrs['artifact_role'] = np.bytes_('physics.full')
    full = reference(root,'full.h5','qcl-negf-physics-v5','application/x-hdf5')
    def bind(node):
        if isinstance(node, dict):
            if node.get('kind') == 'array' and 'file' in node:
                node['file'] = deepcopy(full)
            for child in node.values(): bind(child)
        elif isinstance(node,list):
            for child in node: bind(child)
    bind(state)
    # This owner is deliberately unopened by A1: raw descriptor validation is
    # not full-state/native scientific validation or an exact-restore claim.
    checkpoint = dict(schema='qcl-negf.checkpoint-declaration.v1', contract_set='qcl-negf.results.v2',
        identity=deepcopy(IDENTITY), lineage=dict(parent_state=None), invocation=dict(source_epoch='epoch'),
        state_id='state-1',state_sequence='1',core_layout_sha256=LAYOUT,
        numerical_binding=dict(identity=deepcopy(fields['identity']),execution=typed['ExactExecutionIdentity']),
        history=deepcopy(manifest['history']),state=state,payloads=[full],witness_targets=[])
    cpref=put(root,'checkpoint.json',checkpoint,checkpoint['schema'])
    arrays = wire_values() if values is None else values
    def descriptor(name, eltype, axes, units='1'):
        return dict(dataset='/final_diagnostics/arrays/'+name, eltype=eltype,
                    shape=[str(n) for n in arrays[name].shape], order='F', axes=axes, units=units)
    integrals = dict(
        spectral_integral=descriptor('spectral_integral','ComplexF64',['k','a','b']),
        spectral_integral_eigenvalues=descriptor('spectral_integral_eigenvalues','Float64',['k','state']),
        spectral_trace_cumulative=descriptor('spectral_trace_cumulative','Float64',['E','k']),
        negative_energy_integral_scaled=descriptor('negative_energy_integral_scaled','Float64',['matrix_kind','k'],
            'scaled solver units; broadening carries energy squared'),
        negative_block_count=[integer(0) for _ in KINDS],beyond_backward_error_block_count=[integer(0) for _ in KINDS],
        invalid_block_count=[integer(0) for _ in KINDS],total_blocks_per_kind=integer(6),
        negative_phase_space_weight_per_m2=[ieee() for _ in range(5)])
    positivity = dict(status=status,reason=None if status=='measured' else 'literal missing scan',data=None)
    if status == 'measured':
        positivity['data']=dict(metric_version=integer(1),matrix_kinds=KINDS.copy(),
            witnesses=[dict(status='measured',matrix_kind=k,energy_index=integer(1),momentum_index=integer(2),
                **{n:ieee() for n in SCALARS}) for k in KINDS],strict_ratio_limit=ieee('3fb999999999999a'),
            strict_absolute_limits_scaled=[ieee() for _ in KINDS],strict_threshold_satisfied=[False]*6,
            scan_order='momentum_energy',integrals=integrals)
    value=dict(schema='qcl-negf.final-diagnostics.v1',contract_set='qcl-negf.results.v2',
        identity=deepcopy(IDENTITY),invocation=dict(source_epoch='epoch'),state_id='state-1',state_sequence='1',
        checkpoint=cpref,dimensions=dict(energy='3',momentum='2',basis='2'),
        solution_metrics=dict(summary_metrics=[dict(name='r_D',value=ieee('7ff8000000000001'))],
            strict_limits=[dict(name='r_D',value=ieee('3fb999999999999a'))],approximate_limits=None),positivity=positivity)
    with h5py.File(root/'analysis.h5','w') as handle:
        group=handle.create_group('metadata')
        group.attrs.update({k:np.bytes_(v) for k,v in dict(schema='qcl-negf-physics-analysis-v5',schema_version='5.0',
            contract_set='qcl-negf.results.v2',artifact_role='physics.analysis',representation='final_diagnostics',
            core_layout_sha256=LAYOUT).items()})
        fixed_json(group,'identity_json',IDENTITY); fixed_json(group,'invocation_json',dict(source_epoch='epoch'))
        fixed_json(handle,'final_diagnostics/declaration_json',value)
        if status=='measured':
            for name,array in arrays.items():
                handle.create_dataset('/final_diagnostics/arrays/'+name,data=array,
                                      chunks=chunks if chunks is None else True,compression=compression)
    analysis=reference(root,'analysis.h5','qcl-negf-physics-analysis-v5','application/x-hdf5')
    manifest.update(closed=True,checkpoint=cpref,numerical_binding=deepcopy(checkpoint['numerical_binding']),
        artifacts=[dict(full,role='physics.full',profile='full-state',dependencies=[]),
            dict(cpref,role='recovery',profile='full-state',dependencies=[full['path']]),
            dict(analysis,role='physics.analysis',profile='science',dependencies=[cpref['path']])])
    manifest['status']['process'].update(state='failed',exit_code=1,reason='manufactured failure')
    ref=put(root,'selected.json',manifest,manifest['schema'])
    selected={k:deepcopy(manifest[k]) for k in ('identity','state_id','state_sequence')} | dict(manifest=ref)
    return selected,analysis,value,checkpoint


def refresh(root, selected, analysis):
    analysis.update(reference(root,analysis['path'],analysis['schema'],analysis['media_type']))
    manifest=json.loads((root/selected['manifest']['path']).read_text())
    next(item for item in manifest['artifacts'] if item['path']==analysis['path']).update(analysis)
    selected['manifest']=put(root,'selected.json',manifest,manifest['schema'])


def change_declaration(root, selected, analysis, change):
    with h5py.File(root/analysis['path'],'r+') as handle:
        value=json.loads(bytes(handle['final_diagnostics/declaration_json'][()]))
        change(value)
        del handle['final_diagnostics/declaration_json']
        fixed_json(handle,'final_diagnostics/declaration_json',value)
    refresh(root,selected,analysis)
