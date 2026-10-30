"""Literal tiny native5 consumer inputs, never a solve or a computed oracle."""
from copy import deepcopy
import json
from pathlib import Path
import struct

import h5py
import numpy as np

from result_v2_fixture import encoded, integer, metadata, put, record, reference, symbol


def tag(value):
    return dict(kind='ieee',type='f64',bits=struct.pack('>d',float(value)).hex())


def witness_history(root, *, attempts=1, updates=None, scientific_plan=False, valid_full=False, identity_override=None):
    """Twenty scalar rows/ten odd witnesses per original attempt; one shared owner."""
    from consumer_context_fixture import fixture
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    selected,_,_,checkpoint,_=fixture(root/'original',plan_kind='scientific_plan_v2' if scientific_plan else 'nested_sweep')
    manifest=json.loads((root/'original'/selected['manifest']['path']).read_bytes())
    original_identity=deepcopy(selected['identity']);original_identity.update(identity_override or {})
    checkpoint['identity']=deepcopy(original_identity)
    if valid_full:
        from qcl_negf_results.state import _arrays
        declarations=list(_arrays(checkpoint['state']))
        target=root/'original'/'native-full.h5'
        with h5py.File(root/'original'/'full.h5','r') as source,h5py.File(target,'w') as handle:
            source.copy('metadata',handle)
            del handle['metadata/identity_json'];handle['metadata/identity_json']=np.bytes_(encoded(original_identity))
            for index,d in enumerate(declarations):
                d['dataset']='/arrays/literal-'+str(index)
                dtype={'Float64':'<f8','ComplexF64':'<c16','Int64':'<i8','Bool':'?'}[d['eltype']]
                handle.create_dataset(d['dataset'],shape=tuple(map(int,d['shape'])),dtype=dtype)
        full_ref=reference(root/'original','native-full.h5','qcl-negf-physics-v5','application/x-hdf5')
        for d in declarations:d['file']=deepcopy(full_ref)
        checkpoint['payloads']=[full_ref]
    def rebase(value):
        if isinstance(value,dict):
            if {'path','bytes','sha256','media_type','schema'}<=value.keys():value['path']='original/'+value['path']
            for item in value.values():rebase(item)
        elif isinstance(value,list):
            for item in value:rebase(item)
    rebase(checkpoint)
    full=deepcopy(checkpoint['payloads'][0])
    parent=None;all_records=[]
    for attempt in range(1,attempts+1):
        identity=dict(original_identity,attempt=str(attempt));prefix=f'attempt-{attempt}'
        leafroot=root/prefix;leafroot.mkdir()
        with h5py.File(leafroot/'arrays.h5','w') as handle:
            metadata(handle,representation='array_payload',identity=identity)
            for sequence in range(1,21,2):
                matrix=np.eye(3,dtype='<c16')*-2.34e-12
                vector=np.zeros(3,dtype='<c16')
                vector.view('<u8')[:]=[0x7ff8000000001234,0,0x8000000000000000,0,0,0]
                handle[f'arrays/matrix-{sequence}']=matrix
                handle[f'arrays/vector-{sequence}']=vector
        owner=reference(root,prefix+'/arrays.h5','qcl-negf-scientific-history-v5','application/x-hdf5')
        rows=[]
        for sequence in range(1,21):
            row=record(sequence,sequence);row['identity']=deepcopy(identity)
            row['core']['fields']['row']['fields']['ν']=integer(sequence+(attempt-1)*20)
            row['core']['fields']['inner_iteration']=integer(sequence+(attempt-1)*20)
            row['row_key']['inner_iteration']=integer(sequence+(attempt-1)*20)
            if sequence%2:
                array=lambda name,shape:dict(kind='array',eltype='ComplexF64',shape=list(map(str,shape)),order='F',file=deepcopy(owner),dataset='/arrays/'+name+'-'+str(sequence))
                w={name:tag(0) for name in ('minimum_eigenvalue','block_norm','backward_error','absolute_defect','relative_defect','hermiticity_defect')}
                w.update(matrix_kind=symbol('unoccupied'),energy_index=integer(1),momentum_index=integer(1),
                    ratio=tag(-1 if sequence<5 else (30 if attempt==1 else 3) if sequence==13 else 1),
                    matrices=dict(kind='dict',entries=[dict(key='Gp',value=array('matrix',(3,3)))]),eigenvector=array('vector',(3,)))
                row['core']['fields']['row']['fields']['witness']=dict(type='SCBAPhysicsWitness',fields=w)
            rows.append(row)
        for target_sequence,ratio in (updates or {}).get(attempt,()):
            row=deepcopy(rows[target_sequence-1]);ordinal=len(rows)+1
            row['stream_ordinal']=str(ordinal);row['core']['fields']['sequence']=integer(ordinal)
            row['core']['fields']['action']=symbol('witness_update')
            row['core']['fields']['row']['fields']['witness']['fields']['ratio']=tag(ratio)
            row['target']=dict(scope='same_leaf',source_identity=deepcopy(identity),source_epoch='epoch',
                append_ordinal=str(target_sequence),leaf=None,row_offset=str(target_sequence-1),anchor_state=None,anchor_history=None)
            rows.append(row)
        pieces=[encoded(r) for r in rows];offsets=np.cumsum([0]+list(map(len,pieces)),dtype=np.uint64)
        with h5py.File(leafroot/'records.h5','w') as handle:
            metadata(handle,identity=identity)
            handle['records/offsets']=offsets;handle['records/json_utf8']=np.frombuffer(b''.join(pieces),dtype='u1')
            handle['records/stream_ordinal']=np.arange(1,len(rows)+1,dtype='<u8')
        leaf=reference(root,prefix+'/records.h5','qcl-negf-scientific-history-v5','application/x-hdf5')
        totals=dict(records=str(len(rows)),last_record=str(len(rows)),scba_rows='20',outer_rows='0',witness_updates=str(len(rows)-20))
        page=dict(schema='qcl-negf.scientific-catalog.v1',contract_set='qcl-negf.results.v2',identity=identity,
            invocation=dict(source_epoch='epoch'),generation='1',previous=None,entries='2',totals=totals,
            range=dict(first_record='1',last_record=str(len(rows))),artifacts=[dict(owner,role='science.history',profile='science',dependencies=[]),dict(leaf,role='science.history',profile='science',dependencies=[owner['path']])])
        pageref=put(root,prefix+'/catalog.json',page,page['schema'])
        pin=dict(head=pageref,entries='2',**totals)
        m=deepcopy(manifest);m.update(identity=identity,history=pin,lineage=dict(parent_state=parent),checkpoint=None,artifacts=[])
        missing=dict(state='not_measured',reason='literal_fixture',evidence=None)
        for key in ('convergence','physics'):m['status'][key]=deepcopy(missing)
        if attempt==1:
            checkpoint['history']=deepcopy(pin)
            cpref=put(root,'original/checkpoint.json',checkpoint,checkpoint['schema'])
            m['checkpoint']=cpref;m['artifacts']=[dict(full,role='physics.full',profile='full-state',dependencies=[]),dict(cpref,role='recovery',profile='full-state',dependencies=[full['path']])]
        mref=put(root,prefix+'/manifest.json',m,m['schema'])
        parent=dict(identity=identity,state_id=m['state_id'],state_sequence=m['state_sequence'],manifest=mref)
        all_records.append(rows)
    return parent,all_records


def selected_arrays(reader, plan):
    """Re-enter only each selected exact record, then use the public slice reader."""
    answer={}
    for attempt in plan['attempts']:
        with_reader=reader
        borrowed=[]
        while with_reader.source.state!=attempt['source_state']:
            with_reader=with_reader._parent_reader(with_reader.source.lineage['parent_state']);borrowed.append(with_reader)
        try:
            with_reader.import_history()
            for entry in attempt['selected']:
                batch=with_reader.history_batch(entry['cursor_before'],maximum_records=1)
                w=batch.records[0]['core']['fields']['row']['fields']['witness']['fields']
                arrays={p['key']:with_reader.read_array(p['value'],(slice(None),)*len(p['value']['shape'])).values.tobytes() for p in w['matrices']['entries']}
                arrays['eigenvector']=with_reader.read_array(w['eigenvector'],(slice(None),)).values.tobytes()
                answer[entry['sequence']]=arrays
        finally:
            for child in reversed(borrowed):child.close()
    return answer


def sealed_phase(root, identity, *, previous=None, generation=1,prefix=""):
    """Literal normal8/6 source prefix: file order must not be sorted by maxima."""
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    offset=8*(generation-1)
    source=dict(session_id='native',clock_domain_id='native-clock',utc_anchor_ns='0',monotonic_anchor_ns='0')
    health=dict(events_seen=str(8*generation),enqueued=str(8*generation),sealed_events=str(8*generation),coalesced_events=str(6*generation),queued_events='0',buffered_bytes='0',peak_buffered_bytes='1024',
        inflight_bytes='0',unsealed_events='0',incomplete=False,incomplete_reasons=[],self_cost=[],
        dropped={k:'0' for k in ('capacity','contention','oversized','closed','invalid','writer_failure')})
    def row(first,last,calls,duration):
        return dict(kind='aggregate',stage='scba',phase='dyson' if last==8 else 'mixing',status='ok',task_width='2',counter_scope='process_inclusive_delta_on_end',
            first_sequence=str(first+offset),last_sequence=str(last+offset),first_core_sequence=None,last_core_sequence=None,call_count=str(calls),
            metrics=[dict(name='duration_seconds',unit='s',sum=duration,measured_calls=str(calls),unavailable_calls='0',reasons=[])],
            time_bounds=dict(min_start_monotonic_ns='10',max_end_monotonic_ns='20',measured_calls=str(calls),unavailable_calls='0',reasons=[]))
    segment=dict(schema='qcl-negf.producer-phase-segment.v1',contract_set='qcl-negf.results.v2',identity=deepcopy(identity),source=source,
        mode='normal',segment_ordinal=str(generation),first_sequence=str(1+offset),last_sequence=str(8+offset),observed_events='8',coalesced_events='6',
        rows=[row(1,8,2,4.5),row(2,6,1,1.25)],health=health,sealed=True)
    stem=lambda name:str(Path(prefix)/name)
    sr=put(root,stem(f'phase-segment-{generation}.json'),segment,segment['schema'])
    publication=dict(schema='qcl-negf.producer-phase-publication.v1',contract_set='qcl-negf.results.v2',identity=deepcopy(identity),source=source,
        generation=str(generation),previous=previous,segment=sr,first_sequence=str(1+offset),last_sequence=str(8+offset),rows='2',health=health)
    pr=put(root,stem(f'phase-publication-{generation}.json'),publication,publication['schema'])
    current=dict(schema='qcl-negf.producer-phase-current.v1',contract_set='qcl-negf.results.v2',identity=deepcopy(identity),source=source,
        publication=pr,current_phase=None,health=health,closed=True)
    put(root,stem('phase-current.json'),current,current['schema'])
    return dict(segment=segment,segment_ref=sr,publication=publication,publication_ref=pr,current=current)


def technical_source(root, *, checkpoint=False, execution="ex", identity_override=None, canonical_plan=None, source_binding_path="source-binding.json"):
    """Actual tiny selected source plus explicit caller assertion contexts, no actor proof."""
    from consumer_context_fixture import scientific_plan
    from result_v2_fixture import source, IDENTITY
    root=Path(root);science=root/'science';technical=root/'technical'
    technical.mkdir(parents=True,exist_ok=True)
    plan=scientific_plan(IDENTITY['model_sha256'])
    planref=put(root,'original-plan.json',plan,plan['schema'])
    identity=dict(IDENTITY,attempt='2',plan_sha256=planref['sha256'])
    row=record();row['identity']=deepcopy(identity)
    ref=source(science,[row]);m=json.loads((science/ref['manifest']['path']).read_bytes())
    with h5py.File(science/'records.h5','r+') as handle:
        del handle['metadata/identity_json'];handle['metadata/identity_json']=np.bytes_(encoded(identity))
    leaf=reference(science,'records.h5','qcl-negf-scientific-history-v5','application/x-hdf5')
    page=json.loads((science/'catalog.json').read_bytes());page['identity']=identity
    page['artifacts'][-1].update(leaf)
    pref=put(science,'catalog.json',page,page['schema']);m['history']['head']=pref;m['identity']=identity
    mref=put(science,'manifest.json',m,m['schema']);ref.update(identity=identity,manifest=mref)
    if checkpoint:
        ref,_=witness_history(science,scientific_plan=True,valid_full=True,identity_override=identity_override)
        m=json.loads((science/ref['manifest']['path']).read_bytes())
        planref=reference(root,'science/original/plan.json','qcl-negf-scientific-plan-v2')
    if canonical_plan is not None:planref=deepcopy(canonical_plan[1])
    transport=dict(selection=dict(calcjob_uuid='job3',execution_id=execution,attempt=3,plan_fingerprint='a'*64),frozen_plan=planref)
    original=dict(selection=dict(calcjob_uuid='job2',execution_id=execution,attempt=int(ref['identity']['attempt']),plan_fingerprint='a'*64),frozen_plan=planref)
    binding=dict(schema='qcl-negf.telemetry-source-binding.v1',contract_set='qcl-negf.results.v2',
        enclosing_transport=transport,source_transport=original,selected_state=ref,invocation=m['invocation'],relation='borrowed',
        canonical_plan=planref,producer_provenance=m['provenance'],plan_relation='identical_frozen_bytes')
    binding_ref=put(technical,source_binding_path,binding,binding['schema'])
    return dict(root=root if canonical_plan is None else Path(canonical_plan[0]),science=science,technical=technical,state=ref,transport=transport,source_transport=original,
        plan=planref,source_binding=binding_ref,producer_provenance=m['provenance'])


def telemetry_input(context, writer, *, observed_unix=None, binding_ref=None, phase_context_refs=(), resume_context_refs=(),bind_closed=True):
    """Explicit exact current publication chosen by test caller, never discovery."""
    from qcl_negf_results.performance_views import TelemetrySelection
    selected=TelemetrySelection(context['technical'],commit_ref=deepcopy(writer._previous_ref),
        source_binding_ref=context['source_binding'],expected_transport=context['transport'],
        source_transport=context['source_transport'],canonical_plan_root=context['root'],canonical_plan_ref=context['plan'],
        expected_producer_provenance=context['producer_provenance'],binding_ref=binding_ref,observed_unix=observed_unix,phase_context_refs=phase_context_refs,resume_context_refs=resume_context_refs)

    return retain_close_binding(context,selected) if bind_closed and writer.closed else selected

def diagnostic_phase(root,identity,*,durations=(2.,3.)):
    """Two original end Events; per-call correlation is literal, not aggregate-derived."""
    base=sealed_phase(root,identity);root=Path(root)
    health=deepcopy(base['segment']['health'])
    for key in ('events_seen','enqueued','sealed_events'):health[key]=str(len(durations))
    health['coalesced_events']='0';health['incomplete']=True;health['incomplete_reasons']=['context_missing']
    rows=[]
    for sequence,duration in enumerate(durations,1):
        metrics=[dict(name='duration_seconds',value=duration,unit='s',measurement_status='measured',reason=None)]
        for name,value,unit in (('cpu_seconds',8.,'s'),('gc_seconds',.25,'s'),('allocated_bytes',1024,'B')):
            metrics.append(dict(name=name,value=value if sequence==1 else None,unit=unit,
                measurement_status='measured' if sequence==1 else 'not_measured',reason=None if sequence==1 else 'counter_unavailable'))
        rows.append(dict(kind='event',sequence=str(sequence),core_sequence=None,action='phase_end',stage='scba',label='dyson',status='ok',
            iteration=None,total=None,outer_iteration=None,inner_iteration=None,span_id=None,parent_span_id=None,context_status='incomplete',metrics=metrics))
    segment=dict(base['segment'],mode='diagnostic',rows=rows,last_sequence=str(len(rows)),observed_events=str(len(rows)),coalesced_events='0',health=health)
    sr=put(root,'diagnostic-segment.json',segment,segment['schema'])
    publication=dict(base['publication'],segment=sr,last_sequence=str(len(rows)),rows=str(len(rows)),health=health)
    pr=put(root,'diagnostic-publication.json',publication,publication['schema'])
    current=dict(base['current'],publication=pr,health=health)
    put(root,'phase-current.json',current,current['schema'])
    return dict(segment=segment,segment_ref=sr,publication=publication,publication_ref=pr,current=current)


def outward_series(state,phase,*,artifact_root='science',fingerprint='a'*64):
    """Closed literal outward checkpoint projection, not process/science acceptance."""
    identity=state['identity'];point=identity['point_id'];execution=identity['execution_id']
    native=dict(schema='qcl-negf.native-point-source.v1',kind='checkpoint',artifact_root=artifact_root,
        relation='current_point_attempt',selected_state=deepcopy(state),diagnostics=None,observations=None,reason='manufactured_terminal_checkpoint')
    row=dict(id=point,execution_id=execution,attempt=int(identity['attempt']),
        coordinates=dict(temperature_K=200.,voltage_per_period_V=.05,branch='forward',order=1),
        initialization=dict(kind='cold_start',source_point_id=None,checkpoint=None,fallback_reason=None),
        status='failed',quality='invalid',converged=False,warnings=[],observables={},
        data=dict(native_source=native,phase_source=phase,optical_attachment=None),
        postprocessing=dict(optical=dict(status='not_requested',error=None)))
    return dict(schema='qcl-negf-series-result-v4',contract_set='qcl-negf.results.v2',plan_fingerprint=fingerprint,
        plan_scientific_fingerprint='2'*64,root_definition_id='literal',name='literal',status='failed',model_version='manufactured',inclusions=[],
        executions=[dict(id=execution,definition_id='literal',variant_id='variant',method_id='method',purpose='diagnostic',label='',
            operation='stationary',point_ids=[point],repetition=1)],expected_point_count=1,available_point_count=1,points=[row])


def diagnostic_parts(root,durations):
    from qcl_negf_results.capture import PhaseCaptureSource,PhaseCaptureLimits
    from qcl_negf_results.state import _IOBudget
    from result_v2_fixture import IDENTITY
    literal=diagnostic_phase(root,IDENTITY,durations=durations);answer=[];cursor=None
    with PhaseCaptureSource(root,descriptor=dict(status='available',session_id='native',current_path='phase-current.json',reason=None),
            expected_identity=IDENTITY,route_epoch=0,limits=PhaseCaptureLimits(maximum_rows=1)) as reader:
        while (batch:=reader.harvest(cursor,budget=_IOBudget(1024*1024))) is not None:
            answer.append(batch);cursor=batch.cursor
    return answer


def consumer_writer(root,**options):
    from qcl_negf_results.telemetry import TelemetryWriter
    c=technical_source(root)
    return c,TelemetryWriter(c['technical'],{}, {},transport=c['transport'],**options)


def consumer_export(context,writer,destination,*,observe=False,**options):
    from qcl_negf_results.state import StateReader
    from qcl_negf_results.export import export_snapshot
    observed=writer._previous['published_unix']+1 if observe else None
    with StateReader(context['science'],state_ref=context['state'],expected_identity=context['state']['identity']) as reader:
        return export_snapshot(reader,destination,profile='science',byte_budget=32*1024*1024,reserve_bytes=0,
            telemetry=telemetry_input(context,writer,observed_unix=observed),**options)


def retain_close_binding(context,selection):
    """Manufacture the real public close binder as explicit fixture preparation."""
    import hashlib
    from qcl_negf_results.capture import make_binding
    from qcl_negf_results.state import StateReader
    from qcl_negf_results.performance_views import TelemetrySelection
    if selection.binding_ref is not None:return selection
    name='bindings/'+hashlib.sha256(encoded(dict(commit=selection.commit_ref,state=context['state'],source=selection.source_binding_ref))).hexdigest()+'.json'
    with StateReader(context['science'],state_ref=context['state'],expected_identity=context['state']['identity']) as reader:
        make_binding(reader,telemetry=selection,destination=selection.artifact_root/name,
            capture_status=dict(terminal_tail='unknown',scalar_live_route='direct',scalar_live_reason=None,performance_status='available',reason=None))
    values=selection.document();values['binding_ref']=reference(selection.artifact_root,name,'qcl-negf.telemetry-binding.v1')
    return TelemetrySelection(selection.artifact_root,canonical_plan_root=selection.canonical_plan_root,**values)
