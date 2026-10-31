"""Independently verified stored final observations, without scientific derive."""
from __future__ import annotations

from dataclasses import dataclass, field, replace
import hashlib
import math
import re
from typing import Mapping

from qcl_negf_contracts import schema_validator
from qcl_negf_contracts.messages import ContractError, scientific_plan
from qcl_negf_contracts.results import validate_final_observations, validate_nested_sweep_plan
from .commits import decode_json, safe_path
from .model import canonical_bytes, validate_model
from .state import (StateReader, ResultReadLimits, Document, ArrayDescription, ArrayBlock, _ref,
                    _metadata_weight, _schema_check, _context_copy, _context_contract,
                    _context_before, _context_hold, _context_decoding, _context_arena,
                    _MetadataArena, _IOBudget, _context_json_size, _context_workspace, _context_weight,
                    _context_call, _context_validation_profile)
from .final_diagnostics import FinalDiagnosticsReader, FinalDiagnosticsDescription
from .native import final_observations_metadata, validate_final_observations_array


@dataclass(frozen=True)
class FinalObservationsDescription:
    source: Document
    invocation: Document
    artifact: Document
    checkpoint: Document
    diagnostics: Document
    declaration: Document
    plan: Document
    model: Document
    original_evidence: Document
    integrity: Document
    io_counters: Mapping[str, int | None]
    parent_manifest: Document | None = field(default=None, kw_only=True)
    parent_checkpoint: Document | None = field(default=None, kw_only=True)


@dataclass(frozen=True)
class CheckpointInputsDescription:
    """Detached original inputs of one selected checkpoint, without a verdict."""
    source: Document
    invocation: Document
    checkpoint: Document
    plan_ref: Document
    model_ref: Document
    plan_raw: bytes
    model_raw: bytes
    plan: Document
    model: Document
    integrity: Document
    io_counters: Mapping[str, int | None]


def _read_original_input(reader, reference, role, remaining):
    from ._capture_io import regular, stamp
    from .state import _BudgetStream
    owner=next((row for row in reader._manifest['artifacts'] if row['path']==reference['path']),None)
    if owner is None or _ref(owner)!=reference or owner['role']!=role or owner['profile']!='science' or owner['dependencies']:
        raise ContractError('original input FileRef/dependencies differ','corrupt_result')
    size=int(reference['bytes'])
    if size>remaining:raise ContractError('original input exceeds metadata allowance','budget_exceeded')
    path=safe_path(reader.root,reference['path'])
    with regular(path,size=size) as raw:
        before=stamp(raw,include_change=True)
        stream=_BudgetStream(raw,reader._budget)
        phase,reader._budget.phase=reader._budget.phase,'metadata'
        _context_before('original input read',extra=128+6*size)
        try:payload=stream.read(size)
        finally:reader._budget.phase=phase
        current=path.stat(follow_symlinks=False)
        bound=(current.st_dev,current.st_ino,current.st_size,current.st_mtime_ns,current.st_ctime_ns)
        if (reader._root_identity()!=reader._root_stamp or stamp(raw,include_change=True)!=before or bound!=before
                or hashlib.sha256(payload).hexdigest()!=reference['sha256']):
            raise ContractError('original input immutable binding changed','source_changed')
        reader._note_source_binding(reference['path'], before)
    return payload


def describe_checkpoint_inputs(reader: StateReader, *, plan_ref: Document,
                               model_ref: Document) -> CheckpointInputsDescription:
    """Read explicit original FileRefs using one selected reader operation.

    Counters describe this operation, not the reader's lifetime. The caller owns
    aggregate credit across construction/repeated calls and actual creator proof.
    No final request, restore, full tensor or scientific acceptance is assessed.
    """
    live=[]
    with _context_hold(live):
        if not isinstance(reader, StateReader):
            raise TypeError('explicit selected StateReader required')
        with reader._consumer_operation(verify_source_stability=True):
            maximum = reader.limits.maximum_metadata_bytes
            common = schema_validator('result-common.schema.json')
            file_schema = common.evolve(schema={'$ref': '#/$defs/FileRef', '$defs': common.schema['$defs']})
            for reference in (plan_ref, model_ref):
                _schema_check(file_schema, reference)
            # Reserve caller refs before copying, then the complete borrowed context.
            if 2*_metadata_weight((plan_ref, model_ref)) + reader._source_stability_bytes() > maximum:
                raise ContractError('checkpoint input references exceed metadata allowance', 'budget_exceeded')
            plan_ref, model_ref = _context_copy(dict(plan_ref)), _context_copy(dict(model_ref))
            formats = {('qcl-negf.nested-sweep-plan.v1', 'text/plain'): 'nested_sweep',
                       ('qcl-negf-scientific-plan-v2', 'application/json'): 'scientific_plan_v2'}
            kind = formats.get((plan_ref['schema'], plan_ref['media_type']))
            if kind is None or (model_ref['schema'], model_ref['media_type']) != (
                    'qcl-negf-resolved-configuration-v3', 'application/json'):
                raise ContractError('unsupported original checkpoint input format', 'incompatible_contract')
            if plan_ref['sha256'] != reader._manifest['identity']['plan_sha256']:
                raise ContractError('original checkpoint plan identity digest differs', 'corrupt_result')
            checkpoint = reader._load_checkpoint()
            contexts = (reader._manifest, checkpoint, plan_ref, model_ref)
            retained = []
            live.extend((contexts,retained))

            def allowance(reserve=0):
                available = maximum - _metadata_weight((contexts, retained)) - reader._source_stability_bytes() - reserve
                if available < 1:
                    raise ContractError('aggregate checkpoint inputs exceed metadata allowance', 'budget_exceeded')
                return available

            def original_json(raw, name):
                # Bound ordinary JSON object/string workspace before invoking the
                # existing decoder. Punctuation inside strings only overestimates.
                workspace = 512*(1+sum(raw.count(c) for c in (b'{', b'}', b'[', b']', b',', b':'))) + 4*len(raw)
                allowance(workspace)
                try:
                    _context_decoding(raw,name)
                    return decode_json(raw, name)
                except RecursionError as error:
                    raise ContractError('original checkpoint JSON depth exceeds parser allowance', 'budget_exceeded') from error

            model_raw = _read_original_input(reader, model_ref, 'model', allowance())
            retained.append(model_raw)
            model = original_json(model_raw, 'original checkpoint model')
            retained.append(model)
            envelope = schema_validator('final-observations.schema.json')
            _schema_check(envelope.evolve(schema={'$ref': '#/$defs/ModelEnvelope', '$defs': envelope.schema['$defs']}), model)
            allowance(16*len(model_raw))  # Existing canonical encoder's temporary bytes.
            _context_contract('validate_model',model)
            validate_model(model)
            if model['configuration_hash'] != reader._manifest['identity']['model_sha256']:
                raise ContractError('original checkpoint model identity digest differs', 'corrupt_result')
            plan_raw = _read_original_input(reader, plan_ref, 'plan', allowance())
            retained.append(plan_raw)
            if kind == 'nested_sweep':
                plan = _portable_plan(plan_raw, allowance())
            else:
                plan = original_json(plan_raw, 'original checkpoint scientific plan')
                _context_contract('scientific_plan',plan)
                scientific_plan(plan)
                _schema_check(schema_validator('scientific-plan.schema.json'), plan)
            retained.append(plan)
            allowance()
            # Only the small public metadata needs copying; decoded inputs and raw
            # bytes already belong exclusively to this operation, never the reader.
            output_metadata = (reader._state_ref, reader._manifest['invocation'],
                               reader._manifest['checkpoint'], plan_ref, model_ref)
            allowance(2*_metadata_weight(output_metadata) + 4096)
            source, invocation, checkpoint_ref, plan_ref, model_ref = _context_copy(output_metadata)
            live.extend((source,invocation,checkpoint_ref,plan_ref,model_ref))
            with _context_workspace('input integrity copy arguments',
                    8192+sum(_context_weight(v) for v in (source,checkpoint_ref,plan_ref,model_ref)) if _context_arena.get() is not None else 0):
                integrity = dict(scope='selected_checkpoint_original_inputs', source_state=_context_copy(source),
                    checkpoint=_context_copy(checkpoint_ref),
                    input_refs=dict(plan=_context_copy(plan_ref), model=_context_copy(model_ref)),
                    checks=['checkpoint_declaration_history_lineage', 'manifest_input_rows', 'original_file_bytes',
                            'plan_raw_identity_sha256', 'model_native_profile', 'model_canonical_identity_sha256',
                            'consumed_source_bindings_at_return'],
                    not_assessed=['full_core_payloads', 'resolver_provenance', 'actual_creator', 'core_restore', 'scientific_acceptance'])
            allowance(_metadata_weight((output_metadata, source, invocation, checkpoint_ref, plan_ref, model_ref, integrity)))
            _context_before('input description',extra=8192,roots=(source,invocation,checkpoint_ref,plan_ref,model_ref,integrity))
            description = CheckpointInputsDescription(source, invocation, checkpoint_ref, plan_ref, model_ref,
                plan_raw, model_raw, plan, model, integrity, reader.io_counters)
        # The source-stability context completes its actual consumed-binding check
        # before a description (including one built from a cached CP) can escape.
        return description


def _portable_plan(raw: bytes, maximum_bytes: int) -> dict:
    """Two-pass bounded portable YAML admission, without arbitrary constructors."""
    import yaml
    from yaml.events import (AliasEvent, ScalarEvent, SequenceStartEvent, SequenceEndEvent,
                             MappingStartEvent, MappingEndEvent, DocumentStartEvent)
    def fail(message, code='corrupt_result'):
        raise ContractError('original nested plan: '+message, code)
    with _context_workspace('YAML parser lifetime',2048*(len(raw)+1)+16384):
        try:
            _context_before('YAML UTF8 decode',extra=4*len(raw)+4096,roots=(raw,))
            text=raw.decode('utf-8', errors='strict')
            depth=documents=cost=0
            explicit={}
            tags={"tag:yaml.org,2002:"+name for name in ("str","int","float","bool","null","seq","map")}
            for event in yaml.parse(text, Loader=yaml.BaseLoader):
                if isinstance(event,DocumentStartEvent):
                    documents+=1
                    if documents>1 or event.tags:fail('one document without tag directives required')
                if isinstance(event,AliasEvent) or getattr(event,'anchor',None) is not None:
                    fail('aliases and anchors forbidden')
                if getattr(event,'tag',None) is not None:
                    if event.tag not in tags:fail('nonportable tags forbidden')
                    if (isinstance(event,MappingStartEvent) and event.tag!='tag:yaml.org,2002:map'
                            or isinstance(event,SequenceStartEvent) and event.tag!='tag:yaml.org,2002:seq'
                            or isinstance(event,ScalarEvent) and event.tag in ('tag:yaml.org,2002:map','tag:yaml.org,2002:seq')):
                        fail('portable tag differs from node kind')
                    explicit[event.start_mark.index]=event.tag
                if isinstance(event,(ScalarEvent,SequenceStartEvent,MappingStartEvent)):
                    cost+=512+8
                    if isinstance(event,ScalarEvent):cost+=4*len(event.value.encode('utf-8'))
                    else:
                        depth+=1
                        if depth>64:fail('depth exceeds limit','budget_exceeded')
                    if cost+3*len(raw)>maximum_bytes:fail('event tree exceeds metadata allowance','budget_exceeded')
                elif isinstance(event,(SequenceEndEvent,MappingEndEvent)):depth-=1
            if documents!=1:fail('one document required')
            # Compose only after the complete event admission. BaseLoader retains
            # scalar spelling and style; timestamps/custom Python tags never execute.
            _context_before('YAML compose/convert',extra=3*cost+8*len(raw)+4096,roots=(raw,text))
            node=yaml.compose(text,Loader=yaml.BaseLoader)
            def scalar(n):
                value=n.value
                tag=explicit.get(n.start_mark.index)
                if tag=='tag:yaml.org,2002:str' or tag is None and n.style is not None:return value
                if tag is None:
                    # Exact YAML.jl0.4.17 resolver order, rather than PyYAML's
                    # YAML1.1 yes/on booleans or timestamp object constructors.
                    if value in ('true','True','TRUE','false','False','FALSE'):tag='bool'
                    elif re.fullmatch(r'[+-]?0b[01_]+|[+-]?[0-9]+|0o[0-7]+|0x[0-9a-fA-F]+',value):tag='int'
                    elif re.fullmatch(r'[+-]?(?:\.[0-9]+|[0-9]+(?:\.[0-9]*)?)(?:[eE][+-]?[0-9]+)?|[+-]?\.(?:inf|Inf|INF)|\.(?:nan|NaN|NAN)',value):tag='float'
                    elif value in ('~','null','Null','NULL',''):tag='null'
                    elif re.match(r'[0-9]{4}-[0-9]{1,2}-[0-9]{1,2}(?:$|[Tt \t])',value):fail('nonportable timestamp scalar')
                    else:return value
                else:tag=tag.removeprefix('tag:yaml.org,2002:')
                if tag=='null':return None
                if tag=='bool':
                    if value.lower() not in ('true','false'):fail('invalid Boolean scalar')
                    return value.lower()=='true'
                if tag=='int':
                    number=value.lower().replace('_','')
                    # YAML.jl uses native Int64, including its unsigned-leading-zero
                    # octal constructor branch. Root's exact Julia1.13/YAML0.4.17
                    # scalar probe rejects binary prefixes in that constructor.
                    if re.match(r'[+-]?0b',number):fail('binary integer spelling rejected by owning YAML constructor')
                    result=int(number[2:],16) if number.startswith('0x') else int(number,8) if len(number)>1 and number[0]=='0' else int(number,10)
                    if not -2**63<=result<2**63:fail('integer exceeds actual Int64 domain')
                    return result
                if tag=='float':
                    if value.lower().lstrip('+-') in ('.inf','.nan'):fail('nonfinite scalar forbidden')
                    result=float(value.lower().replace('_',''))
                    if not math.isfinite(result):fail('nonfinite scalar forbidden')
                    return result
                fail('scalar tag differs from node kind')
            def convert(n):
                if isinstance(n,yaml.ScalarNode):return scalar(n)
                if isinstance(n,yaml.SequenceNode):return [convert(v) for v in n.value]
                result={}
                for key,val in n.value:
                    k=scalar(key) if isinstance(key,yaml.ScalarNode) else None
                    if (not isinstance(k,str) or k in result or k=='<<' and key.style is None
                            and explicit.get(key.start_mark.index)!='tag:yaml.org,2002:str'):
                        fail('mapping key/merge/duplicate forbidden')
                    result[k]=convert(val)
                return result
            value=convert(node)
            _context_contract('validate_nested_sweep_plan',value)
            validate_nested_sweep_plan(value)
            level=value['root']
            while level['kind']=='level':
                for axis in level['axes']:
                    _context_before('YAML canonical axis keys',extra=16*sum(_context_json_size(v) for v in axis['values']),roots=(value,))
                    keys=[canonical_bytes(v) for v in axis['values']]
                    if len(set(keys))!=len(keys):fail('canonical duplicate axis value')
                level=level['child']
            if _metadata_weight(value)>maximum_bytes:fail('decoded tree exceeds metadata allowance','budget_exceeded')
            return value
        except ContractError:
            raise
        except (UnicodeError,yaml.YAMLError,ValueError,OverflowError,RecursionError) as error:
            raise ContractError('invalid original nested plan: '+str(error),'corrupt_result') from error


class FinalObservationsReader:
    """Borrow one selected StateReader and its owner/budget/lineage proofs."""
    def __init__(self, reader: StateReader, *, artifact_ref: Document):
        if not isinstance(reader,StateReader):raise TypeError('explicit selected StateReader required')
        if not isinstance(artifact_ref,Mapping) or set(artifact_ref)!={'path','bytes','sha256','media_type','schema'}:
            raise ContractError('exact observations FileRef required','corrupt_result')
        self._reader,self._reference=reader,_context_copy(dict(artifact_ref))
        self._source,self._closed=_context_copy(reader.source.state),False

    def __enter__(self):return self

    def __exit__(self,*args):self.close()

    def close(self):self._closed=True

    def _input(self, reference, role, remaining):
        return _read_original_input(self._reader, reference, role, remaining)

    def _load(self):
        live=[]
        with _context_hold(live,self._source,self._reference):
            if self._closed:raise ValueError('final observations reader is closed')
            reader=self._reader
            if reader.source.state!=self._source:raise ContractError('selected observations source changed','source_changed')
            owner=next((row for row in reader._manifest['artifacts'] if row['path']==self._reference['path']),None)
            if (owner is None or _ref(owner)!=self._reference or owner['role']!='physics.analysis'
                    or owner['schema']!='qcl-negf-physics-analysis-v5' or owner['profile']!='science'):
                raise ContractError('selected exact observations FileRef absent or differs','corrupt_result')
            checkpoint=reader._load_checkpoint()
            handle=reader._open_hdf(owner,identity=reader._manifest['identity'],invocation=reader._manifest['invocation'],representation='final_observations')
            value=final_observations_metadata(handle,identity=reader._manifest['identity'],invocation=reader._manifest['invocation'],maximum_bytes=reader.limits.maximum_metadata_bytes)['declaration']
            live.append(value)
            _schema_check(schema_validator('final-observations.schema.json'),value)
            diagnostics=FinalDiagnosticsReader(reader,artifact_ref=value['diagnostics']).describe().declaration
            live.append(diagnostics)
            parent_manifest=parent_checkpoint=None
            if value['source_mode']=='completed_restore':
                anchor=reader._manifest['lineage']['parent_state']
                if anchor is None:raise ContractError('completed observations require immediate parent','corrupt_result')
                with reader._parent_reader(anchor) as parent:
                    parent_checkpoint=parent._load_checkpoint();parent_manifest=parent._manifest
                    live.extend((parent_checkpoint,parent_manifest))
            contexts=(reader._manifest,checkpoint,value,diagnostics,parent_manifest,parent_checkpoint)
            maximum=reader.limits.maximum_metadata_bytes
            remaining=maximum-_metadata_weight(contexts)
            if remaining<1:raise ContractError('aggregate observations context exceeds metadata limit','budget_exceeded')
            model_raw=self._input(value['request']['model'],'model',remaining)
            live.append(model_raw);_context_decoding(model_raw,'original model')
            model=decode_json(model_raw,'original model');live.append(model)
            _context_contract('validate_model',model)
            validate_model(model)
            remaining=maximum-_metadata_weight((contexts,model))-len(model_raw)
            plan_raw=self._input(value['request']['plan'],'plan',remaining)
            remaining-=len(plan_raw);live.append(plan_raw)
            _context_decoding(plan_raw,'original scientific plan')
            plan=(_portable_plan(plan_raw,remaining) if value['request']['kind']=='nested_sweep'
                  else decode_json(plan_raw,'original scientific plan'))
            live.append(plan)
            if _metadata_weight((contexts,model,plan))+len(model_raw)+len(plan_raw)>maximum:
                raise ContractError('aggregate observations input context exceeds metadata limit','budget_exceeded')
            _context_contract('validate_final_observations',value,checkpoint=checkpoint,manifest=reader._manifest,artifact_ref=self._reference,diagnostics=diagnostics,plan=plan,model=model,parent_manifest=parent_manifest,parent_checkpoint=parent_checkpoint)
            value=validate_final_observations(value,checkpoint=checkpoint,manifest=reader._manifest,artifact_ref=self._reference,
                diagnostics=diagnostics,plan=plan,model=model,parent_manifest=parent_manifest,parent_checkpoint=parent_checkpoint)
            live[0]=value
            for key in ('convergence','physics'):
                if reader._manifest['status'][key]!={'state':'recorded','reason':None,'evidence':self._reference}:
                    raise ContractError('current final evidence must bind selected observations','corrupt_result')
            # A one-owner cache may have evicted O1 while D1 was authenticated.
            handle=reader._open_hdf(owner,identity=reader._manifest['identity'],invocation=reader._manifest['invocation'],representation='final_observations')
            for d in self._arrays(value).values():
                if d is not None:validate_final_observations_array(handle,d)
            return value,handle,plan,model,checkpoint,parent_manifest,parent_checkpoint

    @staticmethod
    def _arrays(value):
        from .native import _observation_profiles
        return {name:(value['context'] if name in value['context'] else value['views'])[name] for name in _observation_profiles()}

    def describe(self) -> FinalObservationsDescription:
        with self._reader._consumer_operation():
            value,_,plan,model,checkpoint,parent_manifest,parent_checkpoint=self._load()
            roots=(self._reader._manifest,checkpoint,value,plan,model,parent_manifest,parent_checkpoint,
                self._source,self._reference)
            copies=(self._source,value['invocation'],self._reference,value['checkpoint'],value['diagnostics'],
                checkpoint['state']['fields']['final_evidence'])
            if parent_manifest is not None:copies+=(parent_manifest,parent_checkpoint)
            with _context_hold(*roots):
                _context_before('observations description',extra=8192,roots=roots,copies=copies)
                # Arena hooks are no-ops for standalone readers. Admit originals,
                # all simultaneous argument copies and headers before either new
                # parent deepcopy using the existing per-operation metadata model.
                if parent_manifest is not None and (8192+_metadata_weight(roots)+
                        sum(_metadata_weight(v) for v in copies)+self._reader._source_stability_bytes()
                        >self._reader.limits.maximum_metadata_bytes):
                    raise ContractError('observations description exceeds metadata allowance','budget_exceeded')
                with _context_workspace('observations copy arguments',
                        8192+sum(_context_weight(v) for v in copies) if _context_arena.get() is not None else 0):
                    return FinalObservationsDescription(_context_copy(self._source),_context_copy(value['invocation']),
                        _context_copy(self._reference),_context_copy(value['checkpoint']),_context_copy(value['diagnostics']),
                        value,plan,model,_context_copy(checkpoint['state']['fields']['final_evidence']),
                        dict(manifest_verified=True,observations_file_verified=True,diagnostics_file_verified=True,
                            original_inputs_verified=True,checkpoint_declaration_verified=True,native_metadata_verified=True,
                            selected_array_metadata_verified=True,full_core_payloads_scope='not_assessed_by_this_operation',scope='selected_bytes'),self._reader.io_counters,
                        parent_manifest=None if parent_manifest is None else _context_copy(parent_manifest),
                        parent_checkpoint=None if parent_checkpoint is None else _context_copy(parent_checkpoint))

    def _array(self,name):
        value,handle,_,_,_,_,_=self._load();arrays=self._arrays(value)
        if name not in arrays:raise ValueError('unknown stored observation array')
        descriptor=arrays[name]
        if descriptor is None:
            issue=next(v for v in value['issues'] if v['target']==name)
            raise ContractError(name+': '+issue['status']+': '+issue['reason']+': '+issue['detail'],'capability_unavailable')
        return descriptor,validate_final_observations_array(handle,descriptor)

    def describe_array(self,name: str) -> ArrayDescription:
        with self._reader._consumer_operation():
            descriptor,dataset=self._array(name)
            return ArrayDescription(_context_copy(descriptor),dataset.shape,dataset.shape,descriptor['eltype'],dataset.dtype.itemsize,'F',_context_copy(self._source))

    def read_array(self,name: str,selection: tuple[int|slice,...],*,maximum_bytes: int=1024*1024) -> ArrayBlock:
        with self._reader._consumer_operation():
            descriptor,dataset=self._array(name)
            return self._reader._read_selected_array(descriptor,dataset,selection,maximum_bytes=maximum_bytes)


def describe_final_observations(reader: StateReader,*,artifact_ref: Document) -> FinalObservationsDescription:
    with FinalObservationsReader(reader,artifact_ref=artifact_ref) as view:return view.describe()


def describe_native_metadata_context(artifact_root, *, state_ref: Document,
        expected_identity: Document, plan_ref: Document, model_ref: Document,
        diagnostics_ref: Document | None, observations_ref: Document | None,
        limits: ResultReadLimits, maximum_context_bytes: int = 128*1024*1024,
        maximum_retained_bytes: int = 4*1024*1024) -> tuple[
            dict, dict, CheckpointInputsDescription,
            FinalDiagnosticsDescription | None, FinalObservationsDescription | None]:
    """One bounded original metadata closure, without native/science authority.

    ``maximum_context_bytes`` is aggregate explicit metadata accounting (at most
    128 MiB), not a Python/HDF allocator or process RSS guarantee. All operations,
    initialization, nested validation, stability and close share one physical
    stream-delivery allowance. On any failure no partial tuple is returned.
    """
    with _context_call():
        from .state import _context_weight
        if type(maximum_retained_bytes) is not int or not 0<maximum_retained_bytes<=128*1024*1024:
            raise ValueError('maximum_retained_bytes must be an integer in 1..134217728')
        if not isinstance(limits,ResultReadLimits):raise TypeError('ResultReadLimits required')
        arena=_MetadataArena(maximum_context_bytes)
        if (diagnostics_ref is None)!=(observations_ref is None):
            raise ValueError('both final FileRefs or neither required')
        # Exact builtin metadata prevents user mappings/objects from allocating or
        # calling back during admission. Full schemas still use the existing owner.
        for value in (state_ref,expected_identity,plan_ref,model_ref,diagnostics_ref,observations_ref):
            if value is not None and type(value) is not dict:raise TypeError('builtin metadata dictionaries required')
            def builtin(item,active=None,depth=0):
                active=set() if active is None else active
                if depth>64:raise ContractError('argument metadata depth exceeds allowance','budget_exceeded')
                if item is None or type(item) in (str,int,float,bool):return
                if type(item) not in (dict,list):raise TypeError('builtin JSON metadata required')
                if id(item) in active:raise ContractError('argument metadata cycle','invalid_input')
                active.add(id(item))
                try:
                    for key,child in (item.items() if type(item) is dict else enumerate(item)):
                        if type(item) is dict and type(key) is not str:raise TypeError('JSON string keys required')
                        builtin(child,active,depth+1)
                finally:active.remove(id(item))
            builtin(value)
            _context_weight(value)
        arena.before('whole-call original arguments',roots=(state_ref,expected_identity,plan_ref,model_ref,diagnostics_ref,observations_ref))
        token=_context_arena.set(arena);reader=StateReader.__new__(StateReader)
        live=[];views=[];primary=None
        arguments=(state_ref,expected_identity,plan_ref,model_ref,diagnostics_ref,observations_ref)
        try:
            with _context_hold(live,arguments):
                # The same budget is supplied BEFORE manifest initialization. It is
                # never replaced by later public/nested _operation calls.
                profile=_context_validation_profile(arena)
                if schema_validator is not __import__('qcl_negf_contracts').schema_validator:
                    raise ContractError('unknown or changed validation profile','incompatible_contract')
                budget=_IOBudget(limits.maximum_io_bytes)
                reader._initialize(artifact_root,state_ref=state_ref,expected_identity=expected_identity,limits=limits,budget=budget)
                with reader._consumer_operation(verify_source_stability=True):
                    manifest=reader._manifest;checkpoint=reader._load_checkpoint()
                    inputs=describe_checkpoint_inputs(reader,plan_ref=plan_ref,model_ref=model_ref)
                    live.append(inputs)
                    diagnostic=observation=None
                    if diagnostics_ref is not None:
                        final=FinalDiagnosticsReader(reader,artifact_ref=diagnostics_ref);views.append(final)
                        diagnostic=final.describe();live.append(diagnostic)
                        views.remove(final);final.close()
                        final=FinalObservationsReader(reader,artifact_ref=observations_ref);views.append(final)
                        observation=final.describe();live.append(observation)
                        views.remove(final);final.close()
                    # These are private original trees, with no caller aliases. Move
                    # ownership, rather than making another complete context copy.
                    _context_before('detached return tuple',extra=16384,roots=(manifest,checkpoint,inputs,diagnostic,observation))
                    result=(manifest,checkpoint,inputs,diagnostic,observation)
                    live.append(result)
                reader.close()  # All delivery during close still uses `budget`.
                _context_validation_profile(arena,profile)
                if schema_validator is not __import__('qcl_negf_contracts').schema_validator:
                    raise ContractError('unknown or changed validation profile','incompatible_contract')
                counters=dict(reader.io_counters)
                counters.update(maximum_context_bytes=arena.maximum,metadata_high_water_bytes=arena.high_water)
                _context_before('whole-call counter headers',extra=16384,roots=(result,counters))
                counters['metadata_high_water_bytes']=arena.high_water
                inputs=replace(inputs,io_counters=dict(counters))
                if diagnostic is not None:diagnostic=replace(diagnostic,io_counters=dict(counters))
                if observation is not None:observation=replace(observation,io_counters=dict(counters))
                result=(manifest,checkpoint,inputs,diagnostic,observation)
                _context_before('final retained tuple',roots=(result,))
                for description in (inputs,diagnostic,observation):
                    if description is not None:description.io_counters['metadata_high_water_bytes']=arena.high_water
                if _context_weight(result)>maximum_retained_bytes:
                    raise ContractError('retained metadata context exceeds allowance','budget_exceeded')
                # Detach moved JSON from the now-closed private owner before return.
                for owner in arena.readers:
                    for name in ('_state_ref','_manifest','_checkpoint','_parent_anchor_cache'):
                        if hasattr(owner,name):setattr(owner,name,None)
                    for name in ('_pages','_index','_known_states','_catalog_by_hash','_catalog_by_leaf','_page_proofs','_declarations','_verified_refs','_metadata_bindings'):
                        value=getattr(owner,name,None)
                        if value is not None:value.clear()
                return result
        except BaseException as error:
            primary=error
            raise
        finally:
            # Never replace the primary error/cancellation with secondary cleanup.
            first=None
            for view in reversed(views):
                try:view.close()
                except BaseException as error:
                    if first is None:first=error
            for owner in reversed(arena.readers):
                if not getattr(owner,'_closed',True):
                    try:owner.close()
                    except BaseException as error:
                        if first is None:first=error
            _context_arena.reset(token)
            if first is not None:
                if primary is None:raise first
                primary.add_note('metadata context cleanup also raised '+type(first).__name__)
