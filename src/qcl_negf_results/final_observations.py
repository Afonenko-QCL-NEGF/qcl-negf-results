"""Independently verified stored final observations, without scientific derive."""
from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import math
import re
from typing import Mapping

from qcl_negf_contracts import schema_validator
from qcl_negf_contracts.messages import ContractError
from qcl_negf_contracts.results import validate_final_observations, validate_nested_sweep_plan
from .commits import decode_json, safe_path
from .model import canonical_bytes, validate_model
from .state import (StateReader, Document, ArrayDescription, ArrayBlock, _ref,
                    _metadata_weight, _schema_check)
from .final_diagnostics import FinalDiagnosticsReader
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


def _portable_plan(raw: bytes, maximum_bytes: int) -> dict:
    """Two-pass bounded portable YAML admission, without arbitrary constructors."""
    import yaml
    from yaml.events import (AliasEvent, ScalarEvent, SequenceStartEvent, SequenceEndEvent,
                             MappingStartEvent, MappingEndEvent, DocumentStartEvent)
    def fail(message, code='corrupt_result'):
        raise ContractError('original nested plan: '+message, code)
    try:
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
        validate_nested_sweep_plan(value)
        level=value['root']
        while level['kind']=='level':
            for axis in level['axes']:
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
        self._reader,self._reference=reader,copy.deepcopy(dict(artifact_ref))
        self._source,self._closed=copy.deepcopy(reader.source.state),False

    def __enter__(self):return self

    def __exit__(self,*args):self.close()

    def close(self):self._closed=True

    def _input(self, reference, role, remaining):
        from ._capture_io import regular, stamp
        from .state import _BudgetStream
        reader=self._reader
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
            try:payload=stream.read(size)
            finally:reader._budget.phase=phase
            current=path.stat(follow_symlinks=False)
            bound=(current.st_dev,current.st_ino,current.st_size,current.st_mtime_ns,current.st_ctime_ns)
            if (reader._root_identity()!=reader._root_stamp or stamp(raw,include_change=True)!=before or bound!=before
                    or hashlib.sha256(payload).hexdigest()!=reference['sha256']):
                raise ContractError('original input immutable binding changed','source_changed')
            reader._note_source_binding(reference['path'], before)
        return payload

    def _load(self):
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
        _schema_check(schema_validator('final-observations.schema.json'),value)
        diagnostics=FinalDiagnosticsReader(reader,artifact_ref=value['diagnostics']).describe().declaration
        parent_manifest=parent_checkpoint=None
        if value['source_mode']=='completed_restore':
            anchor=reader._manifest['lineage']['parent_state']
            if anchor is None:raise ContractError('completed observations require immediate parent','corrupt_result')
            with reader._parent_reader(anchor) as parent:
                parent_checkpoint=parent._load_checkpoint();parent_manifest=parent._manifest
        contexts=(reader._manifest,checkpoint,value,diagnostics,parent_manifest,parent_checkpoint)
        maximum=reader.limits.maximum_metadata_bytes
        remaining=maximum-_metadata_weight(contexts)
        if remaining<1:raise ContractError('aggregate observations context exceeds metadata limit','budget_exceeded')
        model_raw=self._input(value['request']['model'],'model',remaining)
        model=decode_json(model_raw,'original model')
        validate_model(model)
        remaining=maximum-_metadata_weight((contexts,model))-len(model_raw)
        plan_raw=self._input(value['request']['plan'],'plan',remaining)
        remaining-=len(plan_raw)
        plan=(_portable_plan(plan_raw,remaining) if value['request']['kind']=='nested_sweep'
              else decode_json(plan_raw,'original scientific plan'))
        if _metadata_weight((contexts,model,plan))+len(model_raw)+len(plan_raw)>maximum:
            raise ContractError('aggregate observations input context exceeds metadata limit','budget_exceeded')
        value=validate_final_observations(value,checkpoint=checkpoint,manifest=reader._manifest,artifact_ref=self._reference,
            diagnostics=diagnostics,plan=plan,model=model,parent_manifest=parent_manifest,parent_checkpoint=parent_checkpoint)
        for key in ('convergence','physics'):
            if reader._manifest['status'][key]!={'state':'recorded','reason':None,'evidence':self._reference}:
                raise ContractError('current final evidence must bind selected observations','corrupt_result')
        # A one-owner cache may have evicted O1 while D1 was authenticated.
        handle=reader._open_hdf(owner,identity=reader._manifest['identity'],invocation=reader._manifest['invocation'],representation='final_observations')
        for d in self._arrays(value).values():
            if d is not None:validate_final_observations_array(handle,d)
        return value,handle,plan,model,checkpoint

    @staticmethod
    def _arrays(value):
        from .native import _observation_profiles
        return {name:(value['context'] if name in value['context'] else value['views'])[name] for name in _observation_profiles()}

    def describe(self) -> FinalObservationsDescription:
        with self._reader._consumer_operation():
            value,_,plan,model,checkpoint=self._load()
            return FinalObservationsDescription(copy.deepcopy(self._source),copy.deepcopy(value['invocation']),
                copy.deepcopy(self._reference),copy.deepcopy(value['checkpoint']),copy.deepcopy(value['diagnostics']),
                value,plan,model,copy.deepcopy(checkpoint['state']['fields']['final_evidence']),
                dict(manifest_verified=True,observations_file_verified=True,diagnostics_file_verified=True,
                    original_inputs_verified=True,checkpoint_declaration_verified=True,native_metadata_verified=True,
                    selected_array_metadata_verified=True,full_core_payloads_scope='not_assessed_by_this_operation',scope='selected_bytes'),self._reader.io_counters)

    def _array(self,name):
        value,handle,_,_,_=self._load();arrays=self._arrays(value)
        if name not in arrays:raise ValueError('unknown stored observation array')
        descriptor=arrays[name]
        if descriptor is None:
            issue=next(v for v in value['issues'] if v['target']==name)
            raise ContractError(name+': '+issue['status']+': '+issue['reason']+': '+issue['detail'],'capability_unavailable')
        return descriptor,validate_final_observations_array(handle,descriptor)

    def describe_array(self,name: str) -> ArrayDescription:
        with self._reader._consumer_operation():
            descriptor,dataset=self._array(name)
            return ArrayDescription(copy.deepcopy(descriptor),dataset.shape,dataset.shape,descriptor['eltype'],dataset.dtype.itemsize,'F',copy.deepcopy(self._source))

    def read_array(self,name: str,selection: tuple[int|slice,...],*,maximum_bytes: int=1024*1024) -> ArrayBlock:
        with self._reader._consumer_operation():
            descriptor,dataset=self._array(name)
            return self._reader._read_selected_array(descriptor,dataset,selection,maximum_bytes=maximum_bytes)


def describe_final_observations(reader: StateReader,*,artifact_ref: Document) -> FinalObservationsDescription:
    with FinalObservationsReader(reader,artifact_ref=artifact_ref) as view:return view.describe()
