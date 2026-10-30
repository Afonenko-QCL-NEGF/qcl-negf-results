"""Exact small retained performance binding; never walk current/newest/catalogs."""

from __future__ import annotations

import json
import re
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from qcl_negf_contracts.artifacts import relative_path
from qcl_negf_contracts import schema_validator
from qcl_negf_contracts.telemetry import validate_telemetry_summary, validate_telemetry_source_binding, validate_telemetry_binding, validate_telemetry_commit
from qcl_negf_contracts.messages import encode

from ._capture_io import digest, fail, integer, regular, stamp, beneath
from .commits import json_bytes
from .state import StateReader, _metadata_weight, _schema_check
from .commits import decode_json

SELECTION = ("calcjob_uuid", "execution_id", "attempt", "plan_fingerprint")
SHA = re.compile("[0-9a-f]{64}")


@dataclass(frozen=True)
class PerformanceLimits:
    maximum_metadata_bytes: int = 1024 * 1024
    maximum_io_bytes: int = 4 * 1024 * 1024
    maximum_json_bytes: int = 2 * 1024 * 1024
    maximum_objects: int = 4096
    maximum_sessions: int = 16

    def __post_init__(self):
        if any(
            not integer(v, 1, self.__dataclass_fields__[k].default)
            for k, v in vars(self).items()
        ):
            raise ValueError(
                "performance limits may only tighten positive public bounds"
            )


def _same(actual, expected):
    return type(actual) is type(expected) and actual == expected


def _selection(value):
    if (
        not isinstance(value, Mapping)
        or set(value) != set(SELECTION)
        or not integer(value.get("attempt"), 1)
        or any(
            not isinstance(value.get(key), str) or not value[key]
            for key in SELECTION
            if key != "attempt"
        )
        or not SHA.fullmatch(value["plan_fingerprint"])
    ):
        fail("selection_required", "exact enclosing performance selection required")
    return dict(value)


def _bounded_objects(roots, maximum_objects):
    if not integer(maximum_objects, 1, 4096):
        raise ValueError("maximum_objects may only tighten the public bound")
    pending = list(roots)
    count = 0
    while pending:
        node = pending.pop()
        count += 1
        if count > maximum_objects:
            fail("budget_exceeded", "aggregate retained metadata object cap exceeded")
        if isinstance(node, dict):
            pending.extend(node.values())
        elif isinstance(node, list):
            pending.extend(node)


def _exact(value):
    if type(value) is int and abs(value) > 9007199254740991:
        return {"kind": "integer", "decimal": str(value)}
    if isinstance(value, dict):
        return {key: _exact(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_exact(item) for item in value]
    return value



@dataclass(frozen=True,init=False)
class TelemetrySelection:
    """Detached immutable input assertions/refs. Never an ownership certificate."""
    artifact_root: Path
    canonical_plan_root: Path
    _raw: bytes

    def __init__(self,artifact_root,*,commit_ref,source_binding_ref,expected_transport,source_transport,
                 canonical_plan_root,canonical_plan_ref,expected_producer_provenance,binding_ref=None,observed_unix=None,phase_context_refs=(),resume_context_refs=()):
        import math
        if observed_unix is not None and (type(observed_unix) not in (float,int) or not math.isfinite(observed_unix) or observed_unix<0):
            raise ValueError('finite nonnegative explicit observation time required')
        if not isinstance(phase_context_refs,(tuple,list)) or len(phase_context_refs)>128:
            raise ValueError('bounded phase context references required')
        if not isinstance(resume_context_refs,(tuple,list)) or len(resume_context_refs)>16:
            raise ValueError('bounded resume context references required')
        if any(not isinstance(row,dict) or set(row)!={'publication_ref','segment_ref','retained_publication_ref','retained_segment_ref'} for row in phase_context_refs):
            raise ValueError('exact original/retained phase reference quadruples required')
        if any(not isinstance(row,dict) or set(row)!={'commit_ref','summary_ref'} for row in resume_context_refs):
            raise ValueError('exact resume reference pairs required')
        value=dict(commit_ref=commit_ref,source_binding_ref=source_binding_ref,expected_transport=expected_transport,
            source_transport=source_transport,canonical_plan_ref=canonical_plan_ref,expected_producer_provenance=expected_producer_provenance,
            binding_ref=binding_ref,observed_unix=observed_unix,phase_context_refs=list(phase_context_refs),resume_context_refs=list(resume_context_refs))
        raw=json_bytes(value)
        if len(raw)>1024*1024 or _metadata_weight(value)>1024*1024:fail('budget_exceeded','telemetry selection metadata exceeds cap')
        object.__setattr__(self,'artifact_root',Path(artifact_root));object.__setattr__(self,'canonical_plan_root',Path(canonical_plan_root))
        object.__setattr__(self,'_raw',raw)

    def document(self):return decode_json(self._raw,'telemetry input assertions')
    @property
    def commit_ref(self):return self.document()['commit_ref']
    @property
    def source_binding_ref(self):return self.document()['source_binding_ref']
    @property
    def expected_transport(self):return self.document()['expected_transport']
    @property
    def source_transport(self):return self.document()['source_transport']
    @property
    def canonical_plan_ref(self):return self.document()['canonical_plan_ref']
    @property
    def expected_producer_provenance(self):return self.document()['expected_producer_provenance']
    @property
    def binding_ref(self):return self.document()['binding_ref']
    @property
    def observed_unix(self):return self.document()['observed_unix']


def _check_bindings(bindings):
    from qcl_negf_contracts.messages import ContractError
    for (root,path),before in bindings.items():
        try:
            if path=='':
                descriptor=os.open(root,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW|os.O_CLOEXEC)
                try:
                    info=os.fstat(descriptor)
                    if (info.st_dev,info.st_ino)!=before:fail('source_changed','technical artifact root replaced')
                finally:os.close(descriptor)
            else:
                with regular(beneath(root,path)) as stream:
                    if stamp(stream,include_change=True)!=before:fail('source_changed','technical source binding changed')
        except (OSError,ContractError) as error:
            fail('source_changed','technical source binding unavailable: '+str(error))


def _read_reference(reader,root,reference,limits,bindings):
    """One qualified metadata FD; only caller-selected root/whole FileRef."""
    common=schema_validator('result-common.schema.json')
    _schema_check(common.evolve(schema={'$ref':'#/$defs/FileRef','$defs':common.schema['$defs']}),reference)
    if reference['media_type']!='application/json':fail('unsupported_format','technical metadata requires JSON')
    size=int(reference['bytes']);key=(str(Path(root).absolute()),reference['path'])
    if size>limits.maximum_metadata_bytes:fail('budget_exceeded','technical metadata byte cap exceeded')
    rootkey=(key[0],'')
    directory=os.open(rootkey[0],os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW|os.O_CLOEXEC)
    try:
        info=os.fstat(directory);rootstamp=(info.st_dev,info.st_ino)
    finally:os.close(directory)
    if rootkey in bindings and bindings[rootkey]!=rootstamp:fail('source_changed','technical root changed within operation')
    if _metadata_weight((bindings,rootkey,rootstamp))+128>limits.maximum_metadata_bytes:fail('budget_exceeded','technical root witness cap exceeded')
    bindings[rootkey]=rootstamp
    with regular(beneath(root,reference['path']),size=size) as stream:
        before=stamp(stream,include_change=True);reader._budget.require(size)
        phase=reader._budget.phase
        try:
            reader._budget.phase='metadata';raw=stream.read(size);reader._budget.account(len(raw))
        finally:reader._budget.phase=phase
        if len(raw)!=size or stamp(stream,include_change=True)!=before or digest(raw)!=reference['sha256']:
            fail('source_changed','technical exact reference bytes changed')
        original=bindings.get(key)
        if original is not None and original!=before:fail('source_changed','technical immutable path rebound')
        if _metadata_weight((bindings,key,before))+128>limits.maximum_metadata_bytes:fail('budget_exceeded','technical binding metadata cap exceeded')
        bindings[key]=before;_check_bindings({key:before})
    value=decode_json(raw,'selected technical metadata')
    if _metadata_weight(value)>limits.maximum_metadata_bytes:fail('budget_exceeded','decoded technical metadata cap exceeded')
    return value,raw


def _load_telemetry(reader,telemetry,limits,bindings,*,creating_binding=False):
    if not isinstance(reader,StateReader) or not isinstance(telemetry,TelemetrySelection):
        raise TypeError('explicit StateReader and TelemetrySelection required')
    data=telemetry.document();root=telemetry.artifact_root
    references=[];cache={}
    def read(scope,ref):
        key=(str(Path(scope).absolute()),json_bytes(ref))
        if key not in cache:
            retained=_metadata_weight((telemetry._raw,data,cache,bindings,references))
            if int(ref['bytes'])>limits.maximum_metadata_bytes-retained:
                fail('budget_exceeded','aggregate metadata credit exhausted before selected read')
            value=_read_reference(reader,scope,ref,limits,bindings)
            if _metadata_weight((telemetry._raw,data,cache,bindings,references,key,value))>limits.maximum_metadata_bytes:
                fail('budget_exceeded','decoded selected metadata exceeds combined allowance')
            cache[key]=value
            references.append((Path(scope),ref))
        return cache[key]
    commit,commit_raw=read(root,data['commit_ref'])
    if commit.get('schema')!='qcl-negf.telemetry-commit.v1':fail('unsupported_format','separate technical commit required')
    validator=schema_validator('telemetry-envelope.schema.json')
    _schema_check(validator.evolve(schema={'$ref':'#/$defs/TelemetryCommit','$defs':validator.schema['$defs']}),commit)
    summary,summary_raw=read(root,commit['summary'])
    validate_telemetry_summary(summary)
    for key in ('session','transport','generation','closed','baseline','source_cursors'):
        if encode(dict(value=commit[key]))!=encode(dict(value=summary[key])):fail('identity_mismatch','technical commit/summary transaction differs')
    previous=previous_summary=None
    if commit['previous'] is not None:
        previous,_=read(root,commit['previous'])
        previous_summary,_=read(root,previous['summary'])
    phase_refs=[]
    for ar in commit['accepted_ranges']:
        if ar['retained_publication'] is None or ar['retained_segment'] is None:
            fail('capability_unavailable','selected delta original raw context was not retained; explicit raw context required')
        row=dict(publication_ref=ar['publication'],segment_ref=ar['segment'],
            retained_publication_ref=ar['retained_publication'],retained_segment_ref=ar['retained_segment'])
        if row not in phase_refs:phase_refs.append(row)
    phase_refs.extend(data['phase_context_refs'])
    contexts=[];seen=set()
    for row in phase_refs:
        key=json_bytes(row['publication_ref'])
        if key in seen:fail('corrupt_result','duplicate original phase context input')
        seen.add(key)
        for original,retained in (('publication_ref','retained_publication_ref'),('segment_ref','retained_segment_ref')):
            left,right=row[original],row[retained]
            if any(left[k]!=right[k] for k in ('bytes','sha256','media_type','schema')):
                fail('corrupt_result','retained phase reference differs from original bytes')
        publication,_=read(root,row['retained_publication_ref']);segment,_=read(root,row['retained_segment_ref'])
        contexts.append(dict(publication=publication,publication_ref=row['publication_ref'],segment=segment,segment_ref=row['segment_ref']))
    resume=[]
    for row in data['resume_context_refs']:
        c,_=read(root,row['commit_ref']);ss,_=read(root,row['summary_ref'])
        resume.append(dict(commit=c,commit_ref=row['commit_ref'],summary=ss,summary_ref=row['summary_ref']))
    kwargs={} if previous is None else dict(previous=previous,previous_ref=commit['previous'],previous_summary=previous_summary,previous_summary_ref=previous['summary'])
    validate_telemetry_commit(commit,summary=summary,summary_ref=commit['summary'],phase_contexts=contexts,resume_contexts=resume,**kwargs)
    source_binding,source_raw=read(root,data['source_binding_ref'])
    plan,plan_raw=read(telemetry.canonical_plan_root,data['canonical_plan_ref'])
    _schema_check(schema_validator('scientific-plan.schema.json'),plan)
    context=dict(expected_state=reader.source.state,expected_transport=data['expected_transport'],manifest=reader._manifest,
        manifest_ref=reader.source.state['manifest'],source_transport=data['source_transport'],canonical_plan_ref=data['canonical_plan_ref'],
        expected_producer_provenance=data['expected_producer_provenance'])
    validate_telemetry_source_binding(source_binding,**context)
    if commit['transport']!=data['expected_transport']:fail('identity_mismatch','technical transport differs from caller context')
    binding=None
    if commit['closed'] and data['binding_ref'] is None and not creating_binding:
        fail('selection_required','closed selected telemetry requires exact retained close binding')
    if data['binding_ref'] is not None:
        binding,_=read(root,data['binding_ref'])
        validate_telemetry_binding(binding,commit=commit,commit_ref=data['commit_ref'],summary=summary,summary_ref=commit['summary'],
            source_binding=source_binding,source_binding_ref=data['source_binding_ref'],**context)
    _bounded_objects((data,commit,summary,source_binding,plan,binding,previous,previous_summary,contexts,resume),limits.maximum_objects)
    if _metadata_weight((data,cache,bindings))>limits.maximum_metadata_bytes:
        fail('budget_exceeded','aggregate selected technical metadata exceeds allowance')
    if data['observed_unix'] is not None and data['observed_unix']<commit['published_unix']:
        fail('invalid_input','explicit observation precedes selected publication')
    return dict(commit=commit,commit_raw=commit_raw,summary=summary,summary_raw=summary_raw,source_binding=source_binding,
        source_binding_raw=source_raw,plan=plan,plan_raw=plan_raw,binding=binding,input=data,references=references)


def validate_performance_binding(value,*,expected_state,expected_transport,commit,commit_ref,summary,summary_ref,
        source_binding,source_binding_ref,manifest,manifest_ref,source_transport,canonical_plan_ref,expected_producer_provenance):
    """Pure current technical binding validator; old sidecar arguments are retired."""
    result=validate_telemetry_binding(value,expected_state=expected_state,expected_transport=expected_transport,
        commit=commit,commit_ref=commit_ref,summary=summary,summary_ref=summary_ref,source_binding=source_binding,
        source_binding_ref=source_binding_ref,manifest=manifest,manifest_ref=manifest_ref,source_transport=source_transport,
        canonical_plan_ref=canonical_plan_ref,expected_producer_provenance=expected_producer_provenance)
    return decode_json(result['commit']['raw_utf8'].encode()),decode_json(result['summary']['raw_utf8'].encode())


def read_performance_summary(reader: StateReader,*,telemetry: TelemetrySelection | None,limits=None):
    if not isinstance(reader,StateReader):raise TypeError('explicit selected StateReader required')
    limits=limits or PerformanceLimits()
    if not isinstance(limits,PerformanceLimits):raise TypeError('PerformanceLimits required')
    if telemetry is None:return dict(status='unavailable',reason='source_binding_missing')
    bindings={}
    with reader._consumer_operation(limits.maximum_io_bytes,verify_source_stability=True,maximum_stability_bytes=limits.maximum_metadata_bytes):
        loaded=_load_telemetry(reader,telemetry,limits,bindings);commit=loaded['commit'];summary=loaded['summary']
        answer=dict(status='available',reason=None,schema='qcl-negf.performance-summary-view.v2',scope='execution_attempt',
            source_state=reader.source.state,transport=commit['transport'],session_id=commit['session']['source']['session_id'],
            generation=commit['generation'],closed=commit['closed'],published_unix=commit['published_unix'],summary=summary,
            coverage=summary['coverage'],retention=summary['retention'],observer_self_cost=summary['observer_self_cost'],
            units=dict(cpu='seconds',covered_capacity='logical_core_seconds',memory='bytes'),
            integrity_scope='selected_bytes_and_supplied_context_relations',
            integrity=dict(original_selected_metadata_bytes=True,commit_delta_contexts='public_validator_exact_current_delta',
                provenance_origin='caller_assertions; actual P07/AiiDA INPUT_CREATE verification separate',catalogs_or_parquet_opened=False),
            io_counters=reader.io_counters)
        answer=_exact(answer)
        if len(json_bytes(answer))>limits.maximum_json_bytes:fail('budget_exceeded','performance response exceeds cap')
        _check_bindings(bindings)
        return answer
