"""Bounded logical witness selection from authenticated native5 records.

Whole original native owners are retained. This module writes no derived HDF codec.
"""
from __future__ import annotations

import copy
import hashlib
import math
import struct
import sys

from qcl_negf_contracts.artifacts import CONTRACT_SET
from qcl_negf_contracts.messages import ContractError
from .commits import json_bytes
from .diagnostic_views import DiagnosticViewLimits
from .state import StateReader, _metadata_weight

POLICY = "per-attempt-first-first-positive-ratio-worst-last-v1"


def select_history_witnesses(reader: StateReader, *, limits: DiagnosticViewLimits | None = None) -> dict:
    """Select exact stored evidence, with no scalar mutation or physical omission."""
    if not isinstance(reader,StateReader):
        raise TypeError('explicit selected StateReader required')
    limits=limits or DiagnosticViewLimits()
    if not isinstance(limits,DiagnosticViewLimits):raise TypeError('DiagnosticViewLimits required')
    latest={}; groups={}; parents=[]; scalar_hash=hashlib.sha256(); scalars=updates=0
    with reader._consumer_operation(limits.maximum_io_bytes,maximum_owner_bytes=limits.maximum_owner_bytes,
            verify_source_stability=True,maximum_stability_bytes=limits.maximum_metadata_bytes):
        try:
            chain=[reader]; current=reader; seen=set()
            while current.source.lineage['parent_state'] is not None:
                state=current.source.lineage['parent_state']; key=json_bytes(state)
                if key in seen:raise ContractError('witness lineage cycle','corrupt_result')
                seen.add(key)
                if len(chain)>=limits.maximum_source_entries:raise ContractError('witness lineage exceeds limit','budget_exceeded')
                current=current._parent_reader(state);parents.append(current);chain.append(current)
            for current in reversed(chain):
                current.import_history(); source=current.source.state; cursor=None
                while True:
                    before=copy.deepcopy(cursor);batch=current.history_batch(cursor,maximum_records=1)
                    for record in batch.records:
                        core=record['core']['fields'];target=record['target']
                        if core['stage']['value']!='scba':continue
                        if target is None and len(latest)>=limits.maximum_source_entries:
                            raise ContractError('latest witness map entry limit','budget_exceeded')
                        # Reserve the complete input tree plus source/cursor before
                        # copying its smaller selection entry. One bounded raw row
                        # remains live alongside old and prospective map contents.
                        if _metadata_weight((latest,groups))+_metadata_weight((record,source,before))+512>limits.maximum_metadata_bytes:
                            raise ContractError('latest witness working metadata limit','budget_exceeded')
                        if target is None:
                            identity=record['identity'];epoch=record['invocation']['source_epoch'];ordinal=record['stream_ordinal']
                            key=json_bytes(dict(identity=identity,source_epoch=epoch,ordinal=ordinal))
                            entry=dict(sequence=int(core['row']['fields']['ν']['decimal']),row_key=copy.deepcopy(record['row_key']),
                                source_state=copy.deepcopy(source),source_identity=copy.deepcopy(identity),source_epoch=epoch,
                                append_ordinal=ordinal,cursor_before=before,witness_state=copy.deepcopy(source),witness=None)
                            scalars+=1;scalar_hash.update(json_bytes(core['row']))
                            groupkey=json_bytes(dict(identity=identity,source_epoch=epoch))
                            groups.setdefault(groupkey,dict(source_state=copy.deepcopy(source),identity=copy.deepcopy(identity),source_epoch=epoch,keys=[]))['keys'].append(key)
                        else:
                            key=json_bytes(dict(identity=target['source_identity'],source_epoch=target['source_epoch'],ordinal=target['append_ordinal']))
                            if key not in latest:raise ContractError('witness update has no selected original append','corrupt_result')
                            entry=copy.deepcopy(latest[key]);entry.update(cursor_before=before,witness_state=copy.deepcopy(source));updates+=1
                        entry['witness']=copy.deepcopy(core['row']['fields']['witness'])
                        if len(latest)+(key not in latest)>limits.maximum_source_entries:
                            raise ContractError('latest witness map entry limit','budget_exceeded')
                        if _metadata_weight((latest,groups,entry))+128>limits.maximum_metadata_bytes:
                            raise ContractError('latest witness map metadata limit','budget_exceeded')
                        latest[key]=entry
                    cursor=batch.cursor
                    if batch.at_cutoff:break
            # Both the original working map and detached public plan are live
            # until return. Reserve its conservative full-map copy upper bound.
            if 2*_metadata_weight((latest,groups))+4096>limits.maximum_metadata_bytes:
                raise ContractError('witness plan and map exceed shared metadata allowance','budget_exceeded')
            attempts=[]
            for group in groups.values():
                candidates=[latest[k] for k in group.pop('keys') if latest[k]['witness'] is not None]
                if not candidates:continue
                first_positive=worst=None;worst_value=None
                for i,entry in enumerate(candidates):
                    ratio=entry['witness']['fields']['ratio']
                    number=struct.unpack('>d' if ratio['type']=='f64' else '>f',bytes.fromhex(ratio['bits']))[0]
                    if math.isfinite(number):
                        if first_positive is None and number>0:first_positive=i
                        if worst is None or number>worst_value:worst=i;worst_value=number
                chosen=sorted({i for i in (0,len(candidates)-1,first_positive,worst) if i is not None})
                selected=[copy.deepcopy(candidates[i]) for i in chosen]
                omitted=[entry['sequence'] for i,entry in enumerate(candidates) if i not in chosen]
                attempts.append(dict(group,attempt=group['identity']['attempt'],source_witnesses=len(candidates),
                    selected_witnesses=len(selected),omitted_witnesses=len(omitted),selected_sequences=[r['sequence'] for r in selected],
                    selected=selected,omitted_sequences_sha256=hashlib.sha256(json_bytes(omitted)).hexdigest()))
            answer=dict(schema='qcl-negf.witness-selection-proof.v1',contract_set=CONTRACT_SET,policy=POLICY,
                source=copy.deepcopy(reader.source.state),attempts=attempts,source_witnesses=sum(a['source_witnesses'] for a in attempts),
                selected_witnesses=sum(a['selected_witnesses'] for a in attempts),omitted_witnesses=sum(a['omitted_witnesses'] for a in attempts),
                selected_sequences=[s for a in attempts for s in a['selected_sequences']],scalar_history='all rows unchanged',
                scalar_rows=scalars,witness_updates=updates,scalar_records_sha256=scalar_hash.hexdigest(),
                physical_payload_scope='whole original owners retained unchanged',scientific_acceptance='not_assessed')
            if len(json_bytes(answer))>limits.maximum_json_bytes:raise ContractError('witness plan JSON limit','budget_exceeded')
            return answer
        finally:
            primary=sys.exception();first=None
            for parent in reversed(parents):
                try:parent.close()
                except BaseException as error:
                    if first is None:first=error
                    else:first.add_note('additional witness lineage cleanup '+type(error).__name__)
            if first is not None:
                if primary is None:raise first
                primary.add_note('witness lineage cleanup also raised '+type(first).__name__)
