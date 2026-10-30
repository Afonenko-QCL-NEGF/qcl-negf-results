"""Bounded exact telemetry catalog prefix. Paths are local to its declared root."""
from __future__ import annotations

import copy
import hashlib
from pathlib import Path
from typing import Any

from qcl_negf_contracts.artifacts import Artifact, digest_value, relative_path, require_contract_set
from qcl_negf_contracts.messages import ContractError
from ._capture_io import beneath,regular,stamp
from .commits import decode_json

CATALOG_SCHEMA = "qcl-negf.telemetry-catalog.v2"


def _catalog_rows(root: Path,watermark: dict[str,Any],*,limits=None,budget=None,bindings=None):
    """Return exact original rows plus authenticated catalog refs, never a locator."""
    from .state import ResultReadLimits,_IOBudget,_metadata_weight
    from .performance_views import _check_bindings
    limits=limits or ResultReadLimits()
    if not isinstance(limits,ResultReadLimits):raise TypeError('ResultReadLimits required')
    budget=budget if budget is not None else _IOBudget(limits.maximum_io_bytes)
    bindings=bindings if bindings is not None else {}
    root=Path(root)
    if not isinstance(watermark,dict) or set(watermark)!={'path','entries','sha256'}:
        raise ContractError('invalid telemetry catalog watermark','corrupt_result')
    count=watermark['entries']
    if type(count) is not int or not 0<=count<=limits.maximum_catalog_entries:
        raise ContractError('telemetry catalog entry bound exceeded','budget_exceeded')
    chunks=[];catalogs=[];reference=copy.deepcopy(watermark);seen=set();paths=set();remaining=count
    while reference is not None:
        if len(chunks)>=limits.maximum_catalog_pages:raise ContractError('telemetry catalog page bound exceeded','budget_exceeded')
        if not isinstance(reference,dict) or set(reference)!={'path','entries','sha256'}:
            raise ContractError('invalid telemetry catalog reference','corrupt_result')
        name=relative_path(reference['path'])
        if name in seen:raise ContractError('telemetry catalog cycle','corrupt_result')
        seen.add(name)
        with regular(beneath(root,name)) as stream:
            before=stamp(stream,include_change=True);size=before[2]
            if size>limits.maximum_metadata_bytes:raise ContractError('telemetry catalog metadata bound exceeded','budget_exceeded')
            budget.require(size);phase=budget.phase
            try:
                budget.phase='metadata';payload=stream.read(size);budget.account(len(payload))
            finally:budget.phase=phase
            if len(payload)!=size or stamp(stream,include_change=True)!=before:
                raise ContractError('telemetry catalog source changed during read','source_changed')
            if hashlib.sha256(payload).hexdigest()!=digest_value(reference['sha256']):
                raise ContractError('telemetry catalog checksum mismatch','corrupt_result')
            key=(str(root.absolute()),name)
            if key in bindings and bindings[key]!=before:raise ContractError('telemetry catalog path rebound','source_changed')
            if _metadata_weight((bindings,key,before))+128>limits.maximum_catalog_bytes:
                raise ContractError('telemetry catalog binding metadata bound exceeded','budget_exceeded')
            bindings[key]=before;_check_bindings({key:before})
        value=decode_json(payload,'telemetry catalog');require_contract_set(value)
        rows=value.get('artifacts')
        if (set(value)!={'schema','contract_set','entries','previous','artifacts'} or value['schema']!=CATALOG_SCHEMA
                or type(value['entries']) is not int or value['entries']!=remaining or reference['entries']!=remaining
                or not isinstance(rows,list) or not rows or not all(isinstance(r,dict) for r in rows)):
            raise ContractError('telemetry catalog prefix is inconsistent','corrupt_result')
        if len(rows)>remaining:raise ContractError('telemetry catalog exceeds watermark','corrupt_result')
        parsed=[]
        for row in rows:
            artifact=Artifact.parse(row)
            if artifact.path in paths:raise ContractError('duplicate telemetry catalog path','corrupt_result')
            paths.add(artifact.path);parsed.append((artifact,copy.deepcopy(row)))
        remaining-=len(rows)
        ref=dict(path=name,bytes=str(size),sha256=reference['sha256'],media_type='application/json',schema=CATALOG_SCHEMA)
        prospective=(chunks,catalogs,parsed,ref,bindings,paths,seen)
        if _metadata_weight(prospective)>limits.maximum_catalog_bytes:
            raise ContractError('decoded telemetry catalog inventory bound exceeded','budget_exceeded')
        chunks.append(parsed);catalogs.append((ref,value));reference=value['previous']
    if remaining:raise ContractError('telemetry catalog is incomplete','corrupt_result')
    _check_bindings(bindings)
    return [item for chunk in reversed(chunks) for item in chunk],list(reversed(catalogs))


def catalog_artifacts(root: Path,watermark: dict[str,Any],*,limits=None,budget=None) -> tuple[Artifact,...]:
    """Read the bounded exact closed prefix; later appends never change its pin."""
    rows,_=_catalog_rows(root,watermark,limits=limits,budget=budget)
    return tuple(item for item,_ in rows)
