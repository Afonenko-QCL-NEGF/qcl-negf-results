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


def _retained_parquet_header(stream, descriptor, *, maximum_metadata_bytes):
    """One already-qualified budgeted FD; inspect footer/schema, never decode rows."""
    import pyarrow.parquet as pq
    from qcl_negf_contracts.telemetry import TABLE_FIELDS
    from .compaction import validate_performance_schema
    reference = descriptor['ref']; size = int(reference['bytes'])
    if size < 12 or stream.seek(0, 2) != size:
        raise ContractError('retained Parquet size differs', 'corrupt_result')
    stream.seek(-8, 2); trailer = stream.read(8)
    if len(trailer) != 8 or trailer[4:] != b'PAR1':
        raise ContractError('invalid retained Parquet footer', 'corrupt_result')
    footer = int.from_bytes(trailer[:4], 'little')
    fields = TABLE_FIELDS[descriptor['table']]
    if 2*footer+len(fields)*1024+4096 > maximum_metadata_bytes:
        raise ContractError('retained Parquet footer metadata allowance exceeded before decode', 'budget_exceeded')
    stream.seek(0)
    with pq.ParquetFile(stream, pre_buffer=False, buffer_size=0,
            thrift_string_size_limit=max(1, maximum_metadata_bytes),
            thrift_container_size_limit=max(1, maximum_metadata_bytes//128)) as file:
        if (file.metadata.num_columns != len(fields)
                or validate_performance_schema(file.schema_arrow) != descriptor['table']
                or file.metadata.num_rows != int(descriptor['rows'])):
            raise ContractError('retained Parquet footer rows/table differ from catalog', 'corrupt_result')


def _retained_payload(root, descriptor, *, budget, maximum_metadata_bytes, bindings, retain_raw=False):
    """Stream one exact Parquet member and keep its same-open identity witness."""
    from .state import _BudgetStream, _metadata_weight
    from .performance_views import _check_bindings
    reference = descriptor['ref']; size = int(reference['bytes'])
    if reference['media_type'] != 'application/vnd.apache.parquet':
        raise ContractError('retained payload requires explicit Parquet type', 'unsupported_format')
    budget.require(size)
    if retain_raw and 2*size+65536 > maximum_metadata_bytes:
        raise ContractError('retained raw payload exceeds prospective combined allowance', 'budget_exceeded')
    key = (str(Path(root).absolute()), reference['path'])
    if _metadata_weight((bindings, key))+512 > maximum_metadata_bytes:
        raise ContractError('retained payload witness allowance exceeded', 'budget_exceeded')
    with regular(beneath(root, reference['path']), size=size) as raw:
        before = stamp(raw, include_change=True)
        if key in bindings and bindings[key] != before:
            raise ContractError('retained immutable payload path rebound', 'source_changed')
        stream = _BudgetStream(raw, budget); phase = budget.phase
        try:
            budget.phase = 'sha'; digest = hashlib.sha256(); captured = bytearray() if retain_raw else None
            remaining = size
            while remaining:
                chunk = stream.read(min(65536, remaining))
                if not chunk:
                    raise ContractError('retained payload shortened within selected read', 'source_changed')
                remaining -= len(chunk); digest.update(chunk)
                if captured is not None: captured.extend(chunk)
            if digest.hexdigest() != reference['sha256'] or stamp(raw, include_change=True) != before:
                raise ContractError('retained payload bytes/identity differ before footer decode', 'source_changed')
            budget.phase = 'metadata'
            _retained_parquet_header(stream, descriptor,
                maximum_metadata_bytes=maximum_metadata_bytes-(2*size if retain_raw else 0))
            if stamp(raw, include_change=True) != before:
                raise ContractError('retained payload changed within same-open read', 'source_changed')
            bindings[key] = before; _check_bindings({key:before})
            return None, None if captured is None else bytes(captured)
        except BaseException as primary:
            try:
                if stamp(raw, include_change=True) != before:
                    raise ContractError('retained payload changed during failed read', 'source_changed')
                _check_bindings({key:before})
            except BaseException as cleanup:
                primary.add_note('retained payload guard: '+str(cleanup))
            raise
        finally:
            budget.phase = phase


def _retained_metadata(value, reference, *, expected_transport):
    """Validate typed snapshot metadata without following its historical locators."""
    from qcl_negf_contracts.telemetry import validate_telemetry_catalog, validate_telemetry_index
    if reference['schema'] == 'qcl-negf.telemetry-catalog.v3':
        validate_telemetry_catalog(value, expected_transport=expected_transport)
    elif reference['schema'] == 'qcl-negf.performance-index.v3':
        validate_telemetry_index(value, expected_transport=expected_transport)
