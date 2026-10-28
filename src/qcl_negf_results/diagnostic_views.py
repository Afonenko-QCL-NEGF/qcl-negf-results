"""Bounded scalar projections of canonical native5 events; no other resolver."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, replace
import copy
from importlib.resources import files
import json
import math
import struct
from typing import Mapping

from qcl_negf_contracts.messages import ContractError
from .state import StateReader, Document, _metadata_weight
from .provenance import _selected_source_provenance


@dataclass(frozen=True)
class DiagnosticViewLimits:
    maximum_metadata_bytes: int = 1024 * 1024
    maximum_owner_bytes: int = 64 * 1024 * 1024
    maximum_io_bytes: int = 128 * 1024 * 1024
    maximum_selected_bytes: int = 1024 * 1024
    maximum_json_bytes: int = 2 * 1024 * 1024
    maximum_objects: int = 4096
    maximum_rows: int = 512
    maximum_columns: int = 16
    maximum_source_entries: int = 4096

    def __post_init__(self):
        if any(type(value) is not int or value < 1 for value in vars(self).values()):
            raise ValueError('diagnostic limits must be positive integers')
        if any(value > self.__dataclass_fields__[name].default for name,value in vars(self).items()):
            raise ValueError('diagnostic limits may tighten, never enlarge public caps')


def _fields(stage):
    row_type = 'SCBAIteration' if stage == 'scba' else 'OuterIteration'
    node = json.loads(files('qcl_negf_contracts').joinpath('core_layout.json').read_text())['graph']['nodes'][row_type]
    return {field['name']:field['type'] for field in node['fields'] if field['type'] in ('Float32','Float64')}


@dataclass(frozen=True)
class HistoryViewRequest:
    stage: str
    fields: tuple[str, ...]
    maximum_records: int = 256

    def __post_init__(self):
        if self.stage not in ('scba','outer'):
            raise ValueError('stage must be scba or outer')
        if (not isinstance(self.fields,tuple) or not self.fields
                or any(not isinstance(name,str) or name not in _fields(self.stage) for name in self.fields)
                or len(set(self.fields)) != len(self.fields)):
            raise ValueError('unique direct published IEEE scalar fields required')
        if type(self.maximum_records) is not int or self.maximum_records < 1:
            raise ValueError('positive consumed event count required')


@dataclass(frozen=True)
class DiagnosticViewBatch:
    source: Document
    records: tuple[Document, ...]
    rows: tuple[Document, ...]
    witness_updates: tuple[Document, ...]
    cursor: Document
    at_cutoff: bool
    io_counters: Mapping[str, int | None]


def _bounded(value, maximum):
    encoded = json.dumps(value,ensure_ascii=False,allow_nan=False,separators=(',',':')).encode('utf-8')
    if len(encoded) > maximum:
        raise ContractError('diagnostic JSON/selected bytes exceed limit','budget_exceeded')
    return value


def _pairs(values):
    result = {}
    for pair in values:
        fields = pair['fields']; key = fields['first']['value']
        if key in result:
            raise ContractError('duplicate stored units/missing reason','corrupt_result')
        value = fields['second']
        result[key] = value['value'] if isinstance(value,dict) else value
    return result


class DiagnosticReader:
    """Borrow the caller's StateReader and its canonical catalog/cursor/proofs."""
    def __init__(self, reader: StateReader, *, limits: DiagnosticViewLimits | None = None):
        if not isinstance(reader,StateReader):
            raise TypeError('explicit selected StateReader required')
        self.reader, self.limits = reader, limits or DiagnosticViewLimits()
        if not isinstance(self.limits,DiagnosticViewLimits):
            raise TypeError('limits must be DiagnosticViewLimits')

    @contextmanager
    def _operation(self):
        reader, limits = self.reader, self.limits
        original = reader.limits
        reader.limits = replace(original,
            maximum_metadata_bytes=min(original.maximum_metadata_bytes,limits.maximum_metadata_bytes),
            maximum_record_bytes=min(original.maximum_record_bytes,limits.maximum_selected_bytes),
            maximum_batch_bytes=min(original.maximum_batch_bytes,limits.maximum_selected_bytes),
            maximum_array_bytes=min(original.maximum_array_bytes,limits.maximum_selected_bytes),
            maximum_hdf_objects=min(original.maximum_hdf_objects,limits.maximum_objects))
        try:
            with reader._consumer_operation(limits.maximum_io_bytes, maximum_owner_bytes=limits.maximum_owner_bytes):
                yield
        finally:
            reader.limits = original

    def describe(self) -> Document:
        with self._operation():
            source = self.reader.source
            value = dict(source=source.state,invocation=source.invocation,lineage=source.lineage,
                status=source.status,closed=source.closed,history=source.history,
                fields={stage:_fields(stage) for stage in ('scba','outer')},
                units=dict(status='not_loaded',reason='stored_record_read_required'),
                resources=dict(checkpoint=dict(status='declared' if self.reader._manifest['checkpoint'] else 'not_available',
                    reference=copy.deepcopy(self.reader._manifest['checkpoint']),payload_bytes=None,
                    reason='payload_cost_not_recorded'),
                    performance_summary=dict(status='not_available',reason='not_recorded_in_result_v2')),
                provenance=_selected_source_provenance(self.reader),io_counters=self.reader.io_counters)
            _bounded(value,self.limits.maximum_metadata_bytes)
            return _bounded(value,self.limits.maximum_json_bytes)

    def read(self, request: HistoryViewRequest, *, cursor: Document | None = None) -> DiagnosticViewBatch:
        if not isinstance(request,HistoryViewRequest):
            raise TypeError('HistoryViewRequest required')
        limits, reader = self.limits, self.reader
        if (request.maximum_records > min(limits.maximum_rows,reader.limits.maximum_batch_records)
                or len(request.fields)>limits.maximum_columns
                or request.maximum_records*len(request.fields)*16>limits.maximum_selected_bytes
                or len(reader._index)>limits.maximum_source_entries):
            raise ContractError('diagnostic rows/columns/selected/source limit exceeded','budget_exceeded')
        with self._operation():
            batch = reader.history_batch(cursor,maximum_records=request.maximum_records)
            if _metadata_weight(batch.records)>limits.maximum_metadata_bytes:
                raise ContractError('retained diagnostic metadata exceeds limit','budget_exceeded')
            _bounded(batch.records,limits.maximum_selected_bytes)
            rows, updates, previous = [], [], None
            for record in batch.records:
                core = record['core']['fields']
                if core['action']['value'] == 'witness_update':
                    updates.append(copy.deepcopy(record))
                    continue
                if core['stage']['value'] != request.stage:
                    continue
                units, missing = _pairs(core['units']), _pairs(core['missing_reasons'])
                cells = {}
                for field in request.fields:
                    tag = core['row']['fields'][field]
                    value = struct.unpack('>d' if tag['type']=='f64' else '>f',bytes.fromhex(tag['bits']))[0]
                    invalid = None if math.isfinite(value) else 'nan' if math.isnan(value) else '+Inf' if value>0 else '-Inf'
                    cells[field] = dict(tagged=copy.deepcopy(tag),value=value if invalid is None else None,
                        invalid_reason=invalid,units=units.get(field),missing_reason=missing.get(field))
                broken = previous is None or any(cell['invalid_reason'] is not None for cell in cells.values())
                if previous is not None:
                    pcore, pcells = previous
                    broken |= any(cell['invalid_reason'] is not None for cell in pcells.values())
                    broken |= int(core['sequence']['decimal']) != int(pcore['sequence']['decimal'])+1
                    broken |= any(core[key]!=pcore[key] for key in
                        ('domain_revision','discretization_id','basis_id','stage','outer_iteration'))
                    iteration = core['inner_iteration'] if request.stage=='scba' else core['outer_iteration']
                    prior_iteration = pcore['inner_iteration'] if request.stage=='scba' else pcore['outer_iteration']
                    broken |= int(iteration['decimal']) <= int(prior_iteration['decimal'])
                rows.append(dict(row_key=copy.deepcopy(record['row_key']),stream_ordinal=record['stream_ordinal'],
                    sequence=copy.deepcopy(core['sequence']),source_identity=copy.deepcopy(record['identity']),
                    source_epoch=record['invocation']['source_epoch'],status=copy.deepcopy(core['status']),
                    quality=copy.deepcopy(core['quality']),converged=core['converged'],
                    cells=cells,break_before=bool(broken)))
                previous = core, cells
            value = dict(source=batch.source,records=batch.records,rows=rows,witness_updates=updates,
                cursor=batch.cursor,at_cutoff=batch.at_cutoff,io_counters=reader.io_counters)
            _bounded(value,limits.maximum_json_bytes)
            return DiagnosticViewBatch(batch.source,batch.records,tuple(rows),tuple(updates),batch.cursor,
                                       batch.at_cutoff,reader.io_counters)


def describe_saved_diagnostics(reader: StateReader, *, limits: DiagnosticViewLimits | None = None) -> Document:
    return DiagnosticReader(reader,limits=limits).describe()


def read_saved_history(reader: StateReader,request: HistoryViewRequest, *, cursor=None,
                       limits: DiagnosticViewLimits | None = None) -> DiagnosticViewBatch:
    return DiagnosticReader(reader,limits=limits).read(request,cursor=cursor)
