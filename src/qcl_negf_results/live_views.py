"""Native5 selected-source providers and explicit streaming scalar reductions."""
from __future__ import annotations

from dataclasses import dataclass
import copy
import math
import struct

from qcl_negf_contracts.messages import ContractError
from .state import StateReader, ResultDescription, HistoryBatch, _IOBudget, _ref


def describe_live_history(reader: StateReader) -> ResultDescription:
    if not isinstance(reader, StateReader):
        raise TypeError('selected StateReader required')
    return reader.source


def read_live_history(reader: StateReader, *, cursor=None, maximum_records=256) -> HistoryBatch:
    return reader.history_batch(cursor, maximum_records=maximum_records)


@dataclass(frozen=True)
class ScalarReduction:
    stage: str
    field: str
    first_record: str
    last_record: str

    def __post_init__(self):
        if self.stage not in ('scba', 'outer') or not isinstance(self.field, str) or not self.field:
            raise ValueError('explicit stage and direct scalar row field required')
        import json
        from importlib.resources import files
        layout = json.loads(files('qcl_negf_contracts').joinpath('core_layout.json').read_text())
        row_type = 'SCBAIteration' if self.stage == 'scba' else 'OuterIteration'
        scalar_fields = {field['name'] for field in layout['graph']['nodes'][row_type]['fields']
                         if field['type'] in ('Float32', 'Float64')}
        if self.field not in scalar_fields:
            raise ValueError('field must be a direct IEEE scalar in the published Core layout')
        for value in (self.first_record, self.last_record):
            if (not isinstance(value, str) or not value.isascii() or not value.isdecimal()
                    or value.startswith('0') or int(value) > 2**64-1):
                raise ValueError('canonical positive UInt64 record bound required')
        if int(self.first_record) > int(self.last_record):
            raise ValueError('record bounds reversed')


def _locator(reader, ordinal):
    for _, page in reader._pages:
        if int(page['range']['first_record']) <= ordinal <= int(page['range']['last_record']):
            return dict(source_identity=copy.deepcopy(reader._manifest['identity']),
                source_epoch=reader._manifest['invocation']['source_epoch'], append_ordinal=str(ordinal),
                leaf=_ref(page['artifacts'][-1]), row_offset=str(ordinal-int(page['range']['first_record'])))
    raise ContractError('derived row not in imported catalog', 'corrupt_result')


def derive_history(reader: StateReader, request: ScalarReduction, *, maximum_io_bytes: int) -> dict:
    if not isinstance(reader, StateReader) or not isinstance(request, ScalarReduction):
        raise TypeError('selected reader and ScalarReduction required')
    if type(maximum_io_bytes) is not int or maximum_io_bytes < 1:
        raise ValueError('positive derive I/O budget required')
    if int(request.last_record) > int(reader.source.history['last_record']):
        raise ContractError('reduction exceeds frozen history cutoff', 'invalid_input')
    # Reuse the same budget across nested batch operations, including parent
    # target proofs. Nothing invokes a user callback, Julia or a fresh map.
    reader._operation()
    budget = _IOBudget(min(maximum_io_bytes, reader.limits.maximum_io_bytes))
    previous_budget = reader._shared_budget
    reader._shared_budget = budget
    cursor, minimum, maximum, units = None, None, None, None
    observations = updates = finite = nonfinite = missing = 0
    try:
        while True:
            batch = reader.history_batch(cursor)
            for record in batch.records:
                ordinal = int(record['stream_ordinal'])
                if not int(request.first_record) <= ordinal <= int(request.last_record):
                    continue
                core = record['core']['fields']
                if core['stage']['value'] != request.stage:
                    continue
                if core['action']['value'] == 'witness_update':
                    updates += 1
                    continue
                observations += 1
                value = core['row']['fields'].get(request.field)
                if value is None:
                    missing += 1
                    continue
                if not isinstance(value, dict) or value.get('kind') != 'ieee':
                    raise ContractError('reduction requires stored IEEE scalar field', 'invalid_input')
                declared = {pair['fields']['first']['value']: pair['fields']['second'] for pair in core['units']}
                unit = declared.get(request.field)
                if unit is None:
                    raise ContractError('reduction field has no stored unit', 'invalid_data')
                if units is not None and units != unit:
                    raise ContractError('reduction units differ across selected rows', 'invalid_data')
                units = unit
                number = struct.unpack('>d' if value['type'] == 'f64' else '>f', bytes.fromhex(value['bits']))[0]
                if not math.isfinite(number):
                    nonfinite += 1
                    continue
                finite += 1
                sample = dict(value=copy.deepcopy(value), locator=_locator(reader, ordinal))
                if minimum is None or number < minimum[0]:
                    minimum = (number, sample)
                if maximum is None or number > maximum[0]:
                    maximum = (number, sample)
            cursor = batch.cursor
            if batch.at_cutoff or int(cursor['last_record']) >= int(request.last_record):
                break
        return dict(source=copy.deepcopy(reader.source.state), request=vars(request).copy(), units=units,
            observations=observations, updates=updates, finite_count=finite, nonfinite_count=nonfinite,
            missing_count=missing, minimum=minimum[1] if minimum else None, maximum=maximum[1] if maximum else None,
            coverage=dict(first_record=request.first_record, last_record=request.last_record,
                          witness_updates='counted_separately_no_scalar_replacement'),
            io_counters=budget.counters(), derived_by='qcl_negf_results.live_views.derive_history.v1')
    finally:
        reader._shared_budget = previous_budget


def compare_history(left: StateReader, right: StateReader, *, left_request: ScalarReduction,
                    right_request: ScalarReduction, comparability, maximum_io_bytes: int) -> dict:
    keys = {'model','discretization','protocol','source','evidence','reason'}
    if (not isinstance(comparability, dict) or set(comparability) != keys
            or any(comparability[k] not in ('comparable','different','unknown') for k in keys-{'evidence','reason'})
            or not isinstance(comparability['evidence'], list) or not isinstance(comparability['reason'], str)):
        raise ValueError('explicit comparability states/evidence/reason required')
    a = derive_history(left, left_request, maximum_io_bytes=maximum_io_bytes)
    remaining = maximum_io_bytes - a['io_counters']['bytes_read_total']
    if remaining < 1:
        raise ContractError('comparison I/O budget exhausted', 'budget_exceeded')
    b = derive_history(right, right_request, maximum_io_bytes=remaining)
    eligible = (all(comparability[k] == 'comparable' for k in keys-{'evidence','reason'})
        and left_request.field == right_request.field and a['units'] == b['units'])
    differences = None
    if eligible:
        differences = {}
        for key in ('minimum','maximum'):
            if a[key] is None or b[key] is None:
                differences[key] = None
            else:
                decode = lambda item: struct.unpack('>d' if item['type'] == 'f64' else '>f', bytes.fromhex(item['bits']))[0]
                delta = decode(b[key]['value'])-decode(a[key]['value'])
                differences[key] = dict(kind='ieee',type='f64',bits=struct.pack('>d',delta).hex())
    return dict(left=a, right=b, comparability=copy.deepcopy(comparability), differences=differences,
                scientific_acceptance='not_assessed')
