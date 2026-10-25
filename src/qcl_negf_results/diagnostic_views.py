"""Bounded read-only views of committed scalar history and checkpoint cost.

This reader selects science.history independently of physics.analysis. It never
runs native/full-state validators, progress scans, materialization or a solver.
Provider code owns remote selection, cancellation and temporary input lifetime.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import os
from pathlib import Path
import re
from typing import Mapping

import h5py
import numpy as np

from qcl_negf_contracts.artifacts import Artifact, relative_path, validate_commit
from qcl_negf_contracts.messages import ContractError
from .commits import decode_json, safe_path
from .saved_views import AnalysisReader, _bounded, _error, _open_regular, _saved_budget_json
from .state import _BudgetStream, _IOBudget, _same_json_value, _verify_stream

HISTORY_SCHEMA = 'qcl-negf-scientific-history-v4'
PHASES = ('scba', 'outer')
CORE_IDENTITY = ('point_id', 'execution_id', 'attempt', 'plan_fingerprint')
JS_SAFE = 9007199254740991
SHA = re.compile(r'[0-9a-f]{64}')


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
        ceilings = DiagnosticViewLimits.__dataclass_fields__
        if any(value > ceilings[key].default for key, value in vars(self).items()):
            raise ValueError('diagnostic limits may tighten, never enlarge public caps')


def _exact(value):
    """Existing ExactScalar decimal tag; finite floating values keep their type."""
    if type(value) is int and abs(value) > JS_SAFE:
        return {'kind': 'integer', 'decimal': str(value)}
    if isinstance(value, dict):
        return {key: _exact(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_exact(item) for item in value]
    return value


def _positive_int(value):
    return type(value) is int and value > 0


def _matches(pattern, value):
    return isinstance(value, str) and pattern.fullmatch(value) is not None


def _expected(value):
    required = {*CORE_IDENTITY, 'source_commit_sha256'}
    allowed = required | {'run_uuid', 'calcjob_uuid'}
    if not isinstance(value, Mapping) or not required <= set(value) <= allowed:
        _error('selection_required', 'exact point/execution/source attempt/plan/commit binding required')
    expected = dict(value)
    if (not _positive_int(expected['attempt'])
            or any(not isinstance(expected[key], str) or not expected[key] for key in ('point_id', 'execution_id'))
            or not _matches(SHA, expected['plan_fingerprint'])
            or not _matches(SHA, expected['source_commit_sha256'])
            or any(not isinstance(expected[key], str) or not expected[key]
                   for key in ('run_uuid', 'calcjob_uuid') if key in expected)):
        _error('selection_required', 'invalid exact diagnostic identity')
    return expected


class DiagnosticReader:
    """A selected-role reader; only bounded safe primitives are shared.

    These method aliases have no role selection, constructor, full validator or
    numerical scan. AnalysisReader itself is never instantiated or required.
    """
    _charge_metadata = AnalysisReader._charge_metadata
    _attr = AnalysisReader._attr
    _json = AnalysisReader._json
    _storage = AnalysisReader._storage
    _numeric = AnalysisReader._numeric
    _chunk_bound = AnalysisReader._chunk_bound
    close = AnalysisReader.close

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __init__(self, commit_path, *, expected_identity, limits=None):
        self.limits = limits or DiagnosticViewLimits()
        if not isinstance(self.limits, DiagnosticViewLimits):
            raise ValueError('limits must be DiagnosticViewLimits')
        self.expected = _expected(expected_identity)
        self._budget = _IOBudget(self.limits.maximum_io_bytes)
        self._metadata_bytes = 0
        self._handle = self._stream = self._raw = None
        self._verified = False
        self._span_allocation = 0
        self._spans = {}
        self._bounds = {}
        self._phases = []
        self._attempt_provenance = 'unavailable'
        self.identity_binding = {'native_recorded_fields': [],
            'provider_only_fields': [key for key in ('run_uuid', 'calcjob_uuid') if key in self.expected]}
        path = Path(commit_path)
        if path.is_symlink() or any(parent.is_symlink() for parent in path.parents):
            _error('corrupt_result', 'diagnostic commit path traverses a symbolic link')
        self.commit, payload = _saved_budget_json(path, self.limits.maximum_metadata_bytes, self._budget)
        self._charge_metadata(len(payload))
        self.commit_sha256 = hashlib.sha256(payload).hexdigest()
        if self.commit_sha256 != self.expected['source_commit_sha256']:
            _error('identity_mismatch', 'selected diagnostic commit changed')
        rows = self.commit.get('artifacts')
        if not isinstance(rows, list) or len(rows) > self.limits.maximum_objects:
            _error('budget_exceeded', 'diagnostic artifact inventory exceeds cap')
        # Root identity/state contract still uses the canonical validator. An
        # unrelated unsupported analysis schema/size is outside this read role.
        validate_commit({**self.commit, 'artifacts': []})
        identity = self.commit['identity']
        self.identity_binding['provider_only_fields'] = [key for key in ('run_uuid', 'calcjob_uuid')
            if key in self.expected and key not in identity]
        for key, wanted in self.expected.items():
            if key == 'source_commit_sha256':
                continue
            if key in CORE_IDENTITY or key in identity:
                if not _same_json_value(identity.get(key), wanted):
                    _error('identity_mismatch', 'commit differs from exact diagnostic selection')
        by_path = {}
        for row in rows:
            if not isinstance(row, dict):
                _error('corrupt_result', 'artifact inventory row must be an object')
            name = relative_path(row.get('path'))
            if name in by_path:
                _error('corrupt_result', 'duplicate diagnostic artifact path')
            by_path[name] = row
        owners = [row for row in rows if row.get('role') == 'science.history']
        self.owner = None
        self.history_reason = ('history_owner_missing' if not owners else
                               'history_owner_ambiguous' if len(owners) != 1 else None)
        self.resources = self._resources()
        if self.history_reason is not None:
            return
        selected = owners[0]
        if selected.get('schema') != HISTORY_SCHEMA or selected.get('media_type') != 'application/x-hdf5':
            _error('unsupported_contract', 'requires one native v4 scalar history owner')
        self.owner = Artifact.parse(selected)
        if self.owner.size > self.limits.maximum_owner_bytes:
            _error('budget_exceeded', 'scalar history owner exceeds ordinary read cap')
        closure, pending = {}, [self.owner.path]
        while pending:
            name = pending.pop()
            if name in closure:
                continue
            if name not in by_path:
                _error('corrupt_result', 'history dependency is absent from commit')
            artifact = Artifact.parse(by_path[name])
            if name != self.owner.path and artifact.media_type != 'application/json':
                _error('unsupported_contract', 'history dependencies require bounded JSON metadata')
            closure[name] = by_path[name]
            pending.extend(artifact.dependencies)
        validate_commit({**self.commit, 'artifacts': list(closure.values())})
        for name, row in closure.items():
            if name == self.owner.path:
                continue
            artifact = Artifact.parse(row)
            self._charge_metadata(artifact.size)
            dependency = safe_path(path.parent, name)
            with _open_regular(dependency, expected_size=artifact.size) as raw:
                with _BudgetStream(raw, self._budget) as stream:
                    data = stream.read(artifact.size + 1)
            if len(data) != artifact.size or hashlib.sha256(data).hexdigest() != artifact.sha256:
                _error('corrupt_result', 'history dependency differs from committed bytes')
            decode_json(data, 'diagnostic dependency')
        self.path = safe_path(path.parent, self.owner.path)
        try:
            self._raw = _open_regular(self.path, expected_size=self.owner.size)
            self._opened = os.fstat(self._raw.fileno())
            self._stream = _BudgetStream(self._raw, self._budget)
            self._budget.phase = 'hdf'
            self._handle = h5py.File(self._stream, 'r')
            self._storage()
            header = self._handle.get('metadata')
            if not isinstance(header, h5py.Group):
                _error('unsupported_contract', 'scalar history metadata absent')
            for key, wanted in {'schema': HISTORY_SCHEMA, 'schema_version': '4.0',
                                'artifact_role': 'science.history', 'contract_set': 'qcl-negf.results.v1'}.items():
                if self._attr(header, key) != wanted:
                    _error('unsupported_contract', 'scalar history native format/role differs')
            counts = {}
            for phase in PHASES:
                count = self._attr(header, phase + '_rows')
                if type(count) is not int or count < 0:
                    _error('corrupt_result', 'history requires nonnegative native row counts')
                counts[phase] = count
            self._source_inventory(counts)
            self._describe_phases(counts)
            self._require_allocation(self._span_allocation)
            self._stable()
        except BaseException as error:
            self.close()
            if isinstance(error, ContractError):
                raise
            if isinstance(error, Exception):
                raise ContractError('saved history metadata cannot be read', 'corrupt_result') from error
            raise

    def _native_identity(self, value, *, cumulative):
        if not isinstance(value, dict):
            _error('corrupt_result', 'history source identity must be an object')
        owner = self.commit['identity']
        for key in ('point_id', 'execution_id', 'plan_fingerprint'):
            if not _same_json_value(value.get(key), owner[key]):
                _error('identity_mismatch', 'history source point/execution/frozen plan differs')
        attempt = value.get('attempt')
        if (not _positive_int(attempt) or attempt > owner['attempt']
                or (not cumulative and attempt != owner['attempt'])):
            _error('identity_mismatch', 'history source attempt differs or exceeds cutoff')
        # State-sequence identity belongs to the point commit. History recorder
        # identity predates that state; do not fabricate state-level equivalence.
        if attempt == owner['attempt']:
            for key in ('run_uuid', 'calcjob_uuid'):
                if key in value and key in self.expected and not _same_json_value(value[key], self.expected[key]):
                    _error('identity_mismatch', 'native recorded provider identity differs')
        return attempt

    def _source_inventory(self, counts):
        header = self._handle['metadata']
        has_identity = 'identity_json' in header
        has_inventory = 'source_segments_json' in header
        if has_identity == has_inventory:
            _error('corrupt_result', 'history requires exactly one native identity representation')
        if has_identity:
            identity = self._json('metadata/identity_json')
            attempt = self._native_identity(identity, cumulative=False)
            self.identity_binding = {'native_recorded_fields': sorted(identity),
                'provider_only_fields': [key for key in ('run_uuid', 'calcjob_uuid')
                    if key in self.expected and key not in identity and key not in self.commit['identity']]}
            self._attempt_provenance = 'native_identity'
            for phase in PHASES:
                self._spans[phase] = [{'start': 0, 'stop': counts[phase], 'source_attempt': attempt,
                                       'provenance': 'native_identity'}] if counts[phase] else []
                self._bounds[phase] = {}
            self._span_allocation = 192
            return
        sources = self._json('metadata/source_segments_json')
        if not isinstance(sources, list) or not sources:
            _error('corrupt_result', 'cumulative source inventory must be a nonempty list')
        if len(sources) > self.limits.maximum_source_entries:
            _error('budget_exceeded', 'history source inventory exceeds 4096 entries')
        self._attempt_provenance = 'declared_source_counts'
        grouped = {phase: {} for phase in PHASES}
        intervals = {phase: [] for phase in PHASES}
        seen, selected_recorded = set(), set()
        for source in sources:
            if not isinstance(source, dict) or not _matches(SHA, source.get('sha256')):
                _error('corrupt_result', 'source inventory requires original segment hash')
            if source['sha256'] in seen:
                _error('corrupt_result', 'duplicate cumulative source segment')
            seen.add(source['sha256'])
            attempt = self._native_identity(source.get('identity'), cumulative=True)
            if attempt == self.commit['identity']['attempt']:
                selected_recorded.update(source['identity'])
            for phase in PHASES:
                count = source.get(phase + '_rows')
                if type(count) is not int or count < 0:
                    _error('corrupt_result', 'invalid declared source row count')
                grouped[phase][attempt] = grouped[phase].get(attempt, 0) + count
                first_key, last_key = phase + '_sequence_first', phase + '_sequence_last'
                if (first_key in source) != (last_key in source):
                    _error('corrupt_result', 'incomplete declared source sequence bounds')
                if first_key not in source:
                    continue
                first, last = source[first_key], source[last_key]
                if count == 0:
                    if first is not None or last is not None:
                        _error('corrupt_result', 'empty source has nonempty sequence bounds')
                elif (not _positive_int(first) or not _positive_int(last) or last < first
                      or count > last - first + 1):
                    _error('corrupt_result', 'invalid declared source sequence bounds')
                else:
                    intervals[phase].append((first, last, attempt))
        self.identity_binding = {'native_recorded_fields': sorted(selected_recorded),
            'provider_only_fields': [key for key in ('run_uuid', 'calcjob_uuid')
                if key in self.expected and key not in selected_recorded and key not in self.commit['identity']]}
        for phase in PHASES:
            if sum(grouped[phase].values()) != counts[phase]:
                _error('corrupt_result', 'declared source counts differ from history shape')
            ordered = sorted(intervals[phase])
            for left, right in zip(ordered, ordered[1:]):
                if right[0] <= left[1] or right[2] < left[2]:
                    _error('corrupt_result', 'source sequence bounds overlap or reverse chronological attempts')
            spans, offset = [], 0
            for attempt, count in sorted(grouped[phase].items()):
                if count:
                    spans.append({'start': offset, 'stop': offset + count, 'source_attempt': attempt,
                                  'provenance': 'declared_source_counts'})
                offset += count
            self._spans[phase] = spans
            bounds = {}
            # Enforce a bound on selected coordinates only when every positive
            # source of this attempt actually declares sequence bounds.
            for attempt in grouped[phase]:
                rows_with_bounds = [row for row in sources if row['identity']['attempt'] == attempt
                                    and row[phase + '_rows'] > 0]
                if rows_with_bounds and all(phase + '_sequence_first' in row for row in rows_with_bounds):
                    bounds[attempt] = [(row[phase + '_sequence_first'], row[phase + '_sequence_last'])
                                       for row in rows_with_bounds]
            self._bounds[phase] = bounds
        self._span_allocation = len(sources) * 128 + sum(len(v) * 96 for v in self._spans.values())

    def _describe_phases(self, counts):
        for phase in PHASES:
            group = self._handle.get(phase)
            if not isinstance(group, h5py.Group):
                _error('corrupt_result', 'history phase table absent')
            coordinates = {'sequence', 'iteration'} | ({'outer_iteration'} if phase == 'scba' else set())
            if not coordinates <= set(group):
                _error('corrupt_result', 'native history coordinates are missing')
            columns = []
            for name in group:
                column = self._numeric(phase + '/' + name, (counts[phase],))
                if column.dtype.kind not in 'if' or column.dtype.itemsize != 8:
                    _error('unsupported_contract', 'history requires native Int64/Float64 scalar columns')
                if name in coordinates:
                    if column.dtype.kind != 'i':
                        _error('corrupt_result', 'history coordinates require signed Int64')
                else:
                    units = self._attr(column, 'units')
                    if units is not None and not isinstance(units, str):
                        _error('unsupported_contract', 'history units must be fixed scalar text')
                    columns.append({'name': name, 'units': units, 'dtype': str(column.dtype)})
            self._phases.append({'phase': phase, 'total_rows': counts[phase],
                                 'columns': columns, 'attempt_spans': self._spans[phase]})

    def _resources(self):
        performance = self.commit.get('performance')
        fields = ('checkpoint_payload_seconds', 'checkpoint_payload_bytes',
                  'includes_commit_publication', 'jit_separately_measured')
        checkpoint = None
        if isinstance(performance, dict) and any(key in performance for key in fields):
            checkpoint = {key: performance.get(key) for key in fields}
            for key in fields[:2]:
                value = checkpoint[key]
                valid = (((type(value) is int and value >= 0) or
                          (type(value) is float and math.isfinite(value) and value >= 0))
                         if key.endswith('seconds') else type(value) is int and value >= 0)
                if value is not None and not valid:
                    _error('corrupt_result', 'invalid recorded checkpoint cost')
            for key in fields[2:]:
                if checkpoint[key] is not None and type(checkpoint[key]) is not bool:
                    _error('corrupt_result', 'invalid recorded checkpoint coverage flag')
            checkpoint['scope'] = 'point_commit_checkpoint_payload'
        return {'status': 'available' if checkpoint is not None else 'unavailable',
                'reason': None if checkpoint is not None else 'checkpoint_metadata_missing',
                'checkpoint': checkpoint,
                'performance_summary': {'status': 'unavailable', 'reason': 'source_binding_missing'}}

    def _artifact(self):
        return None if self.owner is None else {'artifact_id': self.owner.path, 'role': self.owner.role,
            'schema': self.owner.schema, 'bytes': self.owner.size, 'sha256': self.owner.sha256}

    def _coverage(self):
        return {'history_scope': 'committed_closed_prefix', 'attempt_spans': self._attempt_provenance,
                'full_coordinate_scan': False, 'scientific_acceptance': 'not_assessed'}

    def _integrity(self, *, selected=False):
        return {'metadata_checked': True, 'owner_sha_verified': self._verified,
                'native_validation': 'not_performed',
                'scope': 'owner_sha_and_selected_history_coordinates' if selected else 'committed_history_metadata'}

    def _base(self):
        return {'identity': dict(self.commit['identity']), 'identity_binding': self.identity_binding,
                'source_commit_sha256': self.commit_sha256, 'generation': self.commit['generation'],
                'closed': self.commit.get('closed') if type(self.commit.get('closed')) is bool else None,
                'recorded_statuses': {key: self.commit.get(key) for key in ('terminal_status', 'quality',
                     'converged', 'scientific_accepted', 'stationary_assessment')}}

    def _finish(self, value):
        return _bounded(_exact(value), self.limits.maximum_json_bytes)

    def describe(self):
        return self._finish({'schema': 'qcl-negf.saved-diagnostics.v1', **self._base(),
            'history': {'status': 'unavailable' if self.history_reason else 'available',
                        'reason': self.history_reason, 'source_artifact': self._artifact(), 'phases': self._phases},
            'resources': self.resources, 'integrity': self._integrity(), 'limits': vars(self.limits),
            'coverage': self._coverage(), 'io_counters': self._budget.counters()})

    def _require_allocation(self, count):
        if count > self.limits.maximum_selected_bytes:
            _error('budget_exceeded', 'history selected values/coordinates/masks/source spans/chunks exceed allocation cap')

    def _stable(self):
        now = os.fstat(self._raw.fileno())
        if (now.st_dev, now.st_ino, now.st_size, now.st_mtime_ns) != (
                self._opened.st_dev, self._opened.st_ino, self._opened.st_size, self._opened.st_mtime_ns):
            _error('corrupt_result', 'opened scalar history changed during read')

    def _attempt(self, phase, ordinal):
        for span in self._spans[phase]:
            if span['start'] <= ordinal < span['stop']:
                return span['source_attempt']
        _error('corrupt_result', 'selected history row is outside declared source spans')

    def read(self, phase, selection):
        if phase not in PHASES:
            _error('selection_required', 'phase must be scba or outer')
        if self.history_reason:
            _error('unsupported_contract', self.history_reason)
        if not isinstance(selection, Mapping) or set(selection) != {'start', 'stop', 'step', 'columns'}:
            _error('selection_required', 'explicit start/stop/step/columns required')
        description = next(item for item in self._phases if item['phase'] == phase)
        start, stop, step = (selection[key] for key in ('start', 'stop', 'step'))
        if (any(type(value) is not int for value in (start, stop, step))
                or not 0 <= start < stop <= description['total_rows'] or step < 1):
            _error('selection_required', 'invalid history row range; no implicit clipping')
        count = len(range(start, stop, step))
        columns = selection['columns']
        known = {column['name']: column for column in description['columns']}
        if (not isinstance(columns, list) or not columns
                or any(not isinstance(name, str) or name not in known for name in columns)
                or len(set(columns)) != len(columns)):
            _error('selection_required', 'history columns must be distinct declared scalar names')
        if count > self.limits.maximum_rows or len(columns) > self.limits.maximum_columns:
            _error('budget_exceeded', 'history row/value-column cap exceeded')
        coordinate_names = ['sequence', 'iteration'] + (['outer_iteration'] if phase == 'scba' else [])
        names = [*coordinate_names, *columns]
        datasets = {name: self._handle[phase + '/' + name] for name in names}
        previous = start > 0 and step == 1
        # Fixed conservative allowance covers NumPy outputs, Python scalar/list
        # slots, invalid masks, source-attempt/break vectors and parsed spans.
        required = self._span_allocation + count * (len(names) * 64 + 64)
        indices = range(start, stop, step)
        for name, dataset in datasets.items():
            self._chunk_bound(dataset)
            if dataset.chunks:
                touched = {index // dataset.chunks[0] for index in indices}
                if previous and name in coordinate_names:
                    touched.add((start - 1) // dataset.chunks[0])
                required += len(touched) * math.prod(dataset.chunks) * dataset.dtype.itemsize
            if previous and name in coordinate_names:
                required += 64 + dataset.dtype.itemsize
        self._require_allocation(required)
        # The metadata handle and hash share one nofollow descriptor. A path
        # replacement cannot redirect the hash or the subsequent hyperslab.
        self._budget.require(self.owner.size)
        self._budget.phase = 'sha'
        try:
            _verify_stream(self._stream, self.owner.size, self.owner.sha256)
        finally:
            self._budget.phase = 'hdf'
        self._verified = True
        self._stable()
        self._budget.logical_selected_bytes += count * sum(dataset.dtype.itemsize for dataset in datasets.values())
        if previous:
            self._budget.logical_selected_bytes += sum(datasets[name].dtype.itemsize for name in coordinate_names)
        try:
            vectors = {name: datasets[name][slice(start, stop, step)] for name in names}
            prior = {name: int(datasets[name][start - 1]) for name in coordinate_names} if previous else None
        except (OSError, ValueError, TypeError) as error:
            raise ContractError('saved history selection cannot be read', 'corrupt_result') from error
        self._stable()
        coords = {name: [int(value) for value in vectors[name]] for name in coordinate_names}
        attempts = [self._attempt(phase, index) for index in indices]
        for offset in range(count):
            if any(coords[name][offset] < 1 for name in coordinate_names):
                _error('corrupt_result', 'native history coordinates must be positive')
            sequence = coords['sequence'][offset]
            if offset and sequence <= coords['sequence'][offset - 1]:
                _error('corrupt_result', 'selected native sequence is not strictly increasing')
            bounds = self._bounds[phase].get(attempts[offset])
            if bounds and not any(first <= sequence <= last for first, last in bounds):
                _error('identity_mismatch', 'selected native sequence differs from declared source bounds')
        if prior is not None:
            if any(prior[name] < 1 for name in coordinate_names) or prior['sequence'] >= coords['sequence'][0]:
                _error('corrupt_result', 'previous native coordinate differs from selected order')
        output_columns, invalid_rows = [], [False] * count
        for name in columns:
            values, mask = [], []
            for offset, scalar in enumerate(vectors[name]):
                value = scalar.item()
                reason = None
                if isinstance(value, float) and not math.isfinite(value):
                    reason = 'nan' if math.isnan(value) else 'positive_infinity' if value > 0 else 'negative_infinity'
                    value = None
                    invalid_rows[offset] = True
                values.append(value); mask.append(reason)
            output_columns.append({'name': name, 'units': known[name]['units'], 'values': values, 'invalid_mask': mask})
        breaks = [True] * count
        for offset in range(1, count):
            breaks[offset] = (step != 1 or attempts[offset] != attempts[offset - 1]
                or coords['sequence'][offset] != coords['sequence'][offset - 1] + 1
                or coords['iteration'][offset] <= coords['iteration'][offset - 1]
                or (phase == 'scba' and coords['outer_iteration'][offset] != coords['outer_iteration'][offset - 1])
                or invalid_rows[offset] or invalid_rows[offset - 1])
        return self._finish({'schema': 'qcl-negf.saved-history-slice.v1', **self._base(),
            'source_artifact': self._artifact(), 'phase': phase, 'selection': dict(selection),
            'total_rows': description['total_rows'],
            'coordinates': {'sequence': coords['sequence'], 'iteration': coords['iteration'],
                            'outer_iteration': coords.get('outer_iteration'), 'source_attempt': attempts},
            'columns': output_columns, 'break_before': breaks,
            'coverage': {**self._coverage(), 'selected_rows': count,
                         'previous_coordinate_row_checked': previous, 'omitted_rows_disconnect': True},
            'integrity': self._integrity(selected=True), 'limits': vars(self.limits),
            'io_counters': self._budget.counters()})


def describe_saved_diagnostics(commit_path, *, expected_identity, limits=None):
    with DiagnosticReader(commit_path, expected_identity=expected_identity, limits=limits) as reader:
        return reader.describe()


def read_saved_history(commit_path, phase, selection, *, expected_identity, limits=None):
    with DiagnosticReader(commit_path, expected_identity=expected_identity, limits=limits) as reader:
        return reader.read(phase, selection)
