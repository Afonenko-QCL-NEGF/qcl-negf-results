"""One result-v2 source, canonical cursors and bounded, verified native5 reads.

Metadata import and selected byte verification are separate. Nothing here runs a
solver or converts process completion into scientific acceptance.
"""
from __future__ import annotations

from collections import OrderedDict
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, fields
import copy
import hashlib
import io
import math
import os
import sys
from pathlib import Path
from typing import Any, Mapping

import h5py
import numpy as np
from qcl_negf_contracts import schema_validator
from qcl_negf_contracts.artifacts import relative_path
from qcl_negf_contracts.messages import ContractError
from qcl_negf_contracts.results import (validate_manifest, validate_pointer, validate_catalog_page,
    validate_checkpoint_declaration, validate_scientific_record, validate_cursor)
from .commits import decode_json, json_bytes

Document = Mapping[str, Any]
_REF_KEYS = ('path', 'bytes', 'sha256', 'media_type', 'schema')
_TOTALS = ('records', 'scba_rows', 'outer_rows', 'witness_updates', 'last_record')


def _fail(message, code='corrupt_result'):
    raise ContractError(message, code)


def _ref(value):
    return {key: value[key] for key in _REF_KEYS}


def _schema_check(validator, value):
    from jsonschema import ValidationError
    try:
        validator.validate(value)
    except ValidationError as error:
        raise ContractError('invalid result declaration: ' + error.message, 'corrupt_result') from error


def _arrays(value):
    if isinstance(value, dict):
        if value.get('kind') == 'array' and 'file' in value:
            yield value
        for item in value.values():
            yield from _arrays(item)
    elif isinstance(value, list):
        for item in value:
            yield from _arrays(item)


def _metadata_weight(value):
    """Conservative retained Python-object estimate; not a process RSS limit."""
    pending, seen, total = [value], set(), 0
    while pending:
        item = pending.pop()
        if id(item) in seen:
            continue
        seen.add(id(item))
        total += sys.getsizeof(item)
        if isinstance(item, dict):
            pending.extend(item.keys())
            pending.extend(item.values())
        elif isinstance(item, (tuple, list)):
            pending.extend(item)
    return total


class _IOBudget:
    """Physical stream delivery, including integrity and native validation.

    This counts delivered bytes, not disk traffic or logical dataset bytes.
    Checking an EOF-clipped request before the callback prevents overshoot.
    """
    def __init__(self, maximum: int | None):
        if maximum is not None and (type(maximum) is not int or maximum < 1):
            raise ValueError("maximum_io_bytes must be a positive integer")
        self.maximum = maximum
        self.phase = "metadata"
        self.bytes_read_total = 0
        self.sha_bytes = 0
        self.hdf_bytes = 0
        self.metadata_bytes = 0
        self.logical_selected_bytes = 0

    def require(self, count: int) -> None:
        if self.maximum is not None and count > self.maximum - self.bytes_read_total:
            raise ContractError("total physical stream I/O budget exceeded", "budget_exceeded")

    def account(self, count: int) -> None:
        self.bytes_read_total += count
        if self.phase == "sha":
            self.sha_bytes += count
        elif self.phase == "hdf":
            self.hdf_bytes += count
        else:
            self.metadata_bytes += count

    def counters(self) -> dict[str, int | None]:
        return {name: getattr(self, name) for name in (
            "bytes_read_total", "sha_bytes", "hdf_bytes", "metadata_bytes", "logical_selected_bytes")} | {
                "maximum_io_bytes": self.maximum}


class _BudgetStream(io.RawIOBase):
    """A single read-only file object used by SHA and the HDF5 fileobj driver."""
    def __init__(self, raw: Any, budget: _IOBudget):
        super().__init__()
        self._raw, self._budget = raw, budget

    @property
    def name(self) -> str:
        return self._raw.name

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._raw.tell()

    def seek(self, offset: int, whence: int = 0) -> int:
        return self._raw.seek(offset, whence)

    def effective_request(self, size: int) -> int:
        position = self._raw.tell()
        end = self._raw.seek(0, 2)
        self._raw.seek(position)
        remaining = max(0, end - position)
        return remaining if size < 0 else min(size, remaining)

    def read(self, size: int = -1) -> bytes:
        self._budget.require(self.effective_request(size))
        data = self._raw.read(size)
        self._budget.account(len(data))
        return data

    def readinto(self, target: Any) -> int:
        self._budget.require(self.effective_request(len(target)))
        count = self._raw.readinto(target)
        self._budget.account(count)
        return count

    def close(self) -> None:
        if not self.closed:
            self._raw.close()
        super().close()


def _verify_stream(stream: Any, size: int, digest: str) -> None:
    stream.seek(0, 2)
    if stream.tell() != size:
        _fail("state payload byte length differs from its commit")
    stream.seek(0)
    if hashlib.file_digest(stream, "sha256").hexdigest() != digest:
        _fail("state payload checksum differs from its commit")
    stream.seek(0)


def _same_json_value(left: Any, right: Any) -> bool:
    """Compare decoded JSON values without Python's bool/int/float coercion."""
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(
            _same_json_value(left[key], right[key]) for key in left)
    if isinstance(left, list):
        return len(left) == len(right) and all(
            _same_json_value(a, b) for a, b in zip(left, right))
    return left == right


@dataclass(frozen=True)
class ResultReadLimits:
    maximum_metadata_bytes: int = 16 * 1024 * 1024
    maximum_io_bytes: int = 256 * 1024 * 1024
    maximum_catalog_pages: int = 4096
    maximum_catalog_entries: int = 65536
    maximum_catalog_bytes: int = 64 * 1024 * 1024
    maximum_open_files: int = 8
    maximum_record_bytes: int = 1024 * 1024
    maximum_batch_bytes: int = 4 * 1024 * 1024
    maximum_batch_records: int = 256
    maximum_hdf_objects: int = 65536
    maximum_hdf_depth: int = 64
    maximum_array_bytes: int = 1024 * 1024

    def __post_init__(self):
        if any(type(getattr(self, item.name)) is not int or getattr(self, item.name) < 1
               for item in fields(self)):
            raise ValueError('read limits require positive integers')


@dataclass(frozen=True)
class ResultDescription:
    state: Document
    invocation: Document
    history: Document
    lineage: Document
    status: Document
    closed: bool
    capabilities: Document
    integrity: Document
    io_counters: Mapping[str, int | None]


@dataclass(frozen=True)
class HistoryBatch:
    source: Document
    records: tuple[Document, ...]
    cursor: Document
    at_cutoff: bool
    io_counters: Mapping[str, int | None]


@dataclass(frozen=True)
class ArrayDescription:
    declaration: Document
    logical_shape: tuple[int, ...]
    stored_shape: tuple[int, ...]
    eltype: str
    element_bytes: int
    order: str
    source: Document


@dataclass(frozen=True)
class ArrayBlock:
    values: Any
    selection: tuple[int | slice, ...]
    declaration: Document
    source: Document
    io_counters: Mapping[str, int | None]


def _internal_storage(handle, *, maximum_objects=65536, maximum_depth=64):
    """Bounded ownership walk, without reading dataset values."""
    seen, pending = set(), [(handle, 0)]
    while pending:
        group, depth = pending.pop()
        address = h5py.h5o.get_info(group.id).addr
        if address in seen:
            continue
        if depth > maximum_depth or len(seen) >= maximum_objects:
            _fail('native object inventory exceeds read limit', 'budget_exceeded')
        seen.add(address)
        if len(group) > maximum_objects - len(seen):
            _fail('native group inventory exceeds read limit', 'budget_exceeded')
        for name in group:
            if not isinstance(group.get(name, getlink=True), h5py.HardLink):
                _fail('native source rejects soft/external links')
            child = group[name]
            if isinstance(child, h5py.Group):
                pending.append((child, depth + 1))
            elif isinstance(child, h5py.Dataset):
                if child.external or child.is_virtual:
                    _fail('native source rejects external/virtual storage')
                address = h5py.h5o.get_info(child.id).addr
                seen.add(address)
                if len(seen) > maximum_objects:
                    _fail('native object inventory exceeds read limit', 'budget_exceeded')


class StateReader:
    """One explicitly selected immutable result source; no legacy format branch.

    Each operation has a physical stream budget. Open handles reuse SHA proof;
    cache eviction/reopen is a new verification boundary. Returned JSON is copied.
    """
    def __init__(self, artifact_root: str | Path, *, state_ref: Document,
                 expected_identity: Document, limits: ResultReadLimits | None = None):
        self._initialize(artifact_root, state_ref=state_ref, expected_identity=expected_identity, limits=limits)

    def _initialize(self, artifact_root, *, state_ref, expected_identity, limits=None,
                    verified_manifest=None, budget=None, lineage_depth=0):
        self.limits = limits or ResultReadLimits()
        if not isinstance(self.limits, ResultReadLimits):
            raise TypeError('limits must be ResultReadLimits')
        self.root = Path(os.path.abspath(artifact_root))
        self._root_stamp = self._root_identity()
        self._closed = False
        self._handles = OrderedDict()
        self._pages, self._index, self._known_states = [], {}, {}
        self._catalog_by_hash, self._published_paths, self._catalog_by_leaf = {}, set(), {}
        self._page_proofs = {}
        self._parent_anchor_cache = None
        self._catalog_ready, self._catalog_size = False, 0
        self._index_weight = self._known_weight = self._pending_state_weight = 0
        self._checkpoint = None
        self._declarations, self._verified_refs = {}, {}
        self._pointer_path, self._pointer_raw = None, None
        self._shared_budget = budget
        self._lineage_depth = lineage_depth
        if lineage_depth > self.limits.maximum_hdf_depth:
            _fail('lineage proof depth exceeds metadata limit', 'budget_exceeded')
        self._operation()
        self._state_ref = copy.deepcopy(dict(state_ref))
        _schema_check(schema_validator('result-common.schema.json').evolve(schema={
            '$ref': '#/$defs/StateRef', '$defs': schema_validator('result-common.schema.json').schema['$defs']
        }), self._state_ref)
        if not _same_json_value(self._state_ref['identity'], dict(expected_identity)):
            _fail('complete selected identity differs from StateRef', 'identity_mismatch')
        self._manifest = validate_manifest(verified_manifest if verified_manifest is not None
                                           else self._json(self._state_ref['manifest']))
        for key in ('identity', 'state_id', 'state_sequence'):
            if self._manifest[key] != self._state_ref[key]:
                _fail('manifest differs from selected StateRef')
        self._register_manifest()
        self._remember_state()

    def _root_identity(self):
        from ._capture_io import regular
        # Opening a known child is unnecessary; nofollow traversal is performed
        # by regular for every actual file. Reject symlink root itself as well.
        current = self.root
        for part in (current, *current.parents):
            if part.is_symlink():
                _fail('artifact root traverses a symbolic link')
        stat = self.root.stat()
        return stat.st_dev, stat.st_ino

    def _operation(self):
        if getattr(self, '_closed', False):
            raise ValueError('state reader is closed')
        if hasattr(self, '_root_stamp') and self._root_identity() != self._root_stamp:
            _fail('artifact root replaced', 'source_changed')
        self._budget = getattr(self, '_shared_budget', None) or _IOBudget(self.limits.maximum_io_bytes)
        for item in getattr(self, '_handles', {}).values():
            item['stream']._budget = self._budget

    @contextmanager
    def _consumer_operation(self, maximum_io_bytes=None, *, maximum_owner_bytes=None):
        """One allowance through nested checkpoint/catalog/ancestor operations."""
        self._operation()
        budget, prior = self._budget, self._shared_budget
        prior_owner = getattr(self, '_consumer_owner_limit', None)
        self._consumer_owner_limit = (maximum_owner_bytes if prior_owner is None else prior_owner
            if maximum_owner_bytes is None else min(prior_owner,maximum_owner_bytes))
        maximum = budget.maximum
        if maximum_io_bytes is not None:
            if type(maximum_io_bytes) is not int or maximum_io_bytes < 1:
                raise ValueError('positive consumer I/O allowance required')
            bound = budget.bytes_read_total + maximum_io_bytes
            budget.maximum = bound if maximum is None else min(maximum, bound)
        self._shared_budget = budget
        try:
            yield budget
        finally:
            self._shared_budget = prior
            self._consumer_owner_limit = prior_owner
            budget.maximum = maximum

    @classmethod
    def from_pointer(cls, artifact_root, *, expected_identity, pointer, limits=None):
        if pointer not in ('live', 'recovery'):
            raise ValueError('pointer must be live or recovery')
        selected = dict(expected_identity)
        path = ('executions/' + selected['execution_id'] + '/' + selected['point_id'] +
                '/attempt-' + selected['attempt'] + ('/current.json' if pointer == 'live'
                else '/recovery/current.json'))
        # Validate selection/path through canonical pure schema before file access.
        common = schema_validator('result-common.schema.json')
        _schema_check(common.evolve(schema={'$ref': '#/$defs/Identity', '$defs': common.schema['$defs']}), selected)
        from ._capture_io import beneath, small_json
        cap = limits or ResultReadLimits()
        budget = _IOBudget(cap.maximum_io_bytes)
        value, raw = small_json(beneath(Path(artifact_root), path), cap.maximum_metadata_bytes, budget=budget)
        value = validate_pointer(value)
        if value['identity'] != selected:
            _fail('pointer differs from selected identity', 'identity_mismatch')
        # Capture the hashed manifest once to derive its StateRef; the private
        # initializer consumes that exact verified document and physical budget.
        temporary = cls.__new__(cls)
        temporary.root, temporary.limits, temporary._budget = Path(os.path.abspath(artifact_root)), cap, budget
        manifest = validate_manifest(temporary._json(value['manifest']))
        ref = {key: manifest[key] for key in ('identity', 'state_id', 'state_sequence')}
        ref['manifest'] = value['manifest']
        result = cls.__new__(cls)
        result._initialize(artifact_root, state_ref=ref, expected_identity=selected, limits=cap,
                           verified_manifest=manifest, budget=budget)
        result._shared_budget = None
        if manifest['generation'] != value['generation']:
            result.close()
            _fail('pointer generation differs from manifest')
        result._pointer_path, result._pointer_raw = path, raw
        return result

    def __enter__(self):
        return self

    def __exit__(self, *args):
        try:
            self.close()
        except BaseException as cleanup_error:
            if len(args) < 2 or args[1] is None:
                raise
            args[1].add_note('native owner cleanup also raised ' + type(cleanup_error).__name__)

    def close(self):
        # Mark closed and detach first: cancellation in one stack must not leak
        # another verified owner or make a second close repeat its callbacks.
        owners = tuple(self._handles.values())
        self._handles.clear()
        self._closed = True
        first = None
        for value in owners:
            try:
                value['stack'].close()
            except BaseException as error:
                if first is None:
                    first = error
                else:
                    first.add_note('another native owner cleanup raised ' + type(error).__name__)
        if first is not None:
            raise first

    def _json(self, reference):
        from ._capture_io import beneath, regular, stamp
        if int(reference['bytes']) > self.limits.maximum_metadata_bytes:
            _fail('metadata exceeds read limit', 'budget_exceeded')
        with regular(beneath(self.root, reference['path']), size=int(reference['bytes'])) as raw:
            before = stamp(raw)
            stream = _BudgetStream(raw, self._budget)
            phase, self._budget.phase = self._budget.phase, 'metadata'
            try:
                payload = stream.read(int(reference['bytes']))
            finally:
                self._budget.phase = phase
            if stamp(raw) != before or hashlib.sha256(payload).hexdigest() != reference['sha256']:
                _fail('metadata differs from immutable reference')
        return decode_json(payload, reference['path'])

    def _register(self, reference, *, table=None):
        target = self._index if table is None else table
        relative_path(reference['path'])
        simple = _ref(reference)
        old = target.get(simple['path'])
        if old is not None and _ref(old) != simple:
            _fail('immutable path rebound to different bytes')
        value = copy.deepcopy(dict(reference))
        if target is self._index:
            changed_weight = _metadata_weight(value) - (_metadata_weight(old) if old is not None else 0)
            if (self._index_weight + changed_weight + self._known_weight + self._pending_state_weight
                    + self._catalog_size > self.limits.maximum_catalog_bytes):
                _fail('retained metadata context exceeds limit', 'budget_exceeded')
            self._index_weight += changed_weight
        target[simple['path']] = value
        if len(target) > self.limits.maximum_catalog_entries:
            _fail('imported reference inventory exceeds limit', 'budget_exceeded')

    def _register_manifest(self):
        self._register(self._state_ref['manifest'])
        for row in self._manifest['artifacts']:
            self._register(row)

    def _remember_state(self):
        digest = self._state_ref['manifest']['sha256']
        if digest not in self._known_states:
            weight = _metadata_weight(self._state_ref) + _metadata_weight(self._manifest)
            if self._known_weight + weight + self._index_weight + self._catalog_size > self.limits.maximum_catalog_bytes:
                _fail('retained metadata context exceeds limit', 'budget_exceeded')
            self._known_states[digest] = (self._state_ref, self._manifest)
            self._known_weight += weight
        self._pending_state_weight = 0

    @property
    def io_counters(self):
        return self._budget.counters()

    @property
    def source(self):
        ready = self._catalog_ready
        def cap(value, reason):
            return dict(available=bool(value), reason=None if value else reason)
        checkpoint = self._checkpoint is not None
        history = int(self._manifest['history']['records']) > 0
        capabilities = dict(history_read=cap(ready, 'catalog_import_required'),
            array_slice=cap(bool(self._declarations), 'verified_array_declaration_required'),
            exact_checkpoint=cap(checkpoint, 'checkpoint_verification_required' if self._manifest['checkpoint'] else 'selected_state_has_no_checkpoint'),
            exact_restore=cap(False, 'core_owned'), saved_views=cap(False, 'explicit_final_observations_ref_required'),
            stored_observables=cap(False, 'explicit_final_observations_ref_required'),
            stored_final_diagnostics=cap(False, 'explicit_final_diagnostics_ref_required'),
            streaming_export=cap(history or self._manifest['checkpoint'], 'no_selected_scientific_data'),
            science_export=cap(history, 'history_or_explicit_analysis_ref_required'),
            full_state_export=cap(self._manifest['checkpoint'] is not None, 'selected_state_has_no_checkpoint'),
            scalar_derive=cap(ready, 'catalog_import_required'))
        return ResultDescription(copy.deepcopy(self._state_ref), copy.deepcopy(self._manifest['invocation']),
            copy.deepcopy(self._manifest['history']), copy.deepcopy(self._manifest['lineage']),
            copy.deepcopy(self._manifest['status']), self._manifest['closed'], capabilities,
            dict(manifest_verified=True, catalog_verified=ready,
                 selected_payloads_verified=copy.deepcopy(list(self._verified_refs.values())), scope='selected_bytes'),
            self.io_counters)

    @staticmethod
    def _pin(page, reference):
        return dict(head=_ref(reference), entries=page['entries'], **page['totals'])

    def import_history(self):
        self._operation()
        self._import_history()
        return self.source

    def _import_history(self):
        pin = self._manifest['history']
        known = self._catalog_by_hash
        ref, pending, seen = pin['head'], [], set()
        size, delta, added_paths, delta_weights, index_change = self._catalog_size, {}, set(), {}, 0
        def register(reference):
            nonlocal index_change
            old = delta.get(reference['path'], self._index.get(reference['path']))
            if old is not None and _ref(old) != _ref(reference):
                _fail('immutable path rebound to different bytes')
            self._register(reference, table=delta)
            existing = self._index.get(reference['path'])
            changed = _metadata_weight(delta[reference['path']]) - (_metadata_weight(existing) if existing is not None else 0)
            index_change += changed - delta_weights.get(reference['path'], 0)
            delta_weights[reference['path']] = changed
            if reference['path'] not in self._index:
                added_paths.add(reference['path'])
            if len(self._index) + len(added_paths) > self.limits.maximum_catalog_entries:
                _fail('imported reference inventory exceeds limit', 'budget_exceeded')
            if (size + self._index_weight + index_change + self._known_weight
                    + self._pending_state_weight > self.limits.maximum_catalog_bytes):
                _fail('retained catalog/context metadata exceeds limit', 'budget_exceeded')
        anchors = self._parent_anchors()
        for reference in anchors.values():
            register(dict(reference, role='science.history', dependencies=[]))
        while ref is not None and ref['sha256'] not in known:
            if ref['path'] in seen or len(pending) + len(self._pages) >= self.limits.maximum_catalog_pages:
                _fail('catalog cycle or page limit', 'budget_exceeded')
            seen.add(ref['path'])
            page = self._json(ref)
            size += _metadata_weight(page) + _metadata_weight(ref)
            if size + self._index_weight + self._known_weight + self._pending_state_weight > self.limits.maximum_catalog_bytes:
                _fail('decoded catalog inventory exceeds limit', 'budget_exceeded')
            pending.append((copy.deepcopy(ref), page))
            # Syntax is checked before following an untrusted predecessor.
            _schema_check(schema_validator('scientific-catalog.schema.json'), page)
            ref = page['previous']['head'] if page['previous'] else None
        if self._pages and pending and ref != self._pages[-1][0]:
            _fail('new history does not extend imported head', 'source_changed')
        prior = known.get(ref['sha256']) if ref else None
        previous_pin = self._pin(prior[1], prior[0]) if prior else None
        prior_page = prior[1] if prior else None
        prior_ref = prior[0] if prior else None
        additions, new_paths_published = [], set()
        for reference, raw_page in reversed(pending):
            page = validate_catalog_page(raw_page, previous=previous_pin,
                previous_page=prior_page, previous_ref=prior_ref)
            if page['identity'] != self._manifest['identity'] or page['invocation'] != self._manifest['invocation']:
                _fail('catalog identity or epoch differs from manifest')
            rows = page['artifacts']
            if len(rows) not in (1, 2):
                _fail('native5 batch requires optional array owner then one record leaf')
            new_paths = {row['path'] for row in rows}
            if new_paths & self._published_paths or new_paths & new_paths_published:
                _fail('catalog republishes a nonadjacent immutable path')
            if new_paths & anchors.keys():
                _fail('catalog republishes an ancestor append dependency')
            if len(rows) == 2 and (rows[0]['dependencies'] or rows[0]['path'] not in rows[-1]['dependencies']):
                _fail('ordered native5 array dependency differs')
            for row in rows:
                register(row)
            for row in rows:
                if any(path == row['path'] or not (path in self._published_paths
                        or path in new_paths_published or path in new_paths or path in anchors)
                        for path in row['dependencies']):
                    _fail('catalog dependency absent, future or self-referential')
                if row is rows[0] and len(rows) == 2 and row['dependencies']:
                    _fail('array owner cannot depend on another leaf')
            register(reference)
            new_paths_published.update(new_paths)
            additions.append((reference, page))
            prior_ref, prior_page, previous_pin = reference, page, self._pin(page, reference)
        actual_pin = previous_pin if pending else (self._pin(*reversed(self._pages[-1])) if self._pages else None)
        if pin['head'] is None:
            if self._pages:
                _fail('history disappeared', 'source_changed')
        elif actual_pin != pin:
            _fail('catalog head counters differ from manifest pin')
        first_new = len(self._pages)
        self._pages.extend(additions)
        for offset, (reference, page) in enumerate(additions, first_new):
            leaf = page['artifacts'][-1]
            self._catalog_by_leaf[(leaf['path'], leaf['sha256'])] = (offset, reference, page)
        self._index.update(delta)
        self._index_weight += index_change
        self._catalog_by_hash.update({reference['sha256']: (reference, page) for reference, page in additions})
        self._published_paths.update(new_paths_published)
        self._catalog_size, self._catalog_ready = size, True

    def _check_hdf_owner(self, key, item):
        """Check the actual retained FD and its immutable root-relative binding."""
        from ._capture_io import stamp
        from .commits import safe_path
        try:
            if self._root_identity() != self._root_stamp:
                _fail('artifact root replaced', 'source_changed')
            current = safe_path(self.root, key[0]).stat(follow_symlinks=False)
            value = (current.st_dev, current.st_ino, current.st_size, current.st_mtime_ns, current.st_ctime_ns)
            opened = stamp(item['raw'], include_change=True)
        except (OSError, ValueError, ContractError) as error:
            raise ContractError('trusted payload binding changed', 'source_changed') from error
        if value != item['stamp'] or opened != item['stamp']:
            _fail('trusted payload source changed', 'source_changed')

    def _open_hdf(self, reference, *, identity=None, invocation=None, representation=None):
        from ._capture_io import regular, stamp
        from .native import validate_native_handle
        owner_limit = getattr(self, '_consumer_owner_limit', None)
        if owner_limit is not None and int(reference['bytes']) > owner_limit:
            _fail('selected native owner exceeds consumer byte limit', 'budget_exceeded')
        key = (reference['path'], reference['sha256'])
        from .commits import safe_path
        path = safe_path(self.root, reference['path'])
        if key in self._handles:
            item = self._handles[key]
            self._check_hdf_owner(key, item)
            self._handles.move_to_end(key)
        else:
            stack = ExitStack()
            try:
                raw = stack.enter_context(regular(path, size=int(reference['bytes'])))
                before = stamp(raw, include_change=True)
                stream = _BudgetStream(raw, self._budget)
                self._budget.phase = 'sha'
                _verify_stream(stream, int(reference['bytes']), reference['sha256'])
                self._budget.phase = 'hdf'
                handle = stack.enter_context(h5py.File(stream, 'r', rdcc_nbytes=self.limits.maximum_array_bytes))
                _internal_storage(handle, maximum_objects=self.limits.maximum_hdf_objects,
                                  maximum_depth=self.limits.maximum_hdf_depth)
                validate_native_handle(handle, reference.get('role'), reference['schema'],
                                       maximum_metadata_bytes=self.limits.maximum_metadata_bytes)
                if stamp(raw, include_change=True) != before:
                    _fail('payload changed while being verified')
                item = dict(stack=stack, raw=raw, stream=stream, handle=handle, stamp=before)
                self._check_hdf_owner(key, item)
                self._handles[key] = item
                self._verified_refs[reference['path']] = _ref(reference)
                while len(self._handles) > self.limits.maximum_open_files:
                    _, old = self._handles.popitem(last=False)
                    old['stack'].close()
                    self._verified_refs = {path: value for path, value in self._verified_refs.items()
                        if any(key[0] == path for key in self._handles)}
            except BaseException as error:
                self._handles.pop(key, None)
                self._verified_refs.pop(reference['path'], None)
                try:
                    stack.close()
                except BaseException as cleanup_error:
                    error.add_note('temporary native owner cleanup also raised ' + type(cleanup_error).__name__)
                raise
        self._budget.phase = 'hdf'
        handle = item['handle']
        if owner_limit is not None:
            _internal_storage(handle, maximum_objects=self.limits.maximum_hdf_objects,
                              maximum_depth=self.limits.maximum_hdf_depth)
        from .native import native_metadata
        native_metadata(handle, identity=identity, invocation=invocation,
                        representation=representation, maximum_bytes=self.limits.maximum_metadata_bytes)
        return handle

    def _record(self, leaf, offset, *, identity, invocation, page=None):
        handle = self._open_hdf(leaf, identity=identity, invocation=invocation, representation='record_leaf')
        n = self._record_count(handle)
        if type(offset) is not int or not 0 <= offset < n:
            _fail('record offset outside physical record leaf')
        start, stop = (int(x) for x in handle['records/offsets'][offset:offset+2])
        if stop - start > self.limits.maximum_record_bytes:
            _fail('record JSON exceeds read limit', 'budget_exceeded')
        payload = handle['records/json_utf8'][start:stop].tobytes()
        record = decode_json(payload, 'scientific record')
        ordinal = int(handle['records/stream_ordinal'][offset])
        if record.get('identity') != identity or record.get('invocation') != invocation or record.get('stream_ordinal') != str(ordinal):
            _fail('record identity/epoch/ordinal differs from native coordinates')
        if page is not None and not int(page['range']['first_record']) <= ordinal <= int(page['range']['last_record']):
            _fail('record ordinal outside containing page')
        return record

    def _record_count(self, handle):
        from .native import record_table
        for item in self._handles.values():
            if item['handle'] is handle:
                # A warm coordinate proof authenticates content, not admission
                # under a later consumer's tighter decoded-chunk allowance.
                for name in ('offsets','json_utf8','stream_ordinal'):
                    dataset = handle['records/'+name]
                    if dataset.chunks and math.prod(dataset.chunks)*dataset.dtype.itemsize > self.limits.maximum_record_bytes:
                        _fail('native record chunk decode buffer exceeds limit', 'budget_exceeded')
                if 'record_count' not in item:
                    item['record_count'] = record_table(handle,
                        maximum_block_bytes=self.limits.maximum_record_bytes)
                return item['record_count']
        _fail('record handle is no longer trusted', 'source_changed')

    def _validate_record(self, record, leaf, offset, *, manifest=None):
        source = manifest or self._manifest
        target = record.get('target')
        kwargs = {}
        parent_reader = None
        if target is not None and target.get('scope') == 'parent_checkpoint':
            parent_ref = source['lineage']['parent_state']
            if parent_ref is None:
                _fail('parent update has no actual lineage parent')
            parent_reader = self._parent_reader(parent_ref)
            kwargs = dict(parent_checkpoint=parent_reader._load_checkpoint(),
                parent_state=parent_ref, parent_manifest_ref=parent_ref['manifest'])
        try:
            checked = validate_scientific_record(record, **kwargs)
            core = checked['core']['fields']
            stage, row = core['stage']['value'], core['row']
            if stage not in ('scba', 'outer') or row['type'] != ('SCBAIteration' if stage == 'scba' else 'OuterIteration'):
                _fail('scientific stage and typed row differ')
            counter = row['fields']['ν' if stage == 'scba' else 'μ']
            expected = core['inner_iteration'] if stage == 'scba' else core['outer_iteration']
            if (counter != expected or stage == 'outer' and core['inner_iteration'] is not None
                    or int(core['outer_iteration']['decimal']) < 1
                    or stage == 'scba' and int(expected['decimal']) < 1
                    or int(core['sequence']['decimal']) < 1
                    or int(core['domain_revision']['decimal']) < 0):
                _fail('scientific row counters/domain/sequence differ from snapshot')
            owner = self._index.get(leaf['path'])
            dependencies = set(owner.get('dependencies', ())) if owner is not None else set()
            for declaration in _arrays(checked):
                reference = declaration['file']
                if (reference['path'] not in dependencies or reference['path'] not in self._index
                        or _ref(self._index[reference['path']]) != reference):
                    _fail('record array owner absent from verified leaf dependency closure')
            if target is not None:
                target_leaf = _ref(leaf) if target['scope'] == 'same_leaf' else target['leaf']
                target_offset = int(target['row_offset'])
                if target['scope'] == 'same_leaf' and target_offset >= offset:
                    _fail('same-leaf update does not target an earlier physical append')
                if target['scope'] != 'parent_checkpoint':
                    target_owner = self._index.get(target_leaf['path'])
                    target_context = self._catalog_by_leaf.get((target_leaf['path'], target_leaf['sha256']))
                    if (target_owner is None or target_context is None
                            or _ref(target_owner) != target_leaf
                            or _ref(target_context[2]['artifacts'][-1]) != target_leaf):
                        _fail('update target FileRef differs from frozen catalog record leaf')
                    if target['scope'] == 'sealed_leaf':
                        current_context = self._catalog_by_leaf.get((leaf['path'], leaf['sha256']))
                        if (current_context is None or target_context[0] >= current_context[0]
                                or target_leaf['path'] not in current_context[2]['artifacts'][-1]['dependencies']):
                            _fail('sealed update target is not an earlier declared catalog dependency')
                    target_leaf = _ref(target_owner)
                original = self._record(target_leaf, target_offset, identity=target['source_identity'],
                    invocation=dict(source_epoch=target['source_epoch']))
                validate_scientific_record(original)
                if (original['target'] is not None or original['core']['fields']['action']['value'] != 'append'
                        or original['row_key'] != checked['row_key']
                        or original['stream_ordinal'] != target['append_ordinal']):
                    _fail('update target differs from original append action/key/ordinal')
            return checked
        finally:
            if parent_reader is not None:
                parent_reader.close()

    def _cursor(self, leaf=None, offset=0, ordinal=0):
        return dict(schema='qcl-negf.result-cursor.v1', contract_set='qcl-negf.results.v2',
            identity=copy.deepcopy(self._manifest['identity']), invocation=copy.deepcopy(self._manifest['invocation']),
            manifest=copy.deepcopy(self._state_ref['manifest']), history=copy.deepcopy(self._manifest['history']),
            leaf=None if leaf is None else _ref(leaf), row_offset=str(offset), last_record=str(ordinal))

    def _cursor_page(self, cursor):
        if cursor['leaf'] is None:
            return None
        context = self._catalog_by_leaf.get((cursor['leaf']['path'], cursor['leaf']['sha256']))
        if context is not None:
            _, reference, page = context
            if _ref(page['artifacts'][-1]) == cursor['leaf']:
                return dict(reference=reference, document=page)
        _fail('cursor record leaf is not reachable from pinned source')

    def _check_cursor(self, cursor):
        _schema_check(schema_validator('result-cursor.schema.json'), cursor)
        context = self._cursor_page(cursor)
        validate_cursor(cursor, manifest=self._manifest, manifest_ref=self._state_ref['manifest'], catalog_page=context)
        if cursor['leaf'] is not None:
            row = self._record(cursor['leaf'], int(cursor['row_offset']),
                identity=self._manifest['identity'], invocation=self._manifest['invocation'], page=context['document'])
            self._validate_record(row, cursor['leaf'], int(cursor['row_offset']))
            if row['stream_ordinal'] != cursor['last_record']:
                _fail('cursor physical position differs from scientific ordinal')
            return row
        return None

    def history_batch(self, cursor=None, *, maximum_records=256):
        self._operation()
        if not self._catalog_ready:
            _fail('explicit catalog import required', 'capability_unavailable')
        if type(maximum_records) is not int or not 1 <= maximum_records <= self.limits.maximum_batch_records:
            raise ValueError('maximum_records exceeds configured batch limit')
        current = self._cursor() if cursor is None else copy.deepcopy(dict(cursor))
        previous_row = self._check_cursor(current)
        previous_sequence = int(previous_row['core']['fields']['sequence']['decimal']) if previous_row else 0
        output, consumed, last = [], 0, int(current['last_record'])
        first_page = (self._catalog_by_leaf[(current['leaf']['path'], current['leaf']['sha256'])][0]
                      if current['leaf'] is not None else 0)
        for page_index in range(first_page, len(self._pages)):
            _, page = self._pages[page_index]
            if int(page['range']['last_record']) <= last:
                continue
            leaf = page['artifacts'][-1]
            start = int(current['row_offset']) + 1 if current['leaf'] == _ref(leaf) else 0
            handle = self._open_hdf(leaf, identity=self._manifest['identity'],
                invocation=self._manifest['invocation'], representation='record_leaf')
            count = self._record_count(handle)
            expected = int(page['range']['last_record']) - int(page['range']['first_record']) + 1
            if count != expected:
                _fail('physical record count differs from catalog range')
            for offset in range(start, count):
                raw = self._record(leaf, offset, identity=self._manifest['identity'],
                    invocation=self._manifest['invocation'], page=page)
                row = self._validate_record(raw, leaf, offset)
                if int(row['stream_ordinal']) != last + 1:
                    _fail('scientific record ordinal gap or replay')
                sequence = int(row['core']['fields']['sequence']['decimal'])
                if sequence <= previous_sequence:
                    _fail('Core delivery sequence replay or regression')
                previous_sequence = sequence
                proof = self._page_proofs.setdefault(leaf['sha256'],
                    dict(next_offset=0, scba_rows=0, outer_rows=0, witness_updates=0))
                if proof.get('invalid'):
                    _fail('actual scientific record categories differ from catalog totals')
                if offset == proof['next_offset']:
                    category = ('witness_updates' if row['core']['fields']['action']['value'] == 'witness_update'
                        else 'scba_rows' if row['core']['fields']['stage']['value'] == 'scba' else 'outer_rows')
                    proof[category] += 1
                    proof['next_offset'] += 1
                    if proof['next_offset'] == count:
                        for category in ('scba_rows','outer_rows','witness_updates'):
                            before = int(page['previous'][category]) if page['previous'] else 0
                            if proof[category] != int(page['totals'][category]) - before:
                                proof['invalid'] = True
                                _fail('actual scientific record categories differ from catalog totals')
                consumed += len(json_bytes(row))
                if consumed > self.limits.maximum_batch_bytes:
                    _fail('batch JSON exceeds byte limit', 'budget_exceeded')
                output.append(copy.deepcopy(row))
                last += 1
                current = self._cursor(leaf, offset, last)
                if len(output) == maximum_records:
                    break
            if len(output) == maximum_records:
                break
        declarations = {json_bytes(item): copy.deepcopy(item) for row in output for item in _arrays(row)}
        if self._checkpoint is not None:
            declarations.update({json_bytes(item): copy.deepcopy(item) for item in _arrays(self._checkpoint)})
        self._declarations = declarations
        return HistoryBatch(copy.deepcopy(self._state_ref), tuple(output), current,
            last == int(self._manifest['history']['last_record']), self.io_counters)

    def checkpoint(self):
        self._operation()
        return copy.deepcopy(self._load_checkpoint())

    def _load_checkpoint(self):
        if self._checkpoint is not None:
            return self._checkpoint
        reference = self._manifest['checkpoint']
        if reference is None:
            _fail('selected state has no checkpoint', 'capability_unavailable')
        checkpoint = validate_checkpoint_declaration(self._json(reference), manifest=self._manifest)
        self._import_history()
        parent_reader, parent_checkpoint = None, None
        parent_ref = self._manifest['lineage']['parent_state']
        if parent_ref is not None:
            parent_reader = self._parent_reader(parent_ref)
            try:
                parent_checkpoint = parent_reader._load_checkpoint()
            finally:
                parent_reader.close()
        for ref in checkpoint['payloads']:
            if ref['path'] not in self._index or _ref(self._index[ref['path']]) != ref:
                _fail('checkpoint payload differs from manifest dependencies')
        for target in checkpoint['witness_targets']:
            locator = target['locator']
            local = (locator['source_identity'] == self._manifest['identity']
                     and locator['source_epoch'] == self._manifest['invocation']['source_epoch'])
            if local:
                if not any(_ref(page['artifacts'][-1]) == locator['leaf'] for _, page in self._pages):
                    _fail('checkpoint target is not in frozen scientific catalog')
            elif parent_checkpoint is None or target not in parent_checkpoint['witness_targets']:
                _fail('ancestor checkpoint target is not retained by actual immediate parent')
            row = self._record(locator['leaf'], int(locator['row_offset']), identity=locator['source_identity'],
                invocation=dict(source_epoch=locator['source_epoch']))
            validate_scientific_record(row)
            if (row['target'] is not None or row['row_key'] != target['row_key']
                    or row['stream_ordinal'] != locator['append_ordinal']):
                _fail('checkpoint witness locator differs from original append')
        self._checkpoint = checkpoint
        self._declarations.update({json_bytes(item): copy.deepcopy(item) for item in _arrays(checkpoint)})
        return checkpoint

    def _parent_reader(self, reference):
        parent = StateReader.__new__(StateReader)
        parent._initialize(self.root, state_ref=reference, expected_identity=reference['identity'],
            limits=self.limits, budget=self._budget, lineage_depth=self._lineage_depth + 1)
        parent._consumer_owner_limit = getattr(self, '_consumer_owner_limit', None)
        return parent

    def _parent_anchors(self):
        """Authenticated parent metadata only; no HDF or prefix payload scan."""
        if self._parent_anchor_cache is not None:
            return self._parent_anchor_cache
        reference = self._manifest['lineage']['parent_state']
        anchors = {}
        if reference is not None:
            parent = self._parent_reader(reference)
            try:
                checkpoint_ref = parent._manifest['checkpoint']
                if checkpoint_ref is None:
                    _fail('lineage parent has no exact checkpoint declaration')
                checkpoint = validate_checkpoint_declaration(parent._json(checkpoint_ref), manifest=parent._manifest)
                for target in checkpoint['witness_targets']:
                    leaf = target['locator']['leaf']
                    prior = anchors.get(leaf['path'])
                    if prior is not None and prior != leaf:
                        _fail('parent anchors rebound an immutable path')
                    anchors[leaf['path']] = copy.deepcopy(leaf)
            finally:
                parent.close()
        self._parent_anchor_cache = anchors
        return anchors

    def _array(self, declaration):
        key = json_bytes(declaration)
        if key not in self._declarations:
            _fail('array descriptor not from retained verified state/records', 'source_changed')
        declaration = self._declarations[key]
        reference = declaration['file']
        if reference['path'] not in self._index or _ref(self._index[reference['path']]) != reference:
            _fail('array owner not in verified source dependencies')
        handle = self._open_hdf(self._index[reference['path']], identity=self._manifest['identity'],
            invocation=self._manifest['invocation'], representation='array_payload')
        from .native import validate_native_array
        dataset = validate_native_array(handle, declaration)
        return declaration, dataset

    def describe_array(self, declaration):
        self._operation()
        wire, dataset = self._array(declaration)
        return ArrayDescription(copy.deepcopy(wire), tuple(int(v) for v in wire['shape']),
            tuple(dataset.shape), wire['eltype'], dataset.dtype.itemsize, 'F', copy.deepcopy(self._state_ref))

    def read_array(self, declaration, selection, *, maximum_bytes=1024*1024):
        self._operation()
        wire, dataset = self._array(declaration)
        return self._read_selected_array(wire, dataset, selection, maximum_bytes=maximum_bytes)

    def _read_selected_array(self, wire, dataset, selection, *, maximum_bytes):
        """Shared bounded selection; caller authenticates the descriptor owner."""
        if type(maximum_bytes) is not int or not 0 < maximum_bytes <= self.limits.maximum_array_bytes:
            raise ValueError('maximum_bytes exceeds configured array read limit')
        if not isinstance(selection, tuple) or len(selection) != dataset.ndim:
            raise ValueError('one explicit selector required per logical axis')
        normalized, counts = [], []
        for item, size in zip(selection, dataset.shape):
            if type(item) is int:
                if not 0 <= item < size:
                    raise IndexError('array index outside logical shape')
                normalized.append(item)
            elif isinstance(item, slice):
                if item.step is not None and item.step <= 0:
                    raise ValueError('array selection requires positive step')
                a, b, step = item.indices(size)
                normalized.append(slice(a, b, step))
                counts.append(len(range(a, b, step)))
            else:
                raise ValueError('array selectors must be integers or slices')
        required = math.prod(counts) * dataset.dtype.itemsize
        if required > maximum_bytes:
            _fail('selected array buffer exceeds limit', 'budget_exceeded')
        if dataset.chunks and math.prod(dataset.chunks)*dataset.dtype.itemsize > self.limits.maximum_array_bytes:
            _fail('native array chunk decode buffer exceeds limit', 'budget_exceeded')
        # Local analysis descriptors have no FileRef: bind to the real opened
        # HDF owner, shared with Core arrays, rather than infer a filename.
        owner = next(((key, item) for key, item in self._handles.items()
                      if item['handle'].id == dataset.file.id), None)
        if owner is None:
            _fail('selected array has no retained verified owner', 'source_changed')
        self._check_hdf_owner(*owner)
        values = np.asarray(dataset[tuple(normalized)])
        values.flags.writeable = False
        self._check_hdf_owner(*owner)
        self._budget.logical_selected_bytes += values.nbytes
        return ArrayBlock(values, tuple(normalized), copy.deepcopy(wire), copy.deepcopy(self._state_ref), self.io_counters)

    def poll(self):
        self._operation()
        if self._pointer_path is None:
            return self.source
        from ._capture_io import beneath, small_json
        value, raw = small_json(beneath(self.root, self._pointer_path), self.limits.maximum_metadata_bytes, budget=self._budget)
        if raw == self._pointer_raw:
            return self.source
        pointer = validate_pointer(value)
        if pointer['identity'] != self._manifest['identity']:
            _fail('selected pointer source replaced', 'source_changed')
        candidate = validate_manifest(self._json(pointer['manifest']))
        if (candidate['identity'] != self._manifest['identity'] or candidate['invocation'] != self._manifest['invocation']
                or candidate['lineage'] != self._manifest['lineage']
                or candidate['provenance'] != self._manifest['provenance']
                or int(candidate['generation']) <= int(self._manifest['generation'])
                or int(candidate['state_sequence']) <= int(self._manifest['state_sequence'])
                or candidate['generation'] != pointer['generation']):
            _fail('pointer source/regression differs from current invocation', 'source_changed')
        old_manifest, old_ref = self._manifest, self._state_ref
        changed_paths = {pointer['manifest']['path'], *(row['path'] for row in candidate['artifacts'])}
        original_entries = {path: self._index.get(path) for path in changed_paths}
        original_weight = self._index_weight
        self._manifest = candidate
        self._state_ref = {key: candidate[key] for key in ('identity', 'state_id', 'state_sequence')}
        self._state_ref['manifest'] = pointer['manifest']
        self._pending_state_weight = _metadata_weight(self._state_ref) + _metadata_weight(candidate)
        try:
            self._register_manifest()
            if self._catalog_ready:
                self._import_history()
        except Exception:
            self._manifest, self._state_ref = old_manifest, old_ref
            self._index_weight, self._pending_state_weight = original_weight, 0
            for path, value in original_entries.items():
                if value is None:
                    self._index.pop(path, None)
                else:
                    self._index[path] = value
            raise
        self._checkpoint, self._declarations = None, {}
        self._pointer_raw = raw
        self._remember_state()
        return self.source

    def rebase_cursor(self, cursor):
        self._operation()
        _schema_check(schema_validator('result-cursor.schema.json'), cursor)
        digest = cursor['manifest']['sha256']
        if digest not in self._known_states:
            _fail('cursor source not observed in this invocation', 'source_changed')
        ref, manifest = self._known_states[digest]
        if cursor['identity'] != self._manifest['identity'] or cursor['invocation'] != self._manifest['invocation']:
            _fail('cursor source replaced', 'source_changed')
        validate_cursor(cursor, manifest=manifest, manifest_ref=ref['manifest'], catalog_page=self._cursor_page(cursor))
        rebased = copy.deepcopy(dict(cursor))
        rebased['manifest'], rebased['history'] = copy.deepcopy(self._state_ref['manifest']), copy.deepcopy(self._manifest['history'])
        self._check_cursor(rebased)
        return rebased


def verify_recovery_bundle(*args, **kwargs):
    """Retired v1 directory API; use explicit StateRef and checkpoint()."""
    _fail('legacy recovery directory unsupported; select result-v2 StateRef', 'unsupported_format')
