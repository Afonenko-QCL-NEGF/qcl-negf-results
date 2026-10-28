"""Explicit result-v2 snapshot selection and streaming single-archive export."""
from __future__ import annotations

from collections.abc import Callable
import copy
import hashlib
import os
from pathlib import Path
import tempfile

import h5py
from qcl_negf_contracts.artifacts import CONTRACT_SET, EXPORT_SCHEMA, validate_export_receipt
from qcl_negf_contracts.messages import ContractError
from .commits import json_bytes, atomic_write
from ._atomic_io import fsync_directory
from ._capture_io import beneath, copy_closed
from .export_budget import ExportBudget, validate_options
from .state import StateReader, _ref, _internal_storage
from .native import validate_native_handle


def _selection(reader, profile):
    if not isinstance(reader, StateReader):
        raise TypeError('explicit selected StateReader required')
    if profile not in ('science', 'full-state'):
        raise ValueError('profile must be science or full-state')
    if profile == 'full-state' and reader._manifest['checkpoint'] is None:
        raise ContractError('selected state has no checkpoint', 'capability_unavailable')
    selected, declared, omitted, records, sessions, queue, visited = {}, {}, [], [], [], [reader], set()
    def add(reference, role, dependencies=()):
        row = dict(_ref(reference), role=role, dependencies=list(dependencies))
        old = selected.get(row['path'])
        if old is not None and _ref(old) != _ref(row):
            raise ContractError('export source path has conflicting declarations', 'corrupt_result')
        selected[row['path']] = row
        if len(selected) > reader.limits.maximum_catalog_entries:
            raise ContractError('export selected reference inventory exceeds limit', 'budget_exceeded')
    def declare(row):
        old = declared.get(row['path'])
        if old is not None and _ref(old) != _ref(row):
            raise ContractError('export dependency inventory rebinds source bytes', 'corrupt_result')
        declared[row['path']] = row
        if len(declared) > reader.limits.maximum_catalog_entries:
            raise ContractError('export dependency inventory exceeds limit', 'budget_exceeded')
    try:
        while queue:
            current = queue.pop()
            digest = current._state_ref['manifest']['sha256']
            if digest in visited:
                continue
            visited.add(digest)
            if len(visited) > reader.limits.maximum_catalog_pages:
                raise ContractError('export lineage inventory exceeds limit', 'budget_exceeded')
            current.import_history()
            manifest = current._manifest
            add(current._state_ref['manifest'], 'source.manifest')
            for reference, page in current._pages:
                add(reference, 'science.catalog')
                for artifact in page['artifacts']:
                    declare(artifact)
                    add(artifact, artifact['role'], artifact['dependencies'])
            # Original canonical manifests remain provenance, never rewritten
            # to pretend an omitted optional state is a complete source tree.
            for artifact in manifest['artifacts']:
                declare(artifact)
                if profile == 'full-state' or artifact['role'] == 'science.history':
                    add(artifact, artifact['role'], artifact['dependencies'])
                else:
                    omitted.append(dict(source_state=copy.deepcopy(current._state_ref),
                        source_path=artifact['path'], role=artifact['role'],
                        reason='optional_full_state_excluded_from_science'))
            parent = manifest['lineage']['parent_state']
            if parent is not None:
                child = current._parent_reader(parent)
                sessions.append(child)
                queue.append(child)
                if profile == 'science' and child._manifest['checkpoint'] is not None:
                    # Pinned parent's typed declaration authenticates update
                    # anchors. Its unselected tensors stay explicit omissions.
                    add(child._manifest['checkpoint'], 'parent.checkpoint.proof')
            if profile == 'full-state':
                from qcl_negf_contracts.results import validate_checkpoint_declaration
                validate_checkpoint_declaration(current._json(manifest['checkpoint']), manifest=manifest)
            records.append(dict(source_state=copy.deepcopy(current._state_ref),
                included=[], omitted_by_policy=[]))
        # All selected declared dependencies are mandatory, not optional by role.
        # Full-state needs the complete original source graph. Science proof JSON
        # is provenance; scientific leaves retain their actual own dependencies.
        pending, expanded = list(selected), set()
        while pending:
            path = pending.pop()
            if path in expanded:
                continue
            expanded.add(path)
            for dependency in selected[path]['dependencies']:
                if dependency not in selected:
                    if dependency not in declared:
                        raise ContractError('selected artifact has undeclared dependency: ' + dependency, 'corrupt_result')
                    row = declared[dependency]
                    add(row, row['role'], row['dependencies'])
                pending.append(dependency)
        if profile == 'science' and not any(row['role'] == 'science.history' for row in selected.values()):
            raise ContractError('no published history or independent analysis representation', 'capability_unavailable')
        omitted = [row for row in omitted if row['source_path'] not in selected]
        return selected, omitted, records
    finally:
        for session in sessions:
            session.close()


def preview(reader: StateReader, *, profile: str) -> dict:
    selected, omitted, _ = _selection(reader, profile)
    capabilities = copy.deepcopy(reader.source.capabilities)
    capabilities['exact_restore'] = dict(available=False, reason='core_owned' if profile == 'full-state' else 'science_subset')
    return dict(profile=profile, source=copy.deepcopy(reader.source.state), files=len(selected),
        bytes=sum(int(row['bytes']) for row in selected.values()), omitted_by_policy=omitted,
        capabilities=capabilities, integrity_scope='declared_metadata_availability',
        canonical_source_complete=profile == 'full-state', scientific_accepted=None)


def export_snapshot(reader: StateReader, destination: str | Path, *, profile: str,
                    byte_budget: int, reserve_bytes: int,
                    progress: Callable[[dict], None] | None = None) -> dict:
    if not isinstance(reader, StateReader):
        raise TypeError('explicit selected StateReader required')
    reader._operation()
    previous = reader._shared_budget
    reader._shared_budget = reader._budget
    try:
        return _export_snapshot(reader, destination, profile=profile,
            byte_budget=byte_budget, reserve_bytes=reserve_bytes, progress=progress)
    finally:
        reader._shared_budget = previous


def _export_snapshot(reader, destination, *, profile, byte_budget, reserve_bytes, progress):
    validate_options(byte_budget, reserve_bytes)
    selected, omitted, records = _selection(reader, profile)
    if profile == 'full-state':
        from .archive import _verify_source_tree
        _verify_source_tree(reader.root, reader.source.state, limits=reader.limits, budget=reader._budget)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    snapshot_identity = hashlib.sha256(json_bytes(dict(schema=EXPORT_SCHEMA, contract_set=CONTRACT_SET,
        profile=profile, source_state=reader.source.state, selected=[selected[k] for k in sorted(selected)],
        omitted_by_policy=omitted, native_format='5.0', exporter_revision=9))).hexdigest()
    report = progress or (lambda event: None)
    def transport_report(phase, **values):
        report(dict(phase=phase, **values))
    from .archive import build_archive, verify_archive
    with tempfile.TemporaryDirectory(prefix='.export-pin-', dir=destination) as directory:
        spool = Path(directory)
        budget = ExportBudget(spool, byte_budget=byte_budget, reserve_bytes=reserve_bytes)
        budget.check(sum(int(row['bytes']) for row in selected.values()))
        objects, mapping = {}, {}
        for path, row in sorted(selected.items()):
            suffix = {'application/x-hdf5': '.h5', 'application/json': '.json',
                      'text/plain': '.txt'}.get(row['media_type'], '.bin')
            name = 'objects/' + row['sha256'] + suffix
            mapping[path] = name
            if name not in objects:
                target = spool / (row['sha256'] + suffix)
                size = int(row['bytes'])
                budget.check(size)
                copy_closed(beneath(reader.root, path), target, size=size, sha256=row['sha256'], budget=reader._budget)
                budget._record(target.resolve(), size)
                if row['media_type'] == 'application/x-hdf5':
                    with h5py.File(target, 'r', rdcc_nbytes=reader.limits.maximum_array_bytes) as handle:
                        _internal_storage(handle, maximum_objects=reader.limits.maximum_hdf_objects,
                                          maximum_depth=reader.limits.maximum_hdf_depth)
                        validate_native_handle(handle, declared_schema=row['schema'],
                            maximum_metadata_bytes=reader.limits.maximum_metadata_bytes)
                objects[name] = (target, dict(path=name, bytes=size, sha256=row['sha256'], media_type=row['media_type']))
            report(dict(phase='capturing', files=len(objects), source_path=path))
        included = [dict(source_path=path, role=row['role'], schema=row['schema'], object=mapping[path],
            sha256=row['sha256'], dependencies=[mapping[dep] for dep in row['dependencies']])
            for path, row in sorted(selected.items())]
        # Export selection describes the complete chosen object set, while
        # original canonical source manifests remain unchanged provenance.
        records = [dict(source_state=copy.deepcopy(reader.source.state), included=included, omitted_by_policy=omitted)]
        capabilities = ['complete_exact_source_graph'] if profile == 'full-state' else ['raw_scientific_history', 'stored_diagnostic_witnesses']
        manifest = dict(schema=EXPORT_SCHEMA, contract_set=CONTRACT_SET, profile=profile,
            snapshot_identity=snapshot_identity, records=records, files=[objects[k][1] for k in sorted(objects)],
            omitted_by_policy=omitted, capabilities=capabilities,
            requires_full_state=[] if profile == 'full-state' else ['exact_restart', 'arbitrary_new_optical_response'],
            source_state=copy.deepcopy(reader.source.state),
            source_locator=[dict(reference=_ref(row), object=mapping[path]) for path, row in sorted(selected.items())],
            canonical_source_complete=profile == 'full-state', scientific_accepted=None,
            snapshot_consistent=True, complete=profile == 'full-state',
            coverage=dict(scope='explicit_selected_state', uncommitted_tail='not_included',
                          stored_observables='independent_analysis_not_published'))
        metadata = {'manifest.json': json_bytes(manifest),
                    'README.md': ('# QCL-NEGF selected snapshot\n\nProfile: ' + profile +
                        '\nScientific acceptance is not assessed by export.\n').encode()}
        temporary, info, index = build_archive(spool, objects, metadata, identity=snapshot_identity,
            profile=profile, report=transport_report, budget=budget)
        if verify_archive(temporary, report=transport_report) != index:
            raise ContractError('archive readback inventory changed', 'corrupt_result')
        archive_path = destination / (info['sha256'] + '.tar.xz')
        receipt = dict(schema=EXPORT_SCHEMA, contract_set=CONTRACT_SET,
            transport_schema=index['schema'], profile=profile, snapshot_identity=snapshot_identity,
            sha256=info['sha256'], bytes=info['bytes'], filename=archive_path.name, archive=archive_path.name,
            snapshot_consistent=True, canonical_source_complete=profile == 'full-state', scientific_accepted=None)
        validate_export_receipt(receipt)
        receipt_file = spool / 'receipt.json'
        payload = json_bytes(receipt)
        with budget.open(receipt_file, 'xb') as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        budget.check_actual()
        receipt_path = destination / (snapshot_identity + '.json')
        archive_existed = archive_path.exists()
        previous_receipt = receipt_path.read_bytes() if receipt_path.exists() else None
        receipt_published = False
        os.replace(temporary, archive_path)
        try:
            os.replace(receipt_file, receipt_path)
            receipt_published = True
            fsync_directory(destination)
        except BaseException:
            try:
                if receipt_published:
                    if previous_receipt is None:
                        receipt_path.unlink(missing_ok=True)
                    else:
                        atomic_write(receipt_path, previous_receipt)
            finally:
                if not archive_existed:
                    archive_path.unlink(missing_ok=True)
            raise
        report(dict(phase='completed', archive=receipt['archive'], bytes=receipt['bytes']))
        return receipt
