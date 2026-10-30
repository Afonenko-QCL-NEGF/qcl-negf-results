"""Explicit result-v2 snapshot selection and streaming single-archive export."""
from __future__ import annotations

from collections.abc import Callable
import copy
import hashlib
import os
import sys
from pathlib import Path
import tempfile

import h5py
from qcl_negf_contracts.artifacts import CONTRACT_SET, EXPORT_SCHEMA, validate_export_receipt
from qcl_negf_contracts.messages import ContractError
from .commits import json_bytes, atomic_write
from ._atomic_io import fsync_directory
from ._capture_io import beneath, copy_closed
from .export_budget import ExportBudget, validate_options
from .state import StateReader, Document, _ref, _internal_storage
from .native import validate_native_handle


def _analysis_context(reader, observations_ref, diagnostics_ref):
    """Named exact references select representations, never schema inventory."""
    from .final_observations import FinalObservationsReader
    from .final_diagnostics import FinalDiagnosticsReader
    observations = diagnostics = None
    if observations_ref is not None:
        with FinalObservationsReader(reader, artifact_ref=observations_ref) as view:
            observations = view.describe()
        if diagnostics_ref is not None and diagnostics_ref != observations.diagnostics:
            raise ContractError('selected diagnostics FileRef differs from observations diagnostics', 'corrupt_result')
    if diagnostics_ref is not None:
        with FinalDiagnosticsReader(reader, artifact_ref=diagnostics_ref) as view:
            diagnostics = view.describe()
    for key,item in reader._handles.items():
        reader._check_hdf_owner(key,item)
    return observations, diagnostics


def _selection(reader, profile, *, analysis=(None, None)):

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
            if current is reader:
                for description in analysis:
                    if description is not None:
                        reference = description.artifact
                        row = declared.get(reference['path'])
                        if row is None or _ref(row) != reference:
                            raise ContractError('selected analysis FileRef differs from canonical inventory', 'corrupt_result')
                        add(row, row['role'], row['dependencies'])
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
        if profile == 'science' and not any(analysis) and not any(row['role'] == 'science.history' for row in selected.values()):
            raise ContractError('no published history or independent analysis representation', 'capability_unavailable')
        omitted = [row for row in omitted if row['source_path'] not in selected]
        return selected, omitted, records
    finally:
        primary = sys.exception()
        first = None
        for session in sessions:
            try:
                session.close()
            except BaseException as error:
                if first is None: first = error
                else: first.add_note('another lineage cleanup raised '+type(error).__name__)
        if first is not None:
            if primary is None: raise first
            primary.add_note('lineage cleanup also raised '+type(first).__name__)


def _technical_selection(reader, telemetry, profile, bindings):
    """Exact declared transaction and logical table subset; no newest discovery."""
    if telemetry is None:
        return None
    from .performance_views import _load_telemetry, PerformanceLimits
    from .catalog import _catalog_rows
    from qcl_negf_contracts.telemetry import TABLE_FIELDS, TELEMETRY_SCHEMA
    limits=(PerformanceLimits(maximum_objects=16384) if telemetry.commit_ref['schema']=='qcl-negf.telemetry-commit.v2' else PerformanceLimits())
    loaded = _load_telemetry(reader, telemetry, limits, bindings)
    selected = {}; omitted = []; rows = []
    def add(scope, reference, role, **extra):
        key = (str(Path(scope).absolute()), reference['path'])
        row = dict(reference=_ref(reference), root=Path(scope), role=role, **extra)
        old = selected.get(key)
        if old is not None and old['reference'] != row['reference']:
            raise ContractError('technical path has conflicting selected references', 'corrupt_result')
        if old is None:selected[key] = row
        from .state import _metadata_weight
        if _metadata_weight((selected,omitted,bindings))>reader.limits.maximum_catalog_bytes:
            raise ContractError('technical selected metadata inventory exceeds allowance','budget_exceeded')
        if len(selected)>reader.limits.maximum_catalog_entries:
            raise ContractError('technical selected inventory exceeds allowance', 'budget_exceeded')
    for scope,ref in loaded['references']:
        role = 'performance.summary' if ref==loaded['commit']['summary'] and scope==telemetry.artifact_root else 'performance.provenance'
        add(scope,ref,role)
    commit=loaded['commit'];commits=[commit]
    if loaded.get('technical_version') == 2:
        # S2 already authenticated the complete bounded retained closure. Previous
        # catalog descriptors are opaque leaves, never recursive export promises.
        for ordinal,descriptor in enumerate(loaded['catalog']['artifacts']):
            ref=descriptor['ref'];key=(str(telemetry.artifact_root.absolute()),ref['path'])
            if key not in selected:raise ContractError('authenticated retained member missing','corrupt_result')
            selected[key].update(role=descriptor['role'],table=descriptor['table'],
                catalog_order=(0,ordinal),source_role=descriptor['role'],selection='retained_technical_window')
        scope_fields=dict(technical_contract='qcl-negf.telemetry-retained.v2',scope='retained_technical_window',
            catalog=copy.deepcopy(commit['catalog']),cumulative_scope='enclosing_transport_since_v2_genesis',
            prior_capture=loaded['summary']['cumulative']['prior_capture'])
        return dict(loaded=loaded,selected=selected,omitted=[],scope_fields=scope_fields,
            cutoff=dict(commit=telemetry.commit_ref,source_cursors=commit['source_cursors'],
                uncommitted='not_included',pending_rows=None,**scope_fields),
            freshness=dict(session=commit['session']['source']['session_id'],published_unix=commit['published_unix'],
                observed_unix=telemetry.observed_unix,committed_age_seconds=None if telemetry.observed_unix is None else telemetry.observed_unix-commit['published_unix'],
                maximum_buffer_seconds=loaded['summary']['durability']['maximum_buffer_seconds'],pending_rows=None),
            root=telemetry.artifact_root,plan_root=telemetry.canonical_plan_root)
    if profile=='full-state':
        from .performance_views import _read_reference
        from .state import _schema_check,_metadata_weight
        from qcl_negf_contracts import schema_validator
        from qcl_negf_contracts.telemetry import validate_telemetry_summary
        envelope=schema_validator('telemetry-envelope.schema.json')
        queue=[];seen={json_bytes(telemetry.commit_ref)}
        for key in ('previous','baseline'):
            value=commit[key]
            if value is not None:queue.append(value['commit'] if key=='baseline' else value)
        while queue:
            ref=queue.pop();key=json_bytes(ref)
            if key in seen:continue
            seen.add(key)
            if len(seen)>reader.limits.maximum_catalog_pages:raise ContractError('technical predecessor graph exceeds allowance','budget_exceeded')
            value,_=_read_reference(reader,telemetry.artifact_root,ref,reader.limits,bindings)
            _schema_check(envelope.evolve(schema={'$ref':'#/$defs/TelemetryCommit','$defs':envelope.schema['$defs']}),value)
            add(telemetry.artifact_root,ref,'performance.provenance')
            prior_summary,_=_read_reference(reader,telemetry.artifact_root,value['summary'],reader.limits,bindings)
            validate_telemetry_summary(prior_summary)
            for field in ('session','transport','generation','closed','baseline','source_cursors'):
                if value[field]!=prior_summary[field]:raise ContractError('retained technical transaction header differs','corrupt_result')
            add(telemetry.artifact_root,value['summary'],'performance.provenance')
            for ar in value['accepted_ranges']:
                for name in ('retained_publication','retained_segment'):
                    if ar[name] is not None:add(telemetry.artifact_root,ar[name],'performance.provenance')
            commits.append(value)
            for name in ('previous','baseline'):
                edge=value[name]
                if edge is not None:queue.append(edge['commit'] if name=='baseline' else edge)
            if _metadata_weight((selected,omitted,commits,queue,seen,bindings))>reader.limits.maximum_catalog_bytes:
                raise ContractError('technical metadata graph exceeds allowance','budget_exceeded')
    catalog_groups={}
    for value in commits:
        head=value['catalog']
        if head is not None:catalog_groups.setdefault(str(Path(head['path']).parent),[]).append(head)
    for group_name,heads in catalog_groups.items():
        group_generation=min(int(value['generation']) for value in commits if value['catalog'] is not None and str(Path(value['catalog']['path']).parent)==group_name)
        watermark=max(heads,key=lambda head:head['entries'])
        # Wire root is explicit, while catalog previous/artifact paths are local
        # to the catalog's declared parent. Never infer a session by basename.
        parent=Path(watermark['path']).parent
        catalog_root=telemetry.artifact_root/parent
        local=dict(watermark,path=Path(watermark['path']).name)
        artifacts,pages=_catalog_rows(catalog_root,local,limits=reader.limits,budget=reader._budget,bindings=bindings)
        for ref,_ in pages:
            add(telemetry.artifact_root,dict(ref,path=str(parent/ref['path'])),'performance.catalog')
        witnessed={(ref['path'],ref['sha256'],page['entries']) for ref,page in pages}
        if any((Path(h['path']).name,h['sha256'],h['entries']) not in witnessed for h in heads):
            raise ContractError('technical catalog pins do not share their declared exact prefix','corrupt_result')
        tables={name:[] for name in TABLE_FIELDS}
        for artifact,raw in artifacts:
            if artifact.media_type=='application/vnd.apache.parquet':
                table=raw.get('table')
                if raw.get('schema')!=TELEMETRY_SCHEMA or table not in TABLE_FIELDS:
                    raise ContractError('declared performance table/schema required; no format guessing','unsupported_format')
                tables[table].append(raw)
        tails={raw['path'] for group in tables.values() for raw in group[-3:]}
        for artifact_index,(artifact,raw) in enumerate(artifacts):
            reference=dict(path=str(parent/artifact.path),bytes=str(artifact.size),sha256=artifact.sha256,
                media_type=artifact.media_type,schema=artifact.schema)
            if artifact.schema is None:raise ContractError('explicit technical artifact schema required','unsupported_format')
            chosen=profile=='full-state' or artifact.included('science') or artifact.path in tails
            if chosen:
                projection_role='performance.provenance' if artifact.media_type=='application/json' else 'performance.window' if profile=='science' and artifact.role=='performance.full' else artifact.role
                add(telemetry.artifact_root,reference,projection_role,table=raw.get('table'),
                    catalog_order=(group_generation,artifact_index),source_role=artifact.role,selection='cutoff_tail' if artifact.path in tails else 'declared_profile')
                rows.append(raw)
            else:omitted.append(dict(reference=reference,role=artifact.role,reason='performance_full_excluded_from_science'))
    return dict(loaded=loaded,selected=selected,omitted=omitted,
        cutoff=dict(commit=telemetry.commit_ref,catalog=commit['catalog'],source_cursors=commit['source_cursors'],
            uncommitted='not_included',pending_rows=None),
        freshness=dict(session=commit['session']['source']['session_id'],published_unix=commit['published_unix'],
            observed_unix=telemetry.observed_unix,committed_age_seconds=None if telemetry.observed_unix is None else telemetry.observed_unix-commit['published_unix'],
            maximum_buffer_seconds=loaded['summary']['durability']['maximum_buffer_seconds'],pending_rows=None),
        root=telemetry.artifact_root,plan_root=telemetry.canonical_plan_root)


def _bound_capabilities(reader, profile, analysis):
    capabilities = copy.deepcopy(reader.source.capabilities)
    observations, diagnostics = analysis
    if observations is not None:
        for name in ('saved_views', 'stored_observables', 'stored_final_diagnostics'):
            capabilities[name] = dict(available=True, reason=None)
    elif diagnostics is not None:
        capabilities['stored_final_diagnostics'] = dict(available=True, reason=None)
    if any(analysis):
        capabilities['science_export'] = dict(available=True, reason=None)
    capabilities['exact_restore'] = dict(available=False,
        reason='core_owned' if profile == 'full-state' else 'science_subset')
    return capabilities


def preview(reader: StateReader, *, profile: str,
            observations_ref: Document | None = None, diagnostics_ref: Document | None = None, telemetry=None) -> dict:
    if not isinstance(reader, StateReader):
        raise TypeError('explicit selected StateReader required')
    with reader._consumer_operation():
        analysis = _analysis_context(reader, observations_ref, diagnostics_ref)
        selected, omitted, _ = _selection(reader, profile, analysis=analysis)
        bindings={};technical=_technical_selection(reader,telemetry,profile,bindings)
        from .performance_views import _check_bindings
        _check_bindings(bindings)
        return dict(profile=profile, source=copy.deepcopy(reader.source.state), files=len(selected)+(0 if technical is None else len(technical['selected'])),
            bytes=sum(int(row['bytes']) for row in selected.values())+ (0 if technical is None else sum(int(row['reference']['bytes']) for row in technical['selected'].values())), omitted_by_policy=omitted,
            telemetry=None if technical is None else dict(cutoff=technical['cutoff'],freshness=technical['freshness']),
            capabilities=_bound_capabilities(reader, profile, analysis),
            integrity_scope='selected_analysis_metadata_verified' if any(analysis) else 'declared_metadata_availability',
            canonical_source_complete=profile == 'full-state', scientific_accepted=None)


def export_snapshot(reader: StateReader, destination: str | Path, *, profile: str,
                    byte_budget: int, reserve_bytes: int,
                    progress: Callable[[dict], None] | None = None,
                    observations_ref: Document | None = None, diagnostics_ref: Document | None = None, telemetry=None, witness_policy=None) -> dict:
    if not isinstance(reader, StateReader):
        raise TypeError('explicit selected StateReader required')
    with reader._consumer_operation():
        analysis = _analysis_context(reader, observations_ref, diagnostics_ref)
        return _export_snapshot(reader, destination, profile=profile,
            byte_budget=byte_budget, reserve_bytes=reserve_bytes, progress=progress, analysis=analysis,telemetry=telemetry,witness_policy=witness_policy)


def _export_snapshot(reader, destination, *, profile, byte_budget, reserve_bytes, progress, analysis,telemetry,witness_policy):
    validate_options(byte_budget, reserve_bytes)
    selected, omitted, records = _selection(reader, profile, analysis=analysis)
    witness=None
    if witness_policy is not None:
        from .witness_selection import POLICY,select_history_witnesses
        if witness_policy!=POLICY:raise ValueError('unsupported logical witness selection policy')
        witness=select_history_witnesses(reader)
    technical_bindings={}
    technical=_technical_selection(reader,telemetry,profile,technical_bindings)
    if profile == 'full-state':
        from .archive import _verify_source_tree
        _verify_source_tree(reader.root, reader.source.state, limits=reader.limits, budget=reader._budget, selected_reader=reader)
    # Capture qualified source bindings before any progress callback can mutate
    # them. Copies hash their actual FD; final publication also requires paths
    # and the selected root to remain the same immutable source.
    from ._capture_io import regular, stamp
    bindings = {}
    for path,row in selected.items():
        with regular(beneath(reader.root,path), size=int(row['bytes'])) as stream:
            bindings[path] = stamp(stream, include_change=True)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    snapshot_identity = hashlib.sha256(json_bytes(dict(schema=EXPORT_SCHEMA, contract_set=CONTRACT_SET,
        profile=profile, source_state=reader.source.state, selected=[selected[k] for k in sorted(selected)],
        omitted_by_policy=omitted, selected_analysis=[None if d is None else d.artifact for d in analysis],
        technical_selection=[] if technical is None else [dict(scope='telemetry' if row['root']==technical['root'] else 'canonical_plan',reference=row['reference'],role=row['role'],table=row.get('table')) for _,row in sorted(technical['selected'].items())],
        technical_observation=None if technical is None else technical['freshness']['observed_unix'],witness_policy=witness_policy,native_format='5.0', exporter_revision=11))).hexdigest()
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
        from .provenance import _selected_source_provenance
        provenance = _selected_source_provenance(reader, observations=analysis[0])
        # Every row below has just been copied/hash-checked and its native
        # header validated. Reader cache warming is not archive authority or
        # selected-analysis qualification and must not alter serialized bytes.
        provenance['native_verified_references'] = [_ref(row) for _,row in sorted(selected.items())
            if row['media_type'] == 'application/x-hdf5']
        provenance['native_verified_references_scope'] = 'captured_native_bytes_and_headers'
        records = [dict(source_state=copy.deepcopy(reader.source.state), included=included, omitted_by_policy=omitted,
            native_provenance=provenance)]
        technical_locator=[];compaction=[];telemetry_storage=None
        if technical is not None:
            from .compaction import _CompactionRead,compact_export_records
            footer=_CompactionRead(32*1024*1024,reader._budget)
            technical_included=[]
            for _,row in sorted(technical['selected'].items(),key=lambda pair:(pair[1].get('catalog_order',(-1,-1)),pair[0])):
                ref=row['reference'];suffix='.parquet' if ref['media_type']=='application/vnd.apache.parquet' else '.json'
                name='objects/'+ref['sha256']+suffix
                if name not in objects:
                    with regular(beneath(row['root'],ref['path']),size=int(ref['bytes'])) as source:
                        technical_bindings[(str(row['root'].absolute()),ref['path'])]=stamp(source,include_change=True)
                    from .state import _metadata_weight
                    if _metadata_weight(technical_bindings)>reader.limits.maximum_catalog_bytes:
                        raise ContractError('technical source stability inventory exceeds allowance','budget_exceeded')
                    target=spool/(ref['sha256']+suffix);size=int(ref['bytes']);budget.check(size)
                    copy_closed(beneath(row['root'],ref['path']),target,size=size,sha256=ref['sha256'],budget=reader._budget)
                    budget._record(target.resolve(),size)
                    objects[name]=(target,dict(path=name,bytes=size,sha256=ref['sha256'],media_type=ref['media_type']))
                if row.get('table') is not None:
                    with footer.open(objects[name][0]) as stream:
                        with footer.parquet(stream) as file:
                            _,table=footer.geometry(file)
                            if table!=row['table']:raise ContractError('selected Parquet footer differs from authenticated table declaration','corrupt_result')
                scope='telemetry' if row['root']==technical['root'] else 'canonical_plan'
                technical_locator.append(dict(scope=scope,reference=ref,object=name))
                technical_included.append(dict(role=row['role'],object=name,sha256=ref['sha256'],schema=ref['schema'],
                    dependencies=[],table=row.get('table'),source_path=ref['path'],source_scope=scope,
                    source_role=row.get('source_role',row['role']),selection=row.get('selection','exact_transaction_metadata')))
            footer.finish()
            technical_record=dict(source_state=copy.deepcopy(reader.source.state),source_commit_sha256=telemetry.commit_ref['sha256'],
                technical=True,included=technical_included,omitted_by_policy=technical['omitted'],**technical.get('scope_fields',{}))
            records.append(technical_record)
            compaction=compact_export_records([technical_record],objects,spool,budget=budget,io_budget=reader._budget,retain_original=True)
            originals={row['object'] for row in technical_locator if row['reference']['media_type']=='application/vnd.apache.parquet'}
            telemetry_storage=dict(original_parquet_bytes=sum(objects[name][1]['bytes'] for name in originals),
                projection_parquet_bytes=sum(proof['compacted_bytes'] for proof in compaction),physical_space_saving_claim=False)
        capabilities = ['complete_exact_source_graph'] if profile == 'full-state' else []
        if any(row['role']=='science.history' for row in selected.values()):
            capabilities += ['raw_scientific_history', 'stored_diagnostic_witnesses']
        if analysis[0] is not None: capabilities.append('selected_final_observations_metadata')
        if any(analysis): capabilities.append('selected_final_diagnostics_metadata')
        manifest = dict(schema=EXPORT_SCHEMA, contract_set=CONTRACT_SET, profile=profile,
            snapshot_identity=snapshot_identity, records=records, files=[objects[k][1] for k in sorted(objects)],
            omitted_by_policy=omitted, capabilities=capabilities,
            requires_full_state=[] if profile == 'full-state' else ['exact_restart', 'arbitrary_new_optical_response'],
            source_state=copy.deepcopy(reader.source.state),
            source_locator=[dict(reference=_ref(row), object=mapping[path]) for path, row in sorted(selected.items())],
            canonical_source_complete=profile == 'full-state', scientific_accepted=None,
            snapshot_consistent=True, complete=profile == 'full-state',
            technical_locator=technical_locator,telemetry_compaction=compaction,telemetry_storage=telemetry_storage,
            cutoff=None if technical is None else technical['cutoff'],freshness=dict(telemetry=[] if technical is None else [technical['freshness']]),
            job_complete=False if technical is not None and not technical['loaded']['commit']['closed'] else None,
            witness_selection=witness,
            coverage=dict(scope='explicit_selected_state', uncommitted_tail='not_included',
                          stored_observables='selected_metadata_verified' if analysis[0] is not None else 'explicit_final_observations_ref_required',
                          stored_final_diagnostics='selected_metadata_verified' if any(analysis) else 'explicit_final_diagnostics_ref_required'))
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
        try:
            from .performance_views import _check_bindings
            _check_bindings(technical_bindings)
            for path,before in bindings.items():
                with regular(beneath(reader.root,path), size=int(selected[path]['bytes'])) as stream:
                    if stamp(stream, include_change=True) != before:
                        raise ContractError('export source binding changed before publication', 'source_changed')
            if reader._root_identity() != reader._root_stamp:
                raise ContractError('export source root changed before publication', 'source_changed')
            for key,item in reader._handles.items():
                reader._check_hdf_owner(key,item)
        except (OSError, ValueError, ContractError) as error:
            raise ContractError('export source changed before publication', 'source_changed') from error
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
