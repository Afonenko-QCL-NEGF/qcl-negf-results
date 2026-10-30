from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from qcl_negf_contracts.artifacts import CONTRACT_SET, MODEL_SCHEMA, validate_commit
from qcl_negf_contracts.messages import ContractError
from qcl_negf_contracts.telemetry import TABLE_FIELDS, TELEMETRY_SCHEMA
from qcl_negf_results.compaction import compact_performance, validate_performance_schema
from qcl_negf_results.model import canonical_bytes, validate_model
from test_export import fixture, unpack
from qcl_negf_results.export import export_snapshot
from qcl_negf_results.telemetry import TelemetryWriter


def test_registry_matches_pure_runtime_contracts():
    from qcl_negf_contracts.artifacts import ARTIFACT_SCHEMA_CONTRACTS, NATIVE_SCHEMA_VERSION
    from qcl_negf_contracts import load_schema
    registry = load_schema("results-contract-set.json")
    assert registry["contract_set"] == CONTRACT_SET
    assert registry["native_schema_version"] == NATIVE_SCHEMA_VERSION
    assert registry["previous_contracts_supported"] is False
    assert {(row["role"], row["media_type"]): frozenset(row["schemas"]) for row in registry["artifacts"]} == ARTIFACT_SCHEMA_CONTRACTS


def test_model_envelope_preserves_numeric_types_and_rejects_stale_or_changed_input():
    configuration = {"integer": 1, "float": 1.0, "negative_zero": -0.0, "name": "Φонон"}
    envelope = {"schema": MODEL_SCHEMA, "contract_set": CONTRACT_SET,
                "configuration": configuration, "hash_encoding": "qcl-negf-canonical-bytes-v1",
                "configuration_hash": hashlib.sha256(canonical_bytes(configuration)).hexdigest()}
    assert validate_model(envelope) == configuration
    assert canonical_bytes(1) != canonical_bytes(1.0)
    assert canonical_bytes(-0.0) != canonical_bytes(0.0)
    with pytest.raises(ContractError, match="contract set"):
        validate_model(configuration)
    envelope["configuration"] = {**configuration, "float": 1}
    with pytest.raises(ContractError, match="identity differs"):
        validate_model(envelope)


def test_missing_contract_rejected_before_payload_access(tmp_path):
    from result_v2_fixture import record, source

    root = tmp_path / "run"
    ref = source(root, [record()])
    commit = json.loads((root / ref["manifest"]["path"]).read_bytes())
    validate_commit(commit)  # The current manifest is otherwise valid.
    commit.pop("contract_set")
    with pytest.raises(ContractError, match="unsupported result format") as error:
        validate_commit(commit)
    assert error.value.code == "unsupported_format"


def test_compaction_preserves_order_null_nan_payload_and_signed_zero(tmp_path):
    types = {"string": pa.string(), "int64": pa.int64(), "float64": pa.float64(), "bool": pa.bool_()}
    schema = pa.schema([pa.field(name, types[kind]) for name, kind in TABLE_FIELDS["resources"]],
        metadata={b"schema": TELEMETRY_SCHEMA.encode(), b"contract_set": CONTRACT_SET.encode(), b"table": b"resources"})
    nan = np.array([0x7ff8000000000003], dtype=np.uint64).view(np.float64)[0]
    paths = []
    rows = [{"monotonic_ns": index, "cpu_time_seconds": value, "missing_reason": "пропуск" if value is None else None}
            for index, value in enumerate([0.0, -0.0, nan, None] * 1300)]
    for number, (start, stop) in enumerate(((0, 3), (3, 4098), (4098, 5200))):
        path = tmp_path / f"part-{number}.parquet"
        pq.write_table(pa.Table.from_pylist(rows[start:stop], schema=schema), path)
        paths.append(path)
    proof = compact_performance(paths, tmp_path / "compacted.parquet")
    assert proof["rows"] == 5200 and proof["source_files"] == 3
    assert proof["logical_payload_sha256"] == proof["verified_payload_sha256"]
    values = pq.read_table(tmp_path / "compacted.parquet")["cpu_time_seconds"].to_numpy()
    assert values.view(np.uint64)[1] == 0x8000000000000000
    assert values.view(np.uint64)[2] == 0x7ff8000000000003
    assert pq.read_table(tmp_path / "compacted.parquet")["monotonic_ns"].to_pylist() == list(range(5200))
    with pytest.raises(ContractError, match="contract set"):
        validate_performance_schema(schema.with_metadata({b"schema": b"qcl-negf.performance.v1"}))


def test_export_compaction_preserves_all_registered_resources(tmp_path):
    from native_consumer_fixture import technical_source, telemetry_input
    from qcl_negf_results.state import StateReader
    c=technical_source(tmp_path/'input')
    recorder = TelemetryWriter(c['technical'], {}, {}, transport=c['transport'])
    for index in range(4):
        recorder.record_resource({"monotonic_ns": index + 1, "wall_seconds": float(index), "rss_bytes": 100 + index},
                                 science_window=index == 2)
        recorder.flush()
    recorder.close()
    with StateReader(c['science'],state_ref=c['state'],expected_identity=c['state']['identity']) as reader:
        receipt = export_snapshot(reader, tmp_path / "export",profile='science',byte_budget=32*1024*1024,reserve_bytes=0,
            telemetry=telemetry_input(c,recorder))
    manifest, members = unpack(receipt, tmp_path / "export")
    proofs = [proof for proof in manifest["telemetry_compaction"] if proof["table"] == "resources"]
    assert len(proofs) == 1 and proofs[0]["rows"] == 4 and proofs[0]["source_files"] == 4
    assert proofs[0]["object"] in members
    compacted = pq.read_table(io.BytesIO(members[proofs[0]["object"]]))
    assert compacted["rss_bytes"].to_pylist() == [100, 101, 102, 103]
    assert manifest["contract_set"] == CONTRACT_SET


def test_export_witness_selection_keeps_scalar_rows_and_native_bits(tmp_path):
    from native_consumer_fixture import witness_history, selected_arrays
    from qcl_negf_results.state import StateReader
    from qcl_negf_results.witness_selection import select_history_witnesses
    ref, records = witness_history(tmp_path)
    before = {str(p.relative_to(tmp_path)): p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    with StateReader(tmp_path, state_ref=ref, expected_identity=ref['identity']) as reader:
        proof = select_history_witnesses(reader)
        assert proof['source_witnesses'] == 10 and proof['selected_witnesses'] == 4
        assert proof['selected_sequences'] == [1, 5, 13, 19]
        assert proof['scalar_history'] == 'all rows unchanged'
        assert proof['physical_payload_scope'] == 'whole original owners retained unchanged'
        payloads = selected_arrays(reader, proof)
        assert list(payloads) == [1, 5, 13, 19]
        import numpy as np
        for arrays in payloads.values():
            assert arrays['Gp'] == (np.eye(3,dtype='<c16')*-2.34e-12).tobytes()
            assert np.frombuffer(arrays['eigenvector'],dtype='<u8').tolist() == [0x7ff8000000001234,0,0x8000000000000000,0,0,0]
        reader.import_history()
        batch=reader.history_batch(maximum_records=20)
        assert batch.records == tuple(records[0])
    assert {str(p.relative_to(tmp_path)): p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()} == before


def test_witness_selection_retains_independent_attempt_extrema(tmp_path):
    from native_consumer_fixture import witness_history, selected_arrays
    from qcl_negf_results.state import StateReader
    from qcl_negf_results.witness_selection import select_history_witnesses
    ref, records = witness_history(tmp_path, attempts=2)
    with StateReader(tmp_path,state_ref=ref,expected_identity=ref['identity']) as reader:
        proof=select_history_witnesses(reader)
        assert proof['selected_sequences'] == [1, 5, 13, 19, 21, 25, 33, 39]
        assert proof['selected_witnesses'] == 8 and proof['source_witnesses'] == 20
        assert [row['selected_sequences'] for row in proof['attempts']] == [[1, 5, 13, 19], [21, 25, 33, 39]]
        assert [row['omitted_witnesses'] for row in proof['attempts']] == [6, 6]
        assert len(selected_arrays(reader,proof)) == 8
    # Exact original lineage/CP bytes establish ownership, never source inventory order.
    import json
    path=tmp_path/ref['manifest']['path'];value=json.loads(path.read_bytes())
    parent=value['lineage']['parent_state'];(tmp_path/parent['manifest']['path']).unlink()
    with StateReader(tmp_path,state_ref=ref,expected_identity=ref['identity']) as reader:
        with pytest.raises((ContractError, OSError), match='(source|selected|parent|metadata|manifest|opened)'):
            select_history_witnesses(reader)


@pytest.mark.parametrize('updates,expected',[
    ({1:[(13,0.)]},[1,5,19]),
    ({1:[(5,-2.)]},[1,7,13,19]),
    ({1:[(15,30.)]},[1,5,13,19]),
    ({1:[(13,0.),(13,31.)]},[1,5,13,19]),
])
def test_T10_latest_witness_updates_demote_worst_positive_and_keep_stable_ties(tmp_path,updates,expected):
    from native_consumer_fixture import witness_history
    from qcl_negf_results.state import StateReader
    from qcl_negf_results.witness_selection import select_history_witnesses
    ref,records=witness_history(tmp_path,updates=updates)
    with StateReader(tmp_path,state_ref=ref,expected_identity=ref['identity']) as reader:
        proof=select_history_witnesses(reader)
    assert proof['selected_sequences']==expected
    assert proof['scalar_rows']==20 and proof['source_witnesses']==10
    assert proof['witness_updates']==len(updates[1])
    assert proof['attempts'][0]['identity']==ref['identity']


def test_T10_witness_update_maps_are_bounded_per_whole_operation_and_original_identity(tmp_path):
    from native_consumer_fixture import witness_history
    from qcl_negf_results.state import StateReader
    from qcl_negf_results.diagnostic_views import DiagnosticViewLimits
    from qcl_negf_results.witness_selection import select_history_witnesses
    ref,_=witness_history(tmp_path,attempts=2,updates={1:[(13,0.)],2:[(5,-2.)]})
    with StateReader(tmp_path,state_ref=ref,expected_identity=ref['identity']) as reader:
        proof=select_history_witnesses(reader)
        assert [a['selected_sequences'] for a in proof['attempts']]==[[1,5,19],[21,27,33,39]]
        assert proof['scalar_rows']==40 and proof['witness_updates']==2
        with pytest.raises(ContractError) as error:
            select_history_witnesses(reader,limits=DiagnosticViewLimits(maximum_source_entries=1))
        assert error.value.code=='budget_exceeded'
        assert select_history_witnesses(reader)['selected_sequences']==proof['selected_sequences']


@pytest.mark.parametrize('kind',['bound','symlink'])
def test_T10_catalog_refuses_symlink_and_exhausted_metadata_before_payload(tmp_path,kind):
    from qcl_negf_results.catalog import catalog_artifacts
    from qcl_negf_results.state import ResultReadLimits
    writer=TelemetryWriter(tmp_path,{},{});writer.record_resource(dict(wall_seconds=1.));writer.close()
    if kind=='bound':
        with pytest.raises(ContractError) as error:
            catalog_artifacts(writer.directory,writer.catalog_watermark,limits=ResultReadLimits(maximum_catalog_pages=1))
        assert error.value.code=='budget_exceeded'
        return
    path=writer.directory/writer.catalog_watermark['path'];saved=path.with_suffix('.actual')
    path.rename(saved);path.symlink_to(saved.name)
    with pytest.raises(ContractError):catalog_artifacts(writer.directory,writer.catalog_watermark)


def test_T10_compaction_rejects_decoded_geometry_before_arrow_payload_and_charges_actual_io(tmp_path,monkeypatch):
    from qcl_negf_results.state import _IOBudget
    types={'string':pa.string(),'int64':pa.int64(),'float64':pa.float64(),'bool':pa.bool_()}
    schema=pa.schema([pa.field(n,types[k]) for n,k in TABLE_FIELDS['resources']],
        metadata={b'schema':TELEMETRY_SCHEMA.encode(),b'contract_set':CONTRACT_SET.encode(),b'table':b'resources'})
    source=tmp_path/'known-small.parquet'
    pq.write_table(pa.Table.from_pylist([{'wall_seconds':float(n),'thread_counter_details':'x'*10000} for n in range(32)],schema=schema),source)
    calls=[]
    def forbidden(*args,**kwargs):calls.append(True);raise AssertionError('decoded payload must not be requested')
    with monkeypatch.context() as patch:
        patch.setattr(pq.ParquetFile,'iter_batches',forbidden)
        with pytest.raises(ContractError) as error:
            compact_performance([source],tmp_path/'denied.parquet',maximum_buffer_bytes=1024)
        assert error.value.code=='budget_exceeded' and calls==[]
    io=_IOBudget(64)
    with pytest.raises(ContractError) as error:
        compact_performance([source],tmp_path/'denied-io.parquet',io_budget=io)
    assert error.value.code=='budget_exceeded' and io.bytes_read_total<=64


def test_T10_full_technical_graph_keeps_both_collector_pages_and_catalog_order(tmp_path):
    from native_consumer_fixture import technical_source,telemetry_input
    from qcl_negf_results.state import StateReader
    c=technical_source(tmp_path/'input',checkpoint=True)
    writer=TelemetryWriter(c['technical'],{}, {},transport=c['transport'])
    writer.record_resource(dict(wall_seconds=1.,rss_bytes=100));writer.flush();first=writer.rotate()
    writer.record_resource(dict(wall_seconds=2.,rss_bytes=99));writer.close()
    with StateReader(c['science'],state_ref=c['state'],expected_identity=c['state']['identity']) as reader:
        receipt=export_snapshot(reader,tmp_path/'export',profile='full-state',byte_budget=32*1024*1024,reserve_bytes=0,
            telemetry=telemetry_input(c,writer))
    manifest,members=unpack(receipt,tmp_path/'export')
    assert manifest['canonical_source_complete'] is True and manifest['scientific_accepted'] is None
    original={r['reference']['path']:r for r in manifest['technical_locator'] if r['scope']=='telemetry'}
    assert str(first.relative_to(writer.root)) in original
    assert members[original[str(first.relative_to(writer.root))]['object']]==first.read_bytes()
    proof=next(p for p in manifest['telemetry_compaction'] if p['table']=='resources')
    table=pq.read_table(io.BytesIO(members[proof['object']]))
    assert table['rss_bytes'].to_pylist()==[100,99]
    assert manifest['telemetry_storage']['original_parquet_bytes']>0
    assert manifest['telemetry_storage']['projection_parquet_bytes']==proof['compacted_bytes']
    assert manifest['telemetry_storage']['physical_space_saving_claim'] is False
    assert len([r for r in original.values() if r['reference']['media_type']=='application/vnd.apache.parquet'])==2


def test_T10_technical_locator_scope_and_compaction_originals_are_closed(tmp_path):
    import copy
    from native_consumer_fixture import consumer_writer,consumer_export
    from qcl_negf_results.archive import _manifest_closure
    c,writer=consumer_writer(tmp_path/'input')
    for number in range(2):writer.record_resource(dict(wall_seconds=float(number)));writer.flush()
    writer.close();receipt=consumer_export(c,writer,tmp_path/'export')
    manifest,members=unpack(receipt,tmp_path/'export')
    index=json.loads(members['export-index.json'])
    _manifest_closure(manifest,index)
    bad=copy.deepcopy(manifest)
    item=next(r for r in bad['technical_locator'] if r['reference']['schema']=='qcl-negf.telemetry-source-binding.v1')
    item['scope']='canonical_plan'
    with pytest.raises(ContractError,match='scope'):_manifest_closure(bad,index)
    bad=copy.deepcopy(manifest);bad['technical_locator']=[r for r in bad['technical_locator'] if r['reference']['media_type']!='application/vnd.apache.parquet']
    with pytest.raises(ContractError):_manifest_closure(bad,index)


def test_T10_export_projection_footer_never_bypasses_shared_delivered_io(tmp_path,monkeypatch):
    from qcl_negf_results.compaction import compact_export_records
    from qcl_negf_results.state import _IOBudget
    writer=TelemetryWriter(tmp_path/'technical',{}, {})
    for value in range(2):writer.record_resource(dict(wall_seconds=float(value)));writer.flush()
    writer.close();files={};included=[]
    for path in sorted(writer.directory.glob('resources-*.parquet')):
        name=path.name;raw=path.read_bytes();sha=hashlib.sha256(raw).hexdigest()
        files[name]=(path,dict(path=name,bytes=len(raw),sha256=sha,media_type='application/vnd.apache.parquet'))
        included.append(dict(object=name,dependencies=[],role='performance.full'))
    record=dict(included=included,source_commit_sha256='a'*64);io=_IOBudget(64);direct=[]
    original=pq.ParquetFile
    def observed(source,*args,**kwargs):
        if isinstance(source,(str,Path)):direct.append(source)
        return original(source,*args,**kwargs)
    monkeypatch.setattr(pq,'ParquetFile',observed)
    with pytest.raises(ContractError) as error:
        compact_export_records([record],files,tmp_path,io_budget=io,retain_original=True)
    assert error.value.code=='budget_exceeded' and io.bytes_read_total<=64
    assert direct==[]


def test_T10_exact_technical_export_bytes_ignore_reader_cache_warming(tmp_path):
    from native_consumer_fixture import consumer_writer,telemetry_input
    from qcl_negf_results.state import StateReader
    from qcl_negf_results.performance_views import read_performance_summary
    c,writer=consumer_writer(tmp_path/'input')
    for value in range(3):writer.record_resource(dict(wall_seconds=float(value),rss_bytes=100+value));writer.flush()
    writer.close();selection=telemetry_input(c,writer)
    with StateReader(c['science'],state_ref=c['state'],expected_identity=c['state']['identity']) as reader:
        first=export_snapshot(reader,tmp_path/'cold',profile='science',byte_budget=32*1024*1024,reserve_bytes=0,telemetry=selection)
        assert 0<reader.io_counters['bytes_read_total']<=reader.limits.maximum_io_bytes
        assert read_performance_summary(reader,telemetry=selection)['status']=='available'
        reader.history_batch(maximum_records=1)
        second=export_snapshot(reader,tmp_path/'warm',profile='science',byte_budget=32*1024*1024,reserve_bytes=0,telemetry=selection)
        # Each public operation has its own allowance; nested reads share it.
        assert 0<reader.io_counters['bytes_read_total']<=reader.limits.maximum_io_bytes
    assert first['snapshot_identity']==second['snapshot_identity'] and first['sha256']==second['sha256']
    assert (tmp_path/'cold'/first['archive']).read_bytes()==(tmp_path/'warm'/second['archive']).read_bytes()


def s2_archive_wire(tmp_path,*,fault=None,profile='science',parquet=False,compact=False):
    """Assemble exact v2 wire objects beside an unchanged scientific export.

    This is a manufactured carrier, not the future S3 writer/export activation.
    """
    import copy
    from test_performance_views import s2_wire_case
    from result_v2_fixture import put
    from qcl_negf_results.state import StateReader
    from qcl_negf_results.archive import build_archive
    from qcl_negf_results.commits import json_bytes
    c=s2_wire_case(tmp_path/'wire',parquet=parquet or compact or fault=='footer_rows',declared_rows=3 if fault=='footer_rows' else 2,parquet_files=2 if compact else 1,checkpoint=profile=='full-state')
    v=c['wire'];root=c['technical'];binding_ref=c['selection'].binding_ref
    if fault in ('forged_cost','wrong_origin'):
        summary=copy.deepcopy(v['summary']);commit=copy.deepcopy(v['commit'])
        if fault=='forged_cost':summary['cumulative']['cost']['metrics']['read_seconds']['known']=999.
        else:
            source=json.loads((root/c['source_binding']['path']).read_bytes());source['producer_provenance']['revision']='f'*40
            c['source_binding']=put(root,c['source_binding']['path'],source,source['schema'])
        v['summary']=summary;v['summary_ref']=put(root,v['summary_ref']['path'],summary,summary['schema'])
        commit['summary']=v['summary_ref'];v['commit']=commit;v['commit_ref']=put(root,v['commit_ref']['path'],commit,commit['schema'])
        binding=json.loads((root/binding_ref['path']).read_bytes())
        for key,ref in (('commit',v['commit_ref']),('summary',v['summary_ref']),('source_binding',c['source_binding'])):
            binding[key]=dict(ref=copy.deepcopy(ref),raw_utf8=(root/ref['path']).read_text())
        binding_ref=put(root,binding_ref['path'],binding,binding['schema'])
    with StateReader(c['science'],state_ref=c['state'],expected_identity=c['state']['identity']) as reader:
        receipt=export_snapshot(reader,tmp_path/'scientific-seed',profile=profile,byte_budget=32*1024*1024,reserve_bytes=0)
    manifest,members=unpack(receipt,tmp_path/'scientific-seed')
    scope_fields=dict(technical_contract='qcl-negf.telemetry-retained.v2',scope='retained_technical_window',
        catalog=copy.deepcopy(v['catalog_ref']),cumulative_scope='enclosing_transport_since_v2_genesis',prior_capture='none')
    cutoff=dict(commit=copy.deepcopy(v['commit_ref']),source_cursors=[],uncommitted='not_included',pending_rows=None,**copy.deepcopy(scope_fields))
    record=dict(technical=True,source_state=copy.deepcopy(c['state']),source_commit_sha256=v['commit_ref']['sha256'],included=[],omitted_by_policy=[],**copy.deepcopy(scope_fields))
    refs=[('telemetry',ref,root) for ref in [v['commit_ref'],v['summary_ref'],v['catalog_ref'],c['source_binding'],binding_ref]]
    if len(c['wire_history'])>1:
        prior=c['wire_history'][-2];refs += [('telemetry',prior[key],root) for key in ('commit_ref','summary_ref','catalog_ref')]
    refs += [('telemetry',row['ref'],root) for row in v['catalog']['artifacts']]
    refs += [('canonical_plan',c['plan'],c['root'])]
    seen=set();technical=[]
    if fault=='extra_unselected':
        old=c['wire_history'][0];ref=put(root,'extra-historical-summary.json',old['summary'],old['summary']['schema']);refs.append(('telemetry',ref,root))
    for scope,ref,base in refs:
        key=(scope,ref['path'])
        if key in seen:continue
        seen.add(key)
        if fault=='missing_promised' and ref==c['wire_history'][-2]['catalog_ref']:continue
        raw=(base/ref['path']).read_bytes();suffix='.parquet' if ref['media_type']=='application/vnd.apache.parquet' else '.json';name='objects/'+ref['sha256']+suffix
        members[name]=raw
        if not any(row['path']==name for row in manifest['files']):manifest['files'].append(dict(path=name,sha256=ref['sha256'],bytes=len(raw),media_type=ref['media_type']))
        technical.append(dict(scope=scope,reference=copy.deepcopy(ref),object=name))
        descriptor=next((row for row in v['catalog']['artifacts'] if row['ref']==ref),None)
        record['included'].append(dict(source_scope=scope,source_path=ref['path'],role='performance.provenance' if descriptor is None else descriptor['role'],
            schema=ref['schema'],sha256=ref['sha256'],object=name,dependencies=[],table=None if descriptor is None else descriptor['table']))
    if fault=='wrong_scope':cutoff['scope']=record['scope']='execution_attempt'
    if fault=='missing_cutoff_fields':cutoff.pop('technical_contract');record.pop('technical_contract')
    if fault=='forged_locator':
        for row in technical:
            if row['reference']==v['summary_ref']:
                row['reference']['path']='retained/forged-summary-alias.json'
        for row in record['included']:
            if row['source_path']==v['summary_ref']['path']:row['source_path']='retained/forged-summary-alias.json'
    manifest.update(technical_locator=technical,cutoff=cutoff,telemetry_compaction=[])
    manifest['records'].append(record)
    manifest['snapshot_identity']=hashlib.sha256(json_bytes(dict(source=manifest['source_state'],profile=profile,cutoff=cutoff,
        technical=[dict(scope=row['scope'],reference=row['reference']) for row in technical]))).hexdigest()
    spool=tmp_path/'wire-spool';spool.mkdir();files={}
    for row in manifest['files']:
        path=spool/Path(row['path']).name;path.write_bytes(members[row['path']]);files[row['path']]=(path,row)
    if compact:
        from qcl_negf_results.compaction import compact_export_records
        proofs=compact_export_records([record],files,spool,retain_original=True)
        assert len(proofs)==1 and proofs[0]['rows']==4
        if fault=='projection_order':
            proofs[0]['source_segments'].reverse()
            for index,source in enumerate(proofs[0]['source_segments']):source.update(row_start=2*index,row_stop=2*index+2)
        if fault=='projection_bits':
            target,descriptor=files[proofs[0]['object']]
            table=pq.read_table(target);column=table['cpu_time_seconds'].to_pylist();column[0]=0.0
            changed=table.set_column(table.schema.get_field_index('cpu_time_seconds'),'cpu_time_seconds',pa.array(column,type=pa.float64()))
            pq.write_table(changed,target)
            raw=target.read_bytes();digest=hashlib.sha256(raw).hexdigest();oldname=proofs[0]['object'];name='objects/'+digest+'.parquet'
            descriptor.update(path=name,sha256=digest,bytes=len(raw));files[name]=files.pop(oldname)
            proofs[0].update(object=name,compacted_bytes=len(raw))
            for row in record['included']:
                if row['object']==oldname:row.update(object=name,sha256=digest)
        manifest['telemetry_compaction']=proofs
        manifest['files']=[row for _,row in files.values()]
    metadata={name:raw for name,raw in members.items() if name in ('plan.json','diagnostics.json','README.md')}
    metadata['manifest.json']=json_bytes(manifest)
    path,_,index=build_archive(spool,files,metadata,identity=manifest['snapshot_identity'],profile=profile)
    return path,c,manifest,index


def test_S2_offline_v2_archive_receive_reexport_keeps_opaque_historical_cut(tmp_path):
    import shutil
    from qcl_negf_results.archive import receive,build_archive
    from qcl_negf_results.commits import json_bytes
    archive,c,manifest,index=s2_archive_wire(tmp_path,parquet=True)
    shutil.rmtree(c['root'])  # All original source/canonical/technical roots gone.
    result=receive([archive],tmp_path/'offline')
    assert result['verified'] is True
    assert result['technical_verification']['scope']=='retained_technical_window'
    assert result['technical_verification']['provenance_origin']=='not_authenticated_offline'
    assert not any(row['reference']['path'].endswith('catalog-1.json') for row in manifest['technical_locator'])
    restored=tmp_path/'offline';spool=tmp_path/'repack';spool.mkdir()
    files={row['path']:(restored/row['path'],row) for row in manifest['files']}
    metadata={name:(restored/name).read_bytes() for name in ('manifest.json','README.md') if (restored/name).exists()}
    repacked,_,_=build_archive(spool,files,metadata,identity=manifest['snapshot_identity'],profile='science')
    assert receive([repacked])['technical_verification']==result['technical_verification']


@pytest.mark.parametrize('fault',['missing_promised','extra_unselected','forged_cost','wrong_origin','wrong_scope','missing_cutoff_fields','footer_rows','forged_locator'])
def test_S2_offline_receive_rejects_hash_valid_but_invalid_retained_wire(tmp_path,fault):
    from qcl_negf_results.archive import receive
    path,c,manifest,index=s2_archive_wire(tmp_path,fault=fault)
    with pytest.raises(ContractError):receive([path],tmp_path/'rejected')
    assert not (tmp_path/'rejected').exists()


@pytest.mark.parametrize('fault',[None,'projection_order','projection_bits'])
def test_S2_compacted_archive_binds_catalog_order_and_actual_scalar_bits(tmp_path,fault):
    from qcl_negf_results.archive import receive
    path,c,manifest,index=s2_archive_wire(tmp_path,compact=True,fault=fault)
    if fault is None:
        assert receive([path])['technical_verification']['scope']=='retained_technical_window'
    else:
        with pytest.raises(ContractError):receive([path])


def test_S2_full_state_receive_restores_science_and_same_retained_technical_cut(tmp_path):
    import shutil
    from qcl_negf_results.archive import receive
    from qcl_negf_results.state import StateReader
    path,c,manifest,index=s2_archive_wire(tmp_path,profile='full-state')
    shutil.rmtree(c['root'])
    result=receive([path],tmp_path/'restored')
    assert result['technical_verification']['scope']=='retained_technical_window'
    assert result['source_state']==manifest['source_state']
    assert manifest['canonical_source_complete'] is True and manifest['scientific_accepted'] is None
    with StateReader(result['source_root'],state_ref=result['source_state'],expected_identity=result['source_state']['identity']) as reader:
        reader.import_history()
        assert reader.history_batch(maximum_records=1).records[0]['schema']=='qcl-negf.scientific-record.v1'
