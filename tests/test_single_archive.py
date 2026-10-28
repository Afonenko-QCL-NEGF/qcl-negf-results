"""One immutable selected source, exact single archive and atomic publication."""
import hashlib
import io
import json
from pathlib import Path
import tarfile
import pytest
from qcl_negf_contracts.messages import ContractError
from qcl_negf_results.archive import receive
from qcl_negf_results.commits import json_bytes
from result_v2_fixture import source, record, export_source, append, pointer
from test_export import unpack


def make_export(tmp_path):
    root, output = tmp_path / 'source', tmp_path / 'export'
    ref = source(root, [record()])
    return root, ref, output, export_source(root, ref, output)


def rewrite_archive(path, destination, mutate):
    with tarfile.open(path, 'r:xz') as source:
        members = [(member,source.extractfile(member).read()) for member in source]
    index = json.loads(members[0][1])
    payloads = {member.name:value for member,value in members}
    mutate(payloads)
    for row in index['metadata']:
        payload = payloads[row['path']]
        row.update(bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest())
    payloads['export-index.json'] = json_bytes(index)
    with tarfile.open(destination, 'w:xz') as target:
        for member,_ in members:
            value = payloads[member.name]
            member.size = len(value)
            target.addfile(member,io.BytesIO(value))


def test_export_publishes_one_archive_and_exact_inventory(tmp_path):
    root, ref, output, receipt = make_export(tmp_path)
    path = output / receipt['archive']
    assert list(output.glob('*.tar.xz')) == [path]
    assert receipt['bytes'] == path.stat().st_size
    assert receipt['sha256'] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert not {'parts','multipart','part_count','maximum_part_bytes'} & receipt.keys()
    manifest, members = unpack(receipt,output)
    assert 'part.json' not in members
    assert manifest['source_state'] == ref
    assert set(members) == {'export-index.json','manifest.json','README.md'} | {row['path'] for row in manifest['files']}
    for row in manifest['source_locator']:
        assert members[row['object']] == (root / row['reference']['path']).read_bytes()


def test_science_receive_returns_selected_subset_not_canonical_source(tmp_path):
    _, _, output, receipt = make_export(tmp_path)
    result = receive([output / receipt['archive']], tmp_path / 'restored', receipt=receipt)
    assert result['verified'] and result['archive_count'] == 1
    assert result['source_root'] is None and result['source_state'] is None
    assert result['verification_scope'] == 'transport_hashes_and_hdf5_storage'
    assert not (tmp_path / 'restored/source').exists()


@pytest.mark.parametrize('restore',[False,True])
def test_receipt_profile_must_match_archive_before_restore(tmp_path,restore):
    _, _, output, receipt = make_export(tmp_path)
    changed = dict(receipt,profile='full-state')
    with pytest.raises(ContractError,match='receipt and archive'):
        receive([output / receipt['archive']],tmp_path / 'restored' if restore else None,receipt=changed)
    assert not (tmp_path / 'restored').exists()


def test_missing_inventory_and_bad_checksum_are_rejected(tmp_path):
    _, _, output, receipt = make_export(tmp_path)
    path = output / receipt['archive']
    with pytest.raises(ContractError,match='checksum'):
        receive([path],receipt=dict(receipt,sha256='0'*64))
    damaged = tmp_path / 'missing.tar.xz'
    with tarfile.open(path,'r:xz') as old, tarfile.open(damaged,'w:xz') as new:
        for member in old:
            if not member.name.startswith('objects/'):
                new.addfile(member,old.extractfile(member))
    with pytest.raises(ContractError,match='omitted'):
        receive([damaged],tmp_path / 'bad')
    assert not (tmp_path / 'bad').exists()


@pytest.mark.parametrize('damage',['schema','complete','escape','duplicate','locator_hash'])
def test_manifest_and_source_locator_cannot_overclaim_closure(tmp_path,damage):
    _, _, output, receipt = make_export(tmp_path)
    damaged = tmp_path / 'wrong.tar.xz'
    def change(payloads):
        manifest = json.loads(payloads['manifest.json'])
        if damage=='schema': manifest['schema']='qcl-negf.operational-evidence.v2'
        elif damage=='complete': manifest['canonical_source_complete']=True
        elif damage=='escape': manifest['source_locator'][0]['reference']['path']='../escape'
        elif damage=='duplicate': manifest['source_locator'].append(manifest['source_locator'][0])
        else: manifest['source_locator'][0]['reference']['sha256']='0'*64
        payloads['manifest.json']=json_bytes(manifest)
    rewrite_archive(output / receipt['archive'],damaged,change)
    with pytest.raises(ContractError):
        receive([damaged],tmp_path / 'bad')
    assert not (tmp_path / 'bad').exists()


def test_xz_footer_is_verified_after_valid_tar_members(tmp_path):
    _, _, output, receipt = make_export(tmp_path)
    raw = bytearray((output / receipt['archive']).read_bytes())
    raw[-8] ^= 1
    damaged = tmp_path / 'footer.tar.xz'
    damaged.write_bytes(raw)
    with pytest.raises((ContractError,EOFError)):
        receive([damaged])


def test_scientific_publication_syncs_before_completed(tmp_path,monkeypatch):
    import qcl_negf_results.export as exporter
    root, output = tmp_path / 'source', tmp_path / 'export'
    ref = source(root,[record()])
    events=[]
    replace=exporter.os.replace
    def track(a,b):
        result=replace(a,b)
        if Path(b).parent==output: events.append('receipt' if Path(b).suffix=='.json' else 'archive')
        return result
    monkeypatch.setattr(exporter.os,'replace',track)
    monkeypatch.setattr(exporter,'fsync_directory',lambda path: events.append('sync'))
    export_source(root,ref,output,progress=lambda row: events.append('completed') if row['phase']=='completed' else None)
    assert events==['archive','receipt','sync','completed']


@pytest.mark.parametrize('failure',['compression','receipt','sync','cancel'])
def test_publication_failures_leave_no_new_pair(tmp_path,monkeypatch,failure):
    import qcl_negf_results.export as exporter
    import qcl_negf_results.archive as transport
    root,output=tmp_path / 'source',tmp_path / 'export'
    ref=source(root,[record()])
    class Cancelled(Exception): pass
    def progress(row):
        if failure=='cancel' and row['phase']=='compressing': raise Cancelled
    if failure=='compression':
        def full(*args,**kw): raise OSError(28,'device full')
        monkeypatch.setattr(transport,'_write_archive',full)
    elif failure=='receipt':
        replace=exporter.os.replace
        def full(a,b):
            if Path(b).parent==output and Path(b).suffix=='.json': raise OSError(28,'receipt full')
            return replace(a,b)
        monkeypatch.setattr(exporter.os,'replace',full)
    elif failure=='sync':
        def full(path): raise OSError(5,'fsync failed')
        monkeypatch.setattr(exporter,'fsync_directory',full)
    with pytest.raises((OSError,Cancelled)):
        export_source(root,ref,output,progress=progress)
    assert list(output.iterdir())==[]


def test_failed_sync_preserves_prior_receipt_and_cas(tmp_path,monkeypatch):
    import qcl_negf_results.export as exporter
    root,ref,output,receipt=make_export(tmp_path)
    previous={p.name:p.read_bytes() for p in output.iterdir()}
    def fail(path): raise OSError(5,'fsync failed')
    monkeypatch.setattr(exporter,'fsync_directory',fail)
    with pytest.raises(OSError): export_source(root,ref,output)
    assert {p.name:p.read_bytes() for p in output.iterdir()}==previous


def test_new_pointer_during_compression_does_not_reselect_frozen_state(tmp_path):
    root,output=tmp_path / 'source',tmp_path / 'export'
    ref=source(root,[record()])
    advanced=[]
    def progress(row):
        if row['phase']=='compressing' and not advanced:
            later=append(root,ref,record(2,99))
            pointer(root,later)
            advanced.append(later)
    receipt=export_source(root,ref,output,progress=progress)
    manifest,_=unpack(receipt,output)
    assert advanced and manifest['source_state']==ref
    assert export_source(root,advanced[0],output)['snapshot_identity']!=receipt['snapshot_identity']


def test_corrupt_output_cache_is_replaced_only_by_new_verified_bytes(tmp_path):
    root,ref,output,receipt=make_export(tmp_path)
    (output / receipt['archive']).write_bytes(b'corrupt cache')
    repaired=export_source(root,ref,output)
    assert receive([output / repaired['archive']],receipt=repaired)['verified']


def test_diagnostic_publication_fsyncs_contents_then_final_parent(tmp_path: Path, monkeypatch) -> None:
    import io
    import qcl_negf_results.diagnostic_archive as exporter
    destination = tmp_path / "evidence"
    events = []
    rename = exporter.os.rename
    def capture(sink):
        member = tarfile.TarInfo("log.txt")
        member.size = 3
        sink.addfile(member, io.BytesIO(b"log"))
        return {"run": "fixture"}
    def track_rename(source, target):
        result = rename(source, target)
        if Path(target) == destination:
            events.append("published")
        return result
    def sync(path):
        events.append("parent synced" if path == destination.parent else "contents synced")
    monkeypatch.setattr(exporter.os, "rename", track_rename)
    monkeypatch.setattr(exporter, "fsync_directory", sync, raising=False)
    exporter.export_diagnostics(destination, capture, label="fixture")
    assert events == ["contents synced", "published", "parent synced"]



def test_incompressible_transport_above_200mb_stays_one_archive(tmp_path):
    """Transport capacity is separate from interpreting opaque evidence as science."""
    import random
    from qcl_negf_contracts.artifacts import CONTRACT_SET, EXPORT_SCHEMA
    from qcl_negf_results.archive import build_archive
    from qcl_negf_results.export_budget import ExportBudget
    from result_v2_fixture import encoded, reference
    root, spool = tmp_path / 'source', tmp_path / 'spool'
    ref = source(root, [])
    spool.mkdir()
    evidence = root / 'manufactured-opaque.bin'
    generator = random.Random(79)
    with evidence.open('wb') as output:
        for _ in range(201):
            output.write(generator.randbytes(1024*1024))
    opaque = reference(root,evidence.name,'constructor-opaque-v1','application/octet-stream')
    rows = [ref['manifest'],opaque]
    mapping = {row['path']:'objects/'+row['sha256']+('.json' if row['media_type']=='application/json' else '.bin') for row in rows}
    files = {mapping[row['path']]:(root / row['path'],dict(path=mapping[row['path']],
        bytes=int(row['bytes']),sha256=row['sha256'],media_type=row['media_type'])) for row in rows}
    manifest = dict(schema=EXPORT_SCHEMA,contract_set=CONTRACT_SET,profile='science',snapshot_identity='2'*64,
        source_state=ref,canonical_source_complete=False,files=[item[1] for item in files.values()],
        records=[dict(included=[dict(object=mapping[row['path']],dependencies=[]) for row in rows])],
        source_locator=[dict(reference=row,object=mapping[row['path']]) for row in rows])
    archive,info,index=build_archive(spool,files,{'manifest.json':encoded(manifest)},
        identity='2'*64,profile='science',budget=ExportBudget(spool,byte_budget=1024**3,reserve_bytes=0))
    assert info['bytes']>200_000_000
    assert len(list(spool.glob('*.tar.xz')))==1
    result=receive([archive])
    assert result['verified'] and result['source_root'] is None
    with tarfile.open(archive,'r:xz') as transport:
        assert hashlib.file_digest(transport.extractfile(mapping[evidence.name]),'sha256').hexdigest()==opaque['sha256']
    with evidence.open('rb') as original:
        assert hashlib.file_digest(original,'sha256').hexdigest()==opaque['sha256']
