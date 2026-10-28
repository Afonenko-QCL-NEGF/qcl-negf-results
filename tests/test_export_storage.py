"""Hash-valid native5 bytes still cannot use unowned numerical storage."""
import json
import h5py
import numpy as np
import pytest
from qcl_negf_contracts.artifacts import CONTRACT_SET, EXPORT_SCHEMA
from qcl_negf_contracts.messages import ContractError
from qcl_negf_results.state import StateReader
from qcl_negf_results.export import _selection
from qcl_negf_results.archive import build_archive, receive
from result_v2_fixture import IDENTITY, source, record, reference, put, encoded, export_source


def unowned_fixture(tmp_path, storage):
    root = tmp_path / 'source'
    ref = source(root, [record()])
    payload = root / 'records.h5'
    if storage == 'wrong_native':
        with h5py.File(payload, 'r+') as handle:
            handle['metadata'].attrs['schema_version'] = np.bytes_('4.0')
    elif storage == 'external':
        raw = tmp_path / 'unowned.raw'
        raw.write_bytes(np.array([3,7,11], dtype='<i4').tobytes())
        with h5py.File(payload, 'r+') as handle:
            handle.create_dataset('outside', shape=(3,), dtype='<i4', external=[(str(raw),0,h5py.h5f.UNLIMITED)])
    else:
        raw = tmp_path / 'unowned.h5'
        with h5py.File(raw, 'w') as handle:
            handle['values'] = np.array([3,7,11], dtype='<i4')
        layout = h5py.VirtualLayout(shape=(3,), dtype='<i4')
        layout[:] = h5py.VirtualSource(str(raw), 'values', shape=(3,))
        with h5py.File(payload, 'r+') as handle:
            handle.create_virtual_dataset('outside', layout)
    page = json.loads((root / 'catalog.json').read_text())
    page['artifacts'][-1].update(reference(root, 'records.h5', page['artifacts'][-1]['schema'], 'application/x-hdf5'))
    manifest = json.loads((root / 'manifest.json').read_text())
    manifest['history']['head'] = put(root, 'catalog.json', page, page['schema'])
    ref['manifest'] = put(root, 'manifest.json', manifest, manifest['schema'])
    return root, ref, payload.read_bytes()


@pytest.mark.parametrize('storage', ['external','virtual','wrong_native'])
def test_export_rejects_numerical_storage_outside_hdf5_closure(tmp_path, storage):
    root, ref, original = unowned_fixture(tmp_path, storage)
    with pytest.raises(ContractError, match='external|virtual|native5'):
        export_source(root, ref, tmp_path / 'export')
    assert not list((tmp_path / 'export').iterdir())
    assert (root / 'records.h5').read_bytes() == original


@pytest.mark.parametrize('storage', ['external','virtual','wrong_native'])
@pytest.mark.parametrize('restore', [False,True])
def test_receiver_independently_rejects_hash_valid_unowned_storage(tmp_path, storage, restore):
    root, ref, original = unowned_fixture(tmp_path, storage)
    with StateReader(root, state_ref=ref, expected_identity=IDENTITY) as reader:
        selected, _, _ = _selection(reader, 'science')
    # Hand-build a malicious but hash-consistent transport with the existing
    # archive builder. No producer validator is monkeypatched or trusted here.
    objects, mapping = {}, {}
    for path, row in selected.items():
        name = 'objects/' + row['sha256'] + ('.h5' if row['media_type']=='application/x-hdf5' else '.json')
        mapping[path] = name
        objects[name] = (root / path, dict(path=name, bytes=int(row['bytes']), sha256=row['sha256'], media_type=row['media_type']))
    manifest = dict(schema=EXPORT_SCHEMA, contract_set=CONTRACT_SET, profile='science', snapshot_identity='1'*64,
        canonical_source_complete=False, source_state=ref, files=[item[1] for item in objects.values()],
        records=[dict(included=[dict(object=mapping[path], dependencies=[]) for path in selected])],
        source_locator=[dict(reference={k:row[k] for k in ('path','bytes','sha256','schema','media_type')},
            object=mapping[path]) for path,row in selected.items()])
    spool = tmp_path / 'spool'
    spool.mkdir()
    path, _, _ = build_archive(spool, objects, {'manifest.json':encoded(manifest)},
        identity='1'*64, profile='science')
    destination = tmp_path / 'restored' if restore else None
    with pytest.raises(ContractError, match='external|virtual|native5'):
        receive([path], destination)
    assert destination is None or not destination.exists()
    assert not list(tmp_path.glob('.qcl-receive-*'))
    assert (root / 'records.h5').read_bytes() == original
