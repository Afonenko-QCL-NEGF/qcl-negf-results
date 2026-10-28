"""Native5 export and metadata limits use explicit selected sources, no giant RAM arrays."""
import hashlib
import json
import tarfile
from types import SimpleNamespace

import numpy as np
import pytest
from qcl_negf_contracts.messages import ContractError
from qcl_negf_results.commits import decode_json
from qcl_negf_results.export import export_snapshot
from qcl_negf_results.native import dataset_blocks
from qcl_negf_results.state import StateReader, ResultReadLimits
from result_v2_fixture import source, record, array_source, IDENTITY


def test_large_typed_record_metadata_is_exported_byte_for_byte(tmp_path):
    row = record()
    row['core']['fields']['missing_reasons'] = [dict(type='Type_5a33f89ccbc22e86',
        fields=dict(first=dict(kind='symbol',value='diagnostic-'+str(i)),
                    second=dict(kind='symbol',value='not_measured'))) for i in range(1500)]
    ref = source(tmp_path / 'source', [row])
    payload = (tmp_path / 'source/records.h5').read_bytes()
    assert len(payload) > 100_000
    with StateReader(tmp_path / 'source', state_ref=ref, expected_identity=IDENTITY) as reader:
        receipt = export_snapshot(reader, tmp_path / 'exports', profile='science', byte_budget=8*1024**2, reserve_bytes=0)
    with tarfile.open(tmp_path / 'exports' / receipt['archive'], 'r:xz') as archive:
        assert archive.extractfile('objects/'+hashlib.sha256(payload).hexdigest()+'.h5').read() == payload


def test_raw_payload_above_450mb_streams_to_one_small_archive(tmp_path):
    ref = array_source(tmp_path / 'source', 56_300_000)
    assert (tmp_path / 'source/arrays.h5').stat().st_size > 450_000_000
    with StateReader(tmp_path / 'source', state_ref=ref, expected_identity=IDENTITY,
                     limits=ResultReadLimits(maximum_io_bytes=1024**3)) as reader:
        receipt = export_snapshot(reader, tmp_path / 'exports', profile='science', byte_budget=1024**3, reserve_bytes=0)
    assert receipt['bytes'] < 1_000_000
    assert receipt['snapshot_consistent'] and not receipt['canonical_source_complete']
    assert len(list((tmp_path / 'exports').glob('*.tar.xz'))) == 1


@pytest.mark.parametrize('payload', [b'{"x":1,"x":2}', b'{"x":NaN}', b'{"x":1e999}',
    b'{"x":[1,]}', b'{"x":1} trailing', b'[]', b'{"x":"\xff"}'])
def test_metadata_json_retains_integrity_validation(payload):
    with pytest.raises(ContractError):
        decode_json(payload, 'selected metadata')


def test_wide_native_dataset_is_read_in_bounded_blocks():
    dataset = SimpleNamespace(ndim=2, shape=(3, 2_000_000), dtype=np.dtype('float64'))
    blocks = list(dataset_blocks(dataset))
    sizes = [np.prod([part.stop-part.start for part in block]) for block in blocks]
    assert max(sizes)*8 <= 1024*1024
    assert sum(sizes) == 6_000_000
