"""Strict native5 headers and ownership, separate from independent audit subresources."""
import h5py
import numpy as np
import pytest
from qcl_negf_contracts.messages import ContractError
from qcl_negf_results.native import validate_native_handle, native_metadata
from qcl_negf_results.state import _internal_storage
from result_v2_fixture import metadata, IDENTITY


@pytest.mark.parametrize('field,value', [('schema_version','4.0'), ('contract_set','qcl-negf.results.v1'),
    ('artifact_role','physics.full'), ('schema','qcl-negf-physics-v4'), ('core_layout_sha256','0'*64),
    ('representation','unknown')])
def test_new_header_cannot_disguise_wrong_version_role_layout_or_representation(tmp_path, field, value):
    with h5py.File(tmp_path / 'native.h5', 'w') as handle:
        metadata(handle)
        handle['metadata'].attrs.modify(field, np.bytes_(value))
        with pytest.raises(ContractError):
            validate_native_handle(handle, 'science.history', 'qcl-negf-scientific-history-v5')


@pytest.mark.parametrize('field', ['schema','schema_version','contract_set','artifact_role','core_layout_sha256','representation'])
def test_each_native5_header_is_required(tmp_path, field):
    with h5py.File(tmp_path / 'native.h5', 'w') as handle:
        metadata(handle)
        del handle['metadata'].attrs[field]
        with pytest.raises(ContractError):
            validate_native_handle(handle)


def test_native_source_identity_and_epoch_are_bound_independently_of_filename(tmp_path):
    with h5py.File(tmp_path / 'native.h5', 'w') as handle:
        metadata(handle)
        with pytest.raises(ContractError):
            native_metadata(handle, identity=dict(IDENTITY, attempt='2'))
        with pytest.raises(ContractError):
            native_metadata(handle, invocation=dict(source_epoch='different'))


def test_unbounded_header_string_is_rejected_before_value_read(tmp_path, monkeypatch):
    with h5py.File(tmp_path / 'native.h5', 'w') as handle:
        metadata(handle)
        del handle['metadata'].attrs['schema_version']
        handle['metadata'].attrs.create('schema_version', '5.0', dtype=h5py.string_dtype())
        original = h5py.AttributeManager.__getitem__
        def guarded(attrs, key):
            assert key != 'schema_version', 'unbounded variable-length attribute read'
            return original(attrs, key)
        monkeypatch.setattr(h5py.AttributeManager, '__getitem__', guarded)
        with pytest.raises(ContractError):
            validate_native_handle(handle)


@pytest.mark.parametrize('storage', ['soft','external','raw','virtual'])
def test_unowned_values_are_rejected_without_numeric_read(tmp_path, storage):
    with h5py.File(tmp_path / 'native.h5', 'w') as handle:
        metadata(handle, representation='array_payload')
        if storage == 'soft':
            handle['values'] = h5py.SoftLink('/metadata')
        elif storage == 'external':
            handle['values'] = h5py.ExternalLink('foreign.h5','/values')
        elif storage == 'raw':
            handle.create_dataset('values', shape=(1,), dtype='<f8', external=[('foreign.raw',0,8)])
        else:
            layout = h5py.VirtualLayout(shape=(1,), dtype='<f8')
            layout[:] = h5py.VirtualSource('foreign.h5','/values', shape=(1,))
            handle.create_virtual_dataset('values', layout)
        with pytest.raises(ContractError):
            _internal_storage(handle)
