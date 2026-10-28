import copy

import pytest

from qcl_negf_contracts.messages import ContractError
from qcl_negf_results import state
from result_v2_fixture import IDENTITY, source, record, producer_root, producer_state, append, pointer, put


def test_cursor_resume_uses_physical_record_index_and_detects_tamper(tmp_path):
    ref = source(tmp_path, [record(1, 7), record(2, 99)])
    assert hasattr(state, "ResultReadLimits"), "missing bounded native5 reader API"
    with state.StateReader(tmp_path, state_ref=ref, expected_identity=IDENTITY) as reader:
        reader.import_history()
        first = reader.history_batch(maximum_records=1)
        assert first.cursor["row_offset"] == "0" and first.cursor["last_record"] == "1"
        second = reader.history_batch(first.cursor)
        assert second.records[0]["stream_ordinal"] == "2" and second.at_cutoff
        forged = copy.deepcopy(first.cursor)
        forged["row_offset"] = "1"
        with pytest.raises(ContractError):
            reader.history_batch(forged)


def test_parent_update_uses_pinned_immediate_parent(producer_root):
    assert hasattr(state, "ResultReadLimits"), "missing bounded native5 reader API"
    for name in ("child", "grandchild"):
        ref = producer_state(producer_root, name)
        with state.StateReader(producer_root, state_ref=ref, expected_identity=ref["identity"]) as reader:
            reader.import_history()
            result = reader.history_batch()
            assert len(result.records) == 1
            assert result.records[0]["target"]["scope"] == "parent_checkpoint"
            assert result.cursor["last_record"] == "1"


def test_pointer_poll_is_metadata_only_and_rebase_preserves_record_position(tmp_path, monkeypatch):
    ref = source(tmp_path, [record()])
    pointer(tmp_path, ref)
    with state.StateReader.from_pointer(tmp_path, expected_identity=IDENTITY, pointer='live') as reader:
        reader.import_history()
        cursor = reader.history_batch().cursor
        import qcl_negf_results._capture_io as io
        original, opened = io.regular, []
        from contextlib import contextmanager
        @contextmanager
        def observed(path, **kwargs):
            opened.append(path.name)
            with original(path, **kwargs) as stream:
                yield stream
        monkeypatch.setattr(io, 'regular', observed)
        for _ in range(10):
            reader.poll()
        assert opened == ['current.json'] * 10
        second = append(tmp_path, ref, record(2, 99))
        pointer(tmp_path, second)
        opened.clear()
        reader.poll()
        assert set(opened) == {'current.json', 'manifest-2.json', 'catalog-2.json'}
        cursor = reader.rebase_cursor(cursor)
        assert cursor['row_offset'] == '0' and cursor['last_record'] == '1'
        assert reader.history_batch(cursor).records[0]['stream_ordinal'] == '2'


def test_nonadjacent_catalog_path_republication_rejected(tmp_path):
    import json
    first = source(tmp_path, [record()])
    second = append(tmp_path, first, record(2, 9))
    third = append(tmp_path, second, record(3, 11))
    m = json.loads((tmp_path / third['manifest']['path']).read_text())
    page = json.loads((tmp_path / m['history']['head']['path']).read_text())
    old = json.loads((tmp_path / 'catalog.json').read_text())
    page['artifacts'] = old['artifacts']
    m['history']['head'] = put(tmp_path, 'reused.json', page, page['schema'])
    third['manifest'] = put(tmp_path, 'invalid.json', m, m['schema'])
    with state.StateReader(tmp_path, state_ref=third, expected_identity=IDENTITY) as reader:
        with pytest.raises(ContractError, match='nonadjacent'):
            reader.import_history()


def test_selected_batch_and_metadata_caps_do_not_advance_cursor(tmp_path):
    from dataclasses import replace
    ref = source(tmp_path, [record(), record(2, 9)])
    limits = state.ResultReadLimits(maximum_record_bytes=10)
    with state.StateReader(tmp_path, state_ref=ref, expected_identity=IDENTITY, limits=limits) as reader:
        reader.import_history()
        with pytest.raises(ContractError, match='record JSON'):
            reader.history_batch()
        assert reader.source.history['last_record'] == '2'
    limits = replace(limits, maximum_metadata_bytes=10)
    with pytest.raises(ContractError, match='metadata'):
        state.StateReader(tmp_path, state_ref=ref, expected_identity=IDENTITY, limits=limits)


def test_parent_proof_bytes_share_the_same_physical_operation_budget(producer_root, monkeypatch):
    ref = producer_state(producer_root, 'child')
    with state.StateReader(producer_root, state_ref=ref, expected_identity=ref['identity']) as reader:
        reader.import_history()
        delivered = [0]
        read, readinto = state._BudgetStream.read, state._BudgetStream.readinto
        def counted_read(stream, size=-1):
            value = read(stream, size)
            delivered[0] += len(value)
            return value
        def counted_readinto(stream, target):
            value = readinto(stream, target)
            delivered[0] += value
            return value
        monkeypatch.setattr(state._BudgetStream, 'read', counted_read)
        monkeypatch.setattr(state._BudgetStream, 'readinto', counted_readinto)
        batch = reader.history_batch()
        assert batch.io_counters['bytes_read_total'] == delivered[0]


def test_new_poll_reuses_prefix_metadata_without_lifetime_rescan(tmp_path):
    first = source(tmp_path, [record()])
    second = append(tmp_path, first, record(2, 9))
    pointer(tmp_path, second)
    class PrefixWithoutTraversal(list):
        def __iter__(self):
            raise AssertionError('poll rescanned imported page prefix')
    with state.StateReader.from_pointer(tmp_path, expected_identity=IDENTITY, pointer='live') as reader:
        reader.import_history()
        reader._pages = PrefixWithoutTraversal(reader._pages)
        third = append(tmp_path, second, record(3, 11))
        pointer(tmp_path, third)
        assert reader.poll().history['last_record'] == '3'


def test_cursor_batch_locates_delta_without_traversing_imported_prefix(tmp_path):
    ref = source(tmp_path, [record()])
    for n in range(2, 6):
        ref = append(tmp_path, ref, record(n, n*10))
    with state.StateReader(tmp_path, state_ref=ref, expected_identity=IDENTITY) as reader:
        reader.import_history()
        first = reader.history_batch(maximum_records=4)
        class PrefixWithoutTraversal(list):
            def __iter__(self):
                raise AssertionError('cursor batch traversed prior page prefix')
        reader._pages = PrefixWithoutTraversal(reader._pages)
        assert reader.history_batch(first.cursor).records[0]['stream_ordinal'] == '5'


def test_failed_catalog_category_proof_cannot_be_bypassed_by_retry(tmp_path):
    import json
    ref = source(tmp_path, [record()])
    page = json.loads((tmp_path / 'catalog.json').read_text())
    page['totals'].update(scba_rows='0', outer_rows='1')
    manifest = json.loads((tmp_path / 'manifest.json').read_text())
    manifest['history'].update(head=put(tmp_path, 'catalog.json', page, page['schema']), scba_rows='0', outer_rows='1')
    ref['manifest'] = put(tmp_path, 'manifest.json', manifest, manifest['schema'])
    with state.StateReader(tmp_path, state_ref=ref, expected_identity=IDENTITY) as reader:
        reader.import_history()
        for _ in range(2):
            with pytest.raises(ContractError, match='categories'):
                reader.history_batch()


@pytest.mark.parametrize('cursor', [{}, {'leaf':{}}, {'leaf':None}])
def test_cursor_schema_checked_before_any_untrusted_position_access(tmp_path, cursor):
    ref = source(tmp_path, [record()])
    with state.StateReader(tmp_path, state_ref=ref, expected_identity=IDENTITY) as reader:
        reader.import_history()
        with pytest.raises(ContractError):
            reader.history_batch(cursor)
        with pytest.raises(ContractError):
            reader.rebase_cursor(cursor)


def test_closed_record_coordinates_verified_once_per_open_handle(tmp_path, monkeypatch):
    from qcl_negf_results import native
    ref = source(tmp_path, [record(), record(2, 9)])
    original, validations = native.record_table, []
    def counted(handle, **kwargs):
        validations.append(handle.filename)
        return original(handle, **kwargs)
    monkeypatch.setattr(native, 'record_table', counted)
    with state.StateReader(tmp_path, state_ref=ref, expected_identity=IDENTITY) as reader:
        reader.import_history()
        first = reader.history_batch(maximum_records=1)
        reader.history_batch(first.cursor)
        assert len(validations) == 1


def test_retained_metadata_cap_accounts_decoded_context_not_only_wire_json(tmp_path):
    ref = source(tmp_path, [record()])
    limit = 2 * (tmp_path / 'catalog.json').stat().st_size
    with pytest.raises(ContractError) as error:
        with state.StateReader(tmp_path, state_ref=ref, expected_identity=IDENTITY,
                limits=state.ResultReadLimits(maximum_catalog_bytes=limit)) as reader:
            reader.import_history()
    assert error.value.code == 'budget_exceeded'


def test_history_only_child_import_resolves_exact_parent_anchor_without_hdf(producer_root, tmp_path, monkeypatch):
    import json
    import h5py
    from result_v2_fixture import copy_constructor_closure
    ref = producer_state(producer_root, 'child')
    root = copy_constructor_closure(producer_root, tmp_path / 'source', ref)
    manifest = json.loads((root / ref['manifest']['path']).read_text())
    manifest.update(checkpoint=None, numerical_binding=None, artifacts=[])
    ref['manifest'] = put(root, 'history-only-child.json', manifest, manifest['schema'])
    with state.StateReader(root, state_ref=ref, expected_identity=ref['identity']) as reader:
        def forbidden(*args, **kw):
            raise AssertionError('catalog import opened HDF payload')
        with monkeypatch.context() as context:
            context.setattr(h5py, 'File', forbidden)
            reader.import_history()
        assert reader.history_batch().records[0]['target']['scope'] == 'parent_checkpoint'


@pytest.mark.parametrize('damage', ['missing_manifest', 'tamper_checkpoint', 'identity_mismatch'])
def test_history_only_child_rejects_unverified_parent_metadata(producer_root, tmp_path, damage):
    import json
    from result_v2_fixture import copy_constructor_closure
    ref = producer_state(producer_root, 'child')
    root = copy_constructor_closure(producer_root, tmp_path / 'source', ref)
    manifest = json.loads((root / ref['manifest']['path']).read_text())
    parent = manifest['lineage']['parent_state']
    parent_manifest = json.loads((root / parent['manifest']['path']).read_text())
    if damage == 'missing_manifest':
        (root / parent['manifest']['path']).unlink()
    elif damage == 'tamper_checkpoint':
        path = root / parent_manifest['checkpoint']['path']
        path.write_bytes(path.read_bytes() + b' ')
    else:
        parent['identity']['attempt'] = '999'
    manifest.update(checkpoint=None, numerical_binding=None, artifacts=[])
    ref['manifest'] = put(root, 'history-only-child.json', manifest, manifest['schema'])
    with pytest.raises(ContractError):
        with state.StateReader(root, state_ref=ref, expected_identity=ref['identity']) as reader:
            reader.import_history()


def _write_native_update_records(path, rows):
    import h5py
    import numpy as np
    from result_v2_fixture import encoded, metadata
    pieces = [encoded(row) for row in rows]
    with h5py.File(path, 'w') as handle:
        metadata(handle)
        handle['records/offsets'] = np.cumsum([0] + [len(row) for row in pieces], dtype=np.uint64)
        handle['records/json_utf8'] = np.frombuffer(b''.join(pieces), dtype=np.uint8)
        handle['records/stream_ordinal'] = np.array([int(row['stream_ordinal']) for row in rows], dtype='<u8')


def _sealed_update_context(root, *, tamper=False, mismatch=None, admission='catalog'):
    import json
    from qcl_negf_contracts.results import validate_scientific_record
    from result_v2_fixture import reference, symbol
    original = record()
    first = source(root, [original])
    ref = append(root, first, record(2, 9))
    manifest = json.loads((root / ref['manifest']['path']).read_text())
    page = json.loads((root / manifest['history']['head']['path']).read_text())
    canonical = reference(root, 'records.h5', 'qcl-negf-scientific-history-v5', 'application/x-hdf5')
    if tamper:
        changed = copy.deepcopy(original)
        changed['core']['fields']['row']['fields']['J']['bits'] = '4010000000000000'
        _write_native_update_records(root / 'records.h5', [changed])
    target = reference(root, 'records.h5', canonical['schema'], canonical['media_type'])
    if tamper:
        assert target['bytes'] == canonical['bytes'] and target['sha256'] != canonical['sha256']
    if admission == 'manifest_only':
        (root / 'orphan.h5').write_bytes((root / 'records.h5').read_bytes())
        target = reference(root, 'orphan.h5', canonical['schema'], canonical['media_type'])
        manifest['artifacts'] = [dict(target, role='science.history', profile='science', dependencies=[])]
    update = copy.deepcopy(original)
    update['stream_ordinal'] = '2'
    update['core']['fields']['sequence']['decimal'] = '9'
    update['core']['fields']['action'] = symbol('witness_update')
    update['target'] = dict(scope='sealed_leaf', source_identity=IDENTITY, source_epoch='epoch',
        append_ordinal='1', leaf=target, row_offset='0', anchor_state=None, anchor_history=None)
    validate_scientific_record(update)
    if mismatch:
        field, value = mismatch
        update['target']['leaf'][field] = value(target[field]) if callable(value) else value
    _write_native_update_records(root / 'records-2.h5', [update])
    leaf = reference(root, 'records-2.h5', canonical['schema'], canonical['media_type'])
    page['artifacts'] = [dict(leaf, role='science.history', profile='science',
        dependencies=[] if admission == 'missing_dependency' else ['records.h5'])]
    page['totals'].update(scba_rows='1', witness_updates='1')
    manifest['history'].update(head=put(root, 'catalog-2.json', page, page['schema']),
        scba_rows='1', witness_updates='1')
    ref['manifest'] = put(root, 'manifest-2.json', manifest, manifest['schema'])
    cursor = dict(schema='qcl-negf.result-cursor.v1', contract_set='qcl-negf.results.v2', identity=IDENTITY,
        invocation=manifest['invocation'], manifest=ref['manifest'], history=manifest['history'],
        leaf=leaf, row_offset='0', last_record='2')
    return ref, cursor


@pytest.mark.parametrize('tamper', [False, True])
@pytest.mark.parametrize('resume', [False, True])
def test_sealed_target_cursor_and_full_delivery_bind_original_catalog_bytes(tmp_path, tamper, resume):
    ref, cursor = _sealed_update_context(tmp_path, tamper=tamper)
    with state.StateReader(tmp_path, state_ref=ref, expected_identity=IDENTITY) as reader:
        reader.import_history()
        if tamper:
            with pytest.raises(ContractError):
                reader.history_batch(cursor if resume else None)
        else:
            batch = reader.history_batch(cursor if resume else None)
            assert batch.at_cutoff and len(batch.records) == (0 if resume else 2)


@pytest.mark.parametrize('mismatch', [('bytes', lambda value: str(int(value)+1)),
    ('sha256', '0'*64), ('schema', 'qcl-negf-checkpoint-v5'), ('media_type', 'application/octet-stream')])
def test_sealed_target_complete_fileref_mismatch_refused_before_target_open(tmp_path, monkeypatch, mismatch):
    import qcl_negf_results._capture_io as io
    ref, cursor = _sealed_update_context(tmp_path, mismatch=mismatch)
    regular = io.regular
    def no_target(path, **kwargs):
        assert path.name != 'records.h5', 'unbound target opened before exact canonical FileRef check'
        return regular(path, **kwargs)
    monkeypatch.setattr(io, 'regular', no_target)
    with state.StateReader(tmp_path, state_ref=ref, expected_identity=IDENTITY) as reader:
        reader.import_history()
        with pytest.raises(ContractError):
            reader.history_batch(cursor)


@pytest.mark.parametrize('admission', ['manifest_only', 'missing_dependency'])
def test_sealed_target_requires_catalog_record_leaf_and_declared_dependency(tmp_path, admission):
    ref, cursor = _sealed_update_context(tmp_path, admission=admission)
    with state.StateReader(tmp_path, state_ref=ref, expected_identity=IDENTITY) as reader:
        reader.import_history()
        with pytest.raises(ContractError):
            reader.history_batch(cursor)


@pytest.mark.parametrize('offset', ['0', '1'])
def test_same_leaf_update_retains_earlier_physical_append_rule(tmp_path, offset):
    from result_v2_fixture import symbol
    original = record()
    update = copy.deepcopy(original)
    update['stream_ordinal'] = '2'
    update['core']['fields']['sequence']['decimal'] = '9'
    update['core']['fields']['action'] = symbol('witness_update')
    update['target'] = dict(scope='same_leaf', source_identity=IDENTITY, source_epoch='epoch',
        append_ordinal='1', leaf=None, row_offset=offset, anchor_state=None, anchor_history=None)
    ref = source(tmp_path, [original, update])
    with state.StateReader(tmp_path, state_ref=ref, expected_identity=IDENTITY) as reader:
        reader.import_history()
        if offset == '0':
            assert len(reader.history_batch().records) == 2
        else:
            with pytest.raises(ContractError, match='earlier physical append'):
                reader.history_batch()


@pytest.mark.parametrize('stop_after', [2, 3])
def test_actual_producer_same_and_sealed_updates_allow_cold_cursor_resume(producer_root, stop_after):
    ref = producer_state(producer_root, 'updates')
    with state.StateReader(producer_root, state_ref=ref, expected_identity=ref['identity']) as reader:
        reader.import_history()
        batch = reader.history_batch(maximum_records=stop_after)
        scopes = [None if row['target'] is None else row['target']['scope'] for row in batch.records]
        assert scopes == [None, 'same_leaf', 'sealed_leaf'][:stop_after]
        cursor = batch.cursor
    with state.StateReader(producer_root, state_ref=ref, expected_identity=ref['identity']) as reader:
        reader.import_history()
        resumed = reader.history_batch(cursor)
        assert [row['stream_ordinal'] for row in resumed.records] == [str(i) for i in range(stop_after+1, 5)]
        assert resumed.at_cutoff
