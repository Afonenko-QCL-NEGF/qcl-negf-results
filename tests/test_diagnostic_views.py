"""Tiny literal native-table oracles. Authoring this file performs no HDF I/O.

No solver/full validator/export/materialize/native fixture generator is required.
"""
import hashlib
import json

import h5py
import numpy as np
import pytest

from qcl_negf_contracts.messages import ContractError
from qcl_negf_results.commits import artifact_row, json_bytes
from qcl_negf_results.diagnostic_views import (
    DiagnosticViewLimits, describe_saved_diagnostics, read_saved_history,
)

IDENTITY = {'point_id': 'p0', 'execution_id': 'ex', 'attempt': 10,
            'plan_fingerprint': 'a' * 64}


def make_history(root, *, sequence=(100, 101, 102, 103, 104),
                 iteration=(1, 2, 3, 1, 2), outer=(1, 1, 1, 2, 2),
                 residual=(.1, .001, .0001, .2, .02), currents=(9000., 9100., 9200., 7000., 7100.),
                 sources=None, units=True, chunks=None):
    root.mkdir()
    path = root / 'history.h5'
    with h5py.File(path, 'w') as handle:
        header = handle.create_group('metadata')
        for name, value in {'schema': 'qcl-negf-scientific-history-v4', 'schema_version': '4.0',
                            'artifact_role': 'science.history', 'contract_set': 'qcl-negf.results.v1'}.items():
            header.attrs[name] = np.bytes_(value)
        header.attrs['scba_rows'] = len(sequence)
        header.attrs['outer_rows'] = 0
        if sources is None:
            header['identity_json'] = np.bytes_(json.dumps(IDENTITY))
        else:
            header['source_segments_json'] = np.bytes_(json.dumps(sources))
        for name, values in {'sequence': sequence, 'iteration': iteration, 'outer_iteration': outer}.items():
            handle.create_dataset('scba/' + name, data=np.asarray(values, dtype='i8'), chunks=chunks)
        for name, values in {'r_K': residual, 'J': currents}.items():
            column = handle.create_dataset('scba/' + name, data=np.asarray(values, dtype='f8'), chunks=chunks)
            if units:
                column.attrs['units'] = np.bytes_('A/m^2' if name == 'J' else 'dimensionless')
        for name in ('sequence', 'iteration', 'r_U', 'J'):
            handle['outer/' + name] = np.asarray([], dtype='i8' if name in ('sequence', 'iteration') else 'f8')
    owner = artifact_row(path, relative='history.h5', role='science.history',
                         media_type='application/x-hdf5', schema='qcl-negf-scientific-history-v4')
    commit = {'schema': 'qcl-negf.artifact-commit.v2', 'contract_set': 'qcl-negf.results.v1',
              'generation': 7, 'identity': IDENTITY, 'terminal_status': 'failed',
              'quality': 'unconverged', 'artifacts': [owner],
              'performance': {'checkpoint_payload_seconds': 2., 'checkpoint_payload_bytes': 4096,
                              'includes_commit_publication': False, 'jit_separately_measured': False}}
    target = root / 'commit.json'
    target.write_bytes(json_bytes(commit))
    return target


def expected(commit):
    return {**IDENTITY, 'source_commit_sha256': hashlib.sha256(commit.read_bytes()).hexdigest(),
            'run_uuid': 'run', 'calcjob_uuid': 'job'}


def select(start=0, stop=5, step=1, columns=('r_K', 'J')):
    return {'start': start, 'stop': stop, 'step': step, 'columns': list(columns)}


def rewrite_owner(commit):
    value = json.loads(commit.read_bytes())
    value['artifacts'][0] = artifact_row(commit.parent / 'history.h5', relative='history.h5',
            role='science.history', media_type='application/x-hdf5', schema='qcl-negf-scientific-history-v4')
    commit.write_bytes(json_bytes(value))


def source(attempt, count, first, last):
    return {'identity': {**IDENTITY, 'attempt': attempt}, 'sha256': str(attempt % 10) * 64,
            'scba_rows': count, 'outer_rows': 0,
            'scba_sequence_first': first, 'scba_sequence_last': last}


def test_descriptor_reads_no_numeric_payload_and_failed_needs_no_analysis(tmp_path, monkeypatch):
    commit = make_history(tmp_path / 'one')
    original = h5py.Dataset.__getitem__
    def reject_numeric(dataset, selection):
        if dataset.dtype.kind in 'if':
            pytest.fail('metadata descriptor selected numeric payload')
        return original(dataset, selection)
    monkeypatch.setattr(h5py.Dataset, '__getitem__', reject_numeric)
    data = describe_saved_diagnostics(commit, expected_identity=expected(commit))
    assert data['schema'] == 'qcl-negf.saved-diagnostics.v1'
    assert data['history']['status'] == 'available'
    assert data['history']['phases'][0]['total_rows'] == 5
    assert data['integrity']['owner_sha_verified'] is False
    assert data['recorded_statuses']['terminal_status'] == 'failed'
    assert data['recorded_statuses']['scientific_accepted'] is None
    assert data['resources']['checkpoint'] == {
        'checkpoint_payload_seconds': 2., 'checkpoint_payload_bytes': 4096,
        'includes_commit_publication': False, 'jit_separately_measured': False,
        'scope': 'point_commit_checkpoint_payload'}
    assert data['resources']['performance_summary'] == {'status': 'unavailable', 'reason': 'source_binding_missing'}


def test_exact_selected_rows_native_cycles_units_and_full_owner_hash(tmp_path):
    commit = make_history(tmp_path / 'one')
    data = read_saved_history(commit, 'scba', select(3, 5), expected_identity=expected(commit))
    assert data['coordinates'] == {'sequence': [103, 104], 'iteration': [1, 2],
                                   'outer_iteration': [2, 2], 'source_attempt': [10, 10]}
    assert data['columns'] == [
        {'name': 'r_K', 'units': 'dimensionless', 'values': [.2, .02], 'invalid_mask': [None, None]},
        {'name': 'J', 'units': 'A/m^2', 'values': [7000., 7100.], 'invalid_mask': [None, None]}]
    assert data['break_before'] == [True, False]
    assert data['coverage']['previous_coordinate_row_checked'] is True
    assert data['integrity']['owner_sha_verified'] is True
    assert data['io_counters']['sha_bytes'] >= (commit.parent / 'history.h5').stat().st_size
    all_rows = read_saved_history(commit, 'scba', select(), expected_identity=expected(commit))
    assert all_rows['break_before'] == [True, False, False, True, False]


def test_cumulative_attempts_are_chronological_not_filename_inventory_order(tmp_path):
    commit = make_history(tmp_path / 'one', sequence=(31, 32, 33, 34), iteration=(1, 2, 3, 4),
                          outer=(7, 7, 7, 7), residual=(.1, .2, .3, .4), currents=(1., 2., 3., 4.),
                          sources=[source(10, 2, 33, 34), source(2, 2, 31, 32)])
    data = read_saved_history(commit, 'scba', select(0, 4), expected_identity=expected(commit))
    assert data['coordinates']['source_attempt'] == [2, 2, 10, 10]
    assert data['break_before'] == [True, False, True, False]
    assert data['coverage']['attempt_spans'] == 'declared_source_counts'
    assert data['coverage']['full_coordinate_scan'] is False


@pytest.mark.parametrize('field,value', [('run_uuid', 'foreign-run'), ('calcjob_uuid', 'foreign-job'),
                                        ('run_uuid', 1), ('calcjob_uuid', True)])
def test_current_cumulative_provider_identity_must_match_before_numeric(tmp_path, monkeypatch, field, value):
    current = source(10, 5, 100, 104)
    current['identity'].update(run_uuid='run', calcjob_uuid='job')
    current['identity'][field] = value
    commit = make_history(tmp_path / 'one', sources=[current])
    original = h5py.Dataset.__getitem__
    def no_numeric(dataset, key):
        if dataset.dtype.kind in 'if': pytest.fail('foreign current source selected numeric payload')
        return original(dataset, key)
    monkeypatch.setattr(h5py.Dataset, '__getitem__', no_numeric)
    with pytest.raises(ContractError) as raised:
        describe_saved_diagnostics(commit, expected_identity=expected(commit))
    assert raised.value.code == 'identity_mismatch'


def test_current_matching_provider_and_older_distinct_provider_preserve_source_attempts(tmp_path):
    older, current = source(2, 2, 31, 32), source(10, 2, 33, 34)
    older['identity'].update(run_uuid='historical-run', calcjob_uuid='historical-job')
    current['identity'].update(run_uuid='run', calcjob_uuid='job')
    commit = make_history(tmp_path / 'one', sequence=(31, 32, 33, 34), iteration=(1, 2, 3, 4),
                          outer=(7, 7, 7, 7), residual=(.1, .2, .3, .4), currents=(1., 2., 3., 4.),
                          sources=[current, older])
    data = read_saved_history(commit, 'scba', select(0, 4), expected_identity=expected(commit))
    assert data['coordinates']['source_attempt'] == [2, 2, 10, 10]
    assert data['break_before'] == [True, False, True, False]
    assert data['identity_binding']['native_recorded_fields'] == [
        'attempt', 'calcjob_uuid', 'execution_id', 'plan_fingerprint', 'point_id', 'run_uuid']
    assert data['identity_binding']['provider_only_fields'] == []


def test_older_only_uuid_names_do_not_claim_selected_native_binding(tmp_path):
    older, current = source(2, 2, 31, 32), source(10, 2, 33, 34)
    older['identity'].update(run_uuid='historical-run', calcjob_uuid='historical-job')
    commit = make_history(tmp_path / 'one', sequence=(31, 32, 33, 34), iteration=(1, 2, 3, 4),
                          outer=(7, 7, 7, 7), residual=(.1, .2, .3, .4), currents=(1., 2., 3., 4.),
                          sources=[current, older])
    data = describe_saved_diagnostics(commit, expected_identity=expected(commit))
    assert data['identity_binding']['native_recorded_fields'] == [
        'attempt', 'execution_id', 'plan_fingerprint', 'point_id']
    assert data['identity_binding']['provider_only_fields'] == ['run_uuid', 'calcjob_uuid']
    assert data['history']['phases'][0]['attempt_spans'] == [
        {'start': 0, 'stop': 2, 'source_attempt': 2, 'provenance': 'declared_source_counts'},
        {'start': 2, 'stop': 4, 'source_attempt': 10, 'provenance': 'declared_source_counts'}]


def test_absent_selected_attempt_identity_is_not_invented_from_historical_records(tmp_path):
    older = source(2, 5, 100, 104)
    older['identity'].update(run_uuid='historical-run', calcjob_uuid='historical-job')
    commit = make_history(tmp_path / 'one', sources=[older])
    data = read_saved_history(commit, 'scba', select(0, 1), expected_identity=expected(commit))
    assert data['coordinates']['source_attempt'] == [2]
    assert data['identity_binding']['native_recorded_fields'] == []
    assert data['identity_binding']['provider_only_fields'] == ['run_uuid', 'calcjob_uuid']


def test_nonfinite_masks_preserve_zero_negative_and_disconnect_omissions(tmp_path):
    commit = make_history(tmp_path / 'one', residual=(0., -1., np.nan, np.inf, -np.inf))
    data = read_saved_history(commit, 'scba', select(columns=('r_K',)), expected_identity=expected(commit))
    assert data['columns'][0]['values'] == [0., -1., None, None, None]
    assert data['columns'][0]['invalid_mask'] == [None, None, 'nan', 'positive_infinity', 'negative_infinity']
    assert data['break_before'] == [True, False, True, True, True]
    sparse = read_saved_history(commit, 'scba', select(step=2, columns=('J',)), expected_identity=expected(commit))
    assert sparse['break_before'] == [True, True, True]


def test_missing_history_still_returns_checkpoint_and_missing_summary(tmp_path, monkeypatch):
    commit = make_history(tmp_path / 'one')
    value = json.loads(commit.read_bytes()); value['artifacts'] = []
    commit.write_bytes(json_bytes(value))
    monkeypatch.setattr(h5py, 'File', lambda *a, **kw: pytest.fail('missing history opened HDF5'))
    data = describe_saved_diagnostics(commit, expected_identity=expected(commit))
    assert data['history']['reason'] == 'history_owner_missing'
    assert data['resources']['checkpoint']['checkpoint_payload_bytes'] == 4096
    with pytest.raises(ContractError):
        read_saved_history(commit, 'scba', select(), expected_identity=expected(commit))


def test_ambiguous_history_is_unavailable_without_owner_access(tmp_path, monkeypatch):
    commit = make_history(tmp_path / 'one')
    value = json.loads(commit.read_bytes())
    value['artifacts'].append({**value['artifacts'][0], 'path': 'second-history.h5'})
    commit.write_bytes(json_bytes(value))
    monkeypatch.setattr(h5py, 'File', lambda *a, **kw: pytest.fail('ambiguous role opened an owner'))
    data = describe_saved_diagnostics(commit, expected_identity=expected(commit))
    assert data['history']['reason'] == 'history_owner_ambiguous'
    assert data['history']['source_artifact'] is None
    assert data['resources']['checkpoint']['checkpoint_payload_seconds'] == 2.


def test_recorded_zero_and_exact_checkpoint_bytes_are_not_missing(tmp_path):
    commit = make_history(tmp_path / 'one')
    value = json.loads(commit.read_bytes())
    value['performance']['checkpoint_payload_seconds'] = 0.
    value['performance']['checkpoint_payload_bytes'] = 9007199254740992
    commit.write_bytes(json_bytes(value))
    checkpoint = describe_saved_diagnostics(commit, expected_identity=expected(commit))['resources']['checkpoint']
    assert checkpoint['checkpoint_payload_seconds'] == 0.
    assert checkpoint['checkpoint_payload_bytes'] == {'kind': 'integer', 'decimal': '9007199254740992'}


@pytest.mark.parametrize('field,value', [('point_id', 'wrong'), ('execution_id', 'wrong'),
        ('attempt', 11), ('plan_fingerprint', 'b'*64), ('source_commit_sha256', '0'*64)])
def test_wrong_exact_binding_refuses_before_owner_open(tmp_path, monkeypatch, field, value):
    commit = make_history(tmp_path / 'one')
    bound = {**expected(commit), field: value}
    monkeypatch.setattr(h5py, 'File', lambda *a, **kw: pytest.fail('wrong binding opened owner'))
    with pytest.raises(ContractError):
        describe_saved_diagnostics(commit, expected_identity=bound)


def test_wrong_owner_digest_refuses_before_selected_numeric_read(tmp_path, monkeypatch):
    commit = make_history(tmp_path / 'one')
    bound = expected(commit)
    value = json.loads(commit.read_bytes()); value['artifacts'][0]['sha256'] = '0'*64
    commit.write_bytes(json_bytes(value))
    original = h5py.Dataset.__getitem__
    def guard(dataset, key):
        if dataset.dtype.kind in 'if': pytest.fail('wrong hash touched selected numeric payload')
        return original(dataset, key)
    monkeypatch.setattr(h5py.Dataset, '__getitem__', guard)
    with pytest.raises(ContractError):
        read_saved_history(commit, 'scba', select(), expected_identity=expected(commit))
    # Wrong digest is rejected before any table selection, on the opened owner.
    assert bound['source_commit_sha256'] != expected(commit)['source_commit_sha256']


def test_path_replacement_cannot_redirect_the_opened_owner_hash_or_slice(tmp_path, monkeypatch):
    commit = make_history(tmp_path / 'one')
    original_file = h5py.File
    replaced = []
    def replace_after_open(stream, *args, **kwargs):
        handle = original_file(stream, *args, **kwargs)
        path = commit.parent / 'history.h5'
        if not replaced:
            path.rename(commit.parent / 'detached-original.h5')
            path.write_bytes(b'not the selected HDF5 owner')
            replaced.append(True)
        return handle
    monkeypatch.setattr(h5py, 'File', replace_after_open)
    data = read_saved_history(commit, 'scba', select(3, 5), expected_identity=expected(commit))
    assert replaced == [True]
    assert data['coordinates']['sequence'] == [103, 104]
    assert data['columns'][1]['values'] == [7000., 7100.]
    assert data['integrity']['owner_sha_verified'] is True


def test_unselected_analysis_schema_and_size_never_gate_history(tmp_path):
    commit = make_history(tmp_path / 'one')
    value = json.loads(commit.read_bytes())
    value['artifacts'].append({'path': 'unselected-analysis.h5', 'role': 'physics.analysis',
        'schema': 'unsupported-old-analysis', 'media_type': 'application/x-hdf5', 'profile': 'science',
        'bytes': 999999999, 'sha256': 'e'*64, 'dependencies': []})
    commit.write_bytes(json_bytes(value))
    data = read_saved_history(commit, 'scba', select(0, 1), expected_identity=expected(commit))
    assert data['columns'][1]['values'] == [9000.]
    assert not (commit.parent / 'unselected-analysis.h5').exists()


def test_bad_selection_and_allocation_caps_refuse_before_numeric_read(tmp_path, monkeypatch):
    commit = make_history(tmp_path / 'one')
    original = h5py.Dataset.__getitem__
    def guard(dataset, key):
        if dataset.dtype.kind in 'if': pytest.fail('refused selection touched numeric payload')
        return original(dataset, key)
    monkeypatch.setattr(h5py.Dataset, '__getitem__', guard)
    for selection, limits in [(select(0, 6), DiagnosticViewLimits()),
            (select(columns=('scba/J',)), DiagnosticViewLimits()),
            (select(), DiagnosticViewLimits(maximum_rows=1)),
            (select(), DiagnosticViewLimits(maximum_selected_bytes=1)),
            (select(columns=('r_K', 'J')), DiagnosticViewLimits(maximum_columns=1))]:
        with pytest.raises(ContractError):
            read_saved_history(commit, 'scba', selection, expected_identity=expected(commit), limits=limits)


def test_inventory_future_overlap_and_cap_refuse_without_numeric(tmp_path, monkeypatch):
    original = h5py.Dataset.__getitem__
    def guard(dataset, key):
        if dataset.dtype.kind in 'if': pytest.fail('invalid source inventory touched numeric payload')
        return original(dataset, key)
    monkeypatch.setattr(h5py.Dataset, '__getitem__', guard)
    cases = [[source(11, 5, 100, 104)], [source(2, 3, 100, 102), source(10, 2, 102, 103)]]
    for i, sources in enumerate(cases):
        commit = make_history(tmp_path / str(i), sources=sources)
        with pytest.raises(ContractError): describe_saved_diagnostics(commit, expected_identity=expected(commit))
    commit = make_history(tmp_path / 'cap', sources=[source(2, 2, 100, 101), source(10, 3, 102, 104)])
    with pytest.raises(ContractError):
        describe_saved_diagnostics(commit, expected_identity=expected(commit),
                                   limits=DiagnosticViewLimits(maximum_source_entries=1))


def test_fixed_metadata_vlen_external_and_chunk_bounds(tmp_path):
    commit = make_history(tmp_path / 'vlen')
    with h5py.File(commit.parent / 'history.h5', 'r+') as handle:
        del handle['metadata/identity_json']
        handle['metadata/identity_json'] = json.dumps(IDENTITY)  # deliberately VLEN
    rewrite_owner(commit)
    with pytest.raises(ContractError): describe_saved_diagnostics(commit, expected_identity=expected(commit))
    commit = make_history(tmp_path / 'external')
    with h5py.File(commit.parent / 'history.h5', 'r+') as handle:
        handle['outside'] = h5py.ExternalLink('unowned-file.h5', '/payload')
    rewrite_owner(commit)
    with pytest.raises(ContractError): describe_saved_diagnostics(commit, expected_identity=expected(commit))
    # The source-span and one-row DTO allocations fit 4096 bytes, while this
    # single compressed/decoded native chunk needs8192. A stale smaller cap
    # would reject on an earlier barrier and would not protect chunk admission.
    commit = make_history(tmp_path / 'chunk', sequence=tuple(range(1, 1025)),
                          iteration=tuple(range(1, 1025)), outer=(1,)*1024,
                          residual=(.1,)*1024, currents=(1.,)*1024, chunks=(1024,))
    with pytest.raises(ContractError, match='decoded HDF5 chunk'):
        read_saved_history(commit, 'scba', select(0, 1), expected_identity=expected(commit),
                           limits=DiagnosticViewLimits(maximum_selected_bytes=4096))


def test_missing_coordinates_wrong_schema_and_owner_cap_refuse(tmp_path, monkeypatch):
    commit = make_history(tmp_path / 'coordinate')
    with h5py.File(commit.parent / 'history.h5', 'r+') as handle:
        del handle['scba/sequence']
    rewrite_owner(commit)
    with pytest.raises(ContractError): describe_saved_diagnostics(commit, expected_identity=expected(commit))
    commit = make_history(tmp_path / 'schema')
    with h5py.File(commit.parent / 'history.h5', 'r+') as handle:
        handle['metadata'].attrs['schema_version'] = np.bytes_('3.0')
    rewrite_owner(commit)
    with pytest.raises(ContractError): describe_saved_diagnostics(commit, expected_identity=expected(commit))
    commit = make_history(tmp_path / 'cap')
    monkeypatch.setattr(h5py, 'File', lambda *a, **kw: pytest.fail('over-cap owner opened'))
    with pytest.raises(ContractError):
        describe_saved_diagnostics(commit, expected_identity=expected(commit),
                                   limits=DiagnosticViewLimits(maximum_owner_bytes=1))


def test_json_response_and_common_io_caps_fail_closed(tmp_path):
    commit = make_history(tmp_path / 'one')
    with pytest.raises(ContractError):
        describe_saved_diagnostics(commit, expected_identity=expected(commit),
                                   limits=DiagnosticViewLimits(maximum_json_bytes=1))
    with pytest.raises(ContractError):
        read_saved_history(commit, 'scba', select(0, 1), expected_identity=expected(commit),
                           limits=DiagnosticViewLimits(maximum_io_bytes=1))


def test_exact_large_int_coordinates_and_unknown_units(tmp_path):
    big = 9007199254740992
    commit = make_history(tmp_path / 'one', sequence=(big, big+1, big+2, big+3, big+4), units=False)
    data = read_saved_history(commit, 'scba', select(0, 1, columns=('J',)), expected_identity=expected(commit))
    assert data['coordinates']['sequence'] == [{'kind': 'integer', 'decimal': '9007199254740992'}]
    assert data['columns'][0]['units'] is None


def test_no_full_validator_progress_materialization_or_summary_lookup(tmp_path, monkeypatch):
    import qcl_negf_results.native as native
    import qcl_negf_results.progress as progress
    import qcl_negf_results.presentation as presentation
    commit = make_history(tmp_path / 'one')
    def forbidden(*a, **kw): pytest.fail('unbounded/full/mutating helper called')
    monkeypatch.setattr(native, 'validate_native_handle', forbidden)
    for name in ('live_history', 'materialize_history', '_inspect', '_attempt_spans'):
        monkeypatch.setattr(progress, name, forbidden)
    monkeypatch.setattr(presentation, 'read_performance_summary', forbidden)
    read_saved_history(commit, 'scba', select(0, 1), expected_identity=expected(commit))
