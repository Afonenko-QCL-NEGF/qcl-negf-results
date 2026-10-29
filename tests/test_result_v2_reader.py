import copy
import hashlib
import json

import h5py
import numpy as np
import pytest

from qcl_negf_contracts.messages import ContractError
from qcl_negf_results import state
from result_v2_fixture import IDENTITY, producer_root, producer_state, source, record


def test_empty_source_preserves_unknown_science_and_requires_no_payload(tmp_path):
    ref = source(tmp_path)
    assert hasattr(state, "ResultReadLimits"), "missing bounded native5 reader API"
    with state.StateReader(tmp_path, state_ref=ref, expected_identity=IDENTITY) as reader:
        reader.import_history()
        result = reader.history_batch()
        assert result.records == () and result.at_cutoff
        assert result.cursor["leaf"] is None and result.cursor["last_record"] == "0"
        assert reader.source.status["physics"]["state"] == "not_measured"
        assert not reader.source.capabilities["exact_restore"]["available"]


def test_history_returns_complete_literal_values_and_separate_core_sequence(tmp_path):
    ref = source(tmp_path, [record(1, 7), record(2, 11, "c010000000000000")])
    assert hasattr(state, "ResultReadLimits"), "missing bounded native5 reader API"
    with state.StateReader(tmp_path, state_ref=ref, expected_identity=IDENTITY) as reader:
        reader.import_history()
        result = reader.history_batch()
        assert [r["stream_ordinal"] for r in result.records] == ["1", "2"]
        assert [r["core"]["fields"]["sequence"]["decimal"] for r in result.records] == ["7", "11"]
        assert result.records[1]["core"]["fields"]["row"]["fields"]["J"]["bits"] == "c010000000000000"
        assert result.cursor["row_offset"] == "1" and result.cursor["last_record"] == "2"


def test_actual_producer_outer_vectors_and_witness_bits(producer_root):
    ref = producer_state(producer_root, "history")
    assert hasattr(state, "ResultReadLimits"), "missing bounded native5 reader API"
    with state.StateReader(producer_root, state_ref=ref, expected_identity=ref["identity"]) as reader:
        reader.import_history()
        rows = reader.history_batch().records
        assert len(rows[0]["core"]["fields"]) == 16
        scba = rows[0]['core']['fields']['row']['fields']
        from result_v2_fixture import ieee, integer
        scalar_names = ('r_D','r_A','r_K','r_Σ','r_λ','λ','r_PSD','r_caus','r_roundoff',
            'r_Jchange','r_population','J','raw_charge','target_charge','normalized_charge','lambda_change')
        assert scba == {**{name:ieee() for name in scalar_names},
            'ν':integer(1),'witness':None,'physical_markers':None}
        assert [row['stream_ordinal'] for row in rows] == ['1','2']
        assert [row['core']['fields']['sequence'] for row in rows] == [integer(7),integer(11)]
        for stage, row in zip(('scba','outer'),rows):
            core = row['core']['fields']
            for key,value in dict(action='append',stage=stage,status='running',quality='unresolved').items():
                assert core[key] == dict(kind='symbol',value=value)
            assert core['domain_revision'] == core['strict_streak'] == core['diagnostic_streak'] == integer(0)
            assert core['outer_iteration'] == integer(1) and core['converged'] is False
            assert core['inner_iteration'] == (integer(1) if stage=='scba' else None)
            assert core['basis_id']=='basis-0' and core['discretization_id']=='grid-0'
            units = {pair['fields']['first']['value']:pair['fields']['second'] for pair in core['units']}
            if stage=='scba':
                assert units == {**{name:'1' for name in scalar_names},'ν':'1','J':'A/m^2',
                    'raw_charge':'L0^2 n_sheet','target_charge':'L0^2 n_sheet','normalized_charge':'L0^2 n_sheet'}
                assert [(pair['fields']['first']['value'],pair['fields']['second']['value'])
                    for pair in core['missing_reasons']] == [('witness','not_measured'),('physical_markers','not_sampled')]
            else:
                assert units == {**{name:'1' for name in ('μ','r_P','r_U','r_n','r_neutral','r_J','r_Jchange','r_population','ζ','J')},
                    'r_U':'E0','J':'A/m^2','populations':'target_sheet_fraction','density':'L0^3 n'}
                assert core['missing_reasons'] == []
        outer = rows[1]["core"]["fields"]["row"]["fields"]
        assert {key:value for key,value in outer.items() if key not in ('populations','density')} == {
            **{name:ieee() for name in ('r_P','r_U','r_n','r_neutral','r_J','r_Jchange','r_population','ζ','J')},'μ':integer(1)}
        assert reader.read_array(outer["populations"], (slice(None),)).values.tolist() == [0.25, 0.75]
        assert reader.read_array(outer["density"], (slice(None),)).values.tolist() == [1.0, 2.0, 3.0]
    ref = producer_state(producer_root, "exact", recovery=True)
    with state.StateReader(producer_root, state_ref=ref, expected_identity=ref["identity"]) as reader:
        checkpoint = reader.checkpoint()
        fields = checkpoint["state"]["fields"]
        assert len(fields) == 17
        witness = fields["scba"]["fields"]["history"]["values"][0]["fields"]["witness"]["fields"]
        matrix = witness["matrices"]["entries"][0]["value"]
        assert reader.read_array(matrix, (slice(None), slice(None))).values.tolist() == [[1+2j]]
        assert reader.read_array(witness["eigenvector"], (slice(None),)).values.tolist() == [1+0j]


@pytest.mark.parametrize("case", ["same-size-native-tamper", "unknown-state-field", "unknown-layout", "zero-state-sequence",
    'native-wrong_dtype', 'native-wrong_shape', 'native-external_raw', 'native-external_link', 'native-soft_link',
    'native-wrong_native_version', 'native-wrong_owner', 'native-wrong_representation',
    'wrong-parent-model', 'wrong-parent-branch', 'wrong-anchor-epoch', 'future-anchor'])
def test_actual_producer_corrupt_checkpoint_rejected(producer_root, case):
    manifest_path = producer_root / case / "manifest.json"
    if not manifest_path.exists():
        pytest.skip("negative uses another explicit producer fixture entry")
    manifest = json.loads(manifest_path.read_text())
    ref = dict(identity=manifest["identity"], state_id=manifest["state_id"], state_sequence=manifest["state_sequence"],
               manifest=dict(path=str(manifest_path.relative_to(producer_root)), bytes=str(manifest_path.stat().st_size),
                   sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(), schema=manifest["schema"], media_type="application/json"))
    assert hasattr(state, "ResultReadLimits"), "missing bounded native5 reader API"
    with pytest.raises(ContractError):
        with state.StateReader(producer_root, state_ref=ref, expected_identity=ref["identity"]) as reader:
            cp = reader.checkpoint()
            for declaration in _arrays(cp):
                reader.describe_array(declaration)


def _arrays(value):
    if isinstance(value, dict):
        if value.get("kind") == "array" and "file" in value:
            yield value
        for child in value.values():
            yield from _arrays(child)
    elif isinstance(value, list):
        for child in value:
            yield from _arrays(child)


def test_actual_codec_shape_and_bit_oracle(producer_root):
    from qcl_negf_results.native import validate_native_array
    wires = json.loads((producer_root / 'codec-shapes.json').read_text())['arrays']
    with h5py.File(producer_root / wires[0]['file']['path']) as handle:
        real = validate_native_array(handle, wires[0])[:]
        assert real.view(np.uint64).tolist() == [0, 0x8000000000000000, 0x7ff0000000000000,
            0xfff0000000000000, 0x7ff8000000000001, 0x7ff8000000000002]
        values = validate_native_array(handle, wires[2])
        assert values.shape == (3, 2, 1, 2)
        assert values[1, 1, 0, 1] == 11-11j
        assert values[2, 0, 0, 0] == 3-3j
        assert validate_native_array(handle, wires[4]).shape == (0, 3)
        assert validate_native_array(handle, wires[5]).shape == (0, 2, 1, 1)
        assert validate_native_array(handle, wires[6])[:].tolist() == [False, True, False]
        assert validate_native_array(handle, wires[7])[:].tolist() == [-(2**63), 2**63-1]


def test_native_array_rank_outside_actual_core_graph_rejected(tmp_path):
    from qcl_negf_results.native import validate_native_array
    with h5py.File(tmp_path / 'rank.h5', 'w') as handle:
        handle['arrays/x'] = np.zeros((2, 2, 2), dtype=np.complex128)
        wire = dict(kind='array', order='F', eltype='ComplexF64', shape=['2','2','2'], dataset='/arrays/x')
        with pytest.raises(ContractError, match='rank'):
            validate_native_array(handle, wire)


def test_declared_array_cannot_grant_arbitrary_dataset_access(tmp_path):
    ref = source(tmp_path, [record()])
    with state.StateReader(tmp_path, state_ref=ref, expected_identity=IDENTITY) as reader:
        reader.import_history()
        reader.history_batch()
        with pytest.raises(ContractError, match='descriptor'):
            reader.describe_array(dict(kind='array', file=ref['manifest'], dataset='/unowned', shape=['1'], order='F', eltype='Float64'))


def test_record_array_declaration_requires_owned_leaf_dependency(producer_root, tmp_path):
    ref = producer_state(producer_root, 'exact', recovery=True)
    manifest = json.loads((producer_root / ref['manifest']['path']).read_text())
    checkpoint = json.loads((producer_root / manifest['checkpoint']['path']).read_text())
    witness = checkpoint['state']['fields']['scba']['fields']['history']['values'][0]['fields']['witness']
    manufactured = record()
    manufactured['core']['fields']['row']['fields']['witness'] = witness
    selected = source(tmp_path, [manufactured])
    with state.StateReader(tmp_path, state_ref=selected, expected_identity=IDENTITY) as reader:
        reader.import_history()
        with pytest.raises(ContractError, match='array.*dependency'):
            reader.history_batch()


@pytest.mark.parametrize('change', ['stage_row_type', 'row_counter', 'sequence_replay'])
def test_scientific_record_relations_not_just_schema_are_checked(tmp_path, change):
    rows = [record(), record(2, 9)]
    if change == 'stage_row_type':
        rows[0]['core']['fields']['stage']['value'] = 'outer'
        rows[0]['row_key']['stage']['value'] = 'outer'
    elif change == 'row_counter':
        rows[0]['core']['fields']['row']['fields']['ν']['decimal'] = '2'
    else:
        rows[1]['core']['fields']['sequence']['decimal'] = '7'
    ref = source(tmp_path, rows)
    with state.StateReader(tmp_path, state_ref=ref, expected_identity=IDENTITY) as reader:
        reader.import_history()
        with pytest.raises(ContractError):
            reader.history_batch()


@pytest.mark.parametrize('error_type',[RuntimeError,KeyboardInterrupt])
def test_native_close_attempts_every_owned_stack_when_one_cleanup_raises(tmp_path,error_type):
    from consumer_context_fixture import fixture
    from qcl_negf_results.final_observations import FinalObservationsReader
    selected,artifact,_,_,_=fixture(tmp_path)
    reader=state.StateReader(tmp_path,state_ref=selected,expected_identity=selected['identity'])
    FinalObservationsReader(reader,artifact_ref=artifact).describe()
    handles=[item['handle'] for item in reader._handles.values()]
    assert len(handles)==2
    def raised():raise error_type('literal cleanup failure')
    next(iter(reader._handles.values()))['stack'].callback(raised)
    with pytest.raises(error_type,match='literal cleanup failure'):reader.close()
    assert all(not handle.id.valid for handle in handles)
    with pytest.raises(ValueError,match='closed'):reader.history_batch()
    reader.close()


def test_state_context_cleanup_preserves_body_cancellation_and_other_owners(tmp_path):
    from consumer_context_fixture import fixture
    from qcl_negf_results.final_observations import FinalObservationsReader
    selected,artifact,_,_,_=fixture(tmp_path)
    reader=state.StateReader(tmp_path,state_ref=selected,expected_identity=selected['identity'])
    handles=[]
    def raised():raise RuntimeError('literal secondary close failure')
    with pytest.raises(KeyboardInterrupt,match='literal primary') as caught:
        with reader:
            FinalObservationsReader(reader,artifact_ref=artifact).describe()
            handles=[item['handle'] for item in reader._handles.values()]
            next(iter(reader._handles.values()))['stack'].callback(raised)
            raise KeyboardInterrupt('literal primary')
    assert all(not handle.id.valid for handle in handles)
    assert any('cleanup' in note for note in caught.value.__notes__)
