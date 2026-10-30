"""Original CP inputs: literal raw bytes and independently pinned identities.

These tiny constructor files are transport fixtures, never a solver oracle.
"""
from copy import deepcopy
import hashlib
import json
import os
import struct

import h5py
import pytest

from qcl_negf_contracts.messages import ContractError
from qcl_negf_results import final_observations as api
from qcl_negf_results.state import StateReader, ResultReadLimits
from consumer_context_fixture import fixture, PLAN, canonical


def wire(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False,
                      sort_keys=True, separators=(',', ':')).encode()


def pin(root, path, schema, media_type='application/json'):
    raw = (root / path).read_bytes()
    return dict(path=path, bytes=str(len(raw)), sha256=hashlib.sha256(raw).hexdigest(),
                schema=schema, media_type=media_type)


def save_manifest(root, selected, manifest):
    path = selected['manifest']['path']
    (root / path).write_bytes(wire(manifest))
    selected['manifest'] = pin(root, path, manifest['schema'])


def cp_source(root, *, kind='nested_sweep', plan_bytes=PLAN):
    selected, _, observations, checkpoint, _ = fixture(
        root, plan_kind=kind, plan_bytes=plan_bytes)
    manifest = json.loads((root / selected['manifest']['path']).read_bytes())
    manifest['artifacts'] = [row for row in manifest['artifacts']
                             if row['role'] != 'physics.analysis']
    for name in ('convergence', 'physics'):
        manifest['status'][name] = dict(state='not_measured', reason='CP-only fixture', evidence=None)
    for path in ('analysis.h5', 'observations.h5'):
        (root / path).unlink()
    save_manifest(root, selected, manifest)
    return selected, deepcopy(observations['request']['plan']), deepcopy(observations['request']['model']), checkpoint


def change_input(root, selected, reference, raw):
    """Rebind exact fixture bytes; identity stays unchanged unless explicitly updated."""
    (root / reference['path']).write_bytes(raw)
    reference.update(pin(root, reference['path'], reference['schema'], reference['media_type']))
    manifest = json.loads((root / selected['manifest']['path']).read_bytes())
    next(row for row in manifest['artifacts'] if row['path'] == reference['path']).update(reference)
    save_manifest(root, selected, manifest)


def read(root, selected, plan_ref, model_ref, **kwargs):
    with StateReader(root, state_ref=selected, expected_identity=selected['identity'], **kwargs) as reader:
        return api.describe_checkpoint_inputs(reader, plan_ref=plan_ref, model_ref=model_ref)


@pytest.mark.parametrize('kind', ['nested_sweep', 'scientific_plan_v2'])
def test_cp_only_original_inputs_without_final_owners(tmp_path, monkeypatch, kind):
    selected, plan_ref, model_ref, checkpoint = cp_source(tmp_path, kind=kind)
    assert callable(getattr(api, 'describe_checkpoint_inputs', None)), 'public CP input reader missing'
    plan_raw = (tmp_path / plan_ref['path']).read_bytes()
    model_raw = (tmp_path / model_ref['path']).read_bytes()
    original = h5py.Dataset.__getitem__
    def no_tensor(dataset, key):
        assert '/arrays/' not in dataset.name, 'CP input description read numeric tensor'
        return original(dataset, key)
    monkeypatch.setattr(h5py.Dataset, '__getitem__', no_tensor)
    with StateReader(tmp_path, state_ref=selected, expected_identity=selected['identity']) as reader:
        description = api.describe_checkpoint_inputs(reader, plan_ref=plan_ref, model_ref=model_ref)
        assert isinstance(description, api.CheckpointInputsDescription)
        assert description.source == selected
        assert description.invocation == checkpoint['invocation']
        assert description.checkpoint == pin(tmp_path, 'checkpoint.json', checkpoint['schema'])
        assert description.plan_ref == plan_ref and description.model_ref == model_ref
        assert description.plan_raw == plan_raw and description.model_raw == model_raw
        assert description.model == json.loads(model_raw)
        assert description.model_ref['sha256'] != description.model['configuration_hash']
        assert description.model['configuration_hash'] == selected['identity']['model_sha256']
        assert description.integrity['scope'] == 'selected_checkpoint_original_inputs'
        assert 'scientific_acceptance' in description.integrity['not_assessed']
        assert reader.source.status['physics']['state'] == 'not_measured'
        assert reader.source.integrity['selected_payloads_verified'] == []
        assert reader.checkpoint() == checkpoint
        if kind == 'nested_sweep':
            assert description.plan_raw.startswith(b'# original manufactured input; preserve this comment\n')
            assert description.plan['name'] == 'configured_operating_points'
        else:
            assert description.plan['schema'] == 'qcl-negf-scientific-plan-v2'
            assert description.plan['scientific_fingerprint'] != selected['identity']['plan_sha256']


@pytest.mark.parametrize('role', ['plan', 'model'])
@pytest.mark.parametrize('field,value', [
    ('sha256', 'f' * 64), ('bytes', '12'), ('path', 'different.json'),
    ('schema', 'legacy'), ('media_type', 'application/yaml'), ('bytes', 12),
    ('bytes', '001'), ('extra', True), ('schema', None)])
def test_exact_full_input_refs_not_hash_only(tmp_path, role, field, value):
    selected, plan_ref, model_ref, _ = cp_source(tmp_path)
    target = plan_ref if role == 'plan' else model_ref
    target[field] = value
    with pytest.raises(ContractError):
        read(tmp_path, selected, plan_ref, model_ref)


@pytest.mark.parametrize('role', ['plan', 'model'])
@pytest.mark.parametrize('mutation', ['role', 'profile', 'dependencies', 'missing', 'file_missing', 'duplicate'])
def test_original_manifest_role_profile_dependencies_are_exact(tmp_path, role, mutation):
    selected, plan_ref, model_ref, _ = cp_source(tmp_path)
    target = plan_ref if role == 'plan' else model_ref
    manifest = json.loads((tmp_path / selected['manifest']['path']).read_bytes())
    row = next(row for row in manifest['artifacts'] if row['path'] == target['path'])
    if mutation == 'role':
        row['role'] = 'source'
    elif mutation == 'profile':
        row['profile'] = 'both'
    elif mutation == 'dependencies':
        row['dependencies'] = ['checkpoint.json']
    elif mutation == 'missing':
        manifest['artifacts'].remove(row)
    elif mutation == 'duplicate':
        manifest['artifacts'].append(deepcopy(row))
    else:
        (tmp_path / target['path']).unlink()
    save_manifest(tmp_path, selected, manifest)
    with pytest.raises(ContractError):
        read(tmp_path, selected, plan_ref, model_ref)


@pytest.mark.parametrize('role', ['plan', 'model'])
def test_missing_file_ref_field_is_contract_failure(tmp_path, role):
    selected, plan_ref, model_ref, _ = cp_source(tmp_path)
    del (plan_ref if role == 'plan' else model_ref)['media_type']
    with pytest.raises(ContractError):
        read(tmp_path, selected, plan_ref, model_ref)


def test_reencoded_plan_has_different_original_identity_even_if_decoded_equal(tmp_path):
    selected, plan_ref, model_ref, _ = cp_source(tmp_path)
    change_input(tmp_path, selected, plan_ref, PLAN + b'# another original byte sequence\n')
    with pytest.raises(ContractError, match='plan.*identity|plan.*digest'):
        read(tmp_path, selected, plan_ref, model_ref)


def test_recomputed_file_hash_cannot_hide_changed_actual_model_preimage(tmp_path):
    selected, plan_ref, model_ref, _ = cp_source(tmp_path)
    model = json.loads((tmp_path / model_ref['path']).read_bytes())
    model['configuration']['derived']['effective_seed_eV'] = -0.0
    change_input(tmp_path, selected, model_ref, wire(model))
    with pytest.raises(ContractError, match='identity|digest'):
        read(tmp_path, selected, plan_ref, model_ref)


@pytest.mark.parametrize('mutation', ['extra', 'missing', 'scope', 'section', 'flag', 'production', 'derived'])
def test_closed_native_model_profile_even_with_recomputed_all_hashes(tmp_path, mutation):
    selected, plan_ref, model_ref, checkpoint = cp_source(tmp_path)
    model = json.loads((tmp_path / model_ref['path']).read_bytes())
    c = model['configuration']
    if mutation == 'extra':
        model['claimed_verified'] = True
    elif mutation == 'missing':
        del model['hash_encoding']
    elif mutation == 'scope':
        c['scope'] = 'different_problem'
    elif mutation == 'section':
        del c['inputs']['physical']
    elif mutation == 'flag':
        c['inputs']['scattering']['LO'] = 0
    elif mutation == 'production':
        c['inputs']['production']['verify_fft_roundoff'] = 0
    else:
        c['derived']['extra'] = 1
    model['configuration_hash'] = hashlib.sha256(canonical(c)).hexdigest()
    identity = dict(selected['identity'], model_sha256=model['configuration_hash'])
    selected['identity'] = identity
    checkpoint['identity'] = identity
    (tmp_path / 'checkpoint.json').write_bytes(wire(checkpoint))
    manifest = json.loads((tmp_path / selected['manifest']['path']).read_bytes())
    manifest['identity'] = identity
    cp_ref = pin(tmp_path, 'checkpoint.json', checkpoint['schema'])
    manifest['checkpoint'] = cp_ref
    next(row for row in manifest['artifacts'] if row['path'] == 'checkpoint.json').update(cp_ref)
    save_manifest(tmp_path, selected, manifest)
    change_input(tmp_path, selected, model_ref, wire(model))
    with pytest.raises(ContractError):
        read(tmp_path, selected, plan_ref, model_ref)


def test_independent_canonical_ieee_oracle_and_full_model_envelope(tmp_path):
    selected, plan_ref, model_ref, checkpoint = cp_source(tmp_path)
    model = json.loads((tmp_path / 'model.json').read_bytes())
    physical = dict(bool=True, int=1, float=1.0, minus_zero=-0.0, zero=0.0)
    literal = (b'm5{s4:boolb1;s5:floatf16:3ff0000000000000s3:inti1:1'
               b's10:minus_zerof16:8000000000000000s4:zerof16:0000000000000000}')
    assert canonical(physical) == literal
    model['configuration']['inputs']['physical'] = physical
    expected_hash = hashlib.sha256(canonical(model['configuration'])).hexdigest()
    model['configuration_hash'] = expected_hash
    identity = dict(selected['identity'], model_sha256=expected_hash)
    selected['identity'] = identity
    checkpoint['identity'] = identity
    (tmp_path / 'checkpoint.json').write_bytes(wire(checkpoint))
    manifest = json.loads((tmp_path / selected['manifest']['path']).read_bytes())
    manifest['identity'] = identity
    cp_ref = pin(tmp_path, 'checkpoint.json', checkpoint['schema'])
    manifest['checkpoint'] = cp_ref
    next(row for row in manifest['artifacts'] if row['path'] == 'checkpoint.json').update(cp_ref)
    save_manifest(tmp_path, selected, manifest)
    change_input(tmp_path, selected, model_ref, wire(model))
    result = read(tmp_path, selected, plan_ref, model_ref)
    stored = result.model['configuration']['inputs']['physical']
    assert type(stored['bool']) is bool and type(stored['int']) is int and type(stored['float']) is float
    assert struct.pack('>d', stored['minus_zero']).hex() == '8000000000000000'
    assert struct.pack('>d', stored['zero']).hex() == '0000000000000000'
    assert result.model['configuration_hash'] == expected_hash
    assert hashlib.sha256(result.model_raw).hexdigest() == model_ref['sha256'] != expected_hash


@pytest.mark.parametrize('raw', [
    b'name: x\nroot: &a {kind: leaf, name: end}\n',
    b'name: x\nname: y\nroot: {kind: leaf, name: end}\n',
    b'name: x\nroot: !!seq {kind: leaf, name: end}\n',
    b'name: x\nroot: {kind: level, name: x, axes: [{name: y, values: [0b10]}], child: {kind: leaf, name: end}}\n',
    b'name: x\nroot: {kind: level, name: x, axes: [{name: y, values: [1, 1]}], child: {kind: leaf, name: end}}\n'])
def test_original_yaml_existing_admission_profile(tmp_path, raw):
    selected, plan_ref, model_ref, _ = cp_source(tmp_path, plan_bytes=raw)
    with pytest.raises(ContractError):
        read(tmp_path, selected, plan_ref, model_ref)


@pytest.mark.parametrize('raw', [b'{"a":1,"a":2}', b'{"a":NaN}', b'{"a":"\xff"}'])
def test_original_model_strict_json(tmp_path, raw):
    selected, plan_ref, model_ref, _ = cp_source(tmp_path)
    change_input(tmp_path, selected, model_ref, raw)
    with pytest.raises(ContractError):
        read(tmp_path, selected, plan_ref, model_ref)


@pytest.mark.parametrize('target', ['model.json', 'plan.yaml'])
def test_input_actual_read_fd_is_guarded(tmp_path, monkeypatch, target):
    from qcl_negf_results import state
    selected, plan_ref, model_ref, _ = cp_source(tmp_path)
    original = state._BudgetStream.read
    def replace(stream, *args):
        raw = original(stream, *args)
        if os.readlink('/proc/self/fd/' + str(stream._raw.fileno())).endswith('/' + target):
            path = tmp_path / target
            substitute = tmp_path / 'substitute'
            substitute.write_bytes(path.read_bytes())
            substitute.replace(path)
        return raw
    monkeypatch.setattr(state._BudgetStream, 'read', replace)
    with pytest.raises(ContractError, match='binding|source'):
        read(tmp_path, selected, plan_ref, model_ref)


@pytest.mark.parametrize('target', ['model.json', 'plan.yaml', 'selected.json', 'checkpoint.json'])
def test_consumed_binding_stays_guarded_after_input_read_until_return(tmp_path, monkeypatch, target):
    selected, plan_ref, model_ref, _ = cp_source(tmp_path)
    original = api._portable_plan
    def replace(raw, maximum):
        value = original(raw, maximum)
        path = tmp_path / target
        substitute = tmp_path / 'later-substitute'
        substitute.write_bytes(path.read_bytes())
        substitute.replace(path)
        return value
    monkeypatch.setattr(api, '_portable_plan', replace)
    with pytest.raises(ContractError, match='binding|source'):
        read(tmp_path, selected, plan_ref, model_ref)


def test_root_binding_stays_guarded_after_parse(tmp_path, monkeypatch):
    root = tmp_path / 'source'
    root.mkdir()
    selected, plan_ref, model_ref, _ = cp_source(root)
    original = api._portable_plan
    def replace(raw, maximum):
        value = original(raw, maximum)
        root.rename(tmp_path / 'moved')
        root.mkdir()
        return value
    monkeypatch.setattr(api, '_portable_plan', replace)
    with pytest.raises(ContractError, match='root|source|binding'):
        read(root, selected, plan_ref, model_ref)


@pytest.mark.parametrize('error_type', [RuntimeError, KeyboardInterrupt])
@pytest.mark.parametrize('target', ['model.json', 'plan.yaml'])
def test_temporary_input_fd_closes_on_any_base_exception(tmp_path, monkeypatch, error_type, target):
    from qcl_negf_results import state
    selected, plan_ref, model_ref, _ = cp_source(tmp_path)
    original = state._BudgetStream.read
    opened = []
    def interrupt(stream, *args):
        if os.readlink('/proc/self/fd/' + str(stream._raw.fileno())).endswith('/' + target):
            opened.append(stream._raw)
            raise error_type('literal original input failure')
        return original(stream, *args)
    with StateReader(tmp_path, state_ref=selected, expected_identity=selected['identity']) as reader:
        with monkeypatch.context() as patch:
            patch.setattr(state._BudgetStream, 'read', interrupt)
            with pytest.raises(error_type, match='literal original input failure'):
                api.describe_checkpoint_inputs(reader, plan_ref=plan_ref, model_ref=model_ref)
        assert opened and all(stream.closed for stream in opened)
        assert api.describe_checkpoint_inputs(reader, plan_ref=plan_ref, model_ref=model_ref).source == selected


def test_repeated_operations_use_real_counters_and_detached_outputs(tmp_path, monkeypatch):
    from qcl_negf_results import state
    selected, plan_ref, model_ref, checkpoint = cp_source(tmp_path)
    original = state._IOBudget.account
    deliveries = []
    def account(budget, n):
        deliveries.append((id(budget), n))
        return original(budget, n)
    with StateReader(tmp_path, state_ref=selected, expected_identity=selected['identity']) as reader:
        monkeypatch.setattr(state._IOBudget, 'account', account)
        first = api.describe_checkpoint_inputs(reader, plan_ref=plan_ref, model_ref=model_ref)
        assert len({key for key, _ in deliveries}) == 1
        assert first.io_counters['bytes_read_total'] == sum(n for _, n in deliveries)
        assert first.io_counters['metadata_bytes'] >= len(first.plan_raw) + len(first.model_raw)
        first.source['identity']['run_id'] = 'mutated'
        first.model['configuration']['scope'] = 'mutated'
        first.plan_ref['path'] = 'mutated'
        first.integrity['input_refs']['model']['sha256'] = '0' * 64
        deliveries.clear()
        second = api.describe_checkpoint_inputs(reader, plan_ref=plan_ref, model_ref=model_ref)
        assert second.source == selected and second.plan_ref == plan_ref
        assert second.model['configuration']['scope'] == 'stationary_transport_inputs_v1'
        assert second.io_counters['bytes_read_total'] == sum(n for _, n in deliveries)
        assert second.io_counters['bytes_read_total'] < first.io_counters['bytes_read_total']
        assert reader.checkpoint() == checkpoint


def test_one_physical_allowance_exact_bound_and_one_byte_under(tmp_path):
    selected, plan_ref, model_ref, _ = cp_source(tmp_path)
    measured = read(tmp_path, selected, plan_ref, model_ref).io_counters['bytes_read_total']
    healthy = read(tmp_path, selected, plan_ref, model_ref,
                   limits=ResultReadLimits(maximum_io_bytes=measured))
    assert healthy.io_counters['bytes_read_total'] == measured
    with StateReader(tmp_path, state_ref=selected, expected_identity=selected['identity'],
                     limits=ResultReadLimits(maximum_io_bytes=measured - 1)) as reader:
        with pytest.raises(ContractError, match='budget'):
            api.describe_checkpoint_inputs(reader, plan_ref=plan_ref, model_ref=model_ref)
        assert reader.io_counters['bytes_read_total'] <= measured - 1


def test_aggregate_metadata_admission_precedes_original_file_read(tmp_path, monkeypatch):
    from qcl_negf_results import state
    selected, plan_ref, model_ref, _ = cp_source(tmp_path)
    original = state._BudgetStream.read
    def no_input(stream, *args):
        path = os.readlink('/proc/self/fd/' + str(stream._raw.fileno()))
        assert not path.endswith(('/model.json', '/plan.yaml')), 'underbudget input read before aggregate admission'
        return original(stream, *args)
    with StateReader(tmp_path, state_ref=selected, expected_identity=selected['identity'],
                     limits=ResultReadLimits(maximum_metadata_bytes=40_000)) as reader:
        monkeypatch.setattr(state._BudgetStream, 'read', no_input)
        with pytest.raises(ContractError) as caught:
            api.describe_checkpoint_inputs(reader, plan_ref=plan_ref, model_ref=model_ref)
        assert caught.value.code == 'budget_exceeded'


def test_valid_model_with_recomputed_canonical_hash_must_match_selected_identity(tmp_path):
    selected, plan_ref, model_ref, _ = cp_source(tmp_path)
    model = json.loads((tmp_path / model_ref['path']).read_bytes())
    model['configuration']['derived']['effective_seed_eV'] = -0.0
    model['configuration_hash'] = hashlib.sha256(canonical(model['configuration'])).hexdigest()
    change_input(tmp_path, selected, model_ref, wire(model))
    with pytest.raises(ContractError, match='model identity'):
        read(tmp_path, selected, plan_ref, model_ref)


@pytest.mark.parametrize('path', ['checkpoint.json', 'selected.json'])
def test_cached_checkpoint_does_not_supply_stale_binding_success(tmp_path, path):
    selected, plan_ref, model_ref, _ = cp_source(tmp_path)
    with StateReader(tmp_path, state_ref=selected, expected_identity=selected['identity']) as reader:
        reader.checkpoint()
        substitute = tmp_path / 'cached-substitute'
        substitute.write_bytes((tmp_path / path).read_bytes())
        substitute.replace(tmp_path / path)
        with pytest.raises(ContractError, match='binding|source'):
            api.describe_checkpoint_inputs(reader, plan_ref=plan_ref, model_ref=model_ref)


@pytest.mark.parametrize('mutation', ['duplicate_id', 'point_count', 'schema'])
def test_original_scientific_json_profile_after_raw_identity_binding(tmp_path, mutation):
    selected, plan_ref, model_ref, checkpoint = cp_source(tmp_path, kind='scientific_plan_v2')
    plan = json.loads((tmp_path / plan_ref['path']).read_bytes())
    if mutation == 'duplicate_id':
        plan['points'].append(deepcopy(plan['points'][0]))
        plan['computation_count'] += 1
    elif mutation == 'point_count':
        plan['computation_count'] += 1
    else:
        plan['model_revision'] = 'legacy'
    change_input(tmp_path, selected, plan_ref, wire(plan))
    identity = dict(selected['identity'], plan_sha256=plan_ref['sha256'])
    selected['identity'] = identity
    checkpoint['identity'] = identity
    (tmp_path / 'checkpoint.json').write_bytes(wire(checkpoint))
    manifest = json.loads((tmp_path / selected['manifest']['path']).read_bytes())
    manifest['identity'] = identity
    cp_ref = pin(tmp_path, 'checkpoint.json', checkpoint['schema'])
    manifest['checkpoint'] = cp_ref
    next(row for row in manifest['artifacts'] if row['path'] == 'checkpoint.json').update(cp_ref)
    save_manifest(tmp_path, selected, manifest)
    with pytest.raises(ContractError):
        read(tmp_path, selected, plan_ref, model_ref)


def test_no_checkpoint_is_explicitly_unavailable(tmp_path):
    selected, plan_ref, model_ref, _ = cp_source(tmp_path)
    manifest = json.loads((tmp_path / selected['manifest']['path']).read_bytes())
    manifest['checkpoint'] = None
    save_manifest(tmp_path, selected, manifest)
    with pytest.raises(ContractError) as caught:
        read(tmp_path, selected, plan_ref, model_ref)
    assert caught.value.code == 'capability_unavailable'
