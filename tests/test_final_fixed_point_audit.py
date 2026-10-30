"""Literal D2 format fixtures; never run a physical map or product encoder."""
import copy
import hashlib
import json

import h5py
import numpy as np
import pytest

from qcl_negf_contracts.messages import ContractError
from qcl_negf_results.native import _read_final_fixed_point_audit
from qcl_negf_results import native as native_module
from qcl_negf_results.state import StateReader, _BudgetStream, _IOBudget
from qcl_negf_contracts.final_audit import validate_final_audit_pair
from native_result_fixtures import declare_native
from qcl_negf_contracts.artifacts import CONTRACT_SET


def _pair(handle, declaration, *, final_state):
    """Compose the actual native subresource decoder and public pure validator.

    This does not construct a result container reader or add v4 compatibility.
    """
    value = declaration.get('stationary_assessment', {}).get('final_fixed_point_audit')
    return validate_final_audit_pair(_read_final_fixed_point_audit(handle), value,
        expected_identity=declaration['identity'], final_state=final_state)

SCHEMA = 'qcl-negf-final-fixed-point-audit-v1'
PATH = 'diagnostics/final_fixed_point_audit'
IDENTITY = {'execution_id': 'e', 'point_id': 'p', 'state_id': 's',
            'attempt': 1, 'state_sequence': 1, 'plan_fingerprint': 'a' * 64}


def measure(value=0):
    return dict(status='measured', value=value, nonfinite=None, reason=None)


def record():
    # All numbers are literal manufactured measurements, no scientific verdict.
    optional = dict(status='not_applicable', value=None, nonfinite=None,
                    reason='no_previous_inner_iteration')
    witness = dict(status='measured', e_index=1, k_index=1,
                   numerator_F=measure(), D_F=measure(1), GR_F=measure(1),
                   identity_F=measure(1), denominator=measure(2), ratio=measure())
    norm = dict(numerator_F=measure(), scale_F=measure(), denominator=measure(1e-14),
                ratio=measure(), floor=1e-14)
    return dict(
        schema=SCHEMA, record_status='measured', reason=None, detail='', detail_truncated=False,
        state_identity=dict(IDENTITY), shape=dict(NE=1, Nk=1, Nb=1),
        units=dict(native='1', energy_scale='eV', length_scale='m', index='1-based'),
        context=dict(status='known', reason=None, unknown_fields=[], backend='production',
            energy_shift='linear', hilbert='direct', contraction='reference', kernel_build='reference',
            embedding='finite_chain', electron_electron_mode='none', embedding_periods=4,
            enabled_scattering=[], E0_eV=1, L0_m=1, lambda_P=1,
            map_kernels_shared_with_solver=True, independent_physics_oracle=False,
            state_alignment='producer_aligned', alignment_reason=None, source_owner_role='physics.full',
            source_paths=dict(U='/state_dimensionless/UH', n='/state_dimensionless/density',
                GR='/state_dimensionless/GR', GL='/state_dimensionless/GL',
                GG='/state_dimensionless/GG', A='/state_dimensionless/A')),
        history=dict(inner_last=1, inner_previous=None, outer_last=1, previous_lambda=optional.copy()),
        dyson=dict(ratio=measure(), global_abs_F=measure(), max_abs_F=measure(),
                   ratio_witness=copy.deepcopy(witness), absolute_witness=copy.deepcopy(witness)),
        spectral=copy.deepcopy(norm), keldysh=copy.deepcopy(norm),
        selfenergy=dict(rows=[dict(family=f, component=c, numerator_F=measure(), scale_F=measure(),
                                  denominator=measure(1e-14), ratio=measure())
                            for f in ('embedding_total', 'embedding_plus', 'embedding_minus')
                            for c in ('R', 'L', 'G')],
                        winner=dict(family='embedding_total', component='R'), max_ratio=measure(), floor=1e-14),
        charge={**{k: measure(v) for k, v in dict(target_sheet=1, stored_raw_sheet=1,
            normalized_sheet=1, candidate_raw_sheet=1, stored_lambda=1, candidate_lambda=1,
            stored_constraint=0, candidate_constraint=0, normalized_residual=0).items()},
            'lambda_change': optional.copy()},
        poisson={k: measure(v) for k, v in dict(residual_ratio=0, residual_numerator_F=0,
            rhs_F=0, additive_floor=1e-14, denominator=1e-14, border_scale=1,
            zeta_signed=0, neutrality_signed=0, neutrality_abs=0, target_sheet=1,
            neutrality_ratio=0, rU_max=0).items()})


def raw(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode('utf-8')


def attribute(parent, key, text):
    parent.attrs.create(key, text.encode(), dtype=h5py.string_dtype('utf-8', len(text.encode())))


def native(handle, value, *, payload=None, width=None, dtype=None, shape=(),
           identity_path='metadata/point_identity_json'):
    if identity_path not in handle:
        handle[identity_path] = json.dumps(IDENTITY)
    group = handle.create_group(PATH)
    attribute(group, 'schema', SCHEMA)
    data = raw(value) if payload is None else payload
    dtype = dtype or h5py.string_dtype('utf-8', width or len(data))
    dataset = group.create_dataset('record_json', shape=shape, dtype=dtype)
    dataset[()] = data
    for key, text in dict(units='1', logical_axis_order='scalar',
                         description='Versioned final fixed-point raw measurement record').items():
        attribute(dataset, key, text)
    return dataset


def commit(value=None, *, recovery=False):
    result = dict(identity=dict(IDENTITY), storage_class='recovery' if recovery else 'archive')
    if value is not None:
        result['stationary_assessment'] = dict(final_fixed_point_audit=value)
    return result


def test_healthy_zero_and_right_padding(tmp_path):
    value = record()
    with h5py.File(tmp_path / 'record.h5', 'w') as handle:
        native(handle, value, width=len(raw(value)) + 17)
        result = _pair(handle, commit(value), final_state=True)
    assert result == dict(record=value, status='measured', reason=None)
    assert result['record']['dyson']['ratio'] == measure(0)


@pytest.mark.parametrize('recovery,status,reason', [
    (False, 'not_recorded', None), (True, 'not_applicable', 'not_a_final_state')])
def test_legacy_and_recovery_absence(tmp_path, recovery, status, reason):
    with h5py.File(tmp_path / 'record.h5', 'w') as handle:
        assert _pair(handle, commit(recovery=recovery),
            final_state=not recovery) == dict(record=None, status=status, reason=reason)


@pytest.mark.parametrize('side', ['native', 'commit', 'both_recovery'])
def test_presence_must_agree_and_recovery_never_contains_final(tmp_path, side):
    value = record()
    declaration = commit(value if side != 'native' else None, recovery=side == 'both_recovery')
    if side == 'null_commit':
        declaration['stationary_assessment']['final_fixed_point_audit'] = None
    with h5py.File(tmp_path / 'record.h5', 'w') as handle:
        if side in ('native', 'both_recovery'):
            native(handle, value)
        with pytest.raises(ContractError):
            _pair(handle, declaration, final_state=side != 'both_recovery')


@pytest.mark.parametrize('field,value', [('attempt', 2), ('state_sequence', 2),
    ('plan_fingerprint', 'b' * 64), ('state_id', 'other')])
def test_record_binding_rejects_wrong_state(tmp_path, field, value):
    stored = record()
    stored['state_identity'][field] = value
    with h5py.File(tmp_path / 'record.h5', 'w') as handle:
        native(handle, stored)
        with pytest.raises(ContractError):
            _pair(handle, commit(stored), final_state=True)


def test_equal_binding_does_not_replace_semantic_equality(tmp_path):
    stored, declared = record(), record()
    declared['detail'] = 'different measurement record'
    with h5py.File(tmp_path / 'record.h5', 'w') as handle:
        native(handle, stored)
        with pytest.raises(ContractError, match='differ'):
            _pair(handle, commit(declared), final_state=True)


@pytest.mark.parametrize('payload', [b'{"x":0,"x":1}', b'{"x":NaN}', b'\xff',
                                    b'{"x":"a\x00b"}', b'{}\x00junk', b' ' * 65537])
def test_hostile_payloads_rejected(tmp_path, payload):
    with h5py.File(tmp_path / 'record.h5', 'w') as handle:
        native(handle, record(), payload=payload)
        with pytest.raises(ContractError):
            _pair(handle, commit(record()), final_state=True)


@pytest.mark.parametrize('kind', ['vlen', 'ascii', 'numeric', 'vector', 'oversize',
                                  'child', 'schema', 'units', 'axis', 'description'])
def test_bad_headers_are_rejected_before_record_payload_read(tmp_path, monkeypatch, kind):
    with h5py.File(tmp_path / 'record.h5', 'w') as handle:
        kwargs = {'vlen': dict(dtype=h5py.string_dtype('utf-8')),
                  'ascii': dict(dtype=h5py.string_dtype('ascii', 65536)),
                  'numeric': dict(dtype=np.dtype('i8')),
                  'vector': dict(shape=(1,)), 'oversize': dict(width=65537)}.get(kind, {})
        # Numeric storage is installed separately; the format preflight must reject it.
        if kind == 'numeric':
            native(handle, record())
            del handle[PATH + '/record_json']
            handle[PATH + '/record_json'] = 0
        else:
            native(handle, record(), **kwargs)
        group, dataset = handle[PATH], handle[PATH + '/record_json']
        if kind == 'child':
            group['extra'] = 0
        elif kind == 'schema':
            del group.attrs['schema']
            attribute(group, 'schema', 'wrong')
        elif kind in ('units', 'axis', 'description'):
            key = 'logical_axis_order' if kind == 'axis' else kind
            del dataset.attrs[key]
            attribute(dataset, key, 'wrong')
        def forbidden(*args):
            pytest.fail('record payload read before header validation')
        monkeypatch.setattr(native_module, '_read_audit_scalar_bytes', forbidden)
        with pytest.raises(ContractError):
            _read_final_fixed_point_audit(handle)


def test_native_metadata_uses_same_physical_budget(tmp_path):
    path = tmp_path / 'record.h5'
    with h5py.File(path, 'w') as handle:
        native(handle, record())
    budget = _IOBudget(None)
    budget.phase = 'hdf'
    with _BudgetStream(path.open('rb', buffering=0), budget) as stream:
        with h5py.File(stream, 'r') as handle:
            before = budget.bytes_read_total
            assert _read_final_fixed_point_audit(handle) == raw(record())
            assert budget.bytes_read_total > before
    assert budget.hdf_bytes == budget.bytes_read_total > len(raw(record()))
    # A fresh fileobj run cannot deliver even its opening header under a one-byte cap.
    budget = _IOBudget(1)
    budget.phase = 'hdf'
    with _BudgetStream(path.open('rb', buffering=0), budget) as stream:
        with pytest.raises(ContractError, match='budget'):
            with h5py.File(stream, 'r') as handle:
                _read_final_fixed_point_audit(handle)
    assert budget.bytes_read_total <= 1


@pytest.mark.parametrize('field,replacement', [('chunks', (1,)), ('compression', 'gzip'),
                                             ('external', [('outside', 0, 1)]), ('is_virtual', True)])
def test_storage_features_fail_preflight(tmp_path, monkeypatch, field, replacement):
    with h5py.File(tmp_path / 'record.h5', 'w') as handle:
        native(handle, record())
        monkeypatch.setattr(h5py.Dataset, field, property(lambda _: replacement))
        def forbidden(*args):
            pytest.fail('record payload read before storage preflight')
        monkeypatch.setattr(native_module, '_read_audit_scalar_bytes', forbidden)
        with pytest.raises(ContractError):
            _read_final_fixed_point_audit(handle)


def test_selected_identity_is_independent_of_audit_record_binding(tmp_path):
    value = record()
    declaration = commit(value)
    declaration['identity']['point_id'] = 'different-selected-point'
    with h5py.File(tmp_path / 'record.h5', 'w') as handle:
        native(handle, value)
        with pytest.raises(ContractError):
            _pair(handle, declaration, final_state=True)


def test_error_finite_zero_remains_error(tmp_path):
    value = record()
    value['dyson']['ratio'] = dict(status='error', value=0, nonfinite=None, reason='nonfinite_measurement')
    value.update(record_status='partial_error', reason='measurement_error')
    with h5py.File(tmp_path / 'record.h5', 'w') as handle:
        native(handle, value)
        result = _pair(handle, commit(value), final_state=True)
    assert result['status'] == 'partial_error'
    assert result['record']['dyson']['ratio']['value'] == 0
    assert result['record']['dyson']['ratio']['status'] == 'error'


def publish(root, value=None, *, recovery=False, sidecar=None, sidecar_value=None):
    """Literal commit/receipt JSON, no product artifact encoder or solver arrays."""
    root.mkdir()
    owners = [('physics.h5', 'physics.full', 'qcl-negf-physics-v4', value)]
    if sidecar:
        owners.append(('sidecar.h5', sidecar,
            'qcl-negf-checkpoint-v4' if sidecar == 'recovery' else 'qcl-negf-physics-analysis-v4', sidecar_value))
    rows = []
    for name, role, schema, content in owners:
        path = root / name
        with h5py.File(path, 'w') as handle:
            declare_native(handle, schema, role, scba_rows=0)
            identity_path = ('metadata/identity_json' if role == 'physics.analysis'
                             else 'metadata/point_identity_json')
            handle[identity_path] = json.dumps(IDENTITY)
            if content is not None:
                native(handle, content, identity_path=identity_path)
        payload = path.read_bytes()
        rows.append(dict(path=name, role=role, bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest(),
                         media_type='application/x-hdf5', profile='full-state', dependencies=[], schema=schema))
    declaration = commit(value, recovery=recovery)
    declaration.update(schema='qcl-negf.artifact-commit.v2', contract_set=CONTRACT_SET,
                       generation=1, state_id='s', state_sequence=1, artifacts=rows)
    payload = raw(declaration)
    (root / 'commit.json').write_bytes(payload)
    (root / 'receipt.json').write_bytes(raw(dict(schema='qcl-negf-recovery-receipt-v1',
        status='verified', publication_scope='local_filesystem', identity=dict(IDENTITY),
        state_id='s', state_sequence=1, commit_sha256=hashlib.sha256(payload).hexdigest())))
    return root / 'commit.json'


def test_native5_does_not_autoload_unreferenced_final_audit(tmp_path):
    from result_v2_fixture import source, IDENTITY as SELECTED
    selected = source(tmp_path)
    with h5py.File(tmp_path / 'unreferenced-audit.h5', 'w') as handle:
        native(handle, record())
    with StateReader(tmp_path, state_ref=selected, expected_identity=SELECTED) as reader:
        assert reader.source.status['physics']['state'] == 'not_measured'
        assert reader.source.integrity['selected_payloads_verified'] == []
        assert not reader.source.capabilities['exact_restore']['available']


@pytest.mark.parametrize('sidecar', ['physics.analysis', 'recovery'])
def test_native5_optional_sidecars_do_not_grant_capability(tmp_path, sidecar):
    from result_v2_fixture import source, IDENTITY as SELECTED
    selected = source(tmp_path)
    with h5py.File(tmp_path / 'orphan.h5', 'w') as handle:
        native(handle, record())
    with StateReader(tmp_path, state_ref=selected, expected_identity=SELECTED) as reader:
        capability = 'stored_observables' if sidecar == 'physics.analysis' else 'exact_checkpoint'
        assert not reader.source.capabilities[capability]['available']
        assert reader.source.status['physics']['state'] == 'not_measured'


@pytest.mark.parametrize('sidecar', ['physics.analysis', 'recovery'])
def test_native5_orphan_is_not_an_authoritative_optional_owner(tmp_path, sidecar):
    from result_v2_fixture import source, IDENTITY as SELECTED
    selected = source(tmp_path)
    (tmp_path / ('uncommitted-' + sidecar + '.h5')).write_bytes(b'not a committed native object')
    with StateReader(tmp_path, state_ref=selected, expected_identity=SELECTED) as reader:
        assert reader.source.integrity['selected_payloads_verified'] == []
        assert not reader.source.capabilities['exact_checkpoint']['available']


# Attribute profile A fixtures use an explicit HDF5 file/memory type. Python's
# automatic str/bytes attribute writer would manufacture vlen instead.
def fixed_attribute(parent, name, payload, *, width=None, charset=h5py.h5t.CSET_UTF8,
                    padding=h5py.h5t.STR_NULLPAD, shape=None, datatype=None):
    width = len(payload) if width is None else width
    dtype = h5py.h5t.C_S1.copy() if datatype is None else datatype.copy()
    space = (h5py.h5s.create(h5py.h5s.SCALAR) if shape is None else
             h5py.h5s.create(h5py.h5s.NULL) if shape == 'null' else
             h5py.h5s.create_simple(shape))
    attr = None
    try:
        if datatype is None:
            dtype.set_size(width)
            dtype.set_cset(charset)
            dtype.set_strpad(padding)
        attr = h5py.h5a.create(parent.id, name.encode(), dtype, space)
        if datatype is None and shape is None:
            fill = b' ' if padding == h5py.h5t.STR_SPACEPAD else b'\0'
            stored = payload.ljust(width, fill)
            assert len(stored) == width
            array = np.frombuffer(stored, dtype=f'S{width}').reshape(())
            attr.write(array, mtype=dtype)
            # Independent fixture certification: exact low metadata and bytes.
            readback = np.empty((), dtype=f'S{width}')
            attr.read(readback, mtype=dtype)
            assert readback.tobytes() == stored, 'fixture writer hid adversarial bytes'
            actual, actual_space = attr.get_type(), attr.get_space()
            try:
                assert actual.get_class() == h5py.h5t.STRING
                assert not actual.is_variable_str()
                assert actual.get_size() == width
                assert actual.get_cset() == charset
                assert actual.get_strpad() == padding
                assert actual_space.get_simple_extent_type() == h5py.h5s.SCALAR
                assert actual_space.get_simple_extent_dims() == ()
            finally:
                actual_space.close()
                actual.close()
    finally:
        if attr is not None:
            attr.close()
        space.close()
        dtype.close()


@pytest.mark.parametrize('expected', [SCHEMA, '1', 'scalar',
    'Versioned final fixed-point raw measurement record'])
@pytest.mark.parametrize('charset', [h5py.h5t.CSET_ASCII, h5py.h5t.CSET_UTF8])
@pytest.mark.parametrize('padding', [h5py.h5t.STR_NULLTERM, h5py.h5t.STR_NULLPAD,
                                    h5py.h5t.STR_SPACEPAD])
@pytest.mark.parametrize('width_case', ['exact', 'one_pad', 'upper'])
def test_profile_a_genuine_fixed_encodings(tmp_path, expected, charset, padding, width_case):
    width = {'exact': len(expected), 'one_pad': len(expected) + 1, 'upper': 65536}[width_case]
    with h5py.File(tmp_path / 'attribute.h5', 'w', libver='latest', track_order=True) as handle:
        fixed_attribute(handle, 'constant', expected.encode(), width=width, charset=charset, padding=padding)
        native_module._audit_attribute(handle, 'constant', expected)


@pytest.mark.parametrize('padding', [h5py.h5t.STR_NULLTERM, h5py.h5t.STR_NULLPAD,
                                    h5py.h5t.STR_SPACEPAD])
@pytest.mark.parametrize('payload', [b'scalar\0X', b'sca\0lar', b' scalar', b'Xscalar',
                                     b'scalarX', b'Scalar', b'scalar\xff'])
def test_profile_a_preserves_and_rejects_hidden_suffix_and_wrong_constants(tmp_path, padding, payload):
    with h5py.File(tmp_path / 'attribute.h5', 'w', libver='latest', track_order=True) as handle:
        fixed_attribute(handle, 'constant', payload, padding=padding)
        with pytest.raises(ContractError):
            native_module._audit_attribute(handle, 'constant', 'scalar')


@pytest.mark.parametrize('charset', [h5py.h5t.CSET_ASCII, h5py.h5t.CSET_UTF8])
def test_profile_a_strict_decode(tmp_path, charset):
    with h5py.File(tmp_path / 'attribute.h5', 'w', libver='latest', track_order=True) as handle:
        fixed_attribute(handle, 'constant', b'scalar\xff', charset=charset)
        with pytest.raises(ContractError, match='invalid string bytes'):
            native_module._audit_attribute(handle, 'constant', 'scalar')


def forbid_attribute_value_read(monkeypatch):
    calls = []
    def forbidden(*args):
        calls.append(args)
        pytest.fail('attribute value-read/allocation before preflight')
    monkeypatch.setattr(native_module, '_read_audit_attribute_bytes', forbidden)
    return calls


@pytest.mark.parametrize('encoding', ['ascii', 'utf-8'])
def test_profile_a_genuine_vlen_rejected_before_value_read(tmp_path, monkeypatch, encoding):
    with h5py.File(tmp_path / 'attribute.h5', 'w', libver='latest', track_order=True) as handle:
        handle.attrs.create('constant', 'scalar' if encoding == 'utf-8' else b'scalar',
                            dtype=h5py.string_dtype(encoding))
        attr = handle.attrs.get_id('constant')
        dtype, space = attr.get_type(), attr.get_space()
        try:
            assert dtype.get_class() == h5py.h5t.STRING and dtype.is_variable_str()
            assert space.get_simple_extent_type() == h5py.h5s.SCALAR
        finally:
            space.close()
            dtype.close()
            attr.close()
        calls = forbid_attribute_value_read(monkeypatch)
        with pytest.raises(ContractError, match='fixed-string scalar'):
            native_module._audit_attribute(handle, 'constant', 'scalar')
        assert calls == []


@pytest.mark.parametrize('kind', ['missing', 'null', 'one_element', 'rank_two', 'numeric',
                                  'compound', 'array_type', 'opaque', 'too_short', 'oversize'])
def test_profile_a_bad_type_shape_and_width_before_value_read(tmp_path, monkeypatch, kind):
    with h5py.File(tmp_path / 'attribute.h5', 'w', libver='latest', track_order=True) as handle:
        if kind in ('null', 'one_element', 'rank_two'):
            fixed_attribute(handle, 'constant', b'scalar',
                            shape={'null': 'null', 'one_element': (1,), 'rank_two': (1, 1)}[kind])
        elif kind == 'numeric':
            handle.attrs['constant'] = 1
        elif kind in ('compound', 'array_type', 'opaque'):
            dtype = (h5py.h5t.py_create(np.dtype([('value', 'i4')])) if kind == 'compound' else
                     h5py.h5t.array_create(h5py.h5t.STD_I32LE, (2,)) if kind == 'array_type' else
                     h5py.h5t.create(h5py.h5t.OPAQUE, 4))
            try:
                fixed_attribute(handle, 'constant', b'', datatype=dtype)
            finally:
                dtype.close()
        elif kind in ('too_short', 'oversize'):
            fixed_attribute(handle, 'constant', b'x', width=5 if kind == 'too_short' else 65537)
        calls = forbid_attribute_value_read(monkeypatch)
        with pytest.raises(ContractError):
            native_module._audit_attribute(handle, 'constant', 'scalar')
        assert calls == []


@pytest.mark.parametrize('name', ['schema', 'units', 'logical_axis_order', 'description'])
def test_every_required_new_attribute_is_required(tmp_path, name):
    with h5py.File(tmp_path / 'record.h5', 'w') as handle:
        native(handle, record())
        parent = handle[PATH] if name == 'schema' else handle[PATH + '/record_json']
        del parent.attrs[name]
        with pytest.raises(ContractError, match='missing'):
            _read_final_fixed_point_audit(handle)


@pytest.mark.parametrize('mode', ['success', 'preflight', 'read_error', 'wrong_value'])
def test_profile_a_same_open_ids_and_cleanup(tmp_path, monkeypatch, mode):
    with h5py.File(tmp_path / 'attribute.h5', 'w', libver='latest', track_order=True) as handle:
        fixed_attribute(handle, 'constant', b'wrong!' if mode == 'wrong_value' else b'scalar')
        manager_type = type(handle.attrs)
        original = manager_type.get_id
        opened, reads, lookups = [], [], []
        if mode == 'preflight':
            # A genuine vlen replaces the fixed attribute before the lookup.
            del handle.attrs['constant']
            handle.attrs.create('constant', 'scalar', dtype=h5py.string_dtype('utf-8'))
        class Attribute:
            def __init__(self, wrapped):
                self.wrapped, self.datatype = wrapped, None
            def get_type(self):
                self.datatype = self.wrapped.get_type()
                opened.append(self.datatype)
                return self.datatype
            def get_space(self):
                result = self.wrapped.get_space()
                opened.append(result)
                return result
            def read(self, target, *, mtype):
                assert mtype is self.datatype
                assert all(identifier.valid for identifier in opened)
                reads.append(target.nbytes)
                if mode == 'read_error':
                    raise RuntimeError('fixture raw read failed')
                return self.wrapped.read(target, mtype=mtype)
            def close(self):
                self.wrapped.close()
        def track(manager, name):
            lookups.append(name)
            attr = original(manager, name)
            opened.append(attr)
            return Attribute(attr)
        monkeypatch.setattr(manager_type, 'get_id', track)
        if mode == 'success':
            native_module._audit_attribute(handle, 'constant', 'scalar')
        else:
            with pytest.raises(RuntimeError if mode == 'read_error' else ContractError):
                native_module._audit_attribute(handle, 'constant', 'scalar')
        assert lookups == ['constant']
        assert reads == ([] if mode == 'preflight' else [6])
        assert len(opened) == 3 and all(not identifier.valid for identifier in opened)


@pytest.mark.parametrize('change', ['charset', 'padding', 'absurd_width'])
def test_profile_a_guarded_metadata_adapter(tmp_path, monkeypatch, change):
    # HDF5 cannot create every invalid enum/absurd type. This supplements the
    # genuine fixtures above, with no value allocation/read in this adapter.
    class Identifier:
        valid = True
        def close(self):
            self.valid = False
    class Datatype(Identifier):
        def get_class(self): return h5py.h5t.STRING
        def is_variable_str(self): return False
        def get_size(self): return 2**62 if change == 'absurd_width' else 6
        def get_cset(self): return 999 if change == 'charset' else h5py.h5t.CSET_UTF8
        def get_strpad(self): return 999 if change == 'padding' else h5py.h5t.STR_NULLPAD
    class Space(Identifier):
        def get_simple_extent_type(self): return h5py.h5s.SCALAR
        def get_simple_extent_dims(self): return ()
    datatype, space = Datatype(), Space()
    class Attribute(Identifier):
        def get_type(self): return datatype
        def get_space(self): return space
    attribute_id = Attribute()
    class Attributes:
        def get_id(self, name): return attribute_id
    class Parent:
        attrs = Attributes()
    calls = forbid_attribute_value_read(monkeypatch)
    with pytest.raises(ContractError):
        native_module._audit_attribute(Parent(), 'constant', 'scalar')
    assert calls == []
    assert not any(identifier.valid for identifier in (attribute_id, datatype, space))


def replace_analysis_identity_and_refresh(commit_path, identity, *, fresh_alias=False):
    """Change only analysis container identity, then certify the new bytes.

    Audit records remain the same literal object; SHA/receipt integrity must
    not mask the independent native point-identity requirement.
    """
    declaration = json.loads(commit_path.read_bytes())
    sidecar = commit_path.parent / 'sidecar.h5'
    with h5py.File(sidecar, 'r+') as handle:
        del handle['metadata/identity_json']
        if identity is not None:
            handle['metadata/identity_json'] = json.dumps(identity)
        if fresh_alias:
            handle['metadata/point_identity_json'] = json.dumps(IDENTITY)
    payload = sidecar.read_bytes()
    owner = next(row for row in declaration['artifacts'] if row['role'] == 'physics.analysis')
    owner.update(bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest())
    commit_bytes = raw(declaration)
    commit_path.write_bytes(commit_bytes)
    receipt_path = commit_path.parent / 'receipt.json'
    receipt = json.loads(receipt_path.read_bytes())
    receipt['commit_sha256'] = hashlib.sha256(commit_bytes).hexdigest()
    receipt_path.write_bytes(raw(receipt))
    assert owner['bytes'] == sidecar.stat().st_size
    assert owner['sha256'] == hashlib.sha256(sidecar.read_bytes()).hexdigest()
    assert json.loads(receipt_path.read_bytes())['commit_sha256'] == hashlib.sha256(
        commit_path.read_bytes()).hexdigest()


def test_native5_independent_analysis_is_explicitly_unavailable(tmp_path):
    from result_v2_fixture import source, IDENTITY as SELECTED
    selected = source(tmp_path)
    with StateReader(tmp_path, state_ref=selected, expected_identity=SELECTED) as reader:
        assert reader.source.capabilities['stored_observables'] == dict(
            available=False, reason='explicit_final_observations_ref_required')


@pytest.mark.parametrize('field,value', [('missing', None), ('attempt', 2),
    ('state_sequence', 2), ('state_id', 'stale-state'), ('plan_fingerprint', 'b' * 64),
    ('execution_id', 'stale-execution'), ('point_id', 'stale-point')])
@pytest.mark.parametrize('fresh_alias', [False, True])
def test_native5_rejects_stale_selection_or_retired_identity_alias(tmp_path, field, value, fresh_alias):
    from result_v2_fixture import source, IDENTITY as SELECTED
    selected = source(tmp_path)
    expected = copy.deepcopy(SELECTED)
    if field == 'missing':
        expected = {}
    elif field in ('state_id', 'state_sequence'):
        selected[field] = str(value)
    else:
        expected['plan_sha256' if field == 'plan_fingerprint' else field] = 'c' * 64 if field == 'plan_fingerprint' else str(value)
    if fresh_alias:
        expected['plan_fingerprint'] = 'a' * 64
    with pytest.raises(ContractError):
        StateReader(tmp_path, state_ref=selected, expected_identity=expected)


@pytest.mark.parametrize('canonical_record', [False, True])
def test_native5_never_infers_implicit_identity_from_optional_record(tmp_path, canonical_record):
    from result_v2_fixture import source
    source(tmp_path)
    if canonical_record:
        with h5py.File(tmp_path / 'unreferenced-audit.h5', 'w') as handle:
            native(handle, record())
    with pytest.raises(TypeError):
        StateReader(tmp_path)


def test_actual_runner_native5_final_transport_when_explicitly_bound(tmp_path):
    """Read original Julia constructor CP/D/O bytes, without a solve or actor.

    The existing Runner focused test prints OUTWARD_FIXTURE_ROOT. Its selected
    point is manufactured from a failed terminal; the separate actual empty
    running writer control is not a stationary result or scientific acceptance.
    Old D2 filename/private-wrapper/Dyson-zero assertions remain in the literal
    format tests above; they are not claims about this negative native5 carrier.
    """
    import os
    from pathlib import Path
    import tarfile
    from qcl_negf_results.state import StateReader, ResultReadLimits
    from qcl_negf_results.final_observations import FinalObservationsReader, describe_checkpoint_inputs
    from qcl_negf_results.final_diagnostics import FinalDiagnosticsReader
    from qcl_negf_results.export import preview, export_snapshot
    from qcl_negf_results.archive import receive
    from qcl_negf_contracts.outward import validate_outward_result, validate_outward_native_source
    from qcl_negf_contracts.results import validate_manifest

    supplied = os.environ.get('QCL_NATIVE5_FINAL_FIXTURE_ROOT')
    if supplied is None:
        legacy_bound = 'QCL_SCI05_RUNNER_FIXTURE_ROOT' in os.environ
        assert not legacy_bound, (
            'legacy fixture binding retired; supply QCL_NATIVE5_FINAL_FIXTURE_ROOT')
        pytest.skip('not_run: explicit original native5 constructor fixture root required')
    root = Path(supplied)
    assert root.is_absolute(), 'producer fixture root must be explicitly absolute'
    metadata_cap, io_cap, export_cap = 1024**2, 4*1024**2, 32*1024**2

    def metadata(path):
        with path.open('rb') as stream:
            payload = stream.read(metadata_cap + 1)
        assert len(payload) <= metadata_cap, 'fixture metadata exceeds admission'
        return payload

    def file_hash(path):
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            while block := stream.read(64*1024):
                digest.update(block)
        return digest.hexdigest()

    outward_path = root / 'outward-v4.json'
    assert outward_path.is_file(), 'bound original native5 outward-v4.json is missing'
    outward = validate_outward_result(json.loads(metadata(outward_path)))
    assert outward['schema'] == 'qcl-negf-series-result-v4'
    assert outward['contract_set'] == 'qcl-negf.results.v2' and len(outward['points']) == 1
    source = outward['points'][0]['data']['native_source']
    assert source['kind'] == 'final' and source['relation'] == 'current_point_attempt'
    assert source['artifact_root'] == '.'
    selected, diagnostics_ref, observations_ref = (
        source['selected_state'], source['diagnostics'], source['observations'])
    assert set(selected['identity']) == {
        'run_id', 'execution_id', 'point_id', 'attempt', 'branch_id', 'model_sha256', 'plan_sha256'}
    for name, expected in dict(run_id='run', execution_id='exec', point_id='point',
                               attempt='1', branch_id='branch1').items():
        assert selected['identity'][name] == expected
    assert selected['state_id'] == 'epoch1-state-1' and selected['state_sequence'] == '1'
    manifest_ref = selected['manifest']
    manifest_raw = metadata(root / manifest_ref['path'])
    assert len(manifest_raw) == int(manifest_ref['bytes'])
    assert hashlib.sha256(manifest_raw).hexdigest() == manifest_ref['sha256']
    manifest = validate_manifest(json.loads(manifest_raw))
    references = [manifest_ref] + [
        {key: row[key] for key in ('path', 'bytes', 'sha256', 'media_type', 'schema')}
        for row in manifest['artifacts']]
    assert sum(int(ref['bytes']) for ref in references) <= export_cap
    before = {ref['path']: file_hash(root / ref['path']) for ref in references}
    assert all(before[ref['path']] == ref['sha256'] and
               (root / ref['path']).stat().st_size == int(ref['bytes']) for ref in references)
    outward_before = file_hash(outward_path)
    limits = ResultReadLimits(maximum_io_bytes=io_cap)
    deliveries = []

    def note(label, reader):
        counters = dict(reader.io_counters)
        assert counters['bytes_read_total'] <= io_cap
        deliveries.append((label, counters))

    with StateReader(root, state_ref=selected, expected_identity=selected['identity'], limits=limits) as reader:
        note('construct', reader)
        checkpoint = reader.checkpoint()
        note('checkpoint', reader)
        with FinalDiagnosticsReader(reader, artifact_ref=diagnostics_ref) as view:
            diagnostic = view.describe()
            note('diagnostics', reader)
        with FinalObservationsReader(reader, artifact_ref=observations_ref) as view:
            observation = view.describe()
            note('observations', reader)
        inputs = describe_checkpoint_inputs(reader,
            plan_ref=observation.declaration['request']['plan'],
            model_ref=observation.declaration['request']['model'])
        note('original_inputs', reader)
        assert observation.source == diagnostic.source == inputs.source == selected
        assert observation.invocation == diagnostic.invocation == inputs.invocation == {'source_epoch': 'epoch1'}
        assert observation.checkpoint == diagnostic.checkpoint == inputs.checkpoint == manifest['checkpoint']
        assert observation.diagnostics == diagnostics_ref and diagnostic.artifact == diagnostics_ref
        assert observation.artifact == observations_ref
        assert inputs.plan == observation.plan and inputs.model == observation.model
        assert inputs.plan_raw.startswith(b'# frozen original plan\r\n')
        for ref, payload in ((inputs.plan_ref, inputs.plan_raw), (inputs.model_ref, inputs.model_raw)):
            assert len(payload) == int(ref['bytes']) and hashlib.sha256(payload).hexdigest() == ref['sha256']
            assert metadata(root / ref['path']) == payload
        assert inputs.plan_ref['sha256'] == selected['identity']['plan_sha256']
        assert inputs.model_ref['sha256'] != inputs.model['configuration_hash']
        assert inputs.model['configuration_hash'] == selected['identity']['model_sha256']
        assert validate_outward_native_source(source, manifest=manifest, manifest_ref=manifest_ref,
            checkpoint=checkpoint, diagnostics=diagnostic.declaration,
            observations=observation.declaration, plan=inputs.plan, model=inputs.model) == source
        assert observation.declaration['context']['dimensions'] == dict(Nz='25', NE='9', Nk='3', Nb='2')
        assert diagnostic.declaration['dimensions'] == dict(energy='9', momentum='3', basis='2')
        summary = observation.declaration['summary']
        assert summary['status'] == 'scba_invalid_candidate' and summary['quality'] == 'invalid'
        assert summary['converged'] is False and summary['scientific_accepted'] is False
        assert summary['discretization_verified'] == 'not_measured'
        assert reader.source.status['process']['state'] == 'failed'
        for name in ('convergence', 'physics'):
            assert reader.source.status[name] == dict(state='recorded', reason=None, evidence=observations_ref)
        original = checkpoint['state']['fields']['final_evidence']['fields']
        assert original['original_status']['value'] == 'scba_invalid_candidate'
        assert original['original_converged'] is False
        assert original['report']['fields']['passed'] is False
        assert original['report']['fields']['messages']['values'] == ['constructor current/original report']
        metrics = {row['name']: row['value'] for row in diagnostic.declaration['solution_metrics']['summary_metrics']}
        # Public diagnostic summary labels map the stored r_K/r_U report to
        # keldysh/hartree; the raw report itself remains in CP final_evidence.
        r_k = int(metrics['keldysh']['bits'], 16)
        assert r_k & 0x7ff0000000000000 == 0x7ff0000000000000 and r_k & 0x000fffffffffffff
        assert metrics['hartree']['bits'] == '7ff0000000000000'
        green = checkpoint['state']['fields']['scba']['fields']['green']['fields']['Gᴿ']
        assert next(row for row in manifest['artifacts'] if row['path'] == green['file']['path'])['role'] == 'physics.full'
        description = reader.describe_array(green)
        note('green_metadata', reader)
        assert description.logical_shape == description.stored_shape == (9, 3, 2, 2)
        block = reader.read_array(green, (slice(0, 1), slice(0, 1), slice(None), slice(None)), maximum_bytes=64)
        note('green_slice', reader)
        assert block.values.tobytes() == np.full((1, 1, 2, 2), 1-1j, dtype='<c16').tobytes()
        with FinalObservationsReader(reader, artifact_ref=observations_ref) as view:
            array = view.describe_array('localized_wavefunctions_per_sqrt_nm')
            note('wavefunction_metadata', reader)
            assert array.logical_shape == array.stored_shape == (25, 2)
            block = view.read_array('localized_wavefunctions_per_sqrt_nm', (slice(0, 3), 1), maximum_bytes=48)
            note('wavefunction_slice', reader)
            with h5py.File(root / observations_ref['path'], 'r') as original_hdf:
                assert block.values.tobytes() == original_hdf[array.declaration['dataset']][0:3, 1].tobytes()
        integral_shapes = dict(spectral_integral=(3, 2, 2), spectral_integral_eigenvalues=(3, 2),
                               spectral_trace_cumulative=(9, 3), negative_energy_integral_scaled=(6, 3))
        assert diagnostic.declaration['positivity']['status'] == 'measured'
        assert diagnostic.declaration['positivity']['data'] is not None
        with FinalDiagnosticsReader(reader, artifact_ref=diagnostics_ref) as view:
            for name, shape in integral_shapes.items():
                array = view.describe_integral(name)
                note(name + '_metadata', reader)
                assert array.logical_shape == array.stored_shape == shape
                selection = (slice(0, 1),) + (slice(None),)*(len(shape)-1)
                block = view.read_integral(name, selection, maximum_bytes=64)
                note(name + '_slice', reader)
                with h5py.File(root / diagnostics_ref['path'], 'r') as original_hdf:
                    assert block.values.tobytes() == original_hdf[array.declaration['dataset']][selection].tobytes()
        assert sum(c['bytes_read_total'] for _, c in deliveries) <= io_cap, deliveries
        assert any(c['hdf_bytes'] > 0 for _, c in deliveries)
        assert any(c['metadata_bytes'] > 0 for _, c in deliveries)
        assert sum(c['sha_bytes'] for _, c in deliveries) >= sum(
            int(ref['bytes']) for ref in (diagnostics_ref, observations_ref, green['file']))
        with pytest.raises(ContractError) as underbuffer:
            reader.read_array(green, (slice(0, 1), slice(0, 1), slice(None), slice(None)), maximum_bytes=8)
        assert underbuffer.value.code == 'budget_exceeded'

    stale = copy.deepcopy(selected)
    stale['state_sequence'] = '2'
    with pytest.raises(ContractError):
        StateReader(root, state_ref=stale, expected_identity=selected['identity'], limits=limits)
    with pytest.raises(ContractError) as underbudget:
        StateReader(root, state_ref=selected, expected_identity=selected['identity'],
                    limits=ResultReadLimits(maximum_io_bytes=1))
    assert underbudget.value.code == 'budget_exceeded'
    with StateReader(root, state_ref=selected, expected_identity=selected['identity'], limits=limits) as reader:
        for cls, ref in ((FinalObservationsReader, observations_ref), (FinalDiagnosticsReader, diagnostics_ref)):
            for field, value in (('sha256', '0'*64), ('bytes', '1')):
                wrong = dict(ref, **{field: value})
                with cls(reader, artifact_ref=wrong) as view, pytest.raises(ContractError):
                    view.describe()

    for profile in ('science', 'full-state'):
        receipts = []
        for warm in (False, True):
            destination = tmp_path / profile / str(warm)
            with StateReader(root, state_ref=selected, expected_identity=selected['identity'], limits=limits) as reader:
                if warm:
                    with FinalObservationsReader(reader, artifact_ref=observations_ref) as view:
                        view.describe()
                info = preview(reader, profile=profile, observations_ref=observations_ref, diagnostics_ref=diagnostics_ref)
                assert info['capabilities']['stored_observables']['available']
                assert info['capabilities']['stored_final_diagnostics']['available']
                receipt = export_snapshot(reader, destination, profile=profile, byte_budget=export_cap,
                    reserve_bytes=0, observations_ref=observations_ref, diagnostics_ref=diagnostics_ref)
            receipts.append(receipt)
            assert receipt['canonical_source_complete'] is (profile == 'full-state')
            with tarfile.open(destination / receipt['archive'], 'r:xz') as archive:
                assert archive.getmember('manifest.json').size <= metadata_cap
                export_manifest = json.load(archive.extractfile('manifest.json'))
                copied = 0
                for row in export_manifest['source_locator']:
                    reference = row['reference']
                    digest = hashlib.sha256()
                    count = 0
                    with archive.extractfile(row['object']) as exported, (root / reference['path']).open('rb') as original_file:
                        while chunk := exported.read(64*1024):
                            copied += len(chunk)
                            count += len(chunk)
                            assert copied <= export_cap and chunk == original_file.read(len(chunk))
                            digest.update(chunk)
                        assert not original_file.read(1)
                    assert count == int(reference['bytes']) and digest.hexdigest() == reference['sha256']
            received = receive([destination / receipt['archive']],
                destination / 'received' if profile == 'full-state' else None, receipt=receipt)
            assert received['verified'] and received['profile'] == profile
            if profile == 'science':
                assert received['source_root'] is None
            else:
                assert received['source_state'] == selected
                with StateReader(received['source_root'], state_ref=selected,
                                 expected_identity=selected['identity'], limits=limits) as restored:
                    assert restored.checkpoint() == checkpoint
                    with FinalDiagnosticsReader(restored, artifact_ref=diagnostics_ref) as view:
                        assert view.describe().declaration == diagnostic.declaration
                    with FinalObservationsReader(restored, artifact_ref=observations_ref) as view:
                        assert view.describe().declaration == observation.declaration
                    restored_inputs = describe_checkpoint_inputs(restored,
                        plan_ref=inputs.plan_ref, model_ref=inputs.model_ref)
                    assert restored_inputs.plan_raw == inputs.plan_raw and restored_inputs.model_raw == inputs.model_raw
                copied_model = Path(received['source_root']) / inputs.model_ref['path']
                payload = metadata(copied_model)
                copied_model.write_bytes(payload[:-1] + bytes([payload[-1] ^ 1]))
                with StateReader(received['source_root'], state_ref=selected,
                                 expected_identity=selected['identity'], limits=limits) as corrupted:
                    with pytest.raises(ContractError):
                        describe_checkpoint_inputs(corrupted, plan_ref=inputs.plan_ref, model_ref=inputs.model_ref)
        assert receipts[0]['snapshot_identity'] == receipts[1]['snapshot_identity']
        assert receipts[0]['sha256'] == receipts[1]['sha256']
    with StateReader(root, state_ref=selected, expected_identity=selected['identity'], limits=limits) as reader:
        with FinalObservationsReader(reader, artifact_ref=observations_ref) as view:
            view.describe()
        destination = tmp_path / 'wrong-selected-observations'
        with pytest.raises(ContractError):
            export_snapshot(reader, destination, profile='science', byte_budget=export_cap, reserve_bytes=0,
                            observations_ref=dict(observations_ref, sha256='0'*64))
        assert not list(destination.glob('*.tar.xz'))
    assert file_hash(outward_path) == outward_before
    assert all(file_hash(root / path) == digest for path, digest in before.items())
