"""36 manufactured native-v4 cases; no solver or production archive input.

Oracles use a closed two-by-two inverse and literal weighted full-matrix sums.
The arithmetic bounds below are engineering roundoff bounds, not S01 acceptance.
"""
import hashlib
import importlib
import json
import math
from pathlib import Path

import h5py
import numpy as np
import pytest

from qcl_negf_contracts.artifacts import CONTRACT_SET
from qcl_negf_contracts.messages import ContractError
from qcl_negf_results.commits import artifact_row, json_bytes
from qcl_negf_results.model import canonical_bytes
from qcl_negf_results.state import StateReader
from native_result_fixtures import declare_native

IDENTITY = {"execution_id": "manufactured", "point_id": "analytic", "attempt": 1,
            "plan_fingerprint": "a" * 64, "state_id": "manufactured-state", "state_sequence": 1}
FAMILIES = ("LO", "acoustic", "impurity", "IFR", "alloy")
BOUNDARIES = ("embedding_total", "embedding_plus", "embedding_minus")
E = np.array([-0.20, -0.07, 0.01, 0.15, 0.40])
WE = np.array([0.02, 0.05, 0.08, 0.04, 0.09])
WK = np.array([0.07, 0.19])
H = np.array([[0.30, 0.04 + 0.02j], [0.04 - 0.02j, 0.50]])
GAMMA = np.array([[0.12, 0.03j], [-0.03j, 0.09]])
D = np.array([[0, 1j], [-1j, 0]])
MU = 0.012
KT = 1.380649e-23 / 1.602176634e-19 * 70
DELTA = 0.03
# Both cached native dependencies and owning sources are asserted inside the packet.
EXPECTED_RESULTS = Path('/home/tolya/.codex/worktrees/r09-results-integrity/qcl-negf/components/qcl-negf-results/src')
EXPECTED_CONTRACTS = Path('/home/tolya/Загрузки/qcl-negf/components/qcl-negf-contracts/src')


def api():
    try:
        return importlib.import_module('qcl_negf_results.equilibrium')
    except ModuleNotFoundError as error:
        if error.name != 'qcl_negf_results.equilibrium':
            raise
        pytest.fail('missing stored equilibrium diagnostics API', pytrace=False)


def inverse2(m):
    a, b, c, d = m.ravel()
    return np.array([[d, -b], [-c, a]]) / (a * d - b * c)


def fd(e):
    return 1 / (1 + math.exp((float(e) * 0.1 - MU) / KT))


def analytic():
    gr = np.empty((5, 2, 2, 2), complex)
    a = np.empty_like(gr)
    sl = np.empty_like(gr)
    sg = np.empty_like(gr)
    gl = np.empty_like(gr)
    gg = np.empty_like(gr)
    for n, e in enumerate(E):
        f = fd(e)
        r = inverse2(e * np.eye(2) - H + 1j * GAMMA / 2)
        spectral = r @ GAMMA @ r.conj().T
        for k in range(2):
            gr[n, k], a[n, k] = r, spectral
            sl[n, k], sg[n, k] = 1j * f * GAMMA, -1j * (1 - f) * GAMMA
            gl[n, k], gg[n, k] = 1j * f * spectral, -1j * (1 - f) * spectral
    return dict(GR=gr, A=a, SL=sl, SG=sg, GL=gl, GG=gg)


def literal_sum(values, we=WE, wk=WK):
    return math.fsum(2 / (2 * math.pi) * float(we[e]) * float(wk[k]) * float(values[e, k])
                     for e in range(len(we)) for k in range(len(wk)))


def norm2(values, we=WE, wk=WK):
    return literal_sum(np.sum(np.abs(values) ** 2, axis=(-2, -1)), we, wk)


def number(values, we=WE, wk=WK):
    return literal_sum(np.trace(values, axis1=-2, axis2=-1).real, we, wk)


def publish(root, commit):
    commit['artifacts'][0] = artifact_row(root / 'physics.h5', relative='physics.h5',
        role='physics.full', media_type='application/x-hdf5', profile='full-state', schema='qcl-negf-physics-v4')
    payload = json_bytes(commit)
    (root / 'commit.json').write_bytes(payload)
    (root / 'receipt.json').write_bytes(json_bytes(dict(schema='qcl-negf-recovery-receipt-v1',
        status='verified', publication_scope='local_filesystem', identity=IDENTITY,
        state_id=IDENTITY['state_id'], state_sequence=1, commit_sha256=hashlib.sha256(payload).hexdigest())))


def matrix(handle, path, value):
    for part, data in [('real', value.real), ('imag', value.imag)]:
        ds = handle.create_dataset(f'{path}/{part}', data=data, chunks=(1, 1, 2, 2))
        ds.attrs.update(units='1', logical_axis_order='E,k,a,b',
            axis_coordinate_paths_json=json.dumps(['/grids_dimensionless/energy', '/grids_dimensionless/k',
                                                   '/grids_dimensionless/state_index', '/grids_dimensionless/state_index']))


@pytest.fixture
def archive(tmp_path):
    import qcl_negf_results.state as state
    import qcl_negf_contracts.artifacts as contracts
    assert Path(state.__file__).resolve().is_relative_to(EXPECTED_RESULTS)
    assert Path(contracts.__file__).resolve().is_relative_to(EXPECTED_CONTRACTS)
    root = tmp_path / 'manufactured'
    root.mkdir()
    data = analytic()
    target = number(-1j * data['GL'])
    with h5py.File(root / 'physics.h5', 'w') as handle:
        declare_native(handle, 'qcl-negf-physics-v4', 'physics.full', scba_rows=0)
        handle['metadata/point_identity_json'] = json.dumps(IDENTITY)
        scalars = {'scales/E0_eV': (0.1, 'eV'), 'scales/L0_m': (1e-8, 'm'),
            'scales/J0_A_per_m2': (13.0, 'A/m^2'), 'inputs/F_bias_V_per_m': (0.0, 'V/m'),
            'inputs/V_period_V': (0.0, 'V'), 'inputs/T_L_K': (70.0, 'K'), 'inputs/T_LO_K': (70.0, 'K'),
            'inputs/E_ref_eV': (0.004, 'eV'), 'inputs/N_dop_2D_per_m2': (target / 1e-16, 'm^-2'),
            'inputs/f_ion': (1.0, '1'), 'inputs/spin_degeneracy': (2, '1'),
            'numerical_inputs/NE': (5, '1'), 'numerical_inputs/Nk': (2, '1'), 'numerical_inputs/Nb': (2, '1')}
        scalars.update({f'numerical_inputs/scattering_{n}': (1, '1') for n in FAMILIES})
        for path, (value, units) in scalars.items():
            handle[path] = value
            handle[path].attrs.update(units=units, logical_axis_order='scalar')
        for n, values in [('energy', E), ('wE', WE), ('k', np.array([0.2, 0.7])), ('wk', WK), ('state_index', np.array([0, 1]))]:
            handle[f'grids_dimensionless/{n}'] = values
            handle[f'grids_dimensionless/{n}'].attrs.update(units='1', logical_axis_order='dimension_1')
        for n in ['A', 'GR', 'GL', 'GG']:
            matrix(handle, f'state_dimensionless/{n}', data[n])
        for n, fraction in [(n, 0.1) for n in FAMILIES] + [('embedding_total', 0.5), ('embedding_plus', 0.25), ('embedding_minus', 0.25)]:
            for part in ['SL', 'SG']:
                matrix(handle, f'selfenergy_dimensionless/{n}/{part}', fraction * data[part])
    history = root / 'history.h5'
    with h5py.File(history, 'w') as handle:
        declare_native(handle, 'qcl-negf-scientific-history-v4', 'science.history', scba_rows=0)
    configuration = {'manufactured': True}
    model = root / 'resolved_configuration.json'
    model.write_bytes(json_bytes(dict(schema='qcl-negf-resolved-configuration-v3', contract_set=CONTRACT_SET,
        configuration=configuration, hash_encoding='qcl-negf-canonical-bytes-v1',
        configuration_hash=hashlib.sha256(canonical_bytes(configuration)).hexdigest())))
    context = dict(boundary_condition='stationary_field_periodic_embedding',
        basis_contract='orthonormal finite projected basis', enabled_scattering=list(FAMILIES),
        physical_models=dict(lo_population='thermal', electron_electron=dict(mode='none')))
    commit = dict(schema='qcl-negf.artifact-commit.v2', contract_set=CONTRACT_SET, generation=1,
        identity=IDENTITY, state_id=IDENTITY['state_id'], state_sequence=1, storage_class='archive',
        stationary_assessment=dict(model_context=context, fixture_solver_status='not_run'), artifacts=[{},
            artifact_row(history, relative=history.name, role='science.history', media_type='application/x-hdf5', schema='qcl-negf-scientific-history-v4'),
            artifact_row(model, relative=model.name, role='model', media_type='application/json', schema='qcl-negf-resolved-configuration-v3')])
    publish(root, commit)
    return root, commit, data


def mutate(archive, action):
    root, commit, _ = archive
    with h5py.File(root / 'physics.h5', 'r+') as handle:
        action(handle)
    publish(root, commit)


def replace_complex(handle, path, value):
    for part in ['real', 'imag']:
        handle[f'{path}/{part}'][...] = getattr(value, part)


def diagnose(archive, **kwargs):
    root, _, _ = archive
    a = api()
    limits = kwargs.pop('limits', a.EquilibriumReadLimits(maximum_io_bytes=32 * 1024 * 1024))
    report = a.diagnose_stored_equilibrium(root / 'commit.json', expected_identity=IDENTITY,
        limits=limits, **kwargs)
    json.dumps(report, allow_nan=False)
    assert 'scientific_accepted' not in report and 'pass' not in report
    assert report['source']['commit_sha256'] == hashlib.sha256((root / 'commit.json').read_bytes()).hexdigest()
    assert report['source']['identity'] == IDENTITY
    assert report['budgets']['sha_bytes'] == (root / 'physics.h5').stat().st_size
    return report


def error(archive, code, **kwargs):
    with pytest.raises(ContractError) as caught:
        diagnose(archive, **kwargs)
    assert caught.value.code == code
    return str(caught.value)


def test_R01_known_mu_nonunit_weights(archive):
    r = diagnose(archive, mu_algorithm_abs_tol_eV=1e-12)
    assert r['applicability']['status'] == 'applicable'
    mu = r['mu']
    assert mu['lower_eV'] <= MU <= mu['upper_eV']
    assert abs(mu['value_relative_eV'] - MU) <= mu['uncertainty_eV'] + 2e-16
    assert mu['value_absolute_eV'] == mu['value_relative_eV'] + 0.004
    assert r['numbers']['raw']['value_dimensionless'] == pytest.approx(number(-1j * archive[2]['GL']), abs=2e-15)
    assert r['numbers']['raw']['value_per_m2'] == pytest.approx(number(-1j * archive[2]['GL']) / 1e-16)
    assert r['fdr']['raw']['value'] < 2e-10 and r['fdr']['normalized']['value'] < 2e-10
    for side in ['plus', 'minus']:
        assert abs(r['currents'][side]['raw_outward_A_per_m2']) < 2e-14
        assert r['currents'][side]['units'] == 'A/m^2'
    assert r['domain']['energy_nodes'] == 5
    assert r['spectral']['represented_capacity_dimensionless'] == pytest.approx(number(archive[2]['A']), abs=2e-15)
    assert r['budgets']['workspace_forecast_bytes'] <= 32 * 1024 * 1024


def test_R02_offdiagonal_normalized_defect(archive):
    data = archive[2]
    gn, gp = -1j * data['GL'] + DELTA * D, 1j * data['GG'] - DELTA * D
    mutate(archive, lambda h: (replace_complex(h, 'state_dimensionless/GL', 1j * gn), replace_complex(h, 'state_dimensionless/GG', -1j * gp)))
    r = diagnose(archive, mu_algorithm_abs_tol_eV=1e-13)
    expected = 2 * DELTA ** 2 * 2 * literal_sum(np.ones((5, 2)))
    assert r['fdr']['normalized']['numerator_squared'] == pytest.approx(expected, rel=2e-10)
    assert r['fdr']['raw']['value'] < 2e-11
    assert r['corrections']['occupied']['value'] == pytest.approx(math.sqrt(norm2(DELTA * np.broadcast_to(D, (5, 2, 2, 2))) / norm2(-1j * data['GL'])), rel=2e-12)
    assert r['corrections']['empty']['value'] == pytest.approx(math.sqrt(norm2(DELTA * np.broadcast_to(D, (5, 2, 2, 2))) / norm2(1j * data['GG'])), rel=2e-12)
    j = 13 * DELTA * np.trace(0.25 * GAMMA @ D).real * literal_sum(np.ones((5, 2)))
    assert j > 0
    for side in ['plus', 'minus']:
        assert r['currents'][side]['normalized_outward_A_per_m2'] == pytest.approx(j, abs=2e-14)
        assert abs(r['currents'][side]['raw_outward_A_per_m2']) < 2e-14
    assert r['currents']['minus']['normalized_common_z_A_per_m2'] < 0
    assert r['currents']['normalized_outward_balance_A_per_m2'] == pytest.approx(2 * j, abs=2e-14)


def test_R03_stored_sigma_offdiagonal_defect(archive):
    data = archive[2]
    sl = 0.1 * data['SL'] + 1j * DELTA * D
    sg = 0.1 * data['SG'] - 1j * DELTA * D
    mutate(archive, lambda h: (replace_complex(h, 'selfenergy_dimensionless/LO/SL', sl), replace_complex(h, 'selfenergy_dimensionless/LO/SG', sg)))
    raw_n = -1j * (data['GR'] @ (data['SL'] + 1j * DELTA * D) @ data['GR'].conj().swapaxes(-1, -2))
    raw_p = 1j * (data['GR'] @ (data['SG'] - 1j * DELTA * D) @ data['GR'].conj().swapaxes(-1, -2))
    r = diagnose(archive, mu_algorithm_abs_tol_eV=1e-13)
    assert r['numbers']['raw']['value_dimensionless'] == pytest.approx(number(raw_n), abs=2e-15)
    assert r['fdr']['raw']['numerator_squared'] == pytest.approx(norm2(raw_n + 1j * data['GL']) + norm2(raw_p - 1j * data['GG']), rel=1e-10)
    assert r['fdr']['normalized']['value'] < 2e-11


def test_R04_transpose_one_family(archive):
    data = archive[2]
    mutate(archive, lambda h: [replace_complex(h, f'selfenergy_dimensionless/LO/{p}', 0.1 * data[p].swapaxes(-1, -2)) for p in ['SL', 'SG']])
    raw = -1j * (data['GR'] @ (0.9 * data['SL'] + 0.1 * data['SL'].swapaxes(-1, -2)) @ data['GR'].conj().swapaxes(-1, -2))
    r = diagnose(archive)
    assert r['numbers']['raw']['value_dimensionless'] == pytest.approx(number(raw), abs=2e-15)
    assert r['fdr']['raw']['value'] > 0.001


def test_R05_wrong_matrix_axis_labels(archive):
    mutate(archive, lambda h: h['state_dimensionless/GR/real'].attrs.__setitem__('logical_axis_order', 'E,k,b,a'))
    error(archive, 'unsupported_layout')


def test_R06_weights_changed_target_retained(archive):
    mutate(archive, lambda h: h['grids_dimensionless/wE'].__setitem__(Ellipsis, WE * 1.7))
    r = diagnose(archive)
    assert r['numbers']['raw']['absolute_target_residual_dimensionless'] > 0.001
    assert abs(r['mu']['value_relative_eV'] - MU) > 0.001
    assert r['fdr']['raw']['value'] > 0.01


def test_R07_energy_edge_truncated(archive):
    def truncate(h):
        paths = []
        h.visititems(lambda name, obj: paths.append(name) if isinstance(obj, h5py.Dataset) and (obj.shape == (5,) or obj.shape == (5, 2, 2, 2)) else None)
        for name in paths:
            ds = h[name]
            values, attrs = ds[:-1], dict(ds.attrs)
            del h[name]
            h[name] = values
            h[name].attrs.update(attrs)
        h['numerical_inputs/NE'][()] = 4
    mutate(archive, truncate)
    r = diagnose(archive)
    assert r['domain']['energy_nodes'] == 4
    assert r['domain']['energy_relative_eV'][1] == 0.015
    assert r['numbers']['raw']['value_dimensionless'] == pytest.approx(number(-1j * archive[2]['GL'][:-1], WE[:-1]), abs=2e-15)
    assert r['domain']['outside_window'] == 'not_determined'


def test_R08_negative_wE(archive):
    mutate(archive, lambda h: h['grids_dimensionless/wE'].__setitem__(0, -0.02))
    error(archive, 'invalid_data')


def test_R09_negative_wk(archive):
    mutate(archive, lambda h: h['grids_dimensionless/wk'].__setitem__(0, -0.07))
    error(archive, 'invalid_data')


def test_R10_negative_spectral_q(archive):
    mutate(archive, lambda h: h['state_dimensionless/A/real'].__setitem__((0, slice(None)), -archive[2]['A'][0].real))
    r = diagnose(archive)
    assert r['spectral']['minimum_number_weight'] < 0
    assert r['mu']['reason'] == 'negative_spectral_number_weight'
    assert r['mu']['value_relative_eV'] is None and r['fdr']['raw']['value'] is None


def test_R11_nan_GR(archive):
    mutate(archive, lambda h: h['state_dimensionless/GR/real'].__setitem__((0, 0, 0, 0), np.nan))
    error(archive, 'invalid_data')


def test_R12_nan_weight(archive):
    mutate(archive, lambda h: h['grids_dimensionless/wE'].__setitem__(0, np.nan))
    error(archive, 'invalid_data')


def test_R13_missing_enabled_family(archive):
    mutate(archive, lambda h: h.__delitem__('selfenergy_dimensionless/alloy'))
    assert 'inconsistent_scattering_inventory' in error(archive, 'invalid_data')


def test_R14_extra_known_family(archive):
    archive[1]['stationary_assessment']['model_context']['enabled_scattering'].remove('alloy')
    mutate(archive, lambda h: h['numerical_inputs/scattering_alloy'].__setitem__((), 0))
    assert 'inconsistent_scattering_inventory' in error(archive, 'invalid_data')


def test_R15_requested_zero_kernel_absent(archive):
    archive[1]['stationary_assessment']['model_context']['enabled_scattering'].remove('alloy')
    def remove(h):
        del h['selfenergy_dimensionless/alloy']
        for n in FAMILIES[:-1]:
            for p in ['SL', 'SG']:
                replace_complex(h, f'selfenergy_dimensionless/{n}/{p}', 0.125 * archive[2][p])
    mutate(archive, remove)
    r = diagnose(archive)
    assert r['fdr']['raw']['value'] < 2e-11
    assert r['source_context']['enabled_scattering'] == list(FAMILIES[:-1])


def test_R16_unknown_enabled_name(archive):
    archive[1]['stationary_assessment']['model_context']['enabled_scattering'].append('mystery')
    mutate(archive, lambda h: h.copy('selfenergy_dimensionless/LO', 'selfenergy_dimensionless/mystery'))
    error(archive, 'unsupported_context')


def test_R17_duplicate_enabled_name(archive):
    archive[1]['stationary_assessment']['model_context']['enabled_scattering'].append('LO')
    publish(archive[0], archive[1])
    error(archive, 'invalid_data')


def test_R18_missing_model_context(archive):
    del archive[1]['stationary_assessment']['model_context']
    publish(archive[0], archive[1])
    r = diagnose(archive)
    assert r['applicability']['status'] == 'not_measured'
    assert r['fdr']['raw']['value'] is None
    assert r['numbers']['raw']['status'] == 'measured'
    assert r['currents']['plus']['raw_outward_A_per_m2'] is not None


def test_R19_finite_drive(archive):
    mutate(archive, lambda h: (h['inputs/F_bias_V_per_m'].__setitem__((), 1e4), h['inputs/V_period_V'].__setitem__((), 1e-3)))
    r = diagnose(archive)
    assert r['applicability']['status'] == 'not_applicable' and r['fdr']['raw']['value'] is None
    assert r['applicability']['reason'] == 'finite_drive'


def test_R20_unequal_LO_temperature(archive):
    mutate(archive, lambda h: h['inputs/T_LO_K'].__setitem__((), 71))
    r = diagnose(archive)
    assert r['applicability']['status'] == 'not_applicable' and r['fdr']['normalized']['value'] is None


def test_R21_unsupported_ee_closure(archive):
    archive[1]['stationary_assessment']['model_context']['physical_models']['electron_electron']['mode'] = 'closure'
    publish(archive[0], archive[1])
    r = diagnose(archive)
    assert r['applicability']['status'] == 'unsupported_context' and r['mu']['value_relative_eV'] is None


def test_R22_target_below_finite_bracket(archive):
    capacity = number(archive[2]['A'])
    mutate(archive, lambda h: h['inputs/N_dop_2D_per_m2'].__setitem__((), capacity * 2 ** -120 / 1e-16))
    r = diagnose(archive)
    assert r['mu']['reason'] == 'target_outside_finite_bracket'
    assert r['mu']['number_lower'] > r['numbers']['target_dimensionless']
    assert r['mu']['lower_eV'] == pytest.approx(E[0] * 0.1 - 50 * KT)
    assert r['fdr']['raw']['value'] is None


def test_R23_zero_capacity_zero_corrections(archive):
    mutate(archive, lambda h: [replace_complex(h, f'state_dimensionless/{n}', np.zeros_like(archive[2]['A'])) for n in ['A', 'GR', 'GL', 'GG']])
    r = diagnose(archive)
    assert r['mu']['value_relative_eV'] is None and r['fdr']['raw']['value'] is None
    for part in ['occupied', 'empty']:
        assert r['corrections'][part]['value'] == 0
        assert r['corrections'][part]['reason'] == 'zero_scale_zero_error'
        assert r['corrections'][part]['denominator_squared'] == 0


def test_R24_zero_capacity_nonzero_corrections(archive):
    def zero(h):
        for n in ['A', 'GR']:
            replace_complex(h, f'state_dimensionless/{n}', np.zeros_like(archive[2]['A']))
        replace_complex(h, 'state_dimensionless/GL', np.broadcast_to(1j * D, (5, 2, 2, 2)))
        replace_complex(h, 'state_dimensionless/GG', np.broadcast_to(-1j * D, (5, 2, 2, 2)))
    mutate(archive, zero)
    r = diagnose(archive)
    for part in ['occupied', 'empty']:
        assert r['corrections'][part]['value'] is None
        assert r['corrections'][part]['reason'] == 'undefined_zero_denominator'
        assert r['corrections'][part]['numerator_squared'] > 0


def test_R25_corrections_independent_statuses(archive):
    mutate(archive, lambda h: [replace_complex(h, f'selfenergy_dimensionless/{n}/SL', np.zeros_like(archive[2]['SL'])) for n in FAMILIES + BOUNDARIES])
    r = diagnose(archive)
    assert r['corrections']['occupied']['value'] is None
    assert r['corrections']['empty']['value'] < 2e-14
    assert r['corrections']['empty']['denominator_squared'] > 0


def test_R26_mu_iteration_limit_one(archive):
    r = diagnose(archive, mu_max_iterations=1)
    assert r['mu']['reason'] == 'algorithm_limit' and r['mu']['iterations'] == 1
    assert r['mu']['upper_eV'] > r['mu']['lower_eV']
    assert r['mu']['absolute_number_residual'] is not None and r['fdr']['raw']['value'] is None


def test_R27_public_scalar_producer_layout(archive):
    api()
    with StateReader(archive[0] / 'commit.json', expected_identity=IDENTITY, maximum_io_bytes=32 * 1024 * 1024) as reader:
        value = reader.read_scalar('inputs/T_L_K')
        assert value.value == 70 and value.units == 'K' and value.source_identity == IDENTITY
        before = reader.io_counters['logical_selected_bytes']
        with pytest.raises(ContractError) as caught:
            reader.read_scalar('inputs/T_L_K', maximum_bytes=7)
        assert caught.value.code == 'budget_exceeded'
        assert reader.io_counters['logical_selected_bytes'] == before


def test_R28_public_vectors_producer_layout(archive):
    api()
    with StateReader(archive[0] / 'commit.json', expected_identity=IDENTITY) as reader:
        for name, want in [('energy', E), ('k', [0.2, 0.7]), ('state_index', [0, 1]), ('wE', WE), ('wk', WK)]:
            value = reader.read_vector(f'grids_dimensionless/{name}', maximum_bytes=128)
            np.testing.assert_array_equal(value.values, want)
            assert value.units == '1' and value.source_identity == IDENTITY


def test_R29_public_matrix_reader_regression(archive):
    api()
    with StateReader(archive[0] / 'commit.json', expected_identity=IDENTITY) as reader:
        selection = (slice(1, 3), slice(0, 1), slice(None), slice(None))
        block = reader.read('state_dimensionless/GR/real', selection)
        np.testing.assert_array_equal(block.values, archive[2]['GR'][1:3, 0:1].real)
        assert block.axes == ('E', 'k', 'a', 'b') and block.units == '1' and block.source_identity == IDENTITY
        np.testing.assert_array_equal(block.coordinates['E'], E[1:3])
        np.testing.assert_array_equal(block.coordinates['k'], [0.2])
        np.testing.assert_array_equal(block.coordinates['a'], [0, 1])
        np.testing.assert_array_equal(block.weights['E'], WE[1:3])
        np.testing.assert_array_equal(block.weights['k'], WK[:1])
        with pytest.raises(ContractError) as caught:
            reader.read('state_dimensionless/GR/real', selection, maximum_bytes=1)
        assert caught.value.code == 'budget_exceeded'


def test_R30_energy_axis_cap(archive):
    limits = api().EquilibriumReadLimits(maximum_io_bytes=32 * 1024 * 1024, maximum_energy_nodes=4)
    error(archive, 'budget_exceeded', limits=limits)


def test_R31_workspace_plane_cap(archive):
    limits = api().EquilibriumReadLimits(maximum_io_bytes=32 * 1024 * 1024, maximum_workspace_bytes=128)
    error(archive, 'budget_exceeded', limits=limits)


def test_R32_io_below_mandatory_sha(archive):
    limits = api().EquilibriumReadLimits(maximum_io_bytes=archive[1]['artifacts'][0]['bytes'] - 1)
    error(archive, 'budget_exceeded', limits=limits)


def test_R33_io_hdf_phase_after_integrity(archive, monkeypatch):
    api()
    from qcl_negf_results import state
    records = []
    rejected = []
    raw_reads = []
    original_init = state._BudgetStream.__init__
    class DeliveryProbe:
        def __init__(self, raw, budget):
            self.raw, self.budget = raw, budget
        def __getattr__(self, name):
            return getattr(self.raw, name)
        def read(self, size=-1):
            value = self.raw.read(size)
            raw_reads.append((self.budget.phase, len(value)))
            return value
        def readinto(self, target):
            value = self.raw.readinto(target)
            raw_reads.append((self.budget.phase, value))
            return value
    def initialize(stream, raw, budget):
        original_init(stream, DeliveryProbe(raw, budget), budget)
    monkeypatch.setattr(state._BudgetStream, '__init__', initialize)
    original_read, original_readinto = state._BudgetStream.read, state._BudgetStream.readinto
    def record(method, stream, requested, operation):
        before = stream._budget.bytes_read_total
        effective = stream.effective_request(requested)
        raw_marker = len(raw_reads)
        try:
            result = operation()
        except ContractError:
            assert len(raw_reads) == raw_marker, 'denied callback must not reach real raw stream'
            rejected.append((stream._budget.phase, before, effective, stream._budget.bytes_read_total))
            raise
        count = stream._budget.bytes_read_total - before
        records.append((stream._budget.phase, before, effective, count))
        return result
    def read(stream, size=-1):
        return record('read', stream, size, lambda: original_read(stream, size))
    def readinto(stream, target):
        return record('readinto', stream, len(target), lambda: original_readinto(stream, target))
    monkeypatch.setattr(state._BudgetStream, 'read', read)
    monkeypatch.setattr(state._BudgetStream, 'readinto', readinto)
    path = archive[0] / 'commit.json'
    with StateReader(path, expected_identity=IDENTITY, maximum_io_bytes=32 * 1024 * 1024) as reader:
        assert reader.source['native_verified'] and reader.source['receipt_verified']
        assert reader.io_counters['sha_bytes'] == archive[1]['artifacts'][0]['bytes']
        marker = len(records)
        reader.read('state_dimensionless/GR/real', (slice(4, 5), slice(1, 2), slice(None), slice(None)))
        selected = next(row for row in records[marker:] if row[0] == 'hdf' and row[3] > 0)
        cap = selected[1] + selected[2] - 1
    records.clear()
    raw_reads.clear()
    with StateReader(path, expected_identity=IDENTITY, maximum_io_bytes=cap) as reader:
        assert reader.source['native_verified'] and reader.source['receipt_verified']
        assert reader.io_counters['sha_bytes'] == archive[1]['artifacts'][0]['bytes']
        with pytest.raises(ContractError) as caught:
            reader.read('state_dimensionless/GR/real', (slice(4, 5), slice(1, 2), slice(None), slice(None)))
        assert caught.value.code == 'budget_exceeded'
        assert reader.io_counters['bytes_read_total'] == sum(row[3] for row in records)
        assert reader.io_counters['bytes_read_total'] == sum(count for _, count in raw_reads)
        assert reader.io_counters['bytes_read_total'] <= cap
        assert rejected[-1][0] == 'hdf' and rejected[-1][1] == rejected[-1][3]


def test_R34_owner_bytes_corrupt(archive):
    with (archive[0] / 'physics.h5').open('ab') as stream:
        stream.write(b'corrupt')
    error(archive, 'corrupt_result')


def test_R35_expected_attempt_mismatch(archive):
    a = api()
    with pytest.raises(ContractError) as caught:
        a.diagnose_stored_equilibrium(archive[0] / 'commit.json', expected_identity={**IDENTITY, 'attempt': 2}, limits=a.EquilibriumReadLimits(maximum_io_bytes=32 * 1024 * 1024))
    assert caught.value.code == 'corrupt_result'


def test_R36_receipt_commit_hash_mismatch(archive):
    receipt = json.loads((archive[0] / 'receipt.json').read_text())
    receipt['commit_sha256'] = '0' * 64
    (archive[0] / 'receipt.json').write_bytes(json_bytes(receipt))
    error(archive, 'corrupt_result')
