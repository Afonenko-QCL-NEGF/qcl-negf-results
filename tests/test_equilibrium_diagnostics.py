"""Preserved independent equilibrium literals, emitted as canonical native5; no solver or production archive input.

Oracles use a closed two-by-two inverse and literal weighted full-matrix sums.
The arithmetic bounds below are engineering roundoff bounds, not S01 acceptance.
"""
import hashlib
from copy import deepcopy
from contextlib import contextmanager, nullcontext
from unittest.mock import patch
import importlib
import json
import math

import h5py
import numpy as np
import pytest

from qcl_negf_contracts.messages import ContractError
from qcl_negf_results.state import StateReader
from source_ownership import assert_source_ownership

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
    from equilibrium_native5_fixture import publish as native_publish
    return native_publish(root, commit)


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
    assert_source_ownership(state.__file__, contracts.__file__)
    root = tmp_path / 'manufactured'
    root.mkdir()
    data = analytic()
    target = number(-1j * data['GL'])
    with h5py.File(root / 'physics.h5', 'w') as handle:
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
    context = dict(boundary_condition='stationary_field_periodic_embedding',
        basis_contract='orthonormal finite projected basis', enabled_scattering=list(FAMILIES),
        physical_models=dict(lo_population='thermal', electron_electron=dict(mode='none')))
    commit = dict(stationary_assessment=dict(model_context=context, fixture_solver_status='not_run'),_complete_context=deepcopy(context))
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
    root, commit, _ = archive
    a = api()
    limits = kwargs.pop('limits', a.EquilibriumReadLimits(maximum_io_bytes=32 * 1024 * 1024))
    selected=commit['_selected']
    with StateReader(commit['_native_root'], state_ref=selected, expected_identity=selected['identity']) as reader:
        report = a.diagnose_stored_equilibrium(reader, observations_ref=commit['_observations'], limits=limits, **kwargs)
    json.dumps(report, allow_nan=False)
    assert 'scientific_accepted' not in report and 'pass' not in report
    assert report['source']==selected
    assert report['budgets']['sha_bytes']==sum((commit['_native_root']/n).stat().st_size for n in ('full.h5','observations.h5','analysis.h5'))
    return report


def error(archive, code, **kwargs):
    with pytest.raises(ContractError) as caught:
        diagnose(archive, **kwargs)
    assert caught.value.code == code
    return str(caught.value)



@contextmanager
def missing_context_oracle(archive):
    """Unit branch proof; not an admissible missing-context native5 source.

    First exercise the actual schema refusal. A complete literal context then
    permits the unchanged numerical code to exercise _applicability(None),
    preserving the former missing-mu/current-bound oracle without bypassing the
    production metadata guard or inventing defaults for any real input.
    """
    with pytest.raises(ContractError) as caught:diagnose(archive)
    assert caught.value.code=='corrupt_result' and 'physical_models' in str(caught.value)
    root,commit,_=archive
    commit['stationary_assessment']['model_context']=deepcopy(commit['_complete_context'])
    publish(root,commit)
    original=api()._applicability
    with patch.object(api(),'_applicability',side_effect=lambda context,inputs,requested:original(None,inputs,requested)):
        yield
    del commit['stationary_assessment']['model_context']


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
    # Donor neutrality constrains occupied electrons only. The independent
    # complementary number is generally different from that donor target.
    occupied = number(-1j * archive[2]['GL'])
    empty = number(1j * archive[2]['GG'])
    capacity = number(archive[2]['A'])
    assert empty == pytest.approx(capacity - occupied, abs=2e-15)
    assert abs(empty - occupied) > 0.001
    for name in ['raw', 'normalized']:
        assert r['numbers'][name]['absolute_target_residual_dimensionless'] < 2e-15
    for name in ['raw_empty', 'normalized_empty']:
        row = r['numbers'][name]
        assert row['status'] == 'measured'
        assert row['value_dimensionless'] == pytest.approx(empty, abs=2e-15)
        assert row['value_per_m2'] == pytest.approx(empty / 1e-16)
        assert row['absolute_target_residual_dimensionless'] is None
        assert row['relative_target_residual'] is None
        assert row['target_residual_status'] == 'not_applicable'
        assert row['target_residual_reason'] == 'donor_target_applies_to_occupied_number'


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
    error(archive, 'corrupt_result')


def test_R06_weights_changed_target_retained(archive):
    mutate(archive, lambda h: h['grids_dimensionless/wE'].__setitem__(Ellipsis, WE * 1.7))
    r = diagnose(archive)
    assert r['numbers']['raw']['absolute_target_residual_dimensionless'] > 0.001
    assert abs(r['mu']['value_relative_eV'] - MU) > 0.001
    assert r['fdr']['raw']['value'] > 0.01
    # Deliberately changed measure with retained donor target: occupied
    # residual remains meaningful, empty measurement must not acquire it.
    occupied_target = number(-1j * archive[2]['GL'])
    occupied_changed = number(-1j * archive[2]['GL'], WE * 1.7)
    empty_changed = number(1j * archive[2]['GG'], WE * 1.7)
    for name in ['raw', 'normalized']:
        row = r['numbers'][name]
        assert row['absolute_target_residual_dimensionless'] == pytest.approx(abs(occupied_changed - occupied_target), abs=2e-15)
        assert row['relative_target_residual'] == pytest.approx(abs(occupied_changed - occupied_target) / occupied_target, abs=2e-14)
    for name in ['raw_empty', 'normalized_empty']:
        row = r['numbers'][name]
        assert row['value_dimensionless'] == pytest.approx(empty_changed, abs=2e-15)
        assert row['value_per_m2'] == pytest.approx(empty_changed / 1e-16)
        assert row['absolute_target_residual_dimensionless'] is None
        assert row['relative_target_residual'] is None
        assert row['target_residual_status'] == 'not_applicable'
        assert row['target_residual_reason'] == 'donor_target_applies_to_occupied_number'


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
    assert 'actual five-family projection differs' in error(archive, 'corrupt_result')


def test_R14_extra_known_family(archive):
    archive[1]['stationary_assessment']['model_context']['enabled_scattering'].remove('alloy')
    mutate(archive, lambda h: h['numerical_inputs/scattering_alloy'].__setitem__((), 0))
    assert 'actual five-family projection differs' in error(archive, 'corrupt_result')


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
    error(archive, 'corrupt_result')


def test_R17_duplicate_enabled_name(archive):
    archive[1]['stationary_assessment']['model_context']['enabled_scattering'].append('LO')
    publish(archive[0], archive[1])
    error(archive, 'corrupt_result')


def test_R18_missing_model_context(archive):
    del archive[1]['stationary_assessment']['model_context']
    publish(archive[0], archive[1])
    with missing_context_oracle(archive):
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
    from qcl_negf_results.final_observations import FinalObservationsReader
    import struct
    a=api();commit=archive[1];selected=commit['_selected']
    with StateReader(commit['_native_root'],state_ref=selected,expected_identity=selected['identity']) as reader:
        with FinalObservationsReader(reader,artifact_ref=commit['_observations']) as view:
            desc=view.describe();tagged=desc.declaration['context']['inputs']['T_L_K']
            assert tagged==dict(kind='ieee',type='f64',bits=struct.pack('>d',70.).hex())
            assert desc.source['identity']==selected['identity']
        with reader._consumer_operation():
            before=reader.io_counters['logical_selected_bytes']
            with pytest.raises(ContractError) as caught:
                a.diagnose_stored_equilibrium(reader,observations_ref=commit['_observations'],
                    limits=a.EquilibriumReadLimits(maximum_io_bytes=32*1024*1024,maximum_single_read_bytes=7))
            assert caught.value.code=='budget_exceeded'
            assert reader.io_counters['logical_selected_bytes']==before


def test_R28_public_vectors_producer_layout(archive):
    from qcl_negf_results.final_observations import FinalObservationsReader
    commit=archive[1];selected=commit['_selected']
    with StateReader(commit['_native_root'],state_ref=selected,expected_identity=selected['identity']) as reader:
        with FinalObservationsReader(reader,artifact_ref=commit['_observations']) as view:
            for name,want in [('energy_scaled',E),('momentum_scaled',[.2,.7]),('state_index',[0,1]),('wE',WE),('wk',WK)]:
                block=view.read_array(name,(slice(None),),maximum_bytes=128)
                np.testing.assert_array_equal(block.values,want)
                assert block.declaration['units']=='1' and block.source['identity']==selected['identity']
                assert not block.values.flags.writeable


def test_R29_public_matrix_reader_regression(archive):
    from qcl_negf_results.final_observations import FinalObservationsReader
    commit=archive[1];selected=commit['_selected']
    with StateReader(commit['_native_root'],state_ref=selected,expected_identity=selected['identity']) as reader:
        with FinalObservationsReader(reader,artifact_ref=commit['_observations']) as view:
            desc=view.describe();declaration=desc.declaration['state_bindings']['green']['retarded']
            selection=(slice(1,3),slice(0,1),slice(None),slice(None))
            block=reader.read_array(declaration,selection)
            np.testing.assert_array_equal(block.values,archive[2]['GR'][1:3,0:1])
            np.testing.assert_array_equal(block.values.real,archive[2]['GR'][1:3,0:1].real)
            np.testing.assert_array_equal(block.values.imag,archive[2]['GR'][1:3,0:1].imag)
            assert block.declaration['order']=='F' and block.declaration['eltype']=='ComplexF64'
            assert block.source['identity']==selected['identity'] and not block.values.flags.writeable
            for name,selection,want in [('energy_scaled',(slice(1,3),),E[1:3]),('momentum_scaled',(slice(0,1),),[.2]),
                    ('state_index',(slice(None),),[0,1]),('wE',(slice(1,3),),WE[1:3]),('wk',(slice(0,1),),WK[:1])]:
                np.testing.assert_array_equal(view.read_array(name,selection).values,want)
            with reader._consumer_operation():
                before=reader.io_counters['logical_selected_bytes']
                with pytest.raises(ContractError) as caught:reader.read_array(declaration,(slice(1,3),slice(0,1),slice(None),slice(None)),maximum_bytes=1)
                assert caught.value.code=='budget_exceeded'
                assert reader.io_counters['logical_selected_bytes']==before


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
    from qcl_negf_results.final_observations import FinalObservationsReader
    from qcl_negf_results.state import ResultReadLimits
    commit=archive[1];ref=commit['_selected']
    def prepare(reader):
        with FinalObservationsReader(reader,artifact_ref=commit['_observations']) as view:description=view.describe()
        declaration=description.declaration['state_bindings']['green']['retarded']
        reader.describe_array(declaration)
        assert reader.source.integrity['manifest_verified']
        assert reader.io_counters['sha_bytes']==sum((commit['_native_root']/n).stat().st_size for n in ('full.h5','observations.h5','analysis.h5'))
        return declaration
    with StateReader(commit['_native_root'],state_ref=ref,expected_identity=ref['identity']) as reader, reader._consumer_operation():
        declaration=prepare(reader)
        marker=len(records)
        reader.read_array(declaration,(slice(4,5),slice(1,2),slice(None),slice(None)))
        selected=next(row for row in records[marker:] if row[0]=='hdf' and row[3]>0)
        cap=selected[1]+selected[2]-1
    records.clear()
    raw_reads.clear()
    with StateReader(commit['_native_root'],state_ref=ref,expected_identity=ref['identity'],limits=ResultReadLimits(maximum_io_bytes=cap)) as reader:
        records.clear();raw_reads.clear()
        with reader._consumer_operation():
            declaration=prepare(reader)
            with pytest.raises(ContractError) as caught:
                reader.read_array(declaration,(slice(4,5),slice(1,2),slice(None),slice(None)))
        assert caught.value.code == 'budget_exceeded'
        assert reader.io_counters['bytes_read_total'] == sum(row[3] for row in records)
        assert reader.io_counters['bytes_read_total'] == sum(count for _, count in raw_reads)
        assert reader.io_counters['bytes_read_total'] <= cap
        assert rejected[-1][0] == 'hdf' and rejected[-1][1] == rejected[-1][3]


def test_R34_owner_bytes_corrupt(archive):
    with (archive[1]['_native_root'] / 'full.h5').open('ab') as stream:
        stream.write(b'corrupt')
    error(archive, 'corrupt_result')


def test_R35_expected_attempt_mismatch(archive):
    commit=archive[1];selected=commit['_selected'];wrong=dict(selected['identity'],attempt='2')
    with pytest.raises(ContractError) as caught:
        StateReader(commit['_native_root'],state_ref=selected,expected_identity=wrong)
    assert caught.value.code=='identity_mismatch'


def test_R36_receipt_commit_hash_mismatch(archive):
    commit=archive[1];selected=deepcopy(commit['_selected']);selected['manifest']['sha256']='0'*64
    with pytest.raises(ContractError) as caught:
        StateReader(commit['_native_root'],state_ref=selected,expected_identity=selected['identity'])
    assert caught.value.code=='corrupt_result'


# SCI03-M1: manufactured measurement controls, not a solver equilibrium claim.
# This append-only section uses the existing encoder/provenance plumbing only.
# Expected arithmetic below never calls product reducers, FD, or mu-fit helpers.
M1_M = 3 / (8 * math.pi)
M1_EPSILON = 1 / 128
M1_L = 7 * M1_M / 512
M1_R = 13 * M1_M / 65536
M1_ROUNDOFF = 512 * 2 ** -53


def m1_close(actual, expected):
    # Conservative <=512-operation binary64 fixture regression policy at these
    # fixed scales; not a rigorous libm enclosure or a scientific tolerance.
    assert math.isfinite(actual)
    assert abs(actual - expected) <= M1_ROUNDOFF * max(1, abs(expected))


def m1_bound_namespace(report):
    assert 'represented_current_bound' in report, 'missing SCI03-5.3 measured current-bound instrumentation'
    result = report['represented_current_bound']
    assert result['contract'] == 'SCI03-5.3-represented-current-v1'
    assert result['norm'] == 'absolute weighted full complex Frobenius'
    assert result['weight'] == 'gs*wE*wk/(2*pi)'
    assert result['normalization'] == 'none'
    assert result['reference_mu'] == 'mu.value_relative_eV'
    assert result['arithmetic_error_status'] == 'not_measured'
    assert result['arithmetic_error_bound'] is None
    assert result['scope'] == 'represented exact-arithmetic complex trace; no acceptance'
    return result


def m1_row(row, expected, units='1'):
    assert row['status'] == 'measured' and row['reason'] is None
    assert row['units'] == units
    m1_close(row['value'], expected)


def m1_archive(archive, *, quarter_capacity=False, zero_boundary=False, imaginary=False):
    """Literal D §7.1 entries; existing HDF encoder is format plumbing only."""
    identity = np.eye(2, dtype=complex)
    x = np.array([[0, 1], [1, 0]], dtype=complex)
    y = np.array([[0, 1j], [-1j, 0]], dtype=complex)
    plus = np.array([[1/2, 1j/8], [-1j/8, 1/4]])
    minus = np.array([[1/4, (1+1j)/16], [(1-1j)/16, 1/2]])
    if zero_boundary:
        plus, minus = np.zeros((2, 2), complex), np.zeros((2, 2), complex)
    internal = 2 * identity - plus - minus
    state = {name: np.empty((1, 2, 2, 2), complex) for name in ('A', 'GR', 'GL', 'GG')}
    sides = {side: {part: np.empty((1, 2, 2, 2), complex) for part in ('SL', 'SG')}
             for side in ('embedding_plus', 'embedding_minus')}
    for k, epsilon in enumerate((1/128, 1/64)):
        state['A'][0, k] = 2 * identity
        state['GR'][0, k] = -1j * identity
        state['GL'][0, k] = 1j * (identity + epsilon * (2*x+y) + (1j/32 * identity if imaginary else 0))
        state['GG'][0, k] = -1j * (identity + epsilon * (-x+2*y))
        for side, gamma, defect in [('embedding_plus', plus, x), ('embedding_minus', minus, y)]:
            delta = np.zeros((2, 2), complex) if zero_boundary else epsilon * defect
            sides[side]['SL'][0, k] = 1j * (gamma/2 + delta)
            sides[side]['SG'][0, k] = -1j * (gamma/2 - delta)
    def install(handle):
        for name, values in [('energy', np.array([0.])), ('wE', np.array([3/8])),
                             ('k', np.array([.2, .7])), ('wk', np.array([1/4, 3/4]))]:
            path = 'grids_dimensionless/' + name
            attrs = dict(handle[path].attrs)
            del handle[path]
            handle[path] = values
            handle[path].attrs.update(attrs)
        handle['numerical_inputs/NE'][()] = 1
        handle['inputs/E_ref_eV'][()] = 0
        handle['inputs/N_dop_2D_per_m2'][()] = (M1_M if quarter_capacity else 2*M1_M) / 1e-16
        for name, values in state.items():
            del handle['state_dimensionless/' + name]
            matrix(handle, 'state_dimensionless/' + name, values)
        for name, fraction in zip(FAMILIES, (1/8, 1/8, 1/8, 1/8, 1/2)):
            for part, sign in [('SL', 1j), ('SG', -1j)]:
                path = f'selfenergy_dimensionless/{name}/{part}'
                del handle[path]
                matrix(handle, path, np.broadcast_to(sign * fraction * internal/2, (1, 2, 2, 2)))
        for side in BOUNDARIES:
            for part in ('SL', 'SG'):
                values = (sides['embedding_plus'][part] + sides['embedding_minus'][part]
                          if side == 'embedding_total' else sides[side][part])
                path = f'selfenergy_dimensionless/{side}/{part}'
                del handle[path]
                matrix(handle, path, values)
    mutate(archive, install)
    return archive


def m1_expected(report, *, imaginary=False):
    # Independent closed literal finite-t table from D §7.1. One PUBLIC mu is
    # used for both variants and sides. No product norm/FD/bound builder oracle.
    t = 1 / (1 + math.exp(-report['mu']['value_relative_eV'] / KT)) - 1/2
    rho2 = 1/1024 if imaginary else 0
    return t, {
        'raw': (4*M1_R + 8*t*t*M1_M, 4*M1_R + 8*t*t*M1_M, 0),
        'normalized': (10*M1_R + 8*t*t*M1_M + 2*rho2*M1_M,
                       10*M1_R + 8*t*t*M1_M, 20*M1_R + 2*rho2*M1_M)}


def m1_check_literal_report(report, *, imaginary=False):
    bound = m1_bound_namespace(report)
    t, states = m1_expected(report, imaginary=imaginary)
    m1_row(bound['spectral_norm'], math.sqrt(8*M1_M))
    for part, squared in [('occupied', 2*M1_R + (2*M1_M/1024 if imaginary else 0)), ('empty', 18*M1_R)]:
        m1_row(bound['normalization_correction_norms'][part], math.sqrt(squared))
        # Existing ratio and numerator/denominator semantics stay independent.
        old = report['corrections'][part]
        assert old['status'] == 'measured'
        m1_close(old['numerator_squared'], squared)
        m1_close(old['denominator_squared'], 2*M1_M + 4*M1_R)
        m1_close(old['value'], math.sqrt(squared / (2*M1_M + 4*M1_R)))
    for variant, squared in states.items():
        row = bound['state_defects'][variant]
        assert row['status'] == 'measured' and row['reason'] is None and row['units'] == '1'
        for name, value in zip(('occupied_norm', 'empty_norm', 'sum_norm'), squared):
            m1_close(row[name], math.sqrt(value))
        m1_close(report['fdr'][variant]['numerator_squared'], squared[0] + squared[1])
        m1_close(report['fdr'][variant]['denominator_squared'], 8*M1_M)
    # Literal zero cannot be replaced by n+p, and normalized closure is false.
    m1_close(bound['state_defects']['raw']['sum_norm'], 0)
    assert bound['state_defects']['raw']['occupied_norm'] > 0
    assert bound['state_defects']['normalized']['sum_norm'] > 0
    for side, gamma2, d2, signed in [
            ('plus', 11/32, 2*M1_R+t*t*(11/32)*M1_M, -69/32768),
            ('minus', 21/64, 2*M1_R-t*M1_L/4+t*t*(21/64)*M1_M, 17/32768)]:
        gm = (1/2-t)*math.sqrt(gamma2*M1_M)
        gp = (1/2+t)*math.sqrt(gamma2*M1_M)
        d = math.sqrt(d2)
        row = bound['boundary_coefficients'][side]
        assert row['status'] == 'measured' and row['reason'] is None and row['units'] == '1'
        for key, expected in [('one_minus_f_gamma_norm', gm), ('f_gamma_norm', gp), ('sigma_fdr_defect_norm', d)]:
            m1_close(row[key], expected)
        for variant, (n2, p2, c2) in states.items():
            terms = (13*gm*math.sqrt(n2), 13*gp*math.sqrt(p2), 13*d*(math.sqrt(8*M1_M)+math.sqrt(c2)))
            result = bound['bounds'][side][variant]
            m1_row(result, math.fsum(terms), 'A/m^2')
            for key, expected in zip(('occupied_term', 'empty_term', 'selfenergy_term'), terms):
                m1_close(result[key], expected)
            assert result['selfenergy_term'] > 0
            current = report['currents'][side]
            real = 13*M1_M*(7/2048 if variant == 'raw' else signed)
            imag = 13*3*M1_M/256 if imaginary and variant == 'normalized' else 0
            m1_close(current[variant+'_outward_A_per_m2'], real)
            m1_close(current[variant+'_imaginary_A_per_m2'], imag)
            m1_close(current[variant+'_common_z_A_per_m2'], real if side == 'plus' else -real)
            assert math.hypot(real, imag) <= result['value'] + M1_ROUNDOFF
    for key in ('raw', 'normalized'):
        m1_close(report['numbers'][key]['value_dimensionless'], 2*M1_M)
        m1_close(report['numbers'][key+'_empty']['value_dimensionless'], 2*M1_M)
    assert report['schema'] == 'qcl-negf-stored-equilibrium-diagnostics-v1'
    assert len(json.dumps(bound, separators=(',', ':')).encode()) <= 8*1024
    assert len(json.dumps(report, ensure_ascii=False, separators=(',', ':')).encode()) <= 64*1024


@pytest.mark.parametrize('quarter_capacity', [False, True])
def test_M1_01_literal_noncommuting_norms_terms_signs_and_weights(archive, quarter_capacity):
    # Catches missing instrumentation; dropped Im/offdiagonal; missing deltaS;
    # wrong W/gs/radial factor; n+p closure; side reuse; total double counting.
    m1_archive(archive, quarter_capacity=quarter_capacity)
    report = diagnose(archive, mu_algorithm_abs_tol_eV=1e-12)
    mu = report['mu']
    known = -KT*math.log(3) if quarter_capacity else 0
    assert mu['status'] == 'measured' and mu['reason'] is None
    assert mu['lower_eV'] <= known <= mu['upper_eV']
    assert mu['stop_reason'] in ('absolute_mu_width', 'adjacent_binary64_endpoints')
    assert (mu['bracket_width_eV'] <= mu['algorithm_abs_tol_eV'] or
            math.nextafter(mu['lower_eV'], mu['upper_eV']) == mu['upper_eV'])
    assert abs(mu['value_relative_eV']-known) <= mu['uncertainty_eV'] + M1_ROUNDOFF
    m1_check_literal_report(report)
    if quarter_capacity:
        # Centered f=1/2 alone cannot distinguish swapped f and (1-f).
        for row in report['represented_current_bound']['boundary_coefficients'].values():
            assert row['one_minus_f_gamma_norm'] > 2*row['f_gamma_norm']


def test_M1_02_full_complex_current_bound(archive):
    # Catches abs(ReJ)-only validation and discarded imaginary norm components.
    m1_archive(archive, imaginary=True)
    report = diagnose(archive, mu_algorithm_abs_tol_eV=1e-12)
    m1_check_literal_report(report, imaginary=True)
    assert report['currents']['plus']['normalized_imaginary_A_per_m2'] > 0


def test_M1_03_zero_boundary_is_measured_zero_with_defined_mu(archive):
    m1_archive(archive, zero_boundary=True)
    report = diagnose(archive)
    assert report['mu']['status'] == 'measured'
    bound = m1_bound_namespace(report)
    assert bound['state_defects']['normalized']['sum_norm'] > 0
    for side in ('plus', 'minus'):
        row = bound['boundary_coefficients'][side]
        assert row['status'] == 'measured' and row['reason'] is None
        for name in ('one_minus_f_gamma_norm', 'f_gamma_norm', 'sigma_fdr_defect_norm'):
            assert row[name] == 0
        for variant in ('raw', 'normalized'):
            result = bound['bounds'][side][variant]
            m1_row(result, 0, 'A/m^2')
            for key in ('value', 'occupied_term', 'empty_term', 'selfenergy_term'):
                assert result[key] == 0
            assert report['currents'][side][variant+'_outward_A_per_m2'] == 0


@pytest.mark.parametrize('unavailable,status,reason', [
    ('missing_context', 'not_measured', 'missing_model_context'),
    ('finite_drive', 'not_applicable', 'finite_drive'),
    ('unequal_lo', 'not_applicable', 'unequal_LO_temperature'),
    ('ee', 'unsupported_context', 'unsupported_electron_electron_closure'),
    ('empty', 'not_measured', 'empty_represented_spectrum'),
    ('iteration_limit', 'algorithm_limit', 'algorithm_limit')])
def test_M1_04_missing_mu_precedes_zero_boundary(archive, unavailable, status, reason):
    m1_archive(archive, zero_boundary=True)
    options = {}
    if unavailable == 'missing_context':
        del archive[1]['stationary_assessment']['model_context']
        publish(archive[0], archive[1])
    elif unavailable == 'finite_drive':
        mutate(archive, lambda h: h['inputs/F_bias_V_per_m'].__setitem__((), 1))
    elif unavailable == 'unequal_lo':
        mutate(archive, lambda h: h['inputs/T_LO_K'].__setitem__((), 71))
    elif unavailable == 'ee':
        archive[1]['stationary_assessment']['model_context']['physical_models']['electron_electron']['mode'] = 'closure'
        publish(archive[0], archive[1])
    elif unavailable == 'empty':
        mutate(archive, lambda h: replace_complex(h, 'state_dimensionless/A', np.zeros((1, 2, 2, 2), complex)))
    else:
        options['mu_max_iterations'] = 1
    with missing_context_oracle(archive) if unavailable=='missing_context' else nullcontext():
        report = diagnose(archive, **options)
    assert report['mu']['status'] == status and report['mu']['reason'] == reason
    assert report['mu']['value_relative_eV'] is None
    bound = m1_bound_namespace(report)
    assert bound['spectral_norm']['status'] == 'measured'
    for row in bound['normalization_correction_norms'].values():
        assert row['status'] == 'measured'
    groups = [(bound['state_defects'][v], ('occupied_norm', 'empty_norm', 'sum_norm')) for v in ('raw', 'normalized')]
    groups += [(bound['boundary_coefficients'][s], ('one_minus_f_gamma_norm', 'f_gamma_norm', 'sigma_fdr_defect_norm')) for s in ('plus', 'minus')]
    groups += [(bound['bounds'][s][v], ('value', 'occupied_term', 'empty_term', 'selfenergy_term'))
               for s in ('plus', 'minus') for v in ('raw', 'normalized')]
    for row, keys in groups:
        assert row['status'] == status and row['reason'] == reason
        for key in keys:
            assert row[key] is None


def test_M1_05_additive_healthy_compatibility_and_no_extra_matrix_reads(archive, monkeypatch):
    # Existing nonuniform 5E fixture and its helper arithmetic are unchanged.
    reads=[]
    original=StateReader.read_array
    def read(reader,declaration,*args,**kwargs):
        path=declaration['dataset']
        if path.startswith(('/arrays/state_dimensionless/','/arrays/selfenergy_dimensionless/')):reads.append(path)
        return original(reader,declaration,*args,**kwargs)
    monkeypatch.setattr(StateReader,'read_array',read)
    report=diagnose(archive,mu_algorithm_abs_tol_eV=1e-12)
    bound=m1_bound_namespace(report)
    assert bound['spectral_norm']['status']=='measured'
    assert report['fdr']['raw']['value']<2e-10
    assert report['fdr']['normalized']['value']<2e-10
    # ComplexF64 carries both components in one actual public read; two k
    # planes, A in both passes, every other full matrix exactly once.
    expected={f'/arrays/state_dimensionless/{name}':(4 if name=='A' else 2) for name in ('A','GR','GL','GG')}
    expected.update({f'/arrays/selfenergy_dimensionless/{family}/{component}':2
        for family in FAMILIES+BOUNDARIES for component in ('SL','SG')})
    assert set(reads)==set(expected)
    for path,count in expected.items():assert reads.count(path)==count
    assert report['budgets']['logical_two_pass_forecast_bytes']==report['budgets']['logical_selected_bytes']


def test_M1_06_bound_scalar_overflow_rejected(archive):
    # Finite arrays/norms and small signed currents can coexist with an
    # overflowing absolute bound term. Never publish measured Inf or null.
    m1_archive(archive)
    def enlarge(h):
        for name in ('GL', 'GG'):
            for part in ('real', 'imag'):
                ds = h[f'state_dimensionless/{name}/{part}']
                ds[...] = 64*ds[...]
        h['scales/J0_A_per_m2'][()] = 1e308
    mutate(archive, enlarge)
    error(archive, 'invalid_data')


def test_M1_07_bound_report_cap_is_not_truncated(archive):
    m1_archive(archive)
    report = diagnose(archive)
    m1_bound_namespace(report)
    old = {key: value for key, value in report.items() if key != 'represented_current_bound'}
    old_size = len(json.dumps(old, ensure_ascii=False, separators=(',', ':')).encode('utf-8'))
    limits = api().EquilibriumReadLimits(maximum_io_bytes=32*1024*1024, maximum_report_bytes=old_size)
    error(archive, 'budget_exceeded', limits=limits)


def test_M1_08_missing_boundary_dataset_is_not_zero(archive):
    m1_archive(archive, zero_boundary=True)
    mutate(archive, lambda h: h.__delitem__('selfenergy_dimensionless/embedding_plus'))
    assert 'native array dataset missing' in error(archive, 'corrupt_result')


def test_M1_09_literal_f0_table_from_scalar_entries():
    # Independent exact dyadic entry identities underpin the runtime oracle.
    # This checks the manufactured table, not a second copy of product kernels.
    gamma_plus = ((1/2, 1j/8), (-1j/8, 1/4))
    gamma_minus = ((1/4, (1+1j)/16), ((1-1j)/16, 1/2))
    x = ((0, 1), (1, 0))
    y = ((0, 1j), (-1j, 0))
    for gamma, norm_squared in ((gamma_plus, 11/32), (gamma_minus, 21/64)):
        assert math.fsum(z.real*z.real + z.imag*z.imag for row in gamma for z in row) == norm_squared
    # Both commutators are nonzero at a literal diagonal entry; no matrix
    # multiplication or product arithmetic helper enters these expectations.
    assert gamma_plus[0][1]*x[1][0] - x[0][1]*gamma_plus[1][0] == 1j/4
    assert gamma_minus[0][1]*y[1][0] - y[0][1]*gamma_minus[1][0] == -1j/8
    assert math.fsum((1/4, 3/4)) == 1
    m1_close(math.fsum((1/4*(1/128), 3/4*(1/64)))*M1_M, M1_L)
    m1_close(math.fsum((1/4*(1/128)**2, 3/4*(1/64)**2))*M1_M, M1_R)
    # Full complex offdiagonal n,p,c norms at exactly f0=1/2.
    for nxy, pxy, literal in [((1, 1), (-1, -1), (4, 4, 0)),
                               ((2, 1), (-1, 2), (10, 10, 20))]:
        assert 2*(nxy[0]**2+nxy[1]**2) == literal[0]
        assert 2*(pxy[0]**2+pxy[1]**2) == literal[1]
        assert 2*((nxy[0]+pxy[0])**2+(nxy[1]+pxy[1])**2) == literal[2]
    m1_close(-M1_L/8-2*M1_R, -69*M1_M/32768)
    m1_close(M1_L/8-6*M1_R, 17*M1_M/32768)


def test_arithmetic_opt_in_off_hash_and_caps(archive):
    """Manufactured native-v4 integration; legacy M1 remains not_measured."""
    import struct
    m1_archive(archive, zero_boundary=True)
    a = api()
    plain = diagnose(archive)
    assert 'represented_current_arithmetic_enclosure' not in plain
    cap = a.EquilibriumArithmeticLimits(16, 4, 20_000_000, 65536, 8*1024*1024, 180.)
    report = diagnose(archive, arithmetic_limits=cap)
    enclosure = report['represented_current_arithmetic_enclosure']
    assert enclosure['status'] == 'enclosed'
    assert report['represented_current_bound']['arithmetic_error_status'] == 'not_measured'
    # Hash identifies the existing represented f values, not true FD accuracy.
    f = a._fd(np.array([0.])*report['scales']['E0_eV'], report['mu']['value_relative_eV'], KT)
    assert enclosure['f_count'] == 1
    assert enclosure['f_binary64_sha256'] == hashlib.sha256(b''.join(struct.pack('<d', float(x)) for x in f)).hexdigest()
    for rows in enclosure['rows'].values():
        for row in rows.values():
            assert row['current_real'] == {'lower': 0., 'upper': 0.}
            assert row['rhs'] == {'lower': 0., 'upper': 0.}
    rejected = a.EquilibriumArithmeticLimits(1, 1, 1, 64, 1024, 1.)
    result = diagnose(archive, arithmetic_limits=rejected)['represented_current_arithmetic_enclosure']
    assert result['status'] == 'budget_exceeded'
    assert all(row['rhs'] is None for rows in result['rows'].values() for row in rows.values())


def test_arithmetic_public_minimum_subnormal_not_false_zero(archive):
    m1_archive(archive, zero_boundary=True)
    t = math.ulp(0.)
    z = np.zeros((1, 2, 2, 2), complex)
    spectral = z.copy()
    spectral[0, 1] = 2*np.eye(2)  # Distinct spectral cell makes μ defined.
    gr, gl = z.copy(), z.copy()
    gr[0, 0] = np.eye(2)
    gl[0, 0, 0, 0] = 1j*t
    plus_sg, internal_sl, internal_sg = z.copy(), z.copy(), z.copy()
    plus_sg[0, 0, 0, 0] = -1j*t
    internal_sl[0, 0, 0, 0] = 1j*t
    internal_sg[0, 0, 0, 0] = 1j*t  # Cancels embedding_total SG exactly.
    def install(h):
        for name, values in [('A', spectral), ('GR', gr), ('GL', gl), ('GG', z)]:
            replace_complex(h, 'state_dimensionless/'+name, values)
        for family in FAMILIES+BOUNDARIES:
            for part in ('SL', 'SG'):
                values = z
                if family == 'LO':
                    values = internal_sl if part == 'SL' else internal_sg
                elif family in ('embedding_total', 'embedding_plus') and part == 'SG':
                    values = plus_sg
                replace_complex(h, f'selfenergy_dimensionless/{family}/{part}', values)
    mutate(archive, install)
    cap = api().EquilibriumArithmeticLimits(16, 4, 20_000_000, 65536, 8*1024*1024, 180.)
    report = diagnose(archive, arithmetic_limits=cap)
    assert report['mu']['value_relative_eV'] is not None
    result = report['represented_current_arithmetic_enclosure']
    assert result['status'] == 'enclosed'
    # Exact target J0*(2/(2π))*(3/8)*(1/4)*2^-2148 is strictly positive.
    for variant in ('raw', 'normalized'):
        row = result['rows']['plus'][variant]
        assert row['current_real']['upper'] > 0 and row['rhs']['upper'] > 0


# New native5 I/O controls: independent literal operands remain above unchanged.
def test_B2_asymmetric_complex_E_k_a_b_binding_NB1(archive):
    data=archive[2];gr=data['GR'].copy();gr[:,1,0,1]+=.07+.13j;gr[::2,0,1,0]-=.02j
    mutate(archive,lambda h:replace_complex(h,'state_dimensionless/GR',gr))
    expected_n=-1j*(gr@data['SL']@gr.conj().swapaxes(-1,-2))
    expected_p=1j*(gr@data['SG']@gr.conj().swapaxes(-1,-2))
    report=diagnose(archive,mu_algorithm_abs_tol_eV=1e-12)
    assert report['numbers']['raw']['value_dimensionless']==pytest.approx(number(expected_n),abs=3e-15)
    assert report['numbers']['raw_empty']['value_dimensionless']==pytest.approx(number(expected_p),abs=3e-15)
    expected=norm2(expected_n+1j*data['GL'])+norm2(expected_p-1j*data['GG'])
    assert report['fdr']['raw']['numerator_squared']==pytest.approx(expected,rel=2e-10)
    assert abs(number(expected_n)-number(expected_n[:,::-1]))>1e-6


def test_B2_actual_nonlexicographic_family_order_and_EE_cap(archive,monkeypatch):
    commit=archive[1];order=['alloy','electron_electron','IFR','impurity','LO','acoustic']
    commit['_family_order']=order
    commit['stationary_assessment']['model_context']['enabled_scattering']=[n for n in order if n!='electron_electron']
    commit['stationary_assessment']['model_context']['physical_models']['electron_electron']['mode']='closure'
    mutate(archive,lambda h:h.copy('selfenergy_dimensionless/LO','selfenergy_dimensionless/electron_electron'))
    assert 'channel cap' in error(archive,'budget_exceeded')
    reads=[];original=StateReader.read_array
    def read(reader,declaration,*args,**kwargs):
        path=declaration['dataset']
        if path.endswith('/SL') and '/selfenergy_dimensionless/' in path:reads.append(path.split('/')[-2])
        return original(reader,declaration,*args,**kwargs)
    monkeypatch.setattr(StateReader,'read_array',read)
    report=diagnose(archive,limits=api().EquilibriumReadLimits(maximum_io_bytes=32*1024*1024,maximum_channels=6))
    assert report['source_context']['enabled_scattering']==order
    assert reads==(order+['embedding_total','embedding_plus','embedding_minus'])*2
    expected=-1j*(archive[2]['GR']@(1.1*archive[2]['SL'])@archive[2]['GR'].conj().swapaxes(-1,-2))
    assert report['numbers']['raw']['value_dimensionless']==pytest.approx(number(expected),abs=3e-15)
    assert report['applicability']==dict(status='unsupported_context',reason='unsupported_electron_electron_closure')
    assert report['mu']['value_relative_eV'] is None


@pytest.mark.parametrize('kind', ['single','chunk'])
def test_B2_complex_plane_and_chunk_admission_before_numeric_read(archive,kind):
    commit=archive[1];ref=commit['_selected'];a=api()
    options={'maximum_single_read_bytes':127} if kind=='single' else {'maximum_chunk_uncompressed_bytes':63}
    with StateReader(commit['_native_root'],state_ref=ref,expected_identity=ref['identity']) as reader:
        previous=reader.limits
        with pytest.raises(ContractError) as caught:a.diagnose_stored_equilibrium(reader,observations_ref=commit['_observations'],
            limits=a.EquilibriumReadLimits(maximum_io_bytes=32*1024*1024,**options))
        assert caught.value.code=='budget_exceeded'
        assert reader.io_counters['logical_selected_bytes']==0
        assert reader.limits is previous and reader._shared_budget is None and not reader._closed


def test_B2_contiguous_storage_streams_under_tiny_chunk_cap(archive):
    archive[1]['_chunks']=None;publish(archive[0],archive[1])
    report=diagnose(archive,limits=api().EquilibriumReadLimits(maximum_io_bytes=32*1024*1024,
        maximum_chunk_uncompressed_bytes=1,maximum_single_read_bytes=128))
    assert report['budgets']['energy_tile_nodes']==1
    assert report['numbers']['raw']['value_dimensionless']==pytest.approx(number(-1j*archive[2]['GL']),abs=2e-15)
    assert report['budgets']['logical_two_pass_forecast_bytes']==report['budgets']['logical_selected_bytes']


def test_B2_Int64_spin_tag_cannot_exceed_actual_domain(archive):
    from consumer_context_fixture import change
    c=archive[1]
    change(c['_native_root'],c['_selected'],c['_observations'],lambda v:v['context']['inputs']['g_s'].update(decimal=str(2**63)))
    # Published exact Int64 schema rejects before numeric payload selection.
    error(archive,'corrupt_result')


@pytest.mark.parametrize('mode', ['cancel','after_last_read','during_read'])
def test_B2_operation_source_guard_cleanup_and_cancellation(archive,monkeypatch,mode):
    c=archive[1];ref=c['_selected'];a=api();fired=[]
    def mutate_owner():
        path=c['_native_root']/'full.h5';data=bytearray(path.read_bytes());data[-1]^=1;path.write_bytes(data);fired.append(True)
    if mode=='after_last_read':
        original=a._represented_current_bound
        def bound(*args,**kwargs):
            result=original(*args,**kwargs);mutate_owner();return result
        monkeypatch.setattr(a,'_represented_current_bound',bound)
    else:
        original=h5py.Dataset.__getitem__
        def read(dataset,key):
            if dataset.name=='/arrays/state_dimensionless/GR':
                if mode=='cancel':fired.append(True);raise KeyboardInterrupt('literal cancellation')
                value=original(dataset,key);mutate_owner();return value
            return original(dataset,key)
        monkeypatch.setattr(h5py.Dataset,'__getitem__',read)
    with StateReader(c['_native_root'],state_ref=ref,expected_identity=ref['identity']) as reader:
        previous=reader.limits
        if mode=='cancel':
            with pytest.raises(KeyboardInterrupt,match='literal cancellation'):
                a.diagnose_stored_equilibrium(reader,observations_ref=c['_observations'],limits=a.EquilibriumReadLimits(maximum_io_bytes=32*1024*1024))
        else:
            with pytest.raises(ContractError) as caught:
                a.diagnose_stored_equilibrium(reader,observations_ref=c['_observations'],limits=a.EquilibriumReadLimits(maximum_io_bytes=32*1024*1024))
            assert caught.value.code=='source_changed'
        assert fired and reader.limits is previous and reader._shared_budget is None and not reader._closed


def test_B2_parent_metadata_and_numeric_reads_share_one_owner_allowance(tmp_path,monkeypatch):
    from test_native_provenance import _completed_fixture
    from qcl_negf_results import state
    ref,observations,_,_,_,_=_completed_fixture(tmp_path)
    with StateReader(tmp_path,state_ref=ref,expected_identity=ref['identity']) as reader:
        deliveries=[];budgets=set();account=state._IOBudget.account
        def counted(budget,count):
            account(budget,count);budgets.add(id(budget));deliveries.append(count)
        monkeypatch.setattr(state._IOBudget,'account',counted)
        result=api().diagnose_stored_equilibrium(reader,observations_ref=observations,
            limits=api().EquilibriumReadLimits(maximum_io_bytes=32*1024*1024))
        assert len(budgets)==1 and result['budgets']['bytes_read_total']==sum(deliveries)
        assert result['source_evidence']['restoration']['parent_state']['identity']['attempt']=='1'
        assert result['source_evidence']['original']['fields']['original_status']['value']=='invalid'
        assert result['source_evidence']['current']['scientific_accepted'] is None
        required=result['budgets']['bytes_read_total']
    with StateReader(tmp_path,state_ref=ref,expected_identity=ref['identity']) as reader:
        with pytest.raises(ContractError) as caught:api().diagnose_stored_equilibrium(reader,observations_ref=observations,
            limits=api().EquilibriumReadLimits(maximum_io_bytes=required-1))
        assert caught.value.code=='budget_exceeded'
        assert reader.io_counters['bytes_read_total']<=required-1


@pytest.mark.parametrize('cap', [1,8])
@pytest.mark.parametrize('target,fault', [('observations','replace'),('diagnostics','unlink'),
    ('checkpoint','modify'),('manifest','replace'),('plan','unlink'),('model','modify'),
    ('parent_manifest','replace'),('parent_checkpoint','unlink')])
def test_B2_F1_all_consumed_bindings_survive_owner_eviction(tmp_path,monkeypatch,cap,target,fault):
    from qcl_negf_results.state import ResultReadLimits
    if target.startswith('parent_'):
        from test_native_provenance import _completed_fixture
        root=tmp_path/'completed';selected,observations,value,_,_,parent=_completed_fixture(root)
        parent_manifest=json.loads((root/parent['manifest']['path']).read_bytes())
        paths={'parent_manifest':parent['manifest']['path'],'parent_checkpoint':parent_manifest['checkpoint']['path']}
    else:
        fixture=archive.__wrapped__(tmp_path);c=fixture[1]
        root,selected,observations,value=c['_native_root'],c['_selected'],c['_observations'],c['_value']
        paths={'observations':observations['path'],'diagnostics':value['diagnostics']['path'],
            'checkpoint':value['checkpoint']['path'],'manifest':selected['manifest']['path'],
            'plan':value['request']['plan']['path'],'model':value['request']['model']['path']}
    path=root/paths[target];events=[];original=api()._represented_current_bound
    def bound(*args,**kwargs):
        result=original(*args,**kwargs)
        before=path.stat();payload=path.read_bytes()
        if fault=='replace':path.rename(path.with_name(path.name+'.retained'));path.write_bytes(payload)
        elif fault=='unlink':path.unlink()
        else:
            changed=bytearray(payload);changed[-1]^=1;path.write_bytes(changed)
        events.append((before.st_ino,None if not path.exists() else path.stat().st_ino))
        return result
    monkeypatch.setattr(api(),'_represented_current_bound',bound)
    with StateReader(root,state_ref=selected,expected_identity=selected['identity'],limits=ResultReadLimits(maximum_open_files=cap)) as reader:
        # Exercise the actual cached checkpoint/ancestor proof, not a cold-only fix.
        reader.checkpoint();previous=reader.limits
        with pytest.raises(ContractError) as caught:
            api().diagnose_stored_equilibrium(reader,observations_ref=observations,
                limits=api().EquilibriumReadLimits(maximum_io_bytes=32*1024*1024))
        assert caught.value.code=='source_changed'
        assert events and len(reader._handles)<=cap
        assert reader.limits is previous and reader._shared_budget is None and not reader._closed
        assert getattr(reader._budget,'_source_guard',None) is None


@pytest.mark.parametrize('cap', [1,8])
def test_B2_F1_healthy_coldwarm_and_cancellation_keep_one_owner(tmp_path,monkeypatch,cap):
    from qcl_negf_results.state import ResultReadLimits
    fixture=archive.__wrapped__(tmp_path);c=fixture[1];selected=c['_selected'];a=api()
    with StateReader(c['_native_root'],state_ref=selected,expected_identity=selected['identity'],limits=ResultReadLimits(maximum_open_files=cap)) as reader:
        previous=reader.limits
        def call():return a.diagnose_stored_equilibrium(reader,observations_ref=c['_observations'],limits=a.EquilibriumReadLimits(maximum_io_bytes=32*1024*1024))
        cold=call();warm=call()
        for key in ('source','source_evidence','mu','fdr','numbers','currents','represented_current_bound','corrections','spectral'):
            assert cold[key]==warm[key]
        assert cold['numbers']['raw']['value_dimensionless']==pytest.approx(number(-1j*fixture[2]['GL']),abs=2e-15)
        assert len(reader._handles)<=cap and reader.limits is previous and reader._shared_budget is None
        original=a._represented_current_bound
        def cancel(*args,**kwargs):original(*args,**kwargs);raise KeyboardInterrupt('F1 final cancellation')
        with monkeypatch.context() as patcher:
            patcher.setattr(a,'_represented_current_bound',cancel)
            with pytest.raises(KeyboardInterrupt,match='F1 final cancellation'):call()
        assert reader.limits is previous and reader._shared_budget is None and not reader._closed
        assert getattr(reader._budget,'_source_guard',None) is None
        assert call()['numbers']==cold['numbers']


@pytest.mark.parametrize('cap', [1,8])
def test_B2_F1_binding_workspace_and_cache_metadata_are_charged_before_numeric_read(tmp_path,monkeypatch,cap):
    from qcl_negf_results.state import ResultReadLimits, _metadata_weight
    fixture=archive.__wrapped__(tmp_path);c=fixture[1];selected=c['_selected'];a=api();samples=[]
    with StateReader(c['_native_root'],state_ref=selected,expected_identity=selected['identity'],limits=ResultReadLimits(maximum_open_files=cap)) as reader:
        previous=reader.limits;original=a._represented_current_bound
        def inspect_guard(*args,**kwargs):
            guard=reader._budget._source_guard;book=reader._metadata_bindings
            assert _metadata_weight(guard)<=guard['weight']
            assert _metadata_weight(book)<=book['weight']
            samples.append(reader._source_stability_bytes())
            return original(*args,**kwargs)
        monkeypatch.setattr(a,'_represented_current_bound',inspect_guard)
        result=a.diagnose_stored_equilibrium(reader,observations_ref=c['_observations'],limits=a.EquilibriumReadLimits(
            maximum_io_bytes=32*1024*1024,energy_tile_nodes=1))
        assert samples and 0<samples[0]<result['budgets']['workspace_forecast_bytes']
        # Admit more than the old numeric-only minimum, but less than the
        # complete plane plus its trace. A margin avoids incidental Python int
        # aliasing changes when the limit itself changes the guard header.
        with pytest.raises(ContractError) as caught:
            a.diagnose_stored_equilibrium(reader,observations_ref=c['_observations'],limits=a.EquilibriumReadLimits(
                maximum_io_bytes=32*1024*1024,energy_tile_nodes=1,
                maximum_workspace_bytes=result['budgets']['workspace_forecast_bytes']-samples[0]//2))
        assert caught.value.code=='budget_exceeded' and reader.io_counters['logical_selected_bytes']==0
        assert reader.limits is previous and reader._shared_budget is None and reader._budget._source_guard is None
        assert not reader._closed and len(reader._handles)<=cap


from result_v2_fixture import producer_root as producer_root


@pytest.mark.parametrize('cap', [1,8])
@pytest.mark.parametrize('dependency', ['ancestor_record','ancestor_catalog'])
def test_B2_F1_cached_actual_parent_witness_keeps_original_read_binding(tmp_path,producer_root,cap,dependency):
    from qcl_negf_results.state import ResultReadLimits
    from result_v2_fixture import producer_state, copy_constructor_closure
    selected=producer_state(producer_root,'child');root=tmp_path/'copied-closure'
    copy_constructor_closure(producer_root,root,selected)
    with StateReader(root,state_ref=selected,expected_identity=selected['identity'],limits=ResultReadLimits(maximum_open_files=cap)) as reader:
        checkpoint=reader.checkpoint();assert checkpoint['witness_targets']
        if dependency=='ancestor_record':path=root/checkpoint['witness_targets'][0]['locator']['leaf']['path']
        else:
            manifest=json.loads((root/selected['manifest']['path']).read_bytes())
            parent=manifest['lineage']['parent_state'];parent_manifest=json.loads((root/parent['manifest']['path']).read_bytes())
            path=root/parent_manifest['history']['head']['path']
        previous=reader.limits;before=path.stat();payload=path.read_bytes()
        with pytest.raises(ContractError) as caught:
            with reader._consumer_operation(32*1024*1024,verify_source_stability=True,maximum_stability_bytes=32*1024*1024):
                assert reader.checkpoint()==checkpoint
                assert reader.io_counters['bytes_read_total']==0, 'cached proof must not trigger prefix/hash replay'
                assert reader._source_stability_bytes(seal=True)>0
                path.rename(path.with_name(path.name+'.original'));path.write_bytes(payload)
                assert path.stat().st_ino!=before.st_ino
        assert caught.value.code=='source_changed'
        assert reader.limits is previous and reader._shared_budget is None and reader._budget._source_guard is None
        assert not reader._closed and len(reader._handles)<=cap


@pytest.mark.parametrize('cap', [1,8])
@pytest.mark.parametrize('target', ['plan','parent_manifest'])
def test_B2_F1_actual_FD_binding_precedes_postread_path_replacement(tmp_path,monkeypatch,cap,target):
    import os
    from qcl_negf_results import state
    from qcl_negf_results.state import ResultReadLimits
    if target=='parent_manifest':
        from test_native_provenance import _completed_fixture
        root=tmp_path/'completed';selected,observations,value,_,_,parent=_completed_fixture(root)
        path=root/parent['manifest']['path']
    else:
        fixture=archive.__wrapped__(tmp_path);c=fixture[1]
        root,selected,observations,value=c['_native_root'],c['_selected'],c['_observations'],c['_value']
        path=root/value['request']['plan']['path']
    original=state._BudgetStream.read;inode=path.stat().st_ino;events=[]
    def read(stream,size=-1):
        bound=os.fstat(stream._raw.fileno()).st_ino
        payload=original(stream,size)
        if bound==inode and not events:
            path.rename(path.with_name(path.name+'.original'));path.write_bytes(payload)
            events.append((bound,path.stat().st_ino))
        return payload
    with StateReader(root,state_ref=selected,expected_identity=selected['identity'],limits=ResultReadLimits(maximum_open_files=cap)) as reader:
        previous=reader.limits;monkeypatch.setattr(state._BudgetStream,'read',read)
        with pytest.raises(ContractError) as caught:
            api().diagnose_stored_equilibrium(reader,observations_ref=observations,limits=api().EquilibriumReadLimits(maximum_io_bytes=32*1024*1024))
        assert caught.value.code=='source_changed' and events and events[0][0]!=events[0][1]
        assert reader.io_counters['logical_selected_bytes']==0
        assert reader.limits is previous and reader._shared_budget is None and reader._budget._source_guard is None
        assert not reader._closed
