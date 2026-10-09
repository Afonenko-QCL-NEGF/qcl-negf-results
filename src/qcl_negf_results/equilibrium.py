"""Independent diagnostics of a represented, immutable stored final state.

Stored self-energies reconstruct the raw Keldysh correlations. They are not a
fresh candidate SCBA map. This API measures numbers, matrix FDR, normalization
corrections and outward electron-flow currents; it never decides acceptance.
No solver, Julia import, inferred model defaults or scientific tolerance.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from qcl_negf_contracts.messages import ContractError
from .state import StateReader

_SCATTERING = ('LO', 'acoustic', 'impurity', 'IFR', 'alloy')
_EMBEDDING = ('embedding_total', 'embedding_plus', 'embedding_minus')
_COORDINATES = ('/grids_dimensionless/energy', '/grids_dimensionless/k',
                '/grids_dimensionless/state_index', '/grids_dimensionless/state_index')
_KB_EV = 1.380649e-23 / 1.602176634e-19  # exact SI definitions


@dataclass(frozen=True)
class EquilibriumReadLimits:
    """Engineering bounds; numerical workspace is not total process RSS."""
    maximum_io_bytes: int
    maximum_workspace_bytes: int = 32 * 1024 * 1024
    maximum_energy_nodes: int = 65536
    maximum_momentum_nodes: int = 128
    maximum_basis_states: int = 16
    maximum_channels: int = 5
    maximum_single_read_bytes: int = 1024 * 1024
    maximum_chunk_uncompressed_bytes: int = 1024 * 1024
    energy_tile_nodes: int = 128
    maximum_report_bytes: int = 64 * 1024

    def __post_init__(self) -> None:
        for field in fields(self):
            value = getattr(self, field.name)
            if type(value) is not int or value < 1:
                raise ValueError(f'{field.name} must be a positive integer (not bool)')


def _error(reason: str, code: str = 'invalid_data') -> None:
    raise ContractError(reason, code)


def _finite(value: Any, locator: str) -> Any:
    if not np.all(np.isfinite(value)):
        _error(f'nonfinite numeric input/intermediate: {locator}')
    return value


class _Sum:
    """Neumaier compensated signed sum, without an unbounded term list."""
    def __init__(self) -> None:
        self.total = 0.0
        self.correction = 0.0

    def add(self, value: float) -> None:
        value = float(value)
        combined = self.total + value
        if abs(self.total) >= abs(value):
            self.correction += (self.total - combined) + value
        else:
            self.correction += (value - combined) + self.total
        self.total = combined
        if not math.isfinite(self.total) or not math.isfinite(self.correction):
            _error('nonfinite compensated sum')

    def value(self) -> float:
        value = self.total + self.correction
        if not math.isfinite(value):
            _error('nonfinite compensated result')
        return value


class _Norm:
    """Weighted full-matrix LASSQ: scale retained separately from squares."""
    def __init__(self) -> None:
        self.scale = 0.0
        self.ss = _Sum()

    def add(self, values: np.ndarray, weights: np.ndarray) -> None:
        _finite(values, 'weighted Frobenius input')
        for value, weight in zip(values, weights):
            if weight == 0:
                continue
            factor = math.sqrt(float(weight))
            for component in (value.real, value.imag):
                for item in component.ravel():
                    amplitude = abs(float(item)) * factor
                    if not math.isfinite(amplitude):
                        _error('nonfinite weighted Frobenius component')
                    if amplitude == 0:
                        continue
                    if amplitude > self.scale:
                        previous = self.ss.value() * (self.scale / amplitude) ** 2
                        self.scale = amplitude
                        self.ss = _Sum()
                        self.ss.add(previous)
                        self.ss.add(1.0)
                    else:
                        self.ss.add((amplitude / self.scale) ** 2)

    def squared(self) -> float:
        value = self.scale * (self.scale * self.ss.value())
        if not math.isfinite(value):
            _error('nonfinite squared Frobenius result')
        return value

    def magnitude(self) -> float:
        value = self.scale * math.sqrt(self.ss.value())
        if not math.isfinite(value):
            _error('nonfinite Frobenius magnitude')
        return value


def _ratio(numerator: _Norm, denominator: _Norm) -> dict[str, Any]:
    top, bottom = numerator.magnitude(), denominator.magnitude()
    result = dict(units='1', numerator_squared=numerator.squared(),
                  denominator_squared=denominator.squared())
    if bottom > 0:
        value = top / bottom
        if not math.isfinite(value):
            _error('nonfinite Frobenius ratio')
        return dict(result, status='measured', reason=None, value=value)
    if top == 0:
        return dict(result, status='measured', reason='zero_scale_zero_error', value=0.0)
    return dict(result, status='undefined_ratio', reason='undefined_zero_denominator', value=None)


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    return value


def _layout(reader: StateReader, path: str, limits: EquilibriumReadLimits) -> Any:
    desc = reader.describe(path)
    if desc.object_kind != 'dataset' or desc.dtype.hasobject or desc.dtype.kind not in 'biuf':
        _error(f'fixed-width real numeric storage required: {path}', 'unsupported_layout')
    if desc.filters and any(filter_id not in (1, 2, 3) for filter_id in desc.filters):
        _error(f'unsupported HDF5 filter: {path}', 'unsupported_layout')
    if desc.chunks is not None and math.prod(desc.chunks) * desc.itemsize > limits.maximum_chunk_uncompressed_bytes:
        _error(f'uncompressed HDF5 chunk exceeds byte cap: {path}', 'budget_exceeded')
    return desc


def _scalar(reader: StateReader, path: str, units: str, limits: EquilibriumReadLimits,
            *, integer: bool = False, flag: bool = False) -> float | int | bool:
    desc = _layout(reader, path, limits)
    if desc.shape != () or desc.units != units:
        _error(f'scalar rank/units mismatch: {path}', 'unsupported_layout')
    if integer and desc.dtype.kind not in 'iu':
        _error(f'exact integer storage required: {path}')
    value = reader.read_scalar(path, maximum_bytes=min(64, limits.maximum_single_read_bytes)).value
    _finite(value, path)
    if flag:
        if type(value) not in (bool, int) or value not in (0, 1):
            _error(f'boolean/0-or-1 scalar required: {path}')
        return bool(value)
    if integer:
        return int(value)
    if type(value) is bool:
        _error(f'physical scalar cannot be boolean: {path}')
    return float(value)


def _applicability(context: Any, inputs: Mapping[str, Any], requested: Mapping[str, bool]) -> dict[str, Any]:
    def result(status: str, reason: str | None) -> dict[str, Any]:
        return dict(status=status, reason=reason)
    if not isinstance(context, Mapping):
        return result('not_measured', 'missing_model_context')
    if context.get('boundary_condition') != 'stationary_field_periodic_embedding':
        return result('unsupported_context', 'unsupported_or_missing_boundary_condition')
    if context.get('basis_contract') != 'orthonormal finite projected basis':
        return result('unsupported_context', 'unsupported_or_missing_basis_contract')
    models = context.get('physical_models')
    if not isinstance(models, Mapping):
        return result('not_measured', 'missing_physical_models')
    ee = models.get('electron_electron')
    if not isinstance(ee, Mapping) or 'mode' not in ee:
        return result('not_measured', 'missing_electron_electron_mode')
    if ee['mode'] != 'none':
        return result('unsupported_context', 'unsupported_electron_electron_closure')
    if inputs['F_bias_V_per_m'] != 0 or inputs['V_period_V'] != 0:
        return result('not_applicable', 'finite_drive')
    if requested['LO']:
        if 'lo_population' not in models:
            return result('not_measured', 'missing_lo_population')
        if models['lo_population'] != 'thermal':
            return result('not_applicable', 'nonthermal_LO')
        if inputs['T_LO_K'] != inputs['T_L_K']:
            return result('not_applicable', 'unequal_LO_temperature')
    return result('applicable', None)


def _fd(energies: np.ndarray, mu: float, kt: float) -> np.ndarray:
    # Masked branches avoid evaluating the overflowing alternative expression.
    x = (energies - mu) / kt
    if np.any(np.isnan(x)):
        _error('invalid Fermi argument')
    result = np.empty_like(x)
    nonnegative = x >= 0
    exp_minus = np.exp(-x[nonnegative])
    result[nonnegative] = exp_minus / (1 + exp_minus)
    exp_plus = np.exp(x[~nonnegative])
    result[~nonnegative] = 1 / (1 + exp_plus)
    return result


def _number(q: np.ndarray, energies: np.ndarray, mu: float, kt: float) -> float:
    result = _Sum()
    for weight, f in zip(q, _fd(energies, mu, kt)):
        result.add(float(weight) * float(f))
    return result.value()


def _fit_mu(q: np.ndarray, energies: np.ndarray, target: float, kt: float, e0: float,
            eref: float, applicable: Mapping[str, Any], tolerance: float | None,
            maximum_iterations: int) -> dict[str, Any]:
    report: dict[str, Any] = dict(status='not_measured', reason=applicable['reason'],
        value_relative_eV=None, value_absolute_eV=None, units='eV',
        lower_eV=None, upper_eV=None, bracket_width_eV=None, uncertainty_eV=None,
        number_lower=None, number_upper=None, iterations=0, absolute_number_residual=None,
        algorithm_abs_tol_eV=None, stop_reason=None)
    if applicable['status'] != 'applicable':
        report['status'] = applicable['status']
        return report
    capacity = math.fsum(float(value) for value in q)
    if np.any(q < 0):
        report['reason'] = 'negative_spectral_number_weight'
        return report
    if capacity <= 0:
        report['reason'] = 'empty_represented_spectrum'
        return report
    if target <= 0 or target >= capacity:
        report['reason'] = 'target_outside_represented_capacity'
        return report
    lower, upper = float(energies[0]) - 50 * kt, float(energies[-1]) + 50 * kt
    if not math.isfinite(lower) or not math.isfinite(upper):
        _error('nonfinite chemical-potential bracket')
    tol = tolerance if tolerance is not None else 8 * np.finfo(np.float64).eps * max(abs(float(energies[0])), abs(float(energies[-1])), e0, kt)
    nlower, nupper = _number(q, energies, lower, kt), _number(q, energies, upper, kt)
    report.update(lower_eV=lower, upper_eV=upper, number_lower=nlower, number_upper=nupper,
                  algorithm_abs_tol_eV=float(tol))
    if not nlower <= target <= nupper:
        report.update(reason='target_outside_finite_bracket', bracket_width_eV=upper-lower,
                      uncertainty_eV=(upper-lower)/2, stop_reason='not_bracketed')
        return report
    stop = None
    for iteration in range(maximum_iterations + 1):
        width = upper - lower
        if width <= tol:
            stop = 'absolute_mu_width'
            break
        if math.nextafter(lower, upper) == upper:
            stop = 'adjacent_binary64_endpoints'
            break
        if iteration == maximum_iterations:
            break
        middle = lower + width / 2
        value = _number(q, energies, middle, kt)
        if value < target:
            lower, nlower = middle, value
        else:
            upper, nupper = middle, value
    middle = lower + (upper - lower) / 2
    report.update(lower_eV=lower, upper_eV=upper, bracket_width_eV=upper-lower,
        uncertainty_eV=(upper-lower)/2, number_lower=nlower, number_upper=nupper,
        iterations=iteration, absolute_number_residual=abs(_number(q, energies, middle, kt)-target))
    if stop is None:
        report.update(status='algorithm_limit', reason='algorithm_limit', stop_reason='iteration_limit')
    else:
        report.update(status='measured', reason=None, stop_reason=stop,
                      value_relative_eV=middle, value_absolute_eV=middle+eref)
    return report


def diagnose_stored_equilibrium(commit_path: str | Path, *, expected_identity: Mapping[str, Any],
        limits: EquilibriumReadLimits, mu_algorithm_abs_tol_eV: float | None = None,
        mu_max_iterations: int = 128) -> dict[str, Any]:
    """Two bounded passes over stored full matrices; no acceptance inference.

    Raw correlations use exactly the stored dictionary plus embedding_total.
    Each boundary flux has an outward sign (electron flow, not charge current).
    The finite common-mu fit is a construction, not an equilibrium verdict.
    """
    if not isinstance(limits, EquilibriumReadLimits):
        raise TypeError('limits must be EquilibriumReadLimits')
    if not isinstance(expected_identity, Mapping) or not expected_identity:
        raise ValueError('explicit nonempty expected_identity is required')
    if type(mu_max_iterations) is not int or not 1 <= mu_max_iterations <= 128:
        raise ValueError('mu_max_iterations must be an integer from 1 to 128')
    if mu_algorithm_abs_tol_eV is not None and (type(mu_algorithm_abs_tol_eV) not in (int, float)
            or not math.isfinite(mu_algorithm_abs_tol_eV) or mu_algorithm_abs_tol_eV <= 0):
        raise ValueError('mu_algorithm_abs_tol_eV must be finite and positive')
    with StateReader(commit_path, expected_identity=expected_identity, maximum_io_bytes=limits.maximum_io_bytes) as reader:
        source = reader.source
        if (source['storage_class'] != 'archive' or source['owner']['schema'] != 'qcl-negf-physics-v4'
                or not source['receipt_verified'] or source['state_id'] is None):
            _error('requires immutable final archive physics.full and local receipt', 'unsupported_context')
        dimensions = {name: _scalar(reader, 'numerical_inputs/' + name, '1', limits, integer=True) for name in ('NE', 'Nk', 'Nb')}
        ne, nk, nb = (dimensions[name] for name in ('NE', 'Nk', 'Nb'))
        for size, maximum in ((ne, limits.maximum_energy_nodes), (nk, limits.maximum_momentum_nodes), (nb, limits.maximum_basis_states)):
            if size < 1:
                _error('nonpositive represented dimension')
            if size > maximum:
                _error('represented dimension exceeds engineering cap', 'budget_exceeded')
        scalar_units = {'E0_eV': 'eV', 'L0_m': 'm', 'J0_A_per_m2': 'A/m^2'}
        scales = {name: _scalar(reader, 'scales/' + name, units, limits) for name, units in scalar_units.items()}
        if any(value <= 0 for value in scales.values()):
            _error('scales must be finite and positive')
        input_units = {'F_bias_V_per_m': 'V/m', 'V_period_V': 'V', 'T_L_K': 'K', 'T_LO_K': 'K',
            'E_ref_eV': 'eV', 'N_dop_2D_per_m2': 'm^-2', 'f_ion': '1'}
        inputs = {name: _scalar(reader, 'inputs/' + name, units, limits) for name, units in input_units.items()}
        gs = _scalar(reader, 'inputs/spin_degeneracy', '1', limits, integer=True)
        if gs < 1 or inputs['T_L_K'] <= 0:
            _error('positive spin degeneracy and lattice temperature required')
        requested = {name: _scalar(reader, 'numerical_inputs/scattering_' + name, '1', limits, flag=True) for name in _SCATTERING}
        assessment = reader.commit.get('stationary_assessment')
        context = assessment.get('model_context') if isinstance(assessment, Mapping) else None
        inventory = reader.describe('selfenergy_dimensionless', maximum_children=16)
        if inventory.object_kind != 'group':
            _error('selfenergy inventory must be a group', 'unsupported_layout')
        stored = set(inventory.child_names)
        if stored - set(_SCATTERING + _EMBEDDING):
            _error('unknown stored selfenergy family', 'unsupported_context')
        if not set(_EMBEDDING) <= stored:
            _error('incomplete embedding inventory')
        enabled = sorted(stored - set(_EMBEDDING))
        if isinstance(context, Mapping):
            explicit = context.get('enabled_scattering')
            if explicit is None:
                context = None  # missing context remains unavailable, never default-inferred
            else:
                if not isinstance(explicit, list) or any(not isinstance(name, str) for name in explicit):
                    _error('invalid enabled scattering list')
                if len(set(explicit)) != len(explicit):
                    _error('duplicate enabled scattering name')
                if set(explicit) - set(_SCATTERING):
                    _error('unknown enabled scattering family', 'unsupported_context')
                if set(enabled) != set(explicit) or any(not requested[name] for name in explicit):
                    _error('inconsistent_scattering_inventory')
                enabled = list(explicit)
        if len(enabled) > limits.maximum_channels:
            _error('channel cap exceeded', 'budget_exceeded')
        # Even without model context, a stored known family cannot contradict a
        # false requested flag. Its actual dictionary is used for raw numbers.
        if any(not requested[name] for name in enabled):
            _error('inconsistent_scattering_inventory')
        applicability = _applicability(context, inputs, requested)
        vector_sizes = {'energy': ne, 'wE': ne, 'k': nk, 'wk': nk, 'state_index': nb}
        descriptions = {}
        for name, count in vector_sizes.items():
            path = 'grids_dimensionless/' + name
            desc = _layout(reader, path, limits)
            if desc.shape != (count,) or desc.units != '1':
                _error(f'grid/weight rank or units mismatch: {path}', 'unsupported_layout')
            if name == 'state_index' and desc.dtype.kind not in 'iu':
                _error('state_index must have integer storage', 'unsupported_layout')
            if desc.itemsize != 8:
                _error('grid/weight requires binary64 or int64 storage', 'unsupported_layout')
            descriptions[path] = desc
        complex_paths = ['state_dimensionless/' + name for name in ('A', 'GR', 'GL', 'GG')]
        complex_paths += [f'selfenergy_dimensionless/{name}/{part}' for name in (*enabled, *_EMBEDDING) for part in ('SL', 'SG')]
        for base in complex_paths:
            for component in ('real', 'imag'):
                path = base + '/' + component
                desc = _layout(reader, path, limits)
                if (desc.shape != (ne, nk, nb, nb) or desc.dtype.kind != 'f' or desc.itemsize != 8
                        or desc.axes != ('E', 'k', 'a', 'b') or desc.units != '1' or desc.coordinate_paths != _COORDINATES):
                    _error(f'unsupported matrix shape/dtype/axes/units/coordinate linkage: {path}', 'unsupported_layout')
                descriptions[path] = desc
        # Conservative live-array/scratch forecast. Adaptive tiles retain a
        # complete basis plane; no diagonal-only fallback is permitted.
        vector_scratch = 4096 + 32 * nk + 16 * nb + 512 * len(descriptions)
        fixed_workspace = 10 * 8 * ne + vector_scratch
        per_energy_workspace = 24 * 16 * nb * nb
        if fixed_workspace + per_energy_workspace > limits.maximum_workspace_bytes:
            _error('complete matrix plane and scratch exceed workspace cap', 'budget_exceeded')
        available_tile = (limits.maximum_workspace_bytes-fixed_workspace) // per_energy_workspace
        single_fixed = 8 * (2 + 2 * nb)
        available_single = (limits.maximum_single_read_bytes-single_fixed) // (8 * (nb * nb+2))
        tile_nodes = min(ne, limits.energy_tile_nodes, available_tile, available_single)
        if tile_nodes < 1:
            _error('complete matrix plane and coordinates exceed single-read cap', 'budget_exceeded')
        workspace_forecast = fixed_workspace + tile_nodes * per_energy_workspace
        # Logical forecast includes duplicate coordinates/weights for each real
        # and imaginary read, A twice, every other required matrix once.
        matrix_components = 2 * (5 + 2 * len(enabled) + 6)
        logical_forecast = reader.io_counters['logical_selected_bytes'] + sum(count * 8 for count in vector_sizes.values())
        for k in range(nk):
            for start in range(0, ne, tile_nodes):
                count = min(tile_nodes, ne-start)
                logical_forecast += matrix_components * 8 * (count * nb * nb + 2 * count + 2 + 2 * nb)
        # Logical bytes are a separate forecast; only the physical callback
        # guard decides actual total I/O admission (including compression).
        vectors = {name: reader.read_vector('grids_dimensionless/' + name,
            maximum_bytes=limits.maximum_single_read_bytes).values for name in vector_sizes}
        for name, value in vectors.items():
            _finite(value, name)
        energy, we, momentum, wk, states = (vectors[name] for name in ('energy', 'wE', 'k', 'wk', 'state_index'))
        if np.any(energy[1:] <= energy[:-1]) or np.any(momentum < 0) or np.any(momentum[1:] <= momentum[:-1]):
            _error('energy/k coordinates must be strictly increasing; k nonnegative')
        if not np.array_equal(states, np.arange(nb)):
            _error('state_index must be zero-based complete basis coordinates', 'unsupported_layout')
        if np.any(we < 0) or np.any(wk < 0) or not np.any(we > 0) or not np.any(wk > 0):
            _error('quadrature weights require nonnegative nonzero measure')
        eev = _finite(energy * scales['E0_eV'], 'physical energy')
        kt = _KB_EV * inputs['T_L_K']
        l02 = scales['L0_m'] ** 2
        if not math.isfinite(l02) or l02 <= 0 or not math.isfinite(kt) or kt <= 0:
            _error('invalid squared length or kBT')
        target = inputs['N_dop_2D_per_m2'] * inputs['f_ion'] * l02
        _finite(target, 'ionized target')
        prefactor = gs / (2 * math.pi)

        def blocks():
            for k in range(nk):
                for start in range(0, ne, tile_nodes):
                    stop = min(start + tile_nodes, ne)
                    selection = (slice(start, stop), slice(k, k+1), slice(None), slice(None))
                    weight = _finite(prefactor * we[start:stop] * wk[k], 'quadrature product')
                    yield start, stop, selection, weight

        def read_complex(base: str, selection: tuple[slice, ...]) -> np.ndarray:
            real = reader.read(base + '/real', selection, maximum_bytes=limits.maximum_single_read_bytes).values
            imag = reader.read(base + '/imag', selection, maximum_bytes=limits.maximum_single_read_bytes).values
            values = real[:, 0] + 1j * imag[:, 0]
            return _finite(values, base)

        # Pass one retains just the energy number weights (compensated over k).
        qtotal = np.zeros(ne, dtype=np.float64)
        qcorrection = np.zeros(ne, dtype=np.float64)
        spectral_imag = _Sum()
        for start, stop, selection, weight in blocks():
            a = read_complex('state_dimensionless/A', selection)
            trace = np.trace(a, axis1=-2, axis2=-1)
            for offset, (value, w) in enumerate(zip(trace, weight)):
                index = start + offset
                term = float(w) * float(value.real)
                previous = float(qtotal[index])
                combined = previous + term
                qcorrection[index] += ((previous-combined)+term if abs(previous) >= abs(term)
                                       else (term-combined)+previous)
                qtotal[index] = combined
                spectral_imag.add(float(w) * float(value.imag))
        q = _finite(qtotal + qcorrection, 'spectral number weight')
        capacity = math.fsum(float(value) for value in q)
        mu = _fit_mu(q, eev, target, kt, scales['E0_eV'], inputs['E_ref_eV'], applicability,
                     mu_algorithm_abs_tol_eV, mu_max_iterations)
        # Release per-node accumulator objects before matrix pass two.
        del qtotal, qcorrection
        fitted = mu['value_relative_eV']
        fermi = None if fitted is None else _fd(eev, fitted, kt)
        norms = {name: _Norm() for name in ('A', 'raw_fdr', 'norm_fdr', 'corr_n', 'corr_p', 'raw_n', 'raw_p')}
        numbers = {name: _Sum() for name in ('raw', 'raw_empty', 'normalized', 'normalized_empty')}
        number_imag = {name: _Sum() for name in numbers}
        flux = {side: {kind: (_Sum(), _Sum()) for kind in ('raw', 'normalized')} for side in ('plus', 'minus')}
        for start, stop, selection, weight in blocks():
            a, gr, gl, gg = (read_complex('state_dimensionless/' + name, selection) for name in ('A', 'GR', 'GL', 'GG'))
            sl_total, sg_total = np.zeros_like(gr), np.zeros_like(gr)
            for family in (*enabled, 'embedding_total'):
                sl_total += read_complex(f'selfenergy_dimensionless/{family}/SL', selection)
                sg_total += read_complex(f'selfenergy_dimensionless/{family}/SG', selection)
            ga = gr.conj().swapaxes(-1, -2)
            gn = _finite(-1j * (gr @ sl_total @ ga), 'raw occupied Keldysh')
            gp = _finite(1j * (gr @ sg_total @ ga), 'raw empty Keldysh')
            normalized_n, normalized_p = -1j * gl, 1j * gg
            norms['A'].add(a, weight)
            norms['raw_n'].add(gn, weight)
            norms['raw_p'].add(gp, weight)
            norms['corr_n'].add(normalized_n-gn, weight)
            norms['corr_p'].add(normalized_p-gp, weight)
            for name, values in [('raw', gn), ('raw_empty', gp), ('normalized', normalized_n), ('normalized_empty', normalized_p)]:
                traces = np.trace(values, axis1=-2, axis2=-1)
                for w, value in zip(weight, traces):
                    numbers[name].add(float(w) * float(value.real))
                    number_imag[name].add(float(w) * float(value.imag))
            if fermi is not None:
                f = fermi[start:stop, None, None]
                for accumulator, occupied, empty in [('raw_fdr', gn, gp), ('norm_fdr', normalized_n, normalized_p)]:
                    norms[accumulator].add(occupied-f*a, weight)
                    norms[accumulator].add(empty-(1-f)*a, weight)
            for side, family in [('plus', 'embedding_plus'), ('minus', 'embedding_minus')]:
                sl = read_complex(f'selfenergy_dimensionless/{family}/SL', selection)
                sg = read_complex(f'selfenergy_dimensionless/{family}/SG', selection)
                for kind, lesser, greater in [('normalized', gl, gg), ('raw', 1j*gn, -1j*gp)]:
                    trace = _finite(np.trace(sg @ lesser - sl @ greater, axis1=-2, axis2=-1), 'boundary flux')
                    for w, value in zip(weight, trace):
                        flux[side][kind][0].add(float(w) * float(value.real))
                        flux[side][kind][1].add(float(w) * float(value.imag))
        number_report = dict(target_dimensionless=target, target_per_m2=target/l02)
        for name, accumulator in numbers.items():
            value = accumulator.value()
            difference = abs(value-target)
            number_report[name] = dict(status='measured', value_dimensionless=value, value_per_m2=value/l02,
                units='m^-2', absolute_target_residual_dimensionless=difference,
                relative_target_residual=difference/target if target > 0 else None,
                imaginary_trace_dimensionless=number_imag[name].value(), issues=['negative_charge'] if value < 0 else [])
        currents = {}
        for side in ('plus', 'minus'):
            current = dict(units='A/m^2', convention='outward electron flow')
            for kind in ('raw', 'normalized'):
                value = scales['J0_A_per_m2'] * flux[side][kind][0].value()
                current[kind + '_outward_A_per_m2'] = value
                current[kind + '_common_z_A_per_m2'] = value if side == 'plus' else -value
                current[kind + '_imaginary_A_per_m2'] = scales['J0_A_per_m2'] * flux[side][kind][1].value()
            currents[side] = current
        for kind in ('raw', 'normalized'):
            plus = currents['plus'][kind+'_outward_A_per_m2']
            minus = currents['minus'][kind+'_outward_A_per_m2']
            currents[kind+'_outward_balance_A_per_m2'] = plus+minus
            currents[kind+'_absolute_max_A_per_m2'] = max(abs(plus), abs(minus))
        if fermi is None:
            fdr = {name: dict(status=mu['status'], reason=mu['reason'], value=None, units='1',
                             numerator_squared=None, denominator_squared=norms['A'].squared()) for name in ('raw', 'normalized')}
        else:
            fdr = dict(raw=_ratio(norms['raw_fdr'], norms['A']), normalized=_ratio(norms['norm_fdr'], norms['A']))
        budgets = dict(reader.io_counters)
        budgets.update(workspace_forecast_bytes=workspace_forecast, logical_two_pass_forecast_bytes=logical_forecast,
                       energy_tile_nodes=tile_nodes, workspace_scope='numeric arrays and scratch; not total process RSS')
        report = dict(schema='qcl-negf-stored-equilibrium-diagnostics-v1', source=_plain(reader.source),
            source_context=_plain(context), recorded_solver_status={name: _plain(assessment[name]) for name in (
                'iterative_converged', 'fixed_hartree_converged', 'outer_confirmation_count', 'fixture_solver_status'
                ) if isinstance(assessment, Mapping) and name in assessment},
            applicability=applicability, mu=mu, fdr=fdr, numbers=number_report, currents=currents,
            corrections=dict(occupied=_ratio(norms['corr_n'], norms['raw_n']), empty=_ratio(norms['corr_p'], norms['raw_p'])),
            spectral=dict(represented_capacity_dimensionless=capacity, represented_capacity_per_m2=capacity/l02,
                          minimum_number_weight=float(np.min(q)), imaginary_trace_dimensionless=spectral_imag.value()),
            domain=dict(energy_nodes=ne, momentum_nodes=nk, basis_states=nb,
                energy_relative_eV=[float(eev[0]), float(eev[-1])], momentum_dimensionless=[float(momentum[0]), float(momentum[-1])],
                outside_window='not_determined', basis_contract='finite represented basis'),
            scales=scales, budgets=budgets,
            interpretation='stored-state measurements; no fresh SCBA, convergence, discretization or experimental acceptance')
        try:
            encoded = json.dumps(report, ensure_ascii=False, allow_nan=False, separators=(',', ':')).encode('utf-8')
        except (ValueError, OverflowError) as error:
            raise ContractError('nonfinite report intermediate/result', 'invalid_data') from error
        if len(encoded) > limits.maximum_report_bytes:
            _error('diagnostic report exceeds byte budget', 'budget_exceeded')
        return report
