"""Independent diagnostics of a represented, immutable stored final state.

Stored self-energies reconstruct the raw Keldysh correlations. They are not a
fresh candidate SCBA map. This API measures numbers, matrix FDR, normalization
corrections and outward electron-flow currents; it never decides acceptance.
No solver, Julia import, inferred model defaults or scientific tolerance.
"""
from __future__ import annotations

from dataclasses import dataclass, fields, replace
from contextlib import contextmanager
import json
import math
import struct
from typing import Any, Mapping

import numpy as np

from qcl_negf_contracts.messages import ContractError
from .state import StateReader, Document
from .final_observations import FinalObservationsReader
from ._equilibrium_arithmetic import (
    EquilibriumArithmeticLimits, _Enclosure, _Budget, _Invalid, _report as _arithmetic_report,
)

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


def _represented_current_bound(norms: Mapping[str, _Norm],
        state_norms: Mapping[str, Mapping[str, _Norm]],
        side_norms: Mapping[str, Mapping[str, _Norm]],
        mu: Mapping[str, Any], j0: float) -> dict[str, Any]:
    """SCI03-5.3 exact-arithmetic represented trace RHS, not acceptance.

    Norms use the existing common fitted mu and complete stored matrices.
    The Float64 evaluation is not an outward-rounded arithmetic enclosure.
    """
    defined = mu['value_relative_eV'] is not None
    status, reason = ('measured', None) if defined else (mu['status'], mu['reason'])
    spectral = norms['A'].magnitude()
    corrections = {name: dict(status='measured', reason=None, units='1', value=norms[key].magnitude())
                   for name, key in (('occupied', 'corr_n'), ('empty', 'corr_p'))}
    states = {variant: dict(status=status, reason=reason, units='1',
        **{key: value.magnitude() if defined else None for key, value in values.items()})
        for variant, values in state_norms.items()}
    sides = {side: dict(status=status, reason=reason, units='1',
        **{key: value.magnitude() if defined else None for key, value in values.items()})
        for side, values in side_norms.items()}
    bounds = {}
    for side, coefficients in sides.items():
        bounds[side] = {}
        for variant, defects in states.items():
            row = dict(status=status, reason=reason, units='A/m^2', value=None,
                       occupied_term=None, empty_term=None, selfenergy_term=None)
            # Inherit missing-mu status before any literal-zero shortcut.
            if defined:
                occupied = _finite(coefficients['one_minus_f_gamma_norm'] * defects['occupied_norm'],
                                   'represented current occupied product')
                empty = _finite(coefficients['f_gamma_norm'] * defects['empty_norm'],
                                'represented current empty product')
                spectral_sum = _finite(spectral + defects['sum_norm'], 'represented current spectral/defect sum')
                selfenergy = _finite(coefficients['sigma_fdr_defect_norm'] * spectral_sum,
                                     'represented current selfenergy product')
                total = _Sum()
                for key, term in (('occupied_term', occupied), ('empty_term', empty),
                                  ('selfenergy_term', selfenergy)):
                    value = _finite(j0 * term, 'represented current SI term')
                    row[key] = value
                    total.add(value)
                row['value'] = total.value()
            bounds[side][variant] = row
    return dict(contract='SCI03-5.3-represented-current-v1',
        norm='absolute weighted full complex Frobenius', weight='gs*wE*wk/(2*pi)',
        normalization='none', reference_mu='mu.value_relative_eV',
        spectral_norm=dict(status='measured', reason=None, units='1', value=spectral),
        normalization_correction_norms=corrections, state_defects=states,
        boundary_coefficients=sides, bounds=bounds,
        arithmetic_error_status='not_measured', arithmetic_error_bound=None,
        scope='represented exact-arithmetic complex trace; no acceptance')


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    return value


def _tag_number(value, locator, *, integer=False):
    """Decode only the mandatory public O1 numeric grammar, without defaults."""
    if integer:
        if not isinstance(value,Mapping) or value.get('kind')!='integer' or value.get('type')!='Int64':
            _error('exact Int64 required: '+locator)
        spelling=value['decimal']
        if not isinstance(spelling,str) or len(spelling)>20:
            _error('exact Int64 outside actual domain: '+locator)
        result=int(spelling)
        if not -2**63<=result<2**63:
            _error('exact Int64 outside actual domain: '+locator)
        return result
    if not isinstance(value,Mapping) or value.get('kind')!='ieee' or value.get('type')!='f64':
        _error('exact Float64 tag required: '+locator)
    return float(_finite(struct.unpack('>d',bytes.fromhex(value['bits']))[0],locator))


def _chunk_layout(dataset, limits, locator):
    # Geometry comes from the already authenticated same-owner dataset. Values
    # are still read exclusively through the existing public raw reader APIs.
    if dataset.chunks is not None and math.prod(dataset.chunks)*dataset.dtype.itemsize > limits.maximum_chunk_uncompressed_bytes:
        _error('uncompressed HDF5 chunk exceeds byte cap: '+locator, 'budget_exceeded')


@contextmanager
def _stored_source(reader, observations_ref, limits):
    previous=reader.limits
    with reader._consumer_operation(limits.maximum_io_bytes, verify_source_stability=True,
                                    maximum_stability_bytes=limits.maximum_workspace_bytes):
        try:
            reader.limits=replace(previous,maximum_array_bytes=min(previous.maximum_array_bytes,limits.maximum_single_read_bytes))
            with FinalObservationsReader(reader,artifact_ref=observations_ref) as view:
                yield view,view.describe()
        finally:
            reader.limits=previous


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


def diagnose_stored_equilibrium(reader: StateReader, *, observations_ref: Document,
        limits: EquilibriumReadLimits, mu_algorithm_abs_tol_eV: float | None = None,
        mu_max_iterations: int = 128,
        arithmetic_limits: EquilibriumArithmeticLimits | None = None) -> dict[str, Any]:
    """Two bounded passes over exact O1-bound native5 full matrices.

    Every actual CP family plus embedding.total is used exactly once; stored
    observables or a requested flag never replace the independent arithmetic.
    """
    if not isinstance(reader,StateReader):
        raise TypeError('explicit selected StateReader required')
    if not isinstance(limits, EquilibriumReadLimits):
        raise TypeError('limits must be EquilibriumReadLimits')
    if type(mu_max_iterations) is not int or not 1 <= mu_max_iterations <= 128:
        raise ValueError('mu_max_iterations must be an integer from 1 to 128')
    if mu_algorithm_abs_tol_eV is not None and (type(mu_algorithm_abs_tol_eV) not in (int, float)
            or not math.isfinite(mu_algorithm_abs_tol_eV) or mu_algorithm_abs_tol_eV <= 0):
        raise ValueError('mu_algorithm_abs_tol_eV must be finite and positive')
    with _stored_source(reader,observations_ref,limits) as (observations,description):
        value=description.declaration;context_raw=value['context']
        dimensions={name:int(context_raw['dimensions'][name]) for name in ('NE','Nk','Nb')}
        ne,nk,nb=(dimensions[name] for name in ('NE','Nk','Nb'))
        for size,maximum in ((ne,limits.maximum_energy_nodes),(nk,limits.maximum_momentum_nodes),(nb,limits.maximum_basis_states)):
            if size<1:_error('nonpositive represented dimension')
            if size>maximum:_error('represented dimension exceeds engineering cap','budget_exceeded')
        scales={name:_tag_number(context_raw['scales'][name],name) for name in ('E0_eV','L0_m','J0_A_per_m2')}
        if any(n<=0 for n in scales.values()):_error('scales must be finite and positive')
        inputs={name:_tag_number(context_raw['inputs'][name],name) for name in
            ('F_bias_V_per_m','V_period_V','T_L_K','T_LO_K','E_ref_eV','N_dop_2D_per_m2','f_ion')}
        gs=_tag_number(context_raw['inputs']['g_s'],'g_s',integer=True)
        if gs<1 or inputs['T_L_K']<=0:_error('positive spin degeneracy and lattice temperature required')
        requested=dict(value['request']['scattering_requested'])
        models=context_raw['physical_models']['fields']
        context=dict(boundary_condition=context_raw['boundary_condition'],
            basis_contract='orthonormal finite projected basis' if context_raw['basis_contract']=='orthonormal_finite_projected' else context_raw['basis_contract'],
            physical_models=dict(lo_population=models['lo_population']['value'],
                electron_electron=dict(mode=models['electron_electron']['fields']['mode']['value'])),
            enabled_scattering=[row['family'] for row in value['state_bindings']['scattering']])
        enabled=context['enabled_scattering']
        if len(enabled)>limits.maximum_channels:_error('channel cap exceeded','budget_exceeded')
        applicability=_applicability(context,inputs,requested)
        assessment=value['summary']
        stored_evidence=dict(original=_plain(description.original_evidence),current=_plain(value['summary']),restoration=_plain(value['restoration']))
        vector_sizes={'energy':ne,'wE':ne,'k':nk,'wk':nk,'state_index':nb}
        vector_names=dict(energy='energy_scaled',wE='wE',k='momentum_scaled',wk='wk',state_index='state_index')
        descriptions={}
        for name,count in vector_sizes.items():
            stored=vector_names[name];desc=observations.describe_array(stored)
            if desc.logical_shape!=(count,) or desc.eltype!=('Int64' if name=='state_index' else 'Float64'):
                _error('grid/weight shape or type differs: '+stored,'unsupported_layout')
            _,dataset=observations._array(stored)
            _chunk_layout(dataset,limits,stored)
            descriptions[stored]=desc
        bindings=value['state_bindings']
        matrices={'state_dimensionless/'+name:bindings['green'][field] for name,field in
            [('A','spectral'),('GR','retarded'),('GL','lesser'),('GG','greater')]}
        for row in bindings['scattering']:
            for part,field in [('SL','lesser'),('SG','greater')]:matrices['selfenergy_dimensionless/'+row['family']+'/'+part]=row[field]
        for side,family in [('total','embedding_total'),('plus','embedding_plus'),('minus','embedding_minus')]:
            for part,field in [('SL','lesser'),('SG','greater')]:matrices['selfenergy_dimensionless/'+family+'/'+part]=bindings['embedding'][side][field]
        for name,declaration in matrices.items():
            desc=reader.describe_array(declaration)
            if desc.logical_shape!=(ne,nk,nb,nb) or desc.eltype!='ComplexF64' or desc.element_bytes!=16:
                _error('complete ComplexF64[E,k,a,b] required: '+name,'unsupported_layout')
            _,dataset=reader._array(declaration)
            _chunk_layout(dataset,limits,name)
            descriptions[name]=desc
        # Conservative live-array/scratch forecast. Adaptive tiles retain a
        # complete basis plane; no diagonal-only fallback is permitted.
        vector_scratch = 4096 + 32 * nk + 16 * nb + 512 * len(descriptions)
        # Add fixed scalar-accumulator/report reserve (4 KiB + 8 KiB). This
        # accounting covers numeric scratch, not Python/HDF caches or total RSS.
        bound_scratch = 4096 + 8 * 1024
        fixed_workspace = 10 * 8 * ne + vector_scratch + bound_scratch
        fixed_workspace += reader._source_stability_bytes(seal=True)
        # Eight extra complex tile slots cover at most three live state-defect
        # buffers, or four side-coefficient buffers, plus O(tile) f/1-f and
        # finite-check scratch. State defects are released before side reads;
        # side coefficients before existing flux matrix products. No new GEMM.
        per_energy_workspace = 32 * 16 * nb * nb
        if fixed_workspace + per_energy_workspace > limits.maximum_workspace_bytes:
            _error('complete matrix plane and scratch exceed workspace cap', 'budget_exceeded')
        available_tile = (limits.maximum_workspace_bytes-fixed_workspace) // per_energy_workspace
        single_fixed = 8 * (2 + 2 * nb)
        available_single = (min(limits.maximum_single_read_bytes,reader.limits.maximum_array_bytes)-single_fixed) // (16 * (nb * nb+1))
        tile_nodes = min(ne, limits.energy_tile_nodes, available_tile, available_single,
                         1 if arithmetic_limits is not None else ne)
        if tile_nodes < 1:
            _error('complete matrix plane and coordinates exceed single-read cap', 'budget_exceeded')
        workspace_forecast = fixed_workspace + tile_nodes * per_energy_workspace
        arithmetic_engine = None
        arithmetic_status, arithmetic_reason = 'enclosed', None
        if arithmetic_limits is not None:
            if not isinstance(arithmetic_limits, EquilibriumArithmeticLimits):
                raise ValueError('arithmetic_limits must be EquilibriumArithmeticLimits or None')
            try:
                arithmetic_engine = _Enclosure(arithmetic_limits, ne*nk, nb, len(enabled),
                    limits.maximum_workspace_bytes-workspace_forecast)
            except _Budget as error:
                arithmetic_status, arithmetic_reason = 'budget_exceeded', str(error)

        def exact_call(method, *args):
            nonlocal arithmetic_status, arithmetic_reason
            if arithmetic_engine is None or arithmetic_status != 'enclosed':
                return
            try:
                getattr(arithmetic_engine, method)(*args)
            except (_Budget, _Invalid) as error:
                arithmetic_status = 'budget_exceeded' if isinstance(error, _Budget) else 'invalid_data'
                arithmetic_reason = str(error)
        # Actual complex bytes: A in both passes, every other matrix once.
        logical_forecast = reader.io_counters['logical_selected_bytes'] + sum(count*8 for count in vector_sizes.values()) + 16*ne*nk*nb*nb*(11+2*len(enabled))
        vectors={name:observations.read_array(vector_names[name],(slice(None),),
            maximum_bytes=reader.limits.maximum_array_bytes).values for name in vector_sizes}
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
            values=reader.read_array(matrices[base],selection,
                maximum_bytes=reader.limits.maximum_array_bytes).values
            return _finite(values[:,0],base)

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
        if fermi is not None:
            exact_call('set_fermi', fermi)
        norms = {name: _Norm() for name in ('A', 'raw_fdr', 'norm_fdr', 'corr_n', 'corr_p', 'raw_n', 'raw_p')}
        state_norms = {variant: {key: _Norm() for key in ('occupied_norm', 'empty_norm', 'sum_norm')}
                       for variant in ('raw', 'normalized')}
        side_norms = {side: {key: _Norm() for key in ('one_minus_f_gamma_norm', 'f_gamma_norm', 'sigma_fdr_defect_norm')}
                      for side in ('plus', 'minus')}
        numbers = {name: _Sum() for name in ('raw', 'raw_empty', 'normalized', 'normalized_empty')}
        number_imag = {name: _Sum() for name in numbers}
        flux = {side: {kind: (_Sum(), _Sum()) for kind in ('raw', 'normalized')} for side in ('plus', 'minus')}
        for start, stop, selection, weight in blocks():
            a, gr, gl, gg = (read_complex('state_dimensionless/' + name, selection) for name in ('A', 'GR', 'GL', 'GG'))
            if fermi is not None and arithmetic_limits is not None:
                k_index = selection[1].start
                exact_call('begin', a[0], gr[0], gl[0], gg[0], we[start], wk[k_index], start)
            sl_total, sg_total = np.zeros_like(gr), np.zeros_like(gr)
            for family in (*enabled, 'embedding_total'):
                family_sl = read_complex(f'selfenergy_dimensionless/{family}/SL', selection)
                family_sg = read_complex(f'selfenergy_dimensionless/{family}/SG', selection)
                if fermi is not None and arithmetic_limits is not None:
                    exact_call('total', family_sl[0], family_sg[0])
                sl_total += family_sl
                sg_total += family_sg
                del family_sl, family_sg
            if fermi is not None:
                exact_call('raw')
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
                one_minus_f = 1-f
                for variant, accumulator, occupied, empty in [('raw', 'raw_fdr', gn, gp), ('normalized', 'norm_fdr', normalized_n, normalized_p)]:
                    delta_n = occupied-f*a
                    delta_p = empty-one_minus_f*a
                    norms[accumulator].add(delta_n, weight)
                    norms[accumulator].add(delta_p, weight)
                    state_norms[variant]['occupied_norm'].add(delta_n, weight)
                    state_norms[variant]['empty_norm'].add(delta_p, weight)
                    # Sum the complete defects; never assume N+P=A or use n+p.
                    state_norms[variant]['sum_norm'].add(delta_n+delta_p, weight)
                    del delta_n, delta_p
            for side, family in [('plus', 'embedding_plus'), ('minus', 'embedding_minus')]:
                sl = read_complex(f'selfenergy_dimensionless/{family}/SL', selection)
                sg = read_complex(f'selfenergy_dimensionless/{family}/SG', selection)
                if fermi is not None and arithmetic_limits is not None:
                    exact_call('side', side, sl[0], sg[0])
                if fermi is not None:
                    sn = -1j*sl
                    gamma = sn+1j*sg
                    side_norms[side]['one_minus_f_gamma_norm'].add(one_minus_f*gamma, weight)
                    side_norms[side]['f_gamma_norm'].add(f*gamma, weight)
                    side_norms[side]['sigma_fdr_defect_norm'].add(sn-f*gamma, weight)
                    del sn, gamma
                for kind, lesser, greater in [('normalized', gl, gg), ('raw', 1j*gn, -1j*gp)]:
                    trace = _finite(np.trace(sg @ lesser - sl @ greater, axis1=-2, axis2=-1), 'boundary flux')
                    for w, value in zip(weight, trace):
                        flux[side][kind][0].add(float(w) * float(value.real))
                        flux[side][kind][1].add(float(w) * float(value.imag))
        number_report = dict(target_dimensionless=target, target_per_m2=target/l02)
        for name, accumulator in numbers.items():
            value = accumulator.value()
            occupied = name in ('raw', 'normalized')
            difference = abs(value-target) if occupied else None
            number_report[name] = dict(status='measured', value_dimensionless=value, value_per_m2=value/l02,
                units='m^-2', absolute_target_residual_dimensionless=difference,
                relative_target_residual=difference/target if occupied and target > 0 else None,
                target_residual_status='measured' if occupied else 'not_applicable',
                target_residual_reason=None if occupied else 'donor_target_applies_to_occupied_number',
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
        represented_bound = _represented_current_bound(norms, state_norms, side_norms, mu, scales['J0_A_per_m2'])
        budgets = dict(reader.io_counters)
        budgets.update(workspace_forecast_bytes=workspace_forecast, logical_two_pass_forecast_bytes=logical_forecast,
                       energy_tile_nodes=tile_nodes, workspace_scope='numeric arrays and scratch; not total process RSS')
        report = dict(schema='qcl-negf-stored-equilibrium-diagnostics-v1', source=_plain(description.source),
            source_evidence=stored_evidence,
            source_context=_plain(context), recorded_solver_status={name: _plain(assessment[name]) for name in (
                'iterative_converged', 'fixed_hartree_converged', 'outer_confirmation_count', 'fixture_solver_status'
                ) if isinstance(assessment, Mapping) and name in assessment},
            applicability=applicability, mu=mu, fdr=fdr, numbers=number_report, currents=currents,
            represented_current_bound=represented_bound,
            corrections=dict(occupied=_ratio(norms['corr_n'], norms['raw_n']), empty=_ratio(norms['corr_p'], norms['raw_p'])),
            spectral=dict(represented_capacity_dimensionless=capacity, represented_capacity_per_m2=capacity/l02,
                          minimum_number_weight=float(np.min(q)), imaginary_trace_dimensionless=spectral_imag.value()),
            domain=dict(energy_nodes=ne, momentum_nodes=nk, basis_states=nb,
                energy_relative_eV=[float(eev[0]), float(eev[-1])], momentum_dimensionless=[float(momentum[0]), float(momentum[-1])],
                outside_window='not_determined', basis_contract='finite represented basis'),
            scales=scales, budgets=budgets,
            interpretation='stored-state measurements; no fresh SCBA, convergence, discretization or experimental acceptance')
        if arithmetic_limits is not None:
            namespace = _arithmetic_report(arithmetic_engine, arithmetic_status, arithmetic_reason,
                mu, gs, scales['J0_A_per_m2'], currents, represented_bound)
            if len(json.dumps(namespace, allow_nan=False, separators=(',', ':')).encode('utf-8')) > 8192:
                _error('arithmetic namespace exceeds byte budget', 'budget_exceeded')
            report['represented_current_arithmetic_enclosure'] = namespace
        try:
            encoded = json.dumps(report, ensure_ascii=False, allow_nan=False, separators=(',', ':')).encode('utf-8')
        except (ValueError, OverflowError) as error:
            raise ContractError('nonfinite report intermediate/result', 'invalid_data') from error
        if len(encoded) > limits.maximum_report_bytes:
            _error('diagnostic report exceeds byte budget', 'budget_exceeded')
        return report
