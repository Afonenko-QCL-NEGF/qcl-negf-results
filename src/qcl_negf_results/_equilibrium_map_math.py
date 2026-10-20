"""Private finite MAP measurements with an explicit scalar operation graph.

No acceptance verdict or floating error enclosure is supplied here. Runtime
rounding assumptions and arithmetic error bounds remain not_measured. The
engineering ceilings below bound the already admitted S1 array shapes; they
are neither an admission for a solver run nor a physical discretization rule.
"""
from __future__ import annotations

from collections.abc import Sequence
import math
import sys

import numpy as np

from qcl_negf_contracts.messages import ContractError


_MAX_BASIS = 7
_MAX_SPATIAL = 192
_MAX_OPERATIONS = 8 * _MAX_SPATIAL * _MAX_BASIS * _MAX_BASIS


def _error(reason: str, code: str = 'invalid_data') -> None:
    raise ContractError(reason, code)


def _count(value: int, ceiling: int, name: str, *, zero: bool = False) -> int:
    if type(value) is not int or value < (0 if zero else 1):
        _error(f'{name}: invalid integer count')
    if value > min(sys.maxsize, ceiling):
        _error(f'{name}: engineering count cap exceeded', 'budget_exceeded')
    return value


def _number(value: object, name: str, *, positive: bool = False) -> float:
    # No array/object conversion or Bool-as-integer interpretation.
    if type(value) not in (float, int, np.float64):
        _error(f'{name}: expected a real scalar, not Bool or a coercible object')
    try:
        result = float(value)
    except (OverflowError, ValueError):
        _error(f'{name}: scalar is not finite binary64')
    if not math.isfinite(result) or (positive and result <= 0.0):
        _error(f'{name}: expected finite {"positive " if positive else ""}scalar')
    return result


def _finite(value: float) -> float:
    if not math.isfinite(value):
        _error('nonfinite arithmetic intermediate')
    return value


def _add(a: float, b: float) -> float:
    return _finite(a + b)


def _sub(a: float, b: float) -> float:
    return _finite(a - b)


def _mul(a: float, b: float) -> float:
    result = _finite(a * b)
    if result == 0.0 and a != 0.0 and b != 0.0:
        _error('arithmetic_events.underflow_or_zero_rounding=true: multiplication')
    return result


def _div(a: float, b: float) -> float:
    if b == 0.0:
        _error('zero arithmetic denominator')
    result = _finite(a / b)
    if result == 0.0 and a != 0.0:
        _error('arithmetic_events.underflow_or_zero_rounding=true: division')
    return result


def _array_storage(array: object, dimensions: int, caps: tuple[int, ...],
                   name: str, *, writable: bool = False,
                   allow_empty: bool = False) -> None:
    if type(array) is not np.ndarray:
        _error(f'{name}: expected a native ndarray', 'unsupported_layout')
    if array.dtype != np.dtype(np.float64) or array.ndim != dimensions:
        _error(f'{name}: expected native float64 shape', 'unsupported_layout')
    if not array.flags.c_contiguous:
        _error(f'{name}: storage must be C-contiguous', 'unsupported_layout')
    cells = 1
    for length, cap in zip(array.shape, caps):
        _count(length, cap, name, zero=allow_empty)
        cells *= length
    _count(cells, math.prod(caps), f'{name} cells', zero=allow_empty)
    _count(cells * 8, math.prod(caps) * 8, f'{name} bytes', zero=allow_empty)
    if array.size != cells or array.nbytes != cells * 8:
        _error(f'{name}: inconsistent storage size')
    if writable and not array.flags.writeable:
        _error(f'{name}: output is read-only')


def _plane_storage(plane: object, caps: tuple[int, int], name: str,
                   *, writable: bool = False) -> None:
    if type(plane) not in (tuple, list) or len(plane) != 2:
        _error(f'{name}: expected real/imag pair', 'unsupported_layout')
    for component in plane:
        _array_storage(component, 2, caps, name, writable=writable)
    if plane[0].shape != plane[1].shape:
        _error(f'{name}: real/imag shape mismatch')
    if np.shares_memory(plane[0], plane[1]):
        _error(f'{name}: real/imag components alias')


def _scan(array: np.ndarray, name: str, *, positive: bool = False) -> None:
    # Scalar scan, no full-size isfinite/bool temporary.
    for value in array.flat:
        _number(value, name, positive=positive)


def _scan_plane(plane: tuple[np.ndarray, np.ndarray], name: str) -> None:
    _scan(plane[0], name)
    _scan(plane[1], name)


def _disjoint(outputs: tuple[np.ndarray, ...], inputs: tuple[np.ndarray, ...]) -> None:
    for index, output in enumerate(outputs):
        for other in outputs[index + 1:]:
            if np.shares_memory(output, other):
                _error('output components alias')
        for source in inputs:
            if np.shares_memory(output, source):
                _error('input/output alias')


def _hat_area(x0: float, x1: float, left: float, right: float, side: str) -> float:
    """Integrate one linear hat on its clipped native interval."""
    x0, x1 = _number(x0, 'x0'), _number(x1, 'x1')
    left, right = _number(left, 'left'), _number(right, 'right')
    if x1 <= x0 or right < left or type(side) is not str or side not in ('left', 'right'):
        _error('invalid hat interval or side')
    low, high = max(x0, left), min(x1, right)
    if low >= high:
        return 0.0
    width = _sub(x1, x0)
    t_low, t_high = _div(_sub(low, x0), width), _div(_sub(high, x0), width)
    if side == 'left':
        t_low, t_high = _sub(1.0, t_low), _sub(1.0, t_high)
    return _mul(_mul(_sub(high, low), _add(t_low, t_high)), 0.5)


def _annulus_square_area(b0: float, b1: float, r0: float, r1: float) -> float:
    """Clipped squared-radius measure; caller owns radial constants."""
    b0, b1 = _number(b0, 'b0'), _number(b1, 'b1')
    r0, r1 = _number(r0, 'r0'), _number(r1, 'r1')
    if b0 < 0.0 or r0 < 0.0 or b1 <= b0 or r1 < r0:
        _error('invalid radial interval')
    low, high = max(b0, r0), min(b1, r1)
    if low >= high:
        return 0.0
    return _sub(_mul(high, high), _mul(low, low))


def _overlap_cell(source_left: float, source_right: float,
                  target_left: float, target_right: float) -> float:
    """Length of the intersection of two positive-width cells."""
    sl, sr = _number(source_left, 'source left'), _number(source_right, 'source right')
    tl, tr = _number(target_left, 'target left'), _number(target_right, 'target right')
    if sr <= sl or tr <= tl:
        _error('cell widths must be positive')
    low, high = max(sl, tl), min(sr, tr)
    return 0.0 if low >= high else _sub(high, low)


def _sequence_length(value: object, cap: int, name: str) -> int:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        _error(f'{name}: expected bounded sequence')
    try:
        length = len(value)
    except (OverflowError, TypeError, ValueError):
        _error(f'{name}: invalid sequence length')
    return _count(length, cap, name)


def _edges(edges: Sequence[float], length: int, name: str) -> None:
    previous = _number(edges[0], name)
    width = None
    for index in range(1, length):
        current = _number(edges[index], name)
        delta = _sub(current, previous)
        if delta <= 0.0 or (width is not None and delta != width):
            _error(f'{name}: grid must be strictly increasing and uniform')
        width = delta
        previous = current


def _restrict_cells(source_edges: Sequence[float], target_edges: Sequence[float],
                    values: Sequence[float]) -> list[float]:
    """Signed conservative restriction on one coincident full uniform domain.

    The only growing allocation is the target-length output list (<=192
    Python floats and list slots). Sweep state is constant size; no dense T.
    """
    ns_edges = _sequence_length(source_edges, _MAX_SPATIAL + 1, 'source edges')
    nt_edges = _sequence_length(target_edges, _MAX_SPATIAL + 1, 'target edges')
    nv = _sequence_length(values, _MAX_SPATIAL, 'values')
    if ns_edges < 2 or nt_edges < 2 or nv != ns_edges - 1:
        _error('missing cells or edges')
    ns, nt = ns_edges - 1, nt_edges - 1
    _count(ns + nt - 1, 2 * _MAX_SPATIAL - 1, 'overlap count')
    _count(nt * 8, _MAX_SPATIAL * 8, 'output numeric bytes')
    _edges(source_edges, ns_edges, 'source edges')
    _edges(target_edges, nt_edges, 'target edges')
    for index in range(nv):
        _number(values[index], 'cell value')
    if source_edges[0] != target_edges[0] or source_edges[-1] != target_edges[-1]:
        _error('restriction requires coincident full domains')
    result = []
    source = 0
    for target in range(nt):
        total = 0.0
        tl, tr = float(target_edges[target]), float(target_edges[target + 1])
        while source < ns:
            sl, sr = float(source_edges[source]), float(source_edges[source + 1])
            low, high = max(sl, tl), min(sr, tr)
            if low < high:
                total = _add(total, _mul(_sub(high, low), float(values[source])))
            if sr >= tr:
                if sr == tr:
                    source += 1
                break
            source += 1
        result.append(_div(total, _sub(tr, tl)))
    return result


def _mac(ar: float, ai: float, br: float, bi: float,
         real: float, imag: float) -> tuple[float, float]:
    """Exactly four real multiplies and four additions/subtractions."""
    p, q = _mul(ar, br), _mul(ai, bi)
    r, s = _mul(ar, bi), _mul(ai, br)
    term_real, term_imag = _sub(p, q), _add(r, s)
    return _add(real, term_real), _add(imag, term_imag)


def _complex_matmul(left, right, out) -> None:
    """Square b-by-b product, i,j,k order: 8*b**3 fp primitives."""
    caps = (_MAX_BASIS, _MAX_BASIS)
    _plane_storage(left, caps, 'left')
    _plane_storage(right, caps, 'right')
    _plane_storage(out, caps, 'out', writable=True)
    b = left[0].shape[0]
    if left[0].shape != (b, b) or right[0].shape != (b, b) or out[0].shape != (b, b):
        _error('product requires matching square matrices')
    _count(8 * b * b * b, _MAX_OPERATIONS, 'product operations')
    _disjoint(tuple(out), tuple(left) + tuple(right))
    _scan_plane(left, 'left')
    _scan_plane(right, 'right')
    for i in range(b):
        for j in range(b):
            real, imag = 0.0, 0.0
            for k in range(b):
                real, imag = _mac(float(left[0][i, k]), float(left[1][i, k]),
                                  float(right[0][k, j]), float(right[1][k, j]), real, imag)
            out[0][i, j], out[1][i, j] = real, imag


def _gram(left, right, out) -> None:
    """Rectangular F_left dagger F_right; z,a,b order, 8*Q*a*b ops."""
    _plane_storage(left, (_MAX_SPATIAL, _MAX_BASIS), 'left')
    _plane_storage(right, (_MAX_SPATIAL, _MAX_BASIS), 'right')
    _plane_storage(out, (_MAX_BASIS, _MAX_BASIS), 'out', writable=True)
    q, a = left[0].shape
    qr, b = right[0].shape
    if qr != q or out[0].shape != (a, b):
        _error('Gram dimension mismatch')
    _count(8 * q * a * b, _MAX_OPERATIONS, 'Gram operations')
    _disjoint(tuple(out), tuple(left) + tuple(right))
    _scan_plane(left, 'left')
    _scan_plane(right, 'right')
    for i in range(a):
        for j in range(b):
            out[0][i, j], out[1][i, j] = 0.0, 0.0
    for z in range(q):
        for i in range(a):
            for j in range(b):
                real, imag = _mac(float(left[0][z, i]), -float(left[1][z, i]),
                                  float(right[0][z, j]), float(right[1][z, j]),
                                  float(out[0][i, j]), float(out[1][i, j]))
                out[0][i, j], out[1][i, j] = real, imag


def _observable_storage(basis, matrix, out, *, bond: bool) -> tuple[int, int]:
    _plane_storage(basis, (_MAX_SPATIAL, _MAX_BASIS), 'basis')
    _plane_storage(matrix, (_MAX_BASIS, _MAX_BASIS), 'M')
    _array_storage(out, 1, (_MAX_SPATIAL,), 'out', writable=True, allow_empty=bond)
    z, b = basis[0].shape
    if matrix[0].shape != (b, b) or out.shape != (z - 1 if bond else z,):
        _error('observable dimension mismatch')
    rows = z - 1 if bond else z
    _count(rows * (8 * b * b + 8 * b),
           _MAX_SPATIAL * (8 * _MAX_BASIS * _MAX_BASIS + 8 * _MAX_BASIS),
           'observable operations', zero=bond)
    _count(2 * b * 8, 2 * _MAX_BASIS * 8, 'contraction scratch numeric bytes')
    _disjoint((out,), tuple(basis) + tuple(matrix))
    return z, b


def _row_contraction(basis, matrix, row: int, conjugate_row: int,
                     b: int, vr: list[float], vi: list[float]) -> tuple[float, float]:
    # v = M conj(basis[conjugate_row]); then basis[row] v.
    # Exactly 8*b*b + 8*b fp operations; scratch is two b-length lists.
    for i in range(b):
        real, imag = 0.0, 0.0
        for k in range(b):
            real, imag = _mac(float(matrix[0][i, k]), float(matrix[1][i, k]),
                              float(basis[0][conjugate_row, k]),
                              -float(basis[1][conjugate_row, k]), real, imag)
        vr[i], vi[i] = real, imag
    real, imag = 0.0, 0.0
    for i in range(b):
        real, imag = _mac(float(basis[0][row, i]), float(basis[1][row, i]),
                          vr[i], vi[i], real, imag)
    return real, imag


def _density_from_integrated(chi, M, L0: float, out) -> None:
    """Direct actual-chi density, Re(chi M chi dagger)/L0**3."""
    z, b = _observable_storage(chi, M, out, bond=False)
    L0 = _number(L0, 'L0', positive=True)
    _scan_plane(chi, 'chi')
    _scan_plane(M, 'M')
    scale = _div(1.0, _mul(_mul(L0, L0), L0))
    vr, vi = [0.0] * b, [0.0] * b
    for row in range(z):
        real, _ = _row_contraction(chi, M, row, row, b, vr, vi)
        out[row] = _mul(real, scale)


def _bond_from_integrated(Phi, M, masses, dx: float, cbar: float,
                          J0: float, out) -> None:
    """Internal Dirichlet bonds from actual Phi; no seam or extra dx factor."""
    z, b = _observable_storage(Phi, M, out, bond=True)
    _array_storage(masses, 1, (_MAX_SPATIAL,), 'masses')
    if masses.shape != (z,):
        _error('mass dimension mismatch')
    dx = _number(dx, 'dx', positive=True)
    cbar = _number(cbar, 'cbar', positive=True)
    J0 = _number(J0, 'J0', positive=True)
    _disjoint((out,), (masses,))
    _scan_plane(Phi, 'Phi')
    _scan_plane(M, 'M')
    _scan(masses, 'masses', positive=True)
    vr, vi = [0.0] * b, [0.0] * b
    for row in range(z - 1):
        mass_sum = _add(float(masses[row]), float(masses[row + 1]))
        h = _div(_mul(-cbar, _div(2.0, mass_sum)), _mul(dx, dx))
        _, imag = _row_contraction(Phi, M, row + 1, row, b, vr, vi)
        out[row] = _mul(_mul(_mul(-2.0, J0), h), imag)
