"""Independent literal controls; expected values never use the tested engine."""
from fractions import Fraction as F
import math

from qcl_negf_results._equilibrium_arithmetic import (
    EquilibriumArithmeticLimits, _Arithmetic, _Enclosure,
)


def limits():
    return EquilibriumArithmeticLimits(16, 4, 20_000_000, 65536, 8*1024*1024, 180)


def pi_oracle():
    # Different identity from production: tan(atan(1/2)+atan(1/3))=1.
    lower, upper = F(0), F(0)
    for d in (2, 3):
        s = sum((F((-1)**n, (2*n+1)*d**(2*n+1)) for n in range(160)), F(0))
        lower += 4*s
        upper += 4*(s+F(1, 321*d**321))
    return lower, upper


def contains(interval, lower, upper=None):
    upper = lower if upper is None else upper
    assert F(interval['lower']) <= lower <= upper <= F(interval['upper'])


def matrix(rows):
    return rows


def certificate(a, n, p, sn, sp):
    e = _Enclosure(limits(), 1, len(a), 0, 32*1024*1024)
    e.set_fermi([0.5])
    identity = [[complex(i == j) for j in range(len(a))] for i in range(len(a))]
    e.begin(a, identity, [[1j*x for x in row] for row in n],
            [[-1j*x for x in row] for row in p], 1., 1., 0)
    e.total([[1j*x for x in row] for row in n], [[-1j*x for x in row] for row in p])
    e.raw()
    for side in ('plus', 'minus'):
        e.side(side, [[1j*x for x in row] for row in sn], [[-1j*x for x in row] for row in sp])
    return e


def test_literal_scalar_equality_and_terms():
    # δN=1, δP=-1, D=0, δS=-1: root terms 1,1,2,0; C=4.
    e = certificate([[2]], [[2]], [[0]], [[0]], [[2]])
    assert e.sums == [F(4), F(1), F(1), F(0), F(1), F(1), F(0),
                      F(1), F(1), F(1), F(1), F(1), F(1)]
    lo, hi = pi_oracle()
    row = e.row('plus', 'raw', 2, 1., complex(4/math.pi), 4/math.pi)
    contains(row['current_real'], 4/hi, 4/lo)
    contains(row['rhs'], 4/hi, 4/lo)
    contains(row['current_imag'], F(0))


def test_noncommuting_complex_trace_and_rhs():
    # Γ+=[[1,1],[0,1]], Γ-=transpose; Nraw=I/2+X; Nnorm=I/2+iH.
    e = _Enclosure(limits(), 1, 2, 0, 32*1024*1024)
    e.set_fermi([0.5])
    a = [[1,0],[0,1]]
    nr = [[.5,1],[1,.5]]
    p = [[.5,0],[0,.5]]
    nn = [[.5,1j],[0,.5]]
    e.begin(a, a, [[1j*x for x in r] for r in nn],
            [[-1j*x for x in r] for r in p], 1., 1., 0)
    e.total([[1j*x for x in r] for r in nr], [[-1j*x for x in r] for r in p])
    e.raw()
    for side, gamma in [('plus', [[1,1],[0,1]]), ('minus', [[1,0],[1,1]])]:
        sl = [[.5j*x for x in r] for r in gamma]
        sg = [[-.5j*x for x in r] for r in gamma]
        e.side(side, sl, sg)
    assert e.traces[('plus', 'raw')] == (F(1,2), F(0))
    assert e.traces[('minus', 'raw')] == (F(1,2), F(0))
    assert e.traces[('plus', 'normalized')] == (F(0), F(0))
    assert e.traces[('minus', 'normalized')] == (F(0), F(1,2))
    assert e.sums[:7] == [F(2), F(2), F(0), F(2), F(1), F(0), F(1)]
    assert e.sums[7:] == [F(3,4), F(3,4), F(0)]*2


def test_tiny_exact_positive_and_subnormal_conversion():
    t = math.ldexp(1., -600)
    e = certificate([[0]], [[t]], [[0]], [[0]], [[t]])
    assert e.traces[('plus', 'raw')] == (F(1, 2**1200), F(0))
    row = e.row('plus', 'raw', 2, 1., 0j, 0.)
    assert row['current_real']['upper'] > 0
    assert row['rhs']['upper'] > 0
    # Never split the minimum subnormal t/8 into stored families.
    t = math.ulp(0.)
    e = certificate([[0]], [[t]], [[0]], [[0]], [[t]])
    assert e.traces[('plus', 'raw')][0] == F(1, 2**2148)
    assert e.row('plus', 'raw', 2, 1., 0j, 0.)['rhs']['upper'] > 0


def test_independent_pi_containment_and_scaled_roots():
    a = _Arithmetic(limits(), 256)
    lo, hi = pi_oracle()
    pl, ph = a.pi()
    assert F(3) < pl <= lo < hi <= ph < F(22,7)
    for q in (F(0), F(4,9), F(2), F(1,2**2400), F(2**2400)):
        low, high = a.sqrt(q)
        assert low >= 0 and low*low <= q <= high*high
        if q == F(4,9):
            assert low == high == F(2,3)
    # floor(Z) is a square, but Z itself is not: no false collapse.
    low, high = a.sqrt(F(1)+F(1,2**300))
    assert low < high and low*low <= 1+F(1,2**300) <= high*high


def sqrt_oracle(q):
    # Bisection only; distinct from production's scaled isqrt construction.
    lower, upper = F(0), max(F(1), q)
    for _ in range(400):
        middle = (lower+upper)/2
        if middle*middle <= q:
            lower = middle
        else:
            upper = middle
    return lower, upper


def test_literal_noncommuting_normalized_rhs_and_common_z():
    e = _Enclosure(limits(), 1, 2, 0, 32*1024*1024)
    e.set_fermi([0.5])
    e.begin([[1,0],[0,1]], [[1,0],[0,1]], [[.5j,-1],[0,.5j]],
            [[-.5j,0],[0,-.5j]], 1., 1., 0)
    e.total([[.5j,1j],[1j,.5j]], [[-.5j,0],[0,-.5j]])
    e.raw()
    e.side('minus', [[.5j,0],[.5j,.5j]], [[-.5j,0],[-.5j,-.5j]])
    lo, hi = pi_oracle()
    rlo, rhi = sqrt_oracle(F(3))
    row = e.row('minus', 'normalized', 2, 1., complex(0,1/(2*math.pi)), math.sqrt(3)/(2*math.pi))
    contains(row['current_imag'], 1/(2*hi), 1/(2*lo))
    contains(row['current_common_z_imag'], -1/(2*lo), -1/(2*hi))
    contains(row['rhs'], rlo/(2*hi), rhi/(2*lo))


def test_caps_before_growing_arithmetic_and_no_partial_rows():
    import pytest
    from qcl_negf_results._equilibrium_arithmetic import _Budget, _report
    tiny = EquilibriumArithmeticLimits(1, 1, 1, 64, 1024, 1.)
    with pytest.raises(_Budget):
        _Enclosure(tiny, 2, 1, 0, 1024)
    a = _Arithmetic(limits(), 256)
    a.limits = EquilibriumArithmeticLimits(1, 1, 20_000_000, 64, 1024*1024, 180.)
    with pytest.raises(_Budget):
        a.mul(F(2**40), F(2**40))
    result = _report(None, 'budget_exceeded', 'cap', {'value_relative_eV': 0.}, 2, 1., {}, {})
    assert all(row['current_real'] is None and row['rhs'] is None
               for rows in result['rows'].values() for row in rows.values())
    # Missing μ precedes any zero/budget shortcut; no invented f hash.
    result = _report(None, 'budget_exceeded', 'cap',
        {'value_relative_eV': None, 'status': 'not_applicable', 'reason': 'fixture'}, 2, 1., {}, {})
    assert result['status'] == 'not_applicable' and result['f_binary64_sha256'] is None


def test_signed_outward_endpoints_and_inward_mutation_detection():
    import pytest
    a = _Arithmetic(limits(), 256)
    for q in (F(1,2**2400), F(-1,2**2400), F(1,3), F(-1,3), F(2**1000)):
        interval = a.interval(q, q)
        contains(interval, q)
    # Independent containment assertion rejects an inward one-ulp mutation.
    interval = a.interval(F(1,3), F(1,3))
    interval['upper'] = math.nextafter(interval['upper'], -math.inf)
    with pytest.raises(AssertionError):
        contains(interval, F(1,3))


def test_literal_nonunit_measure_spin_and_current_scale():
    # Independent scalar table: u=3/32, C=u*4=3/8, all four root
    # terms before kappa/J0 are u*(1,1,2,0). Thus J*=B*=45/(16*pi).
    e = _Enclosure(limits(), 1, 1, 1, 32*1024*1024)
    e.set_fermi([0.5])
    e.begin([[2]], [[1]], [[2j]], [[0]], 3/8, 1/4, 0)
    e.total([[2j]], [[0]])
    e.raw()
    for side in ('plus', 'minus'):
        e.side(side, [[0]], [[-2j]])
    assert e.sums == [F(3,8), F(3,32), F(3,32), F(0),
                      F(3,32), F(3,32), F(0),
                      F(3,32), F(3,32), F(3,32),
                      F(3,32), F(3,32), F(3,32)]
    assert all(trace == (F(3,8), F(0)) for trace in e.traces.values())
    pi_lo, pi_hi = pi_oracle()
    lower, upper = F(45,16)/pi_hi, F(45,16)/pi_lo
    for side in ('plus', 'minus'):
        for variant in ('raw', 'normalized'):
            row = e.row(side, variant, 3, 5., 0j, 0.)
            contains(row['current_real'], lower, upper)
            contains(row['rhs'], lower, upper)
            # Explicitly separate wrong-u=1, wrong-gs=2, wrong-J0=1
            # predictions; these tests do not run source mutants.
            for interval in (row['current_real'], row['rhs']):
                assert F(interval['upper']) < F(30)/pi_hi
                assert F(interval['lower']) > F(15,8)/pi_lo
                assert F(interval['lower']) > F(9,16)/pi_lo


def test_literal_legacy_float_error_upper_guarantees():
    # A deliberately displaced reported legacy point tests the error functional;
    # it is not an assertion about the size of actual solver rounding.
    e = certificate([[2]], [[2]], [[0]], [[0]], [[2]])
    row = e.row('plus', 'raw', 2, 1., 10+0j, 0.)
    pi_lo, pi_hi = pi_oracle()
    lower, upper = F(4)/pi_hi, F(4)/pi_lo
    assert 0 < lower <= upper < 10
    contains(row['current_real'], lower, upper)
    contains(row['current_imag'], F(0))
    contains(row['rhs'], lower, upper)
    # Actual |10-J*| is in [10-upper,10-lower]; |0-B*| in [lower,upper].
    # Comparing the independently derived upper endpoints rejects zero-error
    # claims even though the J*/B* intervals themselves remain unchanged.
    assert F(row['current_float_error_upper']) >= 10-lower > 0
    assert F(row['rhs_float_error_upper']) >= upper > 0
    assert F(row['current_abs_upper']) >= upper
    assert F(row['current_float_abs_upper']) >= 10


def test_literal_original_family_sum_and_exact_raw_products():
    # Source inputs: ordered SL/i = 2^54,1,-2^54,0; SG/i = 1,0,0,-1.
    # Last pair is embedding_total, included once. No PSD/S1 claim is made.
    e = _Enclosure(limits(), 1, 1, 3, 32*1024*1024)
    e.set_fermi([0.5])
    gr = complex(1., math.ldexp(1., -52))
    e.begin([[0]], [[gr]], [[0]], [[0]], 1., 1., 0)
    e.total([[complex(0, 2**54)]], [[1j]])
    e.total([[1j]], [[0]])
    e.total([[complex(0, -2**54)]], [[0]])
    e.total([[0]], [[-1j]])
    e.raw()
    e.side('plus', [[0]], [[-1j]])
    # Handwritten scalar multiplication: (1+i*2^-52)(1-i*2^-52)
    # equals 1+2^-104; no production matmul/reducer enters the expected path.
    r = F(1)+F(1, 2**104)
    square = r*r
    assert e.n['raw'] == [[(r, F(0))]]
    assert e.p['raw'] == [[(F(0), F(0))]]
    assert e.traces[('plus', 'raw')] == (r, F(0))
    assert e.traces[('plus', 'normalized')] == (F(0), F(0))
    assert e.sums[:7] == [F(0), square, F(0), square, F(0), F(0), F(0)]
    # The literal targets distinguish both rounded summation and rounded |GR|².
    rounded_total = (float(2**54)+1.)-float(2**54)
    assert rounded_total == 0. and r != F(rounded_total)
    assert float(r) == 1. and r != F(1)
    pi_lo, pi_hi = pi_oracle()
    row = e.row('plus', 'raw', 2, 1., 0j, 0.)
    contains(row['current_real'], r/pi_hi, r/pi_lo)
    contains(row['rhs'], r/pi_hi, r/pi_lo)
