"""Guarded exact arithmetic for one represented binary64 formula, not physics accuracy.

No rounded raw reconstruction, norms, complement or pi is a certificate input.
All growing integer arithmetic is preceded by a conservative transient census.
"""
from dataclasses import dataclass, fields
from fractions import Fraction as F
import hashlib
import math
import struct
import time


@dataclass(frozen=True)
class EquilibriumArithmeticLimits:
    maximum_cells: int
    maximum_basis_states: int
    maximum_exact_operations: int
    maximum_integer_bits: int
    maximum_workspace_bytes: int
    maximum_wall_seconds: float

    def __post_init__(self):
        for field in fields(self):
            x = getattr(self, field.name)
            if field.name == 'maximum_wall_seconds':
                if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or x <= 0:
                    raise ValueError('maximum_wall_seconds must be finite and positive')
            elif type(x) is not int or x < 1:
                raise ValueError(f'{field.name} must be a positive integer')


class _Budget(Exception):
    pass


class _Invalid(Exception):
    pass


class _Arithmetic:
    def __init__(self, limits, slots, workspace=None):
        self.limits = limits
        self.slots = slots
        self.workspace = min(limits.maximum_workspace_bytes,
                             limits.maximum_workspace_bytes if workspace is None else workspace)
        if self.workspace <= 0:
            raise _Budget('no remaining workspace')
        self.deadline = time.monotonic() + limits.maximum_wall_seconds
        self.operations = 0
        self.peak_bits = 0
        self.peak_workspace = 0

    def guard(self, bits, operations=1):
        # slots bounds all simultaneously live rational scalars; 64 additional
        # integer objects cover cross-products, gcd, divmod, shifts and conversion.
        # CPython 30-bit digits rounded up to 8 bytes/digit (conservative).
        peak = max(bits, self.peak_bits)
        workspace = (2*self.slots+64)*(64+8*((peak+29)//30)) + 128*self.slots
        if (bits > self.limits.maximum_integer_bits or workspace > self.workspace
                or self.operations+operations > self.limits.maximum_exact_operations
                or time.monotonic() > self.deadline):
            raise _Budget('transient integer/operation/workspace/deadline cap')
        self.operations += operations
        self.peak_bits = peak
        self.peak_workspace = max(workspace, self.peak_workspace)

    @staticmethod
    def bits(x):
        return max(x.numerator.bit_length(), x.denominator.bit_length(), 1)

    def number(self, x):
        # Every finite binary64 has numerator/denominator <=1075 bits.
        self.guard(1075)
        x = float(x)
        if not math.isfinite(x):
            raise _Invalid('nonfinite represented input')
        n, d = x.as_integer_ratio()
        return F(n, d)

    def add(self, x, y):
        self.guard(self.bits(x)+self.bits(y)+2, 8)
        return x+y

    def neg(self, x):
        self.guard(self.bits(x), 1)
        return -x

    def sub(self, x, y):
        return self.add(x, self.neg(y))

    def mul(self, x, y):
        self.guard(self.bits(x)+self.bits(y)+1, 8)
        return x*y

    def div(self, x, y):
        self.guard(self.bits(x)+self.bits(y)+1, 8)
        if not y:
            raise _Invalid('zero divisor')
        return x/y

    def compare(self, x, y):
        self.guard(self.bits(x)+self.bits(y)+1, 4)
        return (x > y)-(x < y)

    def absolute(self, x):
        self.guard(self.bits(x))
        return abs(x)

    def maximum(self, x, y):
        return x if self.compare(x, y) >= 0 else y

    def shift(self, x, exponent):
        self.guard(self.bits(x)+abs(exponent)+1, 8)
        factor = F(1 << abs(exponent))
        return self.mul(x, factor) if exponent >= 0 else self.div(x, factor)

    def sqrt(self, q):
        if self.compare(q, F(0)) < 0:
            raise _Invalid('negative root')
        if not q:
            return F(0), F(0)
        # Collapse only if numerator AND denominator are exact squares.
        self.guard(self.bits(q)+2, 16)
        rn, rd = math.isqrt(q.numerator), math.isqrt(q.denominator)
        self.guard(self.bits(q)+2, 8)
        if rn*rn == q.numerator and rd*rd == q.denominator:
            return F(rn, rd), F(rn, rd)
        exponent = q.numerator.bit_length()-q.denominator.bit_length()
        if self.compare(q, self.shift(F(1), exponent)) < 0:
            exponent -= 1
        k = exponent//2
        z = self.shift(q, 256-2*k)
        self.guard(self.bits(z)+2, 16)
        m = math.isqrt(z.numerator//z.denominator)
        return self.shift(F(m), k-128), self.shift(F(m+1), k-128)

    def pi(self):
        intervals = []
        for denominator in (5, 239):
            x = F(1, denominator)
            xx = self.mul(x, x)
            power, total = x, F(0)
            for n in range(64):
                term = self.div(power, F(2*n+1))
                total = self.add(total, term if n % 2 == 0 else self.neg(term))
                power = self.mul(power, xx)
            intervals.append((total, self.add(total, self.div(power, F(129)))))
        (l5, h5), (l239, h239) = intervals
        low = self.sub(self.mul(F(16), l5), self.mul(F(4), h239))
        high = self.sub(self.mul(F(16), h5), self.mul(F(4), l239))
        if self.compare(low, F(0)) <= 0 or self.compare(low, high) > 0:
            raise _Invalid('invalid pi interval')
        return low, high

    def outward(self, q, upper):
        # Fraction.__float__ uses bounded integer shifts/divmod. Reserve 2200
        # additional transient bits before entering it or as_integer_ratio.
        self.guard(self.bits(q)+2200, 32)
        try:
            value = float(q)
        except OverflowError as error:
            raise _Invalid('finite endpoint unavailable') from error
        if not math.isfinite(value):
            raise _Invalid('finite endpoint unavailable')
        represented = self.number(value)
        comparison = self.compare(represented, q)
        if (upper and comparison < 0) or (not upper and comparison > 0):
            value = math.nextafter(value, math.inf if upper else -math.inf)
        if not math.isfinite(value):
            raise _Invalid('finite endpoint unavailable')
        comparison = self.compare(self.number(value), q)
        if (upper and comparison < 0) or (not upper and comparison > 0):
            raise _Invalid('outward endpoint containment failed')
        return value

    def interval(self, lo, hi):
        return dict(lower=self.outward(lo, False), upper=self.outward(hi, True))

    def cadd(self, x, y):
        return self.add(x[0], y[0]), self.add(x[1], y[1])

    def cneg(self, x):
        return self.neg(x[0]), self.neg(x[1])

    def cmul(self, x, y):
        return (self.sub(self.mul(x[0], y[0]), self.mul(x[1], y[1])),
                self.add(self.mul(x[0], y[1]), self.mul(x[1], y[0])))

    def cscale(self, x, s):
        return self.mul(x[0], s), self.mul(x[1], s)

    def matrix(self, rows):
        return [[(self.number(complex(x).real), self.number(complex(x).imag)) for x in row] for row in rows]

    def matmul(self, x, y):
        n = len(x)
        result = []
        for i in range(n):
            row = []
            for j in range(n):
                total = (F(0), F(0))
                for k in range(n):
                    total = self.cadd(total, self.cmul(x[i][k], y[k][j]))
                row.append(total)
            result.append(row)
        return result


class _Enclosure:
    def __init__(self, limits, cells, nb, channels, workspace):
        # Fixed live-object census: <=24 complex Nb² planes (96 rational
        # scalars/Nb² including list-comprehension overlap), 128 accumulators,
        # and 128 scalar interval/Machin temporaries. No per-cell retained data.
        self.slots = 96*nb*nb+256
        self.a = _Arithmetic(limits, self.slots, workspace)
        self.forecast = 512*cells*(nb**3+(channels+8)*nb*nb)+500000
        if (cells > limits.maximum_cells or nb > limits.maximum_basis_states
                or self.forecast > limits.maximum_exact_operations):
            raise _Budget('dimension or exact-operation forecast cap')
        self.a.guard(1075, 0)
        self.nb = nb
        self.sums = [F(0) for _ in range(13)]
        self.traces = {(s, v): (F(0), F(0)) for s in ('plus', 'minus') for v in ('raw', 'normalized')}
        self.f_hash = None
        self.f_count = 0

    def set_fermi(self, fermi):
        digest = hashlib.sha256()
        for value in fermi:
            self.a.guard(1075)
            value = float(value)
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise _Invalid('invalid represented f')
            digest.update(struct.pack('<d', value))
        self.f_hash, self.f_count = digest.hexdigest(), len(fermi)
        self.fermi = fermi  # Existing vector; no copied Fraction vector.

    def rotate(self, matrix, positive):
        a = self.a
        return [[(a.neg(y), x) if positive else (y, a.neg(x)) for x, y in row] for row in matrix]

    def begin(self, spectral, gr, gl, gg, we, wk, energy_index):
        a = self.a
        # Drop previous cell before allocating the next cell.
        self.spectral = self.gr = self.n = self.p = self.sl = self.sg = self.defects = None
        self.u = a.mul(a.number(we), a.number(wk))
        self.f = a.number(self.fermi[energy_index])
        self.complement = a.sub(F(1), self.f)
        self.spectral, self.gr = a.matrix(spectral), a.matrix(gr)
        self.n = {'normalized': self.rotate(a.matrix(gl), False)}
        self.p = {'normalized': self.rotate(a.matrix(gg), True)}
        self.sl = [[(F(0), F(0)) for _ in range(self.nb)] for _ in range(self.nb)]
        self.sg = [[(F(0), F(0)) for _ in range(self.nb)] for _ in range(self.nb)]

    def total(self, sl, sg):
        a = self.a
        for target, source in ((self.sl, sl), (self.sg, sg)):
            for i in range(self.nb):
                for j in range(self.nb):
                    z = complex(source[i][j])
                    target[i][j] = a.cadd(target[i][j], (a.number(z.real), a.number(z.imag)))

    def norm(self, index, matrix):
        a = self.a
        for row in matrix:
            for x, y in row:
                square = a.add(a.mul(x, x), a.mul(y, y))
                self.sums[index] = a.add(self.sums[index], a.mul(self.u, square))

    def raw(self):
        a = self.a
        ga = [[(self.gr[j][i][0], a.neg(self.gr[j][i][1])) for j in range(self.nb)] for i in range(self.nb)]
        self.n['raw'] = self.rotate(a.matmul(a.matmul(self.gr, self.sl), ga), False)
        self.p['raw'] = self.rotate(a.matmul(a.matmul(self.gr, self.sg), ga), True)
        self.norm(0, self.spectral)
        self.defects = {}
        for variant, start in (('raw', 1), ('normalized', 4)):
            dn = [[a.cadd(x, a.cneg(a.cscale(y, self.f))) for x, y in zip(nr, ar)]
                  for nr, ar in zip(self.n[variant], self.spectral)]
            dp = [[a.cadd(x, a.cneg(a.cscale(y, self.complement))) for x, y in zip(pr, ar)]
                  for pr, ar in zip(self.p[variant], self.spectral)]
            d = [[a.cadd(x, y) for x, y in zip(nr, pr)] for nr, pr in zip(dn, dp)]
            for index, value in enumerate((dn, dp, d), start):
                self.norm(index, value)
        self.sl = self.sg = self.gr = None

    def side(self, side, sl, sg):
        a = self.a
        sn, sp = self.rotate(a.matrix(sl), False), self.rotate(a.matrix(sg), True)
        gamma = [[a.cadd(x, y) for x, y in zip(nr, pr)] for nr, pr in zip(sn, sp)]
        start = 7 if side == 'plus' else 10
        for offset, coefficient in enumerate((self.complement, self.f)):
            self.norm(start+offset, [[a.cscale(x, coefficient) for x in row] for row in gamma])
        self.norm(start+2, [[a.cadd(x, a.cneg(a.cscale(y, self.f))) for x, y in zip(nr, gr)]
                            for nr, gr in zip(sn, gamma)])
        for variant in ('raw', 'normalized'):
            total = self.traces[(side, variant)]
            for i in range(self.nb):
                for j in range(self.nb):
                    value = a.cadd(a.cmul(sp[i][j], self.n[variant][j][i]),
                                   a.cneg(a.cmul(sn[i][j], self.p[variant][j][i])))
                    total = a.cadd(total, a.cscale(value, self.u))
            self.traces[(side, variant)] = total

    def row(self, side, variant, gs, j0, current_float, rhs_float):
        a = self.a
        pl, ph = a.pi()
        a.guard(abs(gs).bit_length()+1)
        scale = a.mul(a.number(j0), F(gs))
        kl, kh = a.div(scale, a.mul(F(2), ph)), a.div(scale, a.mul(F(2), pl))
        def scaled_interval(x):
            if a.compare(x, F(0)) >= 0:
                return a.mul(kl, x), a.mul(kh, x)
            return a.mul(kh, x), a.mul(kl, x)
        real, imag = (scaled_interval(x) for x in self.traces[(side, variant)])
        start, s = (1 if variant == 'raw' else 4), (7 if side == 'plus' else 10)
        low, high = F(0), F(0)
        for i, j in ((s, start), (s+1, start+1), (s+2, 0), (s+2, start+2)):
            l, h = a.sqrt(a.mul(self.sums[i], self.sums[j]))
            low, high = a.add(low, l), a.add(high, h)
        rhs = a.mul(kl, low), a.mul(kh, high)
        def radius(interval):
            return a.maximum(a.absolute(interval[0]), a.absolute(interval[1]))
        def distance(point, interval):
            p = a.number(point)
            return a.maximum(a.absolute(a.sub(p, interval[0])), a.absolute(a.sub(p, interval[1])))
        def hypot_upper(x, y):
            return a.sqrt(a.add(a.mul(x, x), a.mul(y, y)))[1]
        error = hypot_upper(distance(current_float.real, real), distance(current_float.imag, imag))
        rhs_error = distance(rhs_float, rhs)
        common_real = real if side == 'plus' else (a.neg(real[1]), a.neg(real[0]))
        common_imag = imag if side == 'plus' else (a.neg(imag[1]), a.neg(imag[0]))
        rhs_report = a.interval(*rhs)
        error_upper = a.outward(error, True)
        float_upper = a.outward(a.add(a.number(rhs_report['upper']), a.number(error_upper)), True)
        return dict(status='enclosed', reason=None, units='A/m^2',
            current_real=a.interval(*real), current_imag=a.interval(*imag), rhs=rhs_report,
            current_common_z_real=a.interval(*common_real), current_common_z_imag=a.interval(*common_imag),
            current_abs_upper=a.outward(hypot_upper(radius(real), radius(imag)), True),
            current_float_error_upper=error_upper, rhs_float_error_upper=a.outward(rhs_error, True),
            current_float_abs_upper=float_upper)

    def budgets(self):
        return dict(exact_operation_forecast=self.forecast, actual_exact_operations=self.a.operations,
                    peak_integer_bits=self.a.peak_bits, live_rational_slots=self.slots,
                    transient_workspace_forecast_bytes=self.a.peak_workspace,
                    scope='conservative object/digit census; not measured process RSS')


def _report(engine, status, reason, mu, gs, j0, currents, bound):
    result = dict(contract='SCI06-represented-arithmetic-v1', status=status, reason=reason,
        scope='exact stored binary64 formula at represented Float64 f; no acceptance',
        f_semantics='exact dyadic of existing _fd output; exact complement',
        reference_mu='mu.value_relative_eV', f_binary64_sha256=None, f_count=None,
        pi_method='Machin alternating 64+64 rational terms', sqrt_bits=128, rows={},
        budgets=engine.budgets() if engine is not None else {})
    for side in ('plus', 'minus'):
        result['rows'][side] = {}
        for variant in ('raw', 'normalized'):
            result['rows'][side][variant] = dict(status=status, reason=reason, units='A/m^2',
                current_real=None, current_imag=None, current_abs_upper=None, rhs=None,
                current_common_z_real=None, current_common_z_imag=None,
                current_float_error_upper=None, rhs_float_error_upper=None, current_float_abs_upper=None)
    if mu['value_relative_eV'] is None:
        result['status'], result['reason'] = mu['status'], mu['reason']
        for rows in result['rows'].values():
            for row in rows.values():
                row.update(status=mu['status'], reason=mu['reason'])
        return result
    if engine is not None and engine.f_hash is not None:
        result.update(f_binary64_sha256=engine.f_hash, f_count=engine.f_count)
    if status != 'enclosed':
        return result
    try:
        rows = {}
        for side in ('plus', 'minus'):
            rows[side] = {}
            for variant in ('raw', 'normalized'):
                point = complex(currents[side][variant+'_outward_A_per_m2'], currents[side][variant+'_imaginary_A_per_m2'])
                rows[side][variant] = engine.row(side, variant, gs, j0, point, bound['bounds'][side][variant]['value'])
        result['rows'] = rows
    except (_Budget, _Invalid) as error:
        result['status'] = 'budget_exceeded' if isinstance(error, _Budget) else 'invalid_data'
        result['reason'] = str(error)
        for values in result['rows'].values():
            for row in values.values():
                row.update(status=result['status'], reason=result['reason'])
    result['budgets'] = engine.budgets()
    return result
