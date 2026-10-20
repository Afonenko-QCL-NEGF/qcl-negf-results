"""SCI12 MATH-I1-DYADIC: literal oracle, not a scientific acceptance test.

Expected values are written independently from MAP D2 §9 / OPS §2.
No irrational enclosure, projector, generic relation norm or native IO claim.
Missing future API fails individual tests, never module collection or xfail.
"""

from collections.abc import Sequence
from importlib import import_module, util
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE = "qcl_negf_results._equilibrium_map_math"
HELPERS = (
    "_hat_area", "_annulus_square_area", "_overlap_cell", "_restrict_cells",
    "_complex_matmul", "_gram", "_density_from_integrated", "_bond_from_integrated",
)


class _CallPhaseAPI:
    """Delay only API-presence assertions until test attribute lookup."""

    def __init__(self, candidate):
        self._candidate = candidate

    def __getattr__(self, name):
        assert self._candidate is not None, (
            "NEW_API_MISSING: _equilibrium_map_math; no behavioral verdict"
        )
        actual = getattr(self._candidate, name, None)
        assert name in HELPERS and callable(actual), f"NEW_API_MISSING: {name}"
        return actual


@pytest.fixture
def api():
    # Check the actual selected package before resolving/calling candidate code.
    package = import_module("qcl_negf_results")
    assert Path(package.__file__).resolve() == ROOT / "src/qcl_negf_results/__init__.py"
    spec = util.find_spec(MODULE)
    if spec is None:
        return _CallPhaseAPI(None)
    expected = ROOT / "src/qcl_negf_results/_equilibrium_map_math.py"
    assert spec.origin is not None and Path(spec.origin).resolve() == expected
    candidate = import_module(MODULE)  # Dependency/import errors remain real errors.
    assert Path(candidate.__file__).resolve() == expected
    return _CallPhaseAPI(candidate)


@pytest.fixture
def contract_error():
    return import_module("qcl_negf_contracts.messages").ContractError


def plane(real, imag):
    """Only manufacture inputs; never calculate an expected product."""
    return np.array(real, dtype=np.float64), np.array(imag, dtype=np.float64)


def output(rows, columns):
    return np.full((rows, columns), -999.0), np.full((rows, columns), -999.0)


def assert_plane(actual, real, imag):
    np.testing.assert_array_equal(actual[0], real)
    np.testing.assert_array_equal(actual[1], imag)


# Independent healthy controls still register when the future module is absent.
def test_literal_normalization_oracle_control():
    assert (0.25 * (8.0 + 4.0), 2.0 + 1.0) == (3.0, 3.0)
    assert -(3.0 / 8.0) * (2.0 / (1.0 + 3.0)) / (0.25 * 0.25) == -3.0
    assert 2.0 * (-1.5) == -3.0
    assert 0.25 * (2.0 + 1.0) == 0.75


def test_literal_signed_and_partition_oracle_control():
    assert 0.75 + 3.0 + 2.25 == 6.0
    assert 0.5 * 1.0 + 0.5 * -1.0 == 0.0
    assert 0.5 * abs(1.0) + 0.5 * abs(-1.0) == 1.0


@pytest.mark.parametrize("args,want", [
    ((-1.0, 0.0, -1.0, 0.0, "left"), 0.5),
    ((-1.0, 0.0, -0.5, 0.0, "left"), 0.125),
    ((-1.0, 0.0, -0.5, 0.0, "right"), 0.375),
    ((0.0, 1.0, 0.0, 0.5, "left"), 0.375),
    ((0.0, 1.0, 0.0, 0.5, "right"), 0.125),
    ((0.0, 1.0, 1.0, 2.0, "left"), 0.0),
    ((0.0, 1.0, 0.5, 0.5, "right"), 0.0),
    ((0.0, 1.0, -1.0, 2.0, "right"), 0.5),
])
def test_hat_clips_without_extrapolating(api, args, want):
    assert api._hat_area(*args) == want


def test_hat_signed_integral_and_partition(api):
    # q=[1,3,5] at [-1,0,1]. Coefficients, not interpolated products.
    low = api._hat_area(-1.0, 0.0, -1.0, -0.5, "left")
    low += 3.0 * api._hat_area(-1.0, 0.0, -1.0, -0.5, "right")
    common = api._hat_area(-1.0, 0.0, -0.5, 0.0, "left")
    common += 3.0 * api._hat_area(-1.0, 0.0, -0.5, 0.0, "right")
    common += 3.0 * api._hat_area(0.0, 1.0, 0.0, 0.5, "left")
    common += 5.0 * api._hat_area(0.0, 1.0, 0.0, 0.5, "right")
    high = 3.0 * api._hat_area(0.0, 1.0, 0.5, 1.0, "left")
    high += 5.0 * api._hat_area(0.0, 1.0, 0.5, 1.0, "right")
    assert (low, common, high) == (0.75, 3.0, 2.25)
    assert low + common + high == 6.0
    assert api._hat_area(-1.0, 1.0, -1.0, 1.0, "left") - api._hat_area(
        -1.0, 1.0, -1.0, 1.0, "right"
    ) == 0.0


@pytest.mark.parametrize("args,want", [
    ((0.0, 0.5, 0.0, 2.0), 0.25),
    ((0.5, 1.5, 0.0, 2.0), 2.0),
    ((1.5, 2.0, 0.0, 2.0), 1.75),
    ((0.0, 0.5, 0.75, 1.75), 0.0),
    ((0.5, 1.5, 0.75, 1.75), 1.6875),
    ((1.5, 2.0, 0.75, 1.75), 0.8125),
    ((0.5, 1.5, 1.0, 1.0), 0.0),
])
def test_annulus_squared_measure_has_no_extra_pi(api, args, want):
    assert api._annulus_square_area(*args) == want


@pytest.mark.parametrize("args,want", [
    ((0.0, 0.5, 0.25, 1.0), 0.25),
    ((0.0, 0.5, 0.5, 1.0), 0.0),
    ((0.0, 0.5, 0.0, 0.5), 0.5),
    ((0.0, 0.5, 1.0, 2.0), 0.0),
])
def test_overlap_does_not_drop_partial_cells(api, args, want):
    assert api._overlap_cell(*args) == want


@pytest.mark.parametrize("values,want", [
    ([1.0, 3.0, 2.0, 6.0], [2.0, 4.0]),
    ([-3.0, -1.0, 1.0, 3.0], [-2.0, 2.0]),
    ([1.0, -1.0, 2.0, -2.0], [0.0, 0.0]),
    ([0.0, 0.0, 0.0, 0.0], [0.0, 0.0]),
])
def test_restriction_preserves_signed_cell_integrals(api, values, want):
    result = api._restrict_cells([0.0, 0.5, 1.0, 1.5, 2.0], [0.0, 1.0, 2.0], values)
    assert isinstance(result, list)
    assert result == want


def test_restriction_sheet_literal(api):
    result = api._restrict_cells([0.0, 0.5, 1.0, 1.5, 2.0], [0.0, 1.0, 2.0], [1.0, 3.0, 2.0, 6.0])
    assert result == [2.0, 4.0]
    assert result[0] + result[1] == 6.0


@pytest.mark.parametrize("source,target,values", [
    ([0.0, 0.5, 1.0], [0.25, 1.25], [1.0, 3.0]),
    ([0.0, 0.5, 1.0], [0.0, 2.0], [1.0, 3.0]),
    ([0.0, 0.0, 1.0], [0.0, 1.0], [1.0, 3.0]),
    ([0.0, 0.25, 1.0], [0.0, 1.0], [1.0, 3.0]),
    ([], [0.0, 1.0], []),
    ([0.0, 1.0], [0.0, 1.0], []),
])
def test_restriction_refuses_changed_domain_or_missing_cells(api, contract_error, source, target, values):
    with pytest.raises(contract_error):
        api._restrict_cells(source, target, values)


def test_complex_product_keeps_offdiagonal_imaginary_and_order(api):
    a = plane([[1.0, 0.0], [0.0, 2.0]], [[0.0, 1.0], [0.0, 0.0]])
    b = plane([[2.0, 0.0], [0.0, 1.0]], [[0.0, 0.0], [1.0, 0.0]])
    out = output(2, 2)
    assert api._complex_matmul(a, b, out) is None
    assert_plane(out, [[1.0, 0.0], [0.0, 2.0]], [[0.0, 1.0], [2.0, 0.0]])
    assert_plane(a, [[1.0, 0.0], [0.0, 2.0]], [[0.0, 1.0], [0.0, 0.0]])
    assert_plane(b, [[2.0, 0.0], [0.0, 1.0]], [[0.0, 0.0], [1.0, 0.0]])


def test_gram_conjugates_left_operand(api):
    a = plane([[1.0, 0.0], [0.0, 2.0]], [[0.0, 1.0], [0.0, 0.0]])
    out = output(2, 2)
    assert api._gram(a, a, out) is None
    assert_plane(out, [[1.0, 0.0], [0.0, 5.0]], [[0.0, 1.0], [-1.0, 0.0]])


def test_gram_discrepancy_is_not_renormalized(api):
    f = plane([[1.0, 0.0], [0.0, 2.0]], [[0.0, 0.0], [0.0, 0.0]])
    out = output(2, 2)
    api._gram(f, f, out)
    assert_plane(out, [[1.0, 0.0], [0.0, 4.0]], [[0.0, 0.0], [0.0, 0.0]])
    assert out[0][0, 0] + out[0][1, 1] == 5.0  # native trace(I2)=2, defect3


def test_rectangular_gram_keeps_missing_third_subspace(api):
    fa = plane([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
    fb = plane([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    out = output(2, 3)
    api._gram(fa, fb, out)
    assert_plane(out, [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    density = np.full(3, -999.0)
    m = plane([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    api._density_from_integrated(fb, m, 1.0, density)
    np.testing.assert_array_equal(density, [1.0, 2.0, 3.0])
    assert density[2] == 3.0  # retained missing sheet, no projector/inverse claim


@pytest.mark.parametrize("basis,mreal,mimag,want_n,want_j,dx,cbar", [
    ("identity", [[2.0, 0.0], [0.0, 1.0]], [[0.0, 0.5], [-0.5, 0.0]], [2.0, 1.0], -3.0, 1.0, 6.0),
    ("identity", [[2.0, 0.0], [0.0, 1.0]], [[0.0, 0.0], [0.0, 0.0]], [2.0, 1.0], 0.0, 1.0, 6.0),
    ("identity", [[2.0, 0.0], [0.0, 1.0]], [[0.0, -0.5], [0.5, 0.0]], [2.0, 1.0], 3.0, 1.0, 6.0),
    ("unitary", [[1.0, -0.5], [-0.5, 2.0]], [[0.0, 0.0], [0.0, 0.0]], [2.0, 1.0], -3.0, 1.0, 6.0),
    ("identity", [[0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]], [0.0, 0.0], 0.0, 0.25, 0.375),
])
def test_density_bond_coherence_conjugation_and_unitary(api, basis, mreal, mimag, want_n, want_j, dx, cbar):
    phi = (plane([[1.0, 0.0], [0.0, 1.0]], [[0.0, 0.0], [0.0, 0.0]])
           if basis == "identity" else plane([[0.0, 1.0], [0.0, 0.0]], [[0.0, 0.0], [1.0, 0.0]]))
    m = plane(mreal, mimag)
    density, bond = np.full(2, -999.0), np.full(1, -999.0)
    assert api._density_from_integrated(phi, m, 1.0, density) is None
    assert api._bond_from_integrated(phi, m, np.array([1.0, 3.0]), dx, cbar, 1.0, bond) is None
    np.testing.assert_array_equal(density, want_n)
    np.testing.assert_array_equal(bond, [want_j])


def test_unequal_dx_uses_actual_chi_for_density_and_phi_for_bond(api):
    phi = plane([[1.0, 0.0], [0.0, 1.0]], [[0.0, 0.0], [0.0, 0.0]])
    chi = plane([[2.0, 0.0], [0.0, 2.0]], [[0.0, 0.0], [0.0, 0.0]])
    m = plane([[2.0, 0.0], [0.0, 1.0]], [[0.0, 0.5], [-0.5, 0.0]])
    n, wrong_n, j, wrong_j = np.full(2, -999.0), np.full(2, -999.0), np.full(1, -999.0), np.full(1, -999.0)
    api._density_from_integrated(chi, m, 1.0, n)
    api._density_from_integrated(phi, m, 1.0, wrong_n)
    api._bond_from_integrated(phi, m, np.array([1.0, 3.0]), 0.25, 0.375, 1.0, j)
    api._bond_from_integrated(chi, m, np.array([1.0, 3.0]), 0.25, 0.375, 1.0, wrong_j)
    np.testing.assert_array_equal(n, [8.0, 4.0])
    np.testing.assert_array_equal(wrong_n, [2.0, 1.0])
    np.testing.assert_array_equal(j, [-3.0])
    np.testing.assert_array_equal(wrong_j, [-12.0])
    assert 0.25 * (n[0] + n[1]) == 3.0
    assert 0.25 * (wrong_n[0] + wrong_n[1]) == 0.75


@pytest.mark.parametrize("name,args", [
    ("_hat_area", (0.0, 0.0, 0.0, 1.0, "left")),
    ("_hat_area", (True, 1.0, 0.0, 1.0, "left")),
    ("_hat_area", (0.0, 1.0, 0.0, 1.0, "middle")),
    ("_hat_area", (0.0, 1.0, 1.0, 0.0, "left")),
    ("_hat_area", (0.0, float("inf"), 0.0, 1.0, "left")),
    ("_hat_area", (0.0, 1.0, float("nan"), 1.0, "left")),
    ("_annulus_square_area", (-1.0, 1.0, 0.0, 1.0)),
    ("_annulus_square_area", (0.0, 1.0, 1.0, 0.0)),
    ("_annulus_square_area", (0.0, 1e308, 0.0, 1e308)),
    ("_annulus_square_area", (False, 1.0, 0.0, 1.0)),
    ("_overlap_cell", (0.0, 0.0, 0.0, 1.0)),
    ("_overlap_cell", (0.0, 1.0, 0.0, 0.0)),
    ("_overlap_cell", (0.0, 1.0, None, 1.0)),
    ("_overlap_cell", (0.0, 1.0, "0", 1.0)),
])
def test_scalar_invalid_inputs_are_contract_errors(api, contract_error, name, args):
    with pytest.raises(contract_error):
        getattr(api, name)(*args)


@pytest.mark.parametrize("kind", ["bool", "integer", "object", "noncontiguous", "shape", "nonfinite", "empty", "list"])
def test_product_refuses_implicit_copy_and_invalid_planes(api, contract_error, kind):
    good = plane([[1.0, 0.0], [0.0, 1.0]], [[0.0, 0.0], [0.0, 0.0]])
    if kind in ("bool", "integer", "object"):
        dtype = {"bool": bool, "integer": np.int64, "object": object}[kind]
        bad = good[0].astype(dtype), good[1]
    elif kind == "noncontiguous":
        bad = good[0][:, ::-1], good[1]
    elif kind == "shape":
        bad = good[0], np.zeros((2, 3))
    elif kind == "nonfinite":
        bad = plane([[1.0, float("nan")], [0.0, 1.0]], [[0.0, 0.0], [0.0, 0.0]])
    elif kind == "empty":
        bad = np.empty((0, 0)), np.empty((0, 0))
    else:
        bad = [[1.0, 0.0], [0.0, 1.0]], [[0.0, 0.0], [0.0, 0.0]]
    with pytest.raises(contract_error):
        api._complex_matmul(bad, good, output(2, 2))


@pytest.mark.parametrize("kind", ["input_alias", "component_alias", "readonly", "wrong_shape"])
@pytest.mark.parametrize("name", ["_complex_matmul", "_gram"])
def test_matrix_outputs_reject_alias_and_invalid_storage(api, contract_error, kind, name):
    good = plane([[1.0, 0.0], [0.0, 1.0]], [[0.0, 0.0], [0.0, 0.0]])
    out = output(2, 2)
    if kind == "input_alias":
        out = good
    elif kind == "component_alias":
        out = out[0], out[0]
    elif kind == "readonly":
        out[0].flags.writeable = False
    else:
        out = output(2, 3)
    with pytest.raises(contract_error):
        getattr(api, name)(good, good, out)


@pytest.mark.parametrize("kind", ["zero_scale", "bool_scale", "negative_scale", "readonly", "alias", "shape", "overflow"])
def test_density_refuses_bad_scale_output_and_overflow(api, contract_error, kind):
    chi = plane([[1.0, 0.0], [0.0, 1.0]], [[0.0, 0.0], [0.0, 0.0]])
    m = plane([[2.0, 0.0], [0.0, 1.0]], [[0.0, 0.5], [-0.5, 0.0]])
    scale, out = 1.0, np.full(2, -999.0)
    if kind == "zero_scale":
        scale = 0.0
    elif kind == "bool_scale":
        scale = True
    elif kind == "negative_scale":
        scale = -1.0
    elif kind == "readonly":
        out.flags.writeable = False
    elif kind == "alias":
        out = chi[0].reshape(-1)[:2]
    elif kind == "shape":
        out = np.full(3, -999.0)
    else:
        chi[0][0, 0] = 1e308
    with pytest.raises(contract_error):
        api._density_from_integrated(chi, m, scale, out)


@pytest.mark.parametrize("kind", ["zero_dx", "negative_mass", "zero_mass", "mass_shape", "nonfinite", "bool_cbar", "readonly", "alias", "shape", "overflow"])
def test_bond_refuses_bad_physical_inputs_and_output(api, contract_error, kind):
    phi = plane([[1.0, 0.0], [0.0, 1.0]], [[0.0, 0.0], [0.0, 0.0]])
    m = plane([[2.0, 0.0], [0.0, 1.0]], [[0.0, 0.5], [-0.5, 0.0]])
    masses, dx, cbar, out = np.array([1.0, 3.0]), 0.25, 0.375, np.full(1, -999.0)
    if kind == "zero_dx":
        dx = 0.0
    elif kind == "negative_mass":
        masses[0] = -1.0
    elif kind == "zero_mass":
        masses[0] = 0.0
    elif kind == "mass_shape":
        masses = np.array([1.0])
    elif kind == "nonfinite":
        masses[0] = float("inf")
    elif kind == "bool_cbar":
        cbar = True
    elif kind == "readonly":
        out.flags.writeable = False
    elif kind == "alias":
        out = phi[0].reshape(-1)[:1]
    elif kind == "shape":
        out = np.full(2, -999.0)
    else:
        cbar = 1e308
    with pytest.raises(contract_error):
        api._bond_from_integrated(phi, m, masses, dx, cbar, 1.0, out)


def test_density_mixed_complex_row_uses_conjugation_and_scale(api):
    chi = plane([[1.0, 0.0], [1.0, 0.0]], [[0.0, 1.0], [0.0, -1.0]])
    m = plane([[2.0, 0.0], [0.0, 1.0]], [[0.0, 0.5], [-0.5, 0.0]])
    out = np.full(2, -999.0)
    api._density_from_integrated(chi, m, 2.0, out)
    np.testing.assert_array_equal(out, [0.5, 0.25])  # row forms4,2; L0^3=8


def test_gram_rank_deficiency_is_retained(api):
    f = plane([[1.0, 1.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]])
    out = output(2, 2)
    api._gram(f, f, out)
    assert_plane(out, [[1.0, 1.0], [1.0, 1.0]], [[0.0, 0.0], [0.0, 0.0]])


def test_product_three_by_three_zero_is_known_not_missing(api):
    zero = plane([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    out = output(3, 3)
    api._complex_matmul(zero, zero, out)
    assert_plane(out, [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])


def test_product_overflow_is_not_a_finite_measurement(api, contract_error):
    huge = plane([[1e308, 0.0], [0.0, 1.0]], [[0.0, 0.0], [0.0, 0.0]])
    with pytest.raises(contract_error):
        api._complex_matmul(huge, huge, output(2, 2))


def test_partial_annuli_keep_signed_weighted_numerator(api):
    result = api._annulus_square_area(0.0, 0.5, 0.75, 1.75)
    result += 2.0 * api._annulus_square_area(0.5, 1.5, 0.75, 1.75)
    result += 4.0 * api._annulus_square_area(1.5, 2.0, 0.75, 1.75)
    assert result == 6.625  # 53/8; caller's 1/(4*pi) deliberately absent


def test_partial_spatial_overlap_keeps_uncovered_strips(api):
    interior = api._overlap_cell(0.0, 0.5, 0.25, 1.75)
    interior += 3.0 * api._overlap_cell(0.5, 1.0, 0.25, 1.75)
    interior += 2.0 * api._overlap_cell(1.0, 1.5, 0.25, 1.75)
    interior += 6.0 * api._overlap_cell(1.5, 2.0, 0.25, 1.75)
    assert interior == 4.25
    assert api._overlap_cell(0.0, 0.5, 0.0, 0.25) == 0.25
    assert 6.0 * api._overlap_cell(1.5, 2.0, 1.75, 2.0) == 1.5


class _ReadForbiddenSequence(Sequence):
    """Legitimate bounded length metadata; any payload access is forbidden."""

    def __init__(self, length):
        self._length = length
        self.element_reads = 0

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        self.element_reads += 1
        raise AssertionError("COUNT_CAP_MUST_PRECEDE_ELEMENT_READ")


def test_restriction_cell_cap_rejects_lengths_before_element_reads(api, contract_error):
    source_edges = _ReadForbiddenSequence(194)
    values = _ReadForbiddenSequence(193)
    with pytest.raises(contract_error):
        api._restrict_cells(source_edges, [0.0, 1.0], values)
    assert source_edges.element_reads == 0
    assert values.element_reads == 0
