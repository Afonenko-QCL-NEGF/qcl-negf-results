"""Bounded native dataset access shared by validation and lossless history proofs."""
from __future__ import annotations

from collections.abc import Iterator
import itertools
from typing import Any

from qcl_negf_contracts.artifacts import ARTIFACT_SCHEMA_CONTRACTS, CONTRACT_SET, NATIVE_SCHEMA_VERSION


def dataset_blocks(dataset: Any, maximum_bytes: int = 1024 * 1024) -> Iterator[tuple[slice, ...]]:
    """Cover a nonempty, nonscalar dataset without allocating a wide full row."""
    block = [1] * dataset.ndim
    remaining = max(1, maximum_bytes // max(1, dataset.dtype.itemsize))
    for axis in reversed(range(dataset.ndim)):
        block[axis] = min(dataset.shape[axis], remaining)
        remaining = max(1, remaining // block[axis])
    for starts in itertools.product(*(range(0, size, step) for size, step in zip(dataset.shape, block))):
        yield tuple(slice(start, min(start + step, size))
                    for start, step, size in zip(starts, block, dataset.shape))


HISTORY_SCHEMA = "qcl-negf-scientific-history-v4"
HDF5_SCHEMAS = {role: schemas for (role, media), schemas in ARTIFACT_SCHEMA_CONTRACTS.items()
                if media == "application/x-hdf5"}
MARKER_FIELDS = {
    "seed_mu_eV": "f", "seed_number_ratio": "f",
    "fdt_raw_seed_mu": "f", "fdt_normalized_seed_mu": "f",
    "measured_iteration": "i", "raw_hole_charge": "f", "represented_capacity": "f",
    "occupied_fraction_a": "f", "empty_fraction_c": "f", "relative_correction_Gn": "f",
    "relative_correction_Gp": "f", "equilibrium_applicable": "i", "equilibrium_status": "s",
    "equilibrium_mu_eV": "f", "fdt_raw": "f", "fdt_normalized": "f",
    "equilibrium_abs_current_A_m2": "f", "delta_energy_eV": "f",
    "sampled_gamma_over_dE_q10": "f", "sampled_gamma_over_dE_q50": "f",
    "sampled_gamma_over_dE_q90": "f", "sampled_spectral_weight_underresolved": "f",
    "linewidth_sampled_blocks": "i", "linewidth_status": "s", "linewidth_sampling_method": "s",
    "marker_cadence": "i", "linewidth_max_blocks": "i", "relative_mode_weight_floor": "f",
    "fresh_map_status": "s", "lo_shift_over_dE": "f", "field_shift_over_dE": "f",
    "lo_boundary_occupied_fraction": "f", "lo_boundary_spectral_fraction": "f",
    "field_boundary_occupied_fraction": "f", "field_boundary_spectral_fraction": "f",
}
MARKER_CHILD_FIELDS = {
    "channels": {"sequence": "i", "channel": "s", "component": "s",
                 "residual_absolute": "f", "residual_scale": "f", "residual_relative": "f"},
    "collisions": {"sequence": "i", "channel": "s", "state_kind": "s",
                   **{name: "f" for name in ("particle_signed", "particle_absolute", "particle_imaginary",
                                              "energy_signed", "energy_absolute", "energy_imaginary")}},
}


def _text(value: Any) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def _required_group(parent: Any, name: str, schema: str) -> Any:
    import h5py
    if name not in parent or not isinstance(parent[name], h5py.Group):
        raise ValueError(f"native format 4.0 requires {name}")
    group = parent[name]
    if _text(group.attrs.get("schema", "")) != schema:
        raise ValueError(f"native format 4.0 requires {schema}")
    return group


def _marker_columns(table: Any, count: int) -> None:
    import h5py
    if not {"sequence", "available", *MARKER_FIELDS} <= set(table):
        raise ValueError("native format 4.0 requires complete physical marker columns")
    for name, column in table.items():
        if name in MARKER_CHILD_FIELDS:
            continue
        if not isinstance(column, h5py.Dataset) or column.shape != (count,):
            raise ValueError("physical marker columns must align with every SCBA row")
        kind = MARKER_FIELDS.get(name, "i")
        width = 1 if name in {"available", "equilibrium_applicable"} else 8
        valid = (h5py.check_string_dtype(column.dtype) is not None if kind == "s"
                 else column.dtype.kind == kind and column.dtype.itemsize == width)
        if not valid:
            raise ValueError(f"physical marker {name} has an incompatible native dtype")
    _marker_children(table)


def _marker_children(table: Any) -> None:
    import h5py
    for name, fields in MARKER_CHILD_FIELDS.items():
        child = _required_group(table, name, f"qcl-negf-physical-marker-{name}-v1")
        if set(child) != set(fields):
            raise ValueError(f"physical marker {name} requires exact typed columns")
        rows = len(child["sequence"])
        for field, kind in fields.items():
            column = child[field]
            if not isinstance(column, h5py.Dataset) or column.shape != (rows,):
                raise ValueError("physical marker child columns differ in length")
            valid = (h5py.check_string_dtype(column.dtype) is not None if kind == "s"
                     else column.dtype.kind == kind and column.dtype.itemsize == 8)
            if not valid:
                raise ValueError(f"physical marker {name}/{field} has an incompatible native dtype")
        _marker_child_coordinates(table, child, name)


def _marker_child_coordinates(parent: Any, child: Any, name: str) -> None:
    import numpy as np
    coordinates = parent["sequence"][:]
    available = parent["available"][:]
    status = np.asarray([_text(value) for value in parent["fresh_map_status"][:]])
    measured = set(coordinates[(available == 1) & (status == "available")].tolist())
    field = "component" if name == "channels" else "state_kind"
    allowed = {"retarded", "lesser", "greater"} if name == "channels" else {"fresh", "mixed"}
    identities = set()
    for sequence, channel, state in zip(child["sequence"][:], child["channel"][:], child[field][:]):
        identity = (int(sequence), _text(channel), _text(state))
        if identity[0] not in measured or identity[2] not in allowed or not identity[1] or identity in identities:
            raise ValueError("physical marker child identity is duplicated, unavailable or outside its measured state")
        identities.add(identity)


def validate_scba_diagnostics(parent: Any, count: int) -> None:
    """Required v3 markers/PSD tables exist even for an empty SCBA history."""
    import h5py
    from .witnesses import PACKED_SCHEMA, selected_witness_records
    marker = _required_group(parent, "physical_markers", "qcl-negf-physical-markers-v3")
    _marker_columns(marker, count)
    psd = _required_group(parent, "psd_history", "qcl-negf-psd-history-v2")
    if not {"sequence", "available", "matrix_kind", "selected_blocks"} <= set(psd):
        raise ValueError("native format 4.0 requires complete PSD coordinates")
    for name, column in psd.items():
        if name != "selected_blocks" and (not isinstance(column, h5py.Dataset) or column.shape != (count,)):
            raise ValueError("PSD columns must align with every SCBA row")
    selected = _required_group(psd, "selected_blocks", PACKED_SCHEMA)
    selected_witness_records(selected)
    _diagnostic_coordinates(parent, marker, psd, count)


def _diagnostic_coordinates(parent: Any, marker: Any, psd: Any, count: int) -> None:
    import h5py
    import numpy as np
    scba = parent["scba"]
    sequence_column = scba.get("sequence") if isinstance(scba, h5py.Group) else None
    previous = 0
    for start in range(0, count, 131_072):
        stop = min(start + 131_072, count)
        sequence = marker["sequence"][start:stop]
        expected = sequence_column[start:stop] if sequence_column is not None else np.arange(start + 1, stop + 1)
        if not np.array_equal(sequence, psd["sequence"][start:stop]) or not np.array_equal(sequence, expected):
            raise ValueError("physical marker and PSD sequences disagree with SCBA history")
        if sequence[0] <= previous or np.any(sequence[1:] <= sequence[:-1]):
            raise ValueError("native diagnostic sequences must be positive and strictly increasing")
        previous = int(sequence[-1])
        available = marker["available"][start:stop]
        if np.any(~np.isin(available, (0, 1))) or np.any(~np.isin(psd["available"][start:stop], (0, 1))):
            raise ValueError("native diagnostic availability must be zero or one")
        applicable = marker["equilibrium_applicable"][start:stop][available == 1]
        if np.any(~np.isin(applicable, (0, 1))):
            raise ValueError("equilibrium applicability must be zero or one when measured")


def _scba_count(parent: Any) -> int:
    import h5py
    if "scba" not in parent:
        raise ValueError("native format 4.0 requires an explicit SCBA history")
    scba = parent["scba"]
    if isinstance(scba, h5py.Dataset):
        if scba.ndim != 2 or scba.shape[1] != 18:
            raise ValueError("checkpoint SCBA history must have exactly 18 native columns")
        return int(scba.shape[0])
    if isinstance(scba, h5py.Group):
        if "sequence" in scba:
            return len(scba["sequence"])
        if "row_count" in scba.attrs:
            return int(scba.attrs["row_count"])
    raise ValueError("native SCBA history has no row count")


def validate_native_handle(handle: Any, role: str | None = None,
                           declared_schema: str | None = None) -> str:
    """Accept only the current format; no legacy layout, version or marker fallback."""
    import h5py
    if "metadata" not in handle or not isinstance(handle["metadata"], h5py.Group):
        raise ValueError("native format 4.0 requires metadata")
    metadata = handle["metadata"]
    if _text(metadata.attrs.get("contract_set", "")) != CONTRACT_SET:
        raise ValueError(f"unsupported native contract set: requires {CONTRACT_SET}")
    actual_role = _text(metadata.attrs.get("artifact_role", ""))
    schema = _text(metadata.attrs.get("schema", ""))
    if _text(metadata.attrs.get("schema_version", "")) != NATIVE_SCHEMA_VERSION:
        raise ValueError("unsupported native format: requires schema_version 4.0")
    if schema not in HDF5_SCHEMAS.get(actual_role, frozenset()) or (role is not None and actual_role != role):
        raise ValueError("native schema does not match its declared scientific role")
    if declared_schema is not None and schema != declared_schema:
        raise ValueError("native schema differs from the committed artifact schema")
    parents = {HISTORY_SCHEMA: "", "qcl-negf-physics-analysis-v4": "diagnostics",
               "qcl-negf-checkpoint-v4": "convergence", "qcl-negf-physics-v4": "convergence"}
    if schema in parents:
        path = parents[schema]
        if path and path not in handle:
            raise ValueError(f"native format 4.0 requires {path}")
        parent = handle[path] if path else handle
        validate_scba_diagnostics(parent, _scba_count(parent))
    return schema


_FINAL_AUDIT_SCHEMA = "qcl-negf-final-fixed-point-audit-v1"
_FINAL_AUDIT_PATH = "diagnostics/final_fixed_point_audit"


def _audit_error(message: str) -> None:
    from qcl_negf_contracts.messages import ContractError
    raise ContractError(f"final audit native layout: {message}", "corrupt_result")


def _audit_attribute(parent: Any, name: str, expected: str) -> None:
    """Profile A for the four new constants, on the caller's budgeted handle."""
    from contextlib import ExitStack
    import h5py
    with ExitStack() as ids:
        try:
            attribute = parent.attrs.get_id(name)
        except KeyError:
            _audit_error(f"missing {name} attribute")
        ids.callback(attribute.close)
        datatype = attribute.get_type()
        ids.callback(datatype.close)
        space = attribute.get_space()
        ids.callback(space.close)
        # Do not ask an arbitrary file type for a NumPy dtype before these
        # checks: a vlen pointer size is not a bound on its heap payload.
        if (datatype.get_class() != h5py.h5t.STRING or datatype.is_variable_str()
                or space.get_simple_extent_type() != h5py.h5s.SCALAR
                or space.get_simple_extent_dims() != ()):
            _audit_error(f"{name} requires an atomic fixed-string scalar")
        width = datatype.get_size()
        charset, padding = datatype.get_cset(), datatype.get_strpad()
        if (not len(expected.encode("utf-8")) <= width <= 65536
                or charset not in (h5py.h5t.CSET_ASCII, h5py.h5t.CSET_UTF8)
                or padding not in (h5py.h5t.STR_NULLTERM, h5py.h5t.STR_NULLPAD,
                                   h5py.h5t.STR_SPACEPAD)):
            _audit_error(f"{name} has unsupported width, charset or padding")
        raw = _read_audit_attribute_bytes(attribute, datatype, width)
        raw = raw.rstrip(b" " if padding == h5py.h5t.STR_SPACEPAD else b"\0")
        if b"\0" in raw:
            _audit_error(f"{name} has an embedded NUL or hidden suffix")
        try:
            value = raw.decode("ascii" if charset == h5py.h5t.CSET_ASCII else "utf-8", "strict")
        except UnicodeError:
            _audit_error(f"{name} has invalid string bytes")
        if value != expected:
            _audit_error(f"{name} attribute differs from the D2 contract")


def _read_audit_attribute_bytes(attribute: Any, datatype: Any, width: int) -> bytes:
    import numpy as np
    target = np.empty((), dtype=np.dtype(f"S{width}"))
    attribute.read(target, mtype=datatype)
    return target.tobytes()


def _read_audit_scalar_bytes(dataset: Any, datatype: Any) -> bytes:
    import h5py
    import numpy as np
    target = np.empty((), dtype=dataset.dtype)
    # The identical memory datatype prevents HDF5 string conversion from
    # truncating an embedded NUL or hiding invalid bytes behind NULLTERM.
    dataset.id.read(h5py.h5s.ALL, h5py.h5s.ALL, target, mtype=datatype)
    return target.tobytes()


def _read_final_fixed_point_audit(handle: Any) -> bytes | None:
    """Private fixed-width scalar read; caller owns hashes and stream budget.

    This performs no numerical measurement or scientific acceptance evaluation.
    The supplied handle must use the caller's existing physical-I/O stream.
    """
    import h5py
    from qcl_negf_contracts.final_audit import MAX_FINAL_AUDIT_BYTES
    parent = handle
    for name in _FINAL_AUDIT_PATH.split("/"):
        if name not in parent:
            return None
        if not isinstance(parent.get(name, getlink=True), h5py.HardLink):
            _audit_error("record path must use internal hard links")
        parent = parent[name]
        if not isinstance(parent, h5py.Group):
            _audit_error("record path must contain groups")
    if len(parent) != 1 or "record_json" not in parent:
        _audit_error("record group requires exactly record_json")
    if not isinstance(parent.get("record_json", getlink=True), h5py.HardLink):
        _audit_error("record dataset must use an internal hard link")
    dataset = parent["record_json"]
    if (not isinstance(dataset, h5py.Dataset) or dataset.shape != ()
            or dataset.dtype.kind != "S" or dataset.dtype.hasobject
            or not 1 <= dataset.dtype.itemsize <= MAX_FINAL_AUDIT_BYTES):
        _audit_error("record_json must be a fixed string scalar of 1..65536 bytes")
    datatype = dataset.id.get_type()
    creation = dataset.id.get_create_plist()
    if (datatype.get_class() != h5py.h5t.STRING or datatype.is_variable_str()
            or datatype.get_cset() != h5py.h5t.CSET_UTF8
            or dataset.chunks is not None or dataset.compression is not None
            or creation.get_nfilters() or dataset.external or dataset.is_virtual):
        _audit_error("record_json requires fixed UTF-8 without chunks, filters or external storage")
    _audit_attribute(parent, "schema", _FINAL_AUDIT_SCHEMA)
    for name, expected in (
            ("units", "1"), ("logical_axis_order", "scalar"),
            ("description", "Versioned final fixed-point raw measurement record")):
        _audit_attribute(dataset, name, expected)
    # Preserve every storage byte. NumPy's scalar bytes conversion otherwise
    # drops trailing NULs; reading into the declared dtype keeps padding explicit.
    raw = _read_audit_scalar_bytes(dataset, datatype)
    padding = datatype.get_strpad()
    if padding == h5py.h5t.STR_SPACEPAD:
        raw = raw.rstrip(b" ")
    elif padding in (h5py.h5t.STR_NULLPAD, h5py.h5t.STR_NULLTERM):
        raw = raw.rstrip(b"\0")
    else:
        _audit_error("unsupported fixed-string padding")
    if not raw or b"\0" in raw:
        _audit_error("empty record or embedded NUL")
    return raw
