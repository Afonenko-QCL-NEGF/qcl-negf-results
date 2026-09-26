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
