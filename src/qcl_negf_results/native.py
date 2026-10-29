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


def _native_header(group, maximum_bytes, *, with_size=False):
    import h5py
    from qcl_negf_contracts.messages import ContractError
    result, consumed = {}, 0
    for name in ('schema','schema_version','contract_set','artifact_role','representation','core_layout_sha256'):
        if name not in group.attrs:
            raise ContractError('required native5 metadata attribute missing', 'corrupt_result')
        attribute = group.attrs.get_id(name)
        datatype, space = attribute.get_type(), attribute.get_space()
        try:
            if (datatype.get_class() != h5py.h5t.STRING or datatype.is_variable_str()
                    or datatype.get_size() > maximum_bytes or space.get_simple_extent_type() != h5py.h5s.SCALAR):
                raise ContractError('native5 header requires bounded fixed-width scalar strings', 'corrupt_result')
            consumed += datatype.get_size()
            if consumed > maximum_bytes:
                raise ContractError('aggregate native header exceeds metadata limit', 'budget_exceeded')
            try:
                result[name] = _text(group.attrs[name])
            except UnicodeError as error:
                raise ContractError('native5 header requires valid UTF-8', 'corrupt_result') from error
        finally:
            space.close()
            datatype.close()
            attribute.close()
    return (result, consumed) if with_size else result


def validate_native_handle(handle: Any, role: str | None = None,
                           declared_schema: str | None = None, *,
                           maximum_metadata_bytes: int = 16 * 1024 * 1024) -> str:
    """Validate only native5 metadata on an already verified immutable stream."""
    import h5py
    from qcl_negf_contracts.messages import ContractError
    def fail(message):
        raise ContractError(message, "corrupt_result")
    if "metadata" not in handle or not isinstance(handle["metadata"], h5py.Group):
        fail("native5 metadata missing")
    attrs = _native_header(handle['metadata'], maximum_metadata_bytes)
    if (_text(attrs.get("contract_set", "")) != CONTRACT_SET
            or _text(attrs.get("schema_version", "")) != "5.0"):
        raise ContractError("only result-v2/native5 is supported", "unsupported_format")
    actual_role, schema = _text(attrs.get("artifact_role", "")), _text(attrs.get("schema", ""))
    allowed = {"science.history": "qcl-negf-scientific-history-v5",
               "recovery": "qcl-negf-checkpoint-v5", "physics.full": "qcl-negf-physics-v5",
               "physics.analysis": "qcl-negf-physics-analysis-v5"}
    if allowed.get(actual_role) != schema or role is not None and role != actual_role:
        fail("native schema/role differs from owning declaration")
    if declared_schema is not None and schema != declared_schema:
        fail("native schema differs from FileRef")
    native_metadata(handle, maximum_bytes=maximum_metadata_bytes)
    return schema


def native_metadata(handle, *, identity=None, invocation=None, representation=None,
                    maximum_bytes=16 * 1024 * 1024):
    from importlib.resources import files
    import json
    import numpy as np
    from qcl_negf_contracts.messages import ContractError
    from .commits import decode_json
    def fail(message):
        raise ContractError(message, "corrupt_result")
    group = handle["metadata"]
    attrs = _native_header(group, maximum_bytes)
    if attrs['artifact_role'] == 'physics.analysis' and attrs['schema'] == 'qcl-negf-physics-analysis-v5':
        actual = attrs['representation']
        if actual not in ('final_diagnostics', 'final_observations') or representation not in (None, actual):
            fail('native representation differs from owning declaration')
        guard = final_diagnostics_metadata if actual == 'final_diagnostics' else final_observations_metadata
        return guard(handle, identity=identity, invocation=invocation, maximum_bytes=maximum_bytes)
    layout = json.loads(files("qcl_negf_contracts").joinpath("core_layout.json").read_text())["layout_sha256"]
    if attrs['core_layout_sha256'] != layout:
        raise ContractError("unsupported Core layout", "unsupported_format")
    actual = attrs['representation']
    if actual not in ("array_payload", "record_leaf") or representation is not None and actual != representation:
        fail("native representation differs from ordered catalog/declaration")
    result = dict(representation=actual)
    for name, expected in (("identity_json", identity), ("invocation_json", invocation)):
        if name not in group:
            fail("native source identity metadata missing")
        dataset = group[name]
        # Fixed width makes the allocation bound inspectable before reading.
        if dataset.shape != () or dataset.dtype.kind != "S" or dataset.dtype.itemsize > maximum_bytes:
            fail("native identity requires bounded fixed-width UTF-8 scalar")
        raw = bytes(dataset[()])
        value = decode_json(raw, name)
        if expected is not None and value != expected:
            fail("native source identity/epoch differs from selected reference")
        result[name] = value
    return result


def final_diagnostics_metadata(handle, *, identity, invocation, maximum_bytes):
    """Bound fixed UTF8 scalar metadata before any high-level string read."""
    import h5py
    import json
    from importlib.resources import files
    from qcl_negf_contracts.messages import ContractError
    from .commits import decode_json
    def fail(message, code='corrupt_result'):
        raise ContractError(message, code)
    attrs, consumed = _native_header(handle['metadata'], maximum_bytes, with_size=True)
    layout = json.loads(files('qcl_negf_contracts').joinpath('core_layout.json').read_text())['layout_sha256']
    if (attrs['schema'] != 'qcl-negf-physics-analysis-v5' or attrs['artifact_role'] != 'physics.analysis'
            or attrs['schema_version'] != '5.0' or attrs['contract_set'] != CONTRACT_SET
            or attrs['representation'] != 'final_diagnostics' or attrs['core_layout_sha256'] != layout):
        fail('analysis native5 schema/role/layout/representation differs')
    result = dict(representation='final_diagnostics')
    for path, key, expected in (('metadata/identity_json','identity_json',identity),
            ('metadata/invocation_json','invocation_json',invocation),
            ('final_diagnostics/declaration_json','declaration',None)):
        if path not in handle or not isinstance(handle[path], h5py.Dataset):
            fail('analysis fixed UTF8 metadata missing')
        dataset = handle[path]
        datatype, space = dataset.id.get_type(), dataset.id.get_space()
        try:
            if (space.get_simple_extent_type() != h5py.h5s.SCALAR
                    or space.get_simple_extent_npoints() != 1 or datatype.get_class() != h5py.h5t.STRING
                    or datatype.is_variable_str() or datatype.get_cset() != h5py.h5t.CSET_UTF8
                    or datatype.get_size() < 1):
                fail('analysis metadata requires fixed UTF8 scalar strings')
            consumed += datatype.get_size()
            if consumed > maximum_bytes:
                fail('aggregate analysis metadata exceeds limit', 'budget_exceeded')
        finally:
            space.close(); datatype.close()
        value = decode_json(bytes(dataset[()]), path)
        if expected is not None and value != expected:
            fail('native source identity/epoch differs from selected reference')
        result[key] = value
    result['metadata_bytes'] = consumed
    return result


_FINAL_ARRAYS = {
    'spectral_integral': ('ComplexF64', ('k','a','b'), '1'),
    'spectral_integral_eigenvalues': ('Float64', ('k','state'), '1'),
    'spectral_trace_cumulative': ('Float64', ('E','k'), '1'),
    'negative_energy_integral_scaled': ('Float64', ('matrix_kind','k'),
        'scaled solver units; broadening carries energy squared'),
}


def validate_final_diagnostics_array(handle, declaration):
    """Four local analysis arrays; never broadens the reflected Core graph."""
    import h5py
    import numpy as np
    from qcl_negf_contracts.messages import ContractError
    def fail(message):
        raise ContractError(message, 'corrupt_result')
    path = declaration.get('dataset')
    name = path.removeprefix('/final_diagnostics/arrays/') if isinstance(path,str) else None
    expected = _FINAL_ARRAYS.get(name)
    if (expected is None or path != '/final_diagnostics/arrays/'+name
            or set(declaration) != {'dataset','eltype','shape','order','axes','units'}
            or (declaration['eltype'],tuple(declaration['axes']),declaration['units']) != expected
            or declaration['order'] != 'F' or len(declaration['shape']) != len(expected[1])):
        fail('unsupported final diagnostics array descriptor')
    if any(not isinstance(n,str) or not n.isascii() or not n.isdecimal() or n.startswith('0')
           or int(n)>2**63-1 for n in declaration['shape']):
        fail('analysis array shape requires positive actual Int64 dimensions')
    if path not in handle or not isinstance(handle[path],h5py.Dataset):
        fail('analysis array dataset missing')
    dataset = handle[path]
    if dataset.shape != tuple(map(int,declaration['shape'])):
        fail('analysis array logical shape differs from descriptor')
    kind = dataset.id.get_type()
    try:
        if expected[0] == 'ComplexF64':
            if (kind.get_class()!=h5py.h5t.COMPOUND or kind.get_nmembers()!=2 or kind.get_size()!=16
                    or [kind.get_member_name(i) for i in range(2)] != [b'r',b'i']
                    or [kind.get_member_offset(i) for i in range(2)] != [0,8]
                    or dataset.dtype != np.dtype('<c16')):
                fail('analysis complex requires exact consecutive LE float64 r/i compound')
            for i in range(2):
                member = kind.get_member_type(i)
                try:
                    if (member.get_class()!=h5py.h5t.FLOAT or member.get_size()!=8
                            or member.get_order()!=h5py.h5t.ORDER_LE):
                        fail('analysis complex component precision/endian differs')
                finally:
                    member.close()
        elif (kind.get_class()!=h5py.h5t.FLOAT or kind.get_size()!=8
                or kind.get_order()!=h5py.h5t.ORDER_LE or dataset.dtype!=np.dtype('<f8')):
            fail('analysis array requires exact LE Float64')
    finally:
        kind.close()
    plist = dataset.id.get_create_plist()
    try:
        if any(plist.get_filter(i)[0] not in (1,2,3) for i in range(plist.get_nfilters())):
            fail('analysis array uses unsupported/lossy storage filter')
    finally:
        plist.close()
    return dataset


def record_table(handle, *, maximum_block_bytes=1024*1024):
    """Check offsets in bounded slabs; numerical array payloads are untouched."""
    import h5py
    import numpy as np
    from qcl_negf_contracts.messages import ContractError
    def fail(message):
        raise ContractError(message, "corrupt_result")
    if type(maximum_block_bytes) is not int or maximum_block_bytes < 8:
        raise ContractError('record coordinate buffer requires at least eight bytes', 'budget_exceeded')
    try:
        offsets, data, ordinals = (handle["records/" + name] for name in
                                    ("offsets", "json_utf8", "stream_ordinal"))
    except KeyError:
        fail("native record table missing")
    if (any(not isinstance(x, h5py.Dataset) or x.ndim != 1 for x in (offsets, data, ordinals))
            or offsets.dtype != np.dtype("<u8") or ordinals.dtype != np.dtype("<u8")
            or data.dtype != np.dtype("u1") or len(offsets) != len(ordinals) + 1):
        fail("native record coordinates dtype/shape differs")
    import math
    for dataset in (offsets, data, ordinals):
        if dataset.chunks and math.prod(dataset.chunks)*dataset.dtype.itemsize > maximum_block_bytes:
            raise ContractError('native record chunk decode buffer exceeds limit', 'budget_exceeded')
        if dataset.scaleoffset is not None or dataset.compression not in (None, 'gzip'):
            fail('native record uses unsupported/lossy storage filter')
    if int(offsets[0]) != 0 or int(offsets[-1]) != len(data):
        fail("native record offset endpoints differ")
    prior = 0
    block_entries = min(8192, maximum_block_bytes//offsets.dtype.itemsize)
    for first in range(0, len(offsets), block_entries):
        block = offsets[first:first + block_entries]
        if len(block) and (int(block[0]) < prior or np.any(block[1:] <= block[:-1])):
            fail("native record offsets are not increasing")
        if len(block):
            prior = int(block[-1])
    return len(ordinals)


def validate_native_array(handle, declaration):
    """Match native dataset storage against the verified tagged Core declaration."""
    import h5py
    import numpy as np
    from qcl_negf_contracts.messages import ContractError
    from importlib.resources import files
    import json
    def fail(message):
        raise ContractError(message, "corrupt_result")
    if declaration.get("kind") != "array" or declaration.get("order") != "F":
        fail("native array requires column-major Core declaration")
    layout = json.loads(files('qcl_negf_contracts').joinpath('core_layout.json').read_text())
    supported = {(node['eltype'], node['rank']) for node in layout['graph']['nodes'].values()
                 if node.get('kind') == 'array' and node.get('storage') == 'dataset'}
    if (declaration.get('eltype'), len(declaration.get('shape', []))) not in supported:
        fail('array eltype/rank outside reflected public Core graph')
    path = declaration.get("dataset")
    if not isinstance(path, str) or not path.startswith("/arrays/") or any(x in ("", ".", "..") for x in path[1:].split("/")):
        fail("unsupported native array dataset path")
    if path not in handle or not isinstance(handle[path], h5py.Dataset):
        fail("native array dataset missing")
    dataset = handle[path]
    if dataset.shape != tuple(int(x) for x in declaration["shape"]):
        fail("native array logical shape differs from descriptor")
    dtype = dataset.dtype
    eltype = declaration["eltype"]
    kind = dataset.id.get_type()
    if eltype == "Bool":
        if kind.get_class() != h5py.h5t.ENUM or dtype != np.dtype("bool"):
            fail("native Bool is not exact HDF Boolean enum")
    elif eltype.startswith("ComplexF"):
        width = int(eltype[8:]) // 8
        if (kind.get_class() != h5py.h5t.COMPOUND or kind.get_nmembers() != 2
                or [kind.get_member_name(i) for i in range(2)] != [b"r", b"i"]
                or [kind.get_member_offset(i) for i in range(2)] != [0, width]
                or kind.get_size() != 2 * width or dtype != np.dtype("<c" + str(2 * width))):
            fail("native complex requires exact consecutive r/i compound")
        for i in range(2):
            field = kind.get_member_type(i)
            if field.get_class() != h5py.h5t.FLOAT or field.get_size() != width or field.get_order() != h5py.h5t.ORDER_LE:
                fail("native complex component precision/order differs")
    else:
        prefix = "f" if eltype.startswith("Float") else "u" if eltype.startswith("UInt") else "i" if eltype.startswith("Int") else None
        digits = ''.join(x for x in eltype if x.isdigit())
        if prefix is None or not digits or dtype != np.dtype("<" + prefix + str(int(digits) // 8)):
            fail("native array eltype differs from descriptor")
    if dataset.scaleoffset is not None or dataset.compression not in (None, "gzip"):
        fail("native array uses unsupported/lossy storage filter")
    return dataset


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


def _observation_profiles():
    """Closed local array vocabulary from the published owning schema."""
    import json
    from importlib.resources import files
    definitions=json.loads(files('qcl_negf_contracts').joinpath('schemas/final-observations.schema.json').read_text())['$defs']
    result={}
    for key,value in definitions.items():
        if key.startswith('Array_'):
            props=value['properties']
            descriptor={name:item['const'] for name,item in props.items() if 'const' in item}
            for name in ('axes','coordinate_fields','weight_fields'):
                descriptor[name]=[item['const'] for item in props[name]['prefixItems']]
            result[key[6:]]=descriptor
    return result


def final_observations_metadata(handle, *, identity, invocation, maximum_bytes):
    """Exact closed O1 tree and fixed UTF8 metadata; no numeric value reads."""
    import h5py
    import json
    from importlib.resources import files
    from qcl_negf_contracts.messages import ContractError
    from .commits import decode_json
    def fail(message,code='corrupt_result'):raise ContractError(message,code)
    if set(handle)!={'metadata','final_observations'} or len(handle.attrs):fail('observations root layout differs')
    for path,names in (('metadata',{'identity_json','invocation_json'}),
                       ('final_observations',{'declaration_json','arrays'})):
        group=handle[path]
        if not isinstance(group,h5py.Group) or set(group)!=names:fail('observations closed group layout differs')
    if (set(handle['metadata'].attrs)!={'schema','schema_version','contract_set','artifact_role','representation','core_layout_sha256'}
            or len(handle['final_observations'].attrs)):
        fail('observations group attributes differ')
    attrs,consumed=_native_header(handle['metadata'],maximum_bytes,with_size=True)
    layout=json.loads(files('qcl_negf_contracts').joinpath('core_layout.json').read_text())['layout_sha256']
    if attrs!={'schema':'qcl-negf-physics-analysis-v5','schema_version':'5.0','contract_set':CONTRACT_SET,
               'artifact_role':'physics.analysis','representation':'final_observations','core_layout_sha256':layout}:
        fail('observations native5 schema/role/layout/representation differs')
    result=dict(representation='final_observations');addresses=set()
    for path,key,expected in (('metadata/identity_json','identity_json',identity),
                              ('metadata/invocation_json','invocation_json',invocation),
                              ('final_observations/declaration_json','declaration',None)):
        if not isinstance(handle.get(path,getlink=True),h5py.HardLink):fail('observations metadata link forbidden')
        dataset=handle[path]
        if not isinstance(dataset,h5py.Dataset) or len(dataset.attrs):fail('observations scalar metadata layout differs')
        datatype,space=dataset.id.get_type(),dataset.id.get_space()
        try:
            if (space.get_simple_extent_type()!=h5py.h5s.SCALAR or space.get_simple_extent_npoints()!=1
                    or datatype.get_class()!=h5py.h5t.STRING or datatype.is_variable_str()
                    or datatype.get_cset()!=h5py.h5t.CSET_UTF8 or datatype.get_size()<1):
                fail('observations metadata requires fixed UTF8 scalar strings')
            consumed+=datatype.get_size()
            if consumed>maximum_bytes:fail('aggregate observations metadata exceeds limit','budget_exceeded')
        finally:space.close();datatype.close()
        value=decode_json(bytes(dataset[()]),path)
        if expected is not None and value!=expected:fail('observations source identity/epoch differs')
        result[key]=value;addresses.add(h5py.h5o.get_info(dataset.id).addr)
    value=result['declaration']
    if not isinstance(value,dict) or not isinstance(value.get('context'),dict) or not isinstance(value.get('views'),dict):
        fail('observations declaration missing closed array contexts')
    profiles=_observation_profiles();expected={name for name in profiles if (value['context'] if name in value['context'] else value['views']).get(name) is not None}
    group=handle['final_observations/arrays']
    if not isinstance(group,h5py.Group) or len(group.attrs) or len(group)>26 or set(group)!=expected:
        fail('observations array inventory differs')
    for path in ('metadata','final_observations','final_observations/arrays'):
        if not isinstance(handle.get(path,getlink=True),h5py.HardLink):fail('observations group link forbidden')
        address=h5py.h5o.get_info(handle[path].id).addr
        if address in addresses:fail('observations object alias forbidden')
        addresses.add(address)
    for name in group:
        if not isinstance(group.get(name,getlink=True),h5py.HardLink):fail('observations array link forbidden')
        dataset=group[name]
        if not isinstance(dataset,h5py.Dataset) or len(dataset.attrs) or dataset.external or dataset.is_virtual:
            fail('observations array storage/attributes differ')
        address=h5py.h5o.get_info(dataset.id).addr
        if address in addresses:fail('observations object alias forbidden')
        addresses.add(address)
    result['metadata_bytes']=consumed
    return result


def validate_final_observations_array(handle,declaration):
    """Exactly 26 O1 descriptors; does not register new Core array ranks."""
    import h5py
    import numpy as np
    from qcl_negf_contracts.messages import ContractError
    def fail(message):raise ContractError(message,'corrupt_result')
    path=declaration.get('dataset');prefix='/final_observations/arrays/'
    name=path.removeprefix(prefix) if isinstance(path,str) else None
    profile=_observation_profiles().get(name)
    if (profile is None or set(declaration)!=set(profile)|{'shape'}
            or any(declaration.get(key)!=value for key,value in profile.items())):
        fail('unsupported observations array descriptor')
    shape=declaration['shape']
    if (not isinstance(shape,list) or len(shape)!=len(profile['axes']) or any(type(v) is not str
            or not v.isascii() or not v.isdecimal() or v.startswith('0') or int(v)>2**63-1 for v in shape)):
        fail('observations shape requires positive actual Int64 dimensions')
    if not isinstance(handle.get(path,getlink=True),h5py.HardLink):fail('observations array link forbidden')
    dataset=handle[path]
    if (not isinstance(dataset,h5py.Dataset) or dataset.shape!=tuple(map(int,shape)) or dataset.external
            or dataset.is_virtual or len(dataset.attrs)):
        fail('observations array logical shape/storage differs')
    kind=dataset.id.get_type()
    try:
        eltype=profile['eltype']
        if eltype=='ComplexF64':
            if (kind.get_class()!=h5py.h5t.COMPOUND or kind.get_nmembers()!=2 or kind.get_size()!=16
                    or [kind.get_member_name(i) for i in range(2)]!=[b'r',b'i']
                    or [kind.get_member_offset(i) for i in range(2)]!=[0,8] or dataset.dtype!=np.dtype('<c16')):
                fail('observations complex requires exact LE r/i compound')
            for i in range(2):
                member=kind.get_member_type(i)
                try:
                    if not member.equal(h5py.h5t.IEEE_F64LE):fail('observations complex component is not exact IEEE Float64')
                finally:member.close()
        elif eltype=='Bool':
            if kind.get_class()!=h5py.h5t.ENUM or kind.get_nmembers()!=2 or dataset.dtype!=np.dtype('bool'):
                fail('observations Bool requires exact FALSE/TRUE enum')
            base=kind.get_super()
            try:
                if not base.equal(h5py.h5t.STD_U8LE):fail('observations Bool base differs')
            finally:base.close()
            if {kind.get_member_name(i):kind.get_member_value(i) for i in range(2)}!={b'FALSE':0,b'TRUE':1}:
                fail('observations Bool enum differs')
        elif not kind.equal(h5py.h5t.IEEE_F64LE if eltype=='Float64' else h5py.h5t.STD_I64LE):
            fail('observations array requires exact LE Float64/Int64')
    finally:kind.close()
    plist=dataset.id.get_create_plist()
    try:
        if (plist.get_layout() not in (h5py.h5d.CONTIGUOUS,h5py.h5d.CHUNKED) or plist.get_nfilters()>2
                or any(plist.get_filter(i)[0] not in (1,2) for i in range(plist.get_nfilters()))):
            fail('observations unsupported/lossy storage filter/layout')
    finally:plist.close()
    return dataset
