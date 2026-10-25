"""Bounded read-only semantic views of one committed analytical projection.

This is neither StateReader nor full native/scientific validation. Provider code
owns exact remote selection, copying, authentication and temporary-file lifetime.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
import stat
from pathlib import Path
from typing import Any, Mapping

import h5py
import numpy as np

from qcl_negf_contracts.artifacts import validate_commit
from qcl_negf_contracts.messages import ContractError, scientific_plan
from .commits import decode_json, safe_path
from .provenance import _state_identity
from .state import _BudgetStream, _IOBudget, _same_json_value, _verify_stream


@dataclass(frozen=True)
class SavedViewLimits:
    maximum_metadata_bytes: int = 1024 * 1024
    maximum_owner_bytes: int = 64 * 1024 * 1024
    maximum_io_bytes: int = 128 * 1024 * 1024
    maximum_selected_bytes: int = 1024 * 1024
    maximum_json_bytes: int = 2 * 1024 * 1024
    maximum_cells: int = 65536
    maximum_line_points: int = 4096
    maximum_states: int = 12
    maximum_matrix_states: int = 64
    maximum_objects: int = 4096

    def __post_init__(self):
        if any(type(value) is not int or value < 1 for value in vars(self).values()):
            raise ValueError("saved-view limits must be positive integers")


# view_id -> native path, external axes, exact units, basis, diagonal projection.
VIEWS = {
    **{f"potential_{name}": (f"observables/potential_{name}_eV", ("z",), "eV", None, False)
       for name in ("structure", "external", "hartree", "total")},
    "density": ("observables/density_per_m3", ("z",), "m^-3", None, False),
    "spatial_energy_density": ("observables/spatial_energy_density", ("E", "z"), "m^-3/eV", None, False),
    "spectral_trace": ("observables/spectral", ("E", "k"), "eV^-1", None, False),
    "occupied_trace": ("observables/occupied_spectral", ("E", "k"), "eV^-1", None, False),
    **{f"{basis}_levels": (f"basis/{basis}_levels_eV", ("state",), "eV", basis, False)
       for basis in ("localized", "effective")},
    **{f"{basis}_wavefunctions_{component}": (f"basis/{basis}_wavefunctions/{component}",
        ("z", "state"), "nm^-1/2", basis, False)
       for basis in ("localized", "effective") for component in ("real", "imag")},
    **{f"{basis}_density_matrix_{component}": (f"basis/{native}/{component}",
        ("state", "state_prime"), "m^-2", basis, False)
       for basis, native in (("localized", "sheet_density_matrix_per_m2"), ("effective", "effective_density_matrix_per_m2"))
       for component in ("real", "imag")},
    **{f"{basis}_populations": (f"basis/{native}/real", ("state", "state_prime"), "m^-2", basis, True)
       for basis, native in (("localized", "sheet_density_matrix_per_m2"), ("effective", "effective_density_matrix_per_m2"))},
}
AXES = {"z": "axes/z_nm", "E": "axes/energy_eV", "k": "axes/k_per_nm",
        "state": "axes/state_index", "state_prime": "axes/state_index"}
WEIGHTS = {"z": "axes/spatial_weights", "E": "axes/energy_weights", "k": "axes/momentum_weights"}


def _error(code, reason):
    raise ContractError(reason, code)


def _bounded(value, maximum):
    try:
        payload = json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode()
    except (ValueError, TypeError) as error:
        raise ContractError("saved metadata must contain finite JSON values", "corrupt_result") from error
    if len(payload) > maximum:
        _error("budget_exceeded", "saved response exceeds its JSON budget")
    return value


def _selection(value):
    keys = {"run_uuid", "execution_id", "attempt", "calcjob_uuid"}
    if not isinstance(value, Mapping) or set(value) != keys:
        _error("selection_required", "exact run/execution/attempt/CalcJob selection required")
    if type(value["attempt"]) is not int or value["attempt"] < 1 or any(
            not isinstance(value[k], str) or not value[k] for k in keys - {"attempt"}):
        _error("selection_required", "invalid exact saved selection")
    return dict(value)


def brief_result(metadata, *, selection, maximum_bytes=1024*1024):
    """Pure finite metadata projection in frozen point order; never opens HDF5."""
    selection = _selection(selection)
    if type(maximum_bytes) is not int or maximum_bytes < 1:
        raise ValueError("maximum_bytes must be positive")
    _bounded(metadata, maximum_bytes)
    plan = scientific_plan(metadata.get("plan"))
    series = metadata.get("series", {})
    if series.get("schema") != "qcl-negf-series-result-v3" or series.get("contract_set") != "qcl-negf.results.v1":
        _error("unsupported_contract", "requires saved native series v3")
    if series.get("plan_fingerprint") != plan["fingerprint"]:
        _error("identity_mismatch", "series differs from retained frozen plan")
    source = dict(metadata.get("source", {}))
    if source.get("calcjob_uuid") not in (None, selection["calcjob_uuid"]):
        _error("identity_mismatch", "source CalcJob differs")
    for key in ("execution_id", "attempt", "run_uuid", "calcjob_uuid"):
        if key in series and not _same_json_value(series[key], selection[key]):
            _error("identity_mismatch", "series selection differs")
    execution = [e for e in plan["executions"] if e["id"] == selection["execution_id"]]
    if len(execution) != 1:
        _error("identity_mismatch", "execution absent from retained plan")
    rows = series.get("points")
    if not isinstance(rows, list):
        _error("corrupt_result", "saved series points must be a list")
    by_id = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("id"), str) or row["id"] in by_id:
            _error("corrupt_result", "invalid or duplicate saved point")
        if row.get("execution_id") != selection["execution_id"]:
            # A multi-execution series is legal; foreign rows are never selected.
            continue
        if "attempt" in row and (type(row["attempt"]) is not int or
                                 not 1 <= row["attempt"] <= selection["attempt"]):
            _error("identity_mismatch", "saved point attempt exceeds owning selection or is invalid")
        data = row.get("data") or {}
        if "checkpoint_source_attempt" in data:
            source_attempt = data["checkpoint_source_attempt"]
            if type(source_attempt) is not int or not 1 <= source_attempt <= row.get("attempt", selection["attempt"]):
                _error("identity_mismatch", "checkpoint source attempt exceeds saved point or is invalid")
        by_id[row["id"]] = row
    frozen = [p for p in plan["points"] if p.get("execution_id") == selection["execution_id"]]
    if any(identifier not in {p["id"] for p in frozen} for identifier in by_id):
        _error("identity_mismatch", "saved point absent from frozen execution")
    points, segments, active = [], [], []
    previous_branch = None
    for planned in frozen:
        row = by_id.get(planned["id"])
        observables = (row or {}).get("observables") or {}
        assessment = observables.get("scientific_assessment")
        if assessment is not None and not isinstance(assessment, dict):
            _error("corrupt_result", "native scientific assessment must be an object")
        accepted = assessment.get("scientific_accepted") if assessment is not None else None
        current = observables.get("current_density_A_per_m2")
        finite_current = type(current) in (int, float) and math.isfinite(current)
        missing = ["planned_row_absent"] if row is None else ([] if finite_current else ["current_unavailable"])
        status = (row or {}).get("status")
        # Preserve raw assessments; this is solely a conservative line topology,
        # never a new scientific acceptance judgement.
        connectable = finite_current and status == "completed" and accepted is True
        branch = (planned.get("branch"), planned.get("temperature_K"))
        if not connectable or (active and branch != previous_branch):
            if active: segments.append(active); active = []
        if connectable: active.append(planned["id"])
        previous_branch = branch
        data = (row or {}).get("data") or {}
        coordinates = {k: planned.get(k) for k in ("temperature_K", "voltage_per_period_V", "branch", "order")}
        if row and row.get("coordinates"):
            for k, v in coordinates.items():
                if k in row["coordinates"] and not _same_json_value(row["coordinates"][k], v):
                    _error("identity_mismatch", "saved coordinates differ from frozen point")
        points.append({"point_id": planned["id"], "coordinates": coordinates,
            "status": status, "source_commit_sha256": data.get("source_commit_sha256", (row or {}).get("source_commit_sha256")),
            "source_attempt": data.get("checkpoint_source_attempt", (row or {}).get("attempt")),
            "scalar_observables": observables, "missing": missing, "line_connectable": connectable,
            "recorded_statuses": {**{key: (row or {}).get(key) for key in ("status", "converged", "quality")},
                "scientific_accepted": accepted, "stationary_assessment": assessment,
                "scientific_assessment": assessment,
                "assessment_source_path": "observables.scientific_assessment" if assessment is not None else None}})
    if active: segments.append(active)
    result = {"schema": "qcl-negf.saved-results.v1", "selection": selection,
        "plan_source": metadata.get("plan_source"),
        "identity": {"definition_id": execution[0].get("definition_id"), "variant_id": execution[0].get("variant_id"),
                     "plan_fingerprint": plan["fingerprint"], "plan_scientific_fingerprint": plan.get("scientific_fingerprint"),
                     "configuration_hash": None},
        "statuses": {k: series.get(k) for k in ("process", "converged", "quality", "scientific_accepted", "stationary_assessment")},
        "points": points, "current_segments": segments,
        "capabilities": [{"id": "current_voltage", "operation": "read", "status": "available" if any(not p["missing"] for p in points) else "unavailable", "reason": None}],
        "completeness": {"saved_profile": series.get("saved_profile"), "available_points": len(by_id),
                         "expected_points": len(frozen), "missing": [p["point_id"] for p in points if p["missing"]]},
        "source": {**source, "source_revisions": source.get("source_revisions"),
                   "unknown_fields": [k for k in ("source_revisions", "locator_id") if source.get(k) is None]}}
    return _bounded(result, maximum_bytes)


def _open_regular(path, *, expected_size=None):
    """Own one final-component nofollow descriptor until transfer to FileIO."""
    required = ("O_RDONLY", "O_NOFOLLOW", "O_CLOEXEC", "O_NONBLOCK")
    if (not all(hasattr(os, name) for name in required) or
            not all(callable(getattr(os, name, None)) for name in ("open", "fstat", "fdopen"))):
        _error("unsupported_contract", "saved reads require POSIX nofollow regular-file opens")
    flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ContractError("saved artifact cannot be opened without following links", "corrupt_result") from error
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode):
            _error("corrupt_result", "saved artifact must be an ordinary regular file")
        if expected_size is not None and opened.st_size != expected_size:
            _error("corrupt_result", "opened saved artifact length differs from its commit")
        return os.fdopen(descriptor, "rb", buffering=0)
    except BaseException as error:
        os.close(descriptor)
        if isinstance(error, OSError):
            raise ContractError("saved artifact descriptor cannot be inspected", "corrupt_result") from error
        raise


def _saved_budget_json(path, maximum, budget):
    # Keep state._budget_json's exact max+1 delivery/decode contract locally.
    with _open_regular(path) as raw:
        with _BudgetStream(raw, budget) as stream:
            payload = stream.read(maximum + 1)
    if len(payload) > maximum:
        _error("corrupt_result", "JSON exceeds its declared budget")
    return decode_json(payload, str(path)), payload


class AnalysisReader:
    """One declared analysis owner, metadata-only description and verified reads."""
    def __init__(self, commit_path, *, expected_identity, limits=None):
        self.limits = limits or SavedViewLimits()
        self._budget = _IOBudget(self.limits.maximum_io_bytes)
        self._handle = self._stream = self._raw = None
        self._metadata_bytes = 0
        self._verified = False
        path = Path(commit_path)
        if path.is_symlink() or any(p.is_symlink() for p in path.parents):
            _error("corrupt_result", "commit path traverses symlink")
        try:
            self.commit, payload = _saved_budget_json(path, self.limits.maximum_metadata_bytes, self._budget)
        except ContractError as error:
            raise ContractError("invalid saved commit metadata", error.code) from error
        self._charge_metadata(len(payload))
        self.commit_sha256 = hashlib.sha256(payload).hexdigest()
        artifacts = validate_commit(self.commit)
        expected = dict(expected_identity or {})
        self.expected_identity = expected
        if not {"point_id", "execution_id", "attempt", "plan_fingerprint"} <= expected.keys():
            _error("selection_required", "exact expected point identity required")
        for key, value in expected.items():
            if key == "source_commit_sha256":
                if self.commit_sha256 != value: _error("identity_mismatch", "selected commit changed")
            elif key in self.commit["identity"] or key in {"point_id", "execution_id", "attempt", "plan_fingerprint"}:
                if not _same_json_value(self.commit["identity"].get(key), value):
                    _error("identity_mismatch", "commit differs from selected identity")
        owners = [item for item in artifacts if item.role == "physics.analysis"]
        if len(owners) != 1 or owners[0].schema != "qcl-negf-physics-analysis-v4":
            _error("unsupported_contract", "requires one declared native analysis owner")
        self.owner = owners[0]
        if self.owner.size > self.limits.maximum_owner_bytes:
            _error("budget_exceeded", "analysis owner exceeds ordinary read cap")
        by_path = {item.path: item for item in artifacts}
        pending, checked = list(self.owner.dependencies), set()
        while pending:
            dependency = by_path[pending.pop()]
            if dependency.path in checked:
                continue
            checked.add(dependency.path)
            if dependency.media_type != "application/json":
                _error("unsupported_contract", "analysis dependency is outside bounded metadata closure")
            self._charge_metadata(dependency.size)
            dependency_path = safe_path(path.parent, dependency.path)
            with _open_regular(dependency_path, expected_size=dependency.size) as raw:
                with _BudgetStream(raw, self._budget) as stream:
                    self._budget.phase = "sha"
                    _verify_stream(stream, dependency.size, dependency.sha256)
            self._budget.phase = "metadata"
            pending.extend(dependency.dependencies)
        self.path = safe_path(path.parent, self.owner.path)
        try:
            self._raw = _open_regular(self.path, expected_size=self.owner.size)
            self._stream = _BudgetStream(self._raw, self._budget)
            self._budget.phase = "hdf"
            self._handle = h5py.File(self._stream, "r")
            self._storage()
            header = self._handle.get("metadata")
            if not isinstance(header, h5py.Group): _error("unsupported_contract", "native header absent")
            for key, wanted in {"schema": self.owner.schema, "schema_version": "4.0", "artifact_role": "physics.analysis", "contract_set": "qcl-negf.results.v1"}.items():
                if self._attr(header, key) != wanted: _error("unsupported_contract", "native header differs")
            identity = self._json("metadata/identity_json")
            if not isinstance(identity, dict): _error("corrupt_result", "native identity must be an object")
            for key, value in self.commit["identity"].items():
                if key not in identity or not _same_json_value(identity[key], value):
                    _error("identity_mismatch", "native identity differs from owning commit")
            _state_identity(identity, self.commit["identity"])
            for key, value in expected.items():
                if key in identity and not _same_json_value(identity[key], value):
                    _error("identity_mismatch", "native recorded identity differs from expected selection")
            self.identity_binding = {"native_recorded_fields": sorted(identity),
                "provider_only_fields": sorted(key for key in ("run_uuid", "calcjob_uuid")
                    if key in expected and key not in identity and key not in self.commit["identity"])}
            self.conventions = {key: self._attr(header, key) for key in ("current_convention", "energy_reference")}
            self.source_revisions = self._json("metadata/toolchain_json") if "metadata/toolchain_json" in self._handle else None
        except BaseException as error:
            self.close()
            if isinstance(error, ContractError):
                raise
            if isinstance(error, Exception):
                raise ContractError("saved native metadata cannot be read", "corrupt_result") from error
            raise

    def __enter__(self): return self
    def __exit__(self, *args): self.close()
    def close(self):
        handle, stream, raw = self._handle, self._stream, self._raw
        self._handle = self._stream = self._raw = None
        try:
            if handle is not None:
                handle.close()
        finally:
            try:
                if stream is not None:
                    stream.close()
            finally:
                if raw is not None:
                    raw.close()

    def _charge_metadata(self, count):
        self._metadata_bytes += count
        if self._metadata_bytes > self.limits.maximum_metadata_bytes:
            _error("budget_exceeded", "native metadata exceeds cap")

    def _attr(self, obj, name):
        if name not in obj.attrs: return None
        identifier = obj.attrs.get_id(name)
        dtype, space = identifier.get_type(), identifier.get_space()
        try:
            # Check the low-level atomic type and SCALAR dataspace before any
            # high-level payload read or NumPy conversion: VLEN descriptor width
            # does not bound heap allocation, including compound-with-VLEN types.
            if (space.get_simple_extent_type() != h5py.h5s.SCALAR or
                    space.get_simple_extent_dims() != () or space.get_simple_extent_npoints() != 1):
                _error("unsupported_contract", "metadata attributes require scalar dataspace")
            kind, width = dtype.get_class(), dtype.get_size()
            if kind == h5py.h5t.STRING:
                if (dtype.is_variable_str() or width < 1 or
                        dtype.get_cset() not in {h5py.h5t.CSET_ASCII, h5py.h5t.CSET_UTF8}):
                    _error("unsupported_contract", "bounded metadata requires atomic fixed strings")
            elif kind == h5py.h5t.INTEGER:
                if width not in {1, 2, 4, 8}:
                    _error("unsupported_contract", "unsupported atomic integer metadata width")
            elif kind == h5py.h5t.FLOAT:
                if width not in {4, 8}:
                    _error("unsupported_contract", "unsupported atomic float metadata width")
            else:
                _error("unsupported_contract", "metadata rejects VLEN, compound, array and reference types")
            self._charge_metadata(width)
            value = obj.attrs[name]
            if isinstance(value, bytes): return value.decode("utf-8")
            if isinstance(value, np.generic): return value.item()
            if isinstance(value, (str, int, float, bool)): return value
            _error("unsupported_contract", "metadata attributes must be scalars")
        finally:
            space.close(); dtype.close(); identifier.close()

    def _json(self, path):
        if path not in self._handle: _error("corrupt_result", "required native identity missing")
        obj = self._handle[path]
        if not isinstance(obj, h5py.Dataset) or obj.shape != () or obj.dtype.kind != "S":
            _error("unsupported_contract", "bounded JSON metadata requires fixed-string scalar")
        self._charge_metadata(obj.dtype.itemsize)
        return decode_json(b'{"value":' + bytes(obj[()]) + b'}', "native metadata")["value"]

    def _storage(self):
        pending, seen, count = [self._handle], set(), 0
        while pending:
            group = pending.pop()
            address = h5py.h5o.get_info(group.id).addr
            if address in seen: continue
            seen.add(address)
            for key in group:
                count += 1
                if count > self.limits.maximum_objects: _error("budget_exceeded", "native inventory exceeds cap")
                if not isinstance(group.get(key, getlink=True), h5py.HardLink):
                    _error("corrupt_result", "analysis rejects external and soft links")
                item = group[key]
                if isinstance(item, h5py.Group): pending.append(item)
                elif isinstance(item, h5py.Dataset) and (item.is_virtual or item.external):
                    _error("corrupt_result", "analysis rejects unowned virtual/external storage")

    def _numeric(self, path, shape=None):
        if path not in self._handle: _error("missing_dataset", "saved native dataset unavailable")
        obj = self._handle[path]
        if not isinstance(obj, h5py.Dataset) or obj.dtype.kind not in "biuf" or obj.dtype.hasobject or obj.dtype.itemsize not in {1, 2, 4, 8}:
            _error("unsupported_contract", "view requires supported fixed-width real numeric data")
        if shape is not None and obj.shape != shape: _error("corrupt_result", "saved axis shape differs")
        return obj

    def _describe(self, view_id):
        if view_id not in VIEWS: _error("missing_dataset", "unknown semantic saved view")
        path, axes, units, basis, diagonal = VIEWS[view_id]
        obj = self._numeric(path)
        if obj.ndim != len(axes) or self._attr(obj, "logical_axis_order") != ",".join(axes):
            _error("corrupt_result", "native axes differ from semantic view")
        if self._attr(obj, "units") != units: _error("missing_dataset", "required saved units unavailable")
        logical_shape = self._attr(obj, "logical_shape")
        if logical_shape is not None and logical_shape != ",".join(map(str, obj.shape)):
            _error("corrupt_result", "logical and stored shapes differ")
        raw = self._attr(obj, "axis_coordinate_paths_json")
        try: paths = json.loads(raw) if isinstance(raw, str) else None
        except ValueError as error: raise ContractError("invalid saved axis paths", "corrupt_result") from error
        if paths != [AXES[a] for a in axes] and paths != ["/"+AXES[a] for a in axes]:
            _error("missing_dataset", "explicit native axis mapping unavailable")
        for axis, size in zip(axes, obj.shape):
            coordinate = self._numeric(AXES[axis], (size,))
            expected_units = {"z": "nm", "E": "eV", "k": "nm^-1", "state": "1", "state_prime": "1"}[axis]
            if self._attr(coordinate, "units") != expected_units:
                _error("missing_dataset", "required coordinate units unavailable")
            if axis in {"state", "state_prime"} and coordinate.dtype.kind not in "iu":
                _error("corrupt_result", "saved state indices must be integers")
        gauge = self._attr(self._handle["basis"], "effective_gauge") if basis == "effective" else (
            self._attr(self._handle["basis"], "localization") if basis else None)
        if basis and not gauge: _error("missing_dataset", "saved basis/gauge unavailable")
        if diagonal and (obj.ndim != 2 or obj.shape[0] != obj.shape[1]):
            _error("corrupt_result", "population matrix must be square")
        return {"view_id": view_id, "artifact_id": self.owner.path, "native_paths": [path],
                "shape": list(obj.shape), "dtype": str(obj.dtype), "axes": list(axes), "units": units,
                "basis": basis, "conventions": {**self.conventions, "gauge": gauge},
                "allowed_selection": {"axis_count": 1 if diagonal else obj.ndim, "index_origin": 0, "positive_step": True,
                    "maximum_cells": self.limits.maximum_cells}, "availability": "available", "reason": None}

    def _integrity(self):
        return {"metadata_checked": True, "owner_sha_verified": self._verified, "native_validation": "not_performed"}

    def describe(self):
        views = []
        for view_id in VIEWS:
            try: views.append(self._describe(view_id))
            except ContractError as error:
                if error.code != "missing_dataset": raise
                views.append({"view_id": view_id, "artifact_id": self.owner.path, "availability": "unavailable", "reason": str(error)})
        return _bounded({"files": [{"artifact_id": self.owner.path, "role": self.owner.role, "schema": self.owner.schema,
            "bytes": self.owner.size, "sha256": self.owner.sha256, "commit_sha256": self.commit_sha256}],
            "identity": dict(self.commit["identity"]), "identity_binding": self.identity_binding,
            "views": views, "integrity": self._integrity(),
            "source_commit_sha256": self.commit_sha256,
            "source": {"source_revisions": self.source_revisions,
                       "unknown_fields": ["source_revisions"] if self.source_revisions is None else []},
            "limits": vars(self.limits), "io_counters": self._budget.counters()}, self.limits.maximum_json_bytes)

    def _normalize(self, selection, shape):
        if not isinstance(selection, (list, tuple)) or len(selection) != len(shape):
            _error("selection_required", "selection must name every semantic axis")
        items, counts, normalized = [], [], []
        for item, size in zip(selection, shape):
            if type(item) is int:
                if not 0 <= item < size: _error("selection_required", "native index outside axis")
                items.append(item); counts.append(1); normalized.append(item)
                continue
            if isinstance(item, slice):
                item = {"start": 0 if item.start is None else item.start,
                        "stop": size if item.stop is None else item.stop,
                        "step": 1 if item.step is None else item.step}
            if not isinstance(item, Mapping) or set(item) != {"start", "stop", "step"}:
                _error("selection_required", "requires explicit index or start/stop/step")
            start, stop, step = (item[k] for k in ("start", "stop", "step"))
            if (any(type(v) is not int for v in (start, stop, step)) or
                    not 0 <= start < stop <= size or step < 1):
                _error("selection_required", "invalid native range; no implicit clamping")
            items.append(slice(start, stop, step)); counts.append(len(range(start, stop, step)))
            normalized.append(dict(item))
        return items, counts, normalized

    def _chunk_bound(self, obj):
        if obj.chunks and math.prod(obj.chunks)*obj.dtype.itemsize > self.limits.maximum_selected_bytes:
            _error("budget_exceeded", "decoded HDF5 chunk exceeds selection allocation cap")
        plist = obj.id.get_create_plist()
        try:
            if any(plist.get_filter(i)[0] not in {1, 2, 3, 4, 5, 6} for i in range(plist.get_nfilters())):
                _error("unsupported_contract", "external HDF5 filter unavailable for bounded views")
        finally:
            plist.close()

    def read(self, view_id, selection):
        try:
            return self._read(view_id, selection)
        except ContractError:
            raise
        except (OSError, ValueError, TypeError, KeyError) as error:
            raise ContractError("saved native selection cannot be read", "corrupt_result") from error

    def _read(self, view_id, selection):
        description = self._describe(view_id)
        native, names, units, basis, diagonal = VIEWS[view_id]
        dataset = self._numeric(native)
        semantic_shape = (dataset.shape[0],) if diagonal else dataset.shape
        selected, counts, normalized = self._normalize(selection, semantic_shape)
        if diagonal:
            # Read only the explicitly selected square block, never a full
            # matrix to extract a tiny diagonal. Its full allocation is charged.
            item = selected[0]
            item = slice(item, item+1, 1) if type(item) is int else item
            native_selected = [item, item]
            native_counts = [counts[0], counts[0]]
            output_names = ["state"]
            output_items = [item]
        else:
            native_selected, native_counts = selected, counts
            output_names, output_items = list(names), selected
        cells = math.prod(native_counts)
        if cells > self.limits.maximum_cells:
            _error("budget_exceeded", "selected native cells exceed cap")
        for axis, count in zip(names, native_counts):
            if axis in {"state", "state_prime"}:
                cap = self.limits.maximum_matrix_states if "density_matrix" in view_id else self.limits.maximum_states
                if count > cap: _error("budget_exceeded", "explicit state subset exceeds cap")
        if sum(type(item) is not int for item in native_selected) <= 1 and max(native_counts) > self.limits.maximum_line_points:
            _error("budget_exceeded", "line selection exceeds native point cap")
        self._chunk_bound(dataset)
        required = cells * dataset.dtype.itemsize
        vectors = []
        for axis, item in zip(output_names, output_items):
            size = dataset.shape[names.index(axis)]
            coordinate = self._numeric(AXES[axis], (size,))
            self._chunk_bound(coordinate)
            count = 1 if type(item) is int else len(range(item.start, item.stop, item.step))
            required += count*coordinate.dtype.itemsize
            weight = self._numeric(WEIGHTS[axis], (size,)) if axis in WEIGHTS and WEIGHTS[axis] in self._handle else None
            if weight is not None:
                self._chunk_bound(weight); required += count*weight.dtype.itemsize
            vectors.append((axis, item, coordinate, weight))
        target = None
        target_paths = ("inputs/N_dop_2D_per_m2", "inputs/f_ion")
        if diagonal and all(path in self._handle for path in target_paths):
            donor, ionization = [self._numeric(path, ()) for path in target_paths]
            if self._attr(donor, "units") != "m^-2" or self._attr(ionization, "units") != "1":
                _error("missing_dataset", "saved target doping units unavailable")
            required += donor.dtype.itemsize + ionization.dtype.itemsize
            target = (donor, ionization)
        if required > self.limits.maximum_selected_bytes:
            _error("budget_exceeded", "selected data, coordinates and weights exceed byte cap")
        # SHA obligation covers the whole declared analysis owner, separately
        # from hyperslab allocation and all HDF5 stream delivery counters.
        if not self._verified:
            self._budget.require(self.owner.size)
            self._budget.phase = "sha"
            try: _verify_stream(self._stream, self.owner.size, self.owner.sha256)
            finally: self._budget.phase = "hdf"
            self._verified = True
        axes = []
        for name, item, coordinate, weight in vectors:
            coordinate_values = np.atleast_1d(coordinate[item])
            finite_coordinates, coordinate_invalid = _finite(coordinate_values)
            if any(reason is not None for reason in coordinate_invalid):
                _error("corrupt_result", "saved coordinates must be finite")
            weight_values, weight_invalid = _finite(np.atleast_1d(weight[item])) if weight is not None else (None, None)
            axes.append({"name": name, "values": finite_coordinates, "units": self._attr(coordinate, "units"),
                "source_path": AXES[name], "weights": weight_values, "weight_invalid_mask": weight_invalid,
                "integration_measure": self._attr(weight, "integration_measure") if weight is not None else None,
                "physical_measure_multiplier": self._attr(weight, "physical_measure_multiplier") if weight is not None else None,
                "physical_measure_units": self._attr(weight, "physical_measure_units") if weight is not None else None})
        values = dataset[tuple(native_selected)]
        if diagonal: values = np.diag(values).copy()
        finite_values, invalid = _finite(values)
        fractions = None
        if diagonal:
            denominator = float(target[0][()]) * float(target[1][()]) if target else None
            if denominator is not None and math.isfinite(denominator) and denominator > 0:
                fraction_values, fraction_invalid = _finite(values/denominator)
                fractions = {"values": fraction_values, "invalid_mask": fraction_invalid, "reason": None,
                    "denominator": {"value": denominator, "units": "m^-2", "source_paths": list(target_paths),
                        "kind": "results_derived_from_saved_target_inputs", "rule": "ionized_sheet_density = saved donor_sheet_density * saved ionization_fraction"}}
            else:
                fractions = {"values": None, "invalid_mask": None, "denominator": None,
                             "reason": "saved_target_doping_unavailable" if target is None else "saved_target_doping_nonpositive_or_invalid"}
        self._budget.logical_selected_bytes += required
        result = {"selection": {k: self.expected_identity.get(k, self.commit["identity"].get(k)) for k in ("run_uuid", "execution_id", "attempt", "calcjob_uuid")},
            "point_id": self.commit["identity"]["point_id"], "identity_binding": self.identity_binding,
            "source_commit_sha256": self.commit_sha256,
            "artifact_sha256": self.owner.sha256, "view_id": view_id, "normalized_selection": normalized,
            "native_shape": list(dataset.shape), "returned_shape": list(np.shape(values)),
            "logical_axes": [name for name, item in zip(output_names, output_items) if type(item) is not int],
            "axes": axes, "values": finite_values, "invalid_mask": invalid,
            "units": units, "basis": basis, "conventions": description["conventions"],
            "sampling": {"kind": "exact_native_selection", "step": [item.step if isinstance(item, slice) else None for item in selected]},
            "recorded_statuses": {k: self.commit.get(k) for k in ("terminal_status", "quality", "converged", "scientific_accepted", "stationary_assessment")},
            "integrity": self._integrity(), "limits": vars(self.limits), "io_counters": self._budget.counters(),
            "fractions": fractions,
            "source": {"source_revisions": self.source_revisions,
                       "unknown_fields": ["source_revisions"] if self.source_revisions is None else []}}
        return _bounded(result, self.limits.maximum_json_bytes)


def _finite(value):
    """JSON null plus reason masks distinguish native invalids from finite zero."""
    if isinstance(value, np.ndarray) and value.ndim:
        rows = [_finite(item) for item in value]
        return [item[0] for item in rows], [item[1] for item in rows]
    if isinstance(value, np.generic) or isinstance(value, np.ndarray): value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None, "nan" if math.isnan(value) else "positive_infinity" if value > 0 else "negative_infinity"
    return value, None


def describe_saved_views(commit_path, *, expected_identity, limits=None):
    with AnalysisReader(commit_path, expected_identity=expected_identity, limits=limits) as reader:
        return reader.describe()


def read_saved_view(commit_path, view_id, selection, *, expected_identity, limits=None):
    with AnalysisReader(commit_path, expected_identity=expected_identity, limits=limits) as reader:
        return reader.read(view_id, selection)
