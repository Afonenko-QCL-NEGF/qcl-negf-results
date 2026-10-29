"""Bounded read-only semantic views of one committed analytical projection.

This is neither StateReader nor full native/scientific validation. Provider code
owns exact remote selection, copying, authentication and temporary-file lifetime.
"""
from __future__ import annotations

from contextlib import contextmanager
import copy
import struct
from dataclasses import dataclass, replace
import json
import math
import os
import stat
import sys
from typing import Mapping, Sequence

import numpy as np

from qcl_negf_contracts.messages import ContractError, scientific_plan
from .state import StateReader, Document, _same_json_value
from .final_observations import FinalObservationsReader


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


# view_id -> exact O1 field, logical axes, units, basis, diagonal, component.
VIEWS = {
    **{f"potential_{name}": (f"potential_{name}_eV", ("z",), "eV", None, False, None)
       for name in ("structure", "external", "hartree", "total")},
    "density": ("density_per_m3", ("z",), "m^-3", None, False, None),
    "spatial_energy_density": ("spatial_energy_density_per_eV_m3", ("E", "z"), "eV^-1 m^-3", None, False, None),
    **{name: (field, ("E","k"), "eV^-1", None, False, None) for name,field in
       (("spectral_trace","spectral_trace_per_eV"),("occupied_trace","occupied_trace_per_eV"))},
    **{f"{basis}_levels": (f"{basis}_levels_eV", ("state",), "eV", basis, False, None)
       for basis in ("localized","effective")},
    **{f"{basis}_wavefunctions_{component}": (f"{basis}_wavefunctions_per_sqrt_nm", ("z","state"), "nm^-1/2", basis, False, component)
       for basis in ("localized","effective") for component in ("real","imag")},
    **{f"{basis}_density_matrix_{component}": (field, ("state","state_prime"), "m^-2", basis, False, component)
       for basis,field in (("localized","sheet_density_matrix_per_m2"),("effective","effective_density_matrix_per_m2"))
       for component in ("real","imag")},
    **{f"{basis}_populations": (field, ("state","state_prime"), "m^-2", basis, True, None)
       for basis,field in (("localized","sheet_density_matrix_per_m2"),("effective","effective_density_matrix_per_m2"))},
}

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


class AnalysisReader:
    """Semantic projection borrowing exactly one selected native5 StateReader."""
    def __init__(self, reader: StateReader, *, artifact_ref: Document,
                 limits: SavedViewLimits | None = None):
        if not isinstance(reader, StateReader):
            raise TypeError("explicit selected StateReader required")
        self.reader, self.limits = reader, limits or SavedViewLimits()
        if not isinstance(self.limits, SavedViewLimits):
            raise TypeError("limits must be SavedViewLimits")
        self.raw = FinalObservationsReader(reader, artifact_ref=artifact_ref)
        self._closed = False

    def __enter__(self):
        if self._closed: raise ValueError("saved view is closed")
        return self

    def __exit__(self, *args): self.close()

    def close(self):
        self._closed = True
        self.raw.close()

    @contextmanager
    def _operation(self):
        if self._closed: raise ValueError("saved view is closed")
        reader, limits = self.reader, self.limits
        original = reader.limits
        reader.limits = replace(original,
            maximum_metadata_bytes=min(original.maximum_metadata_bytes, limits.maximum_metadata_bytes),
            maximum_array_bytes=min(original.maximum_array_bytes, limits.maximum_selected_bytes),
            maximum_hdf_objects=min(original.maximum_hdf_objects, limits.maximum_objects))
        try:
            with reader._consumer_operation(limits.maximum_io_bytes,
                    maximum_owner_bytes=limits.maximum_owner_bytes): yield
        except ContractError:
            raise
        except (OSError,ValueError) as error:
            raise ContractError('saved native metadata/selection cannot be read: '+str(error),'corrupt_result') from error
        finally: reader.limits = original

    def _publication_guard(self, source, references):
        # Semantic conversion/cache work happens after raw ArrayBlock checks.
        # Recheck actual retained owner FDs immediately before public return;
        # this is stat/FD identity checking, never a warm full-file SHA rescan.
        if self.reader.source.state != source:
            _error('source_changed','selected source changed before semantic publication')
        for index,reference in enumerate(references):
            key=(reference['path'],reference['sha256'])
            item=self.reader._handles.get(key)
            if item is None:
                if index==0: _error('source_changed','selected owner no longer retained before semantic publication')
                continue  # Optional cached metadata owner can be evicted by cap1.
            self.reader._check_hdf_owner(key,item)

    @staticmethod
    def _conventions(context, basis=None):
        names = ('basis_contract', 'energy_reference', 'effective_gauge', 'localization',
                 'boundary_condition', 'energy_shift_discretization', 'basis_id',
                 'discretization_id', 'domain_revision', 'scales')
        result = {name: copy.deepcopy(context[name]) for name in names}
        result['gauge'] = context['effective_gauge'] if basis == 'effective' else context['localization'] if basis else None
        return result

    def _base(self, description):
        value = description.declaration
        return dict(source_state=description.source, identity=copy.deepcopy(value['identity']),
            invocation=description.invocation, artifact_ref=description.artifact,
            checkpoint_ref=description.checkpoint, diagnostics_ref=description.diagnostics,
            recorded_statuses=dict(current=copy.deepcopy(value['summary']),
                original=copy.deepcopy(description.original_evidence), restoration=copy.deepcopy(value['restoration'])),
            conventions=self._conventions(value['context']), issues=copy.deepcopy(value['issues']),
            integrity=copy.deepcopy(description.integrity), limits=vars(self.limits).copy())

    def _descriptor(self, description, view_id):
        if view_id not in VIEWS: _error('missing_dataset', 'unknown semantic saved view')
        field, axes, units, basis, diagonal, component = VIEWS[view_id]
        value = description.declaration
        native = value['views'][field]
        issue = next((v for v in value['issues'] if v['target'] == field), None)
        return dict(view_id=view_id, field=field, native_descriptor=copy.deepcopy(native),
            shape=None if native is None else list(map(int,native['shape'])),
            axes=list(axes), units=units, basis=basis, component=component,
            conventions=self._conventions(value['context'], basis),
            availability='unavailable' if native is None else 'available', issue=copy.deepcopy(issue),
            allowed_selection=dict(axis_count=1 if diagonal else len(axes), index_origin=0, positive_step=True))

    def describe(self) -> Document:
        with self._operation():
            description = self.raw.describe()
            result = self._base(description)
            result['views'] = {name:self._descriptor(description,name) for name in VIEWS}
            result['io_counters'] = self.reader.io_counters
            _bounded(result,self.limits.maximum_json_bytes)
            self._publication_guard(description.source,(description.artifact,description.diagnostics))
            return result

    def brief(self) -> Document:
        with self._operation():
            description = self.raw.describe(); value = description.declaration
            result = self._base(description)
            result.update(request=copy.deepcopy(value['request']), summary=copy.deepcopy(value['summary']),
                scalar_observables={k:copy.deepcopy(value['views'][k]) for k in
                    ('current_density_A_per_m2','sheet_density_per_m2')},
                io_counters=self.reader.io_counters)
            _bounded(result,self.limits.maximum_json_bytes)
            self._publication_guard(description.source,(description.artifact,description.diagnostics))
            return result

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


    def read(self, view_id: str, selection: Sequence[int | slice | Document]) -> Document:
        with self._operation():
            description = self.raw.describe(); descriptor = self._descriptor(description, view_id)
            field, names, units, basis, diagonal, component = VIEWS[view_id]
            if descriptor['availability'] != 'available':
                _error('capability_unavailable', str(descriptor['issue']))
            native = descriptor['native_descriptor']; shape = descriptor['shape']
            selected, counts, normalized = self._normalize(selection, shape[:1] if diagonal else shape)
            if diagonal:
                item = selected[0]
                item = slice(item,item+1,1) if type(item) is int else item
                native_selected, native_counts = [item,item], [counts[0],counts[0]]
                output_names, output_items = ['state'], [item]
            else:
                native_selected, native_counts = selected, counts
                output_names, output_items = list(names), selected
            cells = math.prod(native_counts)
            if cells > self.limits.maximum_cells: _error('budget_exceeded','selected native cells exceed cap')
            for axis,count in zip(names,native_counts):
                cap = self.limits.maximum_matrix_states if diagonal or 'density_matrix' in view_id else self.limits.maximum_states
                if axis in ('state','state_prime') and count > cap:
                    _error('budget_exceeded','explicit state subset exceeds cap')
            if sum(type(v) is not int for v in native_selected) <= 1 and max(native_counts) > self.limits.maximum_line_points:
                _error('budget_exceeded','line selection exceeds native point cap')
            coordinate_fields = {'z':'z_nm','E':'energy_eV','k':'k_per_nm','state':'state_index','state_prime':'state_index'}
            weight_fields = {'z':'wX','E':'wE','k':'wk'}
            context = description.declaration['context']; vectors=[]
            # Source complex block, native vectors, diagonal/fraction arrays and
            # finite JSON output construction coexist; reject before payload.
            required = 4096+cells*(16 if native['eltype']=='ComplexF64' else 8)
            output_shape=(counts[0],) if diagonal else tuple(count for count,item in zip(counts,selected) if type(item) is not int)
            output_retained,output_peak=_finite_buffers(output_shape)
            required+=output_peak
            if diagonal: required+=16*counts[0]+output_retained+output_peak
            for axis,item in zip(output_names,output_items):
                count = 1 if type(item) is int else len(range(item.start,item.stop,item.step))
                vector_peak=8*count+_finite_buffers((count,))[1]
                required+=vector_peak*(2 if axis in weight_fields else 1)
                vectors.append((axis,item,count))
            cap = min(self.limits.maximum_selected_bytes,self.reader.limits.maximum_array_bytes)
            if required > cap: _error('budget_exceeded','selected complex data, output buffers, coordinates and weights exceed byte cap')
            axes=[]
            for axis,item,count in vectors:
                coordinate_name=coordinate_fields[axis]; coordinate=context[coordinate_name]
                values=self.raw.read_array(coordinate_name,(item,),maximum_bytes=cap).values
                finite,invalid=_finite(np.atleast_1d(values))
                if any(v is not None for v in invalid): _error('corrupt_result','saved coordinates must be finite')
                weight_name=weight_fields.get(axis); weight=context[weight_name] if weight_name else None
                weights,weight_invalid=(None,None) if weight is None else _finite(np.atleast_1d(
                    self.raw.read_array(weight_name,(item,),maximum_bytes=cap).values))
                axes.append(dict(name=axis,values=finite,units=coordinate['units'],source_path=coordinate['dataset'],
                    weights=weights,weight_invalid_mask=weight_invalid,weight_descriptor=copy.deepcopy(weight),
                    integration_measure='stored_dimensionless_quadrature' if weight else None,
                    physical_measure=dict(scales=copy.deepcopy(context['scales']),axis=axis) if weight else None))
            block=self.raw.read_array(field,tuple(native_selected),maximum_bytes=cap)
            values=block.values
            if component is not None: values=getattr(values,component)
            if diagonal: values=np.diag(values.real).copy()
            finite,invalid=_finite(values); fractions=None
            if diagonal:
                inputs=context['inputs']; donor,ionization=(inputs[k] for k in ('N_dop_2D_per_m2','f_ion'))
                denominator=_ieee_value(donor)*_ieee_value(ionization)
                if math.isfinite(denominator) and denominator > 0:
                    fraction_values,fraction_invalid=_finite(values/denominator)
                    fractions=dict(values=fraction_values,invalid_mask=fraction_invalid,reason=None,
                        denominator=dict(value=denominator,units='m^-2',inputs=copy.deepcopy(dict(N_dop_2D_per_m2=donor,f_ion=ionization)),
                            rule='ionized_sheet_density = saved donor_sheet_density * saved ionization_fraction'))
                else: fractions=dict(values=None,invalid_mask=None,denominator=None,reason='saved_target_doping_nonpositive_or_invalid')
            result=self._base(description)
            result.update(view_id=view_id,normalized_selection=normalized,native_shape=shape,returned_shape=list(np.shape(values)),
                logical_axes=[name for name,item in zip(output_names,output_items) if type(item) is not int],
                axes=axes,values=finite,invalid_mask=invalid,units=units,basis=basis,conventions=descriptor['conventions'],
                sampling=dict(kind='exact_native_selection',step=[item.step if isinstance(item,slice) else None for item in selected]),
                fractions=fractions,raw_selection=dict(artifact=copy.deepcopy(description.artifact),dataset=native['dataset'],
                    selection=[dict(start=v.start,stop=v.stop,step=v.step) if isinstance(v,slice) else v for v in native_selected]),
                io_counters=self.reader.io_counters)
            _bounded(result,self.limits.maximum_json_bytes)
            self._publication_guard(description.source,(description.artifact,description.diagnostics))
            return result


SavedAnalysis = AnalysisReader


def _ieee_value(value):
    return struct.unpack('>d',bytes.fromhex(value['bits']))[0]

def _finite_buffers(shape):
    """Retained/peak Python finite-projection bytes on this runtime.

    _finite builds one temporary list of (value,mask) tuples and then two
    result lists at every dimension. list_resize's n+n/8+6 bound includes
    comprehension over-allocation. Scalar36 covers all fixed Int64/Float64
    values; shared None/reason constants are already live. Native array bytes
    and a separate 4096-byte bounded scalar/view scratch reserve are external.
    """
    scalar=max(sys.getsizeof(2**63-1),sys.getsizeof(-2**63),sys.getsizeof(0.))
    pair=sys.getsizeof((None,None));header=sys.getsizeof([]);pointer=struct.calcsize('P')
    def estimate(remaining):
        if not remaining:return scalar,scalar+pair
        n=remaining[0];child,peak=estimate(remaining[1:])
        listing=header+pointer*(n+(n>>3)+6)
        retained=2*listing+n*child
        construction=listing+max(0,n-1)*(pair+child)+peak
        final=3*listing+n*(pair+child)
        return retained,max(construction,final)
    return estimate(tuple(shape))


def _finite(value):
    """JSON null plus reason masks distinguish native invalids from finite zero."""
    if isinstance(value, np.ndarray) and value.ndim:
        rows = [_finite(item) for item in value]
        return [item[0] for item in rows], [item[1] for item in rows]
    if isinstance(value, np.generic) or isinstance(value, np.ndarray): value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None, "nan" if math.isnan(value) else "positive_infinity" if value > 0 else "negative_infinity"
    return value, None


def describe_saved_views(reader: StateReader, *, artifact_ref: Document, limits: SavedViewLimits | None = None) -> Document:
    with AnalysisReader(reader, artifact_ref=artifact_ref, limits=limits) as view:
        return view.describe()


def read_saved_view(reader: StateReader, view_id: str, selection: Sequence[int | slice | Document], *, artifact_ref: Document, limits: SavedViewLimits | None = None) -> Document:
    with AnalysisReader(reader, artifact_ref=artifact_ref, limits=limits) as view:
        return view.read(view_id, selection)
