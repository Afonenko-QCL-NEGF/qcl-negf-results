"""Provenance of an explicitly selected canonical native5 source.

Original request coordinates remain tagged observations; transport identifiers,
field/period reconstruction and scientific acceptance are never inferred.
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Mapping

import h5py
from qcl_negf_contracts.artifacts import digest_value
from qcl_negf_contracts.messages import ContractError
from .commits import decode_json

from .state import StateReader, Document, ResultReadLimits, _IOBudget, _BudgetStream, _internal_storage


def _selected_source_provenance(reader, *, observations=None):
    """Manifest proof, optionally enriched by the verified O1 description."""
    source = reader.source
    result = dict(source=source.state, invocation=source.invocation, lineage=source.lineage,
        git_graph=copy.deepcopy(reader._manifest['provenance']),
        identity=dict(status='matched', scope='verified_selected_manifest',
            fields={name:'matched' for name in source.state['identity']}),
        native_verified_references=source.integrity['selected_payloads_verified'],
        coordinates=dict(status='not_available', reason='explicit_final_observations_ref_required'),
        transport_provenance=dict(status='not_recorded', reason='not_scientific_identity'))
    if observations is not None:
        value = observations.declaration
        checks = value['request']['coordinate_checks']
        result.update(coordinates={name:dict(copy.deepcopy(checks[key]),
                source='final_observations.request.coordinate_checks.'+key)
            for name,key in (('temperature_K','temperature'), ('voltage_per_period_V','voltage'))},
            request=copy.deepcopy(value['request']),
            original_evidence=copy.deepcopy(observations.original_evidence),
            current_evidence=copy.deepcopy(value['summary']),
            restoration=copy.deepcopy(value['restoration']),
            original_inputs=dict(plan=copy.deepcopy(value['request']['plan']),
                                 model=copy.deepcopy(value['request']['model'])))
    return result


def verify_native_provenance(reader: StateReader, *, observations_ref: Document | None = None) -> dict:
    """Verify an exact O1 context, or report a manifest-only unbound context."""
    if not isinstance(reader, StateReader):
        raise TypeError('explicit selected StateReader required')
    from .final_observations import FinalObservationsReader
    with reader._consumer_operation():
        observations = None
        if observations_ref is not None:
            with FinalObservationsReader(reader, artifact_ref=observations_ref) as view:
                observations = view.describe()
        for key,item in reader._handles.items():
            reader._check_hdf_owner(key,item)
        return _selected_source_provenance(reader, observations=observations)


# Optical-v4 has its own existing embedded receipt contract. This narrow API
# never admits retired generic native containers or claims a native5 binding.
IDENTITY_FIELDS = ("point_id", "execution_id", "attempt", "plan_fingerprint")


def verify_optical_provenance(path: str | Path, owner: Mapping[str, Any]) -> dict:
    """Validate only embedded optical-v4 receipt/quality declarations.

    The bounded same-open file is not a canonical source or scientific oracle.
    Ownership still requires the separate published optics/source integration.
    """
    from ._capture_io import regular, stamp
    limit = ResultReadLimits().maximum_metadata_bytes
    with regular(Path(path)) as raw:
        before = stamp(raw, include_change=True)
        if before[2] > limit:
            raise ContractError('optical container exceeds metadata allowance', 'budget_exceeded')
        stream = _BudgetStream(raw, _IOBudget(limit))
        with h5py.File(stream, 'r', rdcc_nbytes=1024*1024) as handle:
            _internal_storage(handle)
            if 'metadata' not in handle or not isinstance(handle['metadata'],h5py.Group):
                raise ContractError('optical container metadata required', 'corrupt_result')
            attrs = handle['metadata'].attrs
            def text(value):return value.decode('utf-8') if isinstance(value,bytes) else value
            if (text(attrs.get('schema')) != 'qcl-negf-optical-v4'
                    or text(attrs.get('schema_version')) != '4.0'
                    or text(attrs.get('artifact_role')) != 'physics.analysis'
                    or text(attrs.get('contract_set')) != 'qcl-negf.results.v2'):
                raise ContractError('optical container schema differs', 'unsupported_format')
            result = _optical_provenance(handle, owner)
        if stamp(raw, include_change=True) != before:
            raise ContractError('optical container changed during metadata read', 'source_changed')
        with regular(Path(path), size=before[2]) as current:
            if stamp(current, include_change=True) != before:
                raise ContractError('optical container path changed', 'source_changed')
    result['integrity'] = dict(scope='embedded_optical_v4_receipt_metadata',
        native5_source_verified=False, scientific_acceptance='not_assessed')
    return result


def _json_scalar(handle: Any, path: str) -> Any:
    dataset = handle[path]
    if not isinstance(dataset, h5py.Dataset) or dataset.shape != ():
        raise ContractError("native identity metadata must be a scalar JSON string", "corrupt_result")
    payload = dataset[()]
    if isinstance(payload, str):
        payload = payload.encode("utf-8")
    if not isinstance(payload, bytes):
        raise ContractError("native identity metadata must be UTF-8 JSON", "corrupt_result")
    return decode_json(b'{"value":' + payload + b'}', source="native identity metadata")["value"]


def _identity_fields(source: Any, owner: Mapping[str, Any], *, cumulative: bool = False) -> set[str]:
    if not isinstance(source, dict):
        raise ContractError("native identity metadata must be an object", "corrupt_result")
    matched: set[str] = set()
    for field in IDENTITY_FIELDS:
        if field not in source or field not in owner:
            continue
        observed, expected = source[field], owner[field]
        if field == "attempt":
            if (type(observed) is not int or type(expected) is not int or observed < 1
                    or not (observed <= expected if cumulative else observed == expected)):
                raise ContractError("native identity attempt differs from its owning commit cutoff", "corrupt_result")
        elif (not isinstance(observed, str) or not isinstance(expected, str)
              or not observed or observed != expected):
            raise ContractError(f"native identity {field} differs from its owning commit", "corrupt_result")
        matched.add(field)
    return matched


def _state_identity(source: Any, owner: Mapping[str, Any]) -> None:
    if "state_id" in owner and (not isinstance(source, dict) or source != owner):
        raise ContractError("native state identity differs from its owning commit", "corrupt_result")


def _optical_provenance(handle: Any, owner: Mapping[str, Any]) -> dict[str, Any]:
    source_path = "/metadata/source_state_receipt_json"
    quality_path = "/metadata/stationary_quality_json"
    if source_path not in handle or quality_path not in handle:
        raise ContractError("optical artifact requires embedded source and stationary quality receipts",
                            "corrupt_result")
    source, quality = _json_scalar(handle, source_path), _json_scalar(handle, quality_path)
    if not isinstance(source, dict) or not isinstance(quality, dict):
        raise ContractError("optical source and stationary quality must be objects", "corrupt_result")
    identity = source.get("identity")
    _state_identity(identity, owner)
    matched = _identity_fields(identity, owner)
    if not set(IDENTITY_FIELDS) <= matched:
        raise ContractError("optical source identity is incomplete", "corrupt_result")
    digest_value(source.get("commit_sha256"))
    if (not isinstance(source.get("state_id"), str) or not source["state_id"]
            or type(source.get("state_sequence")) is not int or source["state_sequence"] < 1):
        raise ContractError("optical source state identity is invalid", "corrupt_result")
    if "state_id" in owner and any(source[key] != owner[key] for key in ("state_id", "state_sequence")):
        raise ContractError("optical source state coordinates differ from its owner", "corrupt_result")
    accepted = quality.get("scientific_accepted")
    if type(accepted) is not bool:
        raise ContractError("optical stationary quality must explicitly declare scientific acceptance",
                            "corrupt_result")
    if accepted and (quality.get("iterative_converged") is not True
                     or quality.get("physical_gates_passed") is not True
                     or quality.get("discretization_verified") not in (True, "verified")):
        raise ContractError("optical stationary quality cannot treat unknown evidence as acceptance",
                            "corrupt_result")
    return {"source_state": source, "stationary_quality": quality}
