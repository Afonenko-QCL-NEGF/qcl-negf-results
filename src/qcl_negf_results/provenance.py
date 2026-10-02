"""Read-only comparison of embedded native metadata with its owning commit."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping

import h5py

from qcl_negf_contracts.messages import ContractError
from .commits import decode_json

IDENTITY_FIELDS = ("point_id", "execution_id", "attempt", "plan_fingerprint")
COORDINATE_PATHS = {
    "temperature_K": ("/metadata/temperature_K", "/inputs/T_L_K"),
    "voltage_per_period_V": ("/metadata/voltage_per_period_V", "/inputs/V_period_V"),
}


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


def _identity_report(matched: set[str], **extra: Any) -> dict[str, Any]:
    return {"status": "matched" if len(matched) == len(IDENTITY_FIELDS) else
            "partial" if matched else "not_available",
            "fields": {field: "matched" if field in matched else "not_available" for field in IDENTITY_FIELDS},
            **extra}


def verify_native_provenance(path: Path, owner: Mapping[str, Any],
                             plan: Mapping[str, Any] | None) -> dict[str, Any]:
    """Check captured original bytes before export derivations; never rewrite them."""
    with h5py.File(path, "r") as handle:
        sources: list[str] = []
        matched: set[str] = set()
        for name in ("/metadata/identity_json", "/metadata/point_identity_json"):
            if name in handle:
                matched |= _identity_fields(_json_scalar(handle, name), owner)
                sources.append(name)
        result = {"identity": _identity_report(matched, sources=sources, scope="owning_commit")}
        if "/metadata/source_segments_json" in handle:
            segments = _json_scalar(handle, "/metadata/source_segments_json")
            if not isinstance(segments, list):
                raise ContractError("native history identity inventory must be a list", "corrupt_result")
            all_fields = set(IDENTITY_FIELDS) if segments else set()
            for segment in segments:
                if not isinstance(segment, dict):
                    raise ContractError("native history identity segment must be an object", "corrupt_result")
                all_fields &= _identity_fields(segment.get("identity"), owner, cumulative=True)
            result["history_segments"] = _identity_report(all_fields, count=len(segments),
                source="/metadata/source_segments_json", attempt_policy="1..owning_commit.attempt")
        point = next((item for item in (plan or {}).get("points", [])
                      if item["id"] == owner.get("point_id")), None)
        coordinates: dict[str, Any] = {}
        for field, aliases in COORDINATE_PATHS.items():
            observations: list[dict[str, Any]] = []
            for name in aliases:
                if name not in handle:
                    continue
                dataset = handle[name]
                if (not isinstance(dataset, h5py.Dataset) or dataset.shape != ()
                        or dataset.dtype.kind != "f" or dataset.dtype.itemsize != 8):
                    raise ContractError(f"native {field} must be a Float64 scalar", "corrupt_result")
                observed = float(dataset[()])
                if not math.isfinite(observed):
                    raise ContractError(f"native {field} must be finite", "corrupt_result")
                observation = {"source": name, "observed": observed}
                if point is None:
                    observation.update(status="not_available", reason="frozen_plan_point_not_available")
                else:
                    expected = float(point[field])
                    observation["expected"] = expected
                    if observed.hex() == expected.hex():
                        observation["status"] = "matched"
                    elif field == "voltage_per_period_V":
                        observation.update(status="not_verified", reason=
                            "producer reconstructs voltage from F_bias * period_length; roundtrip comparison policy is not established")
                    else:
                        raise ContractError(f"native {field} differs from the frozen plan point", "corrupt_result")
                observations.append(observation)
            if not observations:
                coordinates[field] = {"status": "not_available", "reason": "native_scalar_absent"}
            else:
                coordinates[field] = dict(next((value for value in observations if value["status"] != "matched"), observations[0]))
                if len(observations) > 1:
                    coordinates[field]["aliases"] = observations
        result["coordinates"] = coordinates
        return result
