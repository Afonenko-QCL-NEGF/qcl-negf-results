"""Exact small retained performance binding; never walk current/newest/catalogs."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from qcl_negf_contracts.artifacts import relative_path, validate_commit

from ._capture_io import digest, fail, integer, small_json
from .commits import decode_json

SELECTION = ("calcjob_uuid", "execution_id", "attempt", "plan_fingerprint")
SHA = re.compile("[0-9a-f]{64}")


@dataclass(frozen=True)
class PerformanceLimits:
    maximum_metadata_bytes: int = 1024 * 1024
    maximum_io_bytes: int = 4 * 1024 * 1024
    maximum_json_bytes: int = 2 * 1024 * 1024
    maximum_objects: int = 4096
    maximum_sessions: int = 16

    def __post_init__(self):
        if any(
            not integer(v, 1, self.__dataclass_fields__[k].default)
            for k, v in vars(self).items()
        ):
            raise ValueError(
                "performance limits may only tighten positive public bounds"
            )


def _same(actual, expected):
    return type(actual) is type(expected) and actual == expected


def _selection(value):
    if (
        not isinstance(value, Mapping)
        or set(value) != set(SELECTION)
        or not integer(value.get("attempt"), 1)
        or any(
            not isinstance(value.get(key), str) or not value[key]
            for key in SELECTION
            if key != "attempt"
        )
        or not SHA.fullmatch(value["plan_fingerprint"])
    ):
        fail("selection_required", "exact enclosing performance selection required")
    return dict(value)


def _embedded(row, maximum):
    if not isinstance(row, dict) or set(row) != {
        "native_path",
        "bytes",
        "sha256",
        "raw_utf8",
    }:
        fail("corrupt_result", "invalid retained native performance bytes")
    relative_path(row["native_path"])
    if not isinstance(row["raw_utf8"], str):
        fail("corrupt_result", "retained UTF-8 string required")
    raw = row["raw_utf8"].encode("utf-8")
    if (
        not integer(row["bytes"], 0, maximum)
        or len(raw) != row["bytes"]
        or digest(raw) != row["sha256"]
    ):
        fail(
            "identity_mismatch",
            "retained performance bytes differ from their exact pin",
        )
    return decode_json(raw, "retained performance"), raw


def _bounded_objects(roots, maximum_objects):
    if not integer(maximum_objects, 1, 4096):
        raise ValueError("maximum_objects may only tighten the public bound")
    pending = list(roots)
    count = 0
    while pending:
        node = pending.pop()
        count += 1
        if count > maximum_objects:
            fail("budget_exceeded", "aggregate retained metadata object cap exceeded")
        if isinstance(node, dict):
            pending.extend(node.values())
        elif isinstance(node, list):
            pending.extend(node)


def validate_performance_binding(
    value, *, expected_selection, expected_point_pins=None, maximum=1024 * 1024,
    maximum_objects=4096,
):
    expected = _selection(expected_selection)
    if (
        not isinstance(value, dict)
        or value.get("schema") != "qcl-negf.performance-binding.v1"
        or value.get("scope") != "execution_attempt"
        or not isinstance(value.get("selection"), dict)
        or not all(
            _same(value["selection"].get(key), item) for key, item in expected.items()
        )
    ):
        fail(
            "identity_mismatch",
            "performance sidecar belongs to another enclosing execution attempt",
        )
    _selection(value["selection"])
    commit, commit_raw = _embedded(value.get("commit"), maximum)
    summary, summary_raw = _embedded(value.get("summary"), maximum - len(commit_raw))
    _bounded_objects((value, commit, summary), maximum_objects)
    validate_commit(commit)
    identity = commit["identity"]
    if any(not _same(identity.get(key), item) for key, item in expected.items()):
        fail(
            "identity_mismatch",
            "native performance identity contradicts its source CalcJob",
        )
    if (
        commit.get("closed") is not True
        or summary.get("closed") is not True
        or summary.get("schema") != "qcl-negf.performance.v2"
        or not isinstance(identity.get("session_id"), str)
        or identity["session_id"] != summary.get("session_id")
        or not isinstance(summary.get("measurement_coverage"), dict)
        or commit.get("source_cursors")
        != summary["measurement_coverage"].get("sources")
        or not integer(commit.get("generation"), 1)
        or Path(value["commit"]["native_path"]).name
        != f"commit-{commit['generation']:06d}.json"
        or Path(value["summary"]["native_path"]).name
        != f"summary-{commit['generation']:06d}.json"
    ):
        fail("identity_mismatch", "retained session/generation/close/cursors differ")
    owners = [
        row for row in commit["artifacts"] if row.get("role") == "performance.summary"
    ]
    if len(owners) != 1:
        fail("corrupt_result", "unique exact performance summary required")
    owner = owners[0]
    if (
        owner.get("media_type") != "application/json"
        or owner.get("bytes") != len(summary_raw)
        or owner.get("sha256") != digest(summary_raw)
        or str(Path(value["commit"]["native_path"]).parent / owner["path"])
        != value["summary"]["native_path"]
    ):
        fail(
            "identity_mismatch",
            "retained summary does not match exact native commit declaration",
        )
    pins = value.get("point_pins")
    if not isinstance(pins, list) or len(pins) > 256:
        fail("corrupt_result", "invalid performance point pins")
    for row in pins:
        if (
            not isinstance(row, dict)
            or set(row)
            != {
                "point_id",
                "source_attempt",
                "point_commit_path",
                "point_commit_sha256",
            }
            or not isinstance(row["point_id"], str)
            or not row["point_id"]
            or not isinstance(row["point_commit_sha256"], str)
            or not SHA.fullmatch(row["point_commit_sha256"])
            or not integer(row["source_attempt"], 1)
        ):
            fail("corrupt_result", "invalid native source attempt pin")
        relative_path(row["point_commit_path"])
    if len({row["point_id"] for row in pins}) != len(pins):
        fail("corrupt_result", "duplicate performance source point pins")
    if expected_point_pins is not None and pins != expected_point_pins:
        fail(
            "identity_mismatch",
            "performance point source pins differ from independently retained metadata",
        )
    result = value.get("runner_result")
    if (
        not isinstance(result, dict)
        or set(result) != {"path", "bytes", "sha256"}
        or not integer(result["bytes"], 1, maximum)
        or not isinstance(result["sha256"], str)
        or not SHA.fullmatch(result["sha256"])
    ):
        fail("corrupt_result", "bounded runner result pin required")
    relative_path(result["path"])
    return commit, summary


def _exact(value):
    if type(value) is int and abs(value) > 9007199254740991:
        return {"kind": "integer", "decimal": str(value)}
    if isinstance(value, dict):
        return {key: _exact(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_exact(item) for item in value]
    return value


def read_performance_summary(commit_path, *, expected_binding, limits=None):
    """commit_path is the fixed retained sidecar, never a global telemetry pointer."""
    limits = limits or PerformanceLimits()
    if not isinstance(limits, PerformanceLimits):
        raise TypeError("limits must be PerformanceLimits")
    if expected_binding is None:
        return {"status": "unavailable", "reason": "source_binding_missing"}
    if not isinstance(expected_binding, Mapping) or set(expected_binding) != {
        "selection",
        "sidecar_sha256",
        "point_pins",
        "runner_result",
    }:
        fail("selection_required", "exact retained performance binding required")
    from .state import _IOBudget

    budget = _IOBudget(limits.maximum_io_bytes)
    value, raw = small_json(commit_path, limits.maximum_metadata_bytes, budget=budget)
    if digest(raw) != expected_binding["sidecar_sha256"]:
        fail("identity_mismatch", "retained performance sidecar changed")
    commit, summary = validate_performance_binding(
        value,
        expected_selection=expected_binding["selection"],
        expected_point_pins=expected_binding["point_pins"],
        maximum=limits.maximum_metadata_bytes,
        maximum_objects=limits.maximum_objects,
    )
    if value["runner_result"] != expected_binding["runner_result"]:
        fail(
            "identity_mismatch",
            "performance runner result differs from independently retained pin",
        )
    allowed = (
        "cpu_coverage",
        "max_observed_rss_bytes",
        "phase_seconds",
        "phase_calls",
        "producer_phase_seconds",
        "producer_phase_calls",
        "producer_phase_measurements",
        "phase_attribution",
        "measurement_coverage",
        "samples_dropped",
        "uncommitted_resource_batch_at_risk",
        "recorded_until_monotonic_ns",
        "phase_sum_is_not_wall_time",
        "durability",
        "warmup_seconds",
        "warmup_calls",
    )
    answer = {
        "status": "available",
        "reason": None,
        "schema": "qcl-negf.performance-summary-view.v1",
        "binding": dict(expected_binding),
        "scope": "execution_attempt",
        "identity_binding": {"native_recorded_fields": list(SELECTION)},
        "session_id": commit["identity"]["session_id"],
        "generation": commit["generation"],
        "closed": commit["closed"],
        "published_unix": commit.get("published_unix"),
        "recorded_until_monotonic_ns": summary.get("recorded_until_monotonic_ns"),
        "summary": {key: summary.get(key) for key in allowed},
        "capture_status": value.get("capture_status"),
        "units": {
            "cpu": "seconds",
            "covered_capacity": "logical_core_seconds",
            "memory": "bytes",
        },
        "coverage": {
            "scope": "root_julia_process_including_threads",
            "descendants_included": False,
            "total_scheduler_grant": "not_measured",
            "scientific_acceptance": "not_assessed",
            "jit": "not_measured",
            "terminal_tail": "may_be_unobserved",
        },
        "integrity": {
            "exact_embedded_bytes_verified": True,
            "catalogs_or_parquet_opened": False,
        },
        "limits": vars(limits),
        "io_counters": budget.counters(),
    }
    answer = _exact(answer)
    if len(json.dumps(answer, allow_nan=False).encode()) > limits.maximum_json_bytes:
        fail("budget_exceeded", "performance response exceeds cap")
    return answer
