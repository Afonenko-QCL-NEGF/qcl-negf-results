"""Exact raw close binding, independent native source attempts, literal CPU oracle."""

import copy
import hashlib

import pytest
from qcl_negf_contracts.messages import ContractError

from qcl_negf_results.capture import make_binding
from qcl_negf_results.commits import json_bytes
from qcl_negf_results.performance_views import (
    PerformanceLimits,
    _bounded_objects,
    read_performance_summary,
    validate_performance_binding,
)
from qcl_negf_results.telemetry import CpuCoverage, TelemetryWriter

SELECTION = {
    "calcjob_uuid": "job3",
    "execution_id": "ex",
    "attempt": 3,
    "plan_fingerprint": "a" * 64,
}


def test_close_return_raw_bytes_not_newer_session_and_literal_cpu_coverage(tmp_path):
    result = tmp_path / "result"
    result.mkdir()
    sentinel = b'{"scientific":"unchanged"}\n'
    (result / "series_result.json").write_bytes(sentinel)
    pins = [
        {
            "point_id": "p",
            "source_attempt": 2,
            "point_commit_path": "result/p/commit.json",
            "point_commit_sha256": "b" * 64,
        }
    ]
    writer = TelemetryWriter(tmp_path / "performance", SELECTION, {})
    for stamp, cpu in ((0, 1.0), (2_000_000_000, 4.0)):
        writer.record_resource(
            {
                "monotonic_ns": stamp,
                "wall_seconds": stamp / 1e9,
                "cpu_time_seconds": cpu,
                "cpu_count": 2,
                "allocation_id": "alloc",
                "execution_id": "ex",
                "pid": 42,
                "process_start_ticks": 7,
                "cgroup_scope": "root_julia_process_including_threads",
            }
        )
    closed = writer.close()
    newer = TelemetryWriter(
        tmp_path / "performance", {**SELECTION, "calcjob_uuid": "unrelated"}, {}
    )
    newer.close()
    raw = closed.read_bytes()
    value = make_binding(
        tmp_path,
        selection=SELECTION,
        result_path="result/series_result.json",
        point_pins=pins,
        closed_path=closed,
        capture_status={"terminal_tail": "unknown"},
    )
    assert value["commit"]["raw_utf8"].encode() == raw
    assert value["commit"]["native_path"] == str(closed.relative_to(tmp_path))
    assert (result / "series_result.json").read_bytes() == sentinel
    sidecar = tmp_path / "performance-binding.json"
    binding = {
        "selection": SELECTION,
        "sidecar_sha256": hashlib.sha256(sidecar.read_bytes()).hexdigest(),
        "point_pins": pins,
        "runner_result": value["runner_result"],
    }
    view = read_performance_summary(sidecar, expected_binding=binding)
    coverage = view["summary"]["cpu_coverage"]["allocations"][0]
    assert coverage["cpu_seconds"] == 3.0
    assert coverage["allocated_cpu_seconds"] == 4.0
    assert coverage["fraction_of_observed_capacity"] == 0.75
    with pytest.raises(ContractError):
        read_performance_summary(
            sidecar, expected_binding=binding, limits=PerformanceLimits(maximum_objects=1)
        )
    assert (
        view["scope"] == "execution_attempt"
        and view["binding"]["point_pins"][0]["source_attempt"] == 2
    )
    assert view["coverage"]["total_scheduler_grant"] == "not_measured"
    assert (
        read_performance_summary(sidecar, expected_binding=None)["reason"]
        == "source_binding_missing"
    )
    for mutate in ("foreign", "summary", "point", "result"):
        bad = copy.deepcopy(value)
        if mutate == "foreign":
            bad["selection"]["calcjob_uuid"] = "job2"
        if mutate == "summary":
            bad["summary"]["raw_utf8"] += " "
        if mutate == "point":
            bad["point_pins"][0]["source_attempt"] = 3
        if mutate == "result":
            bad["runner_result"]["sha256"] = "c" * 64
        if mutate == "result":
            sidecar.write_bytes(json_bytes(bad))
            binding["sidecar_sha256"] = hashlib.sha256(sidecar.read_bytes()).hexdigest()
            with pytest.raises(ContractError):
                read_performance_summary(sidecar, expected_binding=binding)
        else:
            with pytest.raises(ContractError):
                validate_performance_binding(
                    bad, expected_selection=SELECTION, expected_point_pins=pins
                )
    sidecar.write_bytes(b"{")
    with pytest.raises(ContractError):
        read_performance_summary(sidecar, expected_binding=binding)


def test_three_metadata_roots_share_one_object_budget():
    # Each literal root has exactly three nodes: dict, list, scalar.
    roots = ({"a": [1]}, {"b": [2]}, {"c": [3]})
    _bounded_objects(roots, 9)
    with pytest.raises(ContractError):
        _bounded_objects(roots, 8)


def test_measured_cpu_survives_unknown_capacity_and_missing_is_not_zero():
    coverage = CpuCoverage()
    for stamp, cpu in ((0, 1.0), (2_000_000_000, 4.0)):
        coverage.record({
            "allocation_id": "a", "execution_id": "ex", "pid": 42,
            "process_start_ticks": 7, "clock_domain_id": "clock",
            "monotonic_ns": stamp, "cpu_time_seconds": cpu, "cpu_count": None,
        })
    row = coverage.summary()["allocations"][0]
    assert row["cpu_seconds"] == 3.0 and row["mean_busy_logical_cpus"] == 1.5
    assert row["allocated_cpu_seconds"] is None
    assert row["fraction_of_observed_capacity"] is None
    assert row["capacity_unavailable_reason"]
    missing = CpuCoverage()
    missing.record({"allocation_id": "a", "monotonic_ns": 0, "cpu_time_seconds": None})
    row = missing.summary()["allocations"][0]
    assert row["cpu_seconds"] is None and row["cpu_unavailable_reason"]
