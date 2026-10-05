"""Every series reference must identify its own pinned point and attempt."""
import copy
import hashlib
import json
from pathlib import Path
import shutil

import pytest

from qcl_negf_contracts.messages import ContractError
from qcl_negf_results.commits import json_bytes
from qcl_negf_results.export import export_snapshot
from test_frozen_plan_export import planned_result


def row_and_commit(root):
    plan, generation = planned_result(root)
    commit_path = generation / "commit.json"
    commit = json.loads(commit_path.read_bytes())
    commit["identity"]["attempt"] = 1
    commit_path.write_bytes(json_bytes(commit))
    series = json.loads((root / "series_result.json").read_bytes())
    series["points"][0]["attempt"] = 1
    return plan, series, commit_path, commit


def save_series(root, series):
    (root / "series_result.json").write_bytes(json_bytes(series))


@pytest.mark.parametrize("reference_kind", ["path", "hash"])
def test_series_point_cannot_reference_another_valid_point_in_the_same_plan(tmp_path, reference_kind):
    root, output = tmp_path / "run", tmp_path / "exports"
    plan, series, commit_path, commit = row_and_commit(root)
    second = copy.deepcopy(plan["points"][0])
    second.update(id="point-2", order=2)
    plan["points"].append(second)
    plan["computation_count"] = 2
    plan["maximum_solver_runs"] = 2
    plan["executions"][0]["point_ids"].append("point-2")
    foreign = root / "point-2/artifacts/generation-000001"
    shutil.copytree(commit_path.parent, foreign)
    commit["identity"]["point_id"] = "point-2"
    payload = json_bytes(commit)
    (foreign / "commit.json").write_bytes(payload)
    relative = str((foreign / "commit.json").relative_to(root))
    series["points"][0]["data"]["result_commit"] = relative if reference_kind == "path" else {
        "path": relative, "sha256": hashlib.sha256(payload).hexdigest()}
    save_series(root, series)
    with pytest.raises(ContractError, match="series.*commit"):
        export_snapshot(root, output, plan=json_bytes(plan))
    assert not list(output.glob("*.tar.xz"))
    assert not list(output.glob("*.json"))


@pytest.mark.parametrize("field,value", [("execution_id", "other-execution"), ("attempt", 2),
                                         ("plan_fingerprint", "c" * 64)])
def test_specific_commit_must_match_series_execution_attempt_and_plan(tmp_path, field, value):
    root, output = tmp_path / "run", tmp_path / "exports"
    _, series, commit_path, commit = row_and_commit(root)
    commit["identity"][field] = value
    commit_path.write_bytes(json_bytes(commit))
    save_series(root, series)
    with pytest.raises(ContractError, match="series.*commit"):
        export_snapshot(root, output)
    assert not list(output.glob("*.tar.xz"))


def historical_row(series):
    row = series["points"][0]
    row.update(attempt=2, status="paused")
    row["data"].update(checkpoint_source_attempt=1, pause_reason="resource_pressure",
                       resume_kind="checkpoint", recovery_origin="last_committed_before_resource_pause")


def test_declared_resource_pause_preserves_older_checkpoint_attempt(tmp_path):
    root, output = tmp_path / "run", tmp_path / "exports"
    plan, series, _, _ = row_and_commit(root)
    historical_row(series)
    save_series(root, series)
    receipt = export_snapshot(root, output, plan=json_bytes(plan))
    assert receipt["snapshot_consistent"] is True


@pytest.mark.parametrize("field,value", [("status", "running"), ("pause_reason", "other"),
    ("resume_kind", "cold_start"), ("recovery_origin", "other"),
    ("checkpoint_source_attempt", 2), ("checkpoint_source_attempt", True),
    ("checkpoint_source_attempt", 0)])
def test_historical_checkpoint_requires_exact_declared_resource_pause_lineage(tmp_path, field, value):
    root, output = tmp_path / "run", tmp_path / "exports"
    _, series, _, _ = row_and_commit(root)
    historical_row(series)
    row = series["points"][0]
    if field == "status":
        row[field] = value
    else:
        row["data"][field] = value
    save_series(root, series)
    with pytest.raises(ContractError, match="checkpoint|series.*commit"):
        export_snapshot(root, output)
    assert not list(output.glob("*.tar.xz"))


def test_attempt_history_reference_is_pinned_and_checked(tmp_path):
    root, output = tmp_path / "run", tmp_path / "exports"
    plan, series, commit_path, commit = row_and_commit(root)
    history = copy.deepcopy(series["points"][0])
    series["attempt_history"] = [history]
    next_generation = commit_path.parent.parent / "generation-000002"
    shutil.copytree(commit_path.parent, next_generation)
    commit["identity"]["attempt"] = 2
    commit["generation"] = 2
    payload = json_bytes(commit)
    (next_generation / "commit.json").write_bytes(payload)
    series["points"][0]["attempt"] = 2
    series["points"][0]["data"]["result_commit"] = str((next_generation / "commit.json").relative_to(root))
    save_series(root, series)
    receipt = export_snapshot(root, output, plan=json_bytes(plan), profile="full-state")
    assert receipt["committed_records"] == 2
    history["data"]["result_commit"] = series["points"][0]["data"]["result_commit"]
    save_series(root, series)
    with pytest.raises(ContractError, match="series.*commit"):
        export_snapshot(root, output, plan=json_bytes(plan))
