"""A frozen plan defines the selected export scope and missing planned records."""
import copy
import hashlib
import json
import shutil
import tarfile

import pytest

from qcl_negf_contracts.messages import ContractError
from qcl_negf_results.commits import json_bytes
from qcl_negf_results.export import export_snapshot
from test_series_commit_identity import row_and_commit, save_series


def second_point(plan, *, execution="e-1"):
    point = copy.deepcopy(plan["points"][0])
    point.update(id="point-2", execution_id=execution, order=2)
    plan["points"].append(point)
    plan["computation_count"] = 2
    plan["maximum_solver_runs"] = 2
    return point


def test_selected_execution_rejects_a_valid_row_and_commit_from_another_execution(tmp_path):
    root, output = tmp_path / "run", tmp_path / "exports"
    plan, series, commit_path, commit = row_and_commit(root)
    point = second_point(plan, execution="e-2")
    execution = copy.deepcopy(plan["executions"][0])
    execution.update(id="e-2", point_ids=["point-2"])
    plan["executions"].append(execution)
    node = copy.deepcopy(plan["nodes"][0])
    node["execution_id"] = "e-2"
    plan["nodes"].append(node)
    series["points"][0].update(id="point-2", execution_id="e-2")
    series["points"][0]["coordinates"]["order"] = point["order"]
    commit["identity"].update(point_id="point-2", execution_id="e-2")
    commit_path.write_bytes(json_bytes(commit))
    save_series(root, series)
    with pytest.raises(ContractError, match="selected execution"):
        export_snapshot(root, output, plan=json_bytes(plan))
    assert not list(output.glob("*.tar.xz"))
    assert not list(output.glob("*.json"))


@pytest.mark.parametrize("selected", ["e-1", None])
def test_missing_planned_point_is_reported_without_claiming_complete(tmp_path, selected):
    root, output = tmp_path / "run", tmp_path / "exports"
    plan, series, _, _ = row_and_commit(root)
    second_point(plan)
    plan["executions"][0]["point_ids"].append("point-2")
    series["selected_execution_id"] = selected
    save_series(root, series)
    receipt = export_snapshot(root, output, plan=json_bytes(plan), job_status="completed")
    assert receipt["job_complete"] is True
    assert receipt["complete"] is False
    with tarfile.open(output / receipt["archive"], "r:xz") as archive:
        manifest = json.load(archive.extractfile("manifest.json"))
        missing = next(row for row in manifest["missing_records"] if row["id"] == "point-2")
        assert missing == {"id": "point-2", "execution_id": "e-1", "series_section": "points",
                           "availability": "no_series_point_record"}
        assert manifest["completeness"]["all_planned_points_recorded"] is False


def test_selected_execution_preserves_but_excludes_previous_execution_history(tmp_path):
    """A same-plan E2→E1 output-root reuse must not import E2 payloads into E1."""
    root, output = tmp_path / "run", tmp_path / "exports"
    plan, series, commit_path, commit = row_and_commit(root)
    second_point(plan, execution="e-2")
    execution = copy.deepcopy(plan["executions"][0])
    execution.update(id="e-2", point_ids=["point-2"])
    plan["executions"].append(execution)
    node = copy.deepcopy(plan["nodes"][0])
    node["execution_id"] = "e-2"
    plan["nodes"].append(node)
    historic = root / "point-2/artifacts/generation-000001"
    shutil.copytree(commit_path.parent, historic)
    commit["identity"].update(point_id="point-2", execution_id="e-2")
    historic_bytes = json_bytes(commit)
    (historic / "commit.json").write_bytes(historic_bytes)
    history = copy.deepcopy(series["points"][0])
    history.update(id="point-2", execution_id="e-2")
    history["coordinates"]["order"] = 2
    history["data"]["result_commit"] = {
        "path": "point-2/artifacts/generation-000001/commit.json",
        "sha256": hashlib.sha256(historic_bytes).hexdigest()}
    series["attempt_history"] = [history]
    raw_series = (json.dumps(series, indent=3) + "\n\n").encode()
    (root / "series_result.json").write_bytes(raw_series)
    raw_plan = json_bytes(plan)
    receipt = export_snapshot(root, output, plan=raw_plan, job_status="completed")
    assert receipt["complete"] is True
    assert receipt["committed_records"] == 1
    with tarfile.open(output / receipt["archive"], "r:xz") as archive:
        manifest = json.load(archive.extractfile("manifest.json"))
        assert archive.extractfile("plan.json").read() == raw_plan
        assert [row["identity"]["point_id"] for row in manifest["records"]] == ["point-1"]
        assert [row["id"] for row in manifest["coverage"]] == ["point-1"]
        assert manifest["missing_records"] == []
        excluded = manifest["history_scope"]
        assert excluded["excluded_count"] == 1
        assert excluded["excluded"][0]["identity"] == {
            "point_id": "point-2", "execution_id": "e-2", "attempt": 1}
        assert excluded["excluded"][0]["payload_verification"] == "not_captured"
        original = manifest["series_manifest"]
        assert original["sha256"] == hashlib.sha256(raw_series).hexdigest()
        assert original["bytes"] == len(raw_series)
        assert archive.extractfile(original["object"]).read() == raw_series
        assert any(item["path"] == original["object"] for item in manifest["files"])
    # Historical payload retention is independent of the selected snapshot.
    shutil.rmtree(historic)
    assert export_snapshot(root, output, plan=raw_plan, job_status="completed") == receipt
