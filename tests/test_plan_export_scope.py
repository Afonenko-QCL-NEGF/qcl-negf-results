"""A frozen plan defines the selected export scope and missing planned records."""
import copy
import json
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
