"""Schema-valid test metadata; fingerprints are markers, never Julia evidence."""
import json
from pathlib import Path


def frozen_plan():
    plan = json.loads((Path(__file__).parent / "fixtures/scientific_plan.json").read_bytes())
    original = plan["executions"][0]["id"]
    plan["executions"][0]["id"] = "e-1"
    for inclusion in plan["inclusions"]:
        inclusion["execution_ids"] = ["e-1" if item == original else item for item in inclusion["execution_ids"]]
    for point in plan["points"]:
        if point["execution_id"] == original:
            point["execution_id"] = "e-1"
    for node in plan["nodes"]:
        if node["execution_id"] == original:
            node["execution_id"] = "e-1"
    return plan
