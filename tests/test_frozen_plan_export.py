"""Preserve authoritative frozen-plan bytes and reject foreign provenance."""
import hashlib
import json
from pathlib import Path
import tarfile

import pytest

from qcl_negf_contracts.messages import ContractError
from qcl_negf_results.commits import json_bytes, publish_pointer
from qcl_negf_results.export import export_snapshot
from frozen_plan_fixture import frozen_plan
from test_export import fixture


def planned_result(root):
    generation = fixture(root)
    plan = frozen_plan()
    value = json.loads((generation / "commit.json").read_bytes())
    value["identity"]["plan_fingerprint"] = plan["fingerprint"]
    payload = json_bytes(value)
    (generation / "commit.json").write_bytes(payload)
    publish_pointer(generation.parent / "current.json", "generation-000001/commit.json", payload, 1)
    series = json.loads((root / "series_result.json").read_bytes())
    series.update(plan_fingerprint=plan["fingerprint"],
                  plan_scientific_fingerprint=plan["scientific_fingerprint"],
                  root_definition_id=plan["root_definition_id"], selected_execution_id="e-1")
    series["points"][0]["execution_id"] = "e-1"
    series["points"][0]["coordinates"] = {key: plan["points"][0][key] for key in
                                          ("temperature_K", "voltage_per_period_V", "branch", "order")}
    (root / "series_result.json").write_bytes(json_bytes(series))
    return plan, generation


def test_frozen_plan_bytes_and_source_are_preserved_and_byte_identity_is_pinned(tmp_path):
    root, output = tmp_path / "run", tmp_path / "exports"
    plan, generation = planned_result(root)
    raw = (json.dumps(plan, ensure_ascii=False, indent=3) + "\n\n").encode()
    receipt = export_snapshot(root, output, plan=raw,
                              plan_source="aiida.retrieved:result/scientific_plan.json")
    with tarfile.open(output / receipt["archive"], "r:xz") as archive:
        assert archive.extractfile("plan.json").read() == raw
        manifest = json.load(archive.extractfile("manifest.json"))
        provenance = manifest["frozen_plan"]
        assert provenance["source"] == "aiida.retrieved:result/scientific_plan.json"
        assert provenance["sha256"] == hashlib.sha256(raw).hexdigest()
        assert provenance["bytes"] == len(raw)
        assert provenance["fingerprint"] == plan["fingerprint"]
        assert "contract_set" not in json.loads(raw)
        analysis = manifest["records"][0]["included"][0]
        assert archive.extractfile(analysis["object"]).read() == (generation / analysis["source_path"]).read_bytes()
    same = export_snapshot(root, output, plan=raw,
                           plan_source="aiida.retrieved:result/scientific_plan.json")
    assert same == receipt
    reformatted = export_snapshot(root, output, plan=json_bytes(plan),
                                  plan_source="aiida.retrieved:result/scientific_plan.json")
    assert reformatted["snapshot_identity"] != receipt["snapshot_identity"]


def test_valid_mapping_plan_keeps_the_existing_keyword_api(tmp_path):
    root = tmp_path / "run"
    plan, _ = planned_result(root)
    receipt = export_snapshot(root, tmp_path / "exports", plan=plan)
    with tarfile.open(tmp_path / "exports" / receipt["archive"], "r:xz") as archive:
        assert json.load(archive.extractfile("plan.json")) == plan


@pytest.mark.parametrize("change", ["schema", "model_revision", "contract_set"])
def test_invalid_frozen_plan_schema_is_rejected_before_publication(tmp_path, change):
    root, output = tmp_path / "run", tmp_path / "exports"
    plan, _ = planned_result(root)
    plan[change] = "foreign"
    with pytest.raises(ContractError, match="scientific plan"):
        export_snapshot(root, output, plan=plan)
    assert not list(output.glob("*.tar.xz"))
    assert not list(output.glob("*.json"))


@pytest.mark.parametrize("change", ["fingerprint", "scientific_fingerprint", "execution_id", "coordinates"])
def test_plan_identity_mismatch_is_rejected_before_publication(tmp_path, change):
    root, output = tmp_path / "run", tmp_path / "exports"
    plan, _ = planned_result(root)
    if change in {"fingerprint", "scientific_fingerprint"}:
        plan[change] = "c" * 64 if plan[change] != "c" * 64 else "d" * 64
    elif change == "execution_id":
        plan["points"][0]["execution_id"] = "foreign-execution"
    else:
        plan["points"][0]["temperature_K"] += 1
    with pytest.raises(ContractError, match="plan"):
        export_snapshot(root, output, plan=json_bytes(plan))
    assert not list(output.glob("*.tar.xz"))
    assert not list(output.glob("*.json"))


@pytest.mark.parametrize("raw", [b'{"schema":', b'{"schema":"a","schema":"b"}', b'\xff'])
def test_corrupt_plan_bytes_are_rejected_before_publication(tmp_path, raw):
    root, output = tmp_path / "run", tmp_path / "exports"
    fixture(root)
    with pytest.raises(ContractError):
        export_snapshot(root, output, plan=raw)
    assert not list(output.glob("*.tar.xz"))
