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

# Historical helper retained for consumers pending their owning migration.
# The supported reader binds the exact canonical plan/model hash in Identity.
from result_v2_fixture import IDENTITY, source, record, reference, put, export_source
from qcl_negf_results.state import StateReader
import copy


@pytest.mark.parametrize('field', ['plan_sha256','model_sha256'])
def test_selected_frozen_identity_rejects_foreign_plan_or_model(tmp_path, field):
    ref = source(tmp_path, [record()])
    expected = copy.deepcopy(IDENTITY)
    expected[field] = 'c'*64
    with pytest.raises(ContractError) as error:
        StateReader(tmp_path, state_ref=ref, expected_identity=expected)
    assert error.value.code == 'identity_mismatch'


def test_original_source_manifest_bytes_are_preserved_exactly(tmp_path):
    root, output = tmp_path / 'source', tmp_path / 'export'
    ref = source(root, [record()])
    value = json.loads((root / ref['manifest']['path']).read_text())
    raw = (json.dumps(value, indent=3)+'\n\n').encode()
    (root / ref['manifest']['path']).write_bytes(raw)
    ref['manifest'] = reference(root, ref['manifest']['path'], value['schema'])
    receipt = export_source(root, ref, output)
    with tarfile.open(output / receipt['archive'], 'r:xz') as archive:
        manifest = json.load(archive.extractfile('manifest.json'))
        row = next(row for row in manifest['source_locator'] if row['reference'] == ref['manifest'])
        assert archive.extractfile(row['object']).read() == raw
    (root / ref['manifest']['path']).write_bytes(raw+b' ')
    with pytest.raises(ContractError):
        export_source(root, ref, tmp_path / 'bad')


@pytest.mark.parametrize('override', [{'plan':{}}, {'plan':b'{"schema":'},
    {'plan_source':'other'}, {'job_status':'completed'}])
def test_caller_plan_or_process_override_cannot_relabel_selected_source(tmp_path, override):
    ref = source(tmp_path / 'source', [record()])
    with StateReader(tmp_path / 'source', state_ref=ref, expected_identity=IDENTITY) as reader:
        with pytest.raises(TypeError):
            export_snapshot(reader, tmp_path / 'bad', profile='science',
                byte_budget=1024*1024, reserve_bytes=0, **override)
    assert not (tmp_path / 'bad').exists()
