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

# Point scope is explicit; a point reader cannot invent plan-wide coverage.
from result_v2_fixture import IDENTITY, source, record, producer_root, producer_state, export_source
from qcl_negf_results.state import StateReader


def test_selected_source_never_claims_missing_plan_points_are_complete(tmp_path):
    root, output = tmp_path / 'source', tmp_path / 'export'
    ref = source(root, [record()])
    receipt = export_source(root, ref, output)
    with tarfile.open(output / receipt['archive'], 'r:xz') as archive:
        manifest = json.load(archive.extractfile('manifest.json'))
    assert manifest['coverage']['scope'] == 'explicit_selected_state'
    assert manifest['complete'] is False and manifest['scientific_accepted'] is None
    assert 'all_planned_points_recorded' not in manifest['coverage']


def test_foreign_execution_or_orphan_history_is_not_discovered(tmp_path):
    root, output = tmp_path / 'source', tmp_path / 'export'
    ref = source(root, [record()])
    first = export_source(root, ref, output)
    foreign = root / 'executions/foreign/point/attempt-999'
    foreign.mkdir(parents=True)
    (foreign / 'current.json').write_text('{invalid orphan')
    assert export_source(root, ref, output) == first


def test_selected_execution_cannot_be_reassigned_by_manifest_identity(tmp_path):
    root = tmp_path / 'source'
    ref = source(root, [record()])
    ref['identity']['execution_id'] = 'foreign'
    with pytest.raises(ContractError):
        StateReader(root, state_ref=ref, expected_identity=ref['identity'])
