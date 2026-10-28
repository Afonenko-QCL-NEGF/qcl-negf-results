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


def historical_row(series):
    row = series["points"][0]
    row.update(attempt=2, status="paused")
    row["data"].update(checkpoint_source_attempt=1, pause_reason="resource_pressure",
                       resume_kind="checkpoint", recovery_origin="last_committed_before_resource_pause")

# Canonical StateRef replaces implicit series selection. Old constructor helpers
# above remain for pending native4 consumer tests only.
from result_v2_fixture import IDENTITY, source, record, put, producer_root, producer_state
from qcl_negf_results.state import StateReader


@pytest.mark.parametrize('field,value', [('execution_id','other-execution'),('point_id','other-point'),
    ('attempt','2'),('branch_id','other-branch'),('run_id','other-run'),
    ('plan_sha256','c'*64),('model_sha256','d'*64)])
def test_state_ref_must_match_complete_explicit_identity(tmp_path, field, value):
    ref = source(tmp_path, [record()])
    expected = copy.deepcopy(IDENTITY)
    expected[field] = value
    with pytest.raises(ContractError) as error:
        StateReader(tmp_path, state_ref=ref, expected_identity=expected)
    assert error.value.code == 'identity_mismatch'


@pytest.mark.parametrize('field,value', [('state_id','foreign-state'),('state_sequence','2')])
def test_state_ref_must_match_exact_immutable_manifest(tmp_path, field, value):
    ref = source(tmp_path, [record()])
    ref[field] = value
    with pytest.raises(ContractError):
        StateReader(tmp_path, state_ref=ref, expected_identity=IDENTITY)


@pytest.mark.parametrize('name', ['child','grandchild'])
def test_explicit_older_attempt_parent_survives_exact_cutoff(producer_root, name):
    ref = producer_state(producer_root, name)
    with StateReader(producer_root, state_ref=ref, expected_identity=ref['identity']) as reader:
        reader.import_history()
        batch = reader.history_batch()
        updates = [row for row in batch.records if row['target'] is not None]
        assert updates and all(row['target']['scope'] == 'parent_checkpoint' for row in updates)
        assert reader.source.lineage['parent_state']['identity']['attempt'] != ref['identity']['attempt']


@pytest.mark.parametrize('field,value', [('branch_id','foreign'), ('model_sha256','c'*64), ('plan_sha256','d'*64)])
def test_declared_parent_cannot_cross_canonical_identity(producer_root, tmp_path, field, value):
    from result_v2_fixture import copy_constructor_closure
    ref = producer_state(producer_root, 'child')
    root = copy_constructor_closure(producer_root, tmp_path / 'source', ref)
    manifest = json.loads((root / ref['manifest']['path']).read_text())
    manifest['lineage']['parent_state']['identity'][field] = value
    ref['manifest'] = put(root, 'foreign-parent-manifest.json', manifest, manifest['schema'])
    with pytest.raises(ContractError):
        with StateReader(root, state_ref=ref, expected_identity=ref['identity']) as reader:
            reader.history_batch()
