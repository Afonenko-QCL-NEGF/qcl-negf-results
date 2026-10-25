"""Closed scalar leaves only; literal ordinal/coordinate/gap oracle."""

import hashlib

import pytest
from qcl_negf_contracts.messages import ContractError
from test_diagnostic_views import IDENTITY, make_history

from qcl_negf_results.commits import json_bytes
from qcl_negf_results.live_views import describe_live_history, read_live_history

SELECTION = {
    "run_uuid": "run",
    "calcjob_uuid": "job",
    "execution_id": "ex",
    "attempt": 10,
    "point_id": "p0",
    "source_attempt": 10,
    "plan_fingerprint": "a" * 64,
}


def fixture(root):
    root.mkdir()
    rows = []
    for index, seq, nu, mu, vals in (
        (1, (100, 101, 102), (1, 2, 3), (1, 1, 1), (0.1, 0.001, float("nan"))),
        (2, (103, 104), (1, 2), (2, 2), (0.2, 0.02)),
    ):
        commit = make_history(
            root / f"part-{index}",
            sequence=seq,
            iteration=nu,
            outer=mu,
            residual=vals,
            currents=tuple(9000.0 for _ in seq),
        )
        raw = (commit.parent / "history.h5").read_bytes()
        name = f"segment-{index:06d}.h5"
        (root / name).write_bytes(raw)
        rows.append(
            {
                "path": name,
                "bytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
                "scba_rows": len(seq),
                "outer_rows": 0,
            }
        )
    index = {
        "schema": "qcl-negf-scientific-history-index-v2",
        "contract_set": "qcl-negf.results.v1",
        "identity": IDENTITY,
        "generation": 2,
        "updated_unix": 100.0,
        "scba_rows": 5,
        "outer_rows": 0,
        "segments": rows,
    }
    path = root / "index.json"
    path.write_bytes(json_bytes(index))
    return path


def test_descriptor_no_hdf_and_slice_preserves_cycles_masks_and_units(
    tmp_path, monkeypatch
):
    index = fixture(tmp_path / "history")
    import h5py

    original = h5py.File
    monkeypatch.setattr(
        h5py, "File", lambda *a, **k: pytest.fail("descriptor opened HDF")
    )
    descriptor = describe_live_history(index, expected_selection=SELECTION)
    assert descriptor["cutoff"] == {"scba_rows": 5, "outer_rows": 0}
    monkeypatch.setattr(h5py, "File", original)
    spool = tmp_path / "owned-spool"
    spool.mkdir()
    value = read_live_history(
        index,
        "scba",
        {"start": 0, "stop": 5, "step": 1, "columns": ["r_K"]},
        expected_selection=SELECTION,
        expected_index_sha256=descriptor["index_sha256"],
        expected_generation=2,
        spool=spool,
    )
    assert value["coordinates"]["sequence"] == [100, 101, 102, 103, 104]
    assert value["coordinates"]["iteration"] == [1, 2, 3, 1, 2]
    assert value["coordinates"]["source_attempt"] == [10] * 5
    assert value["columns"][0]["invalid_mask"] == [None, None, "nan", None, None]
    assert value["columns"][0]["units"] == "dimensionless"
    assert value["break_before"] == [True, False, True, True, False]
    assert value["coverage"]["scientific_acceptance"] == "not_assessed"


def test_old_digest_foreign_identity_and_symlink_fail_before_payload(tmp_path):
    path = fixture(tmp_path / "history")
    page = describe_live_history(path, expected_selection=SELECTION)
    path.write_bytes(path.read_bytes() + b" ")
    spool = tmp_path / "spool"
    spool.mkdir()
    with pytest.raises(ContractError, match="source_changed"):
        read_live_history(
            path,
            "scba",
            {"start": 0, "stop": 1, "step": 1, "columns": ["r_K"]},
            expected_selection=SELECTION,
            expected_index_sha256=page["index_sha256"],
            expected_generation=2,
            spool=spool,
        )
    assert list(spool.iterdir()) == []
    with pytest.raises(ContractError):
        describe_live_history(
            path, expected_selection={**SELECTION, "source_attempt": 11}
        )


def test_same_open_unlink_safe_rewrite_rejected_and_nofollow(tmp_path):
    from qcl_negf_results._capture_io import copy_closed

    source = tmp_path / "leaf"
    source.write_bytes(b"closed")
    expected = hashlib.sha256(b"closed").hexdigest()
    target = tmp_path / "copy"
    copy_closed(
        source,
        target,
        size=6,
        sha256=expected,
        before_copy=lambda stream: source.unlink(),
    )
    assert target.read_bytes() == b"closed"
    source.write_bytes(b"closed")
    with pytest.raises(ContractError):
        copy_closed(
            source,
            tmp_path / "changed",
            size=6,
            sha256=expected,
            before_copy=lambda stream: source.write_bytes(b"change"),
        )
    linked = tmp_path / "link"
    linked.symlink_to(source)
    with pytest.raises(ContractError):
        copy_closed(linked, tmp_path / "bad", size=6, sha256=expected)
