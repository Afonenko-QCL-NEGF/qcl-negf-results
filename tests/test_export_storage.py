"""A committed HDF5 hash must cover every exported numerical value."""
import json

import h5py
import numpy as np
import pytest

from qcl_negf_contracts.messages import ContractError
from qcl_negf_results.commits import artifact_row, json_bytes, publish_pointer
from qcl_negf_results.export import export_snapshot
from test_export import fixture


def unowned_fixture(tmp_path, storage):
    root, output = tmp_path / "run", tmp_path / "exports"
    generation = fixture(root)
    payload = generation / "analysis.h5"
    if storage == "external":
        raw = tmp_path / "unowned.raw"
        values = np.array([3, 7, 11], dtype=np.int32)
        raw.write_bytes(values.tobytes())
        with h5py.File(payload, "r+") as handle:
            handle.create_dataset("outside", shape=(3,), dtype=values.dtype,
                external=[(str(raw), 0, h5py.h5f.UNLIMITED)])
    else:
        raw = tmp_path / "unowned.h5"
        with h5py.File(raw, "w") as handle:
            handle.create_dataset("values", data=np.array([3, 7, 11], dtype=np.int32))
        layout = h5py.VirtualLayout(shape=(3,), dtype=np.int32)
        layout[:] = h5py.VirtualSource(str(raw), "values", shape=(3,))
        with h5py.File(payload, "r+") as handle:
            handle.create_virtual_dataset("outside", layout)
    original = payload.read_bytes()
    commit_path = generation / "commit.json"
    commit = json.loads(commit_path.read_bytes())
    commit["artifacts"][0] = artifact_row(payload, relative="analysis.h5",
        role="physics.analysis", schema="qcl-negf-physics-analysis-v4",
        media_type="application/x-hdf5")
    commit_bytes = json_bytes(commit)
    commit_path.write_bytes(commit_bytes)
    publish_pointer(generation.parent / "current.json", "generation-000001/commit.json", commit_bytes, 1)
    return root, output, payload, original


@pytest.mark.parametrize("storage", ["external", "virtual"])
def test_export_rejects_numerical_storage_outside_hdf5_closure(tmp_path, storage):
    root, output, payload, original = unowned_fixture(tmp_path, storage)
    with pytest.raises(ContractError, match="external|virtual|portable"):
        export_snapshot(root, output)
    assert not output.exists() or not list(output.iterdir())
    assert payload.read_bytes() == original


@pytest.mark.parametrize("storage", ["external", "virtual"])
@pytest.mark.parametrize("restore", [False, True])
def test_receiver_rejects_legacy_archive_with_unowned_hdf5_values(tmp_path, storage, restore, monkeypatch):
    from qcl_negf_results.archive import receive
    import qcl_negf_results.state as state
    root, output, payload, original = unowned_fixture(tmp_path, storage)
    # Emulate the old producer's missing ownership gate. The receiver must
    # independently reject those previously published, hash-valid archives.
    with monkeypatch.context() as context:
        context.setattr(state, "_internal_storage", lambda handle: None)
        receipt = export_snapshot(root, output)
    destination = tmp_path / "restored" if restore else None
    with pytest.raises(ContractError, match="external|virtual|portable"):
        receive([output / receipt["archive"]], destination, receipt=receipt)
    assert destination is None or not destination.exists()
    assert not list(tmp_path.glob(".qcl-receive-*"))
    assert payload.read_bytes() == original
