"""Readers validate both declared and actual native 4.0 payloads."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from qcl_negf_contracts.messages import ContractError
from qcl_negf_results.commits import artifact_row, json_bytes
from qcl_negf_results.export import export_snapshot
from qcl_negf_results.native import validate_native_handle
from qcl_negf_results.render import materialize
from native_result_fixtures import declare_native
from test_export import fixture


def _republish(generation):
    commit_path = generation / "commit.json"
    commit = json.loads(commit_path.read_bytes())
    commit["artifacts"][0] = artifact_row(generation / "analysis.h5", relative="analysis.h5",
        role="physics.analysis", schema="qcl-negf-physics-analysis-v4", media_type="application/x-hdf5")
    commit_path.write_bytes(json_bytes(commit))
    return commit, hashlib.sha256(commit_path.read_bytes()).hexdigest()


@pytest.mark.parametrize("change", ["old_version", "old_schema", "missing_markers", "legacy_selected", "missing_metadata"])
def test_new_declaration_cannot_disguise_an_old_or_incomplete_payload(tmp_path: Path, change: str) -> None:
    generation = fixture(tmp_path / "run")
    with h5py.File(generation / "analysis.h5", "a") as handle:
        if change == "old_version":
            handle["metadata"].attrs["schema_version"] = "2.0"
        elif change == "old_schema":
            handle["metadata"].attrs["schema"] = "qcl-negf-physics-analysis-v2"
        elif change == "missing_markers":
            del handle["diagnostics/physical_markers"]
        elif change == "legacy_selected":
            del handle["diagnostics/psd_history/selected_blocks"]
            handle["diagnostics/psd_history"].create_group("selected_blocks").create_group("sequence-1")
        else:
            del handle["metadata"]
    commit, digest = _republish(generation)
    # Bytes and digest are correct; a fake v3 declaration must still be rejected.
    with pytest.raises(ContractError, match="native|schema"):
        export_snapshot(tmp_path / "run", tmp_path / "exports")
    with pytest.raises(ValueError, match="native|schema"):
        materialize(generation, commit, digest, tmp_path / "render")
    assert not list((tmp_path / "exports").glob("*.tar.xz"))


@pytest.mark.parametrize("change", ["availability", "equilibrium_flag", "sequence", "sequence_duplicate"])
def test_diagnostic_coordinates_and_flags_are_required(tmp_path: Path, change: str) -> None:
    path = tmp_path / "analysis.h5"
    with h5py.File(path, "w") as handle:
        declare_native(handle, "qcl-negf-physics-analysis-v4", "physics.analysis", scba_rows=2)
        markers = handle["diagnostics/physical_markers"]
        markers["available"][:] = 1
        if change == "availability":
            markers["available"][0] = 2
        elif change == "equilibrium_flag":
            markers["equilibrium_applicable"][0] = 2
        elif change == "sequence":
            markers["sequence"][0] = 9
        else:
            markers["sequence"][:] = [1, 1]
            handle["diagnostics/psd_history/sequence"][:] = [1, 1]
            handle["diagnostics/scba/sequence"][:] = [1, 1]
    with h5py.File(path, "r") as handle, pytest.raises(ValueError, match="zero or one|sequence"):
        validate_native_handle(handle, "physics.analysis", "qcl-negf-physics-analysis-v4")


def test_checkpoint_requires_exact_current_scba_columns(tmp_path: Path) -> None:
    path = tmp_path / "recovery.h5"
    with h5py.File(path, "w") as handle:
        declare_native(handle, "qcl-negf-checkpoint-v4", "recovery", scba_rows=1)
        del handle["convergence/scba"]
        handle["convergence/scba"] = np.zeros((1, 17))
    with h5py.File(path, "r") as handle, pytest.raises(ValueError, match="18 native columns"):
        validate_native_handle(handle, "recovery", "qcl-negf-checkpoint-v4")


@pytest.mark.parametrize("name,dtype", [("fdt_raw", np.float32), ("measured_iteration", np.int32),
                                       ("available", np.int64), ("equilibrium_applicable", np.int64)])
def test_marker_storage_requires_exact_native_precision(tmp_path: Path, name: str, dtype) -> None:
    path = tmp_path / "analysis.h5"
    with h5py.File(path, "w") as handle:
        declare_native(handle, "qcl-negf-physics-analysis-v4", "physics.analysis", scba_rows=1)
        markers = handle["diagnostics/physical_markers"]
        values = markers[name][()].astype(dtype)
        del markers[name]
        markers[name] = values
    with h5py.File(path, "r") as handle, pytest.raises(ValueError, match="incompatible native dtype"):
        validate_native_handle(handle, "physics.analysis", "qcl-negf-physics-analysis-v4")


def test_marker_native_precision_is_independent_of_endianness(tmp_path: Path) -> None:
    path = tmp_path / "analysis.h5"
    with h5py.File(path, "w") as handle:
        declare_native(handle, "qcl-negf-physics-analysis-v4", "physics.analysis", scba_rows=1)
        markers = handle["diagnostics/physical_markers"]
        for name, dtype in (("fdt_raw", ">f8"), ("measured_iteration", ">i8")):
            values = markers[name][()].astype(dtype)
            del markers[name]
            markers[name] = values
    with h5py.File(path, "r") as handle:
        validate_native_handle(handle, "physics.analysis", "qcl-negf-physics-analysis-v4")
