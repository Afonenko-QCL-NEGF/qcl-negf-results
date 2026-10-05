"""Real Julia SCBA producer and concurrent scientific snapshot integration."""
from __future__ import annotations

import hashlib
import io
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import time

import pytest
import h5py

from qcl_negf_results.export import export_snapshot
from qcl_negf_results.telemetry import TelemetryWriter
from qcl_negf_results.witnesses import selected_witness_records
from qcl_negf_results.native import validate_native_handle, _text
from qcl_negf_results.state import StateReader, verify_recovery_bundle

ROOT = Path(os.environ.get("QCL_NEGF_SOLVER_PROJECT", ".")).resolve()


def test_native_julia_continues_during_scientific_snapshot(tmp_path: Path) -> None:
    julia = os.environ.get("JULIA") or shutil.which("julia")
    if not julia:
        pytest.fail("live producer integration requires Julia")
    if not (ROOT / "Project.toml").is_file():
        pytest.fail("set QCL_NEGF_SOLVER_PROJECT to the installed solver environment")
    run = tmp_path / "run"
    run.mkdir()
    log_path = tmp_path / "julia.log"
    with log_path.open("wb") as log:
        process = subprocess.Popen([julia, "--startup-file=no", f"--project={ROOT}",
            str(Path(__file__).with_name("live_julia_export.jl")), str(run)],
            stdout=log, stderr=subprocess.STDOUT,
            env={**os.environ, "JULIA_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"})
        try:
            preparation_seconds = float(os.environ.get("QCL_NEGF_INTEGRATION_TIMEOUT_SECONDS", "240"))
            if not math.isfinite(preparation_seconds) or not 0 < preparation_seconds <= 900:
                raise ValueError("integration preparation timeout must be in (0, 900] seconds")
            deadline = time.monotonic() + preparation_seconds
            # Six native generations extend the first analysis. More than four
            # witnesses exercise export selection using Julia's source bounds.
            def selection_generation_published() -> bool:
                try:
                    return json.loads((run / "artifacts/current.json").read_bytes())["generation"] >= 6
                except FileNotFoundError:
                    return False
            while not selection_generation_published():
                if process.poll() is not None or time.monotonic() > deadline:
                    pytest.fail("native producer did not publish:\n" + log_path.read_text()[-12000:])
                time.sleep(0.1)
            before = len((run / "native-iterations.csv").read_text().splitlines())
            writer = TelemetryWriter(run / "performance", {"execution_id": "live-execution"}, {})
            native = (run / "native-iterations.csv").read_text().splitlines()[0].split(",")
            writer.record_span({"phase": "native_scba_callback_interval", "duration_seconds": float(native[1]),
                "measurement_kind": "interval", "warmup": True, "iteration": int(native[0])}, science_window=True)
            writer.flush()
            receipt = export_snapshot(run, tmp_path / "exports", job_status="running")
            archive_path = tmp_path / "exports" / f'{receipt["sha256"]}.tar.xz'
            assert hashlib.sha256(archive_path.read_bytes()).hexdigest() == receipt["sha256"]
            with tarfile.open(archive_path, "r:xz") as archive:
                manifest = json.load(archive.extractfile("manifest.json"))
                assert manifest["snapshot_consistent"] and not manifest["job_complete"]
                included = [item for record in manifest["records"] for item in record["included"]]
                models = [item for item in included if item["role"] == "model"]
                assert models
                for item in models:
                    model = json.load(archive.extractfile(item["object"]))
                    assert model["schema"] == "qcl-negf-resolved-configuration-v3"
                    assert type(model["configuration"]["fixture_codec"]["float"]) is float
                    assert type(model["configuration"]["fixture_codec"]["integer"]) is int
                analyses = [item for item in included if item["role"] == "physics.analysis"]
                histories = [item for item in included if item["role"] == "science.history"]
                assert len(analyses) == len(histories) == 1
                assert any(row["analysis_unix"] is not None for row in manifest["freshness"]["records"])
                selection_proofs = manifest["witness_selection"]
                assert len(selection_proofs) == 1
                selection = selection_proofs[0]
                assert selection["policy"] == "per-attempt-first-first-positive-ratio-worst-last-v1"
                assert selection["source_witnesses"] == 6
                assert 2 <= selection["selected_witnesses"] <= 4
                assert selection["omitted_witnesses"] == 6 - selection["selected_witnesses"]
                assert selection["attempts"] == [{
                    "attempt": 1, "source_witnesses": 6,
                    "selected_witnesses": selection["selected_witnesses"],
                    "omitted_witnesses": selection["omitted_witnesses"],
                    "selected_sequences": selection["selected_sequences"],
                }]
                with h5py.File(io.BytesIO(archive.extractfile(analyses[0]["object"]).read()), "r") as native_analysis:
                    assert _text(native_analysis["metadata"].attrs["schema_version"]) == "4.0"
                    assert validate_native_handle(native_analysis, "physics.analysis") == "qcl-negf-physics-analysis-v4"
                    certification = json.loads(native_analysis["metadata/physical_certification_json"][()])
                    assert certification["quantitative_physics_certified"] is False
                    assert "model_capabilities_json" in native_analysis["metadata"]
                    assert "spectral_integral_eigenvalues" in native_analysis["diagnostics/matrix_audit"]
                    psd = native_analysis["diagnostics/psd_history"]
                    assert psd["available"][0] == 1 and len(psd["selected_blocks"]) >= 1
                    assert len(native_analysis["diagnostics/physical_markers/available"]) == 6
                    owner = manifest["records"][0]["identity"]
                    assert json.loads(native_analysis["metadata/identity_json"][()]) == owner
                with h5py.File(io.BytesIO(archive.extractfile(histories[0]["object"]).read()), "r") as native_history:
                    assert _text(native_history["metadata"].attrs["schema_version"]) == "4.0"
                    assert validate_native_handle(native_history, "science.history") == "qcl-negf-scientific-history-v4"
                    assert len(native_history["physical_markers/available"]) == 6
                    sources = json.loads(native_history["metadata/source_segments_json"][()])
                    assert [(row["scba_sequence_first"], row["scba_sequence_last"]) for row in sources] == [
                        (sequence, sequence) for sequence in range(1, 7)
                    ]
                    assert all(row["identity"]["attempt"] == 1 and row["scba_rows"] == 1 for row in sources)
                    psd = native_history["psd_history"]
                    assert psd["sequence"][:].tolist() == list(range(1, 7))
                    assert psd["available"][:].tolist() == [1] * 6
                    selected = selected_witness_records(psd["selected_blocks"])
                    assert set(selected) == set(selection["selected_sequences"])
                    assert {1, 6} <= set(selected)
                    for record in selected.values():
                        assert "critical_eigenvector/real" in record.arrays
                        assert any(name.startswith("matrices_dimensionless/") for name in record.arrays)
                        assert "E0_eV" in record.attributes and "L0_m" in record.attributes
                proofs = [item["replacement_verification"] for record in manifest["records"]
                          for item in record["omitted_by_policy"] if "replacement_verification" in item]
                # The selected generation owns its complete cumulative
                # history; it has no inherited old-analysis/history parent.
                assert proofs == []
            pointer = json.loads((run / "artifacts/current.json").read_bytes())
            committed = run / "artifacts" / pointer["commit_path"]
            proof = verify_recovery_bundle(committed.parent)
            assert proof["commit"]["scientific_accepted"] is False
            with StateReader(committed) as reader:
                block = reader.read("state_dimensionless/GR/real", (slice(0, 2), 0, 0, 0),
                                    maximum_bytes=256)
                assert block.values.shape == (2,) and block.axes == ("E", "k", "a", "b")
                assert block.source_identity == proof["commit"]["identity"]
            # A successful export must leave the numerical producer running and
            # progressing; no mock file writer substitutes for Julia here.
            continuation_deadline = time.monotonic() + 15
            while len((run / "native-iterations.csv").read_text().splitlines()) <= before:
                assert process.poll() is None, log_path.read_text()[-12000:]
                assert time.monotonic() < continuation_deadline
                time.sleep(0.05)
            assert process.poll() is None
            writer.close()
            (run / "stop-request").touch()
            assert process.wait(timeout=30) == 0, log_path.read_text()[-12000:]
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)
