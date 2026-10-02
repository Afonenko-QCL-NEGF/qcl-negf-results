"""Single-file transport preserves the pinned inventory and publishes atomically."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random
import tarfile

import pytest

from qcl_negf_results.commits import artifact_row, json_bytes
from qcl_negf_results.export import export_snapshot
from test_export import fixture


def test_export_publishes_one_v3_archive_and_complete_native_inventory(tmp_path: Path) -> None:
    root, output = tmp_path / "run", tmp_path / "exports"
    generation = fixture(root)
    plan = {"frozen": {"model_identity": "fixture", "run": "e-1", "attempt": 1}}
    receipt = export_snapshot(root, output, profile="full-state", plan=plan, job_id="fixture-job")
    assert receipt["schema"] == "qcl-negf.science-export.v3"
    assert not ({"parts", "multipart", "part_count", "maximum_part_bytes"} & receipt.keys())
    archive_path = output / receipt["archive"]
    assert list(output.glob("*.tar.xz")) == [archive_path]
    assert receipt["bytes"] == archive_path.stat().st_size
    assert hashlib.sha256(archive_path.read_bytes()).hexdigest() == receipt["sha256"]
    assert receipt["filename"].startswith("fixture-job-full-state-")
    with tarfile.open(archive_path, "r:xz") as archive:
        assert "part.json" not in archive.getnames()
        manifest = json.load(archive.extractfile("manifest.json"))
        index = json.load(archive.extractfile("export-index.json"))
        assert index["schema"] == "qcl-negf.export-archive.v1"
        assert json.load(archive.extractfile("plan.json")) == plan
        assert manifest["records"][0]["identity"] == {"execution_id": "e-1", "point_id": "point-1"}
        included = manifest["records"][0]["included"]
        assert {row["role"] for row in included} == {"physics.analysis", "recovery"}
        for row in included:
            assert archive.extractfile(row["object"]).read() == (generation / row["source_path"]).read_bytes()


def test_single_archive_restores_and_rejects_receipt_or_missing_inventory(tmp_path: Path) -> None:
    from qcl_negf_results.archive import receive
    from qcl_negf_contracts.messages import ContractError
    root, output = tmp_path / "run", tmp_path / "exports"
    fixture(root)
    receipt = export_snapshot(root, output)
    path = output / receipt["archive"]
    restored = tmp_path / "restored"
    assert receive([path], restored, receipt=receipt)["verified"] is True
    manifest = json.loads((restored / "manifest.json").read_bytes())
    assert all((restored / row["path"]).stat().st_size == row["bytes"] for row in manifest["files"])
    with pytest.raises(ContractError, match="checksum"):
        receive([path], receipt={**receipt, "sha256": "0" * 64})
    damaged = tmp_path / "missing.tar.xz"
    with tarfile.open(path, "r:xz") as source, tarfile.open(damaged, "w:xz") as target:
        for member in source:
            if not member.name.startswith("objects/"):
                target.addfile(member, source.extractfile(member))
    with pytest.raises(ContractError, match="omitted"):
        receive([damaged], tmp_path / "bad")
    assert not (tmp_path / "bad").exists()


@pytest.mark.parametrize("restore", [False, True])
def test_receipt_profile_must_match_archive_before_restore(tmp_path: Path, restore: bool) -> None:
    from qcl_negf_results.archive import receive
    from qcl_negf_contracts.artifacts import validate_export_receipt
    from qcl_negf_contracts.messages import ContractError
    root, output = tmp_path / "run", tmp_path / "exports"
    fixture(root)
    receipt = export_snapshot(root, output, profile="science")
    changed = {**receipt, "profile": "full-state"}
    validate_export_receipt(changed)  # Independently valid, but labels another profile.
    destination = tmp_path / "restored" if restore else None
    with pytest.raises(ContractError, match="receipt and archive"):
        receive([output / receipt["archive"]], destination, receipt=changed)
    assert not (tmp_path / "restored").exists()


def test_manifest_schema_must_match_archive_before_restore(tmp_path: Path) -> None:
    import io
    from qcl_negf_results.archive import receive
    from qcl_negf_contracts.messages import ContractError
    root, output = tmp_path / "run", tmp_path / "exports"
    fixture(root)
    receipt = export_snapshot(root, output)
    altered = tmp_path / "wrong-schema.tar.xz"
    with tarfile.open(output / receipt["archive"], "r:xz") as source:
        members = [(member, source.extractfile(member).read()) for member in source]
    manifest = json.loads(dict((member.name, value) for member, value in members)["manifest.json"])
    manifest["schema"] = "qcl-negf.operational-evidence.v2"
    payload = json_bytes(manifest)
    index = json.loads(members[0][1])
    for row in index["metadata"]:
        if row["path"] == "manifest.json":
            row.update(bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest())
    with tarfile.open(altered, "w:xz") as target:
        for member, value in members:
            if member.name == "manifest.json":
                value = payload
            elif member.name == "export-index.json":
                value = json_bytes(index)
            member.size = len(value)
            target.addfile(member, io.BytesIO(value))
    with pytest.raises(ContractError, match="manifest.*schema"):
        receive([altered], tmp_path / "restored")
    assert not (tmp_path / "restored").exists()


def test_scientific_publication_fsyncs_final_directory_before_completed(tmp_path: Path, monkeypatch) -> None:
    import qcl_negf_results.export as exporter
    root, output = tmp_path / "run", tmp_path / "exports"
    fixture(root)
    events = []
    replace = exporter.os.replace
    def track_replace(source, target):
        result = replace(source, target)
        if Path(target).parent == output:
            events.append("receipt" if Path(target).suffix == ".json" else "archive")
        return result
    def sync(path):
        if path == output:
            events.append("directory synced")
    def progress(value):
        if value["phase"] == "completed":
            events.append("completed")
    monkeypatch.setattr(exporter.os, "replace", track_replace)
    monkeypatch.setattr(exporter, "fsync_directory", sync, raising=False)
    export_snapshot(root, output, progress=progress)
    assert events == ["archive", "receipt", "directory synced", "completed"]


def test_diagnostic_publication_fsyncs_contents_then_final_parent(tmp_path: Path, monkeypatch) -> None:
    import io
    import qcl_negf_results.diagnostic_archive as exporter
    destination = tmp_path / "evidence"
    events = []
    rename = exporter.os.rename
    def capture(sink):
        member = tarfile.TarInfo("log.txt")
        member.size = 3
        sink.addfile(member, io.BytesIO(b"log"))
        return {"run": "fixture"}
    def track_rename(source, target):
        result = rename(source, target)
        if Path(target) == destination:
            events.append("published")
        return result
    def sync(path):
        events.append("parent synced" if path == destination.parent else "contents synced")
    monkeypatch.setattr(exporter.os, "rename", track_rename)
    monkeypatch.setattr(exporter, "fsync_directory", sync, raising=False)
    exporter.export_diagnostics(destination, capture, label="fixture")
    assert events == ["contents synced", "published", "parent synced"]


@pytest.mark.parametrize("profile", ["science", "diagnostic"])
def test_publication_fsync_failure_rolls_back_new_pair(tmp_path: Path, monkeypatch, profile: str) -> None:
    import io
    import qcl_negf_results.export as exporter
    import qcl_negf_results.diagnostic_archive as diagnostics
    root, output = tmp_path / "run", tmp_path / "exports"
    fixture(root)
    def fail_sync(path):
        if profile == "diagnostic" and path != output.parent:
            return  # Exercise failure after the final directory rename.
        raise OSError(5, "directory fsync failed")
    monkeypatch.setattr(exporter, "fsync_directory", fail_sync, raising=False)
    monkeypatch.setattr(diagnostics, "fsync_directory", fail_sync, raising=False)
    def capture(sink):
        member = tarfile.TarInfo("log.txt")
        member.size = 3
        sink.addfile(member, io.BytesIO(b"log"))
        return {"run": "fixture"}
    with pytest.raises(OSError, match="directory fsync failed"):
        if profile == "science":
            export_snapshot(root, output)
        else:
            diagnostics.export_diagnostics(output, capture, label="fixture")
    assert not output.exists() or list(output.iterdir()) == []


def test_publication_fsync_failure_preserves_existing_receipt_and_cas(tmp_path: Path, monkeypatch) -> None:
    import qcl_negf_results.export as exporter
    root, output = tmp_path / "run", tmp_path / "exports"
    fixture(root)
    receipt = export_snapshot(root, output)
    receipt_path = output / f'{receipt["snapshot_identity"]}.json'
    original = receipt_path.read_bytes()
    archive_path = output / receipt["archive"]
    previous_archive = b"corrupted cached archive"
    archive_path.write_bytes(previous_archive)
    previous_paths = set(output.iterdir())
    def fail_sync(path):
        raise OSError(5, "directory fsync failed")
    monkeypatch.setattr(exporter, "fsync_directory", fail_sync, raising=False)
    with pytest.raises(OSError, match="directory fsync failed"):
        export_snapshot(root, output)
    assert receipt_path.read_bytes() == original
    assert archive_path.read_bytes() == previous_archive
    assert set(output.iterdir()) == previous_paths


def test_compression_cancellation_never_publishes_archive_or_receipt(tmp_path: Path) -> None:
    root, output = tmp_path / "run", tmp_path / "exports"
    fixture(root)
    class Cancelled(Exception):
        pass
    def progress(value: dict) -> None:
        if value["phase"] == "compressing":
            raise Cancelled
    with pytest.raises(Cancelled):
        export_snapshot(root, output, progress=progress)
    assert list(output.iterdir()) == []


def test_live_series_advancement_during_compression_keeps_pinned_prefix(tmp_path: Path) -> None:
    root, output = tmp_path / "run", tmp_path / "exports"
    fixture(root)
    advanced = False
    def progress(value: dict) -> None:
        nonlocal advanced
        if value["phase"] == "compressing" and not advanced:
            advanced = True
            (root / "series_result.json").write_bytes(json_bytes({
                "schema": "qcl-negf-series-result-v3", "contract_set": "qcl-negf.results.v1",
                "points": [{"id": "later-point", "status": "running", "data": {}}]}))
    receipt = export_snapshot(root, output, progress=progress)
    with tarfile.open(output / receipt["archive"], "r:xz") as archive:
        manifest = json.load(archive.extractfile("manifest.json"))
    assert advanced and [row["id"] for row in manifest["coverage"]] == ["point-1"]
    assert export_snapshot(root, output)["snapshot_identity"] != receipt["snapshot_identity"]


def test_single_archive_exposes_scalar_diagnostics_and_repairs_corrupt_cache(tmp_path: Path) -> None:
    from qcl_negf_results.archive import receive
    from test_export import cumulative_fixture
    import numpy as np
    root, output = tmp_path / "run", tmp_path / "exports"
    cumulative_fixture(root)
    receipt = export_snapshot(root, output)
    path = output / receipt["archive"]
    with tarfile.open(path, "r:xz") as archive:
        diagnostics = json.load(archive.extractfile("diagnostics.json"))
        index = json.load(archive.extractfile("export-index.json"))
    assert diagnostics["snapshot_identity"] == receipt["snapshot_identity"]
    row = diagnostics["scalar_history_tails"][0]
    values = row["tables"]["scba"]["columns"]["J"]
    assert values["values"][1] == "NaN" and np.signbit(values["values"][0])
    assert values["rows"] == 6 and values["units"] == "A/m^2"
    assert row["source_sha256"] in {item["sha256"] for item in index["objects"]}
    path.write_bytes(b"corrupt cache")
    repaired = export_snapshot(root, output)
    assert receive([output / repaired["archive"]], receipt=repaired)["verified"] is True


def test_storage_failure_during_archive_write_never_publishes(tmp_path: Path, monkeypatch) -> None:
    import qcl_negf_results.archive as transport
    root, output = tmp_path / "run", tmp_path / "exports"
    fixture(root)
    def full_device(*args, **kwargs):
        args[0].write_bytes(b"unfinished archive")
        raise OSError(28, "No space left on device")
    monkeypatch.setattr(transport, "_write_archive", full_device)
    with pytest.raises(OSError, match="No space left"):
        export_snapshot(root, output)
    assert list(output.iterdir()) == []


def test_receipt_publication_failure_rolls_back_new_archive(tmp_path: Path, monkeypatch) -> None:
    import qcl_negf_results.export as exporter
    root, output = tmp_path / "run", tmp_path / "exports"
    fixture(root)
    replace = exporter.os.replace
    def fail_receipt(source, target):
        if Path(target).parent == output and Path(target).suffix == ".json":
            raise OSError(28, "receipt disk full")
        return replace(source, target)
    monkeypatch.setattr(exporter.os, "replace", fail_receipt)
    with pytest.raises(OSError, match="receipt disk full"):
        export_snapshot(root, output)
    assert list(output.iterdir()) == []


def test_incompressible_export_above_200mb_stays_one_archive(tmp_path: Path) -> None:
    from qcl_negf_results.archive import receive
    root, output = tmp_path / "run", tmp_path / "exports"
    generation = fixture(root)
    source = generation / "large.txt"
    # ASCII evidence avoids an invalid scientific container and stays bounded
    # in memory. A seeded 64-character alphabet exceeds 200 MB compressed.
    generator = random.Random(79)
    alphabet = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_"
    digest = hashlib.sha256()
    with source.open("wb") as stream:
        for _ in range(270):
            raw = generator.randbytes(1024 * 1024)
            block = bytes(alphabet[value & 63] for value in raw)
            digest.update(block)
            stream.write(block)
    commit_path = generation / "commit.json"
    commit = json.loads(commit_path.read_bytes())
    commit["artifacts"].append(artifact_row(source, relative=source.name, role="plan", media_type="text/plain"))
    commit_path.write_bytes(json_bytes(commit))
    receipt = export_snapshot(root, output)
    assert receipt["bytes"] > 200_000_000
    paths = list(output.glob("*.tar.xz"))
    assert len(paths) == 1 and receipt["archive"] == paths[0].name
    assert receive(paths, receipt=receipt)["verified"] is True
    with tarfile.open(paths[0], "r:xz") as archive:
        stream = archive.extractfile("objects/" + digest.hexdigest() + ".txt")
        assert hashlib.file_digest(stream, "sha256").hexdigest() == digest.hexdigest()
    with source.open("rb") as stream:
        assert hashlib.file_digest(stream, "sha256").hexdigest() == digest.hexdigest()
