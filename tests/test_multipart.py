"""Real small-cap transport: finalized byte bounds, immutable data and receiver."""
from __future__ import annotations

import hashlib
import json
import lzma
from pathlib import Path
import subprocess
import sys
import tarfile

import numpy as np
import pytest

from qcl_negf_contracts.artifacts import SCIENCE_MAX_BYTES
from qcl_negf_contracts.messages import ContractError
from qcl_negf_results.commits import artifact_row, json_bytes
from qcl_negf_results.export import export_snapshot
from qcl_negf_results.multipart import receive
from test_export import fixture


def noisy_snapshot(root: Path) -> tuple[Path, bytes]:
    generation = fixture(root)
    rng = np.random.default_rng(121)
    payload = json_bytes({"values": rng.integers(0, 2**32, size=8_000).tolist()})
    source = generation / "evidence.json"
    source.write_bytes(payload)
    path = generation / "commit.json"
    commit = json.loads(path.read_bytes())
    commit["artifacts"].append(artifact_row(source, relative=source.name, role="plan",
        media_type="application/json"))
    path.write_bytes(json_bytes(commit))
    return generation, payload


def part_paths(receipt: dict, output: Path) -> list[Path]:
    return [output / (row["sha256"] + ".tar.xz") for row in receipt["parts"]]


@pytest.mark.parametrize("profile", ["science", "full-state"])
def test_actual_final_parts_bounded_and_restore_exact_objects(tmp_path: Path, profile: str) -> None:
    source, payload = noisy_snapshot(tmp_path / "run")
    updates = []
    output = tmp_path / "exports"
    receipt = export_snapshot(tmp_path / "run", output, profile=profile,
                              maximum_bytes=10_000, progress=updates.append)
    parts = part_paths(receipt, output)
    assert len(parts) >= 4 and receipt["multipart"] is True
    assert all(path.stat().st_size <= 10_000 for path in parts)
    assert receipt["total_archive_bytes"] == sum(path.stat().st_size for path in parts)
    assert receipt["bytes"] == parts[0].stat().st_size
    with tarfile.open(parts[0], "r:xz") as archive:
        assert {"manifest.json", "export-index.json", "README.md", "part.json"} <= set(archive.getnames())
        index = json.load(archive.extractfile("export-index.json"))
    digest = hashlib.sha256(payload).hexdigest()
    row = next(row for row in index["objects"] if row["sha256"] == digest)
    assert row["transport"] == "chunks" and len(row["segments"]) > 1
    assert row["segments"][0]["offset"] == 0
    assert sum(segment["bytes"] for segment in row["segments"]) == len(payload)
    restored = tmp_path / "restored"
    result = receive(list(reversed(parts)), restored, receipt=receipt)
    assert result["verified"] is True
    assert (restored / row["path"]).read_bytes() == payload
    assert (source / "evidence.json").read_bytes() == payload
    assert updates[-1]["completed_parts"] == updates[-1]["total_parts"] == len(parts)
    assert all(row["status"] == "ready" for row in receipt["parts"])


def test_corrupt_missing_duplicate_mixed_parts_never_publish_restored_tree(tmp_path: Path) -> None:
    noisy_snapshot(tmp_path / "run")
    output = tmp_path / "exports"
    receipt = export_snapshot(tmp_path / "run", output, maximum_bytes=10_000)
    parts = part_paths(receipt, output)
    for invalid in (parts[1:], parts[:-1], [*parts, parts[-1]]):
        with pytest.raises(ContractError, match="missing|duplicate"):
            receive(invalid, tmp_path / "invalid")
        assert not (tmp_path / "invalid").exists()
    damaged = tmp_path / "damaged.tar.xz"
    value = bytearray(parts[-1].read_bytes())
    value[len(value) // 2] ^= 32
    damaged.write_bytes(value)
    with pytest.raises((ContractError, lzma.LZMAError, tarfile.ReadError)):
        receive([*parts[:-1], damaged], tmp_path / "invalid", receipt=receipt)
    assert not (tmp_path / "invalid").exists()
    other = export_snapshot(tmp_path / "run", output, maximum_bytes=10_000, job_status="paused")
    with pytest.raises(ContractError, match="different snapshots"):
        receive([parts[0], *part_paths(other, output)[1:]], tmp_path / "invalid")
    assert not (tmp_path / "invalid").exists()


def test_cache_checks_every_part_and_repairs_corrupt_late_part(tmp_path: Path) -> None:
    noisy_snapshot(tmp_path / "run")
    output = tmp_path / "exports"
    first = export_snapshot(tmp_path / "run", output, maximum_bytes=10_000)
    paths = part_paths(first, output)
    assert export_snapshot(tmp_path / "run", output, maximum_bytes=10_000) == first
    paths[-1].write_bytes(b"bad late part")
    repaired = export_snapshot(tmp_path / "run", output, maximum_bytes=10_000)
    assert receive(part_paths(repaired, output), receipt=repaired)["verified"] is True


def test_cli_reassembles_without_combining_a_large_archive(tmp_path: Path) -> None:
    noisy_snapshot(tmp_path / "run")
    output = tmp_path / "exports"
    receipt = export_snapshot(tmp_path / "run", output, maximum_bytes=12_000)
    target = tmp_path / "restored"
    command = [sys.executable, "-m", "qcl_negf_results.multipart", "reassemble",
               "--destination", str(target), *map(str, reversed(part_paths(receipt, output)))]
    completed = subprocess.run(command, text=True, capture_output=True, check=True)
    assert json.loads(completed.stdout)["verified"] is True
    assert (target / "manifest.json").is_file()
    assert not list(target.rglob("*.tar*"))


@pytest.mark.parametrize("profile", ["science", "full-state"])
@pytest.mark.parametrize("maximum", [0, True, 200_000_001, 500_000_000])
def test_all_profiles_reject_limit_above_decimal_200mb(tmp_path: Path, profile: str, maximum: int) -> None:
    assert SCIENCE_MAX_BYTES == 200_000_000
    with pytest.raises(ContractError, match="200000000"):
        export_snapshot(tmp_path / "run", tmp_path / "out", profile=profile, maximum_bytes=maximum)
    assert not (tmp_path / "out").exists()


def test_first_page_provides_scalar_diagnostics_without_later_native_parts(tmp_path: Path) -> None:
    from test_export import cumulative_fixture
    root, output = tmp_path / "run", tmp_path / "export"
    cumulative_fixture(root)
    receipt = export_snapshot(root, output, maximum_bytes=10_000)
    with tarfile.open(part_paths(receipt, output)[0], "r:xz") as archive:
        diagnostics = json.load(archive.extractfile("diagnostics.json"))
        index = json.load(archive.extractfile("export-index.json"))
    assert diagnostics["snapshot_identity"] == receipt["snapshot_identity"]
    assert diagnostics["scalar_history_tails"]
    row = diagnostics["scalar_history_tails"][0]
    values = row["tables"]["scba"]["columns"]["J"]
    assert values["values"][1] == "NaN"
    assert np.signbit(values["values"][0])
    assert values["rows"] == 6 and values["units"] == "A/m^2"
    assert row["source_sha256"] in {item["sha256"] for item in index["objects"]}
    assert row["scientific_accepted"] is False


def test_live_series_advancement_during_multipart_encoding_keeps_pinned_cut(tmp_path: Path) -> None:
    root, output = tmp_path / 'run', tmp_path / 'export'
    _, payload = noisy_snapshot(root)
    advanced = False
    def progress(value: dict) -> None:
        nonlocal advanced
        if value['phase'] == 'compressing' and not advanced:
            advanced = True
            series = json.loads((root / 'series_result.json').read_bytes())
            series['points'] = [{'id': 'later-point', 'status': 'running', 'data': {}}]
            (root / 'series_result.json').write_bytes(json_bytes(series))
    receipt = export_snapshot(root, output, maximum_bytes=10_000, progress=progress)
    assert advanced and receipt['part_count'] > 1
    restored = tmp_path / 'restored'
    receive(part_paths(receipt, output), restored, receipt=receipt)
    manifest = json.loads((restored / 'manifest.json').read_bytes())
    assert [row['id'] for row in manifest['coverage']] == ['point-1']
    assert (restored / 'objects' / (hashlib.sha256(payload).hexdigest() + '.json')).read_bytes() == payload
    following = export_snapshot(root, output, maximum_bytes=10_000)
    assert following['snapshot_identity'] != receipt['snapshot_identity']
