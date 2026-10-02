"""Independent, hand-framed legacy fixture: no current exporter generates parts."""
from __future__ import annotations

import hashlib
import io
import json
import lzma
from pathlib import Path
import random
import subprocess
import sys
import tarfile

import pytest

from qcl_negf_contracts.messages import ContractError
from qcl_negf_results.archive import receive
from qcl_negf_results.multipart import receive as legacy_receive


def canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def legacy_fixture(root: Path, *, identity: str = "a" * 64):
    root.mkdir()
    payload = random.Random(72).randbytes(12_000)
    digest = hashlib.sha256(payload).hexdigest()
    name = "objects/" + digest + ".bin"
    manifest = {"schema": "qcl-negf.science-export.v2", "contract_set": "qcl-negf.results.v1",
                "profile": "science", "snapshot_identity": identity,
                "files": [{"path": name, "bytes": len(payload), "sha256": digest}],
                "records": [{"included": [{"role": "plan", "object": name, "dependencies": []}]}]}
    manifest_payload = canonical(manifest)
    chunks = [payload[:6000], payload[6000:]]
    members = ["chunks/" + digest + "/0000000000000000.bin",
               "chunks/" + digest + "/0000000000006000.bin"]
    index = {"schema": "qcl-negf.export-parts.v1", "contract_set": "qcl-negf.results.v1",
             "snapshot_identity": identity, "profile": "science", "part_count": 3,
             "maximum_part_bytes": 10_000, "index_part": 1,
             "metadata": [{"path": "manifest.json", "bytes": len(manifest_payload),
                           "sha256": hashlib.sha256(manifest_payload).hexdigest()}],
             "objects": [{"path": name, "bytes": len(payload), "sha256": digest,
                          "transport": "chunks", "segments": [
                              {"member": members[i], "offset": i * 6000, "bytes": 6000,
                               "sha256": hashlib.sha256(chunks[i]).hexdigest(), "part": i + 2}
                              for i in range(2)]}]}
    index_payload = canonical(index)
    receipt = {"schema": "qcl-negf.science-export.v2", "contract_set": "qcl-negf.results.v1",
               "snapshot_identity": identity, "parts": []}
    paths = []
    for number in range(1, 4):
        header = {"schema": "qcl-negf.export-part.v1", "contract_set": "qcl-negf.results.v1",
                  "snapshot_identity": identity, "index_sha256": hashlib.sha256(index_payload).hexdigest(),
                  "index": number, "count": 3, "maximum_part_bytes": 10_000}
        rows = {"part.json": canonical(header)}
        if number == 1:
            rows.update({"manifest.json": manifest_payload, "export-index.json": index_payload})
        else:
            rows[members[number - 2]] = chunks[number - 2]
        path = root / f"legacy-{number}.tar.xz"
        with tarfile.open(path, "w:xz", preset=1) as archive:
            for member_name, value in rows.items():
                info = tarfile.TarInfo(member_name)
                info.size = len(value)
                archive.addfile(info, io.BytesIO(value))
        receipt["parts"].append({"index": number, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                 "bytes": path.stat().st_size})
        paths.append(path)
    return paths, receipt, name, payload


@pytest.mark.parametrize("receiver", [receive, legacy_receive])
def test_legacy_multipart_restore_exact_object_without_current_exporter(tmp_path: Path, receiver) -> None:
    paths, receipt, name, payload = legacy_fixture(tmp_path / "legacy")
    target = tmp_path / "restored"
    result = receiver(list(reversed(paths)), target, receipt=receipt)
    assert result["verified"] is True and result["part_count"] == 3
    assert (target / name).read_bytes() == payload
    assert json.loads((target / "manifest.json").read_bytes())["schema"] == "qcl-negf.science-export.v2"


def test_legacy_missing_duplicate_mixed_or_corrupt_parts_never_publish(tmp_path: Path) -> None:
    paths, receipt, _, _ = legacy_fixture(tmp_path / "legacy")
    for invalid in (paths[1:], paths[:-1], [*paths, paths[-1]]):
        with pytest.raises(ContractError, match="missing|duplicate"):
            receive(invalid, tmp_path / "invalid")
        assert not (tmp_path / "invalid").exists()
    other, _, _, _ = legacy_fixture(tmp_path / "other", identity="c" * 64)
    with pytest.raises(ContractError, match="different snapshots"):
        receive([paths[0], *other[1:]], tmp_path / "invalid")
    damaged = tmp_path / "damaged.tar.xz"
    value = bytearray(paths[-1].read_bytes())
    value[len(value) // 2] ^= 32
    damaged.write_bytes(value)
    with pytest.raises((ContractError, lzma.LZMAError, tarfile.ReadError)):
        receive([*paths[:-1], damaged], tmp_path / "invalid", receipt=receipt)
    assert not (tmp_path / "invalid").exists()


def test_receiver_cli_accepts_legacy_fixture(tmp_path: Path) -> None:
    paths, _, name, payload = legacy_fixture(tmp_path / "legacy")
    target = tmp_path / "restored"
    completed = subprocess.run([sys.executable, "-m", "qcl_negf_results.archive", "reassemble",
                                "--destination", str(target), *map(str, paths)],
                               text=True, capture_output=True, check=True)
    assert json.loads(completed.stdout)["verified"] is True
    assert (target / name).read_bytes() == payload
