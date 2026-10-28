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


def test_supported_receiver_rejects_manufactured_legacy_parts_without_publication(tmp_path):
    paths, receipt, _, _ = legacy_fixture(tmp_path / 'legacy')
    for selected in (paths, list(reversed(paths)), paths[:1], paths[1:], [*paths, paths[-1]]):
        with pytest.raises(ContractError):
            receive(selected, tmp_path / 'restored', receipt=receipt)
        assert not (tmp_path / 'restored').exists()


def test_legacy_part_marker_cannot_trigger_a_compatibility_receiver(tmp_path):
    paths, _, _, _ = legacy_fixture(tmp_path / 'legacy')
    with pytest.raises(ContractError) as error:
        receive(paths[:1], tmp_path / 'restored')
    assert error.value.code == 'unsupported_format'
    assert not (tmp_path / 'restored').exists()


def test_receiver_cli_rejects_retired_multipart_fixture(tmp_path):
    paths, _, _, _ = legacy_fixture(tmp_path / 'legacy')
    target = tmp_path / 'restored'
    completed = subprocess.run([sys.executable, '-m', 'qcl_negf_results.archive', 'reassemble',
        '--destination', str(target), *map(str, paths)], text=True, capture_output=True)
    assert completed.returncode == 2
    assert not target.exists()
