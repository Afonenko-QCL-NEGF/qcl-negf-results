"""Exercise the real compressor and committed payloads, without giant arrays."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tarfile

import numpy as np
import pytest

from qcl_negf_contracts.messages import ContractError
from qcl_negf_results.commits import artifact_row, json_bytes
from qcl_negf_results.export import export_snapshot, _validate_container, _dataset_blocks
from test_export import fixture


def _register(generation: Path, name: str, media: str) -> None:
    path = generation / "commit.json"
    commit = json.loads(path.read_bytes())
    commit["artifacts"].append(artifact_row(generation / name, relative=name,
        role="plan", media_type=media))
    path.write_bytes(json_bytes(commit))


def test_native_json_above_100kb_is_exported_byte_for_byte(tmp_path: Path) -> None:
    root, output = tmp_path / "run", tmp_path / "exports"
    generation = fixture(root)
    payload = json_bytes({"matrix_witnesses": [{"eigenvalue": -2.34e-12, "norm": 8.80e-11}] * 5000})
    assert len(payload) > 100_000
    (generation / "witnesses.json").write_bytes(payload)
    _register(generation, "witnesses.json", "application/json")
    receipt = export_snapshot(root, output)
    digest = hashlib.sha256(payload).hexdigest()
    with tarfile.open(output / (receipt["sha256"] + ".tar.xz"), "r:xz") as archive:
        assert archive.extractfile("objects/" + digest + ".json").read() == payload


def test_raw_payload_above_450mb_fits_actual_compressed_archive(tmp_path: Path) -> None:
    root, output = tmp_path / "run", tmp_path / "exports"
    generation = fixture(root)
    # A streaming native table, larger than both the removed raw budget and the
    # old per-object readers. Never construct its whole contents in Python RAM.
    path = generation / "comparison.txt"
    chunk = b"0.125\t0.250\t0.500\n" * 8192
    count = 0
    with path.open("wb") as stream:
        while count <= 450_000_000:
            count += stream.write(chunk)
    _register(generation, path.name, "text/plain")
    progress = []
    receipt = export_snapshot(root, output, progress=progress.append)
    assert receipt["payload_bytes"] > 450_000_000
    assert receipt["bytes"] < 1_000_000
    assert receipt["snapshot_consistent"] is True
    compressed = [row for row in progress if row["phase"] == "compressing"]
    assert max(row["archive_bytes"] for row in compressed) == receipt["bytes"]
    assert all("archive_limit_bytes" not in row for row in compressed)
    processed = [row["completed_bytes"] for row in compressed]
    assert processed == sorted(processed)


@pytest.mark.parametrize("payload", [b'{"x":1,"x":2}', b'{"x":NaN}', b'{"x":1e999}',
    b'{"x":[1,]}', b'{"x":1} trailing', b'[]', b'{"x":"\xff"}'])
def test_streaming_json_retains_integrity_validation(tmp_path: Path, payload: bytes) -> None:
    path = tmp_path / "bad.json"
    path.write_bytes(payload)
    with pytest.raises(ContractError, match="invalid complete UTF-8 JSON"):
        _validate_container(path, "application/json")


def test_wide_native_dataset_is_read_in_bounded_blocks() -> None:
    from types import SimpleNamespace
    dataset = SimpleNamespace(ndim=2, shape=(3, 2_000_000), dtype=np.dtype("float64"))
    blocks = list(_dataset_blocks(dataset))
    sizes = [np.prod([part.stop - part.start for part in block]) for block in blocks]
    assert max(sizes) * 8 <= 1024 * 1024
    assert sum(sizes) == 6_000_000
