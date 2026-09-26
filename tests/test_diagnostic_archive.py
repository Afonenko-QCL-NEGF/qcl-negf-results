"""Diagnostic and scientific snapshots share the verified multipart transport."""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import random
import tarfile

import pytest

from qcl_negf_contracts.messages import ContractError
from qcl_negf_results.diagnostic_archive import export_diagnostics
from qcl_negf_results.multipart import receive


def capture_payload(payload: bytes):
    def capture(sink):
        for name in ("job/results/telemetry.log", "job/results/identical.log"):
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            sink.addfile(info, io.BytesIO(payload))
        return {"job_id": "0123456789abcdef", "warnings": ["live file prefix"]}
    return capture


def test_diagnostics_paginate_restore_and_deduplicate_with_shared_receiver(tmp_path: Path) -> None:
    payload = random.Random(82).randbytes(36_000)
    output = tmp_path / "diagnostics"
    receipt = export_diagnostics(output, capture_payload(payload), label="test-evidence", maximum_bytes=10_000)
    parts = [output / part["filename"] for part in receipt["parts"]]
    assert len(parts) >= 4
    assert all(path.stat().st_size <= 10_000 for path in parts)
    assert receive(list(reversed(parts)), tmp_path / "restored", receipt=receipt)["verified"] is True
    manifest = json.loads((tmp_path / "restored/manifest.json").read_bytes())
    assert manifest["profile"] == "diagnostic"
    assert len(manifest["files"]) == 1
    assert len(manifest["records"][0]["included"]) == 2
    source = manifest["files"][0]
    assert source["sha256"] == hashlib.sha256(payload).hexdigest()
    assert (tmp_path / "restored" / source["path"]).read_bytes() == payload
    assert receipt["source_payload_bytes"] == len(payload) * 2


def test_diagnostic_quota_failure_never_publishes_partial_capture(tmp_path: Path) -> None:
    with pytest.raises(ContractError, match="storage budget"):
        export_diagnostics(tmp_path / "diagnostics", capture_payload(b"0123456789"),
                           label="test", maximum_source_bytes=15)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("maximum", [True, 0, 200_000_001])
def test_shared_writer_rejects_invalid_diagnostic_part_limit(tmp_path: Path, maximum) -> None:
    with pytest.raises(ContractError, match="200000000"):
        export_diagnostics(tmp_path / "diagnostics", capture_payload(b"small"), label="test",
                           maximum_bytes=maximum)
    assert list(tmp_path.iterdir()) == []


def test_captured_stream_truncation_never_publishes(tmp_path: Path) -> None:
    def truncated(sink):
        info = tarfile.TarInfo("job/result.log")
        info.size = 20
        sink.addfile(info, io.BytesIO(b"short"))
        return {}
    with pytest.raises(OSError, match="ended during capture"):
        export_diagnostics(tmp_path / "diagnostics", truncated, label="test")
    assert list(tmp_path.iterdir()) == []
