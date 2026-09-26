"""Compare identical raw events with per-harvest vs bounded publication.

This measures storage organization on the current filesystem, not solver speed.
"""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import time

import pyarrow.parquet as pq

from qcl_negf_results.export import export_snapshot
from qcl_negf_results.telemetry import TelemetryWriter


def benchmark(events: int = 1000) -> dict:
    measurements = {}
    with tempfile.TemporaryDirectory() as temporary:
        for mode in ("per_harvest_durable", "bounded_batch"):
            root = Path(temporary) / mode
            started = time.monotonic()
            writer = TelemetryWriter(root / "performance", {"execution_id": "benchmark"}, {})
            receive = writer.ingest_spans if mode == "per_harvest_durable" else writer.stage_spans
            for index in range(events):
                writer.record_resource({"wall_seconds": float(index), "cpu_time_seconds": index * 0.125,
                    "rss_bytes": 2**30, "execution_id": "benchmark"})
                receive([{"phase": "total", "duration_seconds": (index + 1) / 1000,
                          "iteration": index + 1, "source_sequence": index + 1}],
                        source_cursors={"timings.csv": {"offset": (index + 1) * 64}})
            writer.close()
            write_seconds = time.monotonic() - started
            files = list(writer.directory.iterdir())
            parquet = list(writer.directory.glob("*.parquet"))
            rows = sum(pq.ParquetFile(path).metadata.num_rows for path in parquet)
            assert rows == events * 2
            assert writer.source_cursors["timings.csv"]["offset"] == events * 64
            started = time.monotonic()
            receipt = export_snapshot(root, root / "exports")
            measurements[mode] = {"rows": rows, "parquet_files": len(parquet),
                "session_files": len(files), "local_session_bytes": sum(path.stat().st_size for path in files),
                "manifest_bytes": receipt["manifest_bytes"], "archive_bytes": receipt["bytes"],
                "write_seconds": write_seconds, "export_seconds": time.monotonic() - started}
    return {"events": events, "notes": "identical rows and cursor; synchronous control barriers absent; same zstd encoding in both variants", "measurements": measurements}


if __name__ == "__main__":
    print(json.dumps(benchmark(), indent=2))
