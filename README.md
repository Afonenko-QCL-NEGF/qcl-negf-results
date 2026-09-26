# QCL-NEGF results

Scientific result inspection and export for QCL-NEGF. The package validates immutable artifact inventories, HDF5 payloads and Parquet telemetry, then creates checksummed snapshots suitable for analysis and transfer. It can run independently of the solver and AiiDA.

## Installation

Python 3.14 is supported. Install the `qcl-negf-contracts` 0.2.0 wheel from the same validated release first, then:

```console
uv venv
uv pip install /path/to/qcl_negf_contracts-0.2.0-py3-none-any.whl
uv pip install '.[test]'
.venv/bin/python -m pytest
uv build
```

`qcl-negf` selects component sources with Git submodules and resolves their native Python metadata in one generated `uv.lock`. This package itself declares ordinary versioned dependencies and does not depend on adjacent checkouts.

## Inspect and export

```console
.venv/bin/qcl-negf-results preview /data/calculation --profile science
.venv/bin/qcl-negf-results export /data/calculation /data/exports --profile science --status completed
.venv/bin/qcl-negf-results render /data/calculation/point/artifacts /data/figures
.venv/bin/qcl-negf-receive --help
```

The command prints a JSON receipt. `preview` inspects committed metadata; `export` opens and verifies the payloads. Supply the status recorded by AiiDA. The default is `running`, which does not claim the calculation has finished.

```python
from pathlib import Path
from qcl_negf_results.export import export_snapshot

receipt = export_snapshot(
    Path("calculation"), Path("exports"), profile="science", job_status="completed"
)
```

| Profile | Contents |
| --- | --- |
| `science` | Analysis arrays, scalar histories, selected diagnostic witnesses, model, plan and compacted telemetry |
| `full-state` | Science contents plus full physics and recovery state |

Exports preserve native values and provenance. Compaction validates logical payloads, including NaN bit patterns and signed zero. Diagnostic witness selection keeps per-attempt boundary and extremum records with an explicit selection proof; full scalar history remains available. Source files are not rewritten.

Telemetry identifies CPU intervals by `allocation_id` (for example, a Slurm job/step identity), with separate execution and process identities. Missing counters remain unknown. `TelemetryWriter` is a library for producers; it does not run a background machine sampler.

Every archive part is at most 200,000,000 decimal bytes. Receipts describe all parts and hashes. The receiving command verifies the complete part set before restoring objects. Archive metadata and paths are checked; incomplete, altered or unsupported input fails explicitly. Disk usage includes a temporary captured snapshot in addition to the published archive.

Only `qcl-negf.results.v1` / native HDF5 `4.0` is accepted. Missing declarations are errors. Snapshot consistency, process completion and scientific acceptance are separate fields: none is inferred from the others.

## Modules

- `export`, `multipart`, `diagnostic_archive`: verified snapshots and transport.
- `native`, `model`, `history`, `witnesses`: native payload validation and provenance.
- `telemetry`, `catalog`, `compaction`: durable typed performance records.
- `render`, `progress`, `presentation`: derived local diagnostic views. Scientific source arrays remain authoritative.

The package uses no scheduler queue or deployment service. Artifact file replacement is a durability mechanism for scientific data, not an application update mechanism.

## Tests

```console
.venv/bin/python -m pytest
```

The integration suite runs an actual Julia SCBA producer concurrently with export.
The superproject includes it in `deno task test --julia` and its full test pipeline.
To run it directly from this repository:

```console
QCL_NEGF_SOLVER_PROJECT=/path/to/qcl-negf/julia JULIA=/path/to/julia \
  .venv/bin/python -m pytest integration
```

Use the superproject's prepared Julia environment, containing `QCLNEGF`,
`QCLNEGFRunner`, HDF5 and their pinned dependencies. The runner writes the native
artifacts while the core performs numerical iterations. Missing Julia is a
failure in this explicit suite. Unit tests use clearly synthetic native fixtures;
they do not certify physical predictions. MIT licensed; see [LICENSE](LICENSE).
