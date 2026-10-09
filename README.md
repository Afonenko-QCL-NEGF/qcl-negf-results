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

Intermediate export storage has a finite configurable [byte budget](docs/export-budget.md).
Publication is refused if the complete selected snapshot cannot fit; scientific
finals are preserved.

Bounded native state reads do not deserialize the solver:

```python
from qcl_negf_results.state import StateReader, verify_recovery_bundle

with StateReader("/data/archive/execution/point/final/commit.json") as reader:
    block = reader.read("state_dimensionless/GR/real",
                        (slice(0, 2), 0, slice(None), slice(None)),
                        maximum_bytes=1024 * 1024)

proof = verify_recovery_bundle("/data/recovery/generation-000001",
                               archive_directory="/data/archive")
```

Use the stored external axis order and one selector for every dataset axis.
The budget covers values and selected coordinates/weights. Hash verification
streams the whole file once; hyperslab access returns the selected block.
`maximum_bytes` limits returned buffers, including coordinates/weights. HDF5
chunk decompression, cache and metadata use additional RAM; this is not a limit
on the whole process.
Missing weights remain `None`. Recovery verification checks the publication
receipt, payloads and declared prior-final archive dependencies; it does not
certify application compatibility or availability from another machine.
Optical HDF5 requires embedded source-state and stationary-quality receipts;
unknown discretization/experimental evidence remains unknown.

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
    Path("calculation"), Path("exports"), profile="science", job_status="completed",
    plan=Path("calculation/scientific_plan.json").read_bytes(),
    plan_source="retrieved:scientific_plan.json",
)
```

| Profile | Contents |
| --- | --- |
| `science` | Analysis arrays, scalar histories, selected diagnostic witnesses, model, plan and compacted telemetry |
| `full-state` | Science contents plus full physics and recovery state |

Exports preserve native values and provenance. Compaction validates logical payloads, including NaN bit patterns and signed zero. Diagnostic witness selection keeps per-attempt boundary and extremum records with an explicit selection proof; full scalar history remains available. Source files are not rewritten.

Pass the authoritative frozen plan as exact `bytes` to retain its byte identity in
`plan.json`. A schema-valid mapping remains supported and is serialized to JSON.
The manifest records the source label, byte count, SHA256 and plan fingerprints;
the exact byte hash and source participate in snapshot identity. The exporter
checks the owning scientific-plan schema, execution/point membership and available
series/commit fingerprints and coordinates before publishing. Missing identity
fields are not invented: `identity_verified_against` names the checks actually
performed. Julia's canonical fingerprint computation remains the runner's
responsibility. Scientific plans do not acquire a result `contract_set` field.

Every series point or attempt-history reference is also checked against the
specific pinned commit it names. Matching membership in the same plan is
insufficient: point ID, available execution/plan identity and attempt must agree.
An older checkpoint attempt is accepted only with the runner's explicit paused
resource-pressure lineage and its exact `checkpoint_source_attempt`. Coverage
records these checks and unavailable fields; referenced historical commits are
included without treating missing historical records as missing current points.
When a selected execution is declared, current rows and included commits must
belong to that execution. The runner can retain history from other executions
when reusing one output root with the same plan (`scientific_execution.jl:394–417`).
Historical rows are validated against the whole frozen plan; references outside
the selected execution are excluded from payload capture and coverage. The
manifest's `history_scope` records their identities and count, with payload
verification explicitly `not_captured`. `series_manifest` locates the exact
original series bytes as a checksummed JSON object, so excluded history and its
references remain available alongside the unchanged frozen plan. These excluded
rows do not become missing records in the selected scope. The frozen plan defines
expected point IDs in that scope, or in the whole plan when none is selected. Absent current series rows
remain explicit missing records and prevent `complete=true`, even after a
terminal process status.

Captured native HDF5 metadata is compared with its original owning commit before
derivation. Analysis and closed history segments store `/metadata/identity_json`
(`QCLNEGFRunner` `point_artifacts.jl:314`, `scientific_history.jl:138`); full physics
and recovery store `/metadata/point_identity_json` (`hdf5.jl:288`). Cumulative
history's `/metadata/source_segments_json` intentionally contains earlier attempts:
point/execution/plan identity must agree and source attempts lie between one and
the owning commit cutoff. A borrowed checkpoint is checked against its original
commit, rather than the newer series attempt. Missing metadata or counterpart
fields remain explicit `not_available`; an available identity mismatch fails.

Kelvin scalars `/metadata/temperature_K` and `/inputs/T_L_K` are compared exactly
with the frozen point. Both voltage aliases `/metadata/voltage_per_period_V`
(`point_artifacts.jl:335`) and `/inputs/V_period_V` (`hdf5.jl:345`) are reconstructed
by the producer from `F_bias * period_length`. An exact Float64 match is recorded
as `matched`; a different value retains observed/expected values and producer
path with `not_verified`, since a floating roundtrip comparison policy has not
been established. No scientific tolerance or approximate-match pass is added.

Telemetry identifies CPU intervals by `allocation_id` (for example, a Slurm job/step identity), with separate execution and process identities. Missing counters remain unknown. `TelemetryWriter` is a library for producers; it does not run a background machine sampler.

Each new science, full-state or diagnostic export is one `.tar.xz` archive with no fixed archive size ceiling. The v3 scientific receipt identifies its display filename, storage archive name, SHA256, compressed bytes and pinned snapshot identity. The transport index records whole native objects; it never splits or drops them to fit a transfer limit. Compression and independent verification stream through bounded IO buffers before atomic finalization. Storage exhaustion or cancellation before finalization removes temporary output and publishes no successful receipt. Disk usage includes the captured snapshot, any verified derivations and the compressed archive.

```console
qcl-negf-receive verify EXPORT.tar.xz --receipt RECEIPT.json
qcl-negf-receive reassemble --destination recovered EXPORT.tar.xz
```

The same receiver continues to verify and restore legacy multipart exports, including unordered parts and checksummed chunks. Supply all old parts together. Missing, duplicate, altered, mixed-snapshot or unsupported input fails explicitly. New exports never generate multipart sets.

Receiver results state their `verification_scope`. New archives verify transport
hashes and self-contained HDF5 storage before restoration; external raw storage,
virtual datasets and non-owned links are rejected even for old producer archives.
Checking compressed HDF5 metadata may repeat decompression, but does not create
another numerical payload tree. Legacy multipart verification covers transport
hashes only. Neither scope establishes scientific acceptance or resumability.

Only `qcl-negf.results.v1` / native HDF5 `4.0` is accepted. Missing declarations are errors. Snapshot consistency, process completion and scientific acceptance are separate fields: none is inferred from the others.

## Modules

- `export`, `archive`, `multipart`, `diagnostic_archive`: verified snapshots and transport.
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

## Stored-final equilibrium measurements

```python
from qcl_negf_results.equilibrium import EquilibriumReadLimits, diagnose_stored_equilibrium

report = diagnose_stored_equilibrium(
    "/data/archive/execution/point/final/commit.json",
    expected_identity={"execution_id": "execution", "point_id": "point", "attempt": 1},
    limits=EquilibriumReadLimits(maximum_io_bytes=512 * 1024 * 1024),
)
```

This independent reducer verifies the exact final `physics.full` owner and local
receipt, then streams two passes in stored `E,k,a,b` order. It uses hash-bound
explicit model context, the declared quadratures/scales and full matrices. One
common chemical potential is fitted in the finite represented spectrum using a
bounded bisection; its algorithm width is not a scientific acceptance tolerance.
Raw correlations reconstruct the saved self-energies (enabled dictionary plus
`embedding_total` exactly once); they are not a fresh candidate SCBA map.

The JSON report separates applicability and measurement availability. It includes
raw/normalized occupied and empty numbers, matrix FDR norms, independent occupied
and empty normalization corrections, source hashes and read budgets. Donor-target
residuals apply only to occupied numbers. Empty numbers remain measured; their
donor residuals are `null` with `not_applicable` and the reason
`donor_target_applies_to_occupied_number`. Negative charges are retained.
Nonnegative quadrature weights are required; negative spectral number weights prevent a chemical-potential fit. Unavailable values are
`null` with reasons. Zero norm ratios use explicit zero/undefined branches without
an additive floor. No convergence, discretization or experimental acceptance is
inferred, and outside-window tails remain undetermined.

Boundary currents use **outward electron flow**, `A/m^2`: positive `plus` leaves
toward +z; positive `minus` leaves toward −z. Common-axis values are reported
separately (`Jz,minus = -Jout,minus`); outward balance is their outward sum. These
are not conventional signed charge currents.

`maximum_io_bytes` is mandatory and includes exact commit/receipt bytes, the full
owner SHA and actual EOF-aware HDF5 file-object read/readinto callbacks, including
native validation and repeated coordinates/weights. Physical counters measure
stream-delivered bytes; OS cache does not make them free. Logical selected bytes
and two-pass forecasts are separate. Default numeric workspace is 32 MiB;
energy/momentum/basis/channel, single-read, uncompressed-chunk and report caps are
also explicit. A complete basis plane must fit. Workspace does not bound process
RSS, Python overhead or native HDF caches. No solver/runtime is loaded.

`StateReader.describe` returns immutable dataset/group metadata;
`read_scalar` supports producer rank-0 numeric scalars, and `read_vector` supports
bounded fixed-width grid/weight vectors without requiring their absent coordinate
JSON. Matrix `read` retains its explicit coordinate/weight contract. Optional
`StateReader(maximum_io_bytes=...)` enables the same physical I/O guard for other
callers, while `source` is an immutable certificate built from constructor proofs.
