# Native5 scientific diagnostics

`StateReader.import_history()` imports the selected frozen catalog metadata. `history_batch(cursor)`
returns complete canonical SCBA/outer snapshots, including raw residuals, observables, units,
missing reasons, physical markers and witness array declarations. Outer populations and density
remain separate full vectors. A verified update identifies its original append and, across attempts,
the actual pinned immediate parent checkpoint. Iteration numbers alone never identify a row.

Use explicit array descriptors and bounded `read_array` selections. Native5 HDF shape equals logical
shape; no second Python transpose is required. IEEE bits and complex precision remain intact.
Missing physical markers are not zero defects. The independent final-audit evidence subresource
retains its own version and transport checks; this does not re-enable the retired native4 result
container.

## Scalar event projection

`DiagnosticReader(reader, limits=DiagnosticViewLimits())` borrows an already selected StateReader.
Call `reader.import_history()` explicitly, then
`read(HistoryViewRequest("scba", ("r_K", "J"), maximum_records=128), cursor=cursor)`. The count
limits consumed events, including other stages and witness updates. The result contains complete
`records`, projected append `rows`, separate `witness_updates`, the unchanged canonical `cursor`,
`at_cutoff` and physical `io_counters`. An empty displayed batch can still advance.

Each projected row includes its exact row key, decimal stream ordinal, tagged Core sequence, source
identity/epoch, stored status/quality/converged and `break_before`. A cell contains `tagged`,
`value`, `invalid_reason`, `units`, `missing_reason`. Nonfinite values become null only in the
explicit finite projection; the original IEEE tag remains. Invalid reasons are `nan`, `+Inf`,
`-Inf`. Negative values and signed zero are not clipped. Missing units stay null.

The first displayed row of every batch starts a new segment. Domain/basis/cycle changes, iteration
restarts, legal sequence gaps and invalid samples also break a displayed line. A sequence gap does
not imply corrupt history. An update never replaces scalar append evidence. Use the retained full
record and existing bounded `reader.read_array` for vectors or witness payloads. Only declarations
retained by the current batch/checkpoint remain eligible.

Describe returns metadata and published field types without opening HDF5 or inventing units.
Checkpoint/telemetry costs absent from the result-v2 declaration remain explicitly unavailable; they
are not zero measurements. Provider UUIDs remain transport provenance, separate from the canonical
scientific source. Full physical-coordinate and telemetry integration remains mandatory in the
owning context/provider followup.

Limits can tighten the public ceilings: metadata 1 MiB, owner 64 MiB, physical I/O 128 MiB, selected
bytes 1 MiB, JSON 2 MiB, 512 rows, 16 columns, 4096 objects/source entries. Reader limits also
apply. One allowance covers nested catalog/parent/record checks. A warm coordinate proof cannot
bypass a tighter decoded-chunk allowance. The caller-owned reader remains open; view calls restore
its limits even on failure. Poll/rebase/source replacement follow the original StateReader protocol,
with no second directory resolver or prefix replay.

## Stored final observations

`FinalDiagnosticsReader(reader, artifact_ref=exact_file_ref)` borrows the same verified source.
`describe()` returns
`FinalDiagnosticsDescription(source, invocation, artifact, checkpoint,
declaration, integrity, io_counters)`.
`describe_integral(name)` returns ArrayDescription;
`read_integral(name, explicit_axis_selection, maximum_bytes=...)` returns readonly ArrayBlock.
Closing the view does not close its StateReader. Changing the selected StateRef invalidates the
final view; it never silently moves to another checkpoint.

Only `qcl-negf-physics-analysis-v5`/`physics.analysis`/`final_diagnostics` is accepted. Its fixed
UTF8 scalar identity, invocation and declaration metadata are checked before allocation. The pure
Contracts validator receives mandatory `checkpoint`, `manifest`, `artifact_ref` keyword contexts
from that same reader. The selected exact FileRef, final-state join and dependency DAG are checked;
a path-only substitute is invalid. Whole-file SHA and HDF reads share the safely opened immutable
stream and physical budget. Descriptor validation opens no full checkpoint array owner.

The local integral arrays are spectral integral `(k,a,b)` ComplexF64, spectral eigenvalues
`(k,state)` Float64, cumulative trace `(E,k)` Float64 and negative-energy integrals
`(matrix_kind,k)` Float64. Their shapes are already logical h5py shapes. Endian, complex compound
members, lossless filter whitelist and decoded chunks are checked separately from the unchanged Core
graph. Numeric buffer limits do not promise a hard process RSS or native scratch bound.

Stored metrics/witnesses/limits/counts/status and all IEEE bits are preserved. A measured scan is
not a scientific pass. Error/not-measured scans retain their reason and null data; integral reads
report that capability reason instead of fabricating zero arrays. Core payload validation, fresh
physics, discretization and experimental acceptance are separate. Actual new JuliaHDF5 producer
interoperability remains a separate gate from the small Python constructor fixtures.

## Independent stored equilibrium diagnostics

```python
from qcl_negf_results.state import StateReader
from qcl_negf_results.equilibrium import (
    EquilibriumReadLimits,
    diagnose_stored_equilibrium,
)

with StateReader(root, state_ref=state_ref, expected_identity=identity) as reader:
    report = diagnose_stored_equilibrium(
        reader,
        observations_ref=exact_observations_file_ref,
        limits=EquilibriumReadLimits(maximum_io_bytes=128 * 1024 * 1024),
    )
```

The keyword-only `observations_ref` selects the complete O1 FileRef. Paths and
`expected_identity` are no longer arguments to this operation. The borrowed
`FinalObservationsReader` authenticates the O1 context, original inputs, selected
checkpoint and declared parent evidence through that same StateReader. Missing
mandatory context is corrupt input; it does not authorize inferred defaults.

Scales, physical inputs, quadrature and model applicability come from that exact
context. The computation reads the declared checkpoint's complete
ComplexF64 `[E,k,a,b]` matrices through public reader selections, preserving
complex off-diagonal coherences and actual family order. Every stored scattering
family, including electron-electron when present, contributes once; embedding
`total` contributes once, while `plus` and `minus` remain separate current
observations. Requested scattering flags never select the stored kernels.

The existing independent number, FDR, normalization, current and arithmetic
calculations retain two bounded passes and their missing/undefined statuses.
Original stored evidence, current summary and restoration are reported separately
under `source_evidence`; none replaces an independent arithmetic oracle. The
optional `mu_algorithm_abs_tol_eV`, `mu_max_iterations=128` and
`arithmetic_limits` retain their numerical meaning. No acceptance tolerance is
introduced.

`EquilibriumReadLimits` retains defaults of 32 MiB numerical workspace, 1 MiB
single selection and decoded chunk, 64 KiB report and five scattering channels.
The channel ceiling is an engineering limit: explicitly admit six when the
actual stored inventory includes a sixth family; no family is dropped to fit.
A complete matrix plane must fit. Contiguous datasets stream in bounded tiles;
compressed chunk geometry is checked before numeric reads. These limits do not
promise a process RSS or HDF5/native scratch bound.

One shared physical allowance covers metadata, parent evidence, hashes, vectors
and both matrix passes. The operation restores the caller's limits on all exits,
including cancellation, and leaves the caller-owned reader open. Source FD/path
stability is checked after the final numeric/arithmetic phase before report
publication. This includes consumed O1, D1, checkpoint, original plan/model and
ancestor metadata even when their handles have been closed or evicted under
`maximum_open_files=1`. Witnesses originate from the authenticated read FD;
the final path comparison does not invent an earlier FD proof. Cached metadata
retains its original read binding, so the final guard needs no catalog-prefix or
additional whole-payload hash replay. Existing LRU reopen verification still
applies. Source changes are errors, never partial reports.

Binding bookkeeping is deduplicated by immutable path and bounded by the existing
catalog entry/byte allowances. The operation trace has a metadata ceiling and
shares the numerical workspace allowance, including retained metadata witnesses.
Its complete footprint is reserved before numeric buffers; new unforecast trace
entries then refuse the operation. No extra witness FDs remain open. Normal,
error and cancellation exits release the operation trace and restore the caller's
allowances. The witnesses neither resolve new refs nor authorize metadata or
array values; original FileRef, hash, identity and dependency validation remain
required.

Constructor fixtures and preserved independent formulas qualify this I/O bridge;
actual producer/runtime interoperability and scientific acceptance are separate
claims. Provider ownership, full physical-view transport and the new optics
codec retain their owning gates. Export/process completion and byte consistency
do not assess physical convergence or validate a model.

See [reader, cursor, reduction and export examples](../README.md).
