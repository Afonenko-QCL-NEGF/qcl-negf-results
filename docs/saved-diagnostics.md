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

Stored equilibrium and full saved physical views still require the published actual-grid/scales/
basis/model context carrier. This mandatory followup preserves independent arithmetic and
applicability oracles; the four final-diagnostic integrals do not replace it. Export/process
completion and byte consistency do not assess physical convergence or validate a model.

See [reader, cursor, reduction and export examples](../README.md).
