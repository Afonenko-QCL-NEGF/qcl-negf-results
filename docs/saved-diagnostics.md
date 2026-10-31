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

## Whole native metadata context

`qcl_negf_results.final_observations.describe_native_metadata_context` owns a
single selected metadata read from construction through source-stability checks
and close:

```python
manifest, checkpoint, inputs, diagnostics, observations = describe_native_metadata_context(
    artifact_root,
    state_ref=selected_state_ref,
    expected_identity=complete_identity,
    plan_ref=original_plan_file_ref,
    model_ref=original_model_file_ref,
    diagnostics_ref=selected_diagnostics_file_ref,  # None for checkpoint context
    observations_ref=selected_observations_file_ref,  # both final refs or neither
    limits=read_limits,
    maximum_context_bytes=128 * 1024 * 1024,
    maximum_retained_bytes=4 * 1024 * 1024,
)
```

The function returns original manifest/checkpoint dictionaries and the existing
`CheckpointInputsDescription`, `FinalDiagnosticsDescription` and
`FinalObservationsDescription` types. For checkpoint context the final two
members are `None`. Inputs must be builtin JSON metadata; schemas, complete
identity, exact original FileRefs, role/profile, history/parent bindings, original
plan bytes and model canonical identity retain their existing checks. Missing
mandatory context is an error. There is no inferred final evidence or scientific
acceptance and no numerical array selection.

For `completed_restore`, the observations description additionally contains
`parent_manifest` and `parent_checkpoint`: detached dictionaries read from the
exact immediate `lineage.parent_state` and that parent's original checkpoint
FileRef. Both are present together. For produced observations both are `None`;
checkpoint-only calls still return no observations description. The original
parent epoch, attempt, history cutoff and Core evidence are preserved. A parent
from a different attempt or epoch is valid only through that exact lineage;
latest-state selection and rewriting an identity are not substitutes.

These two fields are optional keyword-only constructor parameters with default
`None`. Existing eleven positional arguments and positional pattern matching
remain compatible; `dataclasses.asdict` includes the two additional fields.
Manually constructing a description does not authenticate it. The public return
tuple still has five elements, and persisted result and binding formats are
unchanged. Parent metadata conveys no native creator, current access, Core
restore or scientific acceptance verdict.

One physical stream-delivery budget is created **before** manifest initialization
and shared by nested checkpoint, history, parent, input and native readers,
source stability and handle close. A successful tuple contains identical final
whole-call I/O counters in all descriptions. The counters include hashes and HDF
metadata delivery; they are not physical disk traffic. Existing standalone
reader operations retain their previous per-operation counters and limits.

The standalone `FinalObservationsReader.describe` and
`describe_final_observations` operations admit the original metadata and all
simultaneous description argument copies against the reader's existing metadata
allowance before copying either parent dictionary. The standalone allowance is
the existing retained-object estimate, not the whole-call arena accounting
below. Whole-call parent copies are also included in the arena's live forests,
temporary copy workspace, final retained allowance and final shared counters.
Parent readers and handles are closed before a successful whole-call return;
consumed parent paths remain covered by its final source-stability check.
Cancellation, a changed parent binding or a close failure returns no partial
tuple. These accounting models do not bound process RSS or physical disk traffic.

The optional keyword-only `maximum_context_bytes` defaults to 128 MiB and accepts
an exact integer in `1..134217728`. It enforces an
explicit aggregate metadata model, including reader caches and parent contexts,
source guards, original raw input bytes, returned descriptions, temporary JSON
and YAML trees, deep copies, nested Contracts clone/encode/validation forecasts,
HDF raw-cache allowances and accounting headers. Admission occurs before each
covered materialization; exhausted capacity raises `budget_exceeded` and returns
no partial tuple. A **112 MiB validation reservation** is admitted before the first
factory or reader initialization and remains charged through source stability
and close. The original Contracts factory builds a fresh 36-document registry
per call; those graphs and its retained process cache are included, never
excluded as shared immutable state. The reservation covers the full shared cache
of 15 fixed public Contracts results/telemetry schema names and three
simultaneous direct bundles, wrapper/builder/layout/inspection workspace, and
the shared standard metaschema graph. In addition to the 12 science names, the
shared cache admits `telemetry-envelope.schema.json`,
`producer-phase-source.schema.json` and `producer-phase-segment.schema.json`:
ordinary performance and producer-phase reads must remain interoperable with a
later closed metadata call. This is a finite public-callsite domain, not every
bundled schema or arbitrary caller-supplied cache key. No factory or cache
semantics are changed; the operation never clears or prewarms a cache.

For the qualified corpus, the 18 simultaneously retained schema bundles and
shared 20 standard metaschemas weigh 94,695,342 W. The explicit fixed registry
URI/wrapper surface allowance is 2,676,240 W (URI string W plus 512 per map/set
entry and 16 KiB per validator). Keeping the separate 6 MiB builder/layout/
inspection allowance gives 103,663,038 W, leaving 13,777,474 W inside the
117,440,512 W reservation. The largest canonical inspection/encoding workspace is 2,087,436 W
inside that separate allowance. These are finite logical accounting estimates
for the qualified implementation, not bounds on Python/Rust allocation or RSS.
The public 128 MiB aggregate and 4 MiB retained defaults remain unchanged.
The larger telemetry sideband corpus requires this separately qualified reserve;
the previous 96 MiB reserve would be short by 2,999,742 W for the same full domain.
This operational revision changes no schema constraints, cache semantics or
physical inputs. A caller can explicitly admit a larger detached result within
the public retained ceiling; the default 4 MiB remains a refusal for complete
original-input final contexts that exceed it.

The operation qualifies exact Contracts schemas/layout/source and the library
content profile, currently CPython 3.14.7, jsonschema 4.26.0, referencing 0.37.0,
rpds-py 2026.6.3 and attrs 26.1.0. It checks actual cached schema contents and
registry aliases before initialization and after close. Bounded read-only
inspection uses the pinned CPython unbounded-lru-cache layout; no cache clear,
prewarm or garbage collection occurs. Unknown source/library/cache content or
layout, mutation, reentry or concurrent operation raises `incompatible_contract`.
Two exact rpds compiled-content profiles (prepared Python and existing Nix Python)
are qualified separately; version equality alone never admits a binary. Other
runtime profiles require a new owning source/profile proof.

The three owner caches qualify the unique actual CPython C-held function
referent, its owning source code/globals and its agreement with `__wrapped__`;
the writable `__wrapped__` attribute alone is not execution authority. The
factory's direct `files`, JSON module/`loads` and package bindings are checked
in both Contracts modules before initialization and after close. Exact loaded
stdlib code fingerprints, the finite `files` decorator closure/defaults and
JSON defaults prevent a foreign callback from hiding behind matching aliases.
These selected checks assume trusted CPython, stdlib internals, imports, guard
machinery, OS and local filesystem. They are not a sandbox for arbitrary process
memory or concurrent privileged mutation of those trusted facilities. No generic
VM verifier, cache manager, factory replacement or new Contracts API is added.

The identity guards share a separate prepaid **6 MiB profile-source read credit**
across entry and exit checks. Each bounded read request, including an EOF probe,
is charged before delivery and never refunded. Hashing uses at most 64 KiB per
read; source rereads before compilation have exact bounded lengths and SHA checks.
Private test/probe hooks record requested and delivered bytes. These identity
reads, the factory's original schema/package reads and vocabulary reads are not
artifact stream delivery. Public `io_counters` keep their existing keys and
artifact scope; they are not a total filesystem-I/O or physical disk counter.
The outer caller must account for package-source scope separately. Local package
vocabulary decoding still has its own forecasts; interpreter baseline is not a
process memory guarantee.

The model uses 64 units per scalar occurrence, `128 + 6*length` per string/bytes
occurrence, and 128 plus entries for a container; an array index adds 64.
Container aliases within the retained forest count once, distinct copies count
separately. The accounting header is 64 KiB plus 2 KiB per preflight and 128 units per visited container
plus 65 traversal frames; active parser
workspace and declared HDF raw caches remain charged until released. It is an
operational admission model, **not** a total Python heap, HDF allocator or process
RSS guarantee. `metadata_high_water_bytes` reports its admitted high-water mark.

All returned objects are detached builtins/raw bytes or existing metadata
Descriptions. Internal readers and HDF/file handles close before return; cancellation
and failures close remaining owners and preserve a primary exception if cleanup
also fails. A later call creates a fresh owner allowance; this does not refund or
reset an outer caller's prepaid aggregate ledger.

The optional keyword-only `maximum_retained_bytes` defaults to 4 MiB and accepts
an exact positive integer up to 128 MiB. It bounds W of the **final returned tuple**,
including all final whole-call counter headers, immediately before ownership
transfer. Exceeding it raises `budget_exceeded`, closes owners and returns no
partial tuple. Smaller arena limits can fail before initialization because the
validation reservation alone requires 112 MiB. The original 4 MiB refusal remains
a valid negative result; it is not reclassified as a successful final context.

An outer consumer must prepay one sequential workspace and each retained output
slot separately, keeping earlier outputs charged while the next call runs.
Only complete detached return and close permit workspace reuse. Error/cancellation
is terminal for that parser attempt; no credit is refunded or reset. Returned
context does not supply creator/parent authority or replace original FileRefs.
Constructor fixtures establish no actual producer, creator, native package,
restore, whole-parser or scientific acceptance. Callers retain all required rows
and report an exhausted or unprovable context as a hold.
