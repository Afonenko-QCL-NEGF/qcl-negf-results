# Export storage and selected-source integrity

```python
preview(reader: StateReader, *, profile: str,
        observations_ref: Document | None = None,
        diagnostics_ref: Document | None = None, telemetry=None) -> dict
export_snapshot(reader: StateReader, destination: str | Path, *, profile: str,
    byte_budget: int, reserve_bytes: int, progress=None,
    observations_ref: Document | None = None,
    diagnostics_ref: Document | None = None, telemetry=None,
    witness_policy=None) -> dict
verify_native_provenance(reader: StateReader, *,
                         observations_ref: Document | None = None) -> dict
```

Every operation requires one explicitly selected result-v2 reader. The named selectors are complete
five-field FileRefs (`path`, `bytes`, `sha256`, `media_type`, `schema`). Both representations share
analysis-v5 schema/role: inventory, filename and schema cannot select D1 or O1. Dedicated readers
verify the expected representation, selected identity/epoch, checkpoint, native metadata and array
metadata. With both selectors, D1 must equal the exact diagnostic FileRef embedded in O1. Describe
and preview do not read numerical scientific arrays or assess scientific acceptance.

Generic `StateReader.source` remains HDF-free before and after dedicated reads. Its `saved_views`
and `stored_observables` capabilities are false with `explicit_final_observations_ref_required`;
`stored_final_diagnostics` is false with `explicit_final_diagnostics_ref_required`. These describe
an unbound operation context. They do not report a missing implementation, absent file or scientific
failure. Preview with verified explicit selectors reports `selected_analysis_metadata_verified` and
context-bound capabilities. Per-view nullable outputs and issues remain the dedicated analysis
reader's responsibility. Without selectors, preview reports `declared_metadata_availability`.

Science defaults to published history and its required closure. Explicit O1 or D1 can select science
without history. D1 requires its checkpoint and actual checkpoint payload frontier; O1 also requires
the exact D1, original plan/model bytes and immediate-parent proof. Parent history remains bound to
its canonical cutoff. Optional unselected full-state owners are omitted unless a selected artifact
actually depends on them. Parent checkpoint proof JSON authenticates ancestry without selecting its
unneeded tensors. Original manifests remain byte-identical provenance declarations with explicit
omissions; science always has `complete=False` and `canonical_source_complete=False`.

Full-state requires a checkpoint and preserves the whole selected canonical graph, including
ancestors, immutable history and exact dependencies. Preserving all analysis files does not qualify
an unselected D1/O1 context. Even with selected metadata, exact restore remains Core-owned and
scientific acceptance is not assessed. Each original root-relative FileRef maps to a verified
content-addressed object. Full-state receive reconstructs those paths under a fresh owned `source/`
directory and validates them with `StateReader`. Shared objects use the same immutable bytes;
science receive exposes objects without claiming a complete canonical source root. Archive readback
proves transport integrity; selected-analysis coverage records the exporter's bounded metadata
proof.

`verify_native_provenance` reports the exact manifest source, invocation, lineage and Git graph.
Without O1 its coordinate context is explicitly ref-required. With O1 it preserves tagged requested
and actual coordinate samples, their recorded verification status/reason, original evidence, current
classification (including nullable scientific acceptance), restoration and original input FileRefs.
It uses the raw reader's plan/model joins. In an archive, the sorted `native_verified_references`
list contains exactly the copied native FileRefs with
`native_verified_references_scope="captured_native_bytes_and_headers"`. This byte/header proof does
not qualify unselected analysis. Reader cache warming cannot alter the archive's provenance or
bytes. It never reconstructs voltage from field times period length, infers coordinates from hashes,
or equates transport identifiers with scientific identity. Snapshot identity also binds which
explicit analysis contexts were qualified.

The private spool accounts simultaneously retained intermediate file bytes and observes free-space
headroom. `byte_budget` is this operation's storage admission limit. The exact selected closure is
estimated before copying. Filesystem allocation/metadata and concurrent writers require additional
administrative reserves; observations are not an OS reservation. One aggregate reader I/O allowance
covers metadata, parent proofs, selected owner hashes and actual source copies. Full-source
validation borrows the same selected reader. All temporary owners/spools close on errors or
cancellation; source FD/path stamps and root stability are checked before archive/receipt
publication. A failure publishes no successful receipt for an incomplete snapshot.

One tar/XZ archive preserves original native5 bytes; multipart fallback and scientific array
re-encoding are unsupported. Hashing, compression and readback stream buffers. Full-file SHA bytes
can exceed a small selected slice; delivered physical bytes and logical array bytes are separate
counters. These limits do not claim a process RSS or native-library scratch bound.

The separate `verify_optical_provenance(path, owner)` retains only the existing declared
`qcl-negf-optical-v4` embedded source-receipt/quality checks. It rejects other schemas, uses a
bounded regular same-open file and returns `embedded_optical_v4_receipt_metadata` scope. It cannot
establish native5 ownership or qualify an optics codec/scientific result. The generic
Path/owner/plan native API remains retired. The new native5 optics codec, typed P07 provider
association, B2 equilibrium bridge and downstream Portal routing require their owning
qualifications.

## Explicit technical selection and logical projection

The optional `telemetry` input is the detached `TelemetrySelection` described in
[live-performance](live-performance.md). Preview and export use the same selected StateReader,
delivered-I/O ledger and actual byte/source guards. Exact current delta validation includes required
retained phase completions and resume contexts. Closed reading requires the exact retained binding.
Missing context cannot yield a header-only available result. Caller assertions do not establish
AiiDA INPUT/CREATE ownership or the production paired admission policy.

`technical_locator` retains each original FileRef, explicit telemetry versus canonical-plan root
scope, and its content-addressed object. The cutoff is the exact caller commit, never a newer
pointer. Uncommitted rows are not included; pending row count is unknown. Observation time is
optional explicit input, not a fabricated freshness clock. A closed collector does not prove job
completion. Original tables require authenticated table/schema declarations and actual matching
Parquet footer metadata. Science selects approved windows/control and cutoff tails, with explicit
omissions. Full-state retains the entire declared technical predecessor/baseline/catalog graph. This
is byte/header and selected current-delta proof, not a new historical actor or scientific authority.

When at least two selected Parquet segments share a table, an additional logical projection
preserves every selected row in declared collector-generation and catalog order, without sorting
values, aggregation or deduplication. Independent rereading compares fixed-4096-row payload hashes
including validity, UTF8, integer/float/NaN payload and signed-zero bits. Source ranges bind exact
retained original segment bytes. Both science and full-state keep these original objects and their
original locators as well as the projection. `telemetry_storage` reports original Parquet bytes and
additional projection bytes separately, with `physical_space_saving_claim=False`. The archive may
grow; logical projection correctness does not establish physical compaction or performance
acceptance.

Parquet reads share the source operation budget. Footer and row-group geometry, fixed type widths
and actual dictionary expansion sizes are admitted before payload expansion. The default 32 MiB
compaction workspace is a policy buffer allowance: six conservative concurrent slots plus 1 MiB
hashing reserve, not a hard RSS or native-library scratch guarantee. Partial owned outputs are
removed on errors and BaseException; unchanged actual sources are checked after reads.

`witness_policy="per-attempt-first-first-positive-ratio-worst-last-v1"` requests a canonical logical
record plan, not HDF rewriting. Scalar history and original witness owners stay byte-identical. Each
source attempt has at most four final selections; latest witness updates determine first finite
positive and worst ratio, stable ties keep the earlier record. The working latest-witness map has
bounded entry and deep-retained-metadata accounting across the operation. Full source identity,
original cursor and witness StateRef remain in the plan. No plot, minimum ratio or selected witness
proves scientific acceptance.
