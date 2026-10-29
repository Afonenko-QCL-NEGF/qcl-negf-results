# Export storage and selected-source integrity

```python
preview(reader: StateReader, *, profile: str,
        observations_ref: Document | None = None,
        diagnostics_ref: Document | None = None) -> dict
export_snapshot(reader: StateReader, destination: str | Path, *, profile: str,
    byte_budget: int, reserve_bytes: int, progress=None,
    observations_ref: Document | None = None,
    diagnostics_ref: Document | None = None) -> dict
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
failure. Preview with verified explicit selectors reports `selected_analysis_metadata_verified`
and context-bound capabilities. Per-view nullable outputs and issues remain the dedicated analysis
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
proves transport integrity; selected-analysis coverage records the exporter's bounded metadata proof.

`verify_native_provenance` reports the exact manifest source, invocation, lineage and Git graph.
Without O1 its coordinate context is explicitly ref-required. With O1 it preserves tagged requested
and actual coordinate samples, their recorded verification status/reason, original evidence,
current classification (including nullable scientific acceptance), restoration and original input
FileRefs. It uses the raw reader's plan/model joins. In an archive, the sorted
`native_verified_references` list contains exactly the copied native FileRefs with
`native_verified_references_scope="captured_native_bytes_and_headers"`. This byte/header proof does
not qualify unselected analysis. Reader cache warming cannot alter the archive's provenance or bytes. It never reconstructs voltage from field times
period length, infers coordinates from hashes, or equates transport identifiers with scientific
identity. Snapshot identity also binds which explicit analysis contexts were qualified.

The private spool accounts simultaneously retained intermediate file bytes and observes free-space
headroom. `byte_budget` is this operation's storage admission limit. The exact selected closure is
estimated before copying. Filesystem allocation/metadata and concurrent writers require additional
administrative reserves; observations are not an OS reservation. One aggregate reader I/O allowance
covers metadata, parent proofs, selected owner hashes and actual source copies. Full-source
validation borrows the same selected reader. All temporary owners/spools close on errors or
cancellation; source FD/path stamps and root stability are checked before archive/receipt publication.
A failure publishes no successful receipt for an incomplete snapshot.

One tar/XZ archive preserves original native5 bytes; multipart fallback, implicit compaction and
array re-encoding are unsupported. Hashing, compression and readback stream buffers. Full-file SHA
bytes can exceed a small selected slice; delivered physical bytes and logical array bytes are
separate counters. These limits do not claim a process RSS or native-library scratch bound.

The separate `verify_optical_provenance(path, owner)` retains only the existing declared
`qcl-negf-optical-v4` embedded source-receipt/quality checks. It rejects other schemas, uses a bounded
regular same-open file and returns `embedded_optical_v4_receipt_metadata` scope. It cannot establish
native5 ownership or qualify an optics codec/scientific result. The generic Path/owner/plan native
API remains retired. The new native5 optics codec, typed P07 provider association, B2 equilibrium
bridge and downstream Portal routing require their owning qualifications.
