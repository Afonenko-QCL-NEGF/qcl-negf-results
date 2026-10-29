# QCL-NEGF results

Independent inspection and export of one explicitly selected QCL-NEGF result. The raw reader
supports **`qcl-negf.results.v2`, native HDF5 `5.0` only**. It needs Python 3.14, the compatible
`qcl-negf-contracts[validation]` package, NumPy and h5py; it does not import Julia, start a solver
or require an AiiDA profile. Legacy result-v1/native4 commits, multipart archives and guessed format
fallbacks are unsupported. Package dependencies are declared in `pyproject.toml`.

## Select a source

Every source has one artifact root and a complete canonical Identity: `run_id`, `execution_id`,
`point_id`, decimal-string `attempt`, `branch_id`, `model_sha256`, `plan_sha256`. Select an
immutable canonical StateRef explicitly:

```python
import json
from pathlib import Path
from qcl_negf_results.state import StateReader, ResultReadLimits

identity = json.loads(Path("identity.json").read_text())
state = json.loads(Path("state-ref.json").read_text())
limits = ResultReadLimits(maximum_io_bytes=256 * 1024 * 1024)

with StateReader("/data/result", state_ref=state,
                 expected_identity=identity, limits=limits) as reader:
    description = reader.source
    reader.import_history()  # explicit bounded metadata import; no tensor read
    batch = reader.history_batch(maximum_records=128)
    cursor = batch.cursor    # publish/persist only after the complete call succeeds
```

Alternatively `StateReader.from_pointer(root, expected_identity=identity,
pointer="live")` opens
`executions/<execution_id>/<point_id>/attempt-<attempt>/current.json`. `pointer="recovery"` selects
`recovery/current.json` under that same attempt. These pointers can legitimately have different
generations. A history-only live manifest does not acquire the earlier checkpoint's exact-recovery
capability. Opening recovery is an explicit request; the reader never substitutes another attempt,
uses a directory glob or searches for the newest file.

`poll()` on a pointer reader checks the small pointer. An unchanged poll opens no catalog or HDF5
payload. An advance imports only new catalog metadata back to the verified head. An explicit
StateRef reader stays frozen. After a verified advance, `reader.rebase_cursor(old_cursor)`
explicitly binds the same original record position to the new manifest/cutoff. Source/epoch/attempt
replacement is an error, not an automatic cursor reset. Serialized cursors are not reusable
integrity proof.

## Records and arrays

`history_batch(cursor)` returns complete canonical scientific records, including all SCBA/outer
scalar fields, units, missing reasons, populations, density and witness declarations. Scientific
stream ordinals are separate from shared Core sequence numbers. A cursor's `row_offset` is the
zero-based physical **last returned record index** in a record leaf, not an iteration number or byte
offset. The initial cursor has null leaf and zero position; EOF retains its last record. Updates
retain exact original-append and immediate-parent checkpoint provenance. Repeated iteration numbers
across attempts are never merged implicitly.

Numeric arrays remain lazy declarations. Select an array declaration from the current complete
record batch or `reader.checkpoint()`, then use:

```python
# declaration is a verified kind=array descriptor from this reader's records/state
info = reader.describe_array(declaration)
block = reader.read_array(declaration, (slice(0, 2), slice(None)),
                          maximum_bytes=1024 * 1024)
```

Use one selector per logical axis and the actual descriptor rank. The on-disk/h5py shape equals the
logical shape; do **not** transpose again to compensate Julia's writer implementation. Numeric
precision, complex compound storage and IEEE bits are preserved. Units remain in the typed snapshot;
the reader does not invent native4 axes, coordinate vectors or quadrature weights for a native5
array. `checkpoint()` verifies the typed declaration and its pinned context, returns all 17
exact-state fields with lazy arrays, and never constructs a Julia solver state. Actual restore and
scientific acceptance remain Core-owned.

`ResultReadLimits` bounds metadata, imported references, record/batch buffers, open handles, HDF
object inventory, returned arrays and physical stream I/O. Before reading a chunked numerical
dataset, the reader checks its fixed dtype and logical decoded chunk size against the working buffer
allowance. Record coordinate/JSON chunks are checked against `maximum_record_bytes`. A small
selection cannot bypass a larger chunk decode obligation; contiguous datasets still support bounded
slices. These checks do not bound every native-library scratch allocation or provide a hard process
RSS quota. Callers can explicitly raise a limit after checking resources. A limit failure returns no
advanced cursor and never truncates scientific evidence. Initial catalog import uses memory
proportional to its bounded metadata index; arbitrary source size does not imply constant memory.
HDF decompression/cache and Python object overhead also require process RAM.

Integrity streams a whole selected immutable file for SHA256 on its safe opened handle. A tiny slice
can therefore require a large integrity read. Reuse the reader to reuse trusted open handles;
eviction/reopen requires new verification. Counters distinguish delivered SHA/HDF/metadata bytes
from logical selected bytes. Metadata/catalog verification and selected payload verification are
separate. Symlink escapes, path rebinding, external/virtual HDF storage and changed source identity
fail explicitly.

## Stored final diagnostics and scalar views

```python
from qcl_negf_results.final_diagnostics import FinalDiagnosticsReader
from qcl_negf_results.diagnostic_views import DiagnosticReader, HistoryViewRequest

# analysis_ref is the exact five-field FileRef from this selected manifest.
with FinalDiagnosticsReader(reader, artifact_ref=analysis_ref) as final:
    observations = final.describe()
    integral = final.read_integral("spectral_integral",
                                   (0, slice(None), slice(None)))

reader.import_history()
view = DiagnosticReader(reader)
batch = view.read(HistoryViewRequest("scba", ("r_K", "J"), maximum_records=128))
# batch.records retains complete typed events; rows is a finite scalar projection.
```

Final diagnostics reads `qcl-negf.final-diagnostics.v1` in the explicit analysis-v5
`final_diagnostics` representation. All three pure validation contexts come from the same reader:
checkpoint, manifest and selected analysis FileRef. The API preserves every metric, witness, count,
limit, status and IEEE bit. It does not recompute observations or infer scientific acceptance. The
four local integral arrays have explicit ranks, axes, units and LE storage; their guard is separate
from the reflected Core array graph. A description reads bounded fixed UTF8 metadata and array
storage metadata, including a physical owner SHA pass, but no numerical integral payload or full
checkpoint tensor. Python constructor tests do not establish JuliaHDF5 producer interoperability.
Science-export selection of this newly supported analysis artifact still needs the owning export
policy followup; current science export selects history. Full-state preserves its declared source
graph. Reading final diagnostics alone does not change either profile's selection policy.

`DiagnosticReader.describe()` opens no HDF5 owner. `read()` consumes one bounded canonical event
batch using the original cursor; filtering a stage can return no displayed rows while advancing the
event cursor. IEEE cells retain their original tag plus a finite value or an explicit invalid
reason. Units and missing reasons come from each stored record. Witness updates remain separate
events; vectors and matrices stay lazy. Numeric source attempts, transport UUID aliases and
full-history consolidation are not alternative source selectors. See
[diagnostic APIs](docs/saved-diagnostics.md).

## Raw stored final observations

```python
from qcl_negf_results.final_observations import FinalObservationsReader

# observations_ref is an exact manifest FileRef, not a guessed HDF path.
with FinalObservationsReader(reader, artifact_ref=observations_ref) as final:
    description = final.describe()
    energy = final.read_array("energy_eV", (slice(0, 3),))
    wavefunction = final.read_array("localized_wavefunctions_per_sqrt_nm",
                                    (slice(0, 4), 0), maximum_bytes=1024)
```

The `qcl-negf.final-observations.v1` declaration uses the explicit analysis-v5 `final_observations`
representation. The borrowed StateReader verifies its checkpoint, linked final diagnostics, original
plan/model bytes and, for completed restore, the exact immediate parent and history cutoff. The
model file SHA and canonical configuration digest are checked independently. Original JSON or
bounded portable nested YAML remains the input authority; caller dictionaries and hash decoding
cannot supply missing inputs.

`describe()` returns detached source, declaration, plan/model, original evidence, integrity and
counters. It authenticates complete owner bytes and array storage metadata but reads no numeric
array values or full checkpoint tensors. The 26 local arrays have closed dtype/shape/axes/units and
coordinate/weight descriptors; they do not widen the Core graph. `describe_array(name)` and
`read_array(name, selection, maximum_bytes=...)` preserve logical h5py axes without transpose and
raw IEEE/complex/integer/Boolean values. Optional null output carries its stored issue reason;
unknown names are errors. All operations share the reader's metadata, physical I/O, chunk and
selected-buffer limits. Borrowed view close leaves the StateReader usable; StateReader close
attempts all owned handles even under cancellation.

This raw API does not implement semantic SavedAnalysis, renderer, equilibrium adapter, transport
association or analysis-export selection. Those mandatory migrations remain separate staged work.
Python literal fixtures do not establish actual Julia 26+4-array interoperability. The production
Python YAML parser's 21 literal portable trees, scalar types and Float64 bits have a separate
comparison against exact Julia1.13/YAML0.4.17 output; this does not qualify every original plan. See
[saved views](docs/saved-views.md).

## Explicit reduction and export

```python
from qcl_negf_results.live_views import ScalarReduction, derive_history

request = ScalarReduction("scba", "J", "1", "100")
reduction = derive_history(reader, request, maximum_io_bytes=64 * 1024 * 1024)
```

The reduction is an explicit streaming min/max of a stored scalar, with units, source locators,
finite/nonfinite/missing counts and separate witness-update counts. It does not run a physical map.
Negative current, signed zero and missing values retain their meaning. Comparison requires explicit
comparability evidence; unknown comparability produces no automatic accuracy or acceptance verdict.

```python
from qcl_negf_results.export import preview, export_snapshot

selection = preview(reader, profile="science")  # metadata and declared sizes
receipt = export_snapshot(reader, "/data/exports", profile="full-state",
                          byte_budget=64 * 1024**3, reserve_bytes=64 * 1024**2)
```

`science` selects actual scientific history/diagnostic owners and their required dependencies. It
preserves explicit omissions of optional full state and original source declarations as provenance.
It is not a complete restorable source. `full-state` requires a matching checkpoint and preserves
the complete selected source graph, including frozen history/parent targets. No full tensor is
loaded into RAM and no native5 tensor or record is re-encoded for export. Both use one checksummed
tar/XZ archive and existing export-v3 selection/capability metadata.

```python
from qcl_negf_results.archive import receive

restored = receive([Path("snapshot.tar.xz")], Path("fresh-destination"),
                   receipt=receipt)
# For full-state only: original root-relative paths and original StateRef
with StateReader(restored["source_root"], state_ref=restored["source_state"],
                 expected_identity=identity) as reader:
    checkpoint = reader.checkpoint()
```

Science subsets expose selected transport capabilities and no canonical full source root. Full-state
receive verifies the reconstructed source using the same reader. Export/receive success proves the
stated byte/container/source-closure checks, not convergence, physics, discretization accuracy or
experimental validation. See [export storage](docs/export-budget.md).

CLI selection is equally explicit:

```console
qcl-negf-results preview /data/result --identity identity.json --state-ref state-ref.json --profile science
qcl-negf-results export /data/result /data/exports --identity identity.json --pointer recovery --profile full-state
```

## Staged research capabilities

This P06 stage provides native5 records, slices, scalar diagnostic projections, stored final
diagnostics, raw stored final observations, reductions and selected exports. Full semantic saved
physical views and stored equilibrium views still require their consumer adapters to the published
actual-grid/scales/basis/model carrier. Their older entrypoints return `capability_unavailable`; the
source description reports exact reasons. This is a staged migration, **not permanent removal of
research functionality**. Required owning Runner/Results context, view and export followup must pass
before final Portal/research release acceptance. Scientific statuses are copied as separate
process/convergence/physics/ discretization/experimental evidence; `recorded` does not mean pass and
`not_measured` never becomes zero or success. Existing technical capture lifecycle and telemetry
migration are owned separately by P29/P30.

## Verification

Use a prepared environment, without bootstrap inside checks:

```console
python -m pytest -q tests/test_result_v2_reader.py tests/test_result_v2_cursor.py tests/test_result_v2_export.py tests/test_final_diagnostics_reader.py tests/test_final_observations_reader.py tests/test_diagnostic_views.py
```

Portable tests manufacture tiny transport sources. To additionally check actual constructor-only
Julia producer output, explicitly set `QCL_RESULT_V2_FIXTURE_ROOT` to a separately supplied,
source-qualified fixture root. It must not contain historical scientific runs. These Python tests do
not invoke Julia or a solver. Constructor transport tests do not establish numerical restart
equivalence or any scientific acceptance. The source graph is selected by the superproject gitlinks;
no install/dependency change is hidden in this check.
