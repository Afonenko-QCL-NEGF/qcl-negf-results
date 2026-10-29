# Native5 saved semantic views

`AnalysisReader` and its direct alias `SavedAnalysis` borrow one explicitly selected `StateReader`
and one exact final-observations FileRef. There is no commit-path or v4 compatibility reader.

```python
from qcl_negf_results.saved_views import AnalysisReader, SavedViewLimits
from qcl_negf_results.render import materialize

# reader is already selected by exact StateRef and canonical native identity.
with AnalysisReader(reader, artifact_ref=observations_ref, limits=SavedViewLimits()) as analysis:
    metadata = analysis.describe()
    brief = analysis.brief()
    heat = analysis.read("spatial_energy_density", [slice(0, 2), slice(0, 3)])
    figures = materialize(analysis, destination,
        selections={"density": [slice(0, 3)],
                    "spatial_energy_density": [slice(0, 2), slice(0, 3)]})
```

`describe_saved_views(reader, *, artifact_ref, limits=None)` and
`read_saved_view(reader, view_id, selection, *, artifact_ref, limits=None)` are one-operation
wrappers. Closing the facade invalidates it; the caller retains ownership of the StateReader.
All operations share its nofollow FD, authenticated owner, source-change and aggregate I/O controls.

Metadata includes exact source StateRef/identity/invocation, O1/CP/D1 FileRefs, conventions,
availability/issues, original/current/restoration evidence and nullable acceptance. Describe and
brief authenticate metadata and owners without reading numerical arrays. Brief retains tagged
current and sheet-density scalars plus exact original request; no coordinator IDs are invented.

The 20 semantic view IDs expose stored potentials, density, E,z density, spectral/occupied traces,
localized/effective levels, real/imaginary wavefunctions and density matrices, and populations.
Every range names every semantic axis with an index, positive slice or exact start/stop/step mapping.
No implicit range clamping is performed. Complex components select the complete stored complex
block. Populations take the real diagonal of the explicitly selected square block, and fractions
use stored `N_dop_2D_per_m2*f_ion`, never the selected population sum. An invalid/nonpositive target
has an explicit unavailable fraction reason. Missing mandatory units/gauge corrupt the selected
contract; optional null output retains its stored issue.

Responses preserve axes, quadrature descriptors, stored scales, canonical units, basis/gauge,
normalized selection and exact raw owner/dataset/selection locator. Negative values and signed zero
remain unchanged. NaN and infinities become JSON null with distinct invalid masks; the raw reader
retains the original IEEE bytes. E,z density uses `eV^-1 m^-3`. No view derives a scientific observable
from Core tensors or changes acceptance.

Selection admission counts full complex blocks, vectors, finite-output buffers, diagonals and
fractions before payload reads. Metadata, opened-owner bytes, delivered I/O, cells, states, line
points, selected buffers and JSON response have separate caps. JSON byte cap is not a memory/RSS
bound. Larger valid sources require explicit engineering limits rather than truncation.

`materialize(analysis, destination, *, selections, history=())` accepts only explicit semantic
selections and optional explicit bounded `DiagnosticViewBatch` values from the same selected source.
It never resolves latest/current or consolidates history. History retains original native event and
iteration coordinates, cycle/attempt boundaries, invalid/log gaps and the latest cycle within the
explicitly supplied batches. It does not claim all-history completeness. Scientific acceptance
remains nullable and separate from presentation readiness.

Heatmaps retain one PNG pixel per selected native cell and all finite native extrema, signed colors
and invalid cells. Nonuniform physical axes use nearest native cells for a bounded display raster;
the annotation states display/native shapes and physical extents. Display reduction changes only
screen sampling. Full native PNG and source arrays remain intact. The private engineering workspace
policy is `16*effective_selected_bytes + 8*maximum_cells`; arithmetic preflight accounts for caller
containers, rows, PNG/compression/output joins, source coordinates, display coordinates/indices and
retained native PNG. It does not bound allocator overhead or native zlib scratch/RSS.

Cache keys bind StateRef, identities/epoch, exact refs, selections, conventions, statuses, explicit
history, limits and renderer revision. Every hit first checks the retained source owners. A warm hit
reuses authenticated HDF handles and verifies asset hashes. One immutable index-byte memo on the
selected `StateReader` binds all rendered metadata to its qualified publication; it is detached from
returned dictionaries. The disk index must match those bytes before decoding, so altered acceptance,
figure inventory, units, source or annotations reject. A new facade borrowing the same owner can
reuse that memo. A new owner has no such proof and recreates the explicit selections from source;
the disk index alone never establishes a warm hit. Changing the key replaces the single entry.
The workspace policy includes memo bytes, binding serialization, index decode containers and asset
buffers; an insufficient allowance rejects before publication. This is not a cross-process cache
proof or a process RSS bound.
Warm index and asset reads open a nofollow, nonblocking regular-file descriptor. Admission uses its
actual opened size. Asset hashes stream in chunks of at most 64 KiB and consume at most that size
plus one growth-detection byte; the descriptor and pathname must retain the same identity, size and
timestamps through the read. Index reads require the exact memo length and a bounded read. Growth,
replacement and nonregular files reject without an unbounded file buffer.
Figures stay in a light theme. Failed, unknown, missing and nonfinite point projections break series
lines; only explicit scientific acceptance can connect points.

History projections must preserve the attached raw record's tagged value, recorded units and missing
reason. Conflicting projections reject; matching missing reasons and changes of recorded units break
the plotted line. Signed current stays in the recorded units without automatic conversion.

The existing transport `brief_result` remains a separate staged boundary. Typed v4 provider
association requires published P07 proof and does not run through this facade or renderer. This B1
increment does not complete provider, equilibrium or optional optics migrations. Those mandatory
research-view obligations remain open before installed research acceptance.

## Raw observations and original input proof

Import `FinalObservationsReader` or `describe_final_observations` from
`qcl_negf_results.final_observations`. Both require the already selected `StateReader` and explicit
`artifact_ref` (all five FileRef fields). This raw reader returns stored observables and context; it
does not reconstruct a private Core problem, diagonalize a Hamiltonian, derive density or certify
science. Original terminal evidence, current revalidation and nullable scientific acceptance remain
separate.

The closed local arrays have ranks one/two, exact LE Float64/ComplexF64/Int64 or unsigned Boolean
enum, logical axes and coordinate/weight references from the owning schema. The native guard
requires fixed UTF8 scalar JSON metadata, internal storage and the producer's contiguous or
shuffle/deflate profile. Selected decoded chunk and returned-buffer bounds apply even to scalar or
empty selections; these limits do not claim a bound on total process RSS or native-library scratch.

Plan/model files are opened through the same reader allowance and checked on their actual nofollow
FD and path before publication. Model canonical identity is independent of the file SHA. Nested YAML
is admitted by bounded events before object construction: no aliases, anchors, merge keys, duplicate
keys, tag directives, arbitrary constructors or nonfinite values. Portable standard tags and scalar
resolution follow exact YAML.jl0.4.17 primary source rather than PyYAML's default implicit resolver;
21 literal portable trees/type distinctions/Float64 bits separately match actual
Julia1.13/YAML0.4.17 output. Full original-plan interoperability remains a separate gate. Binary
integer spellings are rejected consistently with the owning YAML constructor; they are never
silently reinterpreted. JSON plans use the strict duplicate-free, finite decoder and exact
scientific-plan/request joins.

For completed restore, the immediate `lineage.parent_state`, parent checkpoint, original evidence
and frozen history pin are verified under the shared allowance. The reader does not search another
attempt or a newer checkpoint. A valid raw read is not exact-restore numerical equivalence,
independent equilibrium acceptance or a complete scientific archive export.
