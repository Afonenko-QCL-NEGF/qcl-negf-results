# Bounded saved analytical views

`qcl_negf_results.saved_views` exposes read-only metadata and exact native
selections without invoking Julia, solver, export, derive jobs, or full-state
validation. Existing `StateReader` integrity and scientific semantics are unchanged.

- `brief_result(metadata, *, selection, maximum_bytes=1048576)` is pure JSON
  projection. `metadata` contains full retained `plan`, native `series`, optional
  `plan_source`, and provider `source` disclosure. `selection` has exactly
  `run_uuid`, `execution_id`, positive `attempt`, `calcjob_uuid`. The series must
  match the retained plan fingerprint. Frozen point order is authoritative;
  absent planned rows, null/absent currents and failed points break current
  segments. `current_segments` contains point IDs, including isolated endpoints.
  `line_connectable` conservatively requires a completed row with finite saved
  current and explicit saved observables.scientific_assessment.scientific_accepted=true;
  this is display topology, not acceptance inference. Native series v3 rows do not
  promote this flag to top level. recorded_statuses retains the entire raw nested
  assessment, projected accepted flag and explicit assessment_source_path.
  Missing assessment remains unknown; a top-level synthetic flag is not used.
  Raw statuses/assessments remain independent and unknown
  metadata stays null. Multi-execution series is filtered by exact execution.
  The selected CalcJob attempt bounds recorded point attempts: positive exact
  integers from 1 through the selected attempt are allowed, since recovery may
  retain completed earlier rows unchanged. A recorded `data.checkpoint_source_attempt`
  must likewise be a positive exact integer no greater than the row attempt
  (or the selected attempt when legacy row attempt is absent). `source_attempt`
  preserves this explicit checkpoint source, otherwise the recorded row attempt;
  absent legacy provenance remains null. Selection never relabels a historical
  payload, its commit SHA or its saved scientific assessment. These brief-only
  bounds do not relax `AnalysisReader`'s exact native identity requirements.
- `describe_saved_views(commit_path, *, expected_identity, limits=None)` opens
  only the declared single `physics.analysis` owner. Header, embedded identity,
  finite layout and selected dataset metadata are inspected; no numeric dataset
  arrays are read and whole-owner SHA is not yet claimed. Views absent from the
  saved projection are unavailable. The local path is provider-owned, never a
  browser-supplied file or HDF5 dataset path.
- `read_saved_view(commit_path, view_id, selection, *, expected_identity,
  limits=None)` verifies whole-owner SHA, then reads an exact selected native
  hyperslab and matching coordinates/weights. Integer indices are zero-based;
  ranges require `{start,stop,step}` with exclusive stop and positive step.
  No clipping, automatic sampling, rescaling, positivity clamp or synthetic
  default arrays occurs. Public view IDs are the finite `VIEWS` mapping.
- `AnalysisReader` is the corresponding context manager for several operations
  on one opened immutable owner. No HDF handle escapes. It closes HDF before
  closing the budgeted backing stream, including construction/read exceptions.

`expected_identity` requires point_id, execution_id, attempt and plan_fingerprint;
all supplied recorded identity fields are compared with exact JSON types in both
commit and embedded native metadata, including native-only run/CalcJob fields.
State-bound native identity retains whole-identity equality with its commit.
identity_binding discloses native_recorded_fields and genuinely legacy
provider_only_fields; foreign native provenance is never relabelled.
Optional source_commit_sha256 pins the selected commit bytes. Additional
run_uuid/calcjob_uuid come from provider selection; missing legacy embedded fields
are not invented. Provider binding of exact attempt/CalcJob is separate from native
identity verification. No latest-by-time or first-artifact heuristic is used.

Metadata strings must be fixed-width scalars within the common metadata cap;
before reading an attribute payload, the low-level datatype must be atomic fixed
STRING or a supported fixed-width INTEGER/FLOAT and the dataspace must be SCALAR.
VLEN (including numeric), compounds, arrays, references and nonscalar attributes
are rejected before high-level read/NumPy conversion. Fixed strings and scalar
numeric multipliers share the metadata allocation cap; unbounded HDF5 vlen
strings are explicitly unsupported by this narrow reader.
Header requires current native4.0/current contract and analysis schema. Soft,
external and virtual links/storage, external filters, object numeric types and
unbounded inventories are refused. Shape, units and native axes are checked
before numerical reads. Missing required gauge/units make the view unavailable.
Declared JSON metadata dependencies are followed under the common byte cap and
verified by their declared SHA. A dependency requiring another scientific HDF5
owner is explicitly unsupported; this adapter never opens full physics as a
fallback. Retained toolchain_json, when present, is disclosed as producer
provenance; absent provenance is unknown, never replaced by the reader checkout.
The full native validator reads diagnostic arrays and is deliberately not called.
`integrity` separately reports metadata_checked, owner_sha_verified and
native_validation="not_performed"; none establishes scientific acceptance.

Finite JSON `values` use null plus an equally shaped `invalid_mask` whose entries
are null, nan, positive_infinity or negative_infinity. Finite zero/negative values
remain numbers. Every selected original axis is disclosed with actual saved
coordinates/units/source path and optional quadrature weights/measure/multiplier.
An indexed dimension drops from returned_shape/logical_axes but remains disclosed
in axes with its one selected coordinate. Native state-index mappings are kept.
Complex components are distinct explicit real/imag view IDs; they are not dropped.
Effective levels are saved projected k=0 Hamiltonian eigenlevels, not spectral peaks.

Population views explicitly read only the selected square native block, charge
its full allocation, and return its real diagonal with basis and state mapping.
Optional `fractions` use saved target ionized sheet doping: saved
`inputs/N_dop_2D_per_m2` times saved `inputs/f_ion`, explicitly marked as a Results
calculation from those two saved scalar inputs. This matches the target in Core
`state_populations`/`ionized_sheet_density`; it never normalizes to the sum of the
selected subset. If either input is absent, nonfinite or nonpositive, fractions
are unavailable with a reason; raw populations remain available. The current
analysis writer ordinarily omits these inputs. No additional full-state owner
is opened to recover them.

Defaults in `SavedViewLimits`: owner64MiB, delivered I/O128MiB (including metadata,
whole-owner SHA and HDF delivery), metadata1MiB, selected allocations1MiB including
coordinates/weights/target scalars, response2MiB, cells65536, line4096 native points,
wave/population states12, matrix subsets64×64, native inventory4096 objects. Decoded
HDF chunks also fit the selected allocation cap. Limits refuse requests, never
reduce science. Counters distinguish delivered stream bytes, SHA, HDF and logical
selected bytes; they do not claim measured kernel I/O, decompressor RSS or latency.
Provider copy/time/process caps and cancellation/temp cleanup remain provider gates.

Repository provider and future registered read-only NFS provider may call the same
local adapter. This source does not create NFS registration/access/retention or
change ongoing retrieval. Tests are in `tests/test_saved_views.py`; only a prepared,
budgeted focused run is appropriate, with no dependency installation or solver.
