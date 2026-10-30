# Selected native5 capture and execution performance

Results reads immutable producer publications and telemetry transactions. It does not solve, accept
scientific results, infer source ownership from an identifier, or change scientific files after
optional capture fails. Native source attempt strings remain separate from the enclosing telemetry
transport attempt integer.

## Explicit input ports

```python
TelemetrySelection(artifact_root, *, commit_ref, source_binding_ref,
    expected_transport, source_transport, canonical_plan_root,
    canonical_plan_ref, expected_producer_provenance, binding_ref=None,
    observed_unix=None, phase_context_refs=(), resume_context_refs=())
read_performance_summary(reader: StateReader, *,
    telemetry: TelemetrySelection | None, limits=None) -> dict
read_performance_context(reader: StateReader, *,
    telemetry: TelemetrySelection, limits: PerformanceLimits | None = None)
    -> tuple[bytes, tuple[tuple[str, dict, bytes], ...]]
writer.close_context() -> tuple[dict, tuple[dict, ...], tuple[dict, ...]]
observe(spec_path, pid, start_ticks, control_fd, *, transport=None,
    telemetry_selection_factory=None, native_execution_path=None,
    solver_arguments=()) -> int
capture_run(spec_path, solver, arguments, *, native_execution_path=None) -> int
```

`TelemetrySelection` detaches bounded caller INPUT assertions and complete five-field FileRefs. It
has no verified/ownership/causality flag. A selected `StateReader` authenticates the original
manifest and its actual identity, invocation and producer provenance; performance reading rereads
exact referenced bytes and applies the public source-binding, summary, commit and binding
validators. Canonical plan bytes come from the explicit separate root. Neither display fingerprint
nor an arbitrary dictionary supplies provenance authority. Actual immutable AiiDA INPUT/CREATE
association and the paired Runner/launcher operational admission policy are independent mandatory
production gates.

The programmatic factory receives the selected reader and exact closed commit FileRef. Its selection must match
the initial explicit Transport, exact close and technical root. Transport uses the full public
schema before observer output. `None` means unassociated: CLI capture cannot fabricate the transport
or a canonical plan from the display fingerprint. It retains capture-error evidence and returns 2
with `source_binding_missing`. This is not production capture qualification.

Closed selected reading requires the exact retained close binding. A missing selection returns
unavailable; a missing required completion/resume context or a forged reference fails before
successful performance/export publication. Open prefix reading can describe only its explicit
committed prefix. No current, newest, global-index discovery, directory scan for a replacement
reference, or legacy sidecar fallback selects a transaction.

`phase_context_refs` has at most 128 rows, each exactly
`{publication_ref, segment_ref, retained_publication_ref, retained_segment_ref}`. Original and
retained pairs must bind the same bytes, hash, schema and media. `resume_context_refs` has at most
16 exact `{commit_ref, summary_ref}` pairs. These provide only missing prior completions to the full
public current-delta validator. They are not trusted decoded evidence. All actual reads, hashes and
source guards share the existing operation allowance.

`close_context()` closes through the existing writer transaction and returns its exact commit
FileRef, additional phase-context quadruples and resume-context pairs. Accepted phase ranges
already carry their retained references in the commit, so the additional list excludes them. The
writer captures these references before clearing staged contexts and exposes them only after the
durable close and exact commit readback. Repeated calls select the same transaction and detach
mutable return dictionaries; a caller cannot change the retained snapshot. Publication errors and
lost acknowledgements retain the original pending transaction for reconciliation. No current or
newest lookup selects this close. The observer uses this public returned commit for terminal
binding; its existing same-owner rotation helpers retain their previous role.

Terminal binding attaches the additional phase/resume references inside Results to the factory's
public detached selection. The callback remains `(reader, close_ref)`. Equal rows deduplicate;
conflicting references for an original publication/commit path fail before the binding is written.
The detached combined metadata and public selection row limits still apply. This retains the exact
old-completion/resume context for `read_performance_context`; it never reads private writer fields
or discovers another close through a pointer.

## Fixed native ingress

The optional native CLI form is literal argv, without a shell:

```text
qcl-negf-results capture-run --spec capture-spec.json --solver INTERNAL_SOLVER \
  --native-execution native-execution.json -- \
  run-plan plan.json ORIGINAL_ABSOLUTE_OUTPUT --execution-id E --source-graph source-graph.json
```

The immutable Runner wrapper owns its paired native marker branch; this Results increment alone
does not activate an AiiDA script or qualify the installed wrapper. The supervisor starts exactly
one solver before its observer. It forwards the native path and the same original solver arguments
to the disposable child. Only that child reads the original plan/source-graph/native-execution
files. It never imports a module string or deserializes a callback. Ordinary inputs remain optional
and unchanged; mixing native ingress with caller `transport` or a programmatic factory is refused
before observer output.

The fixed parser checks bounded original raw input claims, the public scientific plan and
SourceGraph grammar, exact run-plan arguments, execution/full fingerprint, and the absolute output
suffix declared by `relative_execution_root`. Only direct/staged locator roots equal to that exact
output are accepted. Native `--attempt`, duplicate/additional options, native scratch, rewritten raw
inputs and foreign locators are refused. Transport attempt comes from the original capture spec;
the native point attempt and source/route epochs remain distinct. All original input reads and
native validation share the existing observer I/O credit, 1 MiB aggregate native metadata and
4096-object limits. This adds no lifetime grant or credit reset.

These checks establish input assertions. They do not authenticate the absolute trusted jobs-root
prefix, actual InstalledCode/release/internal-wrapper origin, original source creator or current
storage access. Those remain controller/Platform gates. In particular the existing native input
and manifest `Invocation{source_epoch}` do not supply original selected source transport. Native
ingress therefore retains phase/resource raw and a `source_binding_missing` control event, closes
the technical transaction and returns observer status 2 without a fabricated selected binding.
It does not infer a produced/borrowed association from a point counter, source relation or graph.
An owning AiiDA/Platform contract must supply that missing original operand before native selected
performance can become available; this port does not change their public APIs.

The existing scalar descriptor has a workdir-relative artifact root. External native canonical
roots keep that scalar route unavailable, rather than publish an escaping path or invent a copy
policy. Raw telemetry is retained; scientific source files and the solver exit/signal/reap behavior
are unchanged. This is SOURCE/L ingress support, not native qualification, an offline raw carrier,
retention policy, long-run acceptance or a numerical experiment.

## Selected raw contexts

`read_performance_context()` uses the same loader, public current-delta validator and operation
guards as summary reading. It returns serialized **input assertions** followed by exact consumed
original file bytes. The assertion bytes have the existing `TelemetrySelection.document()` grammar;
they are not measured raw or a producer certificate. Each file tuple is `(scope, whole_FileRef,
original_raw_bytes)`, where scope is `scientific`, `canonical_plan` or `technical`. The closure
contains the selected scientific manifest, original canonical plan, current commit/summary,
immediate predecessor pair, accepted retained phase pairs, supplied old-completion/resume contexts,
source binding and selected close binding. It includes neither scientific arrays nor an all-history
proof of ancestor deltas. Closed selection still requires its exact binding.

Files are deduplicated by scope and complete FileRef in first-consumed order. Conflicting refs for
one scoped path fail. Original phase references retain their original paths; retained raw files
keep their explicit retained refs. No path rewrite, summary reconstruction or directory scan fills
missing raw. A consumer can materialize separate scoped roots and use the existing public reader
again, but must independently verify actual original INPUT/CALL/CREATE provenance. This API does
not certify native access, restore, science acceptance or original creator ownership.

The raw return shares one combined 1 MiB metadata and 4096-object allowance with decoded contexts,
guards, assertion bytes and detached return copies. Its I/O cap is 4 MiB, response cap 2 MiB and
session cap 16; callers can only tighten them through `PerformanceLimits`. Prospective return
storage is debited before copying mutable refs. Source guards complete before any return escapes;
late replacement, missing context, cancellation or exhausted credit returns no partial snapshot.
Operation counters remain public on the reader; the owning caller accounts aggregate credit across
reader construction and repeated operations. These reader/writer APIs add no lifetime renewal,
retention pruning, native origin qualification or AiiDA repository carrier. Those remain separate
required production work.

Performance metadata is bounded by 1 MiB, delivered I/O by 4 MiB, response JSON by 2 MiB, objects by
4096 and supplied context pages by 16; limits only tighten. The response reports execution-attempt
scope and nullable measured coverage. It never attributes execution3 resources to a borrowed source
point from attempt2. Summary reading does not open catalog/Parquet payloads; export validates those
selected original bytes and footers separately. Integer values outside the JSON safe range use
ExactScalar decimal tags. Unknown CPU, capacity, terminal tail, observer cost and scientific
acceptance remain unknown rather than zero/pass.

## Bounded producer and collector pages

```python
PhaseCaptureSource(root, *, descriptor, expected_identity, route_epoch,
                   limits=None)
source.harvest(cursor, *, budget) -> PhaseBatch | None
source.correlation_capabilities() -> dict
writer.stage_phase_batch(batch, *, phase_context_refs=(),
                         resume_context_refs=()) -> None
writer.rotate() -> Path
```

The source consumes sealed P29 phase publications and their exact raw segment, publication and
predecessor bytes. A cursor identifies physical row prefixes; normal aggregates can cover sequence
intervals such as 8/6, so maximum event sequence is not a row count. Quiet polls authenticate
current metadata and reuse the bounded qualified closed payload cache; they do not replay a JSONL
prefix. The raw PhaseBatch bytes and public validators bind every accepted prefix. Whole source
identity, session, clock and route epoch remain explicit.

Normal producer aggregates retain call counts, durations, measured/unavailable counts, coalescing
and health. They do not record per-call counter/wall correlation: its capability is
`correlation_not_recorded`. Diagnostic segments retain original Events; only actual stored,
measured, representable durations and counters can populate per-call span rows. Thus the literal
two-Event example has CPU 8 s, GC .25 s, allocation 1024 B, total duration 5 s and CPU-measured wall
2 s. Normal duration 5 s cannot imply measured wall 2 s or 5 s. Missing admission wait stays null,
and coalesced counts remain independent. This is a diagnostic capability, not parity with an old
summary-level wall field or a science test.

Staging invalid input changes no cursor or rows. Raw retention, Parquet rows, source cursor and
accepted ranges publish together; pointer is last. Retry after publication failure reuses the
candidate rather than duplicating accepted rows. An explicit `rotate` closes the old page, returns
its exact closed path and starts empty active cursors with the same Transport, exact closed baseline
and cumulative totals. It does not bridge old/new CPU endpoints. There is no silent rotation inside
invalid staging and no new allowance after rotation. The capture owner explicitly rotates before the
seventeenth active source and supplies original retained completion/resume refs if an older source
reappears. Retained page-context metadata is bounded by 4 MiB, with no persistent secondary index.
Finite existing credits can still end observation; sixteen is not a lifetime source count cutoff.

## Launcher, storage and cleanup

The immutable launcher may invoke
`qcl-negf-results capture-run --spec
<job>/capture-spec.json --solver <immutable Julia launcher> -- run-plan ...`.
There is one Julia child, one reaper and at most one disposable observer, with inherited
stdout/stderr and no shell. Observer failure never relaunches Julia. TERM/INT reach its process
group; after five seconds the supervisor sends KILL. After Julia reap the observer has five seconds
plus .1 s nonblocking reap, then KILL. Unknown remote cleanup remains the enclosing Slurm cgroup's
concern; the reaper never waits for NFS, Arrow, copy, flush or file locks.

The private immutable capture-spec pins launch selection, exact display-plan bytes, token,
allocation CPU width and admitted performance/scalar/I/O/reserve partitions. Performance is 8..32
MiB and optional scratch scalars 0..64 MiB; these are partitions of the admitted job grant, not
additional campaign quota. Runner's token/full-fingerprint `capture-source.json` appears only after
its operational admission. Spec alone permits no observer output. Direct/staged root is exactly
`<job>/result`; scratch is strictly under its trusted parent. Epoch cannot reverse. Neither this
engineering reader nor synthetic fixtures qualify that owning operational policy.

The observer defaults to 256 pending rows, 4 MiB payload buffer, 15 s flush age and nominal 2 s
sampling. `/proc` root-process CPU includes its threads, not children, host totals or cgroup page
cache. PID start ticks protect identity. Missing samples and allocation width remain null. Last
observed VmHWM is not a guaranteed terminal peak. Stable intervals retain measured CPU with unknown
capacity; no stable interval gives null CPU. Existing independent CPU arithmetic and
allocation-change rules are unchanged.

Direct/staged live publication points at its explicit selected native5 source. Scratch copies only
declared closed scientific history closure with one regular NOFOLLOW FD, byte hash and path/FD
stability. It publishes a local `live-prefix.v2` descriptor with original StateRef/history pin and
explicit omissions. It never presents the subset as complete canonical/restorable state. Copy
buffers are at most 1 MiB, selected leaves at most 64 MiB/4096 refs and metadata at most 1 MiB.
Optional full state, temporary files and undeclared arrays are not mirrored. No HDF scientific
payload is numerically assessed.

One observer I/O ledger covers startup metadata, source reads, hash checks, retained metadata
writes, copies and final bindings. Arrow serialization uses a finite prospective reservation settled
from actual new Parquet sizes. Used storage includes explicitly closed pages as well as the active
page. The finite 4096-entry session inventory and 8 MiB prospective operation allowance are
engineering forecasts, not atomic quotas, physical-I/O guarantees, latency bounds, RSS claims or
native allocator bounds. Protected free space remains an explicit owner input. Source/owned
temporary descriptors close on errors and cancellation; primary failures are preserved.

Exact close bindings live at `performance/performance-binding.json` for one point or
`performance/performance-bindings/<StateRef-sha>.json` for multiple points. They embed original UTF8
technical commit/summary bytes and the exact selected state/source-binding relation. Producer
association remains unbound until actual P07 actor evidence is validated; successful byte/contract
reading cannot promote it. Existing provider/Portal callers need their owning typed-port migration
before production integration acceptance.
