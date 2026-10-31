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
writer.published_context() -> tuple[dict, tuple[dict, ...], tuple[dict, ...]]
writer.close_context() -> tuple[dict, tuple[dict, ...], tuple[dict, ...]]
observe(spec_path, pid, start_ticks, control_fd, *, transport=None,
    telemetry_selection_factory=None, native_execution_path=None,
    solver_arguments=(), native_performance_inputs=None) -> int
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

`published_context()` returns the last durably acknowledged transaction's commit FileRef,
additional phase-context quadruples and resume-context pairs under the writer lock. It does not
flush, close, publish or recover a transaction. Buffered rows and staged cursors remain uncommitted;
publication generation, committed cursors and the closed flag stay unchanged. A constructor normally
acknowledges the genesis transaction. Before the first acknowledgement, or while a failed/uncertain
publication awaits explicit `flush()` recovery, the snapshot raises `ValueError` without inventing
an empty commit. The method also works after close and returns that exact closed transaction.

Each standalone snapshot is an admitted observer read operation. V2 records its operation/self-cost
through the existing observer accounting; it does not write those counters or advance the published
transaction. The ordinary per-operation I/O allowance resets for each snapshot, while any supplied
caller aggregate budget still covers all reads. Repeated snapshots therefore have no lifetime call
quota. All v2 observed operations hold that same writer lock across admission, depth checks,
read-allowance reset, the operation and final accounting. Reentrant calls on the owning thread share
the outer operation; a contending thread waits and receives a separate interval and its own caller
bound. Exceptions, including cancellation, unwind depth/accounting and release the lock before
another thread can enter. The shared close path retains its original combined close-and-readback
I/O allowance, without a new standalone reset inside it.

Both methods preserve the same 1 MiB combined captured-raw/decoded/commit/return-copy metadata
admission, the 128 additional phase / 16 resume reference bounds, and exact selected-commit readback.
The captured snapshot and selected-commit object graph are charged together before allocating
detached returns or readback buffers. Returned dictionaries are detached on every call; neither caller mutation nor a
new current pointer changes the selected snapshot. These references select bytes; they certify no
producer origin, RO lease, scientific acceptance or Portal performance availability.

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

The original seven-token run-plan vector remains supported. The native origin vector appends
`--native-execution native-execution.json --native-calcjob-uuid UUID` in that order, with the
canonical UUID exactly equal to the original capture spec selection. Partial, reordered,
duplicated or foreign values are rejected. Neither form proves origin by itself.

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

For legacy technical v1, the raw return shares one combined 1 MiB metadata and 4096-object allowance
with decoded contexts, guards, assertion bytes and detached return copies. Its I/O cap is 4 MiB, response cap 2 MiB and
session cap 16; callers can only tighten them through `PerformanceLimits`. Prospective return
storage is debited before copying mutable refs. Source guards complete before any return escapes;
late replacement, missing context, cancellation or exhausted credit returns no partial snapshot.
Operation counters remain public on the reader; the owning caller accounts aggregate credit across
reader construction and repeated operations. These reader/writer APIs add no lifetime renewal,
retention pruning, native origin qualification or AiiDA repository carrier. Those remain separate
required production work.

For legacy technical v1, performance metadata is bounded by 1 MiB, delivered I/O by 4 MiB,
response JSON by 2 MiB, objects by 4096 and supplied context pages by 16; limits only tighten. The response reports execution-attempt
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


## Explicit retained technical v2 read-first boundary

The published Contracts revision adds separate technical commit/summary/pointer/binding v2,
retained catalog3 and performance index3. This reader accepts an explicit commit2 FileRef through
the unchanged `TelemetrySelection` and `read_performance_context` ports. The reader does not select a newest/index entry or convert a v1 root. The explicit source-only
writer/window API below is separate from production capture activation and capture-spec/native5. Existing v1 read/export and
recursive full-state semantics remain on their original branch.

```python
selection_raw, original_files = read_performance_context(
    selected_reader, telemetry=explicit_selection,
    limits=PerformanceLimits(maximum_objects=16384),
)
```

The default object allowance remains 4096. An explicit v2 read can request up to 16384; v1 rejects an
allowance above 4096 before technical file reads. All other bounds remain 1 MiB combined metadata,
4 MiB delivered I/O and 2 MiB response, with the same operation ledger, source guards and prospective
return-copy admission. Catalog original bytes have an additional 64 KiB ceiling checked before
opening the declared catalog. Bounds count current/previous metadata, retained original raw
contexts and every selected after artifact together; a required closure that cannot fit is an
explicit `budget_exceeded`, without a partial result or a larger allowance after rotation.
Object credit includes every decoded selected cache member, including catalog/index after leaves;
an opaque historical descriptor stops file traversal, not accounting for its consumed metadata.
These are operation/engineering bounds, not a run-duration quota or a process RSS guarantee.

V2 loads exactly the current commit/summary/catalog and immediate previous commit/summary/catalog,
if any. It validates committed phase context quadruples and resume transaction pairs using the
same public Contracts relations. Optional caller context lists must agree with the committed
inventory. Original producer references keep their source-root meaning; all actual raw reads use
the exact retained locators under the explicitly selected technical root. Raw source files can
contain an unaccepted suffix: only validated accepted physical ranges contribute to local and
cumulative phase summaries. Missing metric profiles remain unavailable. No phase clocks from
unrelated producers become a global wall interval.

Every current after artifact is authenticated by its whole reference. Parquet payloads are hashed
in bounded streams, then their actual table/schema and exact footer row count are checked through
the same open file and I/O ledger, without decoding rows for a summary. Typed catalog/index leaves
receive public shape/membership validation; their referenced old selections are not traversed.
Unlike v1, a v2 summary therefore opens its retained catalog and selected payloads. The integrity
field `catalogs_or_parquet_opened` reports that distinction. Source binding still uses independently
supplied selected native state, original source transport, canonical plan bytes and producer graph;
caller assertions do not authenticate actual AiiDA INPUT/CREATE origin.

The unchanged raw-context return shape contains exact original bytes with `technical`,
`canonical_plan` and `scientific` scopes. For v2 it includes both transaction/catalog triples,
required current phase/resume raw proof, source/close binding, canonical plan, selected scientific
manifest and all current after artifacts, including retained Parquet originals. Refs are detached,
hashes/sizes remain whole-file identities, and late source changes or cancellation return no
partial closure. Reader I/O counters cover the original operation and reset as before.

Each successor catalog inventories its immediate previous catalog as a provenance metadata leaf.
That leaf is authenticated as an original file but its historical artifact descriptors are opaque
for physical retention. Older commit/summary/catalog references inside the immediate previous
metadata are historical provenance, rather than promises to export an entire recursive history.
Pending-unlink debt is also not a retained transport member. Missing required current members fail;
a missing older historical catalog is permitted only through this explicit v2 boundary. Actual
pins, ownership, read/export/copy lifetime, native RO, no-reuse, publication barrier and unlink
reconciliation remain independent owning gates.

Offline receive requires v2 technical record and cutoff fields
`technical_contract='qcl-negf.telemetry-retained.v2'`, `scope='retained_technical_window'`, exact
`catalog`, `cumulative_scope='enclosing_transport_since_v2_genesis'` and `prior_capture`. It checks
contained current/previous originals, raw phase/resume closure, complete after membership and whole
scoped locators, then calls the same public v2 transition/source-binding/close-binding validators.
The current close is joined by its exact embedded commit reference, never by newest session.
An open prefix requires an unambiguous contained source binding for the selected state/transport.
Extra unselected technical locators or transport objects fail. Science and full-state carriers
use the same complete retained technical window; their scientific source-graph capabilities and
full-state restoration rules stay separate.

Existing Parquet compaction format and algorithm are unchanged. V2 receive binds projection source
order and row coverage to the authenticated retained catalog and recomputes the existing
fixed 4096-row validity/native-bit codec from actual contained originals and projection. A signed-zero
or NaN-bit change cannot pass because proof strings agree. This additional verification shares the
4 MiB retained I/O ledger and the existing 32 MiB compaction workspace admission; it stages no second
payload tree and reads no numerical scientific tensors. Metadata bounds do not claim an Arrow or
native allocator RSS limit.

A successful v2 receive adds `technical_verification` with the retained-window scope and
`provenance_origin='not_authenticated_offline'`. It establishes contained byte/declaration
consistency only. It does not refresh native access, actual producer association, pin leases,
writer/pruning readiness, runtime overhead or scientific acceptance. The wire/archive checks use
manufactured fixtures; source publication and later installed/native qualification remain separate.

## Opt-in writer and owned-fixture retention

`TelemetryWriter(..., technical_version=2, prior_capture='none')` writes the retained
technical format in a fresh empty root and requires an explicit Transport. `prior_capture`
may be `unimported`; that evidence remains sticky through rotation and resume. The default
is still version1. Old roots are neither converted nor appended by a version2 constructor.
`resume(..., technical_version=2, prior_capture=...)` consumes one exact current transaction,
its immediate predecessor/catalog and committed contexts; it does not traverse historical
baseline graphs. The caller must have quiesced the previous writer. Rotation keeps the
observer operation ID; process resume starts a new ID and records an uncovered restart.

Each v2 publication stages immutable catalog/summary/commit bytes, checks exact same-open
readback and input guards, then publishes the root pointer and bounded index3. A retry after
lost acknowledgement uses the prepared transaction's original bytes and operation interval.
The current publication tail is excluded from its own cost delta and becomes eligible at
the next completed operation. Every completed outer writer operation classifies all16 cost
metrics. Acknowledged logical payload bytes and process CPU are measured; uninstrumented
read/probe/physical I/O/serialization timings remain unavailable with sticky coverage.
Arrow I/O is separately an admitted forecast. A failed operation cannot claim known zero
bytes for an uncertain write. None of these counters is a physical disk quota.

`retain_window(maximum_retained_bytes=..., minimum_free_bytes=...,
maximum_io_bytes=..., protected_refs=(...))` returns the existing Retention object. It
publishes exact before/introduced/removed membership before any unlink, including the mandatory
previous-catalog metadata leaf. Current/previous transaction/catalog refs, accepted raw delta,
required phase/resume proof and explicit caller-held refs are protected. Older catalog leaves
retire through the authenticated window delta. Unsuccessful unlink remains physical debt;
a later barrier retries it, and only acknowledged success removes that debt from the next
transaction. Whole-file footer rows and accepted physical extents determine omission counts;
missing row counts are never inferred as zero. Per-operation metadata/delivered I/O admission
and Arrow forecasts are capped at4MiB. Before dependent writes the writer checks observed free
space against the supplied reserve, retained closure, physical debt and temporary metadata burst.
Insufficient space refuses that dependent operation; buffered probe data remain uncommitted.
Catalog/raw/phase limits and combined1MiB working metadata still apply, even if individual
cardinality maxima would permit more members. New phase keys that do not fit prospective
current/previous/cache object headroom go into the explicit overflow bucket; established keys
are retained. Inclusive phase totals never become a whole-solver wall interval.

Internal v2 export and `make_binding` explicitly select `PerformanceLimits(maximum_objects=16384)`.
Public `read_performance_context` and performance-view callers must supply that allowance when
needed; their default4096 is unchanged and can reject a valid larger retained closure. Tight
limits still fail through the unchanged whole-cache S2 guards. Both export profiles carry
exactly the same complete retained technical closure and explicit retained-v2 cutoff fields.
Scientific profile selection and the existing Parquet compaction algorithm are unchanged.
The fixed root hardware passport is outside catalog3 membership: S1 permits performance.v2
as Parquet, not arbitrary JSON. It adds fixed root overhead rather than one passport per rotation.

This writer/window implementation is qualified here only for source and exclusively owned local
fixtures. `protected_refs` expresses exact constraints; it does not establish the actual absence
of other RO/export/copy users. Native/shared/NFS activation requires the separate TL-O lifetime,
owner/no-reuse, pin and crash/fault gates. The existing capture observer remains v1 by default;
its historical lifetime credits do not become limits of the explicit v2 writer. Source fixture
throughput and bounded windows do not establish native overhead or scientific acceptance.


## Typed Runner anchors and the source-only native factory

`capture` reads only the fixed Runner `performance-anchor-current.json` and its exact
`performance-anchors/<sha256>.json` original. Original UTF8, including whitespace, is retained.
The public Contracts validator checks the full phase identity and scientific manifest/StateRef
relationship. The immutable copy must already exist and have the same whole bytes. Reading an
anchor never flushes scientific history, publishes a checkpoint or constructs a StateRef.

Phase harvesting uses `phase_identity` and the current phase session, including while
`scientific=None`. Resource samples and original phase rows, cursors, drop/incomplete coverage and
self-cost remain on the enclosing technical stream. A later scientific or borrowed anchor cannot
reset that cursor or move current execution costs to its scientific source attempt. An empty
scientific selection remains `scientific_anchor_pending`; technical evidence alone cannot make
that lifecycle available through StateReader.

The owning source factory has explicit input ports in `qcl_negf_results.capture`:

```python
NativePerformanceInputs(*, provider_selection, original_inputs,
    runner_anchor_root, technical_root, sources)
NativePerformanceCapture(workdir, artifact_root, *, writer, transport, inputs, budget)
capture.remember(anchor, whole_immutable_anchor_ref, original_raw_bytes)
capture.publish(*, close_context=None, capture_status=None) -> FileRef
```

`provider_selection` and `original_inputs` have the published Contracts
`PerformanceProviderSelection` and `PerformanceOriginalInputs` shapes. Each `sources` row has
exact keys `selected_state`, `artifact_root`, `scientific_root`, `scientific_directory`,
`canonical_plan_root`, `canonical_plan_directory`, `source_transport`. RootLocators are explicit
original storage/area/path assertions. The two directory fields are separate internal local
mappings; neither is inferred from the enclosing Runner path or returned in the source index.
A source's canonical plan FileRef is its original `source_transport.frozen_plan`. `selected_state`
may be null only as a future produced-state mapping with the exact enclosing transport; borrowed
sources require an explicit original StateRef. Multiple matching mappings refuse rather than
selecting one. Source attempts are never inferred from native counters.

This typed input detaches assertions and validates shape, normalized paths and combined credit.
It does not verify an actual AiiDA graph, qualify a physical mapping or issue an access lease.
Every scientific, technical, plan and Runner root still needs independent owning qualification.
The factory reads each scientific and canonical-plan mapping independently with the existing
StateReader and public performance reader. It writes immutable source bindings and creates a
closed binding with the existing `make_binding`. A caller callback is still forbidden at native
ingress; the owning factory performs the local binding operation itself.

The factory publishes a full `qcl-negf.performance-source-index.v1`, retaining distinct full
root/state/invocation/source-transport/origin selector keys. Identical keys are deduplicated;
conflicting associations refuse. All entries share the one exact acknowledged execution-attempt
technical context. They do not describe independent copies of that cost. The inventory is at most
256 entries and 1 MiB encoded UTF8. The canonical input JSON, each original metadata read and
every encoded source index remain bounded by1 MiB; capture checks encoded length before calling
its encoder. The F4 private operational allowance is4 MiB for its explicitly retained metadata
forest and prospective owned read/decode/encode/copy/publication buffers, with the existing
16384-object bound. It includes input raw/decoded documents, the complete remembered anchor
inventory, enclosing transport, caller close context/status, accumulated entries and actual
selected reader JSON/guards. Distinct copies consume byte credit; aliased retained objects count
once in that byte weight. The existing public object traversal counts supplied occurrences.
No inventory is truncated to fit. A legal input within the raw/source-count bounds can still
receive an explicit conservative memory refusal.

Owned JSON decoding prepays the full temporary Unicode input, including legal external
whitespace, parser/pair-hook and detached-container capacity, and simultaneously active recursive
diagnostic paths before entering the decoder. Encoding prepays container capacities, diagnostic
paths, Unicode and UTF8 output. Already validated immutable canonical inputs retain their tighter
shape-based decoder allowance. Capture copies also reserve their container/memo workspace.
Incoming read/readback buffers, parsed/detached operands and codec diagnostic/output objects
receive prospective object credit too; object refusal precedes the owned read/write/codec effect.
The declared selected scientific manifest must be≤1 MiB before actual StateReader initialization;
the ordinary initialization/hash check remains and receives a1MiB metadata limit. No extra
manifest read is performed for a generic lifetime recipe.

Reader limits keep their public default4096 and explicit16384 behavior. The performance reader
receives `min(1 MiB, remaining private retained allowance)`, never an enlarged public limit.
Getter metadata1MiB/response2MiB bounds are unchanged. The factory and writer must share the same aggregate
observer I/O ledger; reader construction, source checks, bindings, writes and readback debit it.
Each reader operation reserves its allowance on that ledger before reads begin; direct metadata
I/O can spend only the unreserved remainder. Reader construction and nested performance/binding
reads settle the delivered delta even when validation or cleanup fails. Settlement consumes an
already admitted reservation rather than requesting credit after delivery, so it cannot replace a
primary reader exception with a late admission refusal. The binding payload's known encoded size
is separately admitted before its atomic write. The protected StateReader context profile remains
unchanged.

F4 does **not** bound all transitive callee temporaries, global Python heap, native allocations
or process RSS by4MiB. These limits qualify separate boundaries:

| Callee | Retained existing protection | Not qualified under capture4MiB |
| --- | --- | --- |
| StateReader init/consumer | Exact selected read/hash/identity, supplied1MiB metadata limit, shared I/O and source stability | Init/schema/validator temporaries; its inactive optional arena is not a4MiB certificate |
| performance_views | Existing selected retained cache≤1MiB, response≤2MiB, object/session limits and source guards | Internal decoder, `_exact`, validator copies and diagnostic paths |
| Contracts anchor/index/binding | Existing raw/aggregate byte, object and depth controls | Transitive detachments, embedded decoders, Unicode and diagnostic paths |
| T0 published_context | Stored raw1MiB, context/page bounds, exact readback and its own retained check | Decoder temporaries preceding that retained check |
| catalog/Parquet |64KiB stream chunks, footer preflight and explicit Thrift limits | Native Arrow allocations or RSS |

Historical F1/F3 R1/R2 failures remain failures of their stronger whole-lifetime contract;
F4 is a new qualification identity, not a retroactive whole-lifetime pass. The4MiB operational
choice follows the old closed80 minimum observed forest1099163B and canonical normal80 input
encoding stage1524894B, with approximately15GiB available on the owned host when chosen.
These observations do not establish a mathematical minimum, every-input admission or native
overhead. Representative/adversarial native memory and host-cost qualification remains open;
ordinary existing host/process memory protections still apply.

Open publication uses `writer.published_context()` without closing or flushing the collector.
Closed publication requires the exact context returned by the owning close operation. Source
bindings, closed bindings and all immutable Runner originals precede the immutable
`performance/sources/<sha256>.json`; atomic `performance/current-source.json` and its readback come
last. A terminal close writes those same index bytes as fixed `performance-source.json` for
bounded retrieval. Before the first immutable index write, capture prepares its index reference,
pointer encoding and detached return reference and prepays its own final suffix buffers and
readbacks. This prevents an owned memory refusal first arising after pointer replacement.
It does not preclude I/O or changed-source failure after replacement. A terminal run with no scientific anchor can retain an empty-entry technical
index with an explicit pending control event, while observation reports unavailable. An uncertain
technical ACK, changed original or exhausted credit before pointer replacement leaves the preceding
pointer in place. An atomic replacement can already be durable and visible when its readback fails
or lacks credit: that operation has an uncertain outcome, raises without acknowledging a new owner
generation, and does not prove that the previous pointer is still selected. The visible pointer
may reference a fully validated immutable index; earlier immutable pins remain intact. No blind
rollback is performed over a possible replacement. A later reader must validate its exact selected
bytes independently. After a phase-discovery failure the owner attempts an explicit incomplete close of its last good
accepted prefix, retaining a `capture_failed` control event and incomplete close binding. It does
not retry scientific publication or claim recovery of the missing tail. Failed cleanup preserves
the primary failure and cannot invent an acknowledged index. Existing immutable pins remain
readable when a current pointer advances; consumers use
exact retained operands and closed bindings without a new mutable lookup.

`observe(..., native_performance_inputs=...)` is a source/L opt-in for these explicit typed
operands and technical v2. It is exercised with tiny manufactured native5/history-only fixtures.
The ordinary native CLI still lacks a production transport for all original operands:
`capture-spec.v1.selection` has calcjob/execution/provider-attempt/plan fingerprint, but no
`run_uuid`; original submit-script NodePin is also absent. Native input
`storage.execution_binding_uuid` already carries the native-execution UUID claim, and the AiiDA
actor independently compares it with the actual input node UUID. `request_id` is a different
identity and is never used as `run_uuid`. Plan/source-graph NodePins already exist in that input.

The native calculation path currently also explicitly unsets `QCL_NEGF_CAPTURE_SPEC`; the ordinary
capture-spec generator is called only by the non-native path. The existing native `presubmit`
creates and reads back the genuine `native_submit_script` CREATE after generating the script.

A later narrow owning extension of the existing capture-spec/launch input must carry genuine
original performance operands from the AiiDA graph. It belongs to AiiDA `capture_spec.py` and
`calculation.py`, with Runner's existing exact capture-spec admission and Results `_spec` reader
updated together after design review. A possible ordering is to fix the existing script's capture
path in `_prepare_native`, then fill a versioned capture-spec after the `presubmit` CREATE/readback
from original graph values. The spec can contain the script pin without changing the script bytes.
This is a design proposal, not an implemented launch/admission policy. No extra attestation
sideband or DTO flag supplies that
proof. Until that producer wire, actual graph/root qualification and native lifetime checks are
complete, source fixture success and staging do not establish actual native capture or mandatory
R5 availability. Installed backend, NFS/RO lifetime, AiiDA origin and scientific acceptance remain
separate open gates.
