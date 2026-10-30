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
observe(spec_path, pid, start_ticks, control_fd, *, transport=None,
    telemetry_selection_factory=None) -> int
```

`TelemetrySelection` detaches bounded caller INPUT assertions and complete five-field FileRefs. It
has no verified/ownership/causality flag. A selected `StateReader` authenticates the original
manifest and its actual identity, invocation and producer provenance; performance reading rereads
exact referenced bytes and applies the public source-binding, summary, commit and binding
validators. Canonical plan bytes come from the explicit separate root. Neither display fingerprint
nor an arbitrary dictionary supplies provenance authority. Actual immutable AiiDA INPUT/CREATE
association and the paired Runner/launcher operational admission policy are independent mandatory
production gates.

The factory receives the selected reader and exact closed commit FileRef. Its selection must match
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
