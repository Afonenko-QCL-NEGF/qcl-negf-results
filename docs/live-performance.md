# Live scalar prefixes and execution performance

This is source plumbing for new ordinary jobs. It neither starts scientific work
nor accepts scientific results. Saved/registered L1 APIs are unchanged. Native
mapping, package injection and the later consumers remain separate owning changes.

## Launcher and private operational ports

The immutable Runner launcher may invoke
`qcl-negf-results capture-run --spec <job>/capture-spec.json --solver <immutable internal Julia launcher> -- run-plan ...`.
There is one Julia child, one reaper and at most one disposable observer. Stdout
and stderr are inherited; no shell. If the observer cannot start after Julia has
spawned, Julia continues once. A missing capture executable must be handled by
the owning launcher **before** spawn; never retry a solver after uncertain spawn.
TERM/INT reach Julia's process group; after five seconds without exit the
supervisor sends KILL. After Julia reap the observer has at most five seconds
plus a 0.1 second nonblocking reap opportunity, then KILL. Uninterruptible remote
I/O can leave observer cleanup unknown: the enclosing Slurm cgroup remains the
cleanup authority. No claim of NFS quiescence or successful cleanup is made.
The supervisor never waits for NFS, Arrow, writer flush, copy or file locks.
It also never writes observer warnings to inherited stderr after solver spawn:
that descriptor can be an NFS file. Missing capture remains unavailable.

AiiDA writes a private immutable spec before launch, at most16KiB:

```json
{"schema":"qcl-negf.capture-spec.v1","selection":{"calcjob_uuid":"UUID","execution_id":"execution","attempt":1,"plan_fingerprint":"64hex"},"plan":{"path":"plan.json","bytes":123,"sha256":"64hex"},"capture_token":"32hex","allocation_id":"actual allocation identity","allocated_logical_cpus":2,"scratch_root":null,"credits":{"performance_bytes":33554432,"scalar_bytes":0,"io_bytes":134217728,"reserve_bytes":67108864}}
```

Numbers above illustrate the port, not a new grant. All credits are partitions
of the **existing admitted job output/storage budget**; protected free-space
reserve is supplied by its owner. Performance partition is8..32MiB, optional
scratch scalar partition0..64MiB. `io_bytes` is an existing finite admitted
observer I/O allowance. Unknown actual allocation width is `null`, never host
CPU count or requested width. `plan` pins exact bytes; selected frozen execution
must contain1..256 unique points. Paths and launcher are trusted package/AiiDA
inputs, never browser fields. Fixed job paths are `capture-source.json`,
`result/series_result.json`, `live/current.json`, `performance/` and
`performance-binding.json`. No extra lock or service is introduced.

Runner optionally publishes this private locator atomically, at most16KiB:

```json
{"schema":"qcl-negf.capture-source.v1","capture_token":"same32hex","execution_id":"execution","plan_fingerprint":"same64hex","mode":"direct","actual_root":"/absolute/job/result","epoch":0}
```

Direct/staged root must equal the selected job's `result`; scratch root must be
strictly beneath the trusted worker scratch parent. Epoch cannot decrease.
Production recovery=true uses direct output, so scratch is not assumed. Only
an actual scratch locator enables scalar copies. Verified stage changes locator
to `staged` before cleanup; failure may leave scratch. Journal source identity,
clock, sequence and exact prefix SHA must match across root changes. Native
point attempts can differ from the enclosing input attempt; no native UUID or
attempt is replaced with the current provider selection.

## Observer and publication

A valid input spec alone grants no output credit. Before constructing Writer or
writing any observer output, the disposable observer waits for Runner's existing
token/fullFP locator. Runner publishes it only after admitting the fixed104MiB
operational partition inside the existing grant. Missing locator before terminal
means unavailable with zero observer output; corrupt/foreign locator refuses
before Writer. Startup samples can be missing. Waiting never belongs to reaper.

The observer reuses `TelemetryWriter` defaults256rows,4MiB pending buffer,15s
flush age. Every nominal2s opportunity records `/proc/<root Julia PID>/stat`
utime+stime / clock ticks and status VmRSS/VmHWM KiB×1024, with captured PID
start ticks. Scope is root Julia process including its threads; descendants,
cgroup page cache and host totals are excluded. PID disappearance/replacement
is null plus missing reason, not zero. Last observed VmHWM is not a guaranteed
terminal peak. Actual CPU width comes only from trusted allocation metadata.
Stable measured CPU deltas survive unknown width; capacity/utilization then
remain null with a reason. With no stable measured interval CPU is null, not0.
Covered capacity and sampled utilization are not total scheduler grant, queue
cost, remaining budget or per-point cost. JIT and missing terminal tail remain
unknown. Existing summary's generic CPU scope string is accompanied by the
reader's explicit capture scope; no new attribution formula is introduced.

Only complete native phase-v2 JSONL lines are ingested: at most128rows/64KiB per
batch,64KiB per line,64MiB **processed bytes for the whole job**, including prefix
replay. Fields are selected from native SPAN_FIELDS, with phase-boundary
measurement semantics. `ingest_spans` atomically publishes rows and source byte
cursor; restart cannot use unpublished staged offsets. `flush_due` is called
even without new rows. A partial terminal line remains incomplete. Final drain
uses only the bounded frozen/result point inventory, never a tree walk.

Live publication is an opportunity at most once/30s plus a terminal flush.
Direct output publishes a small descriptor pointing at its existing index.
Scratch copies only declared closed history leaves through one nofollow FD,
hashes copied bytes, fsyncs/links without overwrite, then publishes **exact raw
native index v2**, then the descriptor. Copy buffer≤1MiB, one copy at a time,
whole-job leaves≤64MiB/4096refs, aggregate index metadata≤1MiB. An advancing
index must retain the previous closed prefix. Open temporary HDF, recovery,
analysis/full-state and arbitrary files are not mirrored. Cap/outage/mismatch
disables observation; existing science is unchanged. The last durable prefix
can carry `live_publication_incomplete`; absent prefix is unavailable.

Performance output is checked against its32MiB partition, bounded4096-file
session inventory, an8MiB prospective flush allowance, supplied IO credit and
protected free-space reserve before writer operations. Copy forecasts include
source reads/writes. One observer-local counter charges metadata reads (including
startup spec/plan and final binding), prefix+suffix+lookahead, planned JSON writes
and scalar copies before delivery. The valid grant prepays the bounded16KiB spec
read. Unchanged Arrow serialization/footer/hash I/O uses a finite temporary
reservation and a local forecast from newly published Parquet sizes; an exceeded
forecast leaves capture unknown. Current/global index writes are charged too.
These are conservative
software forecasts, **not atomic quotas, hard physical IO accounting or a
latency/overhead guarantee** against concurrent filesystem writers. Buffers
have finite byte caps; Python metadata overhead/RSS is not claimed to be8MiB.
The admitted job/cgroup and later finite qualification own resource enforcement.

## Exact retained close binding

`TelemetryWriter.close()` is called once; its returned immutable commit path is
authority. No current/newest/global index/catalog selection. After close, fixed
`performance-binding.json` embeds exact raw UTF8 native commit and declared
unique summary, bytes/SHA/native relative paths, execution selection, exact
`runner_result:{path,bytes,sha256}`, up to256 independent native point pins
`{point_id,source_attempt,point_commit_path,point_commit_sha256}`, capture status
and execution-attempt scope. Total encoded sidecar≤1MiB. Only final series and
small point commits are read; scientific commits are never changed. Missing
sidecar means unavailable. Failed/paused points without commits remain absent
pins, not invented measurements. Scratch-only failed stage has no fixed retained
result binding and stays unavailable.

AiiDA must independently verify raw result SHA and point/source relations,
retain the sidecar as SinglefileData and its UUID/SHA, and expose fixed
retrieval only; do not retrieve performance Parquet/catalog trees. Enclosing3
summary describes execution3, not borrowed native point2 cost. Source-point
resource lookup needs originating CalcJob2's retained binding. Native UUID
contradiction is an error, not permission to stamp current UUID.

`read_performance_summary(sidecar_path, expected_binding=..., limits=...)`
expects **exactly** `{selection,sidecar_sha256,point_pins,runner_result}`. It
rechecks raw sidecar/embedded bytes, native close/session/generation/cursors,
summary role/media/size/SHA, independent result and point pins. It never opens
Parquet/catalogs. Missing binding returns `source_binding_missing`. Output is
finite JSON≤2MiB with ExactScalar integer tags; caps only tighten. One session
is read, metadata≤1MiB, IO≤4MiB, objects≤4096.
The object allowance is common to decoded sidecar, native commit and summary;
`maximum_objects` tightens that single allowance.

## Selected active history reader

`describe_live_history(index_path, expected_selection=..., limits=...)` reads
only bounded native index metadata. Exact provider selection is
`{run_uuid,calcjob_uuid,execution_id,attempt,point_id,source_attempt,plan_fingerprint}`;
producer identity stays separate. Descriptor includes raw index SHA,
generation/cutoff/update/observation clocks, uncertain clock skew, closed-prefix
scope and `access_mode:selected_closed_copy`. Publication is not a heartbeat.

`read_live_history(index_path, phase, row_selection, expected_selection=...,
expected_index_sha256=..., expected_generation=..., spool=..., limits=...)`
requires explicit phase scba|outer and `{start,stop,step,columns}`. Index advance
is `source_changed` before payload. Only intersecting closed leaves are copied
into a caller-owned empty local spool. Every source ancestor/leaf is opened
NOFOLLOW via directory FDs; regular/size/SHA/fstat stability use the same open
inode. Unlink after open is safe, in-place rewrite rejected. HDF reads use only
local copies, retained until detached child reap; parent cleanup remains the
existing Portal responsibility, including fail-closed unknown cleanup.

Limits only tighten:16 selected segments each≤64MiB, whole128MiB delivered-read
plus reserved spool-write allowance, spool≤64MiB, metadata≤1MiB, HDF objects4096,
selected allocation≤1MiB,512rows,16columns, JSON≤2MiB. The budget is shared across
source copy, local hash, metadata and actual HDF chunk delivery, not reset per
segment. Conservative forecasts can reject a large leaf before opening it.
Native units, ν/μ, source attempts, ExactScalar, null/nonfinite masks and line
breaks are retained. Source selection is `source_selection` on slice; row
selector is `selection`. Private local adapter commits identify copied leaves
only and are never reported as native scientific commits. No scientific
assessment/formula is evaluated and no SCBA, Poisson, export or materialize runs.
