# Intermediate export storage budget

`export_snapshot(..., byte_budget=64 * 1024**3, reserve_bytes=64 * 1024**2)`
limits the simultaneous **logical file bytes** of one export's private spool.
The CLI exposes the same policy as `--byte-budget` and `--reserve-bytes`, in
bytes. Both parameters have finite defaults; the budget must be a positive
integer and the reserve a nonnegative integer.

The budget includes captured source objects, cumulative-history proof copies,
telemetry compaction, witness selection and HDF5 repacking, superseded captures
that still exist, the compressed archive, and the temporary receipt. Published
archives and source results remain separate owners. An export never omits
scientific objects to fit this operational budget: it publishes the complete
selected archive or refuses publication. There is no fixed scientific archive
size ceiling.

Preflight checks the known captured prefix and proof copies before copying any
payload. Each controlled write checks projected file growth and observed free
space, including the publication reserve. Seeked writes, truncation, typed
buffers, Parquet writes and compressed XZ output use the same accounting.
HDF5 uses its file-object driver. After a refusal it discards subsequent writes
to the private temporary file and reports the original error after HDF5 closes;
raising inside the C callbacks would leave unsafe pending exception state.
Independent readback and a complete spool size reconciliation precede any final
archive or receipt replacement. Refusal cleans only that export's own staging;
existing archives, receipts and source results remain untouched.

A verified receipt and matching archive for the same pinned source are returned
before spool preflight because reuse writes no new payload. The current budget
is an operational policy and does not change scientific snapshot identity.
An invalid or missing cached archive requires a fresh export and the current
budget applies to that work.

This library guard is **not a filesystem quota**. File allocation rounding,
directory metadata, sparse-file accounting and other simultaneous exports or
processes are outside its logical-byte counter. Free-space observations do not
reserve space against concurrent writers. Enforce a hard physical disk limit
with the deployment's existing filesystem quota or a dedicated bounded volume.
The helper does not install, configure or replace that OS facility.

The exporter refuses with `budget_exceeded` for logical budget exhaustion and
`storage_failure` for insufficient observed free space. Ordinary write failures
remain errors and do not acknowledge publication. Export success establishes
transport consistency; it does not establish solver convergence, physical
acceptance, discretization sufficiency or experimental validation.
