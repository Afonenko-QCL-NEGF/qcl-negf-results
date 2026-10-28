# Export storage and selected-source integrity

`export_snapshot(reader, destination, profile=..., byte_budget=..., reserve_bytes=...)` requires one
explicitly selected result-v2 reader. The existing private spool accounts simultaneous intermediate
file bytes and observes free-space headroom. `byte_budget` is a storage admission limit for that
operation, not a scientific history cutoff or campaign credit. A failure preserves source files and
publishes no successful receipt for an incomplete snapshot. Filesystem allocation/metadata and
concurrent writers need additional administrative reserves; observations are not an OS space
reservation.

Science selects actual published scientific artifacts and their own required closure. Optional
checkpoint tensors are listed as omitted, unless a selected science artifact actually depends on
them. Original source manifests can remain byte-identical provenance declarations with explicit
omissions; such a subset must never claim complete canonical or exact-restorable source capability.
Missing independent native5 stored analysis is reported explicitly.

Full-state requires a checkpoint and complete selected canonical dependency graph. It includes
immutable history pages, complete record/array owners and exact parent/append dependencies. Each
original root-relative path maps to a verified content-addressed transport object. Receive
reconstructs those paths under a fresh owned `source/` directory, validates them with `StateReader`,
and returns the original StateRef. Shared archive objects use the same immutable bytes; a second
full tensor copy is unnecessary. Science receive exposes selected objects and capabilities without a
full canonical source root.

One tar/XZ archive preserves original native5 bytes; multipart fallback, implicit compaction and
array re-encoding are unsupported. Hashing, compression and readback stream buffers rather than
materializing scientific tensors. Full-file SHA bytes can substantially exceed a small selected
array slice; physical and logical read counters are separate. Export consistency and integrity are
engineering evidence, not convergence, physics, discretization or experimental acceptance.
