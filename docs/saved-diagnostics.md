# Сохранённая диагностика: bounded read-only API

`qcl_negf_results.diagnostic_views` предоставляет два public entry point:

```python
from qcl_negf_results.diagnostic_views import (
    DiagnosticViewLimits, describe_saved_diagnostics, read_saved_history,
)

describe_saved_diagnostics(commit_path, expected_identity=expected)
read_saved_history(commit_path, 'scba',
                   {'start': 3, 'stop': 5, 'step': 1, 'columns': ['r_K', 'J']},
                   expected_identity=expected)
```

Это локальный reader уже сохранённых artifacts. Он не запускает solver, native/full
validator, materialization, export, live-history scan или performance-index lookup.
Remote provider отвечает за authentication, exact retained selection, ограниченную
доставку выбранного closure, timeout/cancellation и срок жизни temporary inputs.
Paths приходят от него, а не от произвольного HTTP поля. Наличие `physics.analysis`
не требуется: failed checkpoint с отдельным `science.history` допустим. Большой либо
неподдержанный **невыбранный** analysis owner не становится prerequisite истории.

## Exact binding и область проверки

`expected_identity` обязан содержать ровно required `point_id`, `execution_id`,
`attempt`, `plan_fingerprint`, `source_commit_sha256` и optional `run_uuid`,
`calcjob_uuid`. Required hashes — lowercase SHA256, attempt — положительный integer.
Это source attempt выбранного commit; он может быть меньше внешнего CalcJob attempt
при borrowed checkpoint. Записанные identity fields сверяются с expected без type
coercion. Optional provider-only UUID, отсутствующие в native/commit identity,
показываются как provider-only; reader не изобретает их присутствие в native payload.
Commit `state_id/state_sequence` сохраняются, но history recorder identity может
предшествовать этому state. Проверка point/execution/frozen-plan/attempt не выдаётся
за независимую state-level identity каждого исторического row.

Commit metadata ≤1 MiB и его exact expected SHA проверяются до открытия owner.
Canonical validator проверяет root commit/state и выбранный history+JSON dependency
closure. Inventory paths уникальны/нормализованы и ограничены4096 entries; file access
ограничен только выбранным closure. Невыбранные scientific schemas не валидируются
этим operation. Ровно один owner `science.history` с schema
`qcl-negf-scientific-history-v4`, media `application/x-hdf5` нужен для history.
Missing/ambiguous owner даёт descriptor unavailable, сохраняя resource metadata.
Неподдержанный выбранный schema/storage или corrupt identity даёт `ContractError`.

Owner path проверяется existing safe-path/POSIX regular `O_NOFOLLOW` primitives.
HDF5 использует budgeted stream того же opened descriptor. При read SHA проверяется
по полному объявленному owner перед numerical selection, затем hyperslab читается
через тот же descriptor. Подмена pathname не перенаправляет hash/selection. Проверки
opened inode/size/mtime обнаруживают изменение открытого содержимого. Это не snapshot
произвольно изменяемого чужого filesystem; provider передаёт immutable retained inputs.

Descriptor проверяет native4.0 header, role/schema/contract set, fixed scalar metadata,
таблицы/shape/dtypes и source identities, не читая numerical history arrays и не
заявляя owner SHA verification. Storage inventory ограничен4096 objects; soft/external
links, VDS и external raw storage не открываются. Читаемые metadata должны быть atomic
fixed strings/scalars; VLEN JSON/attributes и VLEN/nonnumeric history columns отвергаются.
Root physical-marker/full-state arrays этим API не валидируются и не читаются.

## DTO descriptor

Schema `qcl-negf.saved-diagnostics.v1` содержит:

* `identity`, `identity_binding:{native_recorded_fields,provider_only_fields}`,
  `source_commit_sha256`, `generation`, `closed`.
* `recorded_statuses:{terminal_status,quality,converged,scientific_accepted,stationary_assessment}`.
  Отсутствующие значения `null`. Failed/unconverged сохраняются как записано;
  успешное чтение не означает convergence/physics/discretization/experiment PASS.
* `history:{status,reason,source_artifact,phases}`. Status available/unavailable;
  reason `history_owner_missing`/`history_owner_ambiguous` или `null`.
  `source_artifact` — `null` либо `{artifact_id,role,schema,bytes,sha256}`.
  Каждая phase: `{phase,total_rows,columns:[{name,units,dtype}],attempt_spans}`.
* `resources`, `integrity`, `coverage`, `limits`, `io_counters` ниже.

`closed` — только explicit bool исходного commit, иначе `null`: immutable closed
history prefix не означает, что весь расчёт завершён. Units — native attribute как
сохранён; отсутствующий attribute даёт `null`, не вымышленный dimensionless или
конверсию current units. Native column names — labels, не произвольные HDF paths.

`metadata/identity_json` даёт attempt spans с `provenance:"native_identity"`.
`metadata/source_segments_json` должен быть fixed-string scalar ≤общего metadata
cap; после bounded decode допустимо ≤4096 source entries. Это **не16 pages**.
Each source несёт original segment SHA, core identity, nonnegative scba/outer counts
и optional declared first/last sequence bounds. Duplicate source hashes, future/foreign
attempts, inconsistent counts, overlapping/reversed declared sequence bounds отказаны.
Attempts упорядочиваются хронологически, а не по лексикографическому filename inventory.
Spans имеют `{start,stop,source_attempt,provenance:"declared_source_counts"}`.

Cumulative spans — происхождение из объявленных counts + chronological attempts.
Они не являются independently measured per-row attempt mapping. Selected native
sequence проверяется против bounds, когда они присутствуют для всех положительных
source counts этой попытки. Producer может не записывать outer sequence first/last:
reader не синтезирует их и не утверждает full outer coverage. Full coordinate scan
никогда не вызывается. Missing original source data обозначается ограничением,
не scientific PASS.

## DTO history slice

Phase ровно `scba|outer`. Selection имеет ровно `{start,stop,step,columns}`:
zero-based ordinals, exclusive stop, positive integer step, 1..16 distinct names
из descriptor value columns. Coordinates не являются user value columns.
Нет default tail/latest/full read, clipping или arbitrary native-path request.
Oversize selection отвергается до numerical read/allocation.

Schema `qcl-negf.saved-history-slice.v1` содержит common identity/source/generation/
closed/recorded_statuses, `phase`, echoed `selection`, `total_rows`, а также:

```text
coordinates = {
 sequence:[ExactScalar], iteration:[ExactScalar],
 outer_iteration:[ExactScalar]|null, source_attempt:[ExactScalar]
}
columns = [{name,units:string|null,values:[number|ExactScalar|null],
            invalid_mask:[null|"nan"|"positive_infinity"|"negative_infinity"]}]
break_before = [bool]
```

Int64 values за пределами JS safe integer представлены existing ExactScalar tag
`{kind:"integer",decimal:"9007199254740992"}`. Правило применяется рекурсивно ко
всем integer fields response, включая counters/identity/spans. Клиент не приводит
такие tags к Number. Finite zero/negative values сохраняются; NaN/±Inf становятся
`null` с точной причиной, никогда нулём. Log plot не может превращать nonpositive
значение в положительное.

`break_before` разрывает линии при смене source attempt/outer cycle, reset или
non-increasing iteration, sequence gap, invalid sample и после invalid sample.
Первый row каждой response — отдельное начало. `step>1` разрывает все selected rows:
пропущенные samples не доказывают непрерывность. Если start>0 и step1, читается ровно
одна предыдущая **coordinate** row, charged в тот же allocation/I/O budget; coverage
явно сообщает `previous_coordinate_row_checked`. Values предыдущего row и пропущенные
интервалы не сканируются. Strict positive/nondecreasing immutable sequence проверяется
на selected coordinate scope, а не на полной таблице. Iteration reset сам по себе
допустим и разрывает линию.

## Resource metadata

`resources` имеет `{status,reason,checkpoint,performance_summary}`. Checkpoint —
`null` либо ровно записанные `checkpoint_payload_seconds`,
`checkpoint_payload_bytes`, `includes_commit_publication`, `jit_separately_measured`
плюс `scope:"point_commit_checkpoint_payload"`. Отсутствующие fields `null`,
измеренный zero остаётся0. Это стоимость checkpoint payload в данном point commit,
**не job wall/CPU time**. Coverage flags не исправляются reader.

Current producer публикует performance summary отдельно. Без retained exact
same-attempt/session/cutoff binding этот API возвращает:
`performance_summary={status:"unavailable",reason:"source_binding_missing"}`.
Он не открывает произвольный `/performance/index`, не выбирает newest session,
не восстанавливает binding по pathname и не выдаёт checkpoint bytes/time за полный
CPU/RSS/time summary. Следующая producer задача при необходимости — explicit immutable
retained performance-commit binding; данный API не создаёт новый telemetry catalogue.
Slurm/AiiDA allocated grant accounting также отдельно; sampled CPU не определяет
полный charged grant или remaining quota.

## Общие конечные bounds и disclosure

Defaults: owner64 MiB; delivered stream I/O128 MiB; cumulative metadata1 MiB;
selection allocation1 MiB; JSON response2 MiB; HDF/artifact inventory4096;
selected rows512; selected value columns16; source inventory entries4096.
`DiagnosticViewLimits` позволяет только ужесточить эти public caps.
SHA bytes, HDF stream delivery, metadata/dependencies делят один I/O counter.
Decoded chunks всех touched datasets, selected values, coordinates, masks,
source spans и previous coordinate row входят в conservative allocation admission.
Каждый decoded chunk также проверяется до selection; неизвестные external filters
не вызываются. Returned JSON отдельно проверяется после ExactScalar projection.

`io_counters` сохраняет existing fields: `bytes_read_total`, `sha_bytes`, `hdf_bytes`,
`metadata_bytes`, `logical_selected_bytes`, `maximum_io_bytes`. Это bytes,
переданные read stream, не disk traffic и не performance cost solver.
`logical_selected_bytes` считает raw selected native numeric bytes; additional decoded
chunks/Python DTO/masks/spans учитываются allocation admission, не этим raw-byte counter.

`integrity` имеет `metadata_checked`, `owner_sha_verified`,
`native_validation:"not_performed"`, `scope` (`committed_history_metadata` либо
`owner_sha_and_selected_history_coordinates`). `coverage` содержит
`history_scope:"committed_closed_prefix"`, attempt-spans provenance,
`full_coordinate_scan:false`, `scientific_acceptance:"not_assessed"`.
Read добавляет selected_rows, previous_coordinate_row_checked и
`omitted_rows_disconnect:true`. Наличие valid SHA/таблицы не доказывает численную
сходимость, достаточность сетки или экспериментальную валидацию.

Focused literal fixtures в `tests/test_diagnostic_views.py` предназначены для
отдельно разрешённого конечного Python test stage; authoring source не исполняет
их и не создаёт HDF5 files. Они не являются native Julia integration или SCI evidence.
