# Native integration fixture: the production SCBA loop remains active while Python
# exports its first committed, explicitly unaccepted analytical state.
using LinearAlgebra
using Unitful
using QCLNEGFRunner
import QCLNEGF
include("native_physics_fixture.jl")

struct ExportIntegrationStop <: Exception end
root = only(ARGS)
fixture = native_physics_fixture(; energy_nodes = 33)
identity = Dict{String,Any}(
    "point_id"=>"live-point",
    "execution_id"=>"live-execution",
    "attempt"=>1,
    "plan_fingerprint"=>"live-export-integration",
)
recorder = QCLNEGFRunner.ScientificHistoryRecorder(root, identity, Ref(0), Ref(0))
options = SolverOptions(
    max_scba = 1_000_000,
    max_poisson = 1,
    convergence = ConvergencePolicy(
        required_consecutive_scba_passes = 1_000_001,
        stagnation_window = 0,
        stagnation_relative_improvement = 0.0,
    ),
)
production = ProductionOptions(
    parallel_backend = :blas,
    worker_count = 1,
    checkpoint_every_scba = 1,
    checkpoint_every_outer = 0,
    progress_every_scba = 0,
    progress_every_outer = 0,
)
previous = Ref(time_ns())
function observe(state)
    now = time_ns()
    iteration = last(state.history).ν
    open(joinpath(root, "native-iterations.csv"), "a") do io
        println(io, iteration, ",", (now - previous[])*1e-9)
    end
    previous[] = now
    QCLNEGFRunner.record_scientific_history!(
        recorder,
        :scba,
        1,
        last(state.history),
        fixture.problem,
    )
    if iteration <= 6
        QCLNEGFRunner.flush_scientific_history!(recorder)
        snapshot = NEGFSolution(
            fixture.problem,
            options,
            fixture.Uᴴ,
            QCLNEGF._electron_density_bar(fixture.problem, state.green.Gˡ),
            state,
            OuterIteration[],
            Dict{Symbol,Any}(),
            fixture.report,
            false,
            :running_scba,
        )
        QCLNEGFRunner.commit_point_artifacts(
            root,
            snapshot;
            identity,
            algorithms = AlgorithmOptions(),
            # Explicit codec fixture: inputs used by this small I/O-only model.
            # Positive exponents must keep their Float64 type through Julia's
            # writer, Python validation and Julia's envelope reader.
            configuration = Dict{String,Any}(
                "numerical"=>Dict{String,Any}(),
                "physical"=>Dict{String,Any}(),
                "fixture_codec"=>Dict(
                    "float"=>1.0e9,
                    "integer"=>1_000_000_000,
                    "negative_zero"=>-0.0,
                ),
            ),
            history_paths = [
                joinpath(recorder.directory, row["path"]) for row in recorder.segments
            ],
            # A selected recovery generation does not inherit stale analysis
            # from an earlier state. Publish an analysis of the selected cut.
            analysis = iteration in (1, 6),
        )
        commit=QCLNEGFRunner.YAML.load_file(joinpath(root, "artifacts/current.json"))
        model=joinpath(
            root,
            "artifacts",
            dirname(commit["commit_path"]),
            "resolved_configuration.json",
        )
        restored=QCLNEGFRunner.load_resolved_configuration_envelope(model)
        @assert restored["fixture_codec"]["float"] isa Float64
    end
    isfile(joinpath(root, "stop-request")) && throw(ExportIntegrationStop())
end
try
    solve_scba_production(
        fixture.problem,
        fixture.Uᴴ;
        options,
        production_options = production,
        iteration_callback = observe,
    )
catch error
    error isa ExportIntegrationStop || rethrow()
end
