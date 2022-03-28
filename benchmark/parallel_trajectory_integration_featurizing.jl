# export JULIA_NUM_THREADS=4
using Statistics, BenchmarkTools
using ChaosTools
using ChaosTools.DelayEmbeddings, ChaosTools.DynamicalSystemsBase
using Distributed, OrdinaryDiffEq
addprocs(4)
@everywhere using OrdinaryDiffEq, ChaosTools.DynamicalSystemsBase

#Magnetic pendulum (3 attractors)
function feature_extraction(t, A)
    x, y, z = columns(A)
    return [mean(x), mean(y), std(x), std(z)]
end

ds = Systems.lorenz84()

#parameters
T = 2000
Ttr = 200
Δt = 1.0


#region of interest
N = 1000;
# N = 50000;
min_limits = [-2,-2,-2];
max_limits = [2,2,2];

s, = statespace_sampler(min_bounds=min_limits, max_bounds=max_limits)
ics = Dataset([s() for i=1:N])

# Definition based on just looping over `trajectory`
function featurizer(ds::DynamicalSystem, u0::AbstractVector, feature_extraction; T=100, Ttr=100, Δt=1, diffeq=NamedTuple(), kwargs...)
    u = trajectory(ds, T, u0; Ttr, Δt, diffeq)
    t = Ttr:Δt:T+Ttr
    feature = feature_extraction(t, u)
    return feature
end
function featurizer(integ::SciMLBase.DEIntegrator, feature_extraction, u0; T=100, Ttr=100, Δt=1, diffeq=NamedTuple(), kwargs...)
    u = trajectory(integ, T, u0; Δt, Ttr)
    t = Ttr:Δt:T+Ttr
    feature = feature_extraction(t, u)
    return feature
end

function featurizer_allics_threads(ds, ics::Dataset, feature_extraction::Function;  kwargs...)
    num_samples = size(ics, 1) #number of actual ICs
    feature_array = Vector{Vector{Float64}}(undef, num_samples)
    Threads.@threads for i = 1:num_samples
        ic, _ = iterate(ics, i)
        feature_array[i] = featurizer(ds, ic, feature_extraction; kwargs...)
    end
    return reduce(hcat, feature_array)
end

# Definition based on looping over pre-initialized integrators
function featurizer_allics_threads_integrators(ds, ics::Dataset, feature_extraction::Function;  kwargs...)
    num_samples = size(ics, 1) #number of actual ICs
    feature_array = Vector{Vector{Float64}}(undef, num_samples)
    # TODO: Add DiffEq here
    integrators = [integrator(ds) for i in 1:Threads.nthreads()]
    _featurizer_allics_threads_integrators!(feature_array, integrators, ds, ics::Dataset, feature_extraction::Function;  kwargs...)
end # need a function barrier here because the integrator vector isn't type stable
function _featurizer_allics_threads_integrators!(feature_array, integrators, ds, ics::Dataset, feature_extraction::Function;  kwargs...)
    Threads.@threads for i = 1:length(feature_array)
        j = Threads.threadid()
        integ = integrators[j]
        feature_array[i] = featurizer(integ, feature_extraction, ics[i]; kwargs...)
    end
    return reduce(hcat, feature_array)
end

# Definition based on `EnsembleProblem` from DiffEq.
function featurizer_allics_ensemble(ds, ics::Dataset, feature_extraction::Function; T=100, Ttr=50,
    Δt=1, diffeq=NamedTuple(), EnsembleAlgorithm=EnsembleDistributed(), kwargs...)
    solver = DynamicalSystemsBase._get_solver(diffeq)
    prob = ODEProblem(ds, (ds.t0, T))
    @everywhere function prob_func(prob,i,repeat)
        remake(prob,u0=ics[i])
    end
    @eval @everywhere ics = $ics #way I found here: https://github.com/JuliaLang/julia/issues/9118 to broadcast ics to all workers
    ensemble_prob = EnsembleProblem(prob, prob_func=prob_func)
    sims = solve(ensemble_prob, solver, EnsembleAlgorithm, trajectories=length(ics), saveat=Ttr:Δt:T)

    feature_array = [feature_extraction(sim.t, sim.u) for sim in sims]

    return reduce(hcat, feature_array)
end

# Definitions based on using OnlineStats.jl
using OnlineStats
# First define the features in the form of OnlineStats
feature_extraction_online = (
    (OnlineStats.Mean(Float64), 1),
    (OnlineStats.Mean(Float64), 2),
    (OnlineStats.Variance(Float64), 1),
    (OnlineStats.Variance(Float64), 3),
)
# The define a function that initializes the integrators
function featurizer_threads_integrators_onlinestats(ds, ics::Dataset;  kwargs...)
    num_samples = size(ics, 1) #number of actual ICs
    feature_array = Vector{Vector{Float64}}(undef, num_samples)
    integrators = [integrator(ds) for i in 1:Threads.nthreads()]
    # convert feature extraction to actual online stats
    _featurizer_threads_integrators_onlinestats!(feature_array, integrators, ds, ics::Dataset;  kwargs...)
end # need a function barrier here because the integrator vector isn't type stable
# Finally, list the function that does the low-level computation
function _featurizer_threads_integrators_onlinestats!(feature_array, integrators, ds, ics::Dataset;
    Ttr = 100, T = 1000, Δt = 1.0, kwargs...)

    t0 = ds.t0
    tvec = (t0+Ttr):Δt:(T+t0+Ttr)

    Threads.@threads for i = 1:length(feature_array)
        j = Threads.threadid()
        integ = integrators[j]
        reinit!(integ, ics[i])

        # TODO: the feature extraction is handwritten for now
        # but can be easily made into generizable functions
        # Feature extraction needs to be recreated for each initial condition
        # (currently hardcoded but easily made into a function)
        extractors = (
            (OnlineStats.Mean(Float64), 1),
            (OnlineStats.Mean(Float64), 2),
            (OnlineStats.Variance(Float64), 1),
            (OnlineStats.Variance(Float64), 3),
        )

        step!(integ, Ttr)
        for (i, t) in enumerate(tvec)
            while t > integ.t
                step!(integ)
            end
            # TODO: Accesing state can be done with specified indices a-la `trajectory`
            current_state = integ(t)
            # Now we extract stuff into current state
            # (again, handwritten but easily made into a function)
            for k in 1:length(extractors)
                OnlineStats.fit!(extractors[k][1], current_state[extractors[k][2]])
            end
        end
        # Now transform extractors into feature vector
        current_feature = [OnlineStats.value(x[1]) for x in extractors]
        feature_array[i] = current_feature
    end
    return reduce(hcat, feature_array)
end



# %% Benchmark
@btime features_threads = featurizer_allics_threads(ds, ics, feature_extraction, T=T, Ttr=Ttr, Δt=Δt)
@btime features_threads_integrators = featurizer_allics_threads_integrators(ds, ics, feature_extraction, T=T, Ttr=Ttr, Δt=Δt)
@btime features_threads_onlinestats = featurizer_threads_integrators_onlinestats(ds, ics; T=T, Ttr=Ttr, Δt=Δt)
@btime features_ensemble_threads = featurizer_allics_ensemble(ds, ics, feature_extraction, T=T, Ttr=Ttr, Δt=Δt, EnsembleAlgorithm=EnsembleThreads())
@btime features_ensemble_distributed = featurizer_allics_ensemble(ds, ics, feature_extraction, T=T, Ttr=Ttr, Δt=Δt, EnsembleAlgorithm=EnsembleDistributed())

# Output
