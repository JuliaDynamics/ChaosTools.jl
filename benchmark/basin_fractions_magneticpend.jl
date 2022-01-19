# export JULIA_NUM_THREADS=25
using Statistics, BenchmarkTools
using ChaosTools, DelayEmbeddings, DynamicalSystemsBase
using Distributed, OrdinaryDiffEq
addprocs(25)
@everywhere using OrdinaryDiffEq, DynamicalSystemsBase

#Magnetic pendulum (3 attractors)
function feature_extraction(t, y)
    return y[end][1:2] #final position 
end

ds = Systems.magnetic_pendulum()

#parameters
T = 2000
Ttr = 200 
Δt = 1.0


#region of interest 
N = 1000;
# N = 50000; 
min_limits = [-1,-1,-1,-1]; 
max_limits = [1,1,1,1]; 
sampling_method = "uniform"; 

s = ChaosTools.sampler(min_bounds=min_limits, max_bounds=max_limits, method=sampling_method)
ics = Dataset([s() for i=1:N])

# Definition based on just looping over `trajectory`
function featurizer(ds::DynamicalSystem, u0::AbstractVector, feature_extraction; T=100, Ttr=100, Δt=1, diffeq=NamedTuple(), kwargs...)
    u = trajectory(ds, T, u0; Ttr=Ttr, Δt=Δt, diffeq) 
    t = Ttr:Δt:T+Ttr
    feature = feature_extraction(t, u)
    return feature
end
function featurizer(ds::DynamicalSystem, integ::SciMLBase.DEIntegrator, feature_extraction; T=100, Ttr=100, Δt=1, diffeq=NamedTuple(), kwargs...)
    u = trajectory(ds, integ, T, Δt, Ttr, nothing) 
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
        reinit!(integ, ics[i])
        feature_array[i] = featurizer(ds, integ, feature_extraction; kwargs...)
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

# %% Benchmark
@btime features_threads = featurizer_allics_threads(ds, ics, feature_extraction, T=T, Ttr=Ttr, Δt=Δt)
@btime features_threads_integrators = featurizer_allics_threads_integrators(ds, ics, feature_extraction, T=T, Ttr=Ttr, Δt=Δt)
@btime features_ensemble_threads = featurizer_allics_ensemble(ds, ics, feature_extraction, T=T, Ttr=Ttr, Δt=Δt, EnsembleAlgorithm=EnsembleThreads())
@btime features_ensemble_distributed = featurizer_allics_ensemble(ds, ics, feature_extraction, T=T, Ttr=Ttr, Δt=Δt, EnsembleAlgorithm=EnsembleDistributed())

# all(features_threads .- features_ensemble_distributed .< 1e-4) #true

#=
25 workers, N=50000, T = 2000
Threads:    23.125 s (1077057193 allocations: 31.03 GiB)
Ensemble (Distributed):     106.806 s (1176707851 allocations: 44.56 GiB)
Emseble (Threads):   33.503 s (1242096027 allocations: 41.51 GiB) [It prints a long error message saying "concurrency violation detected". But also outputs the feature values similar to the other methods. Don't know what is happening.]


25 workers, N=1000, T=2000
Threads:   256.846 ms (21540932 allocations: 635.44 MiB)
Ensemble (Distributed):   2.179 s (23598717 allocations: 916.71 MiB)
Ensemble (Threads):   343.171 ms (24875192 allocations: 852.93 MiB)
=#