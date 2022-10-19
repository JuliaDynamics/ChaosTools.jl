"""
Tools for the exploration of chaos and nonlinear dynamics
"""
module ChaosTools

using DelayEmbeddings
using Entropies
using DynamicalSystemsBase

using DynamicalSystemsBase: DS, DDS, CDS
using DynamicalSystemsBase: MDI, TDI
using DynamicalSystemsBase: stateeltype
using DynamicalSystemsBase.SciMLBase: AbstractODEIntegrator, u_modified!, DEIntegrator
using Optim
DEI = DEIntegrator

include("orbitdiagrams/discrete_diagram.jl")
include("orbitdiagrams/produce_orbitdiagram.jl")

include("fractaldim/linear_regions.jl")
include("fractaldim/generalized_dim.jl")
include("fractaldim/correlationsum_vanilla.jl")
include("fractaldim/correlationsum_boxassisted.jl")
include("fractaldim/correlationsum_fixedmass.jl")
include("fractaldim/molteno.jl")
include("fractaldim/kaplanyorke.jl")
include("fractaldim/takens_best_estimate.jl")
include("fractaldim/higuchi.jl")

include("dimreduction/broomhead_king.jl")
include("dimreduction/dyca.jl")

include("stablemotion/periodic_points.jl")
include("stablemotion/period.jl")
include("stablemotion/fixedpoints.jl")
include("stablemotion/yin.jl")

include("rareevents/mean_return_times/mrt_api.jl")

include("chaosdetection/lyapunovs/lyapunovspectrum.jl")
include("chaosdetection/lyapunovs/lyapunov.jl")
include("chaosdetection/lyapunovs/local_growth_rates.jl")
include("chaosdetection/lyapunovs/lyapunov_from_data.jl")
include("chaosdetection/gali.jl")
include("chaosdetection/expansionentropy.jl")
include("chaosdetection/partially_predictable.jl")
include("chaosdetection/testchaos01.jl")

# Exports
# TODO: It is probably better to put all exports here rather than in
# each individual file. Or is it...? Not sure yet.
export lyapunovspectrum, lyapunov, local_growth_rates, lyapuov_from_data



include("deprecations.jl")

end # module
