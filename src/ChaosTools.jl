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
DEI = DEIntegrator

include("orbitdiagrams/discrete_diagram.jl")
include("orbitdiagrams/poincare.jl")
include("orbitdiagrams/produce_orbitdiagram.jl")

include("basins/dict_utils.jl")
include("basins/mapping/attractor_mapping.jl")
include("basins/basins_of_attraction.jl")
include("basins/basins_utilities.jl")
include("basins/fractality_of_basins.jl")
include("basins/tipping.jl")
include("basins/sampler.jl")
include("basins/continuation/basins_fractions_continuation_api.jl")

include("dimensions/linear_regions.jl")
include("dimensions/generalized_dim.jl")
include("dimensions/correlationsum_vanilla.jl")
include("dimensions/correlationsum_boxassisted.jl")
include("dimensions/correlationsum_fixedmass.jl")
include("dimensions/molteno.jl")
include("dimensions/kaplanyorke.jl")
include("dimensions/takens_best_estimate.jl")

include("dimreduction/broomhead_king.jl")
include("dimreduction/dyca.jl")

include("stablemotion/periodic_points.jl")
include("stablemotion/period.jl")
include("stablemotion/fixedpoints.jl")
include("stablemotion/yin.jl")

include("rareevents/transit_times_statistics.jl")

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
