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

include("deprecations.jl")

include("orbitdiagrams/discrete_diagram.jl")
include("orbitdiagrams/poincare.jl")
include("orbitdiagrams/produce_orbitdiagram.jl")

include("basins/attractor_mapping.jl")
include("basins/attractor_mapping_proximity.jl")
include("basins/attractor_mapping_featurizing.jl")
include("basins/attractor_mapping_recurrences.jl")
include("basins/basins_of_attraction.jl")
include("basins/basins_utilities.jl")
include("basins/fractality_of_basins.jl")
include("basins/tipping.jl")
include("basins/sampler.jl")

include("dimensions/linear_regions.jl")
include("dimensions/generalized_dim.jl")
include("dimensions/correlationdim.jl")
include("dimensions/correlation_fixedmass.jl")
include("dimensions/molteno.jl")
include("dimensions/kaplanyorke.jl")
include("dimensions/takens_best_estimate.jl")

include("dimreduction/broomhead_king.jl")
include("dimreduction/dyca.jl")

include("period_return/periodic_points.jl")
include("period_return/period.jl")
include("period_return/transit_times_statistics.jl")
include("period_return/fixedpoints.jl")
include("period_return/yin.jl")

include("chaosdetection/lyapunovs.jl")
include("chaosdetection/gali.jl")
include("chaosdetection/expansionentropy.jl")
include("chaosdetection/partially_predictable.jl")
include("chaosdetection/testchaos01.jl")

end # module
