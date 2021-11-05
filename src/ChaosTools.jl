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
using DynamicalSystemsBase.DiffEqBase: AbstractODEIntegrator, u_modified!, DEIntegrator
DEI = DEIntegrator

include("orbitdiagrams/discrete_diagram.jl")
include("orbitdiagrams/poincare.jl")

include("basins/basins_reinit.jl")
include("basins/basins_highlevel.jl")
include("basins/basins_lowlevel.jl")
include("basins/basins_utilities.jl")
include("basins/uncertainty_exp.jl")
include("basins/tipping.jl")

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

include("chaosdetection/lyapunovs.jl")
include("chaosdetection/gali.jl")
include("chaosdetection/expansionentropy.jl")
include("chaosdetection/partially_predictable.jl")
include("chaosdetection/testchaos01.jl")

include("deprecations.jl")

end # module
