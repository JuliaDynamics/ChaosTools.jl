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

include("basins/basins_highlevel.jl")
include("basins/basins_lowlevel.jl")
include("basins/basins_utilities.jl")
include("basins/uncertainty_exp.jl")
include("basins/tipping.jl")

include("dimensions/linear_regions.jl")
include("dimensions/dims.jl")
include("dimensions/correlationdim.jl")
include("dimensions/correlation_fixedmass.jl")
include("dimensions/molteno.jl")
include("dimensions/kaplanyorke.jl")
include("dimensions/takens_best_estimate.jl")

include("nlts.jl")
include("dyca.jl")

include("period_return/periodic_points.jl")
include("period_return/period.jl")
include("period_return/transit_times_statistics.jl")

include("chaosdetection/lyapunovs.jl")
include("chaosdetection/gali.jl")
include("chaosdetection/expansionentropy.jl")
include("chaosdetection/partially_predictable.jl")
include("chaosdetection/testchaos01.jl")

# Ugly methods that shouldn't exist:
include("ugliness.jl")

include("deprecations.jl")

display_update = false
version = "1.10.0"
update_name = "update_v$version"

if display_update
if !isfile(joinpath(@__DIR__, update_name))
printstyled(stdout,
"""
\nUpdate message: ChaosTools v$version

A method to calculate the expansion entropy for discrete and
continuous systems is now included as `expansionentropy`!

See B. Hunt & E. Ott, ‘Defining Chaos’, Chaos 25.9 (2015).
\n
"""; color = :light_magenta)
touch(joinpath(@__DIR__, update_name))
end
end

end # module
