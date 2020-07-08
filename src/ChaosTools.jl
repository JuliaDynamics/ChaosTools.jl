"""
Tools for the exploration of chaos and nonlinear dynamics
"""
module ChaosTools

using Reexport
@reexport using DynamicalSystemsBase

using DynamicalSystemsBase: DS, DDS, CDS
using DynamicalSystemsBase: MDI, TDI
using DynamicalSystemsBase: stateeltype
using DiffEqBase: AbstractODEIntegrator, u_modified!

include("lyapunovs.jl")

include("orbitdiagrams/discrete_diagram.jl")
include("orbitdiagrams/poincare.jl")

include("dimensions/entropies.jl")
include("dimensions/dims.jl")
include("dimensions/correlationdim.jl")

include("nlts.jl")

include("periodicity/periodic_points.jl")
include("periodicity/period.jl")

include("chaosdetection/gali.jl")
include("chaosdetection/expansionentropy.jl")
include("chaosdetection/partially_predictable.jl")
include("chaosdetection/testchaos01.jl")

# Ugly methods that shouldn't exist:
include("ugliness.jl")

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
