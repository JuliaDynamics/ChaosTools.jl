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

include("nlts.jl")

include("periodicity/periodic_points.jl")
include("periodicity/period.jl")

include("chaosdetection/gali.jl")
include("chaosdetection/partially_predictable.jl")
include("chaosdetection/testchaos01.jl")

# Ugly methods that shouldn't exist:
include("ugliness.jl")

end # module
