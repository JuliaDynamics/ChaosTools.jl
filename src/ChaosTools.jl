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

include("periodic.jl")

include("chaos_detection.jl")

include("partially_predictable.jl")

# Ugly methods that shouldn't exist:
include("ugliness.jl")

end # module
