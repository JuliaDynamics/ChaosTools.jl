"""
Tools for the exploration of chaos and nonlinear dynamics
"""
module ChaosTools

using Reexport
@reexport using DynamicalSystemsBase

using DynamicalSystemsBase: DS, DDS, CDS
using DynamicalSystemsBase: MDI, TDI
using DynamicalSystemsBase: stateeltype
using OrdinaryDiffEq
import OrdinaryDiffEq: ODEIntegrator

# Lyapunovs:
include("lyapunovs.jl")

# Phase space related:
include("orbitdiagrams/discrete_diagram.jl")
include("orbitdiagrams/poincare.jl")

# Entropies and Dimension Estimation:
include("dimensions/entropies.jl")
include("dimensions/dims.jl")

# Nonlinear Timeseries Analysis:
include("nlts.jl")
include("estimate_reconstruction.jl")

# Periodicity:
include("periodic.jl")

# Chaos Detection:
include("chaos_detection.jl")

# Ugly methods that shouldn't exist:
include("ugliness.jl")

end # module
