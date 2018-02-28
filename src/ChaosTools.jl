__precompile__()

"""
Tools for the exploration of chaos and nonlinear dynamics
"""
module ChaosTools

using Reexport
@reexport using DynamicalSystemsBase

using DynamicalSystemsBase: DS, DDS
using DynamicalSystemsBase: CDS, DEFAULT_DIFFEQ_KWARGS
using DynamicalSystemsBase: MDI

export reinit!

# Lyapunovs:
include("lyapunovs.jl")

# Phase space related:
include("orbitdiagram.jl")

# Entropies and Dimension Estimation:
include("dimensions/entropies.jl")
include("dimensions/dims.jl")

# Nonlinear Timeseries Analysis:
include("nlts.jl")

# Periodicity:
include("periodic.jl")

# Chaos Detection:
include("chaos_detection.jl")

# Visualization routines:
using Requires
@require PyPlot include("visualizations.jl")

end # module
