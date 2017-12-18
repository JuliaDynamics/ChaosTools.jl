__precompile__()

"""
Tools for the exploration of chaos and nonlinear dynamics
"""
module ChaosTools

using Reexport
@reexport using DynamicalSystemsBase

# Lyapunovs:
include("lyapunovs.jl")

# Entropies and Dimension Estimation:
include("dimensions/entropies.jl")
include("dimensions/dims.jl")

# Nonlinear Timeseries Analysis:
include("delay_coords.jl")

# Periodicity:
include("periodic.jl")

# Chaos Detection:
include("chaos_detection.jl")

# Visualization routines:
using Requires
@require PyPlot include("visualizations.jl")

end # module
