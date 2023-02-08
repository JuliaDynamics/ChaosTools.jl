"""
Tools for the exploration of chaos and nonlinear dynamics
"""
module ChaosTools

using Reexport

@reexport using DelayEmbeddings
using ComplexityMeasures
@reexport using DynamicalSystemsBase


# include("orbitdiagrams/discrete_diagram.jl")
# include("orbitdiagrams/produce_orbitdiagram.jl")

# include("dimreduction/broomhead_king.jl")
# include("dimreduction/dyca.jl")

# include("stablemotion/periodic_points.jl")
# include("stablemotion/period.jl")
# include("stablemotion/fixedpoints.jl")
# include("stablemotion/yin.jl")

# include("rareevents/mean_return_times/mrt_api.jl")

include("chaosdetection/lyapunovs/lyapunov.jl")
include("chaosdetection/lyapunovs/lyapunov_from_data.jl")
include("chaosdetection/lyapunovs/lyapunovspectrum.jl")
include("chaosdetection/lyapunovs/local_growth_rates.jl")
include("chaosdetection/gali.jl")
include("chaosdetection/expansionentropy.jl")
# include("chaosdetection/partially_predictable.jl")
include("chaosdetection/testchaos01.jl")

# Copy linear regression from FractalDimensions.jl
import Statistics
function linreg(x::AbstractVector, y::AbstractVector)
    mx = Statistics.mean(x)
    my = Statistics.mean(y)
    b = Statistics.covm(x, mx, y, my)/Statistics.varm(x, mx)
    a = my - b*mx
    return a, b
end

include("deprecations.jl")

end # module
