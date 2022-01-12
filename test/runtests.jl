using ChaosTools
using Test
using DynamicalSystemsBase, DelayEmbeddings
using StatsBase
standardize = DelayEmbeddings.standardize
test_value = (val, vmin, vmax) -> @test vmin <= val <= vmax

ti = time()
@testset "ChaosTools tests" begin

include("basins/basins_tests.jl")
include("basins/uncertainty_tests.jl")
include("basins/tipping_points_tests.jl")

include("orbitdiagrams/orbitdiagram_tests.jl")
include("orbitdiagrams/poincare_tests.jl")

include("chaosdetection/lyapunov_exponents.jl")
#include("chaosdetection/gali_tests.jl") # TODO: make those faster
include("chaosdetection/partially_predictable_tests.jl")
include("chaosdetection/01test.jl")
#include("chaosdetection/expansionentropy_tests.jl")  # TODO: make those faster

include("period_return/periodicity_tests.jl")
include("period_return/period_tests.jl")
include("period_return/yin_tests.jl")
# include("period_return/transit_time_tests.jl")

include("dimensions/dims.jl")
include("dimensions/correlationdim.jl")
include("nlts_tests.jl")
# include("dyca_tests.jl") # TODO: fix DyCa tests once and for 
# all by using https://github.com/JuliaDynamics/JuliaDynamics/tree/master/timeseries

end

ti = time() - ti
println("\nTest took total time of:")
println(round(ti, digits = 33), " seconds or ",
round(ti/60, digits = 3), " minutes")
