using Test

include("orbitdiagrams/orbitdiagram_tests.jl")
include("orbitdiagrams/poincare_tests.jl")

include("chaosdetection/lyapunov_exponents.jl")
include("chaosdetection/gali_tests.jl")
include("chaosdetection/partially_predictable_tests.jl")
include("chaosdetection/01test.jl")
include("chaosdetection/expansionentropy_tests.jl")

include("period_return/periodicity_tests.jl")
include("period_return/period_tests.jl")

include("dimensions/entropy_dimension.jl")
include("nlts_tests.jl")

ti = time() - ti
println("\nTest took total time of:")
println(round(ti, digits = 33), " seconds or ",
round(ti/60, digits = 3), " minutes")
