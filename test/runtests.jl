ti = time()

# Continuous
include("lyapunov_exponents.jl")
include("chaos_detection_tests.jl")
include("poincare_tests.jl")

# Discrete
include("orbitdiagram_tests.jl")
include("periodicity_tests.jl")

# Numeric
include("entropy_dimension.jl")
include("nlts_tests.jl")
include("partially_predictable_tests.jl")

ti = time() - ti
println("\nTest took total time of:")
println(round(ti, digits = 33), " seconds or ",
round(ti/60, digits = 3), " minutes")
