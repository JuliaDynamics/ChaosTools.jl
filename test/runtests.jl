ti = time()

# Continuous and discrete
include("lyapunov_exponents.jl")
include("gali_tests.jl")
include("partially_predictable_tests.jl")

# Continuous
include("poincare_tests.jl")

# Discrete
include("orbitdiagram_tests.jl")
include("periodicity_tests.jl")

# Numeric
include("entropy_dimension.jl")
include("nlts_tests.jl")
include("period_tests.jl")
include("01test.jl")

ti = time() - ti
println("\nTest took total time of:")
println(round(ti, digits = 33), " seconds or ",
round(ti/60, digits = 3), " minutes")
