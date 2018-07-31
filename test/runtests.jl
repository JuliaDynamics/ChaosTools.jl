ti = time()

# lyapunov Exponents:
include("lyapunov_exponents.jl")
# Chaos Detection:
include("chaos_detection_tests.jl")
# Orbit Diagrams:
include("orbitdiagram_tests.jl")
# Periodicity:
include("periodicity_tests.jl")
# Entropies (and attractor dimensions)
include("entropy_dimension.jl")
# Nonlinear Timeseries Analysis:
include("nlts_tests.jl")
include("R_params.jl")

ti = time() - ti
println("\nTest took total time of:")
println(round(ti, digits = 33), " seconds or ",
round(ti/60, digits = 3), " minutes")
