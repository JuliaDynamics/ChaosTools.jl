# Additional tests may be run in this test suite according to an environment variable
# `ATTRACTORS_EXTENSIVE_TESTS` which can be true or false.
# If false, a small, but representative subset of tests is used.

# ENV["CHAOSTOOLS_EXTENSIVE_TESTS"] = true or false (set externally)


using ChaosTools
using Test
DO_EXTENSIVE_TESTS = get(ENV, "CHAOSTOOLS_EXTENSIVE_TESTS", "false") == "true"

defaultname(file) = uppercasefirst(replace(splitext(basename(file))[1], '_' => ' '))
testfile(file, testname=defaultname(file)) = @testset "$testname" begin; include(file); end

@testset "ChaosTools" begin

# include("orbitdiagrams/orbitdiagram_tests.jl")
# include("orbitdiagrams/poincare_tests.jl")

testfile("chaosdetection/lyapunovs.jl")
testfile("chaosdetection/gali.jl")
# include("chaosdetection/partially_predictable_tests.jl")
# include("chaosdetection/01test.jl")
tesfile("chaosdetection/expansionentropy.jl")

# include("period_return/periodicity_tests.jl")
# include("period_return/period_tests.jl")
# include("period_return/yin_tests.jl")

# testfile("rareevents/return_time_tests.jl", "Return times")

# include("nlts_tests.jl")
# include("dyca_tests.jl")

end
