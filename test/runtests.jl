# Additional tests may be run in this test suite according to an environment variable
# `CHAOSTOOLS_EXTENSIVE_TESTS` which can be true or false.
# If false, a small, but representative subset of tests is used.

# ENV["CHAOSTOOLS_EXTENSIVE_TESTS"] = true or false (set externally)

using ChaosTools
using Test

DO_EXTENSIVE_TESTS = get(ENV, "CHAOSTOOLS_EXTENSIVE_TESTS", "false") == "true"

function testfile(file, testname=defaultname(file))
    println("running test file $(file)")
    @testset "$testname" begin; include(file); end
    return
end
defaultname(file) = uppercasefirst(replace(splitext(basename(file))[1], '_' => ' '))

@testset "ChaosTools" begin

include("timeevolution/orbitdiagram.jl")

testfile("chaosdetection/lyapunovs.jl")
testfile("chaosdetection/gali.jl")
include("chaosdetection/partially_predictable.jl")
include("chaosdetection/01test.jl")
testfile("chaosdetection/expansionentropy.jl")

include("stability/fixedpoints.jl")
include("periodicity/periodicorbits.jl")
include("periodicity/period.jl")

# testfile("rareevents/return_time_tests.jl", "Return times")

testfile("dimreduction/broomhead_king.jl")
# TODO: simplify and make faster this:
# testfile("dimreduction/dyca.jl")

end
