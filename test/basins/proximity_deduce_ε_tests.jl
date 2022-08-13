using ChaosTools
using ChaosTools.DynamicalSystemsBase
using ChaosTools.DelayEmbeddings
using Test

@testset "Proximity deduce ε" begin
    ds = Systems.henon(zeros(2); a = 1.4, b = 0.3)
    @testset "single attractor, no ε" begin
        attractors = Dict(1 => trajectory(ds, 10000, [0.0, 0.0]; Δt = 1, Ttr=100))
        mapper = AttractorsViaProximity(ds, attractors)
        @test trunc(mapper.ε, digits = 2)  ≈ 0.18 # approximate size of attractor here
    end
    @testset "two attractors, analytically known ε" begin
        attractors = Dict(
            1 => Dataset([0 1.0]; warn = false),
            2 => Dataset([0 2.0]; warn = false)
        )
        mapper = AttractorsViaProximity(ds, attractors)
        @test mapper.ε == 0.5
    end
    @testset "one attractor, single point (invalid)" begin
        attractors = Dict(
            1 => Dataset([0 1.0]; warn = false),
        )
        @test_throws ArgumentError AttractorsViaProximity(ds, attractors)
    end
end
