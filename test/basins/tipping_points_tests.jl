using ChaosTools
using DynamicalSystemsBase
using Test

@testset "Tipping probabilities" begin
    xg = yg = range(-4, 4, length = 100)
    ds = Systems.magnetic_pendulum(d=0.2, α=0.2, ω=0.8, N=3)
    basins_before, = basins_general(xg, yg, ds; dt=1., idxs=1:2)
    ds = Systems.magnetic_pendulum(d=0.2, α=0.2, ω=0.8, N=3, γs = [1.0, 1.0, 0.05])
    basins_after, = basins_general(xg, yg, ds; idxs = 1:2, reltol = 1e-9)
    @testset "Basin fractions" begin
        fs = basin_fractions(basins_before)
        @test sum(values(fs)) ≈ 1
        @test all(0.32 .≤ values(fs) .≤ 0.34)
        fs_after = basin_fractions(basins_after)
        @test length(fs_after) == 2
    end
    @testset "tipping probabilities" begin
        P = tipping_probabilities(basins_before, basins_after)
        @test size(P) == (3,2)
        @test all(P[1, :] .≈ 0.5)
        for i in 1:3; @test sum(P[i, :]) ≈ 1; end
    end
end
