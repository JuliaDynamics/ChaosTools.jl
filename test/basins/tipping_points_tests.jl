using ChaosTools
using DynamicalSystemsBase
using Test

@testset "Tipping points" begin
    xg = yg = range(-4, 4, length = 100)
    ds = Systems.magnetic_pendulum(d=0.2, α=0.2, ω=0.8, N=3)
    pinteg = projected_integrator(ds, 1:2, [0,0];  diffeq=(;:reltol => 1e-9))
    mapper = AttractorsViaRecurrences(pinteg, (xg, yg); Δt = 1.)     
    basins_before, = basins_of_attraction(mapper; show_progress = false)
    pinteg.integ.p.γs = [1., 1., 0.05] # Change magnet strength
    mapper = AttractorsViaRecurrences(pinteg, (xg, yg); Δt = 1.)     
    basins_after, = basins_of_attraction(mapper; show_progress = false)
    @testset "Basin fractions" begin
        fs = basins_fractions(basins_before)
        @test sum(values(fs)) ≈ 1
        @test all(0.32 .≤ values(fs) .≤ 0.34)
        fs_after = basins_fractions(basins_after)
        @test length(fs_after) == 2
    end
    @testset "tipping probabilities" begin
        P = tipping_probabilities(basins_before, basins_after)
        @test size(P) == (3,2)
        @test all(0.495 .≤ P[1, :] .≤ 0.505) # must be ≈ 50%
        for i in 1:3; @test sum(P[i, :]) ≈ 1; end
    end
end
