using ChaosTools
using DynamicalSystemsBase
using Test
using LinearAlgebra
using OrdinaryDiffEq

@testset "Basins tests" begin

@testset "Discrete map" begin
    ds = Systems.henon(zeros(2); a = 1.4, b = 0.3)
    xg = yg = range(-2.,2.,length=100)
    basin, attractors = basins_of_attraction((xg,yg), ds)
    # pcolormesh(xg,yg, basin')
    @test 4260 ≤ count(basin .== 1) ≤ 4280
    @test 5700 ≤ count(basin .== -1) ≤ 5800
end

@testset "Test basin stroboscopic map" begin
    ds = Systems.duffing([0.1, 0.25]; ω = 1., f = 0.2, d = 0.15, β = -1)
    xg = yg = range(-2.2,2.2,length=100)
    T = 2π/1.
    basin,attractors = basins_of_attraction((xg,yg), ds;  T=T, diffeq = (;alg=Tsit5()))
    # pcolormesh(xg, yg, basin')
    @test length(unique(basin)) == 2
    @test 4900 ≤ count(basin .== 1) ≤ 5100
    @test  4900 ≤  count(basin .== 2) ≤ 5100
end

@testset "Test basin poincare map" begin
    ds = Systems.thomas_cyclical(b = 0.1665)
    xg = yg = range(-6.,6.,length=100)
    pmap = poincaremap(ds, (3, 0.), Tmax=1e6; idxs = 1:2, rootkw = (xrtol = 1e-8, atol = 1e-8), reltol=1e-9)
    basin,attractors = basins_of_attraction((xg,yg), pmap)
    # pcolormesh(xg,yg, basin')
    @test length(attractors) == 3
    @test   4610 ≤ count(basin .== 1) ≤ 4641
    @test  2660 ≤ count(basin .== 2)  ≤ 2691
    @test  2640 ≤ count(basin .== 3) ≤ 2691
end

@testset "basin_general" begin
    ds = Systems.magnetic_pendulum(γ=1, d=0.2, α=0.2, ω=0.8, N=3)
    xg=range(-2,2,length=100)
    yg=range(-2,2,length=100)
    complete_state(y) = SVector(0,0)
    basin, attractors = basins_of_attraction((xg,yg), ds; idxs=1:2, complete_state)
    # pcolormesh(xg,yg, basin')
    @test count(basin .== 1) == 3332
    @test count(basin .== 2) == 3332
end

@testset "matching attractors" begin
    d, α, ω = 0.3, 0.2, 0.5
    ds = Systems.magnetic_pendulum(; d, α, ω)
    xg = yg = range(-3, 3, length = 100)
    b₋, a₋ = basins_of_attraction((xg,yg), ds; Δt=1., idxs=1:2)
    @testset "method $method" for method ∈ (:overlap, :distance)
        @testset "γ3 $γ3" for γ3 ∈ [0.2, 0.1] # still 3 at 0.2, but only 2 at 0.1
            ds = Systems.magnetic_pendulum(; d, α, ω,  γs = [1, 1, γ3])
            b₊, a₊ = basins_of_attraction((xg,yg), ds; Δt=1., idxs=1:2)
            match_attractors!(b₋, a₋, b₊, a₊, method)
            for k in keys(a₊)
                dist = minimum(norm(x .- y) for x ∈ a₊[k] for y ∈ a₋[k])
                @test dist < 0.2
            end
        end
    end
end

end
