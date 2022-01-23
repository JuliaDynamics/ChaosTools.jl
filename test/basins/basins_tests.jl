using ChaosTools
using DynamicalSystemsBase
using Test
using LinearAlgebra
using OrdinaryDiffEq

println("\nTesting basins of attraction...")

@testset "Basins tests" begin

@testset "Discrete map" begin
    ds = Systems.henon(zeros(2); a = 1.4, b = 0.3)
    xg = yg = range(-2.,2.,length=100)
    basin, attractors = basins_of_attraction((xg,yg), ds; show_progress = false)
    # pcolormesh(xg,yg, basin')
    @test 4260 ≤ count(basin .== 1) ≤ 4280
    @test 5700 ≤ count(basin .== -1) ≤ 5800
end

@testset "Test basin stroboscopic map" begin
    ds = Systems.duffing([0.1, 0.25]; ω = 1.0, f = 0.2, d = 0.15, β = -1)
    xg = yg = range(-2.2,2.2,length=100)
    T = 2π/1.0
    basin,attractors = basins_of_attraction((xg,yg), ds;
    T, diffeq = (;alg=Tsit5()), show_progress = false)
    # pcolormesh(xg, yg, basin')
    @test length(unique(basin)) == 2
    @test 4900 ≤ count(basin .== 1) ≤ 5100
    @test  4900 ≤  count(basin .== 2) ≤ 5100
end

@testset "Test basin poincare map" begin
    ds = Systems.thomas_cyclical(b = 0.1665)
    xg = yg = range(-6.0, 6.0; length = 100)
    pmap = poincaremap(ds, (3, 0.0), 1e6;
        rootkw = (xrtol = 1e-8, atol = 1e-8), diffeq=(reltol=1e-9,)
    )
    basin,attractors = basins_of_attraction((xg,yg), pmap; show_progress = false)
    # pcolormesh(xg,yg, basin')
    @test length(attractors) == 3
    @test  4610 ≤ count(basin .== 1) ≤ 4641
    @test  2660 ≤ count(basin .== 2) ≤ 2691
    @test  2640 ≤ count(basin .== 3) ≤ 2691
end

@testset "basins_of_attraction" begin
    ds = Systems.magnetic_pendulum(γ=1, d=0.2, α=0.2, ω=0.8, N=3)
    xg = range(-2,2,length=100)
    yg = range(-2,2,length=100)
    complete_state(y) = SVector(0.0, 0.0)
    basin, attractors = basins_of_attraction((xg,yg), ds;
    idxs=1:2, Δt = 1., complete_state, show_progress = false)
    # pcolormesh(xg,yg, basin')
    @test count(basin .== 1) == 3332
    @test count(basin .== 2) == 3332

    # Now test the zoom capability
    xg = yg = range(-2,-1.9,length=50)
    basins, att = basins_of_attraction((xg,yg), ds;
    idxs = 1:2, Δt = 1., complete_state, show_progress = false,
    attractors = attractors, mx_chk_lost = 1000, ε = 1e-3)
    @test count(basins .== 2) == 407
    @test count(basins .== 3) == 737
end

@testset "3D basins" begin
    ds = Systems.lorenz84()
    xg=yg=range(-1.,2.,length=100)
    zg=range(-1.5,1.5,length=30)
    bsn,att = basins_of_attraction((xg, yg, zg), ds; Δt = 1.0, show_progress = false)
    @test length(size(bsn)) == 3
    for i in 1:size(bsn)[3]
        # While there are 4 attractors, because system is chaotic we might
        # miss one of the 4 due to coarse state space partition
        @test sort!(unique(bsn[:, :, i])) ∈ ([1,2,3], [1,2,3,4])
    end
end

@testset "Basins for in-place system" begin
    ds = Systems.henon_iip(zeros(2); a = 1.4, b = 0.3)
    xg = yg = range(-2.,2.;length=100)
    basin, attractors = basins_of_attraction((xg,yg), ds; show_progress = false)
    # pcolormesh(xg,yg, basin')
    @test 4260 ≤ count(basin .== 1) ≤ 4280
    @test 5600 ≤ count(basin .== -1) ≤ 5800

    ds = Systems.lorenz_iip()
    xg = yg = range(-30.0, 30.0; length=10)
    zg = range(0, 50; length=10)
    basin, attractors = basins_of_attraction((xg,yg,zg), ds; show_progress = false, mx_chk_fnd_att = 1000)
    @test length(attractors) == 1
    @test unique(basin) == [1]
end

@testset "matching attractors" begin
    d, α, ω = 0.3, 0.2, 0.5
    ds = Systems.magnetic_pendulum(; d, α, ω)
    xg = yg = range(-3, 3, length = 100)
    b₋, a₋ = basins_of_attraction((xg,yg), ds; Δt=1., idxs=1:2, show_progress = false)
    @testset "method $method" for method ∈ (:overlap, :distance)
        @testset "γ3 $γ3" for γ3 ∈ [0.2, 0.1] # still 3 at 0.2, but only 2 at 0.1
            ds = Systems.magnetic_pendulum(; d, α, ω,  γs = [1, 1, γ3])
            b₊, a₊ = basins_of_attraction((xg,yg), ds; Δt=1., idxs=1:2, show_progress = false)
            match_attractors!(b₋, a₋, b₊, a₊, method)
            for k in keys(a₊)
                dist = minimum(norm(x .- y) for x ∈ a₊[k] for y ∈ a₋[k])
                @test dist < 0.2
            end
        end
    end
end


@testset "labelling mode test" begin
    res = 70
    ds = Systems.henon_iip(zeros(2); a = 1.4, b = 0.3)
    xg = yg = range(-2.,2.,length = res)
    grid = (xg,yg)
    mapper = AttractorMapper(ds; grid = grid)
    bsn = zeros(length(xg),length(yg))
    # Test if basins are (almost) identical
    I = CartesianIndices(bsn)
    for ind in I
        y0 = ChaosTools.generate_ic_on_grid(grid, ind)
        bsn[ind] = mapper(y0)
    end
    basins, att = basins_of_attraction((xg,yg), ds)
    @test sum(basins .!= bsn) < 5

    mapper = AttractorMapper(ds; attractors = att)
    l1 = mapper([1, 1])
    @test l1 == 1
end



end
