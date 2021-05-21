using ChaosTools
using DynamicalSystemsBase
using Test

@testset "Basins tests" begin

@testset "Test basin stroboscopic map" begin
    ω=1.; F = 0.2
    ds =Systems.duffing([0.1, 0.25]; ω = ω, f = F, d = 0.15, β = -1)
    integ_df  = integrator(ds; abstol=1e-8, save_everystep=false)
    xg = range(-2.2,2.2,length=100)
    yg = range(-2.2,2.2,length=100)
    basin,attractors = basins_map2D(xg, yg, integ_df; T=2*pi/ω)
    # pcolormesh(xg, yg, basin')

    @test length(unique(basin)) == 2
    @test 5300 ≤ count(basin .== 1) ≤ 5400
    @test  4600 ≤  count(basin .== 2) ≤ 4700

end

@testset "Test basin poincare map" begin
    b=0.1665
    ds = Systems.thomas_cyclical(b = b)
    xg=range(-6.,6.,length=100)
    yg=range(-6.,6.,length=100)
    pmap = poincaremap(ds, (3, 0.), Tmax=1e6; idxs = 1:2, rootkw = (xrtol = 1e-8, atol = 1e-8), reltol=1e-9)
    basin,attractors = basins_map2D(xg, yg, pmap)
    # pcolormesh(xg, yg, basin')

    @test length(attractors) == 3
    @test   4610 ≤ count(basin .== 1) ≤ 4641
    @test  2660 ≤ count(basin .== 2)  ≤ 2691
    @test  2640 ≤ count(basin .== 3) ≤ 2691
end

@testset "Test basin discrete map" begin
    ds = Systems.henon(zeros(2); a = 1.4, b = 0.3)
    integ_df  = integrator(ds)
    xg = range(-2.,2.,length=100)
    yg = range(-2.,2.,length=100)
    basin,attractors = basins_map2D(xg, yg, integ_df)
    # pcolormesh(xg, yg, basin')

    @test count(basin .== 1) == 4270
    @test count(basin .== -1) == 5730
end

@testset "Test basin_general" begin
    ds = Systems.magnetic_pendulum(γ=1, d=0.2, α=0.2, ω=0.8, N=3)
    xg=range(-2,2,length=100)
    yg=range(-2,2,length=100)
    basin,attractors = basins_general(xg, yg, ds; dt=1., idxs=1:2)
    # pcolormesh(xg, yg, basin')

    @test count(basin .== 1) == 3332
    @test count(basin .== 2) == 3332
end


end
