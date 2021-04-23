using ChaosTools
using DynamicalSystemsBase
using Test
using OrdinaryDiffEq

@testset "Basins tests" begin

@testset "Test basin stroboscopic map" begin
    ω=1.; F = 0.2
    ds =Systems.duffing([0.1, 0.25]; ω = ω, f = F, d = 0.15, β = -1)
    integ_df  = integrator(ds; alg=AutoTsit5(Rosenbrock23()), reltol=1e-8, abstol=1e-8, save_everystep=false)
    xg = range(-2.2,2.2,length=100)
    yg = range(-2.2,2.2,length=100)
    bsn = basin_map(xg, yg, integ_df; T=2*pi/ω)
    @test length(unique(bsn.basin))/2 == 2
    @test count(bsn.basin .== 3) == 5376
    @test count(bsn.basin .== 5) == 4622

end

@testset "Test basin poincare map" begin
    b=0.1665
    ds = Systems.thomas_cyclical(b = b)
    xg=range(-6.,6.,length=100)
    yg=range(-6.,6.,length=100)
    pmap = poincaremap(ds, (3, 0.), Tmax=1e6; idxs = 1:2, rootkw = (xrtol = 1e-8, atol = 1e-8), reltol=1e-9)
    bsn = basin_map(xg, yg, pmap)

    @test length(unique(bsn.basin))/2 == 3
    @test count(bsn.basin .== 3) == 4639
    @test count(bsn.basin .== 5) == 2680
    @test count(bsn.basin .== 7) == 2665
end

@testset "Test basin discrete map" begin
    ds = Systems.henon(zeros(2); a = 1.4, b = 0.3)
    integ_df  = integrator(ds)
    xg = range(-2.,2.,length=100)
    yg = range(-2.,2.,length=100)
    bsn_nfo = basin_map(xg, yg, integ_df)

    @test count(bsn_nfo.basin .== 3) == 4127
    @test count(bsn_nfo.basin .== -1) == 5730
end

@testset "Test basin_general" begin
    ds = Systems.magnetic_pendulum(γ=1, d=0.2, α=0.2, ω=0.8, N=3)
    integ = integrator(ds, u0=[0,0,0,0], reltol=1e-9)
    xg=range(-2,2,length=100)
    yg=range(-2,2,length=100)
    bsn = basin_general_ds(xg, yg, integ; dt=1., idxs=1:2)

    @test count(bsn.basin .== 3) == 3331
    @test count(bsn.basin .== 5) == 3331
end


end
