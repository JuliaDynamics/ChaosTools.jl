using ChaosTools
using Test
using StatsBase
using Statistics

println("\nTesting Takens' best estimate")
@testset "Takens best" begin
  @testset "Henon map" begin
    ds = Systems.henon()
    ts = trajectory(ds, 5000)
    x = ts[:, 1]
    X = embed(x, 2, 1)
    D_C, D_C_95u, D_C_95l = takens_best_estimate(X, std(x)/4)
    @test 1.15 < D_C < 1.25
    @test D_C_95u < 1.05*D_C
    @test D_C_95l > 0.95*D_C
  end
  @testset "Lorenz System" begin
    ds = Systems.lorenz()
    tr = trajectory(ds, 2000; dt = 0.1)
    x = tr[:, 1]
    τ = estimate_delay(x, "mi_min", 1:20)
    X = embed(x, 4, τ)
    D_C, D_C_95u, D_C_95l = takens_best_estimate(X, std(x)/4)
    @test 1.85 < D_C < 2.1
    @test D_C_95u < 1.05*D_C
    @test D_C_95l > 0.95*D_C
  end
end
