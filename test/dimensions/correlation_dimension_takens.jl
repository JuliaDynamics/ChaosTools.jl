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
    D_C = takens_best_estimate(X, std(x)/4)
    @test 1.15 < D_C < 1.25
  end
  @testset "Lorenz System" begin
    ds = Systems.lorenz()
    tr = trajectory(ds, 2000; dt = 0.1)
    x = tr[:, 1]
    τ = estimate_delay(x, "mi_min", 1:20)
    X = embed(x, 4, τ)
    D_C = takens_best_estimate(X, std(x)/4)
    @test 1.85 < D_C < 2.1
  end
end
