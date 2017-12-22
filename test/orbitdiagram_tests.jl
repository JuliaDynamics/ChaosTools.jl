if current_module() != ChaosTools
  using ChaosTools
end
using Base.Test, StaticArrays, OrdinaryDiffEq

@testset "Orbit Diagram" begin
  @testset "DiscreteDS1D" begin
    ds = Systems.logistic()
    i = 1
    parameter = :r
    pvalues = 2:0.01:4
    ics = [rand() for m in 1:10]
    n = 50
    Ttr = 5000
    output = orbitdiagram(ds, i, parameter, pvalues; n = n, Ttr = Ttr)
    @test output[1][1] ≈ 0.5
    @test output[2][1] == 0.5024875621890547
    @test output[end][1] != output[end][2] != output[end][3]
  end

  @testset "DiscreteDS" begin
    ds = Systems.standardmap()
    i = 2
    parameter = :k
    pvalues = 0:0.005:2
    ics = [0.001rand(2) for m in 1:10]
    n = 50
    Ttr = 5000
    output = orbitdiagram(ds, i, parameter, pvalues; n = n, Ttr = Ttr)
    @test length(output[1]) = length(ics)*n
  end

  @testset "BigDiscreteDS" begin
    ds = Systems.coupledstandardmaps(3)
    i = 2
    parameter = :ks
    pvalues = [a.* ones(3) for a in 0:0.005:0.5]
    ics = [0.001rand(6) for m in 1:10]
    n = 50
    Ttr = 5000
    output = orbitdiagram(ds, i, parameter, pvalues; n = n, Ttr = Ttr)
    @test length(output[1]) = length(ics)*n
  end
end
