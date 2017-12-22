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
    @test length(output[1]) == n
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
    @test length(output[1]) == n
  end
end

@testset "Poincare SOS" begin
  @testset "Henon Helies" begin
    ds = Systems.henonhelies([0, .483000, .278980390, 0] )
    psos = poincaresos(ds, 2, 1000.0)
    xcross = psos[:, 1]
    @test length(xcross) > 1
    @test xcross[1] != xcross[2]
  end
  @testset "Duffing" begin
    ds = Systems.duffing()
    psos = poincaresos(ds, 2)
    xcross = psos[:, 1]
    @test length(xcross) > 1
    @test xcross[1] != xcross[2]
  end
end

@testset "Produce OD" begin
  @testset "Duffing" begin
    ds = Systems.duffing()
    ds.state .= [0.1, 0.1, 0]

    pvalues = 2.348:-0.2:1.124
    parameter = :ω
    i = 1
    j = 2
    tf = 200.0
    ics = [vcat(2rand(2), 0.0) for i in 1:5]

    output = produce_orbitdiagram(ds, i, j, parameter, pvalues; tfinal = tf,
    Ttr = 100.0, ics = ics)

    @test length(output) == length(pvalues)
    for out in output
      @test length(out) > 1
    end
  end
  @testset "Shinriki" begin
    ds = Systems.shinriki([-2, 0, 0.2])

    pvalues = linspace(19,22,11)
    parameter = :R1
    i = 1
    j = 2
    tf = 200.0

    de = Dict(:abstol=>1e-9, :reltol => 1e-9)
    output = produce_orbitdiagram(ds, j, i, parameter, pvalues; tfinal = tf,
    Ttr = 100.0, diff_eq_kwargs = de, direction = -1)
    @test length(output) == length(pvalues)
    for out in output
      @test length(out) > 1
    end
  end
end
