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
    xcross = psos[:, 2]
    @test length(xcross) > 1
    for x in xcross
      @test abs(x) < 1e-12
    end

    @inline Vhh(q1, q2) = 1//2 * (q1^2 + q2^2 + 2q1^2 * q2 - 2//3 * q2^3)
    @inline Thh(p1, p2) = 1//2 * (p1^2 + p2^2)
    @inline Hhh(q1, q2, p1, p2) = Thh(p1, p2) + Vhh(q1, q2)
    @inline Hhh(u::AbstractVector) = Hhh(u...)

    E = [Hhh(p) for p in psos]

    @test std(E) < 1e-12
    @test max(@. E - E[1]) < 1e-10
  end
end

@testset "Produce OD" begin
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

    v = round.(output[1], 4)
    s = collect(Set(v))
    @test length(s) == 1
    @test s[1] == -0.856

    v = round.(output[4], 4)
    s = Set(v)
    @test length(s) == 2
    @test s == Set([-0.376, -1.2887])

  end
end

@testset "Stroboscopic" begin
  ds = Systems.duffing(β = -1, ω = 1, f = 0.3)
  a = trajectory(ds, 100000.0, dt = 2π)
  D = information_dim(a)
  @test 1.3 < D < 1.5
end
