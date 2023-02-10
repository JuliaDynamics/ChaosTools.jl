using ChaosTools
using Test

@testset "discrete" begin
    logistic_rule(x, p, n) = @inbounds SVector(p[1]*x[1]*(1-x[1]))
    logistic = DeterministicIteratedMap(logistic_rule, [0.4], [4.0])

    i = 1
    parameter = 1
    pvalues = 2:0.01:4
    n = 50
    Ttr = 5000
    output = orbitdiagram(logistic, i, parameter, pvalues; n, Ttr)
    @test output[1][1] ≈ 0.5
    @test output[2][1] ≈ 0.502487562
    @test output[end][1] != output[end][2] != output[end][3]

    @testset "specific range" begin
        ulims = (0.0, 0.5)
        ics = [[0.0] for _ in 1:length(pvalues)]
        ics[end] = [0.5112661]
        output = orbitdiagram(
            logistic, i, parameter, pvalues;
            n, Ttr, u0 = ics, ulims
        )
        @test all(iszero, output[1])
        @test !all(iszero, output[end])
        @test all(≤(0.5), output[end])
    end
end

@testset "poincare" begin
    function lorenz_rule(u, p, t)
        σ = p[1]; ρ = p[2]; β = p[3]
        du1 = σ*(u[2]-u[1])
        du2 = u[1]*(ρ-u[3]) - u[2]
        du3 = u[1]*u[2] - β*u[3]
        return SVector{3}(du1, du2, du3)
    end
    u0 = fill(10.0, 3)
    p = [10, 28, 8/3]
    plane = (1, 0.0)
    i = 1
    parameter = 2
    pvalues = range(26, 28; length = 11)
    n = 50
    Ttr = 500

    ds = CoupledODEs(lorenz_rule, recursivecopy(u0), p)
    od1 = produce_orbitdiagram(deepcopy(ds), plane, i, parameter, pvalues; n, Ttr)

    pmap = PoincareMap(ds, plane)
    od2 = orbitdiagram(pmap, i, parameter, pvalues; n, Ttr)

    @test od1 == od2

    # TODO: We have a problem here. Don't know why!
    for j in eachindex(od1)
        @test all(x -> abs(x) < 1e-3, od1[j])
    end

end

@testset "stroboscopic" begin
    function duffing_rule(u,p,t)
        d, a, omega = p
        du1 =  u[2]
        du2 =  - u[1] - u[1]*u[1]*u[1] - d*u[2] + a*sin(omega*t)
        return SVector(du1, du2)
    end

    u0 = [0.1, 0.1]
    Trange = range(24, 26; length = 3)
    ωrange = @. 2π / Trange
    p0 = [7, 0.1, ωrange[1]]

    duffing = StroboscopicMap(CoupledODEs(duffing_rule, u0, p0), Trange[1])
    od = orbitdiagram(duffing, [1, 2], 3, ωrange; n = 10, periods = Trange)

    @test od[1][1] isa SVector{2, Float64}
    od1 = [[x[1] for x in v] for v in od]

    # Test that we have same stuff as book figure 9.2
    for u1 in od1
        @test all(x -> abs(x) < 0.1, u1)
    end

end
