using ChaosTools
using Test, StaticArrays

println("\nTesting orbit diagrams...")

@testset "Discrete Orbit Diagram" begin
    @testset "1D" begin
        ds = Systems.logistic(0.5112661)
        i = 1
        parameter = 1
        pvalues = 2:0.01:4
        n = 50
        Ttr = 5000
        output = orbitdiagram(ds, i, parameter, pvalues; n = n, Ttr = Ttr)
        @test output[1][1] ≈ 0.5
        @test output[2][1] == 0.5024875621890547
        @test output[end][1] != output[end][2] != output[end][3]

        @testset "specific range" begin
        ulims = (0.0, 0.5)
        ics = zeros(length(pvalues)); ics[end] = 0.5112661
        output = orbitdiagram(
            ds, i, parameter, pvalues;
            n = n, Ttr = Ttr, u0 = ics, ulims = ulims
        )
        @test all(iszero, output[1])
        @test !all(iszero, output[end])
        @test all(≤(0.5), output[end])
        end
    end

    @testset "2D" begin
        ds = Systems.standardmap()
        i = 2
        parameter = 1
        pvalues = 0:0.005:2
        ics = [0.001rand(2) for m in 1:10]
        n = 50
        Ttr = 5000
        output = orbitdiagram(ds, i, parameter, pvalues; n = n, Ttr = Ttr)
        @test length(output[1]) == n
    end

    @testset "IIP" begin
        ds = Systems.henon_iip()
        i = 2
        parameter = 1
        pvalues = [1.4, 1.5]
        ics = [rand(2), rand(2)]
        n = 50
        Ttr = 5000
        output = orbitdiagram(ds, i, parameter, pvalues; n = n, Ttr = Ttr)
        @test length(output[1]) == n
    end
end
