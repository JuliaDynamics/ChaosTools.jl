using ChaosTools
using Test, StaticArrays

println("\nTesting orbit diagrams...")

@testset "Orbit Diagram" begin
    @testset "Discrete 1D" begin
        ds = Systems.logistic()
        i = 1
        parameter = 1
        pvalues = 2:0.01:4
        ics = [rand() for m in 1:10]
        n = 50
        Ttr = 5000
        output = orbitdiagram(ds, i, parameter, pvalues; n = n, Ttr = Ttr)
        @test output[1][1] ≈ 0.5
        @test output[2][1] == 0.5024875621890547
        @test output[end][1] != output[end][2] != output[end][3]
    end

    @testset "Discrete 2D" begin
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

    @testset "Discrete IIP" begin
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
