using Test, ChaosTools.StaticArrays, OrdinaryDiffEq, LinearAlgebra

println("\nTesting producing continuous orbit diagrams...")

@testset "Produce OD Shinriki" begin
    ds = Systems.shinriki([-2, 0, 0.2])

    pvalues = range(19,stop=22,length=11)
    parameter = 1
    i = 1
    j = 2
    tf = 200.0

    diffeq = (abstol=1e-9, reltol = 1e-9)
    output = produce_orbitdiagram(ds, (j, 0.0), i, parameter, pvalues; tfinal = tf,
    Ttr = 100.0, printparams = false, direction = +1, diffeq)
    @test length(output) == length(pvalues)

    v = round.(output[1], digits = 4)
    s = collect(Set(v))
    @test length(s) == 1
    @test s[1] == -0.856

    v = round.(output[4], digits = 4)
    s = Set(v)
    @test length(s) == 2
    @test s == Set([-0.376, -1.2887])

end
