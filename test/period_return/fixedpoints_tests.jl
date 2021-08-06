using ChaosTools, DynamicalSystemsBase, Test

ds = Systems.henon()
IntervalRootFinding = ChaosTools.IntervalRootFinding
using IntervalRootFinding: (..), (×)

x = 0.0..(2π - 100eps())
box = x × x

ds = Systems.standardmap()
fp, eigs, stable = fixedpoints(ds, box)

println("\nTesting fixed points...")

@testset "Fixed points" begin
    @testset "Henon map" begin
        x = -1.5..1.5
        y = -0.5..0.5
        box = x × y
        
        fp, eigs, stable = fixedpoints(ds, box)
        
        # henon fixed points analytically
        hmfp(a, b) = (x = -(1-b)/2a - sqrt((1-b)^2 + 4a)/2a; return SVector(x, b*x))
        henon_fp = hmfp(ds.p[1], ds.p[2])
        
        @test size(fp) == (2,2)
        @test stable == [false, false]
        @test (henon_fp ≈ fp[1]) || (henon_fp ≈ fp[2])
    end

    @testset "standard map order" begin

    end

    @testset "Lorenz system" begin

    end

end
