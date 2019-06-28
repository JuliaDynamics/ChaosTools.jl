using ChaosTools, Test

println("\nTesting period estimation...")
dt = 0.05

@testset "simple sine" begin

    tsin = 0:dt:22π
    vsin = sin.(tsin)
    vsin2 = sin.(2tsin)

    @testset "ac" begin
        L = length(tsin)÷10
        p1 = estimate_period(vsin, tsin, :ac; L = L)
        @test p1 ≈ 2π   atol = dt
        p2 = estimate_period(vsin2, tsin, :ac; L = L)
        @test p2 ≈  π   atol = dt
    end
end

@testset "Roessler" begin

    ds = Systems.roessler(ones(3))
    T = 5000.0
    tr = trajectory(ds, T; Ttr = 100.0, dt = dt)
    v = tr[:, 1]
    t = 0:dt:T

    @testset "ac" begin
        p = estimate_period(v, 0:dt:T, :ac)
        # one oscillation of Roessler takes on average 6 time units
        # from looking at the plot of v vs t
        @test p ≈ 6  atol = 1
    end

end


@testset "Modified FitzHugh-Nagumo" begin

    function FHN(u, p, t)
        e, b, g = p
        v, w = u
        dv = min(max(-2 - v, v), 2 - v) - w
        dw = e*(v - g*w + b)
        return SVector(dv, dw)
    end

    g  = 0.8
    e = 0.04
    b = 0
    p0 = [e, b, g]

    fhn = ContinuousDynamicalSystem(FHN,SVector(-2, -0.6667),p0)
    T = 1000.0
    v = trajectory(fhn, T; dt = dt)[:, 1]
    real_p = 91

    @testset "ac" begin
        p = estimate_period(v, 0:dt:T, :ac)
        @test p ≈ real_p  atol = 0.1
    end

end
