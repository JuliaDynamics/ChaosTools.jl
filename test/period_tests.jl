using ChaosTools, Test
using Test
println("\nTesting period estimation...")
dt = 0.05

function test_algs(vs, ts, trueperiod, atol; methods = [
            :ac, :periodogram, :welch,
            :bartlett, :multitaper, :lombscargle
    ])

    for alg in methods

        @testset "$alg" begin

            @test estimate_period(vs, alg, ts) ≈ trueperiod atol = atol

        end

    end

end

@testset "simple sine" begin

    tsin = 0:dt:22π
    vsin = sin.(tsin)

    test_algs(vsin, tsin, 2π, dt)
end

@testset "sin(2x)" begin

    tsin = 0:dt:22π
    vsin = sin.(2 .* tsin)

    test_algs(vsin, tsin, π, dt)
end
using DynamicalSystemsBase
@testset "Roessler" begin

    ds = Systems.roessler(ones(3))
    T = 5000.0
    tr = trajectory(ds, T; Ttr = 100.0, dt = dt)
    v = tr[:, 1]
    t = 0:dt:T

    test_algs(v, t, 6, 1)

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

    test_algs(v, 0:dt:T, real_p, 0.1)

end