using Test, ChaosTools
using ChaosTools.DynamicalSystemsBase, ChaosTools.DelayEmbeddings

@testset "magnetic pendulum" begin
    d, α, ω = 0.3, 0.2, 0.5
    ds = Systems.magnetic_pendulum(; d, α, ω)
    xg = yg = range(-3, 3, length = 100)
    ds = projected_integrator(ds, 1:2, [0.0, 0.0])
    mapper = AttractorsViaRecurrences(ds, (xg, yg); Δt = 1.0)

    ps = [[1, 1, γ] for γ in 1.0:-0.1:0]
    pidx = :γs
    sampler, = statespace_sampler(; min_bounds = [-3,-3], max_bounds = [3,3])

    # With this threshold all attractors are mapped to each other, they are within
    # distance 1 in state space.
    fractions_curves = continuation_basins_fractions(mapper, ps, pidx, sampler; threshold = 1.0)

    for (i, p) in enumerate(ps)
        γ = p[3]
        fs = fractions_curves[i]
        k = sort!(collect(keys(fs)))
        @test max(k) ≤ 3
        if γ < 0.3
            @test k == 1:2
        else
            @test k == 1:3
        end
    end
end
