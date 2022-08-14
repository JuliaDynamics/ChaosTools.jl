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

    continuation = RecurrencesSeedingContinuation(mapper)
    # With this threshold all attractors are mapped to each other, they are within
    # distance 1 in state space.
    fractions_curves, attractors_info = basins_fractions_continuation(
        continuation, ps, pidx, sampler
    )

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
    # Plot code for fractions
    # using GLMakie
    # x = [fs[1] for fs in fractions_curves]
    # y = [fs[3] for fs in fractions_curves]
    # z = zeros(length(x))
    # fig = Figure()
    # ax = Axis(fig[1,1])
    # display(fig)
    # γs = [p[3] for p in ps]
    # band!(ax, γs, z, x; color = :violet)
    # band!(ax, γs, x, x .+ y; color = :blue)
    # band!(ax, γs, x .+ y, 1; color = :red)
    # ylims!(ax, 0, 1)
end
