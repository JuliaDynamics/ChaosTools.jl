using Test, ChaosTools
using ChaosTools.DynamicalSystemsBase, ChaosTools.DelayEmbeddings

@testset "magnetic pendulum" begin
    d, α, ω = 0.3, 0.2, 0.
    ds = Systems.magnetic_pendulum(; d, α, ω)
    xg = yg = range(-3, 3, length = 201)
    ds = projected_integrator(ds, 1:2, [0.0, 0.0])
    mapper = AttractorsViaRecurrences(ds, (xg, yg); Δt = 1.0)
    rr = rand(10).*0.2 .+ 0.1
    ps = [[1, 1, γ] for γ in rr]
    pidx = :γs
    sampler, = statespace_sampler(; min_bounds = [-3,-3], max_bounds = [3,3])

    continuation = RecurrencesSeedingContinuation(mapper; threshold = 1.)
    # With this threshold all attractors are mapped to each other, they are within
    # distance 1 in state space.
    fractions_curves, attractors_info = basins_fractions_continuation(
        continuation, ps, pidx, sampler; show_progress = true, samples_per_parameter = 1000)

    finalkeys = collect(keys(fractions_curves[end]))

    for k in attractors_info
        @show collect(keys(k))
    end

    for (i, p) in enumerate(ps)
        γ = p[3]
        fs = fractions_curves[i]
        k = sort!(collect(keys(fs)))
        @show k
        # @test maximum(k) ≤ 3
        # @test all(fk -> fk ∈ k, finalkeys)
        # It is arbitrary what id we get, because the third
        # fixed point that vanishes could have any of the three ids
        # But we can test for sure how many ids we have
        # if γ < 0.3
        #     @test length(k) == 2
        # else
        #     @test length(k) == 3
        # end
        # @test sum(values(fs)) ≈ 1
    end
    # Plot code for fractions
    # using GLMakie
    # x = [fs[finalkeys[1]] for fs in fractions_curves]
    # y = [fs[finalkeys[2]] for fs in fractions_curves]
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
