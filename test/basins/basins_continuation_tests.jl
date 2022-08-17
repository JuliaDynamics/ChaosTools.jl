using Test, ChaosTools
using ChaosTools.DynamicalSystemsBase, ChaosTools.DelayEmbeddings

@testset "magnetic pendulum" begin
    d, α, ω = 0.3, 0.2, 0.
    ds = Systems.magnetic_pendulum(; d, α, ω)
    xg = yg = range(-3, 3, length = 201)
    ds = projected_integrator(ds, 1:2, [0.0, 0.0])
    mapper = AttractorsViaRecurrences(ds, (xg, yg); Δt = 1.0)
    rr = range(1.,0., length = 100)
    ps = [[1, 1, γ] for γ in rr]
    pidx = :γs
    sampler, = statespace_sampler(; min_bounds = [-3,-3], max_bounds = [3,3])

    continuation = RecurrencesSeedingContinuation(mapper; threshold = 0.1)
    # With this threshold all attractors are mapped to each other, they are within
    # distance 1 in state space.
    fractions_curves, attractors_info = basins_fractions_continuation(
        continuation, ps, pidx, sampler; show_progress = true, samples_per_parameter = 100)

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


@testset "Henon map" begin
    function new_henon(x, p, n)
        return SVector{2}(p[1] - x[1]^2 - (1 - p[2])*x[2],  x[1])
    end
    a = 0.0
    ν = 0.01
    u0 = [0.0, 0.6]
    ds = DiscreteDynamicalSystem(new_henon, u0, [a,ν])
    xg = yg = range(-2.5, 2.5, length = 500)
    mapper = AttractorsViaRecurrences(ds, (xg, yg),
            mx_chk_fnd_att = 3000,
            mx_chk_loc_att = 3000)
    ps = range(0.0, 0.4; length = 10)
    pidx = 1
    sampler, = statespace_sampler(; min_bounds = [-2,-2], max_bounds = [2,2])
    continuation = RecurrencesSeedingContinuation(mapper; threshold = 0.2)
    fractions_curves, attractors_info = basins_fractions_continuation(
        continuation, ps, pidx, sampler;
        show_progress = true, samples_per_parameter = 100)

    for k in attractors_info
        @show collect(keys(k))
    end

    for (i, p) in enumerate(ps)
        fs = fractions_curves[i]
        k = sort!(collect(keys(fs)))
        @show k
    end
end
