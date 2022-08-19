using Test, ChaosTools
using ChaosTools.DynamicalSystemsBase, ChaosTools.DelayEmbeddings
using Random

@testset "magnetic pendulum" begin
    d, α, ω = 0.3, 0.2, 0.5
    ds = Systems.magnetic_pendulum(; d, α, ω)
    xg = yg = range(-3, 3, length = 101)
    ds = projected_integrator(ds, 1:2, [0.0, 0.0])
    mapper = AttractorsViaRecurrences(ds, (xg, yg); Δt = 1.0)
    rr = range(1, 0; length = 101)
    ps = [[1, 1, γ] for γ in rr]
    pidx = :γs
    sampler, = statespace_sampler(; min_bounds = [-3,-3], max_bounds = [3,3])

    continuation = RecurrencesSeedingContinuation(mapper; threshold = 0.1)
    # With this threshold all attractors are mapped to each other, they are within
    # distance 1 in state space.
    fractions_curves, attractors_info = basins_fractions_continuation(
        continuation, ps, pidx, sampler; show_progress = false, samples_per_parameter = 100
    )

    finalkeys = collect(keys(fractions_curves[end]))

    # for k in attractors_info
    #     @show collect(keys(k))
    # end

    for (i, p) in enumerate(ps)
        γ = p[3]
        fs = fractions_curves[i]
        attractors = attractors_info[i]
        k = sort!(collect(keys(fs)))
        @test maximum(k) ≤ 3
        # @show k
        attk = sort!(collect(keys(attractors)))
        @test k == attk
        @test all(fk -> fk ∈ k, finalkeys)
        # It is arbitrary what id we get, because the third
        # fixed point that vanishes could have any of the three ids
        # But we can test for sure how many ids we have
        if γ < 0.2
            @test length(k) == 2
        else
            @test length(k) == 3
        end
        @test sum(values(fs)) ≈ 1
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
    # Don't know where this came from or what is happening in this parameter range
    function new_henon(x, p, n)
        return SVector{2}(p[1] - x[1]^2 - (1 - p[2])*x[2],  x[1])
    end
    a = 0.0
    ν = 0.01
    u0 = [0.0, 0.6]
    ds = DiscreteDynamicalSystem(new_henon, u0, [a,ν])
    ps = range(0.0, 0.4; length = 101)

    # This is standard henon map
    ds = Systems.henon(; b = 0.3, a = 1.4)
    ps = range(1.2, 1.25; length = 101)
    # In these parameters we go from a chaotic attractor to a period 7 orbit at a≈1.2265
    # (you can see this by launching our wonderful `interactive_orbitdiagram` app).
    # So we can use this to test different matching processes
    # (because "by distance" matches the two kind of attractors already)
    # Notice that the length=101 is rather sensitive and depending on it, some
    # much smaller periodic windows exist in the range. (For 101, a period-14 exists e.g.)
    acritical = 1.2265

    xg = yg = range(-2.5, 2.5, length = 500)
    mapper = AttractorsViaRecurrences(ds, (xg, yg),
            mx_chk_fnd_att = 3000,
            mx_chk_loc_att = 3000
    )
    pidx = 1
    sampler, = statespace_sampler(Random.MersenneTwister(1234);
        min_bounds = [-2,-2], max_bounds = [2,2]
    )
    distance_function = function (A, B)
        # length of attractors within a factor of 2, then distance is ≤ 1
        return abs(log(2, length(A)) - log(2, length(B)))
    end
    # notice that without this special distance function, even with a
    # really small threshold like 0.2 we still get a "single" attractor
    # throughout the range. Now we get one with period 14, a chaotic,
    # and one with period 7 that spans the second half of the parameter range
    continuation = RecurrencesSeedingContinuation(mapper;
        threshold = 0.99, metric = distance_function
    )
    fractions_curves, attractors_info = basins_fractions_continuation(
        continuation, ps, pidx, sampler;
        show_progress = false, samples_per_parameter = 100
    )

    for (i, p) in enumerate(ps)
        fs = fractions_curves[i]
        attractors = attractors_info[i]
        @test sum(values(fs)) ≈ 1
        # Test that keys are the same (-1 doesn't have attractor)
        k = sort!(collect(keys(fs)))
        -1 ∈ k && deleteat!(k, 1)
        attk = sort!(collect(keys(attractors)))
        @test k == attk
    end

    # unique keys
    unique_keys = unique!(reduce(vcat, [collect(keys(a)) for a in attractors_info]))
    # We can have at most 4 attractors: initial chaotic, period 14 in the middle,
    # chaotic again, and period 7 at the end. ALl of these should be matched to each other.
    @test length(unique_keys) ≤ 4

    # Animation of henon attractors
    # using GLMakie
    # fig = Figure()
    # ax = Axis(fig[1,1]; limits = (-2,2,-1,1))
    # colors = Dict(k => Cycled(i) for (i, k) in enumerate(unique_keys))
    # display(fig)
    # record(fig, "henon_test.mp4", eachindex(ps); framerate = 5) do i
    #     empty!(ax)
    #     p = ps[i]
    #     ax.title = "p = $p"
    #     # fs = fractions_curves[i]
    #     attractors = attractors_info[i]
    #     set_parameter!(ds, pidx, p)
    #     for (k, att) in attractors
    #         tr = trajectory(ds, 1000, att[1])
    #         scatter!(ax, tr[:, 1], tr[:, 2]; color = colors[k])
    #     end
    # end


end
