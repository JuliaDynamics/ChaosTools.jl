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
        continuation, ps, pidx, sampler; show_progress = false, samples_per_parameter = 1000
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
    # # Plot code for fractions
    # using GLMakie
    # x = [fs[finalkeys[1]] for fs in fractions_curves]
    # y = [fs[finalkeys[2]] for fs in fractions_curves]
    # z = zeros(length(x))
    # fig = Figure(resolution = (400, 300))
    # ax = Axis(fig[1,1])
    # display(fig)
    # γs = [p[3] for p in ps]
    # band!(ax, γs, z, x; color = Cycled(1), label = "1")
    # band!(ax, γs, x, x .+ y; color = Cycled(2), label  = "2")
    # band!(ax, γs, x .+ y, 1; color = Cycled(3), label = "3")
    # xlims!(ax, 0, 1)
    # ylims!(ax, 0, 1)
    # ax.ylabel = "fractions"
    # ax.xlabel = "magnet strength"
    # axislegend(ax)
    # Makie.save("magnetic_fracs.png", fig; px_per_unit = 4)

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
    # fig = Figure(); display(fig)
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


@testset "lorenz84" begin

end

using OrdinaryDiffEq
F = 6.886; G = 1.347; a = 0.255; b = 4.0
ds = Systems.lorenz84(; F, G, a, b)
diffeq = (alg = Vern9(), reltol = 1e-9, abstol = 1e-9, maxiters = 1e12)
M = 200; z = 3
xg = yg = zg = range(-z, z; length = M)
grid = (xg, yg, zg)

sampler, = statespace_sampler(Random.MersenneTwister(1234);
    min_bounds = minimum.(grid), max_bounds = maximum.(grid)
)
mapper = AttractorsViaRecurrences(ds, grid;
diffeq, Δt = 0.01, show_progress = false)

# coexistance of periodic and chaotic, and then the chaotic collapses
# into the fixed point via a "crisis" (aka global bifurcation).
# stable fixed point exists always throughout the parameter range,
# but after the collapse, a fixed point and periodic attractor exist
Grange = range(1.32, 1.37; length = 101)
Gidx = 2
# threshold = 0.01 is the ε value we give at the mapper test
continuation = RecurrencesSeedingContinuation(mapper; threshold = 0.01)
fractions_curves, attractors_info = basins_fractions_continuation(
    continuation, Grange, Gidx, sampler;
    show_progress = true, samples_per_parameter = 100
)

# %%
unique_keys = unique!(reduce(vcat, [collect(keys(a)) for a in attractors_info]))
using GLMakie
fig = Figure(); display(fig)
ax = Axis(fig[1,1]; limits = (-1,3,-3,3))
colors = Dict(k => Cycled(i) for (i, k) in enumerate(unique_keys))
att_obs = Dict(k => Observable(Point2f[]) for k in unique_keys)
for k in unique_keys
    scatter!(ax, att_obs[k]; color = colors[k],
    label = "$k", markersize = 8)
end
axislegend(ax)
# display(fig)

record(fig, "lorenz84_test.mp4", eachindex(Grange); framerate = 4) do i
    p = Grange[i]
    ax.title = "p = $p"
    # fs = fractions_curves[i]
    attractors = attractors_info[i]
    set_parameter!(ds, Gidx, p)
    for (k, att) in attractors
        tr = trajectory(ds, 2000, att[1]; Δt = 1)
        att_obs[k][] = vec(tr[:, [1,2]])
        notify(att_obs[k])
    end
    # also ensure that attractors that don't exist are cleared
    for k in setdiff(unique_keys, collect(keys(attractors)))
        att_obs[k][] = Point2f[]; notify(att_obs[k])
    end
end