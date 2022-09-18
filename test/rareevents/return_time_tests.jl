using ChaosTools, Test, Statistics
using ChaosTools.DynamicalSystemsBase
using ChaosTools.DelayEmbeddings

println("\n Return time tests...")

@testset "Standard map (exact)" begin
    # INPUT
    ds = Systems.standardmap()
    T = 10000 # maximum time

    # period 3 of standard map
    u0 = SVector(0.8121, 1.6243) # for default k!
    εs = sort!([4.0, 2.0, 0.01]; rev=true)
    # radius of 4.0 covers the first 2 points of the period 3 while 2.0 covers only
    # the first point. Therefore:
    # all first return times must be 1, and all second return times must be 2,
    # and 3rd must be same as second (because it doesn't matter how close are)
    exits, entries = exit_entry_times(ds, u0, εs, T)
    transits, returns = transit_return(exits, entries)

    @test all(x -> length(x) > 5, exits)
    @test all(x -> length(x) > 5, entries)
    @test all(issorted, exits)
    @test all(issorted, entries)
    @test all(isequal(2), transits[1])
    @test all(isequal(1), transits[2])
    @test transits[2] == transits[3]
    @test all(isequal(1), returns[1])
    @test all(isequal(2), returns[2])
    @test returns[2] == returns[3]

    τ, c = mean_return_times(ds, u0, εs, T)
    @test τ == mean.(returns)
    @test length(unique(c)) == 1
    @test c[1] == T÷3 # should return exactly 1 every 3 steps.

    ### quasiperiodic around period 3:
    u0 = SVector(0.877, 1.565)
    εs = sort!([4.0, 0.5, 0.1]; rev=true)
    exits, entries = exit_entry_times(ds, u0, εs, T)
    transits, returns = transit_return(exits, entries)

    @test all(issorted, exits)
    @test all(issorted, entries)
    @test all(x -> length(x) > 5, exits)
    @test all(x -> length(x) > 5, entries)

    # For ε=4.0, nothing changes with the before
    @test all(isequal(1), returns[1])
    @test all(isequal(2), transits[1])

    # Similarly, 0.5 should be the same as before
    @test all(isequal(1), transits[2])
    @test all(isequal(2), returns[2])

    # But now, the third entry is different, because it has the size of the quasiperiodic
    # stability island torous
    @test returns[3] ≠ returns[2]
    @test transits[3] ≠ transits[2]
    @test any(>(3), returns[3])
    @test all(isequal(1), transits[3]) # still need only one step to exit

    τ, c = mean_return_times(ds, u0, εs, T)
    @test τ == mean.(returns)
    @test length(unique(c)) == 2
    @test c[1] == T÷3
    @test c[2] == T÷3
    @test c[3] < T÷3

end

@testset "Towel map (boxes)" begin

    to = Systems.towel()
    tr = trajectory(to, 5000; Ttr = 10)
    u0 = tr[3000]

    # With these boxes, in the first 5 steps, the trajectory enters the y and z range
    # but not the x range. Therefore it should NOT enter the box. See figure!
    εs = [
        SVector(0.05, 0.05, 0.125),
        SVector(0.005, 0.005, 0.025),
    ]

    # Visual guidance
    # using PyPlot
    # tr0 = trajectory(to, 5, u0)
    # fig, axs = subplots(1,3)
    # comb = ((1, 2), (1, 3), (2, 3))
    # for i in 1:3
    #     j, k = comb[i]
    #     ax = axs[i]
    #     ax.scatter(tr[:, j], tr[:, k], s = 2, color = "C$(i-1)")
    #     ax.scatter([u0[j]], [u0[k]], s = 20, color = "k")
    #     ax.plot(tr0[:, j], tr0[:, k], color = "k")
    #     for l in 1:length(εs)
    #         rect = matplotlib.patches.Rectangle(
    #         u0[[j, k]] .- εs[l][[j, k]], 2εs[l][j], 2εs[l][k],
    #         alpha = 0.25, color = "k"
    #         )
    #         ax.add_artist(rect)
    #     end
    # end

    exits, entries = exit_entry_times(to, u0, εs, 10000)
    transits, returns = transit_return(exits, entries)

    @test all(issorted, exits)
    @test all(issorted, entries)
    @test length(exits[1]) > length(exits[2])
    @test returns[1][1] > 5
    @test mean(returns[1]) < mean(returns[2])

    τ, c = mean_return_times(to, u0, εs, 10000)
    @test τ == mean.(returns)
    @test c[1] > c[2] > 0

end
#
# @testset "Continuous Roessler" begin
# %%
# using OrdinaryDiffEq: Tsit5
alg = DynamicalSystemsBase.DEFAULT_SOLVER
diffeq = (alg = alg,)
ro = Systems.roessler(ones(3))
u0 = SVector(
    4.705494942754781,
    -10.221120945130545,
    0.06186563933318555
)
εs = sort!([1.0, 0.1, 0.01]; rev=true)
avg_period = 6.0
maxε = εs[1]
integ = integrator(ro, u0)

crossing_method =  ChaosTools.CrossingLinearIntersection()
step!(integ)
minimumkwargs = (iterations = 50, )
# optim = ChaosTools.closest_trajectory_point(integ, u0, maxε, 0, minimumkwargs)


exits, entries = exit_entry_times(ro, u0, εs, 1000.0; crossing_method)
transits, returns = transit_return_times(exits, entries)
mrt, ret = mean_return_times(ro, u0, εs, 1000.0; crossing_method)

# Sanity tests
@test mrt == mean.(returns)
for j in 1:length(εs)
    @test all(>(0), exits[j])
    @test all(>(0), entries[j])
    # all transits must be less than the returns (for the specific Roessler case)
    @test all(>(0), returns[j] .- transits[j])
    # all exits modulo the first happen after all entries
    @test all(>(0), transits[j])
    # return times are positive definite
    @test all(>(0), returns[j])

    @test ret[j] == lenth(entries[j])

end

# # Visual guidance
# %%
using GLMakie
using GLMakie.Makie.GeometryBasics

# The integration times have been tested for default integrator!
tr = trajectory(ro, 5000, u0; diffeq)
tr0 = trajectory(ro, avg_period, u0; diffeq)
tr2 = trajectory(ro,  18, u0; diffeq)
tr3 = trajectory(ro,  53, u0; diffeq)
# Trajectory 4 needs to show the actual integrator steps
tr4 = SVector{3, Float64}[]
integ = integrator(ro, u0; diffeq)
while integ.t < 416 # from 415 to 416 you see the first crossing :)
    step!(integ)
    push!(tr4, get_state(integ))
end
tr4 = Dataset(tr4)
# tr4 = trajectory(ro,  420, u0; diffeq)
fig = Figure(;resolution = (1500, 500)); display(fig)
axs = [Axis(fig[1,i]) for i in 1:3]
comb = ((1, 2), (1, 3), (2, 3))
for i in 1:3
    j, k = comb[i]
    ax = axs[i]
    # ax.plot(tr[:, j], tr[:, k], lw = 2.0, color = "C$(i-1)", alpha = 0.5, marker = "o", ms = 2)
    lines!(ax, tr[:, j], tr[:, k]; linewidth = 0.5, color = Cycled(1))
    scatter!(ax, [u0[j]], [u0[k]]; markersize = 5, color = "black")
    scatterlines!(ax, tr4[:, j], tr4[:, k]; color = "darkgreen", linewidth = 2.0, markersize = 6)
    lines!(ax, tr3[:, j], tr3[:, k]; color = "magenta", linewidth = 2.5)
    lines!(ax, tr2[:, j], tr2[:, k]; color = Cycled(2), linewidth = 3.0, linestyle = :dash)
    lines!(ax, tr0[:, j], tr0[:, k]; color = "black", linewidth = 3.0, linestyle = :dashdot)
    if eltype(εs[1]) <: Vector
        for l in 1:length(εs)
            # TODO: Update
            rect = matplotlib.patches.Rectangle(
            u0[[j, k]] .- εs[l][[j, k]], 2εs[l][j], 2εs[l][k],
            alpha = 0.1, color = "k"
            )
            ax.add_artist(rect)
        end
    else
        for l in 1:length(εs)
            poly!(ax, Circle(Point2f(u0[[j, k]]...), εs[l]); color = (:red, 0.1),
            strokecolor = :red, strokewidth = 0.5   )
        end
    end
end

# %%
# We know the average period of the Roessler system. Therefore the return times
# cannot be possibly smaller than it (because u0 is in the xy plane)
# We also know (from the plot) that the first returns after around 3 periods.
# We aaaalso know (by zooming in the plot) that the innermost ball is recurred exactly twice,
# however one of the two crossings is grazing and thus likely to not be spotted by the
# algorithm
exits, entries = exit_entry_times(ro, u0, εs, 30avg_period; diffeq = (alg = alg,))


τ, c = mean_return_times(ro, u0, εs, 3avg_period; diffeq = (alg = alg,), i=20)
@test c[1] == 1
@test c[2] == c[3] == 0
@test 2avg_period < τ[1] < 3avg_period
@test issorted(c; rev=true)

τ, c = mean_return_times(ro, u0, εs, 5000.0; diffeq = (alg = alg,), i=20)
@test all(τ .> avg_period/2)
@test 0 < c[3] ≤ 2
@test issorted(c; rev=true)

x = sort!(MathConstants.e .^ (-4:0.5:-1); rev = true)
Ts = 10.0 .^ range(3, 6, length = 7)
is = range(10; step = 4, length = 7)
τd, cd_ = mean_return_times(ro, u0, x, Ts; i=is, dmin =10.0)
@test issorted(τd)
@test all(z -> z > 0, cd_)

# figure()
# plot(log.(x), log.(τd); marker ="o")
# d = -ChaosTools.slope(log.(x), log.(τd))
# The slope of the above plot should approximate the fractal dimension
# but I don't find this to be true unfortunately...

# Test with hyper rectangles. Both same outcome because we are anyway in the flat part
εs = [
    SVector(0.1, 0.1, 0.5),
    SVector(0.1, 0.1, 0.05),
]
τ2, c2 = mean_return_times(ro, u0, εs, 5000.0; diffeq = (alg = alg,), i=20)

@test τ2[1] < τ[2]

# end
