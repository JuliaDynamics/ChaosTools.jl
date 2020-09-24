using ChaosTools, Test, Statistics

println("\nTesting transit time statistics...")
@testset "Transit time statistics" begin

@testset "Standard map (exact)" begin
# INPUT
ds = Systems.standardmap()
T = 10000 # maximum time

# period 3 of standard map
u0 = SVector(0.8121, 1.6243)
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

τ, c = mean_return_times(to, u0, εs, T)
@test τ == mean.(returns)
@test c[1] > c[2] > 0

end
#
@testset "Continuous Roessler" begin
using OrdinaryDiffEq: Tsit5
alg = Tsit5()

# %%
ro = Systems.roessler(ones(3))
tr = trajectory(ro, 5000; Ttr = 100)
u0 = trajectory(ro, 11; Ttr = 100)[end] # return center
εs = sort!([1.0, 0.1, 0.01]; rev=true)
avg_period = 6.0

# # Visual guidance
# %%
# using PyPlot
# tr = trajectory(ro, 5000, u0)
# tr0 = trajectory(ro, avg_period, u0)
# tr2 = trajectory(ro, 3avg_period, u0)
# fig, axs = subplots(1,3)
# comb = ((1, 2), (1, 3), (2, 3))
# for i in 1:3
#     j, k = comb[i]
#     ax = axs[i]
#     ax.plot(tr[:, j], tr[:, k], lw = 2.0, color = "C$(i-1)", alpha = 0.5)
#     ax.scatter([u0[j]], [u0[k]], s = 20, color = "k")
#     ax.plot(tr2[:, j], tr2[:, k], color = "C4", lw = 1.0, ls = "--")
#     ax.plot(tr0[:, j], tr0[:, k], color = "k", lw = 1.0, ls = "--")
#     if eltype(εs[1]) <: Vector
#         for l in 1:length(εs)
#             rect = matplotlib.patches.Rectangle(
#             u0[[j, k]] .- εs[l][[j, k]], 2εs[l][j], 2εs[l][k],
#             alpha = 0.1, color = "k"
#             )
#             ax.add_artist(rect)
#         end
#     else
#         for l in 1:length(εs)
#             circ = matplotlib.patches.Circle(
#                 u0[[j, k]], εs[l]; alpha = 0.1, color = "k"
#             )
#             ax.add_artist(circ)
#         end
#     end
# end

# %%
# We know the average period of the Roessler system. Therefore the return times
# cannot be possibly larger than it (because u0 is in the xy plane)
# We also know (from the plot) that the first returns after around 3 periods.
# We aaaalso know (by zooming in the plot) that the innermost ball is recurred three times

τ, c = mean_return_times(ro, u0, εs, 5000.0; alg)
@test all(τ .> avg_period/2)

#
# end

end
