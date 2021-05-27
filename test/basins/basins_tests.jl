using ChaosTools
using DynamicalSystemsBase
using Test

@testset "Basins tests" begin

@testset "Test basin stroboscopic map" begin
    ω=1.; F = 0.2
    ds =Systems.duffing([0.1, 0.25]; ω = ω, f = F, d = 0.15, β = -1)
    integ_df  = integrator(ds; abstol=1e-8, save_everystep=false)
    xg = range(-2.2,2.2,length=100)
    yg = range(-2.2,2.2,length=100)
    basin,attractors = basins_map2D(xg, yg, integ_df; T=2*pi/ω)
    # pcolormesh(xg, yg, basin')

    @test length(unique(basin)) == 2
    @test 5300 ≤ count(basin .== 1) ≤ 5400
    @test  4600 ≤  count(basin .== 2) ≤ 4700

end

@testset "Test basin poincare map" begin
    b=0.1665
    ds = Systems.thomas_cyclical(b = b)
    xg=range(-6.,6.,length=100)
    yg=range(-6.,6.,length=100)
    pmap = poincaremap(ds, (3, 0.), Tmax=1e6; idxs = 1:2, rootkw = (xrtol = 1e-8, atol = 1e-8), reltol=1e-9)
    basin,attractors = basins_map2D(xg, yg, pmap)
    # pcolormesh(xg, yg, basin')

    @test length(attractors) == 3
    @test   4610 ≤ count(basin .== 1) ≤ 4641
    @test  2660 ≤ count(basin .== 2)  ≤ 2691
    @test  2640 ≤ count(basin .== 3) ≤ 2691
end

@testset "Test basin discrete map" begin
    ds = Systems.henon(zeros(2); a = 1.4, b = 0.3)
    integ_df  = integrator(ds)
    xg = range(-2.,2.,length=100)
    yg = range(-2.,2.,length=100)
    basin,attractors = basins_map2D(xg, yg, integ_df)
    # pcolormesh(xg, yg, basin')

    @test count(basin .== 1) == 4270
    @test count(basin .== -1) == 5730
end

@testset "Test basin_general" begin
    ds = Systems.magnetic_pendulum(γ=1, d=0.2, α=0.2, ω=0.8, N=3)
    xg=range(-2,2,length=100)
    yg=range(-2,2,length=100)
    basin,attractors = basins_general(xg, yg, ds; dt=1., idxs=1:2)
    # pcolormesh(xg, yg, basin')

    @test count(basin .== 1) == 3332
    @test count(basin .== 2) == 3332
end

@testset "Test matching attractors" begin
end

# %%

d, α, ω = 0.3, 0.2, 0.5
γ3 = 0.3

ds = Systems.magnetic_pendulum(; d, α, ω)
xg = yg = range(-3, 3, length = 100)
b₋, a₋ = basins_general(xg, yg, ds; dt=1., idxs=1:2)

ds = Systems.magnetic_pendulum(; d, α, ω,  γs = [1, 1, γ3])
b₊, a₊ = basins_general(xg, yg, ds; dt=1., idxs=1:2)

if length(a₊) > length(a₋)
    # Set it up so that modification is always done on `+` attractors
    a₋, a₊ = a₊, a₋
    b₋, b₊ = b₊, b₋
end
ids₊, ids₋ = sort!(collect(keys(a₊))), sort!(collect(keys(a₋)))

# TODO: Change the code: modification should be done on the
# basins with the LEAST amount of attractors, not most! Because then every
# single attractor is guaranteed to map to something!

# for testing
using PyPlot
LC =  matplotlib.colors.ListedColormap
fig, axs = subplots(1,2)
cmap = LC([matplotlib.colors.to_rgb("C$k") for k in 0:length(a₋)-1])
axs[1].pcolormesh(xg, yg, b₋'; cmap)
for (k, a) in a₋
    axs[1].scatter(a[1][1], a[1][2], color = "C$(k-1)", edgecolors = "white")
end
cmap = LC([matplotlib.colors.to_rgb("C$k") for k in 0:length(a₊)-1])
axs[2].pcolormesh(xg, yg, b₊'; cmap)
for (k, a) in a₊
    axs[2].scatter(a[1][1], a[1][2], color = "C$(k-1)", edgecolors = "white")
end
title("before")

# Compute normalized overlaps of each basin with each other basin
overlaps = zeros(length(ids₊), length(ids₋))
for (i, ι) in enumerate(ids₊)
    Bi = findall(isequal(ι), b₊)
    for (j, ξ) in enumerate(ids₋)
        Bj = findall(isequal(ξ), b₋)
        overlaps[i, j] = length(Bi ∩ Bj)/length(Bj)
    end
end
overlaps

# # Distances of attractors
using LinearAlgebra
closeness = zeros(length(ids₊), length(ids₋))
for (i, ι) in enumerate(ids₊)
    aι = a₊[ι]
    for (j, ξ) in enumerate(ids₋)
        aξ = a₋[ξ]
        closeness[i, j] = 1 / minimum(norm(x .- y) for x ∈ aι for y ∈ aξ)
    end
end
closeness

match_metric = closeness

# Create the mapping of replacements
replaces = Dict{Int, Int}()
for (i, ι) in enumerate(ids₊)
    v = match_metric[i, :]
    for j in sortperm(v) # go through the match metric in sorted order
        if ids₋[j] ∈ values(replaces)
            continue # do not use keys that have been used
        else
            replaces[ι] = ids₋[j]
        end
    end
end

# Do the actual replacing
replace!(b₊, replaces...)
aorig = copy(a₊)
for (k, v) ∈ replaces
    a₊[v] = aorig[k]
end
# delete unused keys
for k ∈ keys(a₊)
    if k ∉ values(replaces)
        delete!(a₊, k)
    end
end

fig, axs = subplots(1,2)
cmap = LC([matplotlib.colors.to_rgb("C$k") for k in 0:length(a₋)-1])
axs[1].pcolormesh(xg, yg, b₋'; cmap)
for (k, a) in a₋
    axs[1].scatter(a[1][1], a[1][2], color = "C$(k-1)", edgecolors = "white")
end
cmap = LC([matplotlib.colors.to_rgb("C$k") for k in 0:length(a₊)-1])
axs[2].pcolormesh(xg, yg, b₊'; cmap)
for (k, a) in a₊
    axs[2].scatter(a[1][1], a[1][2], color = "C$(k-1)", edgecolors = "white")
end
title("after")

#
#
#

# end
