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

ds = Systems.magnetic_pendulum(d=0.2, α=0.2, ω=0.8)
xg = yg = range(-2,2,length=100)
b₋, a₋ = basins_general(xg, yg, ds; dt=1., idxs=1:2)

ds = Systems.magnetic_pendulum(d=0.2, α=0.2, ω=0.8,  γs = [1, 1, 0.5])
b₊, a₊ = basins_general(xg, yg, ds; dt=1., idxs=1:2)

ids₊, ids₋ = sort!(collect(keys(a1))), sort!(collect(keys(a₊)))
l₊, l₋ = length.((ids₊, ids₋))

if l₊ > l₋
    error("We can only track if the a₊ < a")
    # TODO: just swap pluses and minuses.
end

overlaps = broadcast(
    (i, j) -> length(findall(isequal(i), b₋) ∩ findall(isequal(j), b₊)),
    ids₊, ids₋'
)

# Compute normalized overlaps of each basin with each other basin
overlaps = zeros(length(ids₊), length(ids₋))
for (i, ι) in enumerate(ids₊)
    Bi = findall(isequal(ι), b₋)
    for (ξ, j) in enumerate(ids₋)
        Bj = findall(isequal(ξ), b₊)
        overlaps[i, j] = length(Bi ∩ Bj)/length(Bj)
    end
end
overlaps

# Map indices of maximum overlap
replaces = Dict{Int, Int}()
for (i, ι) in enumerate(ids₊[1:end-1]) # last basin gets whatever index is left
    _, j = findmax(overlaps[i,:])
    if j ∈ values(replaces)
        @warn "Process failed. Found two attractors matching to the same one."
        # TODO: What to do here? Actually probably better to search through sorted
        # overlaps[i, :] until finding the best overlap... But probably also better
        # to sort `overlaps` so that the maximum value is at the first row.
    end
    replaces[ι] = ids₋[j]
end
# assign remaining unmatched attractor
replaces[ids₊[end]] = setdiff(ids₋, collect(values(replaces)))[1]



#
# elseif method == :distance
#     distances =
#         [minimum()]
# end
#
#
#

end
