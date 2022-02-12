using ChaosTools
using DynamicalSystemsBase
using Test
using LinearAlgebra
using OrdinaryDiffEq

@testset "Recurrences method" begin
@testset "Henon map" begin

u1 = [0.0, 0.0] # converges to attractor
u2 = [0, 2.0] # diverges to inf

ds = Systems.henon(zeros(2); a = 1.4, b = 0.3)
xg = yg = range(-2.0, 2.0; length=100)
grid = (xg, yg)

mapper = AttractorsViaRecurrences(ds, grid)

i1 = mapper(u1)
i2 = mapper(u2)

@test i1 == 1
@test i2 == -1

sampler, = statespace_sampler(min_bounds = minimum.(grid), max_bounds = maximum.(grid))

fs = basin_fractions(mapper, sampler)
@test 0.1 < fs[1] < 0.9
@test 0.1 < fs[-1] < 0.9

ics = Dataset([sampler() for i in 1:1000])
fs = basin_fractions(mapper, sampler)
@test 0.1 < fs[1] < 0.9
@test 0.1 < fs[-1] < 0.9

end
end


@testset "Proximity method" begin
@testset "Henon map" begin 
    ds = Systems.henon(zeros(2); a = 1.4, b = 0.3)
    xg = yg = range(-2.,2.,length=100)
    basin, attractors = basins_of_attraction((xg,yg), ds; show_progress = false)



    res = 70
    ds = Systems.henon_iip(zeros(2); a = 1.4, b = 0.3)
    xg = yg = range(-2.,2.,length = res)
    grid = (xg,yg)
    mapper = AttractorMapper(ds; grid = grid)
    bsn = zeros(length(xg),length(yg))
    # Test if basins are (almost) identical
    I = CartesianIndices(bsn)
    for ind in I
        y0 = ChaosTools.generate_ic_on_grid(grid, ind)
        bsn[ind] = mapper(y0)
    end
    basins, att = basins_of_attraction((xg,yg), ds)
    @test sum(basins .!= bsn) < 5

    mapper = AttractorMapper(ds; attractors = att)
    l1 = mapper([1, 1])
    @test l1 == 1
end
end
