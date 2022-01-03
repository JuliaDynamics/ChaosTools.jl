using Revise
using DynamicalSystems
using ChaosTools
using StaticArrays
using BenchmarkTools
res = 70
ds = Systems.henon_iip(zeros(2); a = 1.4, b = 0.3)
xg = yg = range(-2.,2.,length = res)

bsn_nfo, integ = basins_of_attraction((xg,yg), ds; tracking_mode = true)

# Test if basins are (almost) identical
for x0 in xg, y0 in yg
    n = ChaosTools.basin_cell_index([x0, y0], bsn_nfo)
    bsn_nfo.basin[n] = ChaosTools.get_color_point!(bsn_nfo, integ, [x0, y0])
end
ind = iseven.(bsn_nfo.basin)
bsn_nfo.basin[ind] .+= 1
bsn_nfo.basin .= (bsn_nfo.basin .- 1) .÷ 2

basins, att = basins_of_attraction((xg,yg), ds; tracking_mode = false)

@show sum(basins .!= Matrix(bsn_nfo.basin))
@assert sum(basins .!= Matrix(bsn_nfo.basin)) < 10

# Benchmarks
@btime for x0 in xg, y0 in yg
    clr = ChaosTools.get_color_point!(bsn_nfo, integ, [x0, y0])
end

@btime basins, att = basins_of_attraction((xg,yg), ds; tracking_mode = false)
