using ChaosTools
using Test

test_value = (val, vmin, vmax) -> @test vmin <= val <= vmax

@testset "standard map" begin
# Known fixed point locations from publication:
o2x = [0.0
0.0
3.1416
3.1416
1.3346
4.9486]
o2y = [0.0
3.1416
3.1416
0.0
2.6693
3.6139]
o3x = [0.0
0.8121
5.471
0.8121
3.1416
5.471
3.1416
1.7774
4.5058
0.0
0.0
1.7774
6.2832
4.5058
6.2832]
o3y = [0.0
1.6243
2.3295
3.9537
2.3295
4.6589
3.9537
1.7774
4.5058
1.7774
4.5058
3.5548
1.7774
2.7284
4.5058]
ox = [[1.0], o2x, o3x]
oy = [[1.0], o2y, o3y]

@inbounds function standardmap_rule(x, par, n)
    theta = x[1]; p = x[2]
    p += par[1]*sin(theta)
    theta += p
    while theta >= 2π; theta -= 2π; end
    while theta < 0; theta += 2π; end
    while p >= 2π; p -= 2π; end
    while p < 0; p += 2π; end
    return SVector(theta, p)
end
ds = DeterministicIteratedMap(standardmap_rule, [0.001, 0.008], [0.971635])
xs = range(0, stop = 2π, length = 11); ys = copy(xs)
ics = [SVector(x,y) for x in xs for y in ys]

# All permutations of [±1, ±1]:
singss = [[+1, +1], [-1, -1], [+1, -1], [-1, +1]]
# I know from personal research I only need this `inds`:
indss = [[1,2]] # <- must be container of vectors!!!
λs = 0.005 # <- only this allowed to not be vector (could also be vector)

@testset "order = $o" for o in [2, 3]
    FP = periodicorbits(ds, o, ics, λs, indss, singss; spacetol = 1e-7)
    for fp in FP
        @test round(fp[1], digits = 4) ∈ ox[o]
        @test round(fp[2], digits = 4) ∈ oy[o]
    end
end

end
