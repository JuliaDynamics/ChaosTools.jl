using ChaosTools
using DynamicalSystemsBase
using Test
using OrdinaryDiffEq

@testset "Uncertainty exponent / fractal boundaries" begin

@testset "Test uncertainty orginal paper" begin
    ds = Systems.grebogi_map(rand(2))
    θg = range(0, 2π, length = 251)
    xg = range(-0.5, 0.5, length = 251)
    bsn, att = basins_of_attraction((θg, xg), ds; show_progress = false)
    e, f, α = uncertainty_exponent(bsn; range_ε = 3:15)
    # In the paper the value is roughly 0.2
    @test (0.2 ≤ α ≤ 0.3)
end

@testset "Test uncertainty Newton map" begin

function newton_map(dz,z, p, n)
    f(x) = x^p[1]-1
    df(x)= p[1]*x^(p[1]-1)
    z1 = z[1] + im*z[2]
    dz1 = f(z1)/df(z1)
    z1 = z1 - dz1
    dz[1]=real(z1)
    dz[2]=imag(z1)
    return
end

# dummy function to keep the initializator happy
function newton_map_J(J,z0, p, n)
   return
end

ds = DiscreteDynamicalSystem(newton_map,[0.1, 0.2], [3] , newton_map_J)
xg = yg = range(-1.,1.,length=300)
bsn,att = basins_of_attraction((xg, yg), ds; show_progress = false)
e,f,α = uncertainty_exponent(bsn; range_ε = 5:30)

# Value (published) from the box-counting dimension is 1.42. α ≃ 0.6
@test (0.55 ≤ α ≤ 0.65)

end

@testset "Basin entropy and Fractal test" begin
    ds = Systems.grebogi_map()
    θg=range(0,2π,length = 300)
    xg=range(-0.5,0.5,length = 300)
    basin, attractors = basins_of_attraction((θg,xg), ds; show_progress = false)
    Sb, Sbb = basin_entropy(basin, 6)
    @test 0.4 ≤ Sb ≤ 0.42
    @test 0.6 ≤ Sbb ≤ 0.61

    test_res, Sbb = basins_fractal_test(basin; ε = 5)
    @test test_res == :fractal

    ds = Systems.henon(zeros(2); a = 1.4, b = 0.3)
    xg = yg = range(-2.,2.,length = 300)
    basin, attractors = basins_of_attraction((xg,yg), ds; show_progress = false)
    test_res, Sbb = basins_fractal_test(basin; ε = 5)
    @test test_res == :smooth
end

end
