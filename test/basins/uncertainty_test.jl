using ChaosTools
using DynamicalSystemsBase
using Test
using OrdinaryDiffEq

@testset "Uncertainty exponent tests" begin

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
integ  = integrator(ds)

xg=range(-1.,1.,length=300)
yg=range(-1.,1.,length=300)

bsn,att=basins_map2D(xg, yg, integ)

α,e,f=uncertainty_exponent(xg,yg,bsn; precision=1e-5)

@test (0.57 ≤ α ≤ 0.63)

end

end
