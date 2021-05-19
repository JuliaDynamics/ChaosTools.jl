using ChaosTools
using DynamicalSystemsBase
using Test
using OrdinaryDiffEq

@testset "Uncertainty exponent tests" begin

@testset "Test uncertainty orginal paper" begin

    # C. Grebogi, S. W. McDonald, E. Ott, J. A. Yorke, Final state sensitivity: An obstruction to predictability, Physics Letters A, 99, 9, 1983
    function grebogi_map(u, p, n)
        J₀=0.3; a=1.32; b=0.9;
        θ = u[1]; x = u[2]
        dθ= θ + a*sin(2*θ) - b*sin(4*θ) -x*sin(θ)
        dθ = mod(dθ,2π) # to avoid problems with attractor at θ=π
        dx=-J₀*cos(θ)
        return SVector{2}(dθ,dx)
    end

    # dummy function to keep the initializator happy
    function grebogi_map_J(z0, p, n)
       return
    end
    ds = DiscreteDynamicalSystem(grebogi_map,[1., -1.], [] , grebogi_map_J)
    integ  = integrator(ds)

    θg=range(0,2π,length=250)
    xg=range(-0.5,0.5,length=250)

    bsn,att=basins_map2D(θg, xg, integ)

    e,f,α=uncertainty_exponent(θg,xg,bsn; precision=1e-5)

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
integ  = integrator(ds)

xg=range(-1.,1.,length=300)
yg=range(-1.,1.,length=300)

bsn,att=basins_map2D(xg, yg, integ)

e,f,α=uncertainty_exponent(xg,yg,bsn; precision=1e-5)

# Value (published) from the box-counting dimension is 1.42. α ≃ 0.6
@test (0.55 ≤ α ≤ 0.65)

end

end
