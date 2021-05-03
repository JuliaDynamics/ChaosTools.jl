# 1 - Computing the basins of attraction

The technique used to compute the basin of attraction is described in ref. [1]. It consists in tracking the trajectory on the plane and coloring the points of according to the attractor it leads to. This technique is very efficient for 2D basins.

The algorithm gives back a matrix with the N attractors numbered with even numbers from 2 to 2N and their basins with odd numbers. An attractor is numbered with an *even number* 2n and its corresponding basin with an *odd number* 2n+1.

```@docs
basin_map
basin_general_ds
```

## 1.1 - Stroboscopic Maps

First define a dynamical system on the plane, for example with a *stroboscopic* map or Poincaré section. For example we can set up an dynamical system with a stroboscopic map defined:

```jl
using DynamicalSystems
ω=1.; F = 0.2
ds =Systems.duffing([0.1, 0.25]; ω = ω, f = F, d = 0.15, β = -1)
integ  = integrator(ds; reltol=1e-8)
```

Now we define the grid of ICs that we want to analyze and launch the procedure:

```jl
xg = range(-2.2,2.2,length=200)
yg = range(-2.2,2.2,length=200)
basin, attractors = basin_map(xg, yg, integ; T=2π/ω)
```

```jl
using PyPlot
pcolormesh(xg, yg, basin')
```

![image](https://i.imgur.com/R2veb5tl.png)

## 1.2 - Poincaré Maps

Another example with a Poincaré map:
```jl
using Plots
using DynamicalSystems
using Basins

ds = Systems.rikitake(μ = 0.47, α = 1.0)
integ=integrator(ds)
```

Once the integrator has been set, the Poincaré map can defined on a plane:

```jl
xg=range(-6.,6.,length=200)
yg=range(-6.,6.,length=200)
pmap = poincaremap(ds, (3, 0.), Tmax=1e6; idxs = 1:2, rootkw = (xrtol = 1e-8, atol = 1e-8), reltol=1e-9)

@time bsn = basin_poincare_map(xg, yg, pmap)

plot(xg,yg,bsn.basin',seriestype=:heatmap)
```

The arguments are:
* `pmap` : A Poincaré map as defined in [ChaosTools.jl](https://github.com/JuliaDynamics/ChaosTools.jl)


![image](https://i.imgur.com/xjdC8Hh.png)


## 1.3 - Discrete Maps

The process to compute the basin of a discrete map is very similar:

```jl
function newton_map(dz, z, p, n)
    z1 = z[1] + im*z[2]
    dz1 = f(z1, p[1])/df(z1, p[1])
    z1 = z1 - dz1
    dz[1]=real(z1)
    dz[2]=imag(z1)
    return
end
f(x, p) = x^p - 1
df(x, p)= p*x^(p-1)

# dummy Jacobian function to keep the initializator happy
function newton_map_J(J,z0, p, n)
   return
end

ds = DiscreteDynamicalSystem(newton_map,[0.1, 0.2], [3.0], newton_map_J)
integ  = integrator(ds)

xg=range(-1.5,1.5,length=200)
yg=range(-1.5,1.5,length=200)

bsn=basin_discrete_map(xg, yg, integ)
```

![image](https://i.imgur.com/ppHlGPbl.png)

## 1.4 Basins in Higher Dimensions

When you cannot define a Stroboscopic map or a well defined Poincaré map you can always try the general method for higher dimensions.
It is slower and may requires some tuning.
The algorithm looks for attractors on a 2D grid.
The initial conditions are set on this grid and all others variables are set to zero by default.

### Usage

```jl
ds = Systems.magnetic_pendulum(γ=1, d=0.2, α=0.2, ω=0.8, N=3)
integ = integrator(ds, u0=[0,0,0,0], reltol=1e-9)
xg=range(-4,4,length=150)
yg=range(-4,4,length=150)
@time bsn = basin_general_ds(xg, yg, integ; dt=1., idxs=1:2)
```

Keyword parameters are:
* `dt` : this is the time step. It is recommended to use a value above 1. The result may vary a little
depending on this time step.
* `idxs` : Indices of the two variables that define the plane.


![image](https://imgur.com/qgBHZ8Ml.png)

## 1.5 - Custom differential equations and low level functions.

Supose we want to define a custom ODE and compute the basin of attraction on a defined
Poincaré map:

```jl
using DifferentialEquations
using Basins

@inline @inbounds function duffing(u, p, t)
    d = p[1]; F = p[2]; omega = p[3]
    du1 = u[2]
    du2 = -d*u[2] + u[1] - u[1]^3 + F*sin(omega*t)
    return SVector{2}(du1, du2)
end

d=0.15; F=0.2; ω = 0.5848
ds = ContinuousDynamicalSystem(duffing, rand(2), [d, F, ω])
integ = integrator(ds; alg=Tsit5(),  reltol=1e-8, save_everystep=false)
xg = range(-2.2,2.2,length=200)
yg = range(-2.2,2.2,length=200)

iter_f! = (integ) -> step!(integ, 2π/ω, true)
reinit_f! =  (integ,y) ->  reinit!(integ, [y...])
get_u = (integ) -> integ.u[1:2]

bsn = draw_basin(xg, yg, integ, iter_f!, reinit_f!, get_u)
```

The following anonymous functions are important:
* iter_f! : defines a function that iterates the system one step on the map.
* reinit_f! : sets the initial conditions on the map. Remember that only the
initial conditions on the map must be set.
* get_u : it is a custom function to get the state of the integrator only for the variables
defined on the plane
