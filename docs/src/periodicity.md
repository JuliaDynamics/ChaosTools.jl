# Fixed points & Periodicity

## Fixed points
```@docs
fixedpoints
```

A rather simple example of the fixed points can be demonstrated using E.g., the Lorenz-63 system, whose fixed points can be calculated analytically to be
the following three
```math
(0,0,0) \\
\left( \sqrt{\beta(\rho-1)}, \sqrt{\beta(\rho-1)}, \rho-1 \right) \\
\left( -\sqrt{\beta(\rho-1)}, -\sqrt{\beta(\rho-1)}, \rho-1 \right) \\
```

So, let's calculate
```@example MAIN
using ChaosTools

function lorenz_rule(u, p, t)
    σ = p[1]; ρ = p[2]; β = p[3]
    du1 = σ*(u[2]-u[1])
    du2 = u[1]*(ρ-u[3]) - u[2]
    du3 = u[1]*u[2] - β*u[3]
    return SVector{3}(du1, du2, du3)
end
function lorenz_jacob(u, p, t)
    σ, ρ, β = p
    return SMatrix{3,3}(-σ, ρ - u[3], u[2], σ, -1, u[1], 0, -u[1], -β)
end

ρ, β = 30.0, 10/3
lorenz = CoupledODEs(lorenz_rule, 10ones(3), [10.0, ρ, β])
# Define the box within which to find fixed points:
x = y = interval(-20, 20)
z = interval(0, 40)
box = x × y × z

fp, eigs, stable = fixedpoints(lorenz, box, lorenz_jacob)
fp
```
and compare this with the analytic ones:

```@example MAIN
lorenzfp(ρ, β) = [
    SVector(0, 0, 0.0),
    SVector(sqrt(β*(ρ-1)), sqrt(β*(ρ-1)), ρ-1),
    SVector(-sqrt(β*(ρ-1)), -sqrt(β*(ρ-1)), ρ-1),
]

lorenzfp(ρ, β)
```

## Stable and Unstable Periodic Orbits of Maps

Chaotic behavior
of low dimensional dynamical systems is affected by the position and the stability properties of the [periodic orbits](http://www.scholarpedia.org/article/Unstable_periodic_orbits) of a dynamical system.

Finding unstable (or stable) periodic orbits of a discrete mapping analytically
rapidly becomes impossible for higher orders of fixed points.
Fortunately there are numerical algorithms that allow their detection.

### Schmelcher & Diakonos

First of the algorithms was proposed by Schmelcher & Diakonos[^Schmelcher1997].
Notice that even though the algorithm can find stable fixed points, it is mainly 
aimed at *unstable* ones.

The functions `periodicorbits` and `lambdamatrix` implement the algorithm:
```@docs
periodicorbits
lambdamatrix
lambdaperms
```

#### Standard Map example
For example, let's find the fixed points of the Standard map of order 2, 3, 4, 5, 6
and 8. We will use all permutations for the `signs` but only one for the `inds`.
We will also only use one `λ` value, and a 21×21 density of initial conditions.

First, initialize everything
```@example MAIN
using ChaosTools

function standardmap_rule(x, k, n)
    theta = x[1]; p = x[2]
    p += k[1]*sin(theta)
    theta += p
    return SVector(mod2pi(theta), mod2pi(p))
end

standardmap = DeterministicIteratedMap(standardmap_rule, rand(2), [1.0])
xs = range(0, stop = 2π, length = 11); ys = copy(xs)
ics = [SVector{2}(x,y) for x in xs for y in ys]

# All permutations of [±1, ±1]:
singss = lambdaperms(2)[2] # second entry are the signs

# I know from personal research I only need this `inds`:
indss = [[1,2]] # <- must be container of vectors!

λs = 0.005 # <- only this allowed to not be vector (could also be vector)

orders = [2, 3, 4, 5, 6, 8]
ALLFP = Dataset{2, Float64}[]

standardmap
```
Then, do the necessary computations for all orders

```@example MAIN
for o in orders
    FP = periodicorbits(standardmap, o, ics, λs, indss, singss)
    push!(ALLFP, FP)
end
```

Plot the phase space of the standard map
```@example MAIN
using CairoMakie
iters = 1000
dataset = trajectory(standardmap, iters)[1]
for x in xs
    for y in ys
        append!(dataset, trajectory(standardmap, iters, [x, y])[1])
    end
end

fig = Figure()
ax = Axis(fig[1,1]; xlabel = L"\theta", ylabel = L"p",
    limits = ((xs[1],xs[end]), (xs[1],xs[end]))
)
scatter!(ax, dataset[:, 1], dataset[:, 2]; markersize = 1, color = "black")
fig
```

and finally, plot the fixed points
```@example MAIN
markers = [:diamond, :utriangle, :rect, :pentagon, :hexagon, :circle]

for i in 1:6
    FP = ALLFP[i]
    o = orders[i]
    scatter!(ax, columns(FP)...; marker=markers[i], color = Cycled(i),
        markersize = 30 - 2i, strokecolor = "grey", strokewidth = 1, label = "order $o"
    )
end
axislegend(ax)
fig
```

Okay, this output is great, and we can tell that it is correct because:

1. Fixed points of order $n$ are also fixed points of order $2n, 3n, 4n, ...$
2. Besides fixed points of previous orders, *original* fixed points of
   order $n$ come in (possible multiples of) $2n$-sized pairs (see e.g. order 5).
   This is a direct consequence of the Poincaré–Birkhoff theorem.

### Davidchack & Lai

An extension of the previous algorithm was proposed by Davidchack & Lai[^Davidchack1999].
It works similarly, but it uses smarter seeding and improved transformation rule.

The functions `davidchacklai` implements the algorithm:
```@docs
davidchacklai
```

#### Logistic Map example

The idea of periodic orbits can be illustrated easily on 1D maps. Fixed points of a 
map $f$ is a point $x$ such that $f(x) = x$. Hence finding roots of $f(x)-x = 0$ 
yields the fixed points. If we graph $f$ and an identity line $y=x$, their intersection 
is exactly the fixed points. Periodic point is a point $x$ such that $f^{n}(x)=x$, where 
$f^{n}$ is $n$-th composition of $f$. Naturally, solving $f^{n}(x)-x=0$ would yield the 
periodic points. However, this is impossible analytically. Let's see how `davidchacklai` 
deals with it:

First let's start with finding first $9$ periodic orbits of logistic map for parameter $3.72$.

```@example MAIN
using ChaosTools
using CairoMakie

logistic_rule(x, p, n) = @inbounds SVector(p[1]*x[1]*(1 - x[1]))
ds = DeterministicIteratedMap(logistic_rule, SVector(0.4), [3.72])
seeds = [SVector(i) for i in LinRange(0.0, 1.0, 10)]
output = davidchacklai(ds, 9, seeds, 6; abstol=1e-6, disttol=1e-12)
```

Now to check the results, let's plot the periodic orbits of period $6$, $7$, $8$ and $9$. 

```@example MAIN
function ydata(ds, order, xdata)
    ydata = typeof(current_state(ds)[1])[]
    for x in xdata
        reinit!(ds, x)
        step!(ds, order)
        push!(ydata, current_state(ds)[1])
    end
    return ydata
end

fig = Figure()
x = LinRange(0.0, 1.0, 1000)
for (order, position)  in zip([6,7,8,9], [(1,1), (1,2), (2,1), (2,2)])
    fpsx = output[order]
    y = ydata(ds, order, [SVector(x0) for x0 in x])
    fpsy = ydata(ds, order, fpsx)
    axis = Axis(fig[position...])
    axis.title = "Order $order"
    lines!(axis, x, x, color=:black, linewidth=0.5)
    lines!(axis, x, y, color = :blue, linewidth=0.7)
    scatter!(axis, [i[1] for i in fpsx], fpsy, color = :red, markersize=5)
end
fig
```
The result is correct because all the intersection between identity and logistic 
map were found.

#### Henon Map example

Let's try to use `davidchacklai` in higher dimension. We will try to detect 
all periodic points of henon map of period `1` to `14`.

```@example MAIN
using ChaosTools
using CairoMakie

function henon(u0=zeros(2); a = 1.4, b = 0.3)
    return DeterministicIteratedMap(henon_rule, u0, [a,b])
end
henon_rule(x, p, n) = SVector{2}(1.0 - p[1]*x[1]^2 + x[2], p[2]*x[1])

ds = henon()
xs = LinRange(-3.0, 3.0, 10)
ys = LinRange(-10.0, 10.0, 10)
seeds = [SVector{2}(x,y) for x in xs for y in ys]
n = 14
m = 6
output = davidchacklai(ds, n, seeds, m; abstol=1e-7, disstol=1e-10)

fig = Figure()
ax = Axis(fig[1,1])
for result in output
    scatter!(ax, [x[1] for x in result], [x[2] for x in result], markersize=8, color=:blue)
end
fig
```

Note that in this case parameter `m` has to be set to at least `6`. Otherwise, the algorithm 
fails to detect orbits of higher periods correctly.

To check if the detected points are indeed periodic, we can do the following test:

```@example MAIN
    orbit14 = output[end]
    c = 0
    for x in orbit14
        set_state!(ds, x)
        step!(ds, 14)
        xn = current_state(ds)
        if ChaosTools.norm(xn - x) > 1e-10
            c += 1
        end
    end
    println("$c non-periodic points found in the 14th order orbit.")
```

The same test can be applied to orbits of lower periods.


[^Davidchack1999]:
    Ruslan L. Davidchack and Ying-Cheng Lai, Phys. Rev. E 60, 6172 – Published 1 November 1999

[^Schmelcher1997]:
    P. Schmelcher & F. K. Diakonos, Phys. Rev. Lett. **78**, pp 4733 (1997)

## Estimating the Period

The function [`estimate_period`](@ref) offers ways for estimating the period (either exact for periodic timeseries, or approximate for near-periodic ones) of a given timeseries.
We offer five methods to estimate periods, some of which work on evenly sampled data only, and others which accept any data.
The figure below summarizes this:
![](https://raw.githubusercontent.com/JuliaDynamics/JuliaDynamics/master/videos/chaos/periodestimationmethods.png?raw=true)

```@docs
estimate_period
yin
```

### Example
Here we will use a modified FitzHugh-Nagumo system that results in periodic behavior, and then try to estimate its period. First, let's see the trajectory:
```@example MAIN
using ChaosTools, CairoMakie

function FHN(u, p, t)
    e, b, g = p
    v, w = u
    dv = min(max(-2 - v, v), 2 - v) - w
    dw = e*(v - g*w + b)
    return SVector(dv, dw)
end

g, e, b  = 0.8, 0.04, 0.0
p0 = [e, b, g]

fhn = CoupledODEs(FHN, SVector(-2, -0.6667), p0)
T, Δt = 1000.0, 0.1
X, t = trajectory(fhn, T; Δt)
v = X[:, 1]

lines(t, v)
```

Examining the figure, one can see that the period of the system is around `91` time units. To estimate it numerically let's use some of the methods:
```@example MAIN
estimate_period(v, :autocorrelation, t)
```
```@example MAIN
estimate_period(v, :periodogram, t)
```
```@example MAIN
estimate_period(v, :zerocrossing, t)
```
```@example MAIN
estimate_period(v, :yin, t; f0_min=0.01)
```