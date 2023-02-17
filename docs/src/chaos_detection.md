# Detecting & Categorizing Chaos
Being able to detect and distinguish chaotic from regular behavior is crucial in the study of dynamical systems.
Most of the time a positive maximum [`lyapunov`](@ref) exponent
and a bounded system indicate chaos.

However, the convergence of the Lyapunov exponent can be slow, or even misleading, as the types of chaotic behavior vary with respect to their predictability.
There are some alternatives, some more efficient and some more accurate in characterizing chaotic and regular motion.

## Generalized Alignment Index
"GALI" for sort, is a method that relies on the fact that initially orthogonal deviation vectors tend to align towards the direction of the maximum Lyapunov exponent for chaotic motion.
It is one of the most recent and cheapest methods for distinguishing chaotic and regular behavior, introduced first in 2007 by Skokos, Bountis & Antonopoulos.
```@docs
gali
```

### GALI example

As an example let's use the Henon-Heiles system
```@example MAIN
using ChaosTools, CairoMakie
using OrdinaryDiffEq: Vern9

function henonheiles_rule(u, p, t)
    SVector(u[3], u[4],
        -u[1] - 2u[1]*u[2],
        -u[2] - (u[1]^2 - u[2]^2),
    )
end
function henonheiles_jacob(u, p, t)
    SMatrix{4,4}(0, 0, -1 - 2u[2], -2u[1], 0, 0,
     -2u[1], -1 + 2u[2], 1, 0, 0, 0, 0, 1, 0, 0)
end

u0=[0, -0.25, 0.42081, 0]
Δt = 1.0
diffeq = (abstol=1e-9, retol=1e-9, alg = Vern9(), maxiters = typemax(Int))
sp = [0, .295456, .407308431, 0] # stable periodic orbit: 1D torus
qp = [0, .483000, .278980390, 0] # quasiperiodic orbit: 2D torus
ch = [0, -0.25, 0.42081, 0]      # chaotic orbit
ds = CoupledODEs(henonheiles_rule, sp)
```

Let's see what happens with a quasi-periodic orbit:

```@example MAIN
tr = trajectory(ds, 10000.0, qp; Δt)[1]
fig, ax = scatter(tr[:,1], tr[:,3]; label="qp", markersize=2)
axislegend(ax)

ax = Axis(fig[1,2]; yscale = log)
for k in [2,3,4]
    g, t = gali(ds, 10000.0, k; u0 = qp, Δt)
    logt = log.(t)
    lines!(ax, logt, g; label="GALI_$(k)")
    if k == 2
        lines!(ax, logt, 1 ./ t.^(2k-4); label="slope -$(2k-4)")
    else
        lines!(ax, logt, 100 ./ t.^(2k-4); label="slope -$(2k-4)")
    end
end
ylims!(ax, 1e-12, 2)
fig
```

And here is GALI of a continuous system with a chaotic orbit
```@example MAIN
tr = trajectory(ds, 10000.0, ch; Δt)[1]
fig, ax = scatter(tr[:,1], tr[:,3]; label="ch", markersize=2, color = (Main.COLORS[1], 0.5))
axislegend(ax)

ax = Axis(fig[1,2]; yscale = log)
ls = lyapunovspectrum(ds, 5000; Δt, u0 = ch)
for k in [2,3,4]
    ex = sum(ls[1] - ls[j] for j in 2:k)
    g, t = gali(ds, 1000, k; u0 = ch, Δt)
    lines!(t, exp.(-ex.*t); label="exp. k=$k")
    lines!(t, g; label="GALI_$(k)")
end
ylims!(ax, 1e-16, 1)
fig
```


### Using GALI
No-one in their right mind would try to fit power-laws in order to distinguish between chaotic and regular behavior, like the above examples. These were just proofs that the method works as expected.

The most common usage of $\text{GALI}_k$ is to define a (sufficiently) small
amount of time and a (sufficiently) small threshold and see whether $\text{GALI}_k$
stays below it, for a (sufficiently) big $k$.

For example, we utilize parallel integration of `TangentDynamicalSystem` to compute $GALI$ for many initial conditions and produce a color-coded map of regular and chaotic orbits of the standard map.

The following is an example of advanced usage (see [Advanced documentation](@ref)):
```@example MAIN
using ChaosTools, CairoMakie
# Initialize `TangentDynamicalSystem`
@inbounds function standardmap_rule(x, par, n)
    theta = x[1]; p = x[2]
    p += par[1]*sin(theta)
    theta += p
    return mod2pi.(SVector(theta, p))
end
@inbounds standardmap_jacob(x, p, n) = SMatrix{2,2}(
    1 + p[1]*cos(x[1]), p[1]*cos(x[1]), 1, 1
)
ds = DeterministicIteratedMap(standardmap_rule, ones(2), [1.0])
tands = TangentDynamicalSystem(ds; J = standardmap_jacob)
# Collect initial conditions
dens = 101
θs = ps = range(0, stop = 2π, length = dens)
ics = vec(SVector{2, Float64}.(Iterators.product(θs, ps)))
# Initialize as many systems as threads
systems = [deepcopy(tands) for _ in 1:Threads.nthreads()-1]
pushfirst!(systems, tands)
# Perform threaded loop
regularity = zeros(size(ics))
Threads.@threads for i in eachindex(ics)
    u0 = ics[i]
    system = systems[Threads.threadid()]
    reinit!(system, u0)
    regularity[i] = gali(system, 500)[2][end]
end
# Visualize
fig, ax, sc = scatter(ics; color = regularity)
Colorbar(fig[1,2], sc; label = "regularity")
fig
```

## Predictability of a chaotic system

Even if a system is "formally" chaotic, it can still be in phases where it is partially
predictable, because the correlation coefficient between nearby trajectories vanishes
very slowly with time.
[Wernecke, Sándor & Gros](https://www.nature.com/articles/s41598-017-01083-x) have
developed an algorithm that allows one to classify a dynamical system to one of three
categories: strongly chaotic, partially predictable chaos or regular
(called *laminar* in their paper).

We have implemented their algorithm in the function [`predictability`](@ref).
Note that we set up the implementation to always return regular behavior for negative
Lyapunov exponent. You may want to override this for research purposes.

```@docs
predictability
```

### Example Hénon Map
We will create something similar to figure 2 of the paper, but for the Hénon map.

```@example MAIN
fig = Figure()
ax = Axis(fig[1,1]; xlabel = L"a", ylabel = L"x")
henon_rule(x, p, n) = SVector{2}(1.0 - p[1]*x[1]^2 + x[2], p[2]*x[1])
he = DeterministicIteratedMap(henon_rule, zeros(2), [1.4, 0.3])
as = 0.8:0.01:1.225
od = orbitdiagram(he, 1, 1, as; n = 2000, Ttr = 2000)
colors = Dict(:REG => "blue", :PPC => "green", :SC => "red")
for (i, a) in enumerate(as)
    set_parameter!(he, 1, a)
    chaos_type, ν, C = predictability(he; T_max = 400000, Ttr = 2000)
    scatter!(ax, a .* ones(length(od[i])), od[i];
    color = (colors[chaos_type], 0.05), markersize = 2)
end
ax.title = "predictability of Hénon map"
fig
```

## The 0-1 test for chaos

The methods mentioned in this page so far require a `DynamicalSystem` instance.
But of course this is not always the case. The so-called "0 to 1" test for chaos, by
Gottwald & Melbourne, takes as an input a timeseries and outputs a boolean `true` if
the timeseries is chaotic or `false` if it is not.

Notice that the method does have a lot of caveats, so you should read the review paper
before using. Also, it doesn't work for noisy data.

```@docs
testchaos01
```

## Expansion entropy

The expansion entropy is a quantity that is suggested by B. Hunt and E. Ott as a measure
that can define chaos (so far no widely accepted definition of chaos exists).
Positive expansion entropy means chaos.

```@docs
expansionentropy
```
