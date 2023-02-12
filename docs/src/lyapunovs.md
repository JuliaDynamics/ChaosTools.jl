# Lyapunov Exponents
Lyapunov exponents measure exponential rates of separation of nearby trajectories in the flow of a dynamical system. The concept of these exponents is best explained in Chapter 3 of [Nonlinear Dynamics](https://link.springer.com/book/10.1007/978-3-030-91032-7), Datseris & Parlitz, Springer 2022. The explanations of the chapter directly utilize the code of the functions in this page.

## Lyapunov Spectrum

The function `lyapunovspectrum` calculates the entire spectrum of the Lyapunov
exponents of a system:
```@docs
lyapunovspectrum
```

### Example
For example, the Lyapunov spectrum of the [folded towel map](http://www.scholarpedia.org/article/Hyperchaos) is calculated as:

```@example MAIN
using ChaosTools
function towel_rule(x, p, n)
    @inbounds x1, x2, x3 = x[1], x[2], x[3]
    SVector( 3.8*x1*(1-x1) - 0.05*(x2+0.35)*(1-2*x3),
    0.1*( (x2+0.35)*(1-2*x3) - 1 )*(1 - 1.9*x1),
    3.78*x3*(1-x3)+0.2*x2 )
end
function towel_jacob(x, p, n)
    row1 = SVector(3.8*(1 - 2x[1]), -0.05*(1-2x[3]), 0.1*(x[2] + 0.35))
    row2 = SVector(-0.19((x[2] + 0.35)*(1-2x[3]) - 1),  0.1*(1-2x[3])*(1-1.9x[1]),  -0.2*(x[2] + 0.35)*(1-1.9x[1]))
    row3 = SVector(0.0,  0.2,  3.78(1-2x[3]))
    return vcat(row1', row2', row3')
end

ds = DeterministicIteratedMap(towel_rule, [0.085, -0.121, 0.075], nothing)
tands = TangentDynamicalSystem(ds; J = towel_jacob)

λλ = lyapunovspectrum(tands, 10000)
```

`lyapunovspectrum` also works for continuous time systems and will auto-generate a Jacobian function if one is not give. For example,

```@example MAIN
function lorenz_rule(u, p, t)
    σ = p[1]; ρ = p[2]; β = p[3]
    du1 = σ*(u[2]-u[1])
    du2 = u[1]*(ρ-u[3]) - u[2]
    du3 = u[1]*u[2] - β*u[3]
    return SVector{3}(du1, du2, du3)
end

lor = CoupledODEs(lorenz_rule, fill(10.0, 3), [10, 32, 8/3])
λλ = lyapunovspectrum(lor, 10000; Δt = 0.1)
```

`lyapunovspectrum` is also very fast:
```julia
using BenchmarkTools
ds = DeterministicIteratedMap(towel_rule, [0.085, -0.121, 0.075], nothing)
tands = TangentDynamicalSystem(ds; J = towel_jacob)

@btime lyapunovspectrum($tands, 10000)
```
```
  966.500 μs (10 allocations: 576 bytes) # on my laptop
```

Here is an example of using [`reinit!`](@ref) to efficiently iterate over different parameter values, and parallelize via `Threads`, to compute the exponents over a given parameter range.


```@example MAIN
using ChaosTools, CairoMakie

henon_rule(x, p, n) = SVector{2}(1.0 - p[1]*x[1]^2 + x[2], p[2]*x[1])
henon_jacob(x, p, n) = SMatrix{2,2}(-2*p[1]*x[1], p[2], 1.0, 0.0)
ds = DeterministicIteratedMap(henon_rule, zeros(2), [1.4, 0.3])
tands = TangentDynamicalSystem(ds; J = henon_jacob)

as = 0.8:0.005:1.225;
λs = zeros(length(as), 2)

# Since `DynamicalSystem`s are mutable, we need to copy to parallelize
systems = [deepcopy(tands) for _ in 1:Threads.nthreads()-1]
pushfirst!(systems, tands)

Threads.@threads for i in eachindex(as)
    system = systems[Threads.threadid()]
    set_parameter!(system, 1, as[i])
    λs[i, :] .= lyapunovspectrum(system, 10000; Ttr = 500)
end

fig = Figure()
ax = Axis(fig[1,1]; xlabel = L"a", ylabel = L"\lambda")
for j in 1:2
    lines!(ax, as, λs[:, j])
end
fig
```

## Maximum Lyapunov Exponent

It is possible to get only the maximum Lyapunov exponent simply by giving
`1` as the third argument of [`lyapunovspectrum`](@ref). However, there is a second algorithm that calculates the maximum exponent:

```@docs
lyapunov
```

For example:
```@example MAIN
using ChaosTools
henon_rule(x, p, n) = SVector{2}(1.0 - p[1]*x[1]^2 + x[2], p[2]*x[1])
henon = DeterministicIteratedMap(henon_rule, zeros(2), [1.4, 0.3])
λ = lyapunov(henon, 10000; d0 = 1e-7, d0_upper = 1e-4, Ttr = 100)
```


## Local Growth Rates
```@docs
local_growth_rates
```

Here is a simple example using the Henon map
```@example MAIN
using ChaosTools
using Statistics, CairoMakie

ds = Systems.henon()
points = trajectory(ds, 2000; Ttr = 100)[1]

λlocal = local_growth_rates(ds, points; Δt = 1)

λmeans = mean(λlocal; dims = 2)
λstds = std(λlocal; dims = 2)
x, y = columns(points)
fig, ax, obj = scatter(x, y; color = vec(λmeans))
Colorbar(fig[1,2], obj)
fig
```


## Lyapunov exponent from data

```@docs
lyapunov_from_data
NeighborNumber
WithinRange
```

Let's apply the method to a timeseries from a continuous time system. In this case, one must be a bit more thoughtful when choosing parameters.
The following example helps the users get familiar with the process:
```@example MAIN
using ChaosTools, CairoMakie

function lorenz_rule(u, p, t)
    σ = p[1]; ρ = p[2]; β = p[3]
    du1 = σ*(u[2]-u[1])
    du2 = u[1]*(ρ-u[3]) - u[2]
    du3 = u[1]*u[2] - β*u[3]
    return SVector{3}(du1, du2, du3)
end

ds = CoupledODEs(lorenz_rule, fill(10.0, 3), [10, 32, 8/3])
# create a timeseries of 1 dimension
Δt = 0.05
x = trajectory(ds, 1000.0; Ttr = 10, Δt)[1][:, 1]
```

From prior knowledge of the system, we know we need to use `k` up to about `150`.
However, due to the dense time sampling, we don't have to compute for every `k` in the range `0:150`. Instead, we can use
```@example MAIN
ks = 0:4:150
```
Now we plot some example computations using delay embeddings to "reconstruct" the chaotic attractor
```@example MAIN
using DelayEmbeddings: embed
fig = Figure()
ax = Axis(fig[1,1]; xlabel="k (0.05×t)", ylabel="E - E(0)")
ntype = NeighborNumber(5) #5 nearest neighbors of each state

for d in [4, 8], τ in [7, 15]
    r = embed(x, d, τ)

    # E1 = lyapunov_from_data(r, ks1; ntype)
    # λ1 = ChaosTools.linreg(ks1 .* Δt, E1)[2]
    # plot(ks1,E1.-E1[1], label = "dense, d=$(d), τ=$(τ), λ=$(round(λ1, 3))")

    E2 = lyapunov_from_data(r, ks; ntype)
    λ2 = ChaosTools.linreg(ks .* Δt, E2)[2]
    lines!(ks, E2.-E2[1]; label = "d=$(d), τ=$(τ), λ=$(round(λ2, digits = 3))")
end
axislegend(ax; position = :lt)
ax.title = "Continuous Reconstruction Lyapunov"
fig
```

As you can see, using `τ = 15` is not a great choice! The estimates with
`τ = 7` though are very good (the actual value is around `λ ≈ 0.89...`).
Notice that above a linear regression was done over the whole curves, which doesn't make sense. One should identify a linear scaling region and extract the slope of that one. The function `linear_region` from [FractalDimensions.jl](https://github.com/JuliaDynamics/FractalDimensions.jl) does this!