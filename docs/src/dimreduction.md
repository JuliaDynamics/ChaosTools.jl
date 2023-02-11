# Dimensionality reduction

## Broomhead-King Coordinates
```@docs
broomhead_king
```

This alternative/improvement of the traditional delay coordinates can be a very
powerful tool. An example where it shines is noisy data where there is the effect
of superficial dimensions due to noise.

Take the following example where we produce noisy data from a system and then use
Broomhead-King coordinates as an alternative to "vanilla" delay coordinates:

```@example MAIN
using ChaosTools, CairoMakie

function gissinger_rule(u, p, t)
    μ, ν, Γ = p
    du1 = μ*u[1] - u[2]*u[3]
    du2 = -ν*u[2] + u[1]*u[3]
    du3 = Γ - u[3] + u[1]*u[2]
    return SVector{3}(du1, du2, du3)
end

gissinger = CoupledODEs(gissinger_rule, ones(3), [0.12, 0.1, 0.9])
X, t = trajectory(ds, 500.0; Ttr = 10, Δt = 0.05)
x = X[:, 1]

L = length(x)
s = x .+ 0.5rand(L) #add noise

U, S = broomhead_king(s, 20)
summary(U)
```

Now let's simply compare the above result with the one you get from doing a standard delay coordinates embedding
```@example MAIN
using DelayEmbeddings: embed, estimate_delay

fig = Figure()
axs = [Axis3(fig[1, i]) for i in 1:2]
lines!(axs[1], U[:, 1], U[:, 2], U[:, 3])
axs[1].title = "Broomhead-King of s"

R = embed(s, 3, estimate_delay(x, "mi_min"))
lines!(axs[2], columns(R)...)
axs[2].title = "2D embedding of s"
fig
```

we have used the same system as in the [Delay Coordinates Embedding](@ref) example, and picked the optimal
delay time of `τ = 30` (for same `Δt = 0.05`). Regardless, the vanilla delay coordinates is much worse than the Broomhead-King coordinates.

## DyCA - Dynamical Component Analysis
```@docs
dyca
```
