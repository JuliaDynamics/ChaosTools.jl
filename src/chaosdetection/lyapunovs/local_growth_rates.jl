export local_growth_rates

"""
    local_growth_rates(ds::DynamicalSystem, points::StateSpaceSet; kwargs...) → λlocal

Compute the local exponential growth rate(s) of perturbations of the dynamical system
`ds` for initial conditions given in `points`. For each initial condition `u ∈ points`,
`S` total perturbations are created and evolved exactly for time `Δt`. The exponential local
growth rate is defined simply by `log(g/g0)/Δt` with `g0` the initial perturbation size
and `g` the size after `Δt`. Thus, `λlocal` is a matrix of size `(length(points), S)`.

This function is a modification of [`lyapunov`](@ref). It uses the full nonlinear dynamics
and a [`ParallelDynamicalSystem`](@ref) to evolve the perturbations, but does not do any
re-scaling, thus allowing probing state and time dependence of perturbation growth.
The actual growth is given by `exp(λlocal * Δt)`.

The output of this function is sometimes called "Nonlinear Local Lyapunov Exponent".

## Keyword arguments

* `S = 100`
* `Δt = 5`
* `perturbation`: If given, it should be a function `perturbation(ds, u, j)` that
  outputs a perturbation vector (preferrably `SVector`) given the system, current initial
  condition `u` and the counter `j ∈ 1:S`. If not given, a random perturbation is
  generated with norm given by the keyword `e = 1e-6`.
"""
function local_growth_rates(ds::DynamicalSystem, points;
        S = 100, Δt = 5, e = 1e-6,
        perturbation = (ds, u, j) -> _random_Q0(ds, u, j, e),
    )

    Q0 = perturbation(ds, points[1], 1)
    states = [points[1], points[1] .+ Q0]
    pds = ParallelDynamicalSystem(ds, states)
    # Function barrier
    return local_growth_rates(pds, states, points, ds, S, Δt, perturbation)
end

function local_growth_rates(pds, states, points, ds, S, Δt, perturbation)
    λlocal = zeros(length(points), S)
    for (i, u) in enumerate(points)
        for j in 1:S
            Q0 = perturbation(ds, u, j)
            states[1] = u
            states[2] = states[1] .+ Q0
            g0 = norm(Q0)
            reinit!(pds, states)
            step!(pds, Δt, true)
            g = norm(current_state(pds, 2) .- current_state(pds, 1))
            tspan = (current_time(pds) - initial_time(pds))
            λlocal[i, j] = log(g/g0)/tspan
        end
    end
    return λlocal
end

function _random_Q0(ds, u, j, e)
    D, T = dimension(ds), eltype(ds)
    Q0 = randn(SVector{D, T})
    Q0 = e * Q0 / norm(Q0)
end
