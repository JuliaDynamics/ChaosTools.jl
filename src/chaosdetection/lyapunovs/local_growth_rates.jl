"""
    local_growth_rates(ds, points::Dataset; S=100, Δt=5, kwargs...) → λlocal
Compute the exponential local growth rate(s) of perturbations of the dynamical system
`ds` for initial conditions given in `points`. For each initial condition `u ∈ points`,
`S` total perturbations are created and evolved for time `Δt`. The exponential local growth
rate is defined simply by `log(g/g0)/Δt` with `g0` the initial pertrubation size
and `g` the size after `Δt`. Thus, `λlocal` is a matrix of size `(length(points), S)`.

This function is a modification of [`lyapunov`](@ref). It uses the full nonlinear dynamics
to evolve the perturbations, but does not do any re-scaling, thus allowing
probing state and time dependence of perturbation growth. The actual growth
is given by `exp(λlocal * Δt)`.

The output of this function is sometimes referred as "Nonlinear Local Lyapunov Exponent".

## Keyword Arguments
* `perturbation`: If given, it should be a function `perturbation(ds, u, j)` that
  outputs a pertrubation vector (preferrably `SVector`) given the system, current initial
  condition `u` and the counter `j ∈ 1:S`. If not given, a random perturbation is
  generated with norm given by the keyword `e = 1e-6`.
* `diffeq` is a `NamedTuple` (or `Dict`) of keyword arguments propagated into
  `init` of DifferentialEquations.jl.
  See [`trajectory`](@ref) for examples. Only valid for continuous systems.
"""
function local_growth_rates(ds::DynamicalSystem, points;
        S = 100, Δt = 5, e = 1e-6,
        perturbation = (ds, u, j) -> _random_Q0(ds, u, j, e),
        diffeq = NamedTuple(), kwargs...
    )

    if !isempty(kwargs)
        @warn DIFFEQ_DEP_WARN
        diffeq = NamedTuple(kwargs)
    end

    λlocal = zeros(length(points), S)
    Q0 = perturbation(ds, points[1], 1)
    states = [points[1], points[1] .+ Q0]
    pinteg = parallel_integrator(ds, states; diffeq)

    for (i, u) in enumerate(points)
        for j in 1:S
            Q0 = perturbation(ds, u, j)
            states[1] = u
            states[2] = states[1] .+ Q0
            g0 = norm(Q0)
            reinit!(pinteg, states)
            step!(pinteg, Δt, true)
            g = norm(get_state(pinteg, 2) .- get_state(pinteg, 1))
            λlocal[i, j] = log(g/g0) / (pinteg.t - pinteg.t0)
        end
    end
    return λlocal
end

function _random_Q0(ds, u, j, e)
    D, T = dimension(ds), eltype(ds)
    Q0 = randn(Random.GLOBAL_RNG, SVector{D, T})
    Q0 = e * Q0 / norm(Q0)
end
