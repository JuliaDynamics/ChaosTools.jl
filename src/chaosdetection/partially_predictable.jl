using LinearAlgebra, Distributions
using Statistics: mean
export predictability

"""
    predictability(ds::DynamicalSystem; kwargs...) -> chaos_type, ν, C

Determine whether `ds` displays strongly chaotic, partially-predictable chaotic
or regular behaviour, using the method by Wernecke et al. described in [1].

Return the type of the behavior, the cross-distance scaling coefficient `ν`
and the correlation coefficient `C`.
Typical values for `ν`, `C` and `chaos_type` are given in Table 2 of [1]:

| `chaos_type` | `ν` | `C` |
|--------------|-----|-----|
|    `:SC`     |  0  |  0  |
|    `:PPC`    |  0  |  1  |
|    `:REG`    |  1  |  1  |


## Keyword Arguments
* `Ttr = 200` : Extra "transient" time to evolve the system before sampling from
   the trajectory. Should be `Int` for discrete systems.
* `T_sample = 1e4` : Time to evolve the system for taking samples. Should be
  `Int` for discrete systems.
* `n_samples = 500` : Number of samples to take for use in calculating statistics.
* `λ_max = lyapunov(ds, 5000)` : Value to use for largest Lyapunov exponent
  for finding the Lyapunov prediction time. If it is less than zero a regular
  result is returned immediatelly.
* `d_tol = 1e-3` : tolerance distance to use for calculating Lyapunov prediction time.
* `T_multiplier = 10` : Multiplier from the Lyapunov prediction time to the evaluation time.
* `T_max = Inf` : Maximum time at which to evaluate trajectory distance. If the internally
   computed evaluation time is larger than `T_max`, stop at `T_max` instead.
* `δ_range = 10.0 .^ (-9:-6)` : Range of initial condition perturbation distances
   to use to determine scaling `ν`.
* `diffeq...` : Keyword arguments propagated into `init` of DifferentialEquations.jl.
  See [`trajectory`](@ref) for examples. Only valid for continuous systems.

## Description
Samples points from a trajectory of the system to be used as initial conditions. Each of
these initial conditions is randomly perturbed by a distance `δ`, and the trajectories for
both the original and perturbed initial conditions are computed to the 'evaluation time'
`T`.

The average (over the samples) distance and cross-correlation coefficient
of the state at time `T` is
computed. This is repeated for a range of `δ` (defined by `δ_range`), and linear
regression is used to determine how the distance and cross-correlation scale with `δ`,
allowing for identification of chaos type.

The evaluation time `T` is calculated as `T = T_multiplier*Tλ`, where the Lyapunov
prediction time `Tλ = log(d_tol/δ)/λ_max`. This may be very large if the `λ_max` is small,
e.g. when the system is regular, so this internally computed time `T` can be overridden by
a smaller `T_max` set by the user.

## Performance Notes
For continuous systems, it is likely that the `maxiters` used by the integrators needs to
be increased, e.g. to 1e9. This is part of the `diffeq` kwargs.
In addition, be aware that this function does a *lot* of internal computations.
It is operating in a different speed than e.g. [`lyapunov`](@ref).

## References

[1] : Wernecke, H., Sándor, B. & Gros, C.
      *How to test for partially predictable chaos*. [Scientific Reports **7**, (2017)](https://www.nature.com/articles/s41598-017-01083-x).
"""
function predictability(ds::DynamicalSystem;
                        Ttr::Real = 200,
                        T_sample::Real = 1e4,
                        n_samples::Integer = 500,
                        λ_max::Real = lyapunov(ds, 5000),
                        d_tol::Real = 1e-3,
                        T_multiplier::Real = 10,
                        T_max::Real = Inf,
                        δ_range::AbstractArray = 10.0 .^ (-9:-6),
                        diffeq...
                        )

    λ_max < 0 && return :REG, 1.0, 1.0
    # Internal Constants
    ν_threshold = 0.5
    C_threshold = 0.5

    # Sample points from a single trajectory of the system
    samples = sample_trajectory(ds, Ttr, T_sample, n_samples; diffeq...)

    # Calculate the mean position and variance of the trajectory. ([1] pg. 5)
    # Using samples 'Monte Carlo' approach instead of direct integration
    μ = mean(samples)
    s² = mean(map(x->(x-μ)⋅(x-μ), samples))

    # Calculate cross-distance scaling and correlation scaling
    distances = Float64[] # Mean distances at time T for different δ
    correlations = Float64[] # Cross-correlation at time T for different δ
    p_integ = parallel_integrator(ds, samples[1:2]; diffeq...)
    for δ in δ_range
        # some kind of warning should be thrown for very large
        # Tλ.
        Tλ = log(d_tol/δ)/λ_max
        T = min(T_multiplier * Tλ, T_max)
        Σd = 0
        Σd² = 0
        for u in samples
            # Sample perturbation from surface of sphere
            n = rand(Normal(), size(u))
            n /= norm(n)
            û = u + δ*n
            # Update integrator with new initial conditions
            reinit!(p_integ, [u, û])
            # Simulate trajectory until T
            while p_integ.t < T
                step!(p_integ)
            end
            # Accumulate distance and square-distance
            d = norm(p_integ.u[1]-p_integ.u[2], 2)
            Σd  += d
            Σd² += d^2
        end
        # Calculate mean distance and square-distance
        d = Σd/length(samples)
        D = Σd²/length(samples)
        # Convert mean square-distance into cross-correlation
        C = 1 - D/2s²
        # Store mean distance and cross-correlation
        push!(distances, d)
        push!(correlations, C)
    end

    # Perform regression to check cross-distance scaling
    ν = slope(log.(δ_range), log.(distances))
    C = mean(correlations)

    # Determine chaotic nature of the system
    if ν > ν_threshold && C > C_threshold
        chaos_type = :REG
    elseif ν <= ν_threshold && C > C_threshold
        chaos_type = :PPC
    elseif ν <= ν_threshold && C ≤ C_threshold
        chaos_type = :SC
    else
        # Covers the case when ν > ν_threshold but C <= C_threshold
        chaos_type = :INDETERMINATE
    end

    return chaos_type, ν, C
end


function sample_trajectory(ds::ContinuousDynamicalSystem,
                           Ttr::Real, T_sample::Real,
                           n_samples::Real;
                           diffeq...)
    # Samples *approximately* `n_samples` points.
    β = T_sample/n_samples
    D_sample = Exponential(β)
    sample_trajectory(ds, Ttr, T_sample, D_sample; diffeq...)
end

function sample_trajectory(ds::DiscreteDynamicalSystem,
                           Ttr::Real, T_sample::Real,
                           n_samples::Real;
                           diffeq...)
    @assert n_samples < T_sample "discrete systems must satisfy n_samples < T_sample"
    # Samples *approximately* `n_samples` points.
    p = n_samples/T_sample
    D_sample = Geometric(p)
    sample_trajectory(ds, Ttr, T_sample, D_sample; diffeq...)
end

function sample_trajectory(ds::DynamicalSystem,
                           Ttr::Real, T_sample::Real,
                           D_sample::UnivariateDistribution;
                           diffeq...)
    # Simulate initial transient
    integ = integrator(ds; diffeq...)
    while integ.t < Ttr
        step!(integ)
    end

    # Time to the next sample is sampled from the distribution D_sample
    # e.g. Continuous systems: D_sample is Exponential distribution
    samples = typeof(integ.u)[]
    while integ.t < Ttr + T_sample
        step!(integ, rand(D_sample))
        push!(samples, integ.u)
    end
    samples
end
