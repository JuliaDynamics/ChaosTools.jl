#=
This file contains main functions for calculating the expansion entropy
of dynamical systems.
Expansion entropy is defined in (Hunt and Ott, 2015) [1] as a quantitative measure
of chaos. For details, read the docstrings of the functions below.

[1] : B. & E. Ott, ‘Defining Chaos’, [Chaos 25.9 (2015)](https://doi.org/10/gdtkcf)
=#
export boxregion, expansionentropy, expansionentropy_batch, expansionentropy_sample
using LinearAlgebra
using Statistics

"""
    expansionentropy(ds::DynamicalSystem, sampler, restraining; kwargs...)

Calculate the expansion entropy[^Hunt2015] of `ds`, in the restraining region ``S`` defined by
`restraining`, by estimating the slope of the biggest linear region
of the curve ``\\log E_{t0+T, t0}(f, S)`` versus ``T`` (using [`linear_region`](@ref)).
This is an approximation of the expansion entropy ``H_0``, according to[^Hunt2015].

`sampler` is a 0-argument function that generates a random initial condition (a sample)
of `ds`. `restraining` is a 1-argument function `restraining(u)` that given the state
`u` it returns `true` if the state is inside the restraining region ``S``.

Use [`boxregion`](@ref) for an easy way to define `sampler` and `restraining` on a
multidimension box.

## Keyword Arguments
* `N = 1000` : Number of samples taken at each batch (same as ``N`` of [1]).
* `steps = 40` : The maximal steps for which the system will be run.
* `Ttr = 0` : Transient time to evolve each initial condition before starting to comute ``E``.
  This is `t0` of [1] and of the following notation.
* `batches = 100` : Number of batches to run the calculation, see below.
* `Δt = 1` : Integration step size.
* `diffeq` is a `NamedTuple` (or `Dict`) of keyword arguments propagated into
  `init` of DifferentialEquations.jl.
  See [`trajectory`](@ref) for examples. Only valid for continuous systems.

## Description
`N` samples are initialized and propagated forwards in time (along with their tangent space).
At every time ``t`` in `[t0+Δt, t0+2dt, ... t0+steps*Δt]` we calculate ``H``:
```math
H[t] = \\log E_{t0+T, t0}(f, S),
```
with
```math
E_{t0+T, t0}(f, S) = \\frac 1 N \\sum_{i'} G(Df_{t0+t, t0}(x_i))
```
(using same notation as [^Hunt2015]). In principle ``E`` is the average largest possible
growth ratio within the restraining region (sampled by the initial conditions).
The summation is only over ``x_i`` that stay inside the region ``S``
defined by the boolean function `restraining`.
This process is done by the [`expansionentropy_sample`](@ref) function.

Then, this is repeated for `batches` amount of times, as recommended in[^Hunt2015].
From all these batches, the mean and std of ``H`` is computed at every time point.
This is done by the [`expansionentropy_batch`](@ref) function.
When plotted versus ``t``, these create the curves and error bars of e.g. Figs 2, 3 of [1].

This function `expansionentropy` simply returns the slope of the biggest linear region
of the curve ``H`` versus ``t``, which approximates the expansion entropy ``H_0``.
It is therefore *recommended* to use [`expansionentropy_batch`](@ref) directly and
evaluate the result yourself, as this step is known to be inaccurate for
non-chaotic systems (where ``H`` fluctuates strongly around 0).

[^Hunt2015]: B. Hunt & E. Ott, ‘Defining Chaos’, [Chaos 25.9 (2015)](https://doi.org/10/gdtkcf)
"""
function expansionentropy(system, sampler, restraining; kwargs...)
    times, means, stds = expansionentropy_batch(system, sampler, restraining; kwargs...)
    if any(isnan, stds)
        i = findfirst(isnan, stds)
        @warn "All (or all except one) samples have escaped the given region at time = $(times[i])."
        times = times[1:i-1]
        means = means[1:i-1]
    end
    _, slope = linear_region(times, means)
    return slope
end

"""
    boxregion(as, bs) -> sampler, restraining

Define a box in ``\\mathbb{R}^d`` with edges the `as` and `bs` and then
return two functions: `sampler`, which generates a random initial condition in that box
and `restraining` that returns `true` if a given state is in the box.
"""
function boxregion(as, bs)
    @assert length(as) == length(bs) > 0
    gen() = [rand()*(bs[i]-as[i]) + as[i] for i in 1:length(as)]
    restraining(x) = all(as .< x .< bs)
    return gen, restraining
end

# Specialized 1-d version
function boxregion(a::Real, b::Real)
    a, b = extrema((a, b))
    gen() = rand()*(b-a) + a
    restraining = x -> a < x < b
    return gen, restraining
end

#####################################################################################
# Actual implementation of expansion entropy
#####################################################################################
"""
    maximalexpansion(M)

Calculates the maximal expansion rate of M,
i.e. the product of all singular values of M that are greater than 1. In the
notation of [1], it is the function ``G``.
"""
maximalexpansion(M) = prod(filter(x -> x > 1.0, svdvals(M)))

"""
    expansionentropy_sample(ds, sampler, restraining; kwargs...)

Return `times, H` for one sample of `ds` (see [`expansionentropy`](@ref)).
Accepts the same argumets as `expansionentropy`, besides `batches`.
"""
function expansionentropy_sample(system::DynamicalSystem, sampler, restraining;
    N=1000, steps=40, Δt=1, Ttr=0, diffeq = NamedTuple(), kwargs...)

    if !isempty(kwargs)
        @warn DIFFEQ_DEP_WARN
        diffeq = NamedTuple(kwargs)
    end

    D = dimension(system)
    M = zeros(steps)
    # M[t] will be Σᵢ G(Dfₜ₀,ₜ₀₊ₜ(xᵢ))
    # The summation is over all sampled xᵢ that stay inside S during [t0, t0 + t].

    times = @. (system.t0+Ttr)+Δt*(1:steps)
    t_identity = SMatrix{D, D, Float64}(I)
    t_integ = tangent_integrator(system; diffeq)
    Ttr > 0 && (u_integ = integrator(system))

    for i ∈ 1:N
        u = sampler() # New sample point.
        if Ttr > 0 # Evolve through transient time.
            reinit!(u_integ, u)
            step!(u_integ, Ttr, true) # stepping must be exact here for correct time vector
            u = u_integ.u
        end
        reinit!(t_integ, u, t_identity; t0 = t_integ.t0 + Ttr)
        for i ∈ 1:steps # Evolve the sample point for the duration [t0, t0+steps*Δt]
            step!(t_integ, Δt, true)
            u = get_state(t_integ)
            !restraining(u) && break # Stop the integration if the orbit leaves the region.
            Df = get_deviations(t_integ)
            M[i] += maximalexpansion(Df)
        end
    end
    return times, log.(M./N)
end

# This version only deals with 1-dimensional discrete dynamical systems, but does
# it really fast, by avoiding `tangent_integrator` and `maximalexpansion`.
function expansionentropy_sample(system::DiscreteDynamicalSystem{IIP, S, 1}, sampler, restraining;
    N=1000, steps=40, Δt=1, Ttr=0, kwargs...) where {IIP, S}
    f = system.f
    p = system.p
    jacob = system.jacobian
    t0 = system.t0
    times = @. (t0+Ttr)+Δt*(1:steps)

    Δt = max(Int(floor(Δt)), 1)
    Ttr = max(Int(floor(Ttr)), 0)
    M = zeros(steps)

    for i ∈ 1:N
        x = sampler()
        t = t0

        # Evolve for a transient time.
        for _ ∈ 1:Ttr
            x = f(x, p, t)
            t += 1
        end

        Df = 1.0
        for step ∈ 1:steps # Evolve point x for `steps` steps.
            for _ ∈ 1:Δt # Evolve Δt steps at a time.
                Df = jacob(x, p, t) * Df
                x = f(x, p, t)
                t += 1
            end

            !restraining(x) && break
            M[step] += max(1.0, abs(Df))
        end
    end
    return times, log.(M./N)
end

"""
    expansionentropy_batch(ds, sampler, restraining; kwargs...)

Run [`expansionentropy_sample`](@ref) `batch` times, and return
`times, mean(H), std(H)` for all resulting `H`, see [`expansionentropy`](@ref).

Accepts the same arguments as `expansionentropy`.
"""
function expansionentropy_batch(system, sampler, restraining; batches=100, steps=40, kwargs...)
    # TODO: It is a mistake that `expansionentropy_batch` doen't create an integrator that
    # us just reinited inside `expansionentropy_sample`...
    means = fill(NaN, steps)
    stds = fill(NaN, steps)
    eesamples = zeros(batches, steps)
    # eesamples[k, t] = The k-th sample of expansion entropy from t0 to (t0 + t)

    times = undef
    # Collect all the samples
    for k in 1:batches
        times, eesamples[k, :] = expansionentropy_sample(
            system, sampler, restraining; steps, kwargs...
        )
    end

    # Calculate the mean and standard deviations
    for t in 1:steps
        entropysamples = filter(isfinite, @view eesamples[:, t])
        # remove -Inf entries, which indicate all samples failed to stay inside the given region.
        if length(entropysamples) ≤ 1
        @warn "All (or all except one) samples have escaped the given region. "*
        "Consider increasing sample or batch number. Terminating at step = $(t)."
            break
        end
        means[t] = mean(entropysamples)
        stds[t] = std(entropysamples)
    end
    return times, means, stds
end
