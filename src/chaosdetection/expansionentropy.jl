export expansionentropy
using LinearAlgebra
using Statistics

"""
    expansionentropy(ds::DynamicalSystem, sampler, isinside; kwargs...)

Calculate the expansion entropy[^Hunt2015] of `ds`, in the restraining region ``S``
by estimating the slope (via linear regression)
of the curve ``\\log E_{t0+T, t0}(f, S)`` versus ``T`` (using [`linear_region`](@ref)).
This is an approximation of the expansion entropy ``H_0``, according to[^Hunt2015].
Return ``T``,  ``\\log E`` and the calculated slope.

`sampler` is a 0-argument function that generates a random initial conditions of `ds`
and `isinside` is a 1-argument function that given a state it returns true if
the state is inside the restraining region.
Typically `sampler, isinside` are the output of [`statespace_sampler`](@ref).

## Keyword arguments

* `N = 1000`: Number of samples taken at each batch (same as ``N`` of [^Hunt2015]).
* `steps = 40`: The maximal steps for which the system will be run.
* `batches = 100`: Number of batches to run the calculation, see below.
* `Δt = 1`: Time evolution step size.
* `J = nothing`: Jacobian function given to [`TangentDynamicalSystem`](@ref).

## Description

`N` samples are initialized and propagated forwards in time (along with their tangent space).
At every time ``t`` in `[t0+Δt, t0+2Δt, ..., t0+steps*Δt]` we calculate ``H``:
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
defined by the boolean function `insinside`.
This process is done by the `ChaosTools.expansionentropy_sample` function.

Then, this is repeated for `batches` amount of times, as recommended in[^Hunt2015].
From all these batches, the mean and std of ``H`` is computed at every time point.
This is done by the [`expansionentropy_batch`](@ref) function.
When plotted versus ``t``, these create the curves and error bars of e.g. Figs 2, 3 of [1].

This function `expansionentropy` simply returns the slope of the biggest linear region
of the curve ``H`` versus ``t``, which approximates the expansion entropy ``H_0``.
It is therefore *recommended* to use [`expansionentropy_batch`](@ref) directly and
evaluate the result yourself, as this step is known to be inaccurate for
non-chaotic systems (where ``H`` fluctuates strongly around 0).

[^Hunt2015]: Hunt & Ott, ‘Defining Chaos’, [Chaos 25.9 (2015)](https://doi.org/10/gdtkcf)
"""
function expansionentropy(ds::CoreAnalyticSystem, sampler, restraining; kwargs...)
    times, means, stds = expansionentropy_batch(ds, sampler, restraining; kwargs...)
    if any(isnan, stds)
        i = findfirst(isnan, stds)
        @warn "All (or all except one) samples have escaped the given region at time = $(times[i])."
        times = times[1:i-1]
        means = means[1:i-1]
    end
    slope = ChaosTools.linreg(times, means)[2]
    return times, means, slope
end

# TODO: Restore a `Ttr` keyword argument. Will need to also keep `ds`
# being used in `expansionentropy_sample` so that its own state is stepped
# without stepping the tangent space as well (for efficiency)

#####################################################################################
# Actual implementation of expansion entropy
#####################################################################################
"""
    expansionentropy_batch(ds, sampler, restraining; kwargs...)

Run [`expansionentropy_sample`](@ref) `batch` times, and return
`times, mean(H), std(H)` for all resulting `H`, see [`expansionentropy`](@ref).

Accepts the same arguments as `expansionentropy`.
"""
function expansionentropy_batch(ds, sampler, restraining; Δt = 1, J = nothing, batches=100, steps=40, kwargs...)
    means = fill(NaN, steps)
    stds = fill(NaN, steps)
    # eesamples[k, t] = The k-th sample of expansion entropy from t0 to (t0 + t)
    eesamples = zeros(batches, steps)
    times = initial_time(ds) .+ Δt .* (1:steps)
    tands = TangentDynamicalSystem(ds; J)

    # Collect all the samples
    for k in 1:batches
        eesamples[k, :] = expansionentropy_sample(
            tands, sampler, restraining; steps, Δt, kwargs...
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

"""
    expansionentropy_sample(tands::TangentDynamicalSystem, sampler, isinside; kwargs...)

Return `times, H` for one sample of `ds` (see [`expansionentropy`](@ref)).
Accepts the same argumets as `expansionentropy`, besides `batches`.
"""
function expansionentropy_sample(
        tands::TangentDynamicalSystem, sampler, restraining;
        N=1000, steps=40, Δt=1
    )

    M = zeros(steps)
    reinit!(tands)
    # M[t] will be Σᵢ G(Dfₜ₀,ₜ₀₊ₜ(xᵢ))
    # The summation is over all sampled xᵢ that stay inside S during [t0, t0 + t].

    for i ∈ 1:N
        u = sampler() # New sample point.
        reinit!(tands, u)
        for i ∈ 1:steps # Evolve the sample point for the duration [t0, t0+steps*Δt]
            step!(tands, Δt, true)
            u = current_state(tands)
            !restraining(u) && break # Stop the integration if the orbit leaves the region.
            Df = current_deviations(tands)
            M[i] += maximalexpansion(Df)
        end
    end
    return log.(M./N)
end

"""
    maximalexpansion(M)

Calculate the maximal expansion rate of M,
i.e. the product of all singular values of M that are greater than 1. In the
notation of [1], it is the function ``G``.
"""
maximalexpansion(M) = prod(filter(x -> x > 1.0, svdvals(M)))
