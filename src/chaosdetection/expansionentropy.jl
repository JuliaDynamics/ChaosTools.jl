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

Calculate the expansion entropy [1] of `ds`, in the restraining region ``S`` defined by
`restraining`, by estimating the slope of the biggest linear region
of the curve ``\\log E_{t0+T, t0}(f, S)`` versus ``T`` (using [`linear_region`](@ref)).
This is an approximation of the expansion entropy ``H_0``, according to [1].

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
* `diffeq...` : Other keywords are propagated to the solvers of DifferentialEquations.jl.

## Description
`N` samples are initialized and propagated forwards in time (along with their tangent space).
At every time ``t`` in `[t0+dt, t0+2dt, ... t0+steps*dt]` we calculate ``H``:
```math
H[t] = \\log E_{t0+T, t0}(f, S),
```
with
```math
E_{t0+T, t0}(f, S) = \\frac 1 N \\sum_{i'} G(Df_{t0+t, t0}(x_i))
```
(using same notation as [1]). In principle ``E`` is the average largest possible
growth ratio within the restraining region (sampled by the initial conditions).
The summation is only over ``x_i`` that stay inside the region ``S``
defined by the boolean function `restraining`.
This process is done by the [`expansionentropy_sample`](@ref) function.

Then, this is repeated for `batches` amount of times, as recommended in [1].
From all these batches, the mean and std of ``H`` is computed at every time point.
This is done by the [`expansionentropy_batch`](@ref) function.
When plotted versus ``t``, these create the curves and error bars of e.g. Figs 2, 3 of [1].

This function `expansionentropy` simply returns the slope of the biggest linear region
of the curve ``H`` versus ``t``, which approximates the expansion entropy ``H_0``.
It is therefore *recommended* to use [`expansionentropy_batch`](@ref) directly and
evaluate the result yourself.

[1] : B. & E. Ott, ‘Defining Chaos’, [Chaos 25.9 (2015)](https://doi.org/10/gdtkcf)
"""
function expansionentropy(system, sampler, restraining; kwargs...)
    times, meanlist, stdlist = expansionentropy_batch(system, sampler, restraining; kwargs...)
    if any(isnan, stdlist)
        i = findfirst(isnan, stdlist)
        println("Warning: All samples have escaped the given region at time = ", times[i])
        times = times[1:i-1]
        meanlist = meanlist[1:i-1]
    end
    _, slope = linear_region(times, meanlist)
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
# Maxmal expansion rate
#####################################################################################

"""
    maximalexpansion(M)

Calculates the maximal expansion rate of M,
i.e. the product of all singular values of M that are greater than 1. In the
notation of [1], it is the function ``G``.
"""
maximalexpansion(M) = prod(filter(x -> x > 1.0, svdvals(M)))

#####################################################################################
# Expansion entropy
#####################################################################################

"""
    expansionentropy_sample(ds, sampler, restraining; kwargs...)

Return `times, H` for one sample of `ds` (see [`expansionentropy`](@ref)).
Accepts the same argumets as `expansionentropy`, besides `batches`.
"""
function expansionentropy_sample(system::DynamicalSystem, sampler, restraining;
    N=1000, steps=40, dt=1, Ttr=0, diffeq...)
    D = dimension(system)
    M = zeros(steps)
    # M[t] will be Σᵢ G(Dfₜ₀,ₜ₀₊ₜ(xᵢ))
    # The summation is over all sampled xᵢ that stay inside S during [t0, t0 + t].

    t0 = system.t0
    times = @. (t0+Ttr)+dt*(1:steps)

    t_identity = SMatrix{D, D, Float64}(I)

    # TODO: Fix this
    if Ttr > 0
        shifted_system = _timeshift(system, Ttr)
        t_integ = tangent_integrator(shifted_system)
    else
        t_integ = tangent_integrator(system)
    end

    u_integ = integrator(system, system.u0)

    if isa(system, DiscreteDynamicalSystem)
        dt = max(Int(floor(dt)), 1)
        Ttr = max(Int(floor(Ttr)), 0)
    end

    for i ∈ 1:N
        u = sampler() # New sample point.

        # TODO: This is correct, but fix in general
        if Ttr > 0 # Evolve through transient time.
            reinit!(u_integ, u)
            step!(u_integ, Ttr, true) # stepping must be exact here for correct time vector
            u = u_integ.u
        end
        # TODO: reduce this
        reinit!(t_integ, u, t_identity) # Start integrating the tangents.

        for t ∈ 1:steps # Evolve the sample point for the duration [t0, t0+steps*dt]
            step!(t_integ, dt, true)
            # step!(t_integ, dt, exactstepping)

            u = get_state(t_integ)
            !restraining(u) && break # Stop the integration if the orbit leaves the region.
            Df = get_deviations(t_integ)
            M[t] += maximalexpansion(Df)
        end
    end
    return times, log.(M./N)
end

function _timeshift(system, Ttr)
    # TODO: implement this better.
    f = system.f
    u0 = system.u0
    jac = system.jacobian
    p = system.p
    t0 = system.t0
    if isa(system, DiscreteDynamicalSystem)
        shifted_system = DiscreteDynamicalSystem(f, u0, p, jac; t0=t0+Ttr)
    else
        shifted_system = ContinuousDynamicalSystem(f, u0, p, jac; t0=t0+Ttr)
    end
    return shifted_system
end

#=
This version only deals with 1-dimensional discrete dynamical systems, but does
it really fast, by avoiding `tangent_integrator` and `maximalexpansion`.
=#
function expansionentropy_sample(system::DiscreteDynamicalSystem{IIP, S, 1}, sampler, restraining;
    N=1000, steps=40, dt=1, Ttr=0, kwargs...) where {IIP, S}
    f = system.f
    p = system.p
    jacob = system.jacobian
    t0 = system.t0
    times = @. (t0+Ttr)+dt*(1:steps)

    dt = max(Int(floor(dt)), 1)
    Ttr = max(Int(floor(Ttr)), 0)
    M = zeros(steps)

    # whether the state points of system is represented by naked numbers or vectors
    isnumber = typeof(system.u0) <: Number

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
            for _ ∈ 1:dt # Evolve dt steps at a time.
                Df = jacob(x, p, t) * Df
                x = f(x, p, t)
                t += 1
            end

            !restraining(x) && break
            M[step] += isnumber ? max(1.0, abs(Df)) : max(1.0, abs(Df[1]))
        end
    end
    return times, log.(M./N)
end


#####################################################################################
# Graphing functions
#####################################################################################
"""
    expansionentropy_batch(ds, sampler, restraining; kwargs...)

Run [`expansionentropy_sample`](@ref) `batch` times, and return
`times, mean(H), std(H)` for all resulting `H`, see [`expansionentropy`](@ref).

Accepts the same arguments as `expansionentropy`.
"""
function expansionentropy_batch(system, sampler, restraining; batches=100, steps=40, kwargs...)
    meanlist = zeros(steps)
    stdlist = zeros(steps)
    eesamples = zeros(batches, steps)
    # eesamples[k, t] = The k-th sample of expansion entropy from t0 to (t0 + t)

    times = undef
    # Collect all the samples
    for k in 1:batches
        times, eesamples[k, :] = expansionentropy_sample(system, sampler, restraining; steps=steps, kwargs...)
    end

    # Calculate the mean and standard deviations
    for t in 1:steps
        entropysamples = filter(isfinite, @view eesamples[:, t])
        # remove -Inf entries, which indicate all samples failed to stay inside the given region.
        if length(entropysamples) < 1
            println("Warning: All samples have escaped the given region. Consider increasing sample or batch number. Terminating at step = ", t)
            break
        end
        meanlist[t] = mean(entropysamples)
        stdlist[t] = std(entropysamples)
    end
    return times, meanlist, stdlist
end
