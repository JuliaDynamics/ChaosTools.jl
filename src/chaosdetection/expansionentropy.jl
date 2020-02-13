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
* `batch = 100` : Number of batches to run the calculation of ``E``. Across these
  batches the mean value and standard deviation of ``\\log E_{t0+T, t0}(f, S)`` are computed
  which when plotted create the curves and error bars of e.g. Figs 2, 3 of [1].
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

This function `expansionentropy` simply returns the slope of the biggest linear region
of the curve ``H`` versus ``t``, which approximates the expansion entropy ``H_0``.

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
    boxregion(as::Vector{Real}, bs::Vector{Real})

`as`, `bs` defines a box in `\\mathbb{R}`, the smallest box that is parallel to
the axes, and contains `as, bs`.

Returns two functions `gen, restraining`. `gen` generates points uniformly randomly
in the box, and is used as `gen()`.

`restraining` takes one point and returns `true` if the point is in the box, else
it returns `false`.

Points are represented as `Vector{Real}` objects.
"""
function boxregion(as::Vector{T}, bs::Vector{T}) where T <: Real
    @assert length(as) == length(bs) > 0
    zipped = zip(as, bs)
    as = map(minimum, zipped)
    bs = map(maximum, zipped)
    gen() = [rand()*(bs[i]-as[i])+as[i] for i in 1:length(as)]
    restraining(x) = all(as .< x .< bs)
    return gen, restraining
end

"""
    boxregion(a::Real, b::Real)

Specialized version in one dimension.
"""
function boxregion(a::Real, b::Real)
    zipped = (a,b)
    a = minimum(zipped)
    b = maximum(zipped)
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

[1] : [Hunt, Brian R., and Edward Ott, ‘Defining Chaos’, Chaos: An Interdisciplinary Journal of Nonlinear Science, 25.9 (2015)](https://doi.org/10/gdtkcf)
"""
maximalexpansion(M) = prod(filter(x -> x > 1.0, svdvals(M)))

#####################################################################################
# Expansion entropy
#####################################################################################

"""
    expansionentropy_sample(system, sampler, restraining; samplecount=1000, steps=40, dt=1, Ttr=0, exactstepping=false, diffeq...)

* `system` : The dynamical system on which to calculate the expansion entropy (EE).
* `sampler` : A nullary function that upon calling, returns a point in
  state space that falls within the region implicitly defined by `restraining`.
  This sampling function must sample evenly, otherwise the result will be biased.
  Future implementation may allow importance sampling.
* `restraining` : A boolean function that implicitly defines a subset of the state
  space. All orbits must stay within this subset to be counted.
* `samplecount=1000` : The number of samples.
* `steps=40` : The maximal time for which the system will be run.
* `dt=1` : The size of one step of time-evolution of the system.
* `Ttr=0` : Transient time to evolve the system with, after each point sampling,
  before starting the expansion entropy sampling.
* `exactstepping=false` : If `system` is a ContinuousDynamicalSystem, and
  exactstepping is true, then step by exactly dt. If exactstepping is false,
  then the step-size is only guaranteed to be at least dt, since it would be chosen
  by the integrator.
  Setting this to true would improve accuracy, although usually unnecessary.
* `diffeq...` : Keyword arguments propagated into `init` of DifferentialEquations.jl.
  Only valid for continuous systems.

Returns `times` and `H`. `times` is a vector of [t0+dt, t0+2dT, ... t0+steps*dt].
`H` is a Vector{Float64} of length `steps`, such that
for any 1 ≤ t ≤ T, where T = `steps`,
```math
H[t] = \\log E_{t0+T, t0}(f, S),
```
with
```math
E_{t0+T, t0}(f, S) = \\left[\\frac 1 N \\sum_{i} G(Df_{t0+t, t0}(x_i)) \\right]
```
in the notation of (Ott, 2015) [1]:

* ``N`` is `samplecount`,
* ``f_{steps+t0, t0}`` is the equation of motion for the system from time ``t0`` to ``t0+steps``,
* ``D`` is the differential with respect to ``x``,
* ``p`` is the set of other parameters for ``f`,
* ``S`` is a region implicitly defined by the boolean function `restraining`,
* ``G`` denotes the product of all singular values greater than ``1``,
* ``x_i`` are random points in the region ``S``, sampled by `sampler`.
* ``i`` is the index of the sample points, ranging from 1 to `samplecount`.

The summation is only over ``x_i`` that stays inside the region ``S`` implicitly
defined by the boolean function `restraining` as ``x`` moves during ``[t0, t0+T]``,
with its motion defined by ``f_{T+t0, t0}``. That is, for all ``t0 \\le t \\le t0+T``,
``f_{t, t0}(x) \\in S``, that is, ``f_{t, t0}(x)`` evaluates to `true` by `restraining`.

[1] : [Hunt, Brian R., and Edward Ott, ‘Defining Chaos’, Chaos: An Interdisciplinary Journal of Nonlinear Science, 25.9 (2015)](https://doi.org/10/gdtkcf)
"""
function expansionentropy_sample(system::DynamicalSystem, sampler, restraining; samplecount=1000, steps=40, dt=1, Ttr=0, exactstepping=false, diffeq...)
    D = dimension(system)
    M = zeros(steps)
    # M[t] will be Σᵢ G(Dfₜ₀,ₜ₀₊ₜ(xᵢ))
    # The summation is over all sampled xᵢ that stay inside S during [t0, t0 + t].

    t0 = system.t0
    times = @. (t0+Ttr)+dt*(1:steps)

    t_identity = SMatrix{D, D, Float64}(I)

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

    for i ∈ 1:samplecount
        u = sampler() # New sample point.

        if Ttr > 0 # Evolve through transient time.
            reinit!(u_integ, u)
            step!(u_integ, Ttr)
            u = u_integ.u
        end

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
    return times, log.(M./samplecount)
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
function expansionentropy_sample(system::DiscreteDynamicalSystem{IIP, S, 1}, sampler, restraining; samplecount=1000, steps=40, dt=1, Ttr=0, kwargs...) where {IIP, S}
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

    for i ∈ 1:samplecount
        x = sampler()
        t = t0

        # Evolve for a transient time.
        for _ ∈ 1:Ttr
            x = f(x, p, t)
            t += 1
        end

        Df = isnumber ? 1.0 : Matrix{Float64}(I, 1, 1)
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
    return times, log.(M./samplecount)
end


#####################################################################################
# Graphing functions
#####################################################################################

#=
This section provides functions for plotting in an easy-to-read format.
=#
"""
    expansionentropy_batch(system, sampler, restraining; batchcount=100, steps=40, kwargs...)

* `system` : The dynamical system on which to calculate the expansion entropy (EE).
* `sampler` : A nullary function that upon calling, returns a point in
  state space that falls within the region implicitly defined by `restraining`.
  This sampling function must sample evenly, otherwise the result will be biased.
  Future implementation may allow importance sampling.
* `restraining` : A boolean function that implicitly defines a subset of the state
  space. All orbits must stay within this subset to be counted.
* `batchcount=100` : The number of batches to run the experiment with.
  These batches are then aggregated to produce the mean and standard deviation.
* `steps=40` : The maximal steps for which the system will be run.
* `kwargs...` : Keyword arguments propagated into `expansionentropy_sample`.

Runs the `expansionentropy_sample` function `batchcount` times, each time calculating the
expansion entropy out to `steps` time-steps. After that, it combines the simulated
results from each run, thus obtaining `batchcount` number of estimates of
log Eₜ₀₊ₜ,ₜ₀(f, S) for each t ∈ 1:steps.

It then calculates the mean and standard deviation for each t ∈ 1:steps, and return
them as two lists.

In the notation of (Hunt and Ott, 2015) [1], ``f`` is the equation of motion of
the dynamical system, ``S`` is the region defined implicitly by `restraining`.

As an example usage, this reproduces plot 1 of [1]:

```julia
using Plots
tent_eom(x, p, n) = (x < -0.2 ? -0.6 : (x < 0.4 ? 3x : 2(1-x)))
tent_jacob(x, p, n) = (x < -0.2 ? 0 : (x < 0.4 ? 3 : -2))
tent = DiscreteDynamicalSystem(tent_eom, 0.2, nothing, tent_jacob)
times, tent_meanlist, tent_stdlist = expansionentropy_batch(tent, rand, x -> 0 < x < 1; batchcount=100, samplecount=100000, steps=60, dt=1)
plot(times, tent_meanlist, yerr=tent_stdlist, leg=false)
```

[1] : [Hunt, Brian R., and Edward Ott, ‘Defining Chaos’, Chaos: An Interdisciplinary Journal of Nonlinear Science, 25.9 (2015)](https://doi.org/10/gdtkcf)
"""
function expansionentropy_batch(system, sampler, restraining; batchcount=100, steps=40, kwargs...)
    meanlist = zeros(steps)
    stdlist = zeros(steps)
    eesamples = zeros(batchcount, steps)
    # eesamples[k, t] = The k-th sample of expansion entropy from t0 to (t0 + t)

    times = undef
    # Collect all the samples
    for k in 1:batchcount
        times, eesamples[k, :] = expansionentropy_sample(system, sampler, restraining; steps=steps, kwargs...)
    end

    # Calculate the mean and standard deviations
    for t in 1:steps
        entropysamples = filter(isfinite, eesamples[:, t])
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
