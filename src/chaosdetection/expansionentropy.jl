#=
This file contains two main functions for calculating the expansion entropy
of dynamical systems. One handles the discrete case, the other the continuous.

Expansion entropy is defined in (Hunt and Ott, 2015) [1] as a quantitative measure
of chaos. For details, read the docstrings of the functions below.

[1] : [Hunt, Brian R., and Edward Ott, ‘Defining Chaos’, Chaos: An Interdisciplinary Journal of Nonlinear Science, 25.9 (2015)](https://doi.org/10/gdtkcf)
=#
export EEgraph, expansionentropy_sample, maximalexpansion
using LinearAlgebra
using Statistics


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
    expansionentropy_sample(system, samplegenerator, isinside; samplecount=1000, steps=40, dT=1, exactstepping=false)

* `system` : The dynamical system on which to calculate the expansion entropy (EE).
* `samplegenerator` : A nullary function that upon calling, returns a point in
  state space that falls within the region implicitly defined by `isinside`.
  This sampling function must sample evenly, otherwise the result will be biased.
  Future implementation may allow importance sampling.
* `isinside` : A boolean function that implicitly defines a subset of the state
  space. All orbits must stay within this subset to be counted.
* `samplecount=1000` : The number of samples.
* `steps=40` : The maximal time for which the system will be run.
* `dT=1` : The size of one step of time-evolution of the system.
* `exactstepping=false` : If `system` is a ContinuousDynamicalSystem, and
  exactstepping is true, then step by exactly dT. If exactstepping is false,
  then the step-size is only guaranteed to be at least dT, since it would be chosen
  by the integrator.
  Setting this to true would improve accuracy, although usually unnecessary.

Returns `H`, an Array{Float64,2} of dimensions 1×steps, such that
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
* ``S`` is a region implicitly defined by the boolean function `isinside`,
* ``G`` denotes the product of all singular values greater than ``1``,
* ``x_i`` are random points in the region ``S``, sampled by `samplegenerator`.
* ``i`` is the index of the sample points, ranging from 1 to `samplecount`.

The summation is only over ``x_i`` that stays inside the region ``S`` implicitly
defined by the boolean function `isinside` as ``x`` moves during ``[t0, t0+T]``,
with its motion defined by ``f_{T+t0, t0}``. That is, for all ``t0 \\le t \\le t0+T``,
``f_{t, t0}(x) \\in S``, that is, ``f_{t, t0}(x)`` evaluates to `true` by `isinside`.

[1] : [Hunt, Brian R., and Edward Ott, ‘Defining Chaos’, Chaos: An Interdisciplinary Journal of Nonlinear Science, 25.9 (2015)](https://doi.org/10/gdtkcf)
"""
function expansionentropy_sample(system::DynamicalSystem, samplegenerator, isinside; samplecount=1000, steps=40, dT=1, exactstepping=false, diffeq...)
    D = dimension(system)
    M = zeros(steps)
    # M[t] will be Σᵢ G(Dfₜ₀,ₜ₀₊ₜ(xᵢ))
    # The summation is over all sampled xᵢ that stay inside S during [t0, t0 + t].

    t_identity = SMatrix{D, D, Float64}(I)
    t_integ = tangent_integrator(system)

    for i ∈ 1:samplecount
        u = samplegenerator() # New sample point.
        reinit!(t_integ, u, t_identity)
        for t ∈ 1:steps # Evolve the sample point for the duration [t0, t0+steps*dT]
            step!(t_integ, dT, exactstepping)
            u = get_state(t_integ)
            !isinside(u) && break # Stop the integration if the orbit leaves the region.
            Df = get_deviations(t_integ)
            M[t] += maximalexpansion(Df)
        end
    end
    return log.(M./samplecount)
end

#=
This version only deals with 1-dimensional discrete dynamical systems, but does
it really fast, by avoiding `tangent_integrator` and `maximalexpansion`.
=#
function expansionentropy_sample(system::DiscreteDynamicalSystem{IIP, S, 1}, samplegenerator, isinside; samplecount=1000, steps=40, dT=1) where {IIP, S}
    f = system.f
    p = system.p
    jacob = system.jacobian
    t0 = system.t0

    dT_int = max(Int(floor(dT)), 1)
    M = zeros(steps)

    # whether the state points of system is represented by naked numbers or vectors
    isnumber = typeof(system.u0) <: Number

    if IIP # Make some dummy variables to take in-place updates.
        Df_new = Matrix{Float64}(I, 1, 1)
        x_new = zeros(1)
    end

    for i ∈ 1:samplecount
        Df = isnumber ? 1.0 : Matrix{Float64}(I, 1, 1)
        x = samplegenerator()

        for t ∈ 1:steps # Evolve point x for `steps` steps.
            for _ ∈ 1:dT_int # Evolve dT_int steps at a time.
                if isinplace
                    f(x_new, x, p, t0 + t-1)
                    jacob(Df_new, x, p, t0 + t-1)
                    Df = Df_new * Df
                    x .= x_new
                else
                    Df = jacob(x, p, t0 + t-1) * Df
                    x = f(x, p, t0 + t-1)
                end
            end

            !isinside(x) && break
            M[t] += isnumber ? max(1.0, abs(Df)) : max(1.0, abs(Df[1]))
        end
    end
    return log.(M./samplecount)
end


#####################################################################################
# Graphing functions
#####################################################################################

#=
This section provides functions for plotting in an easy-to-read format.
=#
"""
    EEgraph(system, samplegenerator, isinside; batchcount=100, samplecount=1000, steps=40, dT=1, exactstepping=false, diffeq...)

* `system` : The dynamical system on which to calculate the expansion entropy (EE).
* `samplegenerator` : A nullary function that upon calling, returns a point in
  state space that falls within the region implicitly defined by `isinside`.
  This sampling function must sample evenly, otherwise the result will be biased.
  Future implementation may allow importance sampling.
* `isinside` : A boolean function that implicitly defines a subset of the state
  space. All orbits must stay within this subset to be counted.
* `batchcount=100` : The number of batches to run the experiment with.
  These batches are then aggregated to produce the mean and standard deviation.
* `samplecount=1000` : The number of samples.
* `steps=40` : The maximal steps for which the system will be run.
* `dT=1` : The size of one step of time-evolution of the system.
* `exactstepping=false` : If `system` is a ContinuousDynamicalSystem, and
  exactstepping is true, then step by exactly dT. If exactstepping is false,
  then the step-size is only guaranteed to be at least dT, since it would be chosen
  by the integrator.
  Setting this to true would improve accuracy, although usually unnecessary.
* `diffeq...` : Keyword arguments propagated into `init` of DifferentialEquations.jl.
  Only valid for continuous systems.

Runs the `expansionentropy_sample` function `batchcount` times, each time calculating the
expansion entropy out to `steps` time-steps. After that, it combines the simulated
results from each run, thus obtaining `batchcount` number of estimates of
log Eₜ₀₊ₜ,ₜ₀(f, S) for each t ∈ 1:steps.

It then calculates the mean and standard deviation for each t ∈ 1:steps, and return
them as two lists.

In the notation of (Hunt and Ott, 2015) [1], ``f`` is the equation of motion of
the dynamical system, ``S`` is the region defined implicitly by `isinside`.

As an example usage, this reproduces plot 1 of [1]:

```julia
using Plots
tent_eom(x, p, n) = (x < -0.2 ? -0.6 : (x < 0.4 ? 3x : 2(1-x)))
tent_jacob(x, p, n) = (x < -0.2 ? 0 : (x < 0.4 ? 3 : -2))
tent = DiscreteDynamicalSystem(tent_eom, 0.2, nothing, tent_jacob)

tent_meanlist, tent_stdlist = EEgraph(tent, rand, x -> 0 < x < 1; batchcount=100, samplecount=1000, steps=40)
plot(tent_meanlist, yerr=tent_stdlist, leg=false)
```

[1] : [Hunt, Brian R., and Edward Ott, ‘Defining Chaos’, Chaos: An Interdisciplinary Journal of Nonlinear Science, 25.9 (2015)](https://doi.org/10/gdtkcf)
"""
function EEgraph(system, samplegenerator, isinside; batchcount=100, samplecount=1000, steps=40, dT=1, exactstepping=false, diffeq...)
    meanlist = zeros(steps)
    stdlist = zeros(steps)
    EEsamples = zeros(batchcount, steps)
    # EEsamples[k, t] = The k-th sample of expansion entropy from t0 to (t0 + t)

    # Collect all the samples
    for k in 1:batchcount
        if dimension(system) == 1 && isa(system, DiscreteDynamicalSystem)
            EEsamples[k, :] = EEsample_1dim(system, samplegenerator, isinside; samplecount=samplecount, steps=steps, dT=dT)
        else
            EEsamples[k, :] = expansionentropy_sample(system, samplegenerator, isinside; samplecount=samplecount, steps=steps, dT=dT, exactstepping=exactstepping, diffeq...)
        end
    end

    # Calculate the mean and standard deviations
    for t in 1:steps
        entropysamples = filter(isfinite, EEsamples[:, t])
        # remove -Inf entries, which indicate all samples failed to stay inside the given region.
        if length(entropysamples) < 1
            println("Warning: All samples have escaped the given region. Consider increasing sample or batch number. Terminating at step = ", t)
            break
        end
        meanlist[t] = mean(entropysamples)
        stdlist[t] = std(entropysamples)
    end
    return meanlist, stdlist
end
