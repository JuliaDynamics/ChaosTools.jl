export discreteEEsample
using LinearAlgebra

#=
This package contains two main functions for calculating the expansion entropy
of dynamical systems. One handles the discrete case, the other the continuous.

Expansion entropy is defined in (Hunt and Ott, 2015) [1] as a quantitative measure
of chaos.

[1] : : [Hunt, Brian R., and Edward Ott, ‘Defining Chaos’, Chaos: An Interdisciplinary Journal of Nonlinear Science, 25.9 (2015)](https://doi.org/10/gdtkcf)

=#


"""
Hunt and Ott in [1].

## Keyword Arguments ## TODO
*
*

## Description


## Performance Notes


## References

[1] : [Hunt, Brian R., and Edward Ott, ‘Defining Chaos’, Chaos: An Interdisciplinary Journal of Nonlinear Science, 25.9 (2015)](https://doi.org/10/gdtkcf)
"""

#####################################################################################
# Maxmal expansion rate
#####################################################################################

"""
Calculates the maximal expansion rate of M among all possible subspaces,
the product of all singular values of M that are greater than 1. In the
notation of [1], it is

[1] : [Hunt, Brian R., and Edward Ott, ‘Defining Chaos’, Chaos: An Interdisciplinary Journal of Nonlinear Science, 25.9 (2015)](https://doi.org/10/gdtkcf)
"""
maximalexpansion(M) = prod(filter(x -> x>1.0, svdvals(M)))

@assert maximalexpansion([1. 0. 0. 0. 2.; 0. 0. 3. 0. 0.; 0. 0. 0. 0. 0.; 0. 2. 0. 0. 0.]) == 3.0 * 2.23606797749979 * 2.0
@assert maximalexpansion([1 0; 0 1]) == 1


#####################################################################################
# Discrete expansion entropy
#####################################################################################

"""
    discreteEEsample(system, samplegenerator, isinside; samplecount=1000, T=40)

* `system` : The dynamical system on which to calculate the expansion entropy (EE).
* `samplegenerator` : A nullary function that upon calling, returns a point in
  state space that falls within the region implicitly defined by `isinside`.
  This sampling function must sample evenly, otherwise the result will be biased.
  Future implementation may allow importance sampling.
* `isinside` : A boolean function that implicitly defines a subset of the state
  space. All orbits must stay within this subset to be counted.
* `samplecount=1000` : The number of samples.
* `T=40` : The maximal time for which the system will be run.

Returns H::1×T Array{Float64,2}, such that for any 1 ≤ t ≤ T,
```math
H[t] = \\log E_{t0+T, t0}(f, S),
```
with
```math
E_{t0+T, t0}(f, S) = \\left[\\frac 1 N \\sum_{i} G(Df_{t0+t, t0}(x_i)) \\right]
```
in the notation of (Ott, 2015) [1].

Here,
* ``N`` is `samplecount`,
* ``f_{T+t0, t0}`` is the equation of motion for the system from time ``t0`` to ``t0+T``,
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
function discreteEEsample(system, samplegenerator, isinside; samplecount=1000, T=40)
    dim = dimension(system)
    f = system.f
    p = system.p
    jacob = system.jacobian
    t0 = system.t0

    M = zeros(T)
    # M[t] = Σᵢ G(Dfₜ₀₊ₜ,ₜ₀(xᵢ)), if xᵢ stayed inside S during [t0, t0 + t].

    isnumber = typeof(system.u0) <: Number
    # whether the state points of system is represented by naked numbers or vectors
    isinplace = DynamicalSystemsBase.isinplace(system)
    # whether the system is in-place or out-of-place
    @assert !(dim > 1 && isnumber) "The system \n $system \n somehow represents its states with a single number, but still has $dim dimensions."
    @assert !(dim == 1 && isinplace) "The system \n $system \n somehow updates in-place despite represents its states with a single number, which is not updatable."

    if isinplace # Make some dummy variables to take in-place updates.
        Df_new = Matrix{Float64}(I, dim, dim)
        x_new = zeros(dim)
    end

    for i ∈ 1:samplecount
        Df = isnumber ? 1.0 : SMatrix{dim, dim}(Diagonal(fill(1.0, dim)))
        x = samplegenerator()
        @assert isinside(x) "The sample generator \n $samplegenerator \n generated a point that does not exist in the region implicitly defined by the boolean function \n $isinside."
        # sampled points must stay inside the region.

        for t ∈ 1:T
            if isinplace
                f(x_new, x, p, t0 + t-1)
                jacob(Df_new, x, p, t0 + t-1)
                Df = Df_new * Df
                x .= x_new
            else
                Df = jacob(x, p, t0 + t-1) * Df
                x = f(x, p, t0 + t-1)
            end
            !isinside(x) && break
            M[t] += isnumber ? max(1.0, abs(Df)) : maximalexpansion(Df)
        end
    end
    return log.(M./samplecount)
end



"""
    continuousEEsample(system::ContinuousDynamicalSystem, samplegenerator, isinside; samplecount=1000::Int, T=40.0::Real)

This function is essentially the same as discreteEEsample, with one difference
being that
"""
function continuousEEsample(system::ContinuousDynamicalSystem, samplegenerator, isinside; samplecount=1000::Int, T=40.0::Real)
    dim = dimension(system)
    f = system.f
    p = system.p
    jacob = system.jacobian
    t0 = system.t0

    M = zeros(T)
    # M[t] = Σᵢ G(Dfₜ₀₊ₜ,ₜ₀(xᵢ)), if xᵢ stayed inside S during [t0, t0 + t].

    isnumber = typeof(system.u0) <: Number
    # whether the state points of system is represented by naked numbers or vectors
    isinplace = DynamicalSystemsBase.isinplace(system)
    # whether the system is in-place or out-of-place
    @assert !(dim > 1 && isnumber) "The system \n $system \n somehow represents its states with a single number, but still has $dim dimensions."
    @assert !(dim == 1 && isinplace) "The system \n $system \n somehow updates in-place despite represents its states with a single number, which is not updatable."

    if isinplace # Make some dummy variables to take in-place updates.
        Df_new = Matrix{Float64}(I, dim, dim)
        x_new = zeros(dim)
    end

    for i ∈ 1:N
        Df = isnumber ? 1.0 : SMatrix{dim, dim}(Diagonal(fill(1.0, dim)))
        x = samplegenerator()
        @assert isinside(x) "The sample generator \n $samplegenerator \n generated a point that does not exist in the region implicitly defined by the boolean function \n $isinside."
        # sampled points must stay inside the region.

        for t ∈ 1:T
            if isinplace
                f(x_new, x, p, t0 + t-1)
                jacob(Df_new, x, p, t0 + t-1)
                Df = Df_new * Df
                x .= x_new
            else
                Df = jacob(x, p, t0 + t-1) * Df
                x = f(x, p, t0 + t-1)
            end
            !isinside(x) && break
            M[t] += isnumber ? max(1.0, abs(Df)) : maximalexpansion(Df)
        end
    end
    return log.(M./N)
end


#####################################################################################
# Graphing functions
#####################################################################################

#=
This section provides functions for plotting in an easy-to-read format.
=#
"""
    EEgraph(system, samplegenerator, isinside; batchcount=100, samplecount=1000, T=40, dT=1.0)

Runs the `EEsample` function `batchcount` times, each time calculating the
expansion entropy out to `T` time-steps. After that, it combines the simulated
results from each run, thus obtaining `batchcount` number of estimates of
log Eₜ₀₊ₜ,ₜ₀(f, S) for each t ∈ 1:T.

It then calculates the mean and standard deviation for each t ∈ 1:T, and return
them as two lists.

In the notation of (Hunt and Ott, 2015) [1], ``f`` is the equation of motion of
the dynamical system, ``S`` is the region defined implicitly by `isinside`.

[1] : [Hunt, Brian R., and Edward Ott, ‘Defining Chaos’, Chaos: An Interdisciplinary Journal of Nonlinear Science, 25.9 (2015)](https://doi.org/10/gdtkcf)
"""

function EEgraph(system, samplegenerator, isinside; batchcount=100, samplecount=1000, T=40, dT=1.0)
    meanlist = zeros(T)
    stdlist = zeros(T)
    EEsample = zeros(batchcount, T)
    # EEsample[k, t] = The k-th sample of expansion entropy from t0 to (t0 + t)

    for k in 1:batchcount
        if isa(system, DiscreteDynamicalSystem)
            EEsample[k, :] = discreteEEsample(system, samplegenerator, isinside; samplecount=samplecount, T=T)
        else
            EEsample[k, :] = continuousEEsample(system, samplegenerator, isinside; samplecount=samplecount, T=T, dT=dT)
        end
    end
    for t in 1:T
        entropysamples = filter(isfinite, EEsample[:, t])
        # remove -Inf entries, which indicate all samples failed to stay inside the given region.
        if length(entropysamples) ≤ 1
            println("Warning: All samples have escaped the given region. Consider increasing sample or batch number. Terminating at T = ", t)
            break
        end
        meanlist[t] = mean(entropysamples)
        stdlist[t] = std(entropysamples)
    end
    return meanlist, stdlist
end

### Tests
# The following should be put into the documentation once this is finished.

# Expand 2: is not chaotic, thus the plot should be a flat line.
expand2_eom(x, p, n) = 2x
expand2_jacob(x, p, n) = 2
expand2 = DiscreteDynamicalSystem(expand2_eom, 0.2, nothing, expand2_jacob)
@time meanlist, stdlist = EEgraph(expand2, rand, x -> 0 < x < 1; batchcount=100, samplecount=100000, T=10)
plot(meanlist, yerr=stdlist, leg=false)

(system, samplegenerator, isinside; batchcount=100, samplecount=1000, T=40, dT=1.0)

# Modified tent map. This replicates Example B in Hunt and Ott.
tent_eom(x, p, n) = (x < -0.2 ? -0.6 : (x < 0.4 ? 3x : 2(1-x)))
tent_jacob(x, p, n) = (x < -0.2 ? 0 : (x < 0.4 ? 3 : -2))
tent = DiscreteDynamicalSystem(tent_eom, 0.2, nothing, tent_jacob)

# This replicates Figure 2.
@time meanlist, stdlist = EEgraph(tent, rand, x -> 0 < x < 1; batchcount=100, samplecount=1000, T=40)
plot(meanlist, yerr=stdlist, leg=false)

# This replicates Figure 3.
tent_gen() = rand()*2.5 - 1
@time meanlist, stdlist = EEgraph(tent, tent_gen, x -> -1 < x < 1.5; batchcount=100, samplecount=100000, T=60)
plot(meanlist, yerr=stdlist, leg=false)

# This replicates Figure 7
henon_iip = Systems.henon_iip(zeros(2); a = 4.2, b = 0.3)
henon_gen() = rand(2).*6 .- 3
henon_isinside(x) = -3<x[1]<3 &&  -3<x[2]<3
@time meanlist, stdlist = EEgraph(henon_iip, henon_gen, henon_isinside; batchcount=100, samplecount=100000, T=25)
plot(meanlist, yerr=stdlist, leg=false)

#=
The Lorenz attractor is roughly bounded within the box [-20, 20]×[-30, 30]×[0, 50]
=#
lor = Systems.lorenz()
lor_gen() = [rand()*40-20, rand()*60-30, rand()*50]
lor_isinside(x) = -20 < x[1] < 20 && -30 < x[2] < 30 && 0 < x[3] < 50
@time meanlist, stdlist = EEgraph(lor, lor_gen, lor_isinside; batchcount=100, samplecount=100000, T=20, dT=10.0)
plot((1:20)*10.0, meanlist, yerr=stdlist, leg=false)
