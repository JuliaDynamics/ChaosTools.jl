export discreteEEsample, continuousEEsample, EEgraph, maximalexpansion
using LinearAlgebra, DifferentialEquations, Statistics
using OrdinaryDiffEq: Vern9
using DynamicalSystemsBase, DifferentialEquations

#=
This package contains two main functions for calculating the expansion entropy
of dynamical systems. One handles the discrete case, the other the continuous.

Expansion entropy is defined in (Hunt and Ott, 2015) [1] as a quantitative measure
of chaos.

[1] : : [Hunt, Brian R., and Edward Ott, ‘Defining Chaos’, Chaos: An Interdisciplinary Journal of Nonlinear Science, 25.9 (2015)](https://doi.org/10/gdtkcf)

=#

#####################################################################################
# Maxmal expansion rate
#####################################################################################

"""
Calculates the maximal expansion rate of M among all possible subspaces,
the product of all singular values of M that are greater than 1. In the
notation of [1], it is the function ``G``.

[1] : [Hunt, Brian R., and Edward Ott, ‘Defining Chaos’, Chaos: An Interdisciplinary Journal of Nonlinear Science, 25.9 (2015)](https://doi.org/10/gdtkcf)
"""
maximalexpansion(M) = prod(filter(x -> x>1.0, svdvals(M)))

@assert maximalexpansion([1. 0. 0. 0. 2.; 0. 0. 3. 0. 0.; 0. 0. 0. 0. 0.; 0. 2. 0. 0. 0.]) == 3.0 * 2.23606797749979 * 2.0
@assert maximalexpansion([1 0; 0 1]) == 1


#####################################################################################
# Discrete expansion entropy
#####################################################################################

"""
    EEsample(system, samplegenerator, isinside; samplecount=1000, steps=40, dT=1)

* `system` : The dynamical system on which to calculate the expansion entropy (EE).
* `samplegenerator` : A nullary function that upon calling, returns a point in
  state space that falls within the region implicitly defined by `isinside`.
  This sampling function must sample evenly, otherwise the result will be biased.
  Future implementation may allow importance sampling.
* `isinside` : A boolean function that implicitly defines a subset of the state
  space. All orbits must stay within this subset to be counted.
* `samplecount=1000` : The number of samples.
* `steps=40` : The maximal time for which the system will be run.


Returns H::1×steps Array{Float64,2}, such that for any 1 ≤ t ≤ steps,
letting ``T=```steps`,
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
function EEsample(system::DiscreteDynamicalSystem, samplegenerator, isinside; samplecount=1000, steps=40, dT=1)
    dim = dimension(system)
    M = zeros(steps)
    t_identity = SMatrix{dim, dim, Float64}(I)
    # The identity matrix in tangent space
    t_integ = tangent_integrator(system)

    dT_int = Int(floor(dT))

    for i ∈ 1:samplecount
        u = samplegenerator()
        reinit!(t_integ, u, t_identity)

        for t ∈ 1:steps
            step!(t_integ, dT_int)
            u = get_state(t_integ)
            !isinside(u) && break

            Df = get_deviations(t_integ)
            M[t] += maximalexpansion(Df)
        end
    end
    return log.(M./samplecount)
end

function EEsample(system::ContinuousDynamicalSystem, samplegenerator, isinside; samplecount=1000, steps=40, dT=1.0)
    dim = dimension(system)
    M = zeros(steps)
    t_identity = SMatrix{dim, dim, Float64}(I)
    # The identity matrix in tangent space
    t_integ = tangent_integrator(system, alg=Vern9())

    for i ∈ 1:samplecount
        u = samplegenerator()
        reinit!(t_integ, u, t_identity)

        for t ∈ 1:steps
            step!(t_integ, dT, true)
            u = get_state(t_integ)
            !isinside(u) && break

            Df = get_deviations(t_integ)
            M[t] += maximalexpansion(Df)
        end
    end
    return log.(M./samplecount)
end

"""
This specialized version only deals with 1-dimensional discrete dynamical systems,
but does it really fast. Haven't tested it yet.
"""
function EEsample_1dim(system::DiscreteDynamicalSystem, samplegenerator, isinside; samplecount=1000, steps=40)
    f = system.f
    p = system.p
    jacob = system.jacobian
    t0 = system.t0

    M = zeros(steps)
    # M[t] = Σᵢ G(Dfₜ₀₊ₜ,ₜ₀(xᵢ)), if xᵢ stayed inside S during [t0, t0 + t].

    isnumber = typeof(system.u0) <: Number
    # whether the state points of system is represented by naked numbers or vectors
    isinplace = DynamicalSystemsBase.isinplace(system)
    # whether the system is in-place or out-of-place

    if isinplace # Make some dummy variables to take in-place updates.
        Df_new = Matrix{Float64}(I, 1, 1)
        x_new = zeros(1)
    end

    for i ∈ 1:samplecount
        Df = isnumber ? 1.0 : Matrix{Float64}(I, 1, 1)
        x = samplegenerator()

        for t ∈ 1:steps
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
    EEgraph(system, samplegenerator, isinside; batchcount=100, samplecount=1000, steps=40, dT=1.0)

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
    * `dT=1.0` : If the system is a ContinuousDynamicalSystem, then each step
      lasts this long.


Runs the `EEsample` function `batchcount` times, each time calculating the
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
function EEgraph(system, samplegenerator, isinside; batchcount=100, samplecount=1000, steps=40, dT=1.0)
    meanlist = zeros(steps)
    stdlist = zeros(steps)
    EEsamples = zeros(batchcount, steps)
    # EEsamples[k, t] = The k-th sample of expansion entropy from t0 to (t0 + t)

    # Collect all the samples
    for k in 1:batchcount
        EEsamples[k, :] = EEsample(system, samplegenerator, isinside; samplecount=samplecount, steps=steps, dT=dT)
    end

    # Calculate the mean and standard deviations
    for t in 1:steps
        entropysamples = filter(isfinite, EEsamples[:, t])
        # remove -Inf entries, which indicate all samples failed to stay inside the given region.
        if length(entropysamples) ≤ 1
            println("Warning: All samples have escaped the given region. Consider increasing sample or batch number. Terminating at steps = ", t)
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
@time meanlist, stdlist = EEgraph(expand2, rand, x -> 0 < x < 1; batchcount=100, samplecount=100000, steps=10)
plot(meanlist, yerr=stdlist, leg=false)

(system, samplegenerator, isinside; batchcount=100, samplecount=1000, steps=40, dT=1.0)

# Modified tent map. This replicates Example B in Hunt and Ott.
tent_eom(x, p, n) = (x < -0.2 ? -0.6 : (x < 0.4 ? 3x : 2(1-x)))
tent_jacob(x, p, n) = (x < -0.2 ? 0 : (x < 0.4 ? 3 : -2))
tent = DiscreteDynamicalSystem(tent_eom, 0.2, nothing, tent_jacob)

# This replicates Figure 2.
@time tent_meanlist, tent_stdlist = EEgraph(tent, rand, x -> 0 < x < 1; batchcount=100, samplecount=100000, steps=60)
plot(tent_meanlist, yerr=tent_stdlist, leg=false)

# This replicates Figure 3.
@time meanlist, stdlist = EEgraph(tent, () -> rand()*2.5 - 1, x -> -1 < x < 1.5; batchcount=100, samplecount=100000, steps=60)
plot(meanlist, yerr=stdlist, leg=false)

# This replicates Figure 7
henon_iip = Systems.henon_iip(zeros(2); a = 4.2, b = 0.3)
henon_gen() = rand(2).*6 .- 3
henon_isinside(x) = -3<x[1]<3 &&  -3<x[2]<3
@time meanlist, stdlist = EEgraph(henon_iip, henon_gen, henon_isinside; batchcount=100, samplecount=100000, steps=25)
plot(meanlist, yerr=stdlist, leg=false)

# Exponential system
exp_eom(x, p, t) = x*p[1]
exp_u0 = [0.1]
exp_p = [1.0]
exp_jacob(x, p, t) = [1.0]
exp_sys = ContinuousDynamicalSystem(exp_eom, exp_u0, exp_p, exp_jacob)

exp_gen() = [rand()-0.5]
exp_isinside(x) = -10.0 < x[1] < 10.0
@time exp_meanlist, exp_stdlist = EEgraph(exp_sys, exp_gen, exp_isinside; batchcount=100, samplecount=10000, steps=20, dT=1.0)
plot((1:20)*1.0, exp_meanlist, yerr=exp_stdlist, leg=false)


#=
The Lorenz attractor is roughly bounded within the box [-20, 20]×[-30, 30]×[0, 50]
As can be seen in the plot, the Lorentz attractor has an expansion entropy of
about 0.92
=#

lor = Systems.lorenz()
lor_gen() = [rand()*40-20, rand()*60-30, rand()*50]
lor_isinside(x) = -20 < x[1] < 20 && -30 < x[2] < 30 && 0 < x[3] < 50
@time meanlist, stdlist = EEgraph(lor, lor_gen, lor_isinside; batchcount=100, samplecount=1000, steps=20, dT=1.0)
plot(meanlist, yerr=stdlist, leg=false)
plot!(x->0.92x+1.5, xlims=(0, 20))
