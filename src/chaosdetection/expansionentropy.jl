export discreteEEsample, continuousEEsample, EEgraph, maximalexpansion
using LinearAlgebra, DifferentialEquations

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
    discreteEEsample(system, samplegenerator, isinside; samplecount=1000, steps=40)

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
function discreteEEsample(system, samplegenerator, isinside; samplecount=1000, steps=40)
    dim = dimension(system)
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
        Df_new = Matrix{Float64}(I, dim, dim)
        x_new = zeros(dim)
    end

    for i ∈ 1:samplecount
        Df = isnumber ? 1.0 : SMatrix{dim, dim}(Diagonal(fill(1.0, dim)))
        x = samplegenerator()
        @assert isinside(x) "The sample generator \n $samplegenerator \n generated a point that does not exist in the region implicitly defined by the boolean function \n $isinside."
        # sampled points must stay inside the region.

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
            M[t] += isnumber ? max(1.0, abs(Df)) : maximalexpansion(Df)
        end
    end
    return log.(M./samplecount)
end

"""
    __oop_ify(system)

Make an oop version of an iip dynamical system. Written so I don't have to deal
with iip dynamical systems. Only used in `continuousEEsample`.
"""
function __oop_ify(system)
    f = system.f
    u0 = system.u0
    p = system.p
    jacob = system.jacobian
    t0 = system.t0

    function f_oop(u, p, t)
        new_u = copy(u)
        f(new_u, u, p, t)
        return new_u
    end

    function jacob_oop(u, p, t)
        new_jacob = copy(system.J)
        jacob(new_jacob, u, p, t)
        return new_jacob
    end

    return ContinuousDynamicalSystem(f_oop, u0, p, jacob_oop)
end

"""
    continuousEEsample(system::ContinuousDynamicalSystem, samplegenerator, isinside; samplecount=1000::Int, steps=40.0::Real)

This function is essentially the same as discreteEEsample, with one extra option
to specify dT. Also, due to technical difficulties, instead of checking if
the whole trajectory remains inside the given region, it only checks if the
trajectory remains inside at time steps {t0, t0 + dT, ... t0 + steps * dT}.
"""
function continuousEEsample(system::ContinuousDynamicalSystem, samplegenerator, isinside; samplecount=1000::Int, dT=1.0::Real, steps=20::Int)
    if DynamicalSystemsBase.isinplace(system)
        system = __oop_ify(system)
    end

    dim = dimension(system)
    eom = system.f
    p = system.p
    jacob = system.jacobian
    t0 = system.t0
    t = 0

    M = zeros(steps)
    # M[t] = Σᵢ G(Dfₜ₀₊ₜ,ₜ₀(xᵢ)), if xᵢ stayed inside S during [t0, t0 + t].

    if DynamicalSystemsBase.isinplace(system)
        system = __oop_ify(system)
    end

    # The tangent dynamic
    function uv_eom(uv, p, t)
        u = @view uv[:, 1]
        v = @view uv[:, 2:dim+1]
        return hcat(eom(u, p, t), jacob(u, p, t)*v)
    end

    u0 = zeros(dim)
    v0 = Matrix{Float64}(I, dim, dim)
    uv0 = hcat(u0, v0)

    prob = ODEProblem(uv_eom, uv0, (t0, t0+steps*dT), p)
    integ = DiffEqBase.init(prob, Tsit5())

    for i ∈ 1:samplecount
        u0 = samplegenerator()
        @assert isinside(u0) "The sample generator \n $samplegenerator \n generated a point that does not exist in the region implicitly defined by the boolean function \n $isinside."

        uv0[:, 1] .= u0

        # Reset integrator to t0, at uv0.
        # reinit!(integ, uv0, t0)
        # For some strange reason, this reinit! fails.

        prob = ODEProblem(uv_eom, uv0, (t0, t0+steps*dT), p)
        integ = DiffEqBase.init(prob, Tsit5())
        for t in 1:steps
            DifferentialEquations.step!(integ, dT, true)
            u = @view integ.u[:, 1]
            !isinside(u) && break
            v = @view integ.u[:, 2:dim+1]
            M[t] += maximalexpansion(v)
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
    EEsample = zeros(batchcount, steps)
    # EEsample[k, t] = The k-th sample of expansion entropy from t0 to (t0 + t)

    for k in 1:batchcount
        if isa(system, DiscreteDynamicalSystem)
            EEsample[k, :] = discreteEEsample(system, samplegenerator, isinside; samplecount=samplecount, steps=steps)
        else
            EEsample[k, :] = continuousEEsample(system, samplegenerator, isinside; samplecount=samplecount, steps=steps, dT=dT)
        end
    end
    for t in 1:steps
        entropysamples = filter(isfinite, EEsample[:, t])
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
@time tent_meanlist, tent_stdlist = EEgraph(tent, rand, x -> 0 < x < 1; batchcount=100, samplecount=1000, steps=40)
plot(tent_meanlist, yerr=tent_stdlist, leg=false)

# This replicates Figure 3.
tent_gen() = rand()*2.5 - 1
@time meanlist, stdlist = EEgraph(tent, tent_gen, x -> -1 < x < 1.5; batchcount=100, samplecount=100000, steps=60)
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
