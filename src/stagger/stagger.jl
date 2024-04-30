export stagger_and_step!, stagger_trajectory!
using LinearAlgebra:norm
using Random
using ProgressMeter


"""
    stagger_trajectory!(x0 ,ds, isinside; kwargs...) -> xi

This function returns a point `xi` which _guarantees_ `T(xi) > 
Tm`. This is an auxiliary function for [`stagger_and_step`](@
ref). Keyword arguments and definitions are identical for both 
functions. 

The initial search radius is much bigger, `δ = 1.` by default.

"""
function stagger_trajectory!(x0 ,ds, isinside; δ = 1., Tm = 30, stagger_mode = :exp, max_steps = Int(1e5), f = 1.1)
    T = escape_time!(x0, ds, isinside)
    xi = deepcopy(x0) 
    while T < Tm  # we must have T ≥ Tm at each step 
        xi, T = get_stagger!(xi, ds, δ, T, isinside; f, stagger_mode, max_steps)
        if T < 0
            error("Cannot find a stagger trajectory. Choose a different starting point or search radius δ.")
        end 
            
    end
    return xi
end


"""
    stagger_and_step!(x0, ds::DynamicalSystem , N::Int, isinside::function; kwargs...) -> trajectory

This function implements the stagger-and-step method 
[^Sweet2001] to approximate the invariant non-attracting set 
governing the chaotic transient dynamics of a system, namely 
the stable manifold of a chaotic saddle. 

Given the dynamical system `ds` and a initial guess `x0` in a 
region *with no attractors* defined by the membership function 
`isinside`, the algorithm provides `N` points close to the 
stable manifold that escape from the region after a least `Tm` 
steps of `ds`. The search is stochastic and depends on the 
parameter `δ` defining a (small) neighborhood of search. 

The function `isinside(x)` returns `true` if the point `x` is 
inside the chosen bounded region and `false` otherwise. See 
[`statespace_sampler`](@ref) to construct this function.

## Description 

The method relies on the stagger-and-step algorithm that 
computes points close to the saddle that escapes in a time 
`T(x_n) > Tm`. The function `T` represents the escape time 
from a region defined by the user (see the argument 
`isinside`).

Given the dynamical mapping `F`, if the iteration `x_{n+1} = 
F(x_n)` respects the condition `T(x_{n+1}) > Tm` we accept 
this next point, this is the _step_ part of the method. If 
not, the method search randomly the next point in a 
neighborhood following a given probability distribution, this 
is the _stagger_ part. This part sometimes fails to find a new 
candidate and a new starting point of the trajectory is chosen 
within the defined region. See the keyword argument 
`stagger_mode` for the different available methods.   

The method produces a pseudo-trajectory of `N` points δ-close
to the stable manifold of the chaotic saddle. 

## Keyword arguments
* `δ = 1e-10`: it is a small number constraining the random
  search around a particular point. The interpretation of this 
  number will depend on the distribution chosen for the 
  generation (see `stagger_mode`).

* `Tm = 30`: The minimum number of iterations of `ds` before 
  the trajectory escapes from the bounding box defined by
  `isinside`. 

* `max_steps = 10^5`: The search for a new candidate point may 
  fail at some point. If the search fails after `max_steps`, 
  a new initial point is set and the method starts from a new
  point 

* `stagger_mode = :exp`: There are several ways to produce 
  candidate points `x` that have to fulfill the condition 
  `T(x) > Tm`. The available methods are: 

    * `:exp`: An candidate following an truncated exponential 
      distribution in a random direction `u` around the current
      `x` such that `x_c = x + u*r`. `r = 10^-s` with `s` taken
      from a uniform distribution in [-15, δ]. This mode fails
      often but stills manage to provide long enough stretch of
      trajectories. 

    * `:unif`: The next candidate is `x_c = x + u*r` with `r` 
      taken from a uniform distribution [0,δ]. 
    * `:adaptive`: The next candidate is `x_c = x + u*r` with 
      `r` drawn from a gaussian distribution with variance  δ.
      The variance changes according to a free parameter `f`  
      such that `δ = δ/f` if no candidate is found and `δ = δ*f`
      when it succeeds.    
* `f = 1.1`: It is the free parameter for the adaptive stagger
  method. 

[^Sweet2001]: D. Sweet, *et al.*, Phys. Rev. Lett. **86**, pp 2261  (2001)
[^Sala2016]: M. Sala, *et al.*, Chaos **26**, pp 123124 (2016)
"""
function stagger_and_step!(x0 ,ds, N, isinside; δ = 1e-10, Tm  = 30, 
    f = 1.1, max_steps = Int(1e5), stagger_mode = :exp)

    xi = stagger_trajectory!(x0, ds, isinside; δ = 1., Tm, stagger_mode = :exp, max_steps) 
    v = Vector{Vector{Float64}}(undef,N)
    v[1] = xi
@showprogress   for n in 1:N
        if escape_time!(xi, ds, isinside) > Tm
            set_state!(ds, xi)
        else
            xp, Tp = get_stagger!(xi, ds, δ, Tm, isinside; stagger_mode, max_steps, f)
            # The stagger step may fail. We reinitiate the algorithm from a new initial condition.
            if Tp < 0
                xp = stagger_trajectory!(x0, ds, isinside; δ = 1., Tm, stagger_mode = :exp, max_steps, f) 
                δ = 0.1
            end
            set_state!(ds,xp)
        end 
        step!(ds)
        xi = get_state(ds)
        v[n] = xi
    end
    return v
end


    
function escape_time!(x0, ds, isinside) 
    x = deepcopy(x0) 
    set_state!(ds,x)
    ds.t = 1
    k = 1; max_step = 10000;
    while isinside(x) 
        k > max_step && break
        step!(ds)
        x = get_state(ds)
        k += 1
    end
    return ds.t
end

function rand_u(δ, n; stagger_mode = :exp)
    if stagger_mode == :exp 
        a = -log10(δ)
        s = (15-a)*rand() + a
        u = (rand(n).- 0.5)
        u = u/norm(u)
        return u*10^-s
    elseif stagger_mode == :unif
        s = δ*rand()
        u = (rand(n).- 0.5)
        u = u/norm(u)
        return u*s
    elseif stagger_mode == :adaptive
        s = δ*randn()
        u = (rand(n).- 0.5)
        u = u/norm(u)
        return u*s
    end
end

# This function searches a new candidate in a neighborhood of x0 
# with a random search depending on some distribution. 
# If the search fails it returns a negative time.
function get_stagger!(x0, ds, δ, Tm, isinside; max_steps = Int(1e6), f = 1.1, stagger_mode = :exp, verbose = false)

    Tp = 0; xp = zeros(length(x0)); k = 1; 
    T0 = escape_time!(x0, ds, isinside)
    if !isinside(x0)
        error("x0 must be in grid")
    end
    while Tp ≤ Tm 
        xp = x0 .+ rand_u(δ,length(x0); stagger_mode)

        if k > max_steps 
           if verbose 
           @warn "Stagger search fails, δ is too small or T is too large. 
                We reinitiate the algorithm
           "
           end
           return 0,-1
        end
        Tp = escape_time!(xp, ds, isinside)
        if stagger_mode == :adaptive
            # We adapt the variance of the search
            # if the alg. can't find a candidate
            if Tp < T0
                δ = δ/f
            elseif Tp == T0
                δ = δ*f
            end
            if Tp == Tm
                # The adaptive alg. accepts T == Tp
                return xp, Tp
            end
        end
        k = k + 1
    end
    return xp, Tp
end

