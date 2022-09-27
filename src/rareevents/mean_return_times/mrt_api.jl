# Docstrings, includes, and exports for mean return time functionality
export exit_entry_times, transit_return_times, mean_return_times, first_return_time
export CrossingLinearIntersection, CrossingAccurateInterpolation

"""
    exit_entry_times(ds::DynamicalSystem, u₀, εs, T; kwargs...) → exits, entries
Collect exit and entry times for balls or boxes centered at `u₀` with radii `εs`,
in the state space of the given dynamical system (discrete or continuous).
Return the exit and (re-)entry return times to the set(s), where each of these is a vector
containing all collected times for the respective `ε`-radius set, for `ε ∈ εs`.
The dynamical system is evolved up to `T` total time.

Use `transit_return_times(exits, entries)` to transform the output into transit and return
times, and see also [`mean_return_times`](@ref).

The keyword `show_progress` displays a progress bar. It is `false` for discrete and
`true` for continuous systems by default.

## Description
Transit and return time statistics are important for the transport properties of dynamical
systems[^Meiss1997] and can be connected with fractal dimensions of chaotic sets[^Boev2014].

The current algorithm collects exit and re-entry times to given sets in the state space,
which are centered at the state `u₀`. **The system evolution always starts from `u₀`**
and the initial state of `ds` is irrelevant. `εs` is always a `Vector`.

### Specification of sets to return to
If each entry of `εs` is a real number, then sets around `u₀` are nested hyper-spheres
of radius `ε ∈ εs`. The sets can also be hyper-rectangles (boxes), if each entry of `εs`
is a vector itself.
Then, the `i`-th box is defined by the space covered by `u0 .± εs[i]` (thus the actual
box size is `2εs[i]`!).
In the future, state space sets will be specified more conveniently and
a single argument `sets` will be given instead of `u₀, εs`.

The reason to input multiple `εs` at once is purely for performance optimization
(much faster than doing each `ε` individually).

### Discrete systems

For discrete systems, exit time is recorded immediatelly after exitting of the set, and
re-entry is recorded immediatelly on re-entry. This means that if an orbit needs
1 step to leave the set and then it re-enters immediatelly on the next step,
the return time is 1.

### Continuous systems
For continuous systems, a steppable integrator supporting interpolation is used.
The way to specify how to estimate exit and entry times is via the keyword `crossing_method`
whose values can be:
1. `CrossingLinearIntersection()`: Linear interpolation is used between integrator steps
   and the intersection between lines and spheres is used to find the crossing times.
2. `CrossingAccurateInterpolation(; abstol=1e-12, reltol=1e-6)`: Extremely accurate high
   order interpolation is used between integrator steps. First, a minimization with Optim.jl
   finds the minimum distance of the trajectory to the set center. Then, Roots.jl is used
   to find the exact crossing point. The tolerances are given to both procedures.

Clearly, `CrossingAccurateInterpolation` is much more accurate than
`CrossingLinearIntersection`, but also much slower. However, the smaller the steps
the integrator takes (in case some very high accuracy solver is used), the closer
the linear intersection gets to the accurate version.
Benchmarks are advised for the individual specific case the algorithm is applied at,
in order to choose the best method.

The keyword `threshold_distance = Inf` provides a means to skip the interpolation check,
if the current state of the integrator is too far from the set center.
If the distance of the current state of the integrator is `threshold_distance` or more
distance away from the set center, attempts to interpolate are skipped.
By default `threshold_distance = Inf` and hence this never happens.
Typically you'd want this to be 10-100 times the distance the trajectory covers
at an average integrator step.

[^Meiss1997]:
    Meiss, J. D. *Average exit time for volume-preserving maps*,
    [Chaos (1997)](https://doi.org/10.1063/1.166245)

[^Boev2014]:
    Boev, Vadivasova, & Anishchenko, *Poincaré recurrence statistics as an indicator of
    chaos synchronization*, [Chaos (2014)](https://doi.org/10.1063/1.4873721)
"""
function exit_entry_times(ds::DynamicalSystem, u0, εs, T; diffeq = NamedTuple(), kwargs...)
    check_εs_sorting(εs, length(u0))
    # TODO: Improve the algorithm so that starting within u0 is not mandatory.
    # Useful because `u0` can often be a fixed point.
    # The logic that needs to change is first `transit_return_times` and then
    # to actually check in `exit_entry_times` if we start inside the set,
    # and thus set the `prev_outside` accordingly.
    integ = integrator(ds, u0; diffeq)
    exit_entry_times(integ, u0, εs, T; kwargs...)
end

"""
    transit_return_times(exits, entries) → transits, returns
Convert the output of [`exit_entry_times`](@ref) to the transit and return times.
The outputs here are vectors of vectors just like in [`exit_entry_times`](@ref).
"""
function transit_return_times(exits, entries)
    # the main reason this function exists is because entry times are most likely one less
    # than exit times. (otherwise you could just directly subtract)
    returns = [en .- view(ex, 1:length(en)) for (en, ex) in zip(entries, exits)]
    transits = similar(entries)
    for (j, (en, ex)) in enumerate(zip(entries, exits))
        enlen, exlen = length(en), length(ex)
        # Remember, algorithm always starts from the set center. So exit is
        # always guaranteed. So if exits are equal to entries, we need to skip first exit!
        if exlen == enlen + 1 # typical case
            transits[j] = view(ex, 2:exlen) .- en
        else
            transits[j] = view(ex, 2:exlen) .- view(en, 1:(enlen - 1))
        end
        # transits[j] = view(ex, x) .- en
    end
    return transits, returns
end

"""
    mean_return_times(ds::DynamicalSystem, u₀, εs, T; kwargs...) → τ, c
Return the mean return times `τ`, as well as the amount of returns `c`, for
subsets of the state space of `ds` defined by `u₀, εs`.
The `ds` is evolved for a maximum of `T` time.

This function is a convenience wrapper around calls to [`exit_entry_times`](@ref)
and then to [`transit_return`](@ref) and then some averaging.
Thus see [`exit_entry_times`](@ref) for the meaning of `u₀` and `εs` and further info.
"""
function mean_return_times(ds::DynamicalSystem, u0, εs, T; kwargs...)
    exits, entries = exit_entry_times(ds, u0, εs, T; kwargs...)
    transits, returns = transit_return_times(exits, entries)
    mean_return_times(returns)
end
function mean_return_times(returns::AbstractVector)
    mrt = mean.(returns)
    ret = length.(returns)
    return mrt, ret
end

"""
    first_return_time(ds::DynamicalSystem, u0, ε, T; diffeq = NamedTuple(), kwargs...) → t
Return the first return time `t` to the set centered at `u0` with radius `ε` for the given
dynamical system. Time evolution of `ds` always starts from `u0`.

This function operates on the same principles as
[`exit_entry_times`](@ref), so see that docstring for more info.
The only differences here are:
1. If the system did not return to the set within time `T`, then `NaN` is returned.
2. For continuous systems, the exact returned time is from start of time evolution,
   up to the time to get closest back to `u0`, provided that this is at least `ε`-close.
"""
function first_return_time(ds::DynamicalSystem, u0, ε, T; diffeq = NamedTuple(), kwargs...)
    check_εs_sorting([ε], length(u0))
    integ = integrator(ds, u0; diffeq)
    first_return_time(integ, u0, ε, T; kwargs...)
end

include("mrt_distances_utils.jl")
include("mrt_discrete.jl")
include("mrt_continuous.jl")