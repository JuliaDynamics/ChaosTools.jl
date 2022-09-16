# Docstrings, includes, and exports for mean return time functionality
export exit_entry_times, transit_return, mean_return_times

"""
    mean_return_times(ds::DynamicalSystem, u₀, εs, T; kwargs...) → τ, c
Return the mean return times `τ`, as well as the amount of returns `c`, for
subsets of the state space of `ds` defined by `u₀, εs`.
The `ds` is evolved for a maximum of `T` time.
This function behaves similarly to [`exit_entry_times`](@ref) and thus see that one for
the meaning of `u₀` and `εs`.

This function supports both discrete and continuous systems, however the optimizations
done in discrete systems (where all nested `ε`-sets are checked at the same time),
are not done here yet, which leads to disproportionally lower performance since
each `ε`-related set is checked individually from start.

Continuous systems allow for the following keywords:

* `i=10` How many points to interpolate the trajectory in-between steps to find
  candidate crossing regions.
* `dmin` If the trajectory is at least `dmin` distance away from `u0`,
  the algorithm that checks for crossings of the `ε`-set is not initiated.
  By default obtains the a value 4 times as large as the radius of the maximum ε-set.
* `diffeq` is a `NamedTuple` (or `Dict`) of keyword arguments propagated into
  `init` of DifferentialEquations.jl.
  See [`trajectory`](@ref) for examples. Only valid for continuous systems.

For continuous systems `T, i, dmin` can be vectors with same size as `εs`, to help increase
accuracy of small `ε`.
"""
function mean_return_times end


"""
    exit_entry_times(ds, u₀, εs, T; diffeq = NamedTuple()) → exits, entries
Collect exit and entry times for a ball or box centered at `u₀` with radii `εs` (see below),
in the state space of the given discrete dynamical system (function not yet available
for continuous systems).
Return the exit and (re-)entry return times to the set(s), where each of these is a vector
containing all collected times for the respective `ε`-radius set, for `ε ∈ εs`.

Use `transit_return(exits, entries)` to transform the output into transit and return
times, and see also [`mean_return_times`](@ref) for both continuous and discrete systems.

## Description
Transit time statistics are important for the transport properties of dynamical systems[^Meiss1997]
and can even be connected with the fractal dimension of chaotic sets[^Boev2014].

The current algorithm collects exit and re-entry times to given sets in the state space,
which are centered at `u₀` (**algorithm always starts at `u₀`** and the initial state of
`ds` is irrelevant). `εs` is always a `Vector`.

The sets around `u₀` are nested hyper-spheres of radius `ε ∈ εs`, if each entry of
`εs` is a real number. The sets can also be
hyper-rectangles (boxes), if each entry of `εs` is a vector itself.
Then, the `i`-th box is defined by the space covered by `u0 .± εs[i]` (thus the actual
box size is `2εs[i]`!).

The reason to input multiple `εs` at once is purely for performance.

For discrete systems, exit time is recorded immediatelly after exitting of the set, and
re-entry is recorded immediatelly on re-entry. This means that if an orbit needs
1 step to leave the set and then it re-enters immediatelly on the next step, the return time
is 1. For continuous systems high-order
interpolation is done to accurately record the time of exactly crossing the `ε`-ball/box.

[^Meiss1997]:
    Meiss, J. D. *Average exit time for volume-preserving maps*,
    [Chaos (1997)](https://doi.org/10.1063/1.166245)

[^Boev2014]:
    Boev, Vadivasova, & Anishchenko, *Poincaré recurrence statistics as an indicator of
    chaos synchronization*, [Chaos (2014)](https://doi.org/10.1063/1.4873721)
"""
function exit_entry_times end


"""
    transit_return(exits, entries) → transit, return
Convert the output of [`exit_entry_times`](@ref) to the transit and return times.
"""
function transit_return(exits, entries)
    # the main reason this function exists is because entry times can be one less
    # than exit times. (otherwise you could just directly subtract)
    returns = [en .- view(ex, 1:length(en)) for (en, ex) in zip(entries, exits)]
    transits = similar(entries)
    for (j, (en, ex)) in enumerate(zip(entries, exits))
        M, N = length(en), length(ex)
        enr, exr = M == N ? (1:N-1, 2:N) : (1:M, 2:N)
        transits[j] = view(ex, exr) .- view(en, enr)
    end
    return transits, returns
end

include("mrt_distances_utils.jl")
include("mrt_discrete.jl")
include("mrt_continuous.jl")