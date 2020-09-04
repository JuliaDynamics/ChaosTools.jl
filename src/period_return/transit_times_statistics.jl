using LinearAlgebra
using Roots
using Distances

# Shortcuts (already defined)
# MDI = DynamicalSystemsBase.MinimalDiscreteIntegrator
# DEI = DynamicalSystemsBase.DiffEqBase.DEIntegrator

# TODO: Add example with rectangle, perhaps not clear

export transit_time_statistics, transit_return

"""
    transit_time_statistics(ds::DynamicalSystem, u₀, εs, T; diffeq...) → exits, entries
Collect transit time statistics for a ball/box centered at `u₀` with radii `εs` (see below),
in the state space of the given dynamical system.
Return the exit and (re-)entry return times to the set(s), where each of these is a vector
containing all collected times for the respective `ε`-radius set, for `ε ∈ εs`.

Use `transit_return(exits, entries)` to transform the result to transit and return time
statistics instead.

## Keywords
* `diffeq...`: All keywords are propagated to the integrators of DifferentialEquations.jl
  for continuous systems (discrete have no keywords).

## Description
Transit time statistics are important for the transport properties of dynamical systems[^Meiss1997]
and can even be connected with the fractal dimension of chaotic sets[^Boev2014].

The current algorithm collects exit and re-entry times to given sets in the state space,
which are centered at `u₀` (**algorithm always starts at `u₀`** and the initial state of
`ds` is irrelevant). `εs` is always a `Vector`.

The sets around `u₀` are either nested hyper-spheres of radius `ε ∈ εs`, if each entry of
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

[^Meiss1997]: Meiss, J. D. *Average exit time for volume-preserving maps*, Chaos (1997)](https://doi.org/10.1063/1.166245)

[^Boev2014]: Boev, Vadivasova, & Anishchenko, *Poincaré recurrence statistics as an indicator of chaos synchronization*, Chaos (2014)](https://doi.org/10.1063/1.4873721)
"""
function transit_time_statistics(ds::DynamicalSystem, u0, εs, T; diffeq...)
    check_εs_sorting(εs, length(u0))
    integ = integrator(ds, u0; diffeq...)
    transit_time_statistics(integ, u0, εs, T)
end

"""
    transit_return(exits, entries) → transit, return
Convert the output of [`return_time_statistics`](@ref) to the transit and return times.
"""
function transit_return(exits, entries)
    returns = [en .- view(ex, 1:length(en)) for (en, ex) in zip(entries, exits)]
    transits = similar(entries)
    for (j, (en, ex)) in enumerate(zip(entries, exits))
        M, N = length(en), length(ex)
        enr, exr = M == N ? (1:N-1, 2:N) : (1:M, 2:N)
        transits[j] = view(ex, exr) .- view(en, enr)
    end
    return transits, returns
end

##########################################################################################
# ε-distances
##########################################################################################
function check_εs_sorting(εs, L)
    correct = if εs[1] isa Real
        issorted(εs; rev = true)
    elseif εs[1] isa AbstractVector
        @assert all(e -> length(e) == L, εs) "Boxes must have same dimension as state space!"
        issorted(εs; by = maximum, rev = true)
    end
    if !correct
        throw(ArgumentError("`εs` must be sorted from largest to smallest ball/box size."))
    end
    return correct
end

# Support both types of sets: balls and boxes
"Return `true` if state is outside ε-ball"
function isoutside(u, u0, ε::AbstractVector)
    @inbounds for i in 1:length(u)
        abs(u[i] - u0[i]) > ε[i] && return true
    end
    return false
end
isoutside(u, u0, ε::Real) = euclidean(u, u0) > ε

"Return the **signed** distance of state to ε-ball (negative means inside ball)"
function εdistance(u, u0, ε::AbstractVector)
    m = eltype(u)(-Inf)
    @inbounds for i in 1:length(u)
        m2 = abs(u[i] - u0[i]) - ε[i]
        m2 > m && (m = m2)
    end
    return m
end
εdistance(u, u0, ε::Real) = euclidean(u, u0) - ε

##########################################################################################
# Discrete systems
##########################################################################################
function transit_time_statistics(integ::MDI, u0, εs, T)
    E = length(εs); check_εs_sorting(εs, length(u0))
    pre_outside = fill(false, length(εs)) # `true` if outside the ball. Previous step
    cur_outside = copy(pre_outside)       # current step.
    exits = [typeof(integ.t)[] for _ in 1:length(εs)]
    entries = [typeof(integ.t)[] for _ in 1:length(εs)]

    while integ.t < T
        step!(integ)

        # here i gives the index of the largest ε-ball that the trajectory is out of.
        # It is guaranteed that the trajectory is thus outside all other boxes
        i = first_outside_index(integ, u0, εs, E) # TODO: Continuous version
        cur_outside[i:end] .= true
        cur_outside[1:i-1] .= false

        update_exit_times!(exits, i, pre_outside, cur_outside, integ)
        update_entry_times!(entries, i, pre_outside, cur_outside, integ)
        pre_outside .= cur_outside
    end
    return_times = [en .- view(ex, 1:length(en)) for (en, ex) in zip(entries, exits)]
    return exits, entries
end

function first_outside_index(integ::MDI, u0, εs, E)::Int
    i = findfirst(e -> isoutside(integ.u, u0, e), εs)
    return isnothing(i) ? E+1 : i
end

function update_exit_times!(exits, i, pre_outside, cur_outside, integ::MDI)
    @inbounds for j in i:length(pre_outside)
        cur_outside[j] && !pre_outside[j] && push!(exits[j], integ.t)
    end
end

function update_entry_times!(entries, i, pre_outside, cur_outside, integ::MDI)
    # TODO: Can I use `i` here?
    @inbounds for j in 1:length(pre_outside)
        pre_outside[j] && !cur_outside[j] && push!(entries[j], integ.t)
    end
end


##########################################################################################
# Continuous
##########################################################################################
