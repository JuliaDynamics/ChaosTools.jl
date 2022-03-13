#####################################################################################
# Type definition and documentation
#####################################################################################
# TODO: Move here the docstring of `basins_of_attraction`. THere, say that
# the "supervised" version actually just uses `AttractorsViaProximity`.
"""
    AttractorsViaRecurrences(ds::GeneralizedDynamicalSystem, grid::Tuple; kwargs...)
Map initial conditions to attractors by identifying attractors on the fly based on
recurrences in the state space, as outlined by Datseris & Wagemakers[^Datseris2022].
Works for any case encapsulated by [`GeneralizedDynamicalSystem`](@ref).

`grid` is a tuple of ranges partitioning the state space so that a finite state
machine can operate on top of it. For example
`grid = (xg, yg)` where `xg = yg = range(-5, 5; length = 100)` for a two-dimensional
system. The grid has to be the same dimensionality as the state space, use a
[`projected_integrator`](@ref) if you want to search for attractors in a lower
dimensional subspace.

## Keyword Arguments
* `Δt`: Approximate time step of the integrator, which is `1` for discrete systems.
  For continuous systems, an automatic value is calculated using
  [`automatic_Δt_basins`](@ref).
* `diffeq = NamedTuple()`: Keyword arguments propagated to [`integrator`](@ref). Only
  valid for `ContinuousDynamicalSystem`. It is recommended to choose high accuracy
  solvers for this application, e.g. `diffeq = (alg=Vern9(), reltol=1e-9, abstol=1e-9)`.
* `mx_chk_att = 2`: A parameter that sets the maximum checks of consecutives hits of
  an attractor before deciding the basin of the initial condition.
* `mx_chk_hit_bas = 10`: Maximum check of consecutive visits of the same basin of
  attraction. This number can be increased for higher accuracy.
* `mx_chk_fnd_att = 100`: Maximum check of unnumbered cell before considering we have
  an attractor. This number can be increased for higher accuracy.
* `mx_chk_loc_att = 100`: Maximum check of consecutive cells marked as an attractor
  before considering that we have all the available pieces of the attractor.
* `mx_chk_lost = 20`: Maximum check of iterations outside the defined grid before we
  consider the orbit lost outside. This number can be increased for higher accuracy.
* `horizon_limit = 1e6`: If the norm of the integrator state reaches this
  limit we consider that the orbit diverges.

## Description
An initial condition given to an instance of `AttractorsViaRecurrences` is iterated
based on the integrator corresponding to `ds`. A finite state machine (FSM) follows the
trajectory in the state space, and constantly maps it to the given `grid`. An array,
internally called "basins", stores the state of the FSM on the grid, according to the
indexing system described in [^Datseris2022]. As the system is integrated more and more,
the information of the "basins" becomes richer and richer with more identified attractors
or with grid cells that belong to basins of already found attractors.

The iteration of a given initial condition continues until one of the following happens:
1. The trajectory hits `mx_chk_fnd_att` times in a row grid cells previously visited:
   it is considered that an attractor is found and is labelled with a new number.
1. The trajectory hits an already identified attractor `mx_chk_att` consecutive times:
   the initial condition is numbered with the attractor's number.
1. The trajectory hits a known basin `mx_chk_hit_bas` times in a row: the initial condition
   belongs to that basin and is numbered accordingly.
1. The trajectory spends `mx_chk_lost` steps outside the defined grid or the norm
   of the integrator state becomes > than `horizon_limit`: the initial
   condition is set to -1.

[^Datseris2022]:
    G. Datseris and A. Wagemakers, *Effortless estimation of basins of attraction*,
    [Chaos 32, 023104 (2022)](https://doi.org/10.1063/5.0076568)
"""
struct AttractorsViaRecurrences{I, B, K} <: AttractorMapper
    integ::I
    bsn_nfo::B
    kwargs::K
end

function AttractorsViaRecurrences(ds::GeneralizedDynamicalSystem, grid;
        Δt=nothing, diffeq = NamedTuple(), kwargs...
    )
    bsn_nfo, integ = basininfo_and_integ(ds, grid, Δt, diffeq)
    return AttractorsViaRecurrences(integ, bsn_nfo, kwargs)
end

function (mapper::AttractorsViaRecurrences)(u0)
    # Low level code of `basins_of_attraction` function
    lab = get_label_ic!(mapper.bsn_nfo, mapper.integ, u0; mapper.kwargs...)
    # Transform to integers indexing from odd-even indexing
    return iseven(lab) ? (lab ÷ 2) : (lab - 1) ÷ 2
end

function Base.show(io::IO, mapper::AttractorsViaRecurrences)
    ps = generic_mapper_print(io, mapper)
    println(io, rpad(" attractors: ", ps), mapper.bsn_nfo.attractors)
    println(io, rpad(" grid_maxima: ", ps), mapper.bsn_nfo.grid_maxima)
    println(io, rpad(" grid_steps: ", ps), mapper.bsn_nfo.grid_steps)
    println(io, rpad(" grid_minima: ", ps), mapper.bsn_nfo.grid_minima)
    return
end



#####################################################################################
# Definition of `BasinInfo` and initialization
#####################################################################################
# TODO: the functionality of proximty must be completely removed from `BasinsInfo`.
# A loop calling `AttractorsViaProximity` will be called in `basins_of_attraction`.

# TODO: Decide, for projected system, do we save the "full state" attractor,
# or the projected version...?
mutable struct BasinsInfo{B, IF, D, T, Q}
    basins::Array{Int16, B}
    grid_steps::SVector{B, Float64}
    grid_maxima::SVector{B, Float64}
    grid_minima::SVector{B, Float64}
    iter_f!::IF
    state::Symbol
    current_att_label::Int
    visited_cell::Int
    consecutive_match::Int
    consecutive_lost::Int
    prev_label::Int
    attractors::Dict{Int16, Dataset{D, T}}
    visited_list::Q
end

function basininfo_and_integ(ds::GeneralizedDynamicalSystem, grid, Δt, diffeq)
    integ = integrator(ds; diffeq)
    isdiscrete = isdiscretetime(integ)
    Δt = isnothing(Δt) ? automatic_Δt_basins(integ, grid) : Δt
    integ = integrator(ds; diffeq)
    iter_f! = if (isdiscrete && Δt == 1)
        (integ) -> step!(integ)
    else
        (integ) -> step!(integ, Δt)
    end
    bsn_nfo = init_bsn_nfo(grid, integ, iter_f!)
    return bsn_nfo, integ
end

function init_bsn_nfo(grid::Tuple, integ, iter_f!::Function)
    D = length(grid)
    grid_steps = step.(grid)
    grid_maxima = maximum.(grid)
    grid_minima = minimum.(grid)
    bsn_nfo = BasinsInfo(
        zeros(Int16, map(length, grid)),
        SVector{D, Float64}(grid_steps),
        SVector{D, Float64}(grid_maxima),
        SVector{D, Float64}(grid_minima),
        iter_f!,
        :att_search,
        2,4,0,1,0,
        Dict{Int16,Dataset{D, eltype(get_state(integ))}}(),
        Vector{CartesianIndex{D}}(),
    )
    reset_basins_counters!(bsn_nfo)
    return bsn_nfo
end


using LinearAlgebra


"""
    automatic_Δt_basins(integ, grid; N = 5000) → Δt
Calculate an optimal `Δt` value for [`basins_of_attraction`](@ref).
This is done by evaluating the dynamic rule `f` (vector field) at `N` randomly chosen
points of the grid. The average `f` is then compared with the diagonal length of a grid
cell and their ratio provides `Δt`.

Notice that `Δt` should not be too small which happens typically if the grid resolution
is high. It is okay for [`basins_of_attraction`](@ref) if the trajectory skips a few cells.
But if `Δt` is too small the default values for all other keywords such
as `mx_chk_hit_bas` need to be increased drastically.

Also, `Δt` that is smaller than the internal step size of the integrator will cause
a performance drop.
"""
function automatic_Δt_basins(integ, grid; N = 5000)
    isdiscretetime(integ) && return 1
    steps = step.(grid)
    s = sqrt(sum(x^2 for x in steps)) # diagonal length of a cell
    indices = CartesianIndices(length.(grid))
    random_points = [generate_ic_on_grid(grid, ind) for ind in rand(indices, N)]
    dudt = 0.0
    udummy = copy(get_state(integ))
    for p in random_points
        reinit!(integ, p)
        # TODO:  Fix this
        deriv = if get_state(integ) isa SVector
            integ.f(integ.u, integ.p, 0.0)
        else
            integ.f(udummy, integ.u, integ.p, 0.0)
            udummy
        end
        dudt += norm(deriv)
    end
    Δt = 10*s*N/dudt
    @info "Automatic Δt estimation yielded Δt = $(Δt)"
    return Δt
end


#####################################################################################
# Implementation of the Finite State Machine (low level code)
#####################################################################################
"""
    get_label_ic!(bsn_nfo::BasinsInfo, integ, u0; kwargs...) -> ic_label
This function returns the number that matches the initial condition `u0` to an attractor.
`u0` must be of the same dimension as the grid used in [`ic_labelling`](@ref).
`bsn_nfo` and `integ` are generated by the function [`ic_labelling`](@ref).

Notice the numbering system `cell_label` is as in `_identify_basin_of_cell!`.
"""
function get_label_ic!(bsn_nfo::BasinsInfo, integ, u0; kwargs...)
    # This routine identifies the attractor using the previously defined basin.
    # reinitialize integrator
    reinit!(integ, u0)
    reset_basins_counters!(bsn_nfo)
    cell_label = 0

    while cell_label == 0
        bsn_nfo.iter_f!(integ)
        new_y = get_state(integ)
        n = basin_cell_index(new_y, bsn_nfo)
        u_att = get_state(integ) # in case we need the full state to save the attractor
        cell_label = _identify_basin_of_cell!(bsn_nfo, n, u_att; kwargs...)
    end
    return cell_label
end


"""
Main procedure. Directly implements the algorithm of Datseris & Wagemakers 2021,
see the flowchart (Figure 2).

The basins and attractors are coded in the array with odd numbers for the basins and
even numbers for the attractors. The attractor `2n` has the corresponding basin `2n+1`.
This codification is changed when the basins and attractors are returned to the user.
Diverging trajectories and the trajectories staying outside the grid are coded with -1.

The label `1` (initial value) outlined in the paper is `0` here instead.
"""
function _identify_basin_of_cell!(
        bsn_nfo::BasinsInfo, n::CartesianIndex, u_full_state;
        mx_chk_att = 2, mx_chk_hit_bas = 10, mx_chk_fnd_att = 100, mx_chk_loc_att = 100,
        horizon_limit = 1e6, mx_chk_lost = 20,
        show_progress = true, # show_progress only used when finding new attractor.
    )

    #if n[1]==-1 means we are outside the grid
    ic_label = (n[1]==-1  || isnan(u_full_state[1])) ? -1 : bsn_nfo.basins[n]

    check_next_state!(bsn_nfo, ic_label)

    if bsn_nfo.state == :att_hit
        if ic_label == bsn_nfo.prev_label
             bsn_nfo.consecutive_match += 1
        end
        if bsn_nfo.consecutive_match ≥ mx_chk_att
            # Wait if we hit the attractor a mx_chk_att times in a row just
            # to check if it is not a nearby trajectory
            hit_att = ic_label + 1
            relabel_visited_cell!(bsn_nfo, bsn_nfo.visited_cell, 0)
            reset_basins_counters!(bsn_nfo)
            return hit_att
         end
         bsn_nfo.prev_label = ic_label
         return 0
    end

    if bsn_nfo.state == :att_search
        if ic_label == 0
            # unlabeled box, label it with current odd label and reset counter
            bsn_nfo.basins[n] = bsn_nfo.visited_cell
            push!(bsn_nfo.visited_list,n) # keep track of visited cells
            bsn_nfo.consecutive_match = 1
        elseif ic_label == bsn_nfo.visited_cell
            # hit a previously visited box with the current label, possible attractor?
            bsn_nfo.consecutive_match += 1
        end

        if bsn_nfo.consecutive_match >= mx_chk_fnd_att
            bsn_nfo.basins[n] = bsn_nfo.current_att_label
            store_attractor!(bsn_nfo, u_full_state, show_progress)
            bsn_nfo.state = :att_found
            bsn_nfo.consecutive_match = 1
        end
        bsn_nfo.prev_label = ic_label
        return 0
    end

    if bsn_nfo.state == :att_found
        if ic_label == 0 || ic_label == bsn_nfo.visited_cell
            # Maybe chaotic attractor, perodic or long recursion.
            # label this box as part of an attractor
            bsn_nfo.basins[n] = bsn_nfo.current_att_label
            bsn_nfo.consecutive_match = 1
            store_attractor!(bsn_nfo, u_full_state, show_progress)
        elseif iseven(ic_label) && (bsn_nfo.consecutive_match <  mx_chk_loc_att)
            # We make sure we hit the attractor another mx_chk_loc_att consecutive times
            # just to be sure that we have the complete attractor
            bsn_nfo.consecutive_match += 1
        elseif iseven(ic_label) && bsn_nfo.consecutive_match >= mx_chk_loc_att
            # We have checked the presence of an attractor: tidy up everything
            # and get a new cell
            relabel_visited_cell!(bsn_nfo, bsn_nfo.visited_cell, 0)
            # pick the next label for labeling the basin.
            bsn_nfo.visited_cell += 2
            bsn_nfo.current_att_label += 2
            reset_basins_counters!(bsn_nfo)
            return ic_label + 1;
        end
        return 0
    end

    if bsn_nfo.state == :bas_hit
        # hit a labeled basin point of the wrong basin, happens all the time,
        # we check if it happens mx_chk_hit_bas times in a row or if it happens
        # N times along the trajectory whether to decide if it is another basin.
        if bsn_nfo.prev_label == ic_label
            bsn_nfo.consecutive_match += 1
        else
            bsn_nfo.consecutive_match = 1
        end
        if  bsn_nfo.consecutive_match > mx_chk_hit_bas
            relabel_visited_cell!(bsn_nfo, bsn_nfo.visited_cell, 0)
            reset_basins_counters!(bsn_nfo)
            return ic_label
        end
        bsn_nfo.prev_label = ic_label
        return 0
    end

    if bsn_nfo.state == :lost
        bsn_nfo.consecutive_lost += 1
        if bsn_nfo.consecutive_lost > mx_chk_lost || norm(u_full_state) > horizon_limit
            relabel_visited_cell!(bsn_nfo, bsn_nfo.visited_cell, 0)
            reset_basins_counters!(bsn_nfo)
            # problematic IC : diverges or wanders outside the defined grid
            return -1
        end
        bsn_nfo.prev_label = ic_label
        return 0
    end
end

function store_attractor!(bsn_nfo::BasinsInfo{B, IF, D, T, Q},
    u_full_state, show_progress = true) where {B, IF, D, T, Q}
    # bsn_nfo.current_att_label is the number of the attractor multiplied by two
    attractor_id = bsn_nfo.current_att_label ÷ 2
    V = SVector{D, T}
    if haskey(bsn_nfo.attractors, attractor_id)
        push!(bsn_nfo.attractors[attractor_id], V(u_full_state))
    else
        # initialize container for new attractor
        bsn_nfo.attractors[attractor_id] = Dataset([V(u_full_state)])
        if show_progress
            @info "AttractorsViaRecurrences found new attractor with id: $(attractor_id)"
        end
    end
end

function relabel_visited_cell!(bsn_nfo::BasinsInfo, old_label, new_label)
    while !isempty(bsn_nfo.visited_list)
        ind = pop!(bsn_nfo.visited_list)
        if bsn_nfo.basins[ind] == old_label
            bsn_nfo.basins[ind] = new_label
        end
    end
end

function basin_cell_index(y_grid_state, bsn_nfo::BasinsInfo{B}) where {B}
    iswithingrid = true
    @inbounds for i in 1:length(bsn_nfo.grid_minima)
        if !(bsn_nfo.grid_minima[i] ≤ y_grid_state[i] ≤ bsn_nfo.grid_maxima[i])
            iswithingrid = false
            break
        end
    end
    if iswithingrid
        # Snap point to grid
        ind = @. round(Int, (y_grid_state - bsn_nfo.grid_minima)/bsn_nfo.grid_steps) + 1
        return CartesianIndex{B}(ind...)
    else
        return CartesianIndex{B}(-1)
    end
end

function reset_basins_counters!(bsn_nfo::BasinsInfo)
    bsn_nfo.consecutive_match = 0
    bsn_nfo.consecutive_lost = 0
    bsn_nfo.prev_label = 0
    bsn_nfo.state = :att_search
end

function check_next_state!(bsn_nfo, ic_label)
    next_state = :undef
    current_state = bsn_nfo.state
    if current_state == :att_found
        # this is a terminal state, once reached you don't get out
        return
    end

    if ic_label == 0 || ic_label == bsn_nfo.visited_cell
        # unlabeled box or previously visited box with the current label
        next_state = :att_search
    elseif iseven(ic_label)
        # hit an attractor box
        next_state = :att_hit
    elseif ic_label == -1
        # out of the grid we do not reset the counter of other state
        # since the trajectory can follow an attractor that spans outside the grid
        bsn_nfo.state = :lost
        return
    elseif isodd(ic_label)
        # hit an basin box
        next_state = :bas_hit
    end

    if next_state != current_state
        # reset counter except in lost state (the counter freezes in this case)
        if current_state == :lost
            bsn_nfo.consecutive_lost = 1
        else
            bsn_nfo.consecutive_match = 1
        end
    end
    bsn_nfo.state = next_state
end
