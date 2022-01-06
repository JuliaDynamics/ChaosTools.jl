import ProgressMeter
using Statistics: mean

mutable struct BasinsInfo{B, IF, RF, UF, D, T, Q, K}
    basins::Array{Int16, B}
    grid_steps::SVector{B, Float64}
    grid_maxima::SVector{B, Float64}
    grid_minima::SVector{B, Float64}
    iter_f!::IF
    complete_and_reinit!::RF
    get_projected_state::UF
    state::Symbol
    current_att_label::Int
    visited_cell::Int
    consecutive_match::Int
    consecutive_lost::Int
    prev_lab::Int
    attractors::Dict{Int16, Dataset{D, T}}
    visited::Q
    search_trees::K
end


"""
This is the low level function that creates & computes the basins of attraction, and is
agnostic of the dynamical system. `integ` is an integrator, `iter_f!` a function that
steps the integrator, `complete_and_reinit!` a function that re-inits the integrator
at a new full state, given the state on the grid.
"""
function estimate_basins!(
        grid::Tuple, integ, iter_f!::Function, complete_and_reinit!, get_projected_state::Function;
        show_progress = true, attractors = nothing, kwargs...,
    )
    bsn_nfo = init_bsn_nfo(grid, integ, iter_f!, complete_and_reinit!, get_projected_state; attractors)
    basins_computation!(bsn_nfo, grid, integ, show_progress; kwargs...)
    return bsn_nfo
end

# function for structure initialization.
function init_bsn_nfo(
        grid::Tuple, integ, iter_f!::Function, complete_and_reinit!, get_projected_state::Function;
        attractors = nothing
    )
    B = length(grid)
    D = length(get_state(integ)) # dimension of the full state space
    trees = if isnothing(attractors)
        nothing
    else
        Dict(k => searchstructure(KDTree, att, Euclidean()) for (k, att) in attractors)
    end
    grid_steps = step.(grid)
    grid_maxima = maximum.(grid)
    grid_minima = minimum.(grid)
    bsn_nfo = BasinsInfo(
        zeros(Int16, map(length, grid)),
        SVector{B, Float64}(grid_steps),
        SVector{B, Float64}(grid_maxima),
        SVector{B, Float64}(grid_minima),
        iter_f!,
        complete_and_reinit!,
        get_projected_state,
        :att_search,
        2,4,0,1,0,
        Dict{Int16,Dataset{D,eltype(get_state(integ))}}(),
        Vector{CartesianIndex{B}}(),
        trees
    )
    reset_basins_counters!(bsn_nfo)
    return bsn_nfo
end


function basins_computation!(bsn_nfo::BasinsInfo, grid::Tuple, integ, show_progress; kwargs...)
    I = CartesianIndices(bsn_nfo.basins)
    complete = false
    j = 1
    progress = ProgressMeter.Progress(
        length(bsn_nfo.basins); desc = "Basins of attraction: ", dt = 1.0
    )

    while !complete
        ind, complete, j = next_unlabeled_cell(bsn_nfo, j, I)
        show_progress && ProgressMeter.update!(progress, j)
        complete && break
        # Tentatively assign a label: odd is for basins, even for attractors.
        # First label is 2 for attractor and 3 for basins
        bsn_nfo.basins[ind] = bsn_nfo.visited_cell
        y0 = generate_ic_on_grid(grid, ind)
        bsn_nfo.basins[ind] = get_label_ic!(bsn_nfo, integ, y0; kwargs...)
    end
    # remove attractors and rescale from 1 to max number of attractors
    ind = iseven.(bsn_nfo.basins)
    bsn_nfo.basins[ind] .+= 1
    bsn_nfo.basins .= (bsn_nfo.basins .- 1) .÷ 2
    return bsn_nfo
end


function next_unlabeled_cell(bsn_nfo, j, I)
    @inbounds for k in j:length(bsn_nfo.basins)
        if bsn_nfo.basins[I[k]] == 0
            j = k
            ind = I[k]
            return ind, false, j
        end
    end
    return I[1], true, length(bsn_nfo.basins)
end


function get_label_ic!(bsn_nfo::BasinsInfo, integ, y0; kwargs...)
    # This routine identifies the attractor using the previously defined basin.
    # reinitialize integrator
    bsn_nfo.complete_and_reinit!(integ, y0)
    reset_basins_counters!(bsn_nfo)
    celllabel = 0

    while celllabel == 0
        bsn_nfo.iter_f!(integ)
        new_y = bsn_nfo.get_projected_state(integ)
        n = basin_cell_index(new_y, bsn_nfo)
        u_att = get_state(integ) # in case we need the full state to save the attractor
        celllabel = _identify_basin_of_cell!(bsn_nfo, n, u_att; kwargs...)
    end
    return celllabel
end

"""
Main procedure motivated by Nusse & Yorke for the grid cell `n`. The algorithm can be
though as a finite state machine with five states: :att_hit, :att_search, :att_found,
:bas_hit, :lost. The transition between states depends on the current number in the
cell being visited by the trajectory of the dynamical systems. When the automata switches
to a state a counter starts and one of the following action can happen depending on the internal
counter and the input number:
* The automata stays in the same state and the counter increases.
* The automata switches to another state.
* The automata has decided upon the basin of the initial condition and returns a code corresponding
  to this basin.

The basins and attractors are coded in the array with odd numbers for the basins and even numbers
for the attractors. The attractor `2n` has the corresponding basin `2n+1`. This codification
is changed when the basins and attractors are returned to the user. Diverging trajectories
and the trajectories staying outside the grid are coded with -1.
"""
function _identify_basin_of_cell!(
        bsn_nfo::BasinsInfo, n::CartesianIndex, u_full_state;
        mx_chk_att = 2, mx_chk_hit_bas = 10, mx_chk_fnd_att = 100, mx_chk_loc_att = 100,
        horizon_limit = 1e6, ε = 1e-3,
        mx_chk_lost = isnothing(bsn_nfo.search_trees) ? 20 : 1000,
    )

    #if n[1]==-1 means we are outside the grid
    nxt_clr = (n[1]==-1  || isnan(u_full_state[1])) ? -1 : bsn_nfo.basins[n]

    # search attractors directly
    if !isnothing(bsn_nfo.search_trees)
        for (k, t) in bsn_nfo.search_trees # this is a `Dict`
            idxs = isearch(t, u_full_state, WithinRange(ε))
            if !isempty(idxs)
                nxt_clr = 2*k + 1
                break
            end
        end
    end

    check_next_state!(bsn_nfo,nxt_clr)

    if bsn_nfo.state == :att_hit
        if nxt_clr == bsn_nfo.prev_lab
             bsn_nfo.consecutive_match += 1
        end
        if bsn_nfo.consecutive_match ≥ mx_chk_att
            # Wait if we hit the attractor a mx_chk_att times in a row just
            # to check if it is not a nearby trajectory
            hit_att = nxt_clr + 1
            relabel_visited_cell!(bsn_nfo, bsn_nfo.visited_cell, 0)
            reset_basins_counters!(bsn_nfo)
            return hit_att
         end
         bsn_nfo.prev_lab = nxt_clr
         return 0
    end

    if bsn_nfo.state == :att_search
        if nxt_clr == 0
            # unlabeled box, label it with current odd label and reset counter
            bsn_nfo.basins[n] = bsn_nfo.visited_cell
            push!(bsn_nfo.visited,n) # keep track of visited cells
            bsn_nfo.consecutive_match = 1
        elseif nxt_clr == bsn_nfo.visited_cell
            # hit a previously visited box with the current label, possible attractor?
            bsn_nfo.consecutive_match += 1
        end

        if bsn_nfo.consecutive_match >= mx_chk_fnd_att
            bsn_nfo.basins[n] = bsn_nfo.current_att_label
            store_attractor!(bsn_nfo, u_full_state)
            bsn_nfo.state = :att_found
            bsn_nfo.consecutive_match = 1
        end
        bsn_nfo.prev_lab = nxt_clr
        return 0
    end

    if bsn_nfo.state == :att_found
        if nxt_clr == 0 || nxt_clr == bsn_nfo.visited_cell
            # Maybe chaotic attractor, perodic or long recursion.
            # label this box as part of an attractor
            bsn_nfo.basins[n] = bsn_nfo.current_att_label
            bsn_nfo.consecutive_match = 1
            store_attractor!(bsn_nfo, u_full_state)
        elseif iseven(nxt_clr) && (bsn_nfo.consecutive_match <  mx_chk_loc_att)
            # We make sure we hit the attractor another mx_chk_loc_att consecutive times
            # just to be sure that we have the complete attractor
            bsn_nfo.consecutive_match += 1
        elseif iseven(nxt_clr) && bsn_nfo.consecutive_match >= mx_chk_loc_att
            # We have checked the presence of an attractor: tidy up everything
            # and get a new cell
            relabel_visited_cell!(bsn_nfo, bsn_nfo.visited_cell, 0)
            # pick the next label for labeling the basin.
            bsn_nfo.visited_cell += 2
            bsn_nfo.current_att_label += 2
            reset_basins_counters!(bsn_nfo)
            return nxt_clr + 1;
        end
        return 0
    end

    if bsn_nfo.state == :bas_hit
        # hit a labeled basin point of the wrong basin, happens all the time,
        # we check if it happens mx_chk_hit_bas times in a row or if it happens
        # N times along the trajectory whether to decide if it is another basin.
        if bsn_nfo.prev_lab == nxt_clr
            bsn_nfo.consecutive_match += 1
        else
            bsn_nfo.consecutive_match = 1
        end
        if  bsn_nfo.consecutive_match > mx_chk_hit_bas
            relabel_visited_cell!(bsn_nfo, bsn_nfo.visited_cell, 0)
            reset_basins_counters!(bsn_nfo)
            return nxt_clr
        end
        bsn_nfo.prev_lab = nxt_clr
        return 0
    end

    if bsn_nfo.state == :lost
        #grid_mid_point = (bsn_nfo.grid_maxima - bsn_nfo.grid_minima) ./2 + bsn_nfo.grid_minima
        bsn_nfo.consecutive_lost += 1
        if   bsn_nfo.consecutive_lost > mx_chk_lost || norm(u_full_state) > horizon_limit
            relabel_visited_cell!(bsn_nfo, bsn_nfo.visited_cell, 0)
            reset_basins_counters!(bsn_nfo)
            # problematic IC : diverges or wanders outside the defined grid
            return -1
        end
        bsn_nfo.prev_lab = nxt_clr
        return 0
    end
end

function store_attractor!(bsn_nfo::BasinsInfo{B, IF, RF, UF, D, T, Q},
    u_full_state) where {B, IF, RF, UF, D, T, Q}
    # bsn_nfo.current_att_label is the number of the attractor multiplied by two
    attractor_id = bsn_nfo.current_att_label ÷ 2
    V = SVector{D, T}
    if haskey(bsn_nfo.attractors, attractor_id)
        push!(bsn_nfo.attractors[attractor_id], V(u_full_state))
    else
        # initialize container for new attractor
        bsn_nfo.attractors[attractor_id] = Dataset([V(u_full_state)])
    end
end

function relabel_visited_cell!(bsn_nfo::BasinsInfo, old_c, new_c)
    while !isempty(bsn_nfo.visited)
        ind = pop!(bsn_nfo.visited)
        if bsn_nfo.basins[ind] == old_c
            bsn_nfo.basins[ind] = new_c
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
    bsn_nfo.prev_lab = 0
    bsn_nfo.state = :att_search
end

function check_next_state!(bsn_nfo, nxt_clr)
    next_state = :undef
    current_state = bsn_nfo.state
    if current_state == :att_found
        # this is a terminal state, once reached you don't get out
        return
    end

    if nxt_clr == 0 || nxt_clr == bsn_nfo.visited_cell
        # unlabeled box or previously visited box with the current label
        next_state = :att_search
    elseif iseven(nxt_clr)
        # hit an attractor box
        next_state = :att_hit
    elseif nxt_clr == -1
        # out of the grid we do not reset the counter of other state
        # since the trajectory can follow an attractor that spans outside the grid
        bsn_nfo.state = :lost
        return
    elseif isodd(nxt_clr)
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
