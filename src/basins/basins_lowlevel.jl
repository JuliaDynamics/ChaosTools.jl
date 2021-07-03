import ProgressMeter

mutable struct BasinInfo{B, IF, RF, UF, D, T, Q}
    basin::Array{Int16, B}
    grid_steps::SVector{B, Float64}
    grid_maxima::SVector{B, Float64}
    grid_minima::SVector{B, Float64}
    iter_f!::IF
    complete_and_reinit!::RF
    get_projected_state::UF
    state::Symbol
    current_att_color::Int
    current_bas_color::Int
    consecutive_match::Int
    consecutive_lost::Int
    prev_clr::Int
    attractors::Dict{Int16, Dataset{D, T}}
    visited::Q
end


"""
This is the low level function that creates & computes the basins of attraction, and is
agnostic of the dynamical system. `integ` is an integrator, `iter_f!` a function that
steps the integrator, `complete_and_reinit!` a function that re-inits the integrator
at a new full state, given the state on the grid.
"""
function draw_basin!(
        grid::Tuple, integ, iter_f!::Function, complete_and_reinit!, get_projected_state::Function;
        show_progress = true, kwargs...,
    )
    B = length(grid)
    D = length(get_state(integ)) # dimension of the full state space
    complete = false
    grid_steps = step.(grid)
    grid_maxima = maximum.(grid)
    grid_minima = minimum.(grid)
    bsn_nfo = BasinInfo(
        ones(Int16, map(length, grid)),
        SVector{B, Float64}(grid_steps),
        SVector{B, Float64}(grid_maxima),
        SVector{B, Float64}(grid_minima),
        iter_f!,
        complete_and_reinit!,
        get_projected_state,
        :att_search,
        2,4,0,1,1,
        Dict{Int16,Dataset{D,eltype(get_state(integ))}}(),
        Vector{CartesianIndex{B}}()
    )
    reset_basin_counters!(bsn_nfo)
    I = CartesianIndices(bsn_nfo.basin)
    j = 1
    progress = ProgressMeter.Progress(
        length(bsn_nfo.basin); desc = "Basins of attraction: ", dt = 1.0
    )

    while !complete
        ind, complete, j = next_uncolored_cell(bsn_nfo, j, I)
        show_progress && ProgressMeter.update!(progress, j)
        complete && break
        # Tentatively assign a color: odd is for basins, even for attractors.
        # First color is 2 for attractor and 3 for basins
        bsn_nfo.basin[ind] = bsn_nfo.current_bas_color
        y0 = generate_ic_on_grid(grid, ind)
        bsn_nfo.basin[ind] = get_color_point!(bsn_nfo, integ, y0; kwargs...)
    end
    # remove attractors and rescale from 1 to max nmb of attractors
    ind = iseven.(bsn_nfo.basin)
    bsn_nfo.basin[ind] .+= 1
    bsn_nfo.basin .= (bsn_nfo.basin .- 1) .÷ 2
    return bsn_nfo
end

function next_uncolored_cell(bsn_nfo, j, I)
    @inbounds for k in j:length(bsn_nfo.basin)
        if bsn_nfo.basin[I[k]] == 1
            j = k
            ind = I[k]
            return ind, false, j
        end
    end
    return I[1], true, length(bsn_nfo.basin)
end

@generated function generate_ic_on_grid(grid::NTuple{B, T}, ind) where {B, T}
    gens = [:(grid[$k][ind[$k]]) for k=1:B]
    quote
        Base.@_inline_meta
        @inbounds return SVector{$B, Float64}($(gens...))
    end
end


function get_color_point!(bsn_nfo::BasinInfo, integ, y0; kwargs...)
    # This routine identifies the attractor using the previously defined basin.
    # reinitialize integrator
    bsn_nfo.complete_and_reinit!(integ, y0)
    reset_basin_counters!(bsn_nfo)
    cellcolor = 0

    while cellcolor == 0
        bsn_nfo.iter_f!(integ)
        new_y = bsn_nfo.get_projected_state(integ)
        n = basin_cell_index(new_y, bsn_nfo)
        u_att = get_state(integ) # in case we need the full state to save the attractor
        cellcolor = _identify_basin_of_cell!(bsn_nfo, n, u_att; kwargs...)
    end
    return cellcolor
end

"""
Main procedure described by Nusse & Yorke for the grid cell `n`. The algorithm can be
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
        bsn_nfo::BasinInfo, n::CartesianIndex, u_full_state;
        mx_chk_att = 2, mx_chk_hit_bas = 10, mx_chk_fnd_att = 100, mx_chk_lost = 100,
        horizon_limit = 1e6
    )
    #if n[1]==-1 means we are outside the grid
    nxt_clr = (n[1]==-1  || isnan(u_full_state[1])) ? -1 : bsn_nfo.basin[n]
    check_next_state!(bsn_nfo,nxt_clr)

    if bsn_nfo.state == :att_hit
        if nxt_clr == bsn_nfo.prev_clr
             bsn_nfo.consecutive_match += 1
        end
        if bsn_nfo.consecutive_match ≥ mx_chk_att
            # Wait if we hit the attractor a mx_chk_att times in a row just
            # to check if it is not a nearby trajectory
            hit_att = nxt_clr + 1
            recolor_visited_cell!(bsn_nfo, bsn_nfo.current_bas_color, 1)
            reset_basin_counters!(bsn_nfo)
            return hit_att
         end
         bsn_nfo.prev_clr = nxt_clr
         return 0
    end

    if bsn_nfo.state == :att_search
        if nxt_clr == 1
            # uncolored box, color it with current odd color and reset counter
            bsn_nfo.basin[n] = bsn_nfo.current_bas_color
            push!(bsn_nfo.visited,n) # keep track of visited cells
            bsn_nfo.consecutive_match = 1
        elseif nxt_clr == bsn_nfo.current_bas_color
            # hit a previously visited box with the current color, possible attractor?
            bsn_nfo.consecutive_match += 1
        end

        if bsn_nfo.consecutive_match >= mx_chk_fnd_att
            bsn_nfo.basin[n] = bsn_nfo.current_att_color
            store_attractor!(bsn_nfo, u_full_state)
            bsn_nfo.state = :att_found
            bsn_nfo.consecutive_match = 1
        end
        bsn_nfo.prev_clr = nxt_clr
        return 0
    end

    if bsn_nfo.state == :att_found
        if nxt_clr == 1 || nxt_clr == bsn_nfo.current_bas_color
            # Maybe chaotic attractor, perodic or long recursion.
            # Color this box as part of an attractor
            bsn_nfo.basin[n] = bsn_nfo.current_att_color
            bsn_nfo.consecutive_match = 1
            store_attractor!(bsn_nfo, u_full_state)
        elseif iseven(nxt_clr) && (bsn_nfo.consecutive_match <  mx_chk_fnd_att)
            # We make sure we hit the attractor another mx_chk_fnd_att consecutive times
            # just to be sure that we have the complete attractor
            bsn_nfo.consecutive_match += 1
        elseif iseven(nxt_clr) && bsn_nfo.consecutive_match >= mx_chk_fnd_att
            # We have checked the presence of an attractor: tidy up everything
            # and get a new cell
            recolor_visited_cell!(bsn_nfo, bsn_nfo.current_bas_color, 1)
            # pick the next color for coloring the basin.
            bsn_nfo.current_bas_color += 2
            bsn_nfo.current_att_color += 2
            reset_basin_counters!(bsn_nfo)
            return nxt_clr + 1;
        end
        return 0
    end

    if bsn_nfo.state == :bas_hit
        # hit a colored basin point of the wrong basin, happens all the time,
        # we check if it happens mx_chk_hit_bas times in a row or if it happens
        # N times along the trajectory whether to decide if it is another basin.
        if bsn_nfo.prev_clr == nxt_clr
            bsn_nfo.consecutive_match += 1
        else
            bsn_nfo.consecutive_match = 1
        end
        if  bsn_nfo.consecutive_match > mx_chk_hit_bas
            recolor_visited_cell!(bsn_nfo, bsn_nfo.current_bas_color, 1)
            reset_basin_counters!(bsn_nfo)
            return nxt_clr
        end
        bsn_nfo.prev_clr = nxt_clr
        return 0
    end

    if bsn_nfo.state == :lost
        #grid_mid_point = (bsn_nfo.grid_maxima - bsn_nfo.grid_minima) ./2 + bsn_nfo.grid_minima
        bsn_nfo.consecutive_lost += 1
        if   bsn_nfo.consecutive_lost > mx_chk_lost || norm(u_full_state) > horizon_limit
            recolor_visited_cell!(bsn_nfo, bsn_nfo.current_bas_color, 1)
            reset_basin_counters!(bsn_nfo)
            # problematic IC : diverges or wanders outside the defined grid
            return -1
        end
        bsn_nfo.prev_clr = nxt_clr
        return 0
    end
end

function store_attractor!(bsn_nfo::BasinInfo{B, IF, RF, UF, D, T, Q},
    u_full_state) where {B, IF, RF, UF, D, T, Q}
    # bsn_nfo.current_att_color is the number of the attractor multiplied by two
    attractor_id = bsn_nfo.current_att_color ÷ 2
    V = SVector{D, T}
    if haskey(bsn_nfo.attractors, attractor_id)
        push!(bsn_nfo.attractors[attractor_id], V(u_full_state))
    else
        # initialize container for new attractor
        bsn_nfo.attractors[attractor_id] = Dataset([V(u_full_state)])
    end
end

function recolor_visited_cell!(bsn_nfo::BasinInfo, old_c, new_c)
    while !isempty(bsn_nfo.visited)
        ind = pop!(bsn_nfo.visited)
        if bsn_nfo.basin[ind] == old_c
            bsn_nfo.basin[ind] = new_c
        end
    end
end

function basin_cell_index(y_grid_state, bsn_nfo::BasinInfo{B}) where {B}
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

function reset_basin_counters!(bsn_nfo::BasinInfo)
    bsn_nfo.consecutive_match = 0
    bsn_nfo.consecutive_lost = 0
    bsn_nfo.prev_clr = 1
    bsn_nfo.state = :att_search
end

function check_next_state!(bsn_nfo, nxt_clr)
    next_state = :undef
    current_state = bsn_nfo.state
    if current_state == :att_found
        # this is a terminal state, once reached you don't get out
        return
    end

    if nxt_clr == 1 || nxt_clr == bsn_nfo.current_bas_color
        # uncolored box or previously visited box with the current color
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
