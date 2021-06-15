mutable struct BasinInfo{B, G, IF, RF, UF, D, T, Q}
    basin::Array{Int16, B}
    grid_steps::G
    grid_maxima::G
    grid_minima::G
    iter_f!::IF
    reinit_f!::RF
    get_grid_state::UF
    state::Symbol
    current_att_color::Int
    current_bas_color::Int
    consecutive_match::Int
    prev_clr::Int
    attractors::Dict{Int16, Dataset{D, T}}
    visited::Q
end


"""
    draw_basin!(grid, integ, iter_f!, reinit_f!; kwargs...)
This is the low level function that creates & computes the basins of attraction, and is
agnostic of the dynamical system. `integ` is an integrator, `iter_f!` a function that
steps the integrator, `reinit_f!` a function that re-inits the integrator.

The keywords are the actual algorithm-tuning keywords, expanded first time here,
and are described in the high level function.
"""
function draw_basin!(
        grid::Tuple, integ, iter_f!::Function, reinit_f!::Function, get_grid_state::Function;
        mx_chk_att = 2, mx_chk_hit_bas = 10, mx_chk_fnd_att = 100, mx_chk_lost=2000
    )
    D = length(get_state(integ)) # dimension of the full dynamical system
    B = length(grid)             # dimension of the grid, i.e. projected dynamics
    complete = false
    grid_steps = step.(grid)
    grid_maxima = maximum.(grid)
    grid_minima = minimum.(grid)
    bsn_nfo = BasinInfo(
        ones(Int16, map(length, grid)),
        SVector(grid_steps),
        SVector(grid_maxima),
        SVector(grid_minima),
        iter_f!,
        reinit_f!,
        get_grid_state,
        :att_search,
        2,4,0,1,
        Dict{Int16,Dataset{D,eltype(get_state(integ))}}(),
        Vector{CartesianIndex}()
    )
    reset_basin_counters!(bsn_nfo)
    I = CartesianIndices(bsn_nfo.basin)
    j = 1
    T = eltype(grid[1]);

    while !complete
        ind, complete, j = next_uncolored_cell(bsn_nfo, j, I)
        complete && break
        # Tentatively assign a color: odd is for basins, even for attractors.
        # First color is 2 for attractor and 3 for basins
        bsn_nfo.basin[ind] = bsn_nfo.current_bas_color
        y0 = generate_ic_on_grid(grid, ind)
        bsn_nfo.basin[ind] = get_color_point!(bsn_nfo, integ, y0, mx_chk_att, mx_chk_hit_bas, mx_chk_fnd_att, mx_chk_lost)
    end
    # remove attractors and rescale from 1 to max nmb of attractors
    ind = iseven.(bsn_nfo.basin)
    bsn_nfo.basin[ind] .+= 1
    bsn_nfo.basin = (bsn_nfo.basin .- 1) .÷ 2
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


function get_color_point!(bsn_nfo::BasinInfo, integ, y0, mx_chk_att, mx_chk_hit_bas, mx_chk_fnd_att, mx_chk_lost)
    # This routine identifies the attractor using the previously defined basin.
    # reinitialize integrator
    bsn_nfo.reinit_f!(integ, y0)
    reset_basin_counters!(bsn_nfo)
    cellcolor = inlimbo = 0

    while cellcolor == 0
        old_y = bsn_nfo.get_grid_state(integ)
        bsn_nfo.iter_f!(integ)
        new_y = bsn_nfo.get_grid_state(integ)
        n = basin_cell_index(new_y, bsn_nfo)
        u_att = get_state(integ) # in case we need the full state to save the attractor
        cellcolor = _identify_basin_of_cell!(bsn_nfo, n, u_att, mx_chk_att, mx_chk_hit_bas, mx_chk_fnd_att, mx_chk_lost)
    end
    return cellcolor
end

"""
Main procedure described by Nusse & Yorke for the grid cell `n`.
The idea is to color the grid with the current color. When an attractor box is hit
(even color), the initial condition is colored with the color of its basin (odd color).
If the trajectory hits another basin many times times in row, the IC is colored with the
same color as this basin.

# TODO: In this docstring, we should state that the numbering system is different.
(i.e. clarify the even / odd numbers, as we decided that in the high level interface,
there isn't any even/odd distinction and the attractors are numbered according to the
integers)
"""
function _identify_basin_of_cell!(
        bsn_nfo::BasinInfo, n::CartesianIndex, u,
        mx_chk_att::Int, mx_chk_hit_bas::Int, mx_chk_fnd_att::Int, mx_chk_lost::Int
    )
    #if n[1]==-1 means we are outside the grid
    nxt_clr = (n[1]==-1) ? -1 : bsn_nfo.basin[n]
    check_next_state!(bsn_nfo,nxt_clr)

    if bsn_nfo.state == :att_hit
        if nxt_clr == bsn_nfo.prev_clr
             bsn_nfo.consecutive_match += 1
        end
        if bsn_nfo.consecutive_match ≥ mx_chk_att
            # Wait if we hit the attractor a mx_chk_att times in a row just to check if it is not a nearby trajectory
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
            store_attractor!(bsn_nfo, u)
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
            store_attractor!(bsn_nfo, u)
        elseif iseven(nxt_clr) && (bsn_nfo.consecutive_match <  mx_chk_fnd_att)
            # We make sure we hit the attractor another mx_chk_fnd_att consecutive times
            # just to be sure that we have the complete attractor
            bsn_nfo.consecutive_match += 1
        elseif iseven(nxt_clr) && bsn_nfo.consecutive_match >= mx_chk_fnd_att
            # We have checked the presence of an attractor: tidy up everything
            # and get a new box
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
        #grid_maximum = maximum(abs.(bsn_nfo.grid_maxima)); # this is the largest size of the phase space
        bsn_nfo.consecutive_match += 1
        if   bsn_nfo.consecutive_match > mx_chk_lost || norm(u) == Inf
            # TODO: all numeric constants in the above line must be replaced with
            # named variables with intention-revealing name. They must also be tunable
            # as keyword arguments in `draw_basin!`.
            # TODO: Comparing the `norm` of the `new_u` is a mistake, as it assumes
            # that the grid starts from 0. We instead need to compare the `norm`
            # of the `new_u` from the center of the grid.
            recolor_visited_cell!(bsn_nfo, bsn_nfo.current_bas_color, 1)
            reset_basin_counters!(bsn_nfo)
            # problematic IC : diverges or wanders outside the defined grid
            return -1
        end
        bsn_nfo.prev_clr = nxt_clr
        return 0
    end
end

function store_attractor!(bsn_nfo::BasinInfo, u)
    # bsn_nfo.current_color is the number of the attractor multiplied by two
    if haskey(bsn_nfo.attractors , bsn_nfo.current_att_color ÷ 2)
        push!(bsn_nfo.attractors[bsn_nfo.current_att_color ÷ 2],  u) # store attractor
    else
        bsn_nfo.attractors[bsn_nfo.current_att_color ÷ 2] = Dataset([SVector(u...)])  # init dic
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

function basin_cell_index(u, bsn_nfo::BasinInfo)
    iswithingrid = true
    @inbounds for i in 1:length(bsn_nfo.grid_minima)
        if !(bsn_nfo.grid_minima[i] ≤ u[i] ≤ bsn_nfo.grid_maxima[i])
            iswithingrid = false
            break
        end
    end
    if iswithingrid
        # Snap point to grid
        ind = @. round(Int, (u - bsn_nfo.grid_minima)/bsn_nfo.grid_steps) + 1
        return CartesianIndex(ind...)
    else
        return CartesianIndex(-1)
    end
end

function reset_basin_counters!(bsn_nfo::BasinInfo)
    bsn_nfo.consecutive_match = 0
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
        # out of the grid
        next_state = :lost
    elseif isodd(nxt_clr)
        # hit an basin box
        next_state = :bas_hit
    end

    if next_state != current_state
        # reset counter
        bsn_nfo.consecutive_match = 1
    end
    bsn_nfo.state = next_state
end
