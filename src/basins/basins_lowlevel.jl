mutable struct BasinInfo{B, G, IF, RF, UF, D, T, Q}
    basin::Array{Int16, B}
    grid_steps::G
    grid_maxima::G
    grid_minima::G
    iter_f!::IF
    reinit_f!::RF
    # TODO: Does `get_u` return the *FULL* system state, or only the state in the
    # subspace given by the grid...? `get_u` should be renamed to convey this info.
    get_u::UF
    current_color::Int
    next_avail_color::Int
    consecutive_match::Int
    consecutive_other_basins::Int
    prevConsecutives::Int
    prev_attr::Int
    prev_bas::Int
    prev_step::Int
    step::Int
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
        grid::Tuple, integ, iter_f!::Function, reinit_f!::Function, get_u::Function;
        # TODO: The way these keyword arguments are named is very confusing.
        # Why doesn't their name reveal their purpose? We need to rename all of them.
        mc_att = 10, mc_bas = 10, mc_unmb = 60,
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
        get_u,
        2,4,0,0,0,1,1,0,0,
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
        # First color is one
        bsn_nfo.basin[ind] = bsn_nfo.current_color + 1
        u0 = generate_ic_on_grid(grid, ind)
        bsn_nfo.basin[ind] = get_color_point!(bsn_nfo, integ, u0, mc_att, mc_bas, mc_unmb)
    end
    # remove attractors and rescale from 1 to Na
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


function get_color_point!(bsn_nfo::BasinInfo, integ, u0, mc_att, mc_bas, mc_unmb)
    # This routine identifies the attractor using the previously defined basin.
    # reinitialize integrator
    bsn_nfo.reinit_f!(integ, u0)
    reset_basin_counters!(bsn_nfo)
    cellcolor = inlimbo = 0

    while cellcolor == 0
        old_u = bsn_nfo.get_u(integ)
        bsn_nfo.iter_f!(integ)
        new_u = bsn_nfo.get_u(integ)
        n = basin_cell_index(new_u, bsn_nfo)

        if !isnothing(n) # apply procedure only for boxes in the defined space
            cellcolor = _identify_basin_of_cell!(
                # TODO: Why is `get_state(integ)` called here?
                # It is really confusing at which point is the integrator re-init
                # to a new state...
                bsn_nfo, n, get_state(integ), mc_att, mc_bas, mc_unmb
            )
            inlimbo = 0
        else
            # We are outside the defined grid
            inlimbo += 1
        end

        if inlimbo > 60 # TODO: This `60` should be a named keyword
            cellcolor = check_outside_the_grid!(bsn_nfo, new_u, old_u, inlimbo)
        end
    end
    return cellcolor
end

"""
Main procedure described by Nusse & Yorke for the grid cell `n`.
The idea is to color the grid with the current color. When an attractor box is hit
(even color), the initial condition is colored with the color of its basin (odd color).
If the trajectory hits another basin many times times in row, the IC is colored with the
same color as this basin.
"""
function _identify_basin_of_cell!(
        bsn_nfo::BasinInfo, n::CartesianIndex, u,
        mc_att::Int, mc_bas::Int, mc_unmb::Int
    )
    next_c = bsn_nfo.basin[n]
    bsn_nfo.step += 1

    if iseven(next_c) && bsn_nfo.consecutive_match < mc_unmb
        # check wether or not we hit an attractor (even color). Make sure we hit mc_att consecutive times.
        if bsn_nfo.prev_attr == next_c
            bsn_nfo.prevConsecutives += 1
        else
            # reset prevConsecutives
            bsn_nfo.prev_attr = next_c
            bsn_nfo.prevConsecutives = 1
            return 0;
        end

        if bsn_nfo.prevConsecutives ≥ mc_att
            # Wait if we hit the attractor a mc_att times in a row just to check if it is not a nearby trajectory
            c3 = next_c + 1
            if mc_att == 2
                # For maps we can color the previous steps as well. Every point of the trajectory lead
                # to the attractor
                recolor_visited_cell!(bsn_nfo, bsn_nfo.current_color + 1, c3)
            else
                # For higher dimensions we erase the past iterations and visited boxes
                recolor_visited_cell!(bsn_nfo, bsn_nfo.current_color + 1, 1)
            end
            reset_basin_counters!(bsn_nfo)
            return c3
         end
    end

    if next_c == 1 && bsn_nfo.consecutive_match < mc_unmb
        # uncolored box, color it with current odd color
        bsn_nfo.basin[n] = bsn_nfo.current_color + 1
        push!(bsn_nfo.visited,n) # keep track of visited cells
        bsn_nfo.consecutive_match = 0
        return 0
    elseif next_c == 1 && bsn_nfo.consecutive_match >= mc_unmb
        # Maybe chaotic attractor, perodic or long recursion.
        # Color this box as part of an attractor
        bsn_nfo.basin[n] = bsn_nfo.current_color
        # reinit consecutive match to ensure that we have an attractor
        bsn_nfo.consecutive_match = mc_unmb
        store_attractor!(bsn_nfo, u)
        return 0
    elseif next_c == bsn_nfo.current_color + 1
        # hit a previously visited box with the current color, possible attractor?
        if bsn_nfo.consecutive_match < mc_unmb
            bsn_nfo.consecutive_match += 1
            return 0
        else
            bsn_nfo.basin[n] = bsn_nfo.current_color
            store_attractor!(bsn_nfo, u)
            # We continue iterating until we hit again the same attractor. In which case we stop.
            return 0;
        end
    elseif (
            isodd(next_c) && 0 < next_c < bsn_nfo.current_color && 
            bsn_nfo.consecutive_match < mc_unmb && mc_att == 2
        )
        # hit a colored basin point of the wrong basin, happens all the time, we check if it happens
        # mc_bas times in a row or if it happens N times along the trajectory whether to decide if it is another basin.
        bsn_nfo.consecutive_other_basins += 1

        if bsn_nfo.prev_bas == next_c &&  bsn_nfo.prev_step == bsn_nfo.step - 1
            bsn_nfo.prevConsecutives += 1
            bsn_nfo.prev_step += 1
        else
            bsn_nfo.prev_bas = next_c
            bsn_nfo.prev_step = bsn_nfo.step
            bsn_nfo.prevConsecutives = 1
        end

        if bsn_nfo.consecutive_other_basins > 60 || bsn_nfo.prevConsecutives > mc_bas
            recolor_visited_cell!(bsn_nfo, bsn_nfo.current_color + 1, next_c)
            reset_basin_counters!(bsn_nfo)
            return next_c
        end
        return 0
    elseif iseven(next_c) && (mc_unmb <= bsn_nfo.consecutive_match < 2 * mc_unmb)
        # We make sure we hit the attractor 60 consecutive times
        bsn_nfo.consecutive_match += 1
        return 0
    elseif iseven(next_c) && bsn_nfo.consecutive_match >= mc_unmb * 2
        # We have checked the presence of an attractor: tidy up everything and get a new box.
        recolor_visited_cell!(bsn_nfo, bsn_nfo.current_color + 1, 1)
        bsn_nfo.basin[n] = bsn_nfo.current_color
        store_attractor!(bsn_nfo, u)
        # pick the next color for coloring the basin.
        bsn_nfo.current_color = bsn_nfo.next_avail_color
        bsn_nfo.next_avail_color += 2
        reset_basin_counters!(bsn_nfo)
        return next_c + 1;
    else
        return 0
    end
end

function store_attractor!(bsn_nfo::BasinInfo, u)
    # TODO: This comment is confusing, because `Na` is not used anywhere in the function
    # We divide by to order the attractors from 1 to Na
    if haskey(bsn_nfo.attractors , bsn_nfo.current_color ÷ 2)
        push!(bsn_nfo.attractors[bsn_nfo.current_color ÷ 2],  u) # store attractor
    else
        bsn_nfo.attractors[bsn_nfo.current_color ÷ 2] = Dataset([SVector(u...)])  # init dic
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
        return nothing
    end
end

function check_outside_the_grid!(bsn_nfo::BasinInfo, new_u, old_u, inlimbo)
    # TODO: Not sure if this is a good decider of maximum, perhaps the diagonal of the grid
    # is better...?
    grid_maximum = maximum(abs.(bsn_nfo.grid_maxima)); # this is the largest size of the phase space
    if norm(new_u-old_u) < 1e-5 || inlimbo > 60*20 || norm(new_u) > 10*grid_maximum
        # TODO: all numeric constants in the above line must be replaced with
        # named variables with intention-revealing name. They must also be tunable
        # as keyword arguments in `draw_basin!`.
        # TODO: Comparing the `norm` of the `new_u` is a mistake, as it assumes
        # that the grid starts from 0. We instead need to compare the `norm`
        # of the `new_u` from the center of the grid.
        recolor_visited_cell!(bsn_nfo, bsn_nfo.current_color + 1, 1)
        reset_basin_counters!(bsn_nfo)
        # problematic IC : diverges or wanders outside the defined grid
        return -1
    end
    return 0
end

function reset_basin_counters!(bsn_nfo::BasinInfo)
    bsn_nfo.consecutive_match = 0
    bsn_nfo.consecutive_other_basins = 0
    bsn_nfo.prevConsecutives = 0
    bsn_nfo.prev_attr = 1
    bsn_nfo.prev_bas = 1
    bsn_nfo.prev_step = 0
    bsn_nfo.step = 0
end
