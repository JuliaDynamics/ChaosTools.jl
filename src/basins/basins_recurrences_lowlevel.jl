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
    prev_label::Int
    attractors::Dict{Int16, Dataset{D, T}}
    visited_list::Q
    search_trees::K
    dist::Vector{Float64}
    neighborindex::Vector{Int64};
end


function basininfo_and_integ(
        ds, attractors, grid, Δt, T, idxs, complete_state, diffeq
    )

    fixed_solver = haskey(diffeq, :dt) && haskey(diffeq, :adaptive)
    if ds isa ContinuousDynamicalSystem && isnothing(Δt) && isnothing(T) && !fixed_solver
        Δt = automatic_Δt_basins(ds, grid; idxs, complete_state, diffeq)
        @info "Automatic Δt estimation yielded Δt = $(Δt)"
    end


    @assert length(idxs) == length(grid)
    integ = ds isa PoincareMap ? ds : integrator(ds; diffeq)
    D = length(get_state(integ))
    if complete_state isa AbstractVector && (length(complete_state) ≠ D-length(idxs))
        error("Vector `complete_state` must have length D-Dg!")
    end

    idxs = SVector(idxs...)

    complete_and_reinit! = CompleteAndReinit(complete_state, idxs, length(get_state(integ)))
    get_projected_state = (integ) -> view(get_state(integ), idxs)
    MDI = DynamicalSystemsBase.MinimalDiscreteIntegrator
    if !isnothing(T)
        iter_f! = (integ) -> step!(integ, T, true)
    elseif (integ isa PoincareMap) || (integ isa MDI) || fixed_solver
        iter_f! = step!
    else # generic case
        iter_f! = (integ) -> step!(integ, Δt) # we don't have to step _exactly_ `Δt` here
    end
    bsn_nfo = init_bsn_nfo(
        grid, integ, iter_f!, complete_and_reinit!,
        get_projected_state, attractors
    )
    return bsn_nfo, integ
end



function init_bsn_nfo(
        grid::Tuple, integ, iter_f!::Function, complete_and_reinit!,
        get_projected_state::Function, attractors = nothing
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
        trees,
        [Inf],
        [0]
    )
    reset_basins_counters!(bsn_nfo)
    return bsn_nfo
end

"""
This is the low level function that computes the full basins of attraction,
given the already initialized `BasinsInfo` object and the integrator.
It simply loops over the `get_label_ic!` function, that maps initial conditions
to attractors.
"""
function estimate_basins!(
        grid::Tuple,
        bsn_nfo::BasinsInfo, integ;
        show_progress = true, kwargs...,
    )
    I = CartesianIndices(bsn_nfo.basins)
    progress = ProgressMeter.Progress(
        length(bsn_nfo.basins); desc = "Basins of attraction: ", dt = 1.0
    )

    for (k,ind) in enumerate(I)
        if bsn_nfo.basins[ind] == 0
            show_progress && ProgressMeter.update!(progress, k)
            y0 = generate_ic_on_grid(grid, ind)
            bsn_nfo.basins[ind] =
            get_label_ic!(bsn_nfo, integ, y0; show_progress, kwargs...)
        end
    end

    # remove attractors and rescale from 1 to max number of attractors
    ind = iseven.(bsn_nfo.basins)
    bsn_nfo.basins[ind] .+= 1
    bsn_nfo.basins .= (bsn_nfo.basins .- 1) .÷ 2

    return bsn_nfo
end


"""
    get_label_ic!(bsn_nfo::BasinsInfo, integ, u0; kwargs...) -> ic_label
This function returns the number that matches the initial condition `u0` to an attractor.
`u0` must be of the same dimension as the grid used in [`ic_labelling`](@ref).
`bsn_nfo` and `integ` are generated by the function [`ic_labelling`](@ref).

Notice the numbering system `celllabel` is as in `_identify_basin_of_cell!`.
"""
function get_label_ic!(bsn_nfo::BasinsInfo, integ, u0; kwargs...)
    # This routine identifies the attractor using the previously defined basin.
    # reinitialize integrator
    bsn_nfo.complete_and_reinit!(integ, u0)
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
        horizon_limit = 1e6, ε = 1e-3,
        mx_chk_lost = isnothing(bsn_nfo.search_trees) ? 20 : 1000,
        show_progress = true, # show_progress only used when finding new attractor.
    )

    #if n[1]==-1 means we are outside the grid
    ic_label = (n[1]==-1  || isnan(u_full_state[1])) ? -1 : bsn_nfo.basins[n]

    # search attractors directly
    if !isnothing(bsn_nfo.search_trees)
        bsn_nfo.consecutive_lost = (ic_label == -1 ? bsn_nfo.consecutive_lost + 1 : 0);
        if norm(u_full_state) > horizon_limit || bsn_nfo.consecutive_lost > mx_chk_lost
            return -1
        end
        for (k, t) in bsn_nfo.search_trees # this is a `Dict`
            Neighborhood.NearestNeighbors.knn_point!(
                t, u_full_state, false, bsn_nfo.dist,
                bsn_nfo.neighborindex, Neighborhood.alwaysfalse
            )
            if bsn_nfo.dist[1] < ε
                ic_label = 2*k + 1
                return ic_label
            end
        end
        return 0
    end

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
        if   bsn_nfo.consecutive_lost > mx_chk_lost || norm(u_full_state) > horizon_limit
            relabel_visited_cell!(bsn_nfo, bsn_nfo.visited_cell, 0)
            reset_basins_counters!(bsn_nfo)
            # problematic IC : diverges or wanders outside the defined grid
            return -1
        end
        bsn_nfo.prev_label = ic_label
        return 0
    end
end

function store_attractor!(bsn_nfo::BasinsInfo{B, IF, RF, UF, D, T, Q},
    u_full_state, show_progress = true) where {B, IF, RF, UF, D, T, Q}
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
