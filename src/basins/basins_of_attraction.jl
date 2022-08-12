# TODO:
# This entire file is DEPRECATED and will be removed in future versions.

function basins_of_attraction(grid::Tuple, ds;
        Δt=nothing, diffeq = NamedTuple(), attractors = nothing,
        # T, idxs, compelte_state are DEPRECATED!
        T=nothing, idxs = nothing, complete_state = nothing,
        # `kwargs` tunes the basin finding algorithm, e.g. `mx_chk_att`.
        # these keywords are actually expanded in `_identify_basin_of_cell!`
        kwargs...
    )

    @warn("""
    The function `basins_of_attraction(grid::Tuple, ds::DynamicalSystem; ...)` is
    deprecated in favor of the more generic
    `basins_of_attraction(mapper::AttractorMapper, grid::Tuple`) which works for
    any instance of `AttractorMapper`. Please use that method in the future.

    The only reason the existing method is kept in and can be used is because the paper
    G. Datseris and A. Wagemakers, *Effortless estimation of basins of attraction*
    had this old call signature published.
    """)

    if !isnothing(T)
        @warn("Using `T` is deprecated. Initialize a `stroboscopicmap` and pass it.")
        integ = stroboscopicmap(ds, T; diffeq)
    elseif ds isa PoincareMap
        integ = ds
    elseif length(grid) ≠ dimension(ds) && isnothing(idxs)
        @warn("Giving a `grid` with dimension lower than `ds` is deprecated. "*
        "Initialize a `projected_integrator` instead.")
        idxs = 1:length(grid)
        c = zeros(eltype(get_state(ds)), length(get_state(ds)) - length(grid))
        integ = projected_integrator(ds, idxs, c; diffeq)
    elseif !isnothing(idxs)
        @warn("Using `idxs` is deprecated. Initialize a `projeted_integrator` instead.")
        @assert length(idxs) == length(grid)
        remidxs = setdiff(1:dimension(ds), idxs)
        if isnothing(complete_state)
            c = zeros(eltype(get_state(ds)), length(get_state(ds)) - length(grid))
        elseif complete_state isa Function
            c = function(y)
                    v = zeros(eltype(get_state(ds)), dimension(ds))
                    v[idxs] .= y
                    v[remidxs] .= complete_state(y)
                    return v
                end
        else
            c = complete_state
        end
        integ = projected_integrator(ds, idxs, c; diffeq)
    else
        integ = ds
    end

    if !isnothing(attractors) # proximity version
        # initialize Proximity and loop.
        mapper = AttractorsViaProximity(integ, attractors::Dict, getindex(kwargs, :ε) ;
        Δt=isnothing(Δt) ? 1 : Δt, kwargs...)
        return estimate_basins_proximity!(mapper, grid; kwargs...)
    else # (original) recurrences version
        bsn_nfo, integ = basininfo_and_integ(integ, grid, Δt, diffeq, false)
        bsn_nfo = estimate_basins_recurrences!(grid, bsn_nfo, integ; kwargs...)
        return bsn_nfo.basins, bsn_nfo.attractors
    end
end


import ProgressMeter
using Statistics: mean

function estimate_basins_proximity!(mapper, grid; show_progress = true, kwargs...)
    basins = zeros(Int16, map(length, grid))
    progress = ProgressMeter.Progress(
        length(basins); desc = "Basins of attraction: ", dt = 1.0
    )
    for (k,ind) in enumerate(CartesianIndices(basins))
        show_progress && ProgressMeter.update!(progress, k)
        y0 = generate_ic_on_grid(grid, ind)
        basins[ind] = mapper(y0)
    end
    return basins, mapper.attractors
end


"""
This is the low level function that computes the full basins of attraction,
given the already initialized `BasinsInfo` object and the integrator.
It simply loops over the `get_label_ic!` function, that maps initial conditions
to attractors.
"""
function estimate_basins_recurrences!(
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
