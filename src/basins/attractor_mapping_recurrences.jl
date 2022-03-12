#####################################################################################
# Type definition and documentation
#####################################################################################
include("basins_recurrences_lowlevel.jl")

# TODO: Move here the docstring of `basins_of_attraction`. THere, say that
# the "supervised" version actually just uses `AttractorsViaProximity`.

"""
    AttractorsViaRecurrences(ds::DynaicalSystem, grid::Tuple; kwargs...)
Map initial conditions to attractors by identifying attractors on the fly based on
recurrences in the state space, as outlined by[^Datseris2022] and the
[`basins_of_attraction`](@ref) function.

This attractor mapper method is exactly the [`basins_of_attraction`](@ref) function
but operating on-the-fly instead, and hence has exactly the same keywords
and `grid` structure.

[^Datseris2022]:
    G. Datseris and A. Wagemakers,
    [Chaos 32, 023104 (2022)](https://doi.org/10.1063/5.0076568)
"""
struct AttractorsViaRecurrences{I, B<:BasinsInfo, K} <: AttractorMapper
    integ::I
    bsn_nfo::B
    kwargs::K
end

function AttractorsViaRecurrences(ds::GeneralizedDynamicalSystem, grid;
        Δt=nothing, diffeq = NamedTuple(), kwargs...
    )
    bsn_nfo, integ = basininfo_and_integ(ds, nothing, grid, Δt, diffeq)
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
mutable struct BasinsInfo{B, IF, RF, UF, D, T, Q, K}
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
    search_trees::K
    dist::Vector{Float64}
    neighborindex::Vector{Int64};
end

function basininfo_and_integ(ds::GeneralizedDynamicalSystem, attractors, grid, Δt, diffeq)
    integ = integrator(ds; diffeq)
    isdiscrete = isdiscretetime(integ)
    if isnothing(Δt)
        if !isdiscrete
            Δt = automatic_Δt_basins(integ, grid)
            @info "Automatic Δt estimation yielded Δt = $(Δt)"
        else
            Δt = 1
        end
    end
    integ = integrator(ds; diffeq)
    iter_f! = if (isdiscrete && Δt == 1)
        (integ) -> step!(integ)
    else
        (integ) -> step!(integ, Δt)
    end
    bsn_nfo = init_bsn_nfo(grid, integ, iter_f!, attractors)
    return bsn_nfo, integ
end

function init_bsn_nfo(grid::Tuple, integ, iter_f!::Function, attractors = nothing)
    D = length(grid)
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
        SVector{D, Float64}(grid_steps),
        SVector{D, Float64}(grid_maxima),
        SVector{D, Float64}(grid_minima),
        iter_f!,
        :att_search,
        2,4,0,1,0,
        Dict{Int16,Dataset{D,eltype(get_state(integ))}}(),
        Vector{CartesianIndex{D}}(),
        trees,
        [Inf],
        [0]
    )
    reset_basins_counters!(bsn_nfo)
    return bsn_nfo
end