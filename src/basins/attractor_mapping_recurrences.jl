struct AttractorMapperFAFAFAF{B<:BasinsInfo, I, K}
    bsn_nfo::B
    integ::I
    kwargs::K
end


# TODO: At the moment this is not useful. BUT, when we add the sparse matrices,
# this will really be fundamentally different than the `basins_of_attraction`.
"""
    AttractorsViaRecurrences(ds::DynaicalSystem, grid::Tuple; kwargs...)
Map initial conditions to attractors by identifying attractors on the fly based on
recurrences in the state space, as outlined by[^Datseris2022] and the 
[`basins_of_attraction`](@ref) function.

This attractor mapper method is exactly the [`basins_of_attraction`](@ref) function
but operating on-the-fly instead, and hence has exactly the same keywords
and `grid` structure.

[^Datseris2022]: G. Datseris and A. Wagemakers, [Chaos 32, 023104 (2022)](https://doi.org/10.1063/5.0076568)
"""
struct AttractorsViaRecurrences{I, B<:BasinsInfo, K} <: AttractorMapper
    integ::I
    bsn_nfo::B
    kwargs::K
end



function AttractorsViaRecurrences(ds, grid;
        # Notice that all of these are the same keywords as in `basins_of_attraction`
        Δt=nothing, T=nothing, idxs = 1:length(grid),
        complete_state = zeros(eltype(get_state(ds)), length(get_state(ds)) - length(grid)),
        diffeq = NamedTuple(), kwargs...
    )
    if isnothing(complete_state)
        complete_state = zeros(eltype(get_state(ds)), length(get_state(ds)) - length(grid))
    end
    bsn_nfo, integ = basininfo_and_integ(
        ds, attractors, grid, Δt, T, SVector(idxs...), complete_state, diffeq
    )
    return AttractorMapper(integ, bsn_nfo, kwargs)
end

function (mapper::AttractorsViaRecurrences)(u0)
    # Low level code of `basins_of_attraction` function
    lab = get_label_ic!(mapper.bsn_nfo, mapper.integ, u0; mapper.kwargs...)
    # Transform to integers indexing from odd-even indexing
    return iseven(lab) ? (lab ÷ 2) : (lab - 1) ÷ 2
end
