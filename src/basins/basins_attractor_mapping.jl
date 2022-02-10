struct AttractorMapperFAFAFAF{B<:BasinsInfo, I, K}
    bsn_nfo::B
    integ::I
    kwargs::K
end


"""
    AttractorsViaProximity(ds::DynamicalSystem, attractors::Dict; kwargs...)
Map initial conditions to attractors based on whether the trajectory reaches `ε`-distance
close to any of the user-provided `attractors`. They have to be in a form of a dictionary
mapping attractor labels to `Dataset`s containing the attractors.

The state of the system gets stepped, and at each step the minimum distance to all
attractors is computed. If any of these distances is `≤ ε`, then the label of the nearest
attractor is returned. 

Because in this method all possible attractors are already known to the user,
the method can also be called _supervised_.

## Keywords
* `Δt = 1`: Step size for `ds`.
* `horizon_limit = 1e6`: If `norm(get_state(ds))` exceeds this number, it is assumed
  that the trajectory diverged (gets labelled as `-1`).
* `mx_chk_lost = 1000`: If the integrator has been stepped this many times without
  coming `ε`-near to any attractor,  it is assumed
  that the trajectory diverged (gets labelled as `-1`).
"""
struct AttractorsViaProximity{DS, D, T} <: AttractorMapper
    ds::DS
    attractors::Dict{Int16, Dataset{D, T}}
    ε::Float64
    Δt::T
    mx_chk_lost::Int
    horizon_limit::T
    lost_count::Ref{Int}
end
function AttractorsViaProximity(ds::DynamicalSystem, attractors::Dict; 
        ε=1e-3, Δt=1, mx_chk_lost=1000, horizon_limit=1e6
    )
    @assert dimension(ds) == dimension(first(attractors))
    return AttractorsViaProximity(ds, attractors, ε, Δt, horizon_limit)
end

function FAFAFA()
    if isnothing(grid) && isnothing(attractors)
        @error "At least one of `grid` of `attractor` must be provided."
    end
    if isnothing(grid)
        # dummy grid for initialization if the second mode is used
        grid = ntuple(x -> range(-1, 1,step = 0.1), length(ds.u0))
    end
    if isnothing(idxs)
        idxs = 1:length(grid)
    end
    if isnothing(complete_state)
        complete_state = zeros(eltype(get_state(ds)), length(get_state(ds)) - length(grid))
    end
    bsn_nfo, integ = basininfo_and_integ(ds, attractors, grid, Δt, T, SVector(idxs...), complete_state, diffeq)
    return AttractorMapper(bsn_nfo, integ, kwargs)
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

[^Datseris2022]: G. Datseris and A. Wagemakers, [Chaos 32, 023104 (2022)]( https://doi.org/10.1063/5.0076568)
"""
struct AttractorsViaRecurrences{DS, B<:BasinsInfo, I, K} <: AttractorMapper
    ds::DS
    bsn_nfo::B
    integ::I
    kwargs::K
end



function AttractorsViaRecurrences(ds, grid;
        # Notice that all of these are the same keywords as in `basins_of_attraction`
        Δt=nothing, T=nothing, idxs = 1:length(grid),
        complete_state = zeros(eltype(get_state(ds)), length(get_state(ds)) - length(grid)),
        diffeq = NamedTuple(), kwargs...
    )
    if isnothing(grid) && isnothing(attractors)
        @error "At least one of `grid` of `attractor` must be provided."
    end
    if isnothing(grid)
        # dummy grid for initialization if the second mode is used
        grid = ntuple(x -> range(-1, 1,step = 0.1), length(ds.u0))
    end
    if isnothing(idxs)
        idxs = 1:length(grid)
    end
    if isnothing(complete_state)
        complete_state = zeros(eltype(get_state(ds)), length(get_state(ds)) - length(grid))
    end
    bsn_nfo, integ = basininfo_and_integ(ds, attractors, grid, Δt, T, SVector(idxs...), complete_state, diffeq)
    return AttractorMapper(bsn_nfo, integ, kwargs)
end


function (mapper::AttractorMapper)(u0)
    lab = get_label_ic!(mapper.bsn_nfo, mapper.integ, u0)
    return iseven(lab) ? (lab ÷ 2) : (lab - 1) ÷ 2
end
