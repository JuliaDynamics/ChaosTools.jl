"""
    AttractorMapper(ds, mapping_method; kwargs...) → mapper
Initialize a structure that maps initial conditions of `ds` to attractors,
using the given `mapping_method`.

Currently available mapping methods:
* [`AttractorsViaRecurrences`](@ref)
* [`AttractorsViaProximity`](@ref)

`AttractorMapper` can always be used directly with [`basin_fractions`](@ref).

In addition, some mappers can be called as a function of an initial condition:
```julia
label::Int = mapper(u0)
```
and this will on the fly comput and return the label of the attractor `u0` converges at.
The mappers that can do this are:
* [`AttractorsViaRecurrences`](@ref)
* [`AttractorsViaProximity`](@ref)
"""
struct AttractorMapper{B<:BasinsInfo, I, K}
    bsn_nfo::B
    integ::I
    kwargs::K
end

abstract type AttractorMappingMethod end

"""
    AttractorsViaProximity(attractors::Dataset, ε = 1e-3)
Map initial conditions to attractors based on whether the trajectory reaches `ε`-distance
close to any of the user-provided `attractors`.

The process works identically as "Refining basins of attraction" of 
[`basins_of_attraction`](@ref). 

Because in this method all possible attractors are already known to the user,
the method can also be called **supervised**.
"""
struct AttractorsViaProximity <: AttractorMappingMethod
    attractors::A
    ε::Float64
end

"""
    AttractorsViaRecurrences(args...)
Map initial conditions to attractors by identifying attractors on the fly based on
recurrences in the state space, as outlined by[^Datseris2022] and the 
[`basins_of_attraction`](@ref) function.

[^Datseris2022]: G. Datseris and A. Wagemakers, [Chaos 32, 023104 (2022)]( https://doi.org/10.1063/5.0076568)
"""
struct AttractorsViaRecurrences <: AttractorMappingMethod
    field
end



function AttractorMapper(ds;
        # Notice that all of these are the same keywords as in `basins_of_attraction`
        grid = nothing, attractors = nothing,
        Δt=nothing, T=nothing, idxs = nothing,
        complete_state = nothing,
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

# TODO: Notice, currently this code assumes that all versions of `AttractorMapper`
# use the low level code of `basins_of_attraction`.
function (mapper::AttractorMapper)(u0; kwargs...)
    lab = get_label_ic!(mapper.bsn_nfo, mapper.integ, u0; mapper.kwargs...)
    return iseven(lab) ? (lab ÷ 2) : (lab - 1) ÷ 2
end
