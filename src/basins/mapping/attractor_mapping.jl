# Definition of the attracting mapping API and exporting
# At the end it also includes all files related to mapping

export AttractorMapper,
    AttractorsViaRecurrences,
    AttractorsViaRecurrencesSparse,
    AttractorsViaProximity,
    AttractorsViaFeaturizing,
    ClusteringConfig,
    basins_fractions,
    basins_of_attraction,
    automatic_Δt_basins

#########################################################################################
# AttractorMapper structure definition
#########################################################################################
"""
    AttractorMapper(ds::GeneralizedDynamicalSystem, args...; kwargs...) → mapper
Subtypes of `AttractorMapper` are structures that map initial conditions of `ds` to
attractors. Currently available mapping methods:
* [`AttractorsViaProximity`](@ref)
* [`AttractorsViaRecurrences`](@ref)
* [`AttractorsViaFeaturizing`](@ref)

All `AttractorMapper` subtypes can be used with [`basins_fractions`](@ref)
or [`basins_of_attraction`](@ref).

In addition, some mappers can be called as a function of an initial condition:
```julia
label = mapper(u0)
```
and this will on the fly compute and return the label of the attractor `u0` converges at.
The mappers that can do this are:
* [`AttractorsViaProximity`](@ref)
* [`AttractorsViaRecurrences`](@ref)
"""
abstract type AttractorMapper end

# Generic pretty printing
function generic_mapper_print(io, mapper)
    ps = 14
    text = "$(nameof(typeof(mapper)))"
    println(io, text)
    println(io, rpad(" rule f: ", ps), get_rule_for_print(mapper))
    return ps
end
Base.show(io::IO, mapper::AttractorMapper) = generic_mapper_print(io, mapper)

#########################################################################################
# Generic basin fractions method structure definition
#########################################################################################
# It works for all mappers that define the function-like-object behavior
"""
    basins_fractions(mapper::AttractorMapper, ics::Union{Dataset, Function}; kwargs...)

Approximate the state space fractions `fs` of the basins of attraction of a dynamical
stystem by mapping initial conditions to attractors using `mapper`
(which contains a reference to a [`GeneralizedDynamicalSystem`](@ref)).
The fractions are simply the ratios of how many initial conditions ended up
at each attractor.

Initial conditions to use are defined by `ics`. It can be:
* a `Dataset` of initial conditions, in which case all are used.
* a 0-argument function `ics()` that spits out random initial conditions.
  Then `N` random initial conditions are chosen.
  See [`statespace_sampler`](@ref) to generate such functions.

The returned arguments are `fs`.
If `ics` is a `Dataset` then the `labels` of each initial condition and roughly approximated
attractors are also returned: `fs, labels, attractors`.

The output `fs` is a dictionary whose keys are the labels given to each attractor
(always integers enumerating the different attractors), and the
values are their respective fractions. The label `-1` is given to any initial condition
where `mapper` could not match to an attractor (this depends on the `mapper` type).
`attractors` has the same structure, mapping labels to `Dataset`s.

See [`AttractorMapper`](@ref) for all possible `mapper` types.

## Keyword arguments
* `N = 1000`: Number of random initial conditions to generate in case `ics` is a function.
* `show_progress = true`: Display a progress bar of the process.
"""
function basins_fractions(mapper::AttractorMapper, ics::Union{AbstractDataset, Function};
        show_progress = true, N = 1000, additional_fs::Dict = Dict(),
    )
    used_dataset = ics isa AbstractDataset
    N = used_dataset ? size(ics, 1) : N
    if show_progress
        progress=ProgressMeter.Progress(N; desc="Mapping initial conditions to attractors:")
    end
    fs = Dict{Int, Int}()
    used_dataset && (labels = Vector{Int}(undef, N))
    # TODO: If we want to parallelize this, then we need to initialize as many
    # mappers as threads. Use a `threading` keyword and `deepcopy(mapper)`
    for i ∈ 1:N
        ic = _get_ic(ics, i)
        label = mapper(ic)
        fs[label] = get(fs, label, 0) + 1
        used_dataset && (labels[i] = label)
        show_progress && next!(progress)
    end
    # the non-public-API `additional_fs` is used in the continuation methods
    additive_dict_merge!(fs, additional_fs)
    N = N + (isempty(additional_fs) ? 0 : sum(values(additional_fs)))
    # Transform count into fraction
    ffs = Dict(k => v/N for (k, v) in fs)
    if used_dataset
        attractors = extract_attractors(mapper, labels, ics)
        return ffs, labels, attractors
    else
        return ffs
    end
end

_get_ic(ics::Function, i) = ics()
_get_ic(ics::AbstractDataset, i) = ics[i]


#########################################################################################
# Generic basins of attraction method structure definition
#########################################################################################
# It works for all mappers that define a `basins_fractions` method.
"""
    basins_of_attraction(mapper::AttractorMapper, grid::Tuple) → basins, attractors
Compute the full basins of attraction as identified by the given `mapper`,
which includes a reference to a [`GeneralizedDynamicalSystem`](@ref) and return them
along with (perhaps approximated) found attractors.

`grid` is a tuple of ranges defining the grid of initial conditions that partition
the state space into boxes with size the step size of each range.
For example, `grid = (xg, yg)` where `xg = yg = range(-5, 5; length = 100)`.
The grid has to be the same dimensionality as the state space expected by the
integrator/system used in `mapper`. E.g., a [`projected_integrator`](@ref)
could be used for lower dimensional projections, etc. A special case here is
a [`poincaremap`](@ref) with `plane` being `Tuple{Int, <: Real}`. In this special
scenario the grid can be one dimension smaller than the state space, in which case
the partitioning happens directly on the hyperplane the Poincaré map operates on.

`basins_of_attraction` function is a convenience 5-lines-of-code wrapper which uses the
`labels` returned by [`basins_fractions`](@ref) and simply assings them to a full array
corresponding to the state space partitioning indicated by `grid`.
"""
function basins_of_attraction(mapper::AttractorMapper, grid::Tuple; kwargs...)
    basins = zeros(Int32, map(length, grid))
    I = CartesianIndices(basins)
    A = Dataset([generate_ic_on_grid(grid, i) for i in I])
    fs, labels, attractors = basins_fractions(mapper, A; kwargs...)
    vec(basins) .= vec(labels)
    return basins, attractors
end

# Type-stable generation of an initial condition given a grid array index
@generated function generate_ic_on_grid(grid::NTuple{B, T}, ind) where {B, T}
    gens = [:(grid[$k][ind[$k]]) for k=1:B]
    quote
        Base.@_inline_meta
        @inbounds return SVector{$B, Float64}($(gens...))
    end
end

#########################################################################################
# Includes
#########################################################################################
include("attractor_mapping_proximity.jl")
include("attractor_mapping_recurrences.jl")
include("attractor_mapping_featurizing.jl")